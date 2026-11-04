"""The 30 Hz runtime: policy -> interventions -> robot, telemetry, anomaly triggers and the XAI pipeline.

Threading model: the control loop runs in a dedicated thread (hardware I/O and the policy are
blocking); the Meta API pipeline runs as coroutines on the FastAPI event loop. Every frame the
thread publishes a packet (telemetry JSON + overhead JPEG) to the websocket ``Hub``. Commands from
the API (apply fix, e-stop, reset, ...) take ``self._lock`` so they never interleave with a tick.

Interventions only touch what the *policy* sees or what the *servos* are sent:

* mask suppression rewrites the overhead image fed to the policy (the raw frame is still recorded),
* a waypoint bypass pre-empts the policy until the clamped trajectory completes,
* a grasp offset shifts the policy's grasp target.

Demo staging (``--demo-distractor-on-goal``) also edits the policy's input: a SAM pre-flight scan finds
the distractor and the goal, and the policy is shown the distractor on the goal wherever it really is.
Both views are recorded, and the UI and the replay label the staging.

Every failure starts a live triage (Muse Spark, seconds) and an offline replay of the run
(``xai_engine.replay``): heavier attention maps over the recorded frames, then a Muse Wonder breakdown
that can self-correct the failure when no human is needed.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.config import Settings
from backend.meta_clients.base import MetaClient
from backend.meta_clients.sam_client import SAMClient
from backend.meta_clients.spark_explainer import DiagnosisContext, SparkExplainer, forced_diagnosis
from backend.meta_clients.voice_and_image import CounterfactualRenderer, counterfactual_prompt, image_region
from backend.meta_clients.wonder_breakdown import AUTOMATIC, WonderBreakdown
from backend.meta_clients.wonder_planner import WonderPlanner
from backend.robot.calibration import Homography
from backend.robot.policy_hook import ACTRunner, CameraObserver, MockPolicy, PolicyStep
from backend.robot.sam3_observer import Sam3Observer
from backend.robot.vla_observer import SmolVLAObserver
from backend.robot.so101_interface import (
    SCENARIOS, MockSO101, Observation, PlanarKinematics, RealSO101, WebcamSource,
)
from backend.schemas import (
    AnalysisResult, AnomalyEvent, AskResponse, Counterfactual, RecoveryPlan, ReplayResult, ServoTelemetry,
)
from backend.xai_engine import counterfactual as cfx
from backend.xai_engine.attention_reference import AttentionReference, Deviation
from backend.xai_engine.attribution import (
    NON_DISTRACTOR_LABELS,
    AnomalyDetector,
    SegmentedObject,
    centroid,
    compute_attributions,
    joint_error,
    saliency_centroid,
)
from backend.xai_engine.intervention import (
    DistractorStager, GraspOffset, MaskSuppressor, WaypointExecutor, clamp_joints,
)
from backend.xai_engine.replay import LazyModel, ReplayAnalyzer, Run, list_runs
from backend.xai_engine.ring_buffer import FrameRecord, RingBuffer
from backend.xai_engine.visualize import composite_keyframe, draw_path, encode_jpeg, heatmap_overlay

log = logging.getLogger(__name__)

RUNNING, HOLD, ESTOP, COMPLETE = "RUNNING", "HOLD", "ESTOP", "COMPLETE"
START_PROMPT = "ready: press ▶ Start policy"
LIVE_ATTRIBUTION_EVERY = 3  # ticks: 10 Hz attribution/polygon refresh at 30 Hz control


class ConflictError(RuntimeError):
    """Request is valid but the runtime is in the wrong state for it (HTTP 409)."""


@dataclass
class AnalysisRecord:
    result: AnalysisResult
    objects: list[SegmentedObject]
    frame: np.ndarray
    saliency: np.ndarray | None
    keyframe_jpeg: bytes
    ctx: DiagnosisContext
    qpos: np.ndarray | None = None  # joint state at the keyframe (the counterfactual asks the policy at this state)
    counterfactual_views: dict[str, bytes] = field(default_factory=dict)  # image | attention_before | attention_after
    applied: bool = False


@dataclass(eq=False)  # identity hash: clients live in a set
class _Client:
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    events: deque = field(default_factory=deque)
    frame: tuple | None = None  # latest (meta, raw_jpeg, policy_jpeg); older frames are dropped
    view: str = "policy"  # "policy": what the policy sees (suppression visible) | "raw"


class Hub:
    """Fan-out to websocket clients. Frames are coalesced per client; events are never dropped."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[_Client] = set()

    def subscribe(self) -> _Client:
        c = _Client()
        self.clients.add(c)
        return c

    def unsubscribe(self, c: _Client) -> None:
        self.clients.discard(c)

    def _frame(self, item: tuple) -> None:
        for c in self.clients:
            c.frame = item
            c.wake.set()

    def _event(self, event: dict) -> None:
        for c in self.clients:
            c.events.append(event)
            c.wake.set()

    def publish_frame(self, item: tuple) -> None:
        if self.loop is not None and self.clients:
            self.loop.call_soon_threadsafe(self._frame, item)

    def publish_event(self, event: dict) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._event, event)


def _r(values, nd: int = 2) -> list[float]:
    return [round(float(v), nd) for v in values]


class ControlLoop:
    def __init__(self, settings: Settings) -> None:
        s = self.s = settings
        self.hub = Hub()
        self.kin = PlanarKinematics()
        self.size_wh = (s.overhead_width, s.overhead_height)
        self.homography = Homography.load_or_default(s.homography_path, *self.size_wh)
        self.mock = s.mock_hardware and not s.camera_only
        self.camera_only = s.camera_only
        if self.mock:
            self.robot: MockSO101 | RealSO101 | WebcamSource = MockSO101(s, self.homography, s.mock_scenario)
            self.plate = self.robot.background_plate()
        else:
            self.robot = WebcamSource(s) if self.camera_only else RealSO101(s)
            self.plate = cv2.imread(str(s.background_plate_path)) if s.background_plate_path.exists() else None
            if self.plate is not None and self.plate.shape[:2] != self.size_wh[::-1]:
                self.plate = cv2.resize(self.plate, self.size_wh, interpolation=cv2.INTER_AREA)
            if self.plate is None:
                log.warning("No background plate at %s: suppression will inpaint (run calibration.py)", s.background_plate_path)

        self.policy: MockPolicy | ACTRunner | CameraObserver | SmolVLAObserver | None = None
        self.policy_kind = "none"
        self.suppressor = MaskSuppressor(self.plate, s.tabletop_bgr)
        self.stager = DistractorStager(self.plate, s.tabletop_bgr)  # demo staging (see module docstring)
        self.self_correct = s.self_correct
        self._preflight_pending = s.demo_distractor_on_goal and not s.camera_only  # hold until staged
        self._preflight_task: asyncio.Task | None = None
        self._run_t0 = 0.0  # timestamp the current run started (after pre-flight): the replay starts here
        self._awaiting_start = False  # --start-paused: HOLD until the operator presses Start
        self.replay: ReplayAnalyzer | None = None
        self.executor = WaypointExecutor()
        self.grasp = GraspOffset()
        self.reference: AttentionReference | None = None
        self.attn_dev: Deviation | None = None
        self.deviation_threshold: float | None = None
        if s.policy_path and not self.mock and s.attention_reference_path.exists():
            self.reference = AttentionReference.load(s.attention_reference_path)
            self.deviation_threshold = s.attention_deviation_threshold or self.reference.threshold
            if self.deviation_threshold is None:
                log.warning("%s has no calibrated threshold: run scripts/calibrate_attention_reference.py", s.attention_reference_path)
                self.reference = None
            else:
                log.info("Attention reference: %d demo samples, deviation threshold %.2f", len(self.reference.maps), self.deviation_threshold)
        elif s.policy_path and not self.mock:
            log.warning("No attention reference at %s: falling back to the fixed A(i) hijack rule", s.attention_reference_path)
        disabled = set(s.disabled_triggers)
        if s.policy_path and s.policy_observe_every > 0:
            # Disagreement between observe passes fires on normal demonstration runs too (see docs/04), so it
            # is only a trigger when the policy genuinely replans with overlapping chunks.
            disabled.add("variance")
        disabled = frozenset(disabled)
        self.detector = AnomalyDetector(
            s.control_hz, s.joint_error_threshold_deg, s.joint_error_hold_s, s.chunk_variance_threshold,
            s.attention_hijack_threshold, s.stall_progress_s, disabled,
            deviation_threshold=self.deviation_threshold, deviation_hold_s=s.attention_deviation_hold_s,
            hijack_excess_threshold=s.attention_hijack_excess_threshold,
        )
        self.ring = RingBuffer(s.ring_buffer_frames)
        self.history: deque[ServoTelemetry] = deque(maxlen=int(10 * s.control_hz))

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = RUNNING
        self.status_text = "starting"
        self.phase = ""
        self.tick_count = 0
        self.fps = 0.0
        self.saliency: np.ndarray | None = None
        self.saliency_method = ""
        self.last_variance = 0.0
        self.last_target: np.ndarray | None = None
        self.last_q = np.zeros(6)
        self.last_anomaly: AnomalyEvent | None = None
        self.recovery_note = ""
        self.live_objects: list[SegmentedObject] = []
        self.live_attrs = []
        self.live_views: list[dict] = []
        self._sam_live_objects: list[SegmentedObject] = []
        self._tracked: dict[int, SegmentedObject] = {}
        self._track_task: asyncio.Task | None = None
        self._live_sam_task: asyncio.Task | None = None
        self._last_publish = 0.0

        self.analyses: dict[str, AnalysisRecord] = {}
        self.latest_analysis_id: str | None = None
        self._analysis_lock: asyncio.Lock | None = None
        self._epoch = 0  # bumped by reset(); ticks and analyses that straddle a reset are not published
        self.loop: asyncio.AbstractEventLoop | None = None
        self.meta: MetaClient | None = None

    # ================================================================== lifecycle
    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.hub.loop = self.loop
        self._analysis_lock = asyncio.Lock()
        self.meta = MetaClient(self.s)
        # The colour references describe the real overhead camera, not the simulator's renders.
        self.sam = SAMClient(self.s, self.meta, self._mock_truth if self.mock else None,
                             None if self.mock else self.s.appearance_refs)
        self.spark = SparkExplainer(self.s, self.meta)
        self.wonder = WonderPlanner(self.s, self.meta, self.kin)
        self.images = CounterfactualRenderer(self.s, self.meta)

        await asyncio.to_thread(self.robot.connect)
        if self.camera_only and self.s.vla_path:
            self.policy = await asyncio.to_thread(
                SmolVLAObserver, self.s.vla_path, self.s.policy_device, self.s.overhead_camera_key,
                self.s.vla_camera_key, self.size_wh, self.s.task_description, self.s.vla_attention,
            )
            self.policy_kind = "smolvla-observe"
        elif self.camera_only:
            self.policy = CameraObserver(self.s.overhead_camera_key)
            self.policy_kind = "camera-only"
        elif self.s.policy_path:
            self.policy = await asyncio.to_thread(
                ACTRunner, self.s.policy_path, self.s.policy_device, self.s.overhead_camera_key, self.size_wh,
                self.s.policy_n_action_steps, self.s.task_description, self.s.policy_observe_every,
            )
            self.policy_kind = "act"
        elif self.mock:
            self.policy = MockPolicy(self.kin, self.homography, n_action_steps=10)
            self.policy_kind = "mock"
        else:
            log.warning("No --policy-path given: the arm holds position; streaming and analysis still work")
        self.status_text = (
            f"camera-only: SmolVLA {self.s.vla_attention} attention, observe only (no arm commands)"
            if self.policy_kind == "smolvla-observe"
            else "camera-only: observing the workspace (no arm connected)" if self.camera_only
            else "running" if self.policy is not None else "no policy loaded: holding position"
        )
        if self.s.start_paused and self.policy is not None and not self.camera_only:
            # Nothing is sent to the arm in HOLD, so it stays where it is while everything loads.
            self.state, self._awaiting_start = HOLD, True
            self.status_text = START_PROMPT
        if not self.meta.enabled:
            log.warning("MODEL_API_KEY not set: all Meta models will use local fallbacks")
        if self.s.replay_enabled and not self.camera_only:
            self.replay = ReplayAnalyzer(
                self.s, self.sam, WonderBreakdown(self.s, self.meta), self.hub.publish_event,
                busy=self._replay_should_wait, vla=self._replay_vla(), on_breakdown=self._on_breakdown,
                counterfactual=self.counterfactual, live_attenders=self.attenders,
            )
        if not self.mock and self.meta.enabled and self.s.live_sam_interval_s > 0:
            self._live_sam_task = asyncio.create_task(self._live_sam_loop())
        self._thread = threading.Thread(target=self._run, name="glassbox-control", daemon=True)
        self._thread.start()
        self._schedule_preflight()

    def _replay_vla(self) -> LazyModel | None:
        """The replay's observe-only lens, loaded in the background: SAM 3 (perception) or SmolVLA (a second
        policy's attention). Either only ever looks at recorded frames."""
        s = self.s
        if s.replay_sam3:
            prompts = {c.label: c.prompt for c in s.concepts if c.label in s.sam3_lens_labels}

            def load_sam3() -> Sam3Observer:
                return Sam3Observer(str(s.sam3_python), prompts, s.sam3_model)

            return LazyModel(f"SAM 3 ({s.sam3_model})", load_sam3)
        path = s.replay_vla_path
        if not path:
            return None

        def load() -> SmolVLAObserver:
            return SmolVLAObserver(path, s.policy_device, s.overhead_camera_key, s.vla_camera_key, self.size_wh,
                                   s.task_description, "language")

        return LazyModel(f"SmolVLA ({path})", load)

    def _replay_should_wait(self) -> bool:
        """The replay lens and the ACT policy share the CPU: don't starve the arm while it moves."""
        return self.state == RUNNING and self.policy_kind == "act" and not self._preflight_pending

    async def stop(self) -> None:
        self._stop.set()
        for task in (self._track_task, self._live_sam_task, self._preflight_task):
            if task is not None:
                task.cancel()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 2.0)
        try:
            with self._lock:
                self.robot.hold()
        except Exception as e:  # noqa: BLE001
            log.warning("Could not hold robot on shutdown: %s", e)
        await asyncio.to_thread(self.robot.disconnect)
        lens = self.replay.vla.model if self.replay is not None and self.replay.vla is not None else None
        if isinstance(lens, Sam3Observer):
            lens.close()
        if self.meta is not None:
            await self.meta.aclose()

    # ================================================================== control thread
    def _run(self) -> None:
        period = 1.0 / self.s.control_hz
        last = time.perf_counter()
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001 - keep the loop alive, but stop moving
                log.exception("Control tick failed")
                with self._lock:
                    if self.state == RUNNING:
                        self._enter_hold(f"control error: {e}")
                time.sleep(0.2)
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)
            last = now
            time.sleep(max(0.0, period - (now - t0)))

    def _policy_step(self, obs: Observation, policy_frame: np.ndarray) -> PolicyStep:
        if isinstance(self.policy, MockPolicy):
            return self.policy.step(obs.qpos.copy(), policy_frame, self.robot)
        images = dict(obs.images_rgb)
        if policy_frame is not obs.overhead_bgr:
            images[self.s.overhead_camera_key] = cv2.cvtColor(policy_frame, cv2.COLOR_BGR2RGB)
        step = self.policy.step(obs.qpos.astype(np.float32), images)
        if self.grasp.dx_mm or self.grasp.dy_mm:
            step.action = step.action.copy()
            step.action[0] += self.grasp.pan_delta_deg(self.kin.fk(obs.qpos))
        return step

    def _tick(self) -> None:
        epoch = self._epoch
        obs = self.robot.read()  # outside the lock: camera reads can take a full frame period
        raw = obs.overhead_bgr
        with self._lock:
            if epoch != self._epoch:
                return  # reset() ran during the read: this observation belongs to the previous scene
            self.tick_count += 1
            self.last_q = obs.qpos.copy()
            policy_frame = self.stager.apply(raw) if self.stager.active else raw
            if self.suppressor.active:
                policy_frame = self.suppressor.apply(policy_frame)
            target = self.last_target if self.last_target is not None else obs.qpos.copy()
            variance = self.last_variance
            if self.state == RUNNING:
                if self._preflight_pending:  # pre-flight scan: keep still until the scene is staged
                    target, self.phase, variance = obs.qpos.copy(), "PREFLIGHT", 0.0
                elif self.executor.active:
                    target, self.recovery_note = self.executor.next()
                    self.phase, variance = "RECOVERY", 0.0
                    if not self.executor.active:
                        self._recovery_finished()
                elif self.policy is not None:
                    step = self._policy_step(obs, policy_frame)
                    target, variance = np.asarray(step.action, np.float64), step.chunk_variance
                    if step.saliency is not None:
                        self.saliency, self.saliency_method = step.saliency, step.saliency_method
                        if self.reference is not None:
                            self.attn_dev = self.reference.score(obs.qpos, step.saliency)
                    self.phase = (
                        self.policy.phase if isinstance(self.policy, MockPolicy)
                        else "OBSERVE" if self.camera_only else "ACT"
                    )
                else:
                    target = obs.qpos.copy()
                target = self.robot.send(clamp_joints(target))
            self.last_target, self.last_variance = target, variance

            tel = ServoTelemetry(
                timestamp=obs.timestamp,
                qpos_actual=_r(obs.qpos),
                qpos_target=_r(target),
                qvel=_r(obs.qvel),
                torque_load=_r(obs.load),
                action_chunk_variance=round(float(variance), 3),
                joint_error=round(joint_error(target, obs.qpos), 3) if self.state == RUNNING else 0.0,
                attention_deviation=round(self.attn_dev.score, 4) if self.attn_dev is not None else 0.0,
            )
            self.history.append(tel)

            if self.tick_count % LIVE_ATTRIBUTION_EVERY == 0:
                self._refresh_live_attribution()

            # camera-only: there is no arm to fail, so the triggers stay off and analysis is on demand
            if (self.state == RUNNING and not self.executor.active and self.phase != "DONE" and not self.camera_only
                    and not self._preflight_pending):
                event = self.detector.update(tel, self.live_attrs)
                if event is not None:
                    self._on_anomaly(event)
            if self.mock and self.state == RUNNING and (self.robot.delivered or self.phase == "DONE"):
                self.state, self.status_text = COMPLETE, "task complete: cube delivered to the green hex tile"
                self.robot.hold()
                self.hub.publish_event({"type": "state", "state": self.state, "status": self.status_text})
            state, phase = self.state, self.phase
            suppression = {
                "active": self.suppressor.active or (self.stager.active and self.stager.suppressed),
                "objects": [{"mask_id": k, "label": v} for k, v in self.suppressor.labels.items()],
                "staging": self.stager.info(),
            }
            recovery = {"active": self.executor.active, "note": self.recovery_note if self.executor.active else ""}

        jpeg = encode_jpeg(raw, self.s.stream_jpeg_quality)
        # What the policy saw is recorded too when it differs: the replay analyses the policy's own input.
        policy_jpeg = encode_jpeg(policy_frame, self.s.stream_jpeg_quality) if policy_frame is not raw else None
        rec = self.ring.push(obs.timestamp, jpeg, self.saliency, tel, phase, policy_jpeg)

        now = time.time()
        if now - self._last_publish < 0.9 / self.s.stream_fps or not self.hub.clients:
            return
        self._last_publish = now
        meta = {
            "type": "frame",
            "index": rec.index,
            "t": obs.timestamp,
            "state": state,
            "phase": phase,
            "status": self.status_text,
            "scenario": getattr(self.robot, "scenario", None),
            "fps": round(self.fps, 1),
            "telemetry": tel.model_dump(),
            "saliency": self._saliency_grid(),
            "suppression": suppression,
            "recovery": recovery,
            "grasp_offset_mm": self._grasp_offset(),
            "objects": self.live_views,  # refreshed every LIVE_ATTRIBUTION_EVERY ticks; frames may be coalesced
        }
        self.hub.publish_frame((meta, jpeg, policy_jpeg))

    def _saliency_grid(self) -> dict | None:
        if self.saliency is None:
            return None
        gw, gh = self.s.saliency_grid
        small = cv2.resize(self.saliency, (gw, gh), interpolation=cv2.INTER_AREA)
        data = np.clip(small * 255, 0, 255).astype(np.uint8).tobytes()
        return {"w": gw, "h": gh, "method": self.saliency_method, "data": base64.b64encode(data).decode()}

    def _grasp_offset(self) -> list[float]:
        if isinstance(self.policy, MockPolicy):
            return _r(self.policy.grasp_offset_mm, 1)
        return [round(self.grasp.dx_mm, 1), round(self.grasp.dy_mm, 1)]

    def _refresh_live_attribution(self) -> None:
        """Live attention shares for the hijack trigger: simulator masks in mock, last SAM pass on hardware."""
        if self.mock:
            objs = [
                SegmentedObject(i, g.label, g.prompt, g.mask, None, "sim-ground-truth")
                for i, g in enumerate(self._mock_truth())
            ]
        else:
            objs = self._sam_live_objects
        self.live_objects = objs
        self.live_attrs = compute_attributions(objs, self.saliency, self.homography, self.attn_dev) if objs and self.saliency is not None else []
        shares = {a.mask_id: a.attention_share for a in self.live_attrs}
        self.live_views = [
            {**o.view().model_dump(), "source": o.source, "attention_share": shares.get(o.mask_id, 0.0)} for o in objs
        ]

    def _mock_truth(self) -> list:
        """Simulator masks of the scene the policy sees: under demo staging the distractor sits on the goal."""
        objs = self.robot.ground_truth_objects()
        info = self.stager.info()
        if info is None or info["suppressed"]:
            return objs
        dx, dy = info["to_bbox"][0] - info["from_bbox"][0], info["to_bbox"][1] - info["from_bbox"][1]
        shift = np.float32([[1, 0, dx], [0, 1, dy]])
        h, w = self.size_wh[1], self.size_wh[0]
        return [
            type(o)(o.label, o.prompt, cv2.warpAffine(o.mask.astype(np.uint8), shift, (w, h)).astype(bool))
            if o.label == info["label"] else o
            for o in objs
        ]

    def _enter_hold(self, reason: str) -> None:
        self.state = HOLD
        self.status_text = f"holding: {reason}"
        self.executor.cancel()
        self.last_target = None
        try:
            self.robot.hold()
        except Exception as e:  # noqa: BLE001
            log.error("hold() failed: %s", e)
        self.hub.publish_event({"type": "state", "state": self.state, "status": self.status_text})

    def _on_anomaly(self, event: AnomalyEvent) -> None:
        self.last_anomaly = event
        log.warning("Anomaly: %s", "; ".join(event.reasons))
        self._enter_hold("anomaly detected: " + "; ".join(event.reasons))
        self.hub.publish_event({"type": "anomaly", "event": event.model_dump()})
        if self.s.auto_analyze and self.loop is not None:
            fut = asyncio.run_coroutine_threadsafe(self.analyze("auto", event), self.loop)
            fut.add_done_callback(self._log_future_error)

    def _forced_anomaly(self, rec: FrameRecord, attributions, epoch: int) -> AnomalyEvent:
        """Demo: the operator pressed Analyze with no anomaly on record. The press becomes the failure: the arm is
        held, and the run is diagnosed, replayed and self-corrected like a detected one (labelled as forced)."""
        tel = rec.telemetry
        distractors = [a for a in attributions if a.label not in NON_DISTRACTOR_LABELS | {"hand", "obstacle"}]
        top = max(distractors, key=lambda a: a.attention_share, default=None)
        reasons = ["DEMO: failure triggered by the operator"]
        if top is not None:
            reasons.append(f"attention share on non-target '{top.label}' #{top.mask_id} A={top.attention_share:.2f}")
        event = AnomalyEvent(
            timestamp=rec.timestamp, reasons=reasons, joint_error=round(tel.joint_error, 2),
            chunk_variance=round(tel.action_chunk_variance, 2),
            max_attention_share=round(top.attention_share, 3) if top is not None else 0.0,
            attention_deviation=round(tel.attention_deviation, 3),
        )
        with self._lock:
            if epoch != self._epoch:
                return event
            self.last_anomaly = event
            self.status_text = "holding: " + "; ".join(reasons)
        log.warning("Anomaly: %s", "; ".join(reasons))
        self.hub.publish_event({"type": "state", "state": self.state, "status": self.status_text})
        self.hub.publish_event({"type": "anomaly", "event": event.model_dump()})
        return event

    @staticmethod
    def _log_future_error(fut) -> None:
        if not fut.cancelled() and fut.exception() is not None and not isinstance(fut.exception(), ConflictError):
            log.error("Auto analysis failed: %r", fut.exception())

    def _recovery_finished(self) -> None:
        self.status_text = "recovery trajectory complete: policy resumed"
        self.recovery_note = ""
        if self.policy is not None:
            self.policy.queue.clear()
        self.detector.reset()
        self.hub.publish_event({"type": "state", "state": self.state, "status": self.status_text})

    # ================================================================== XAI pipeline (event loop)
    def _diagnosis_context(self, anomaly, attributions, objects, saliency, rec) -> DiagnosisContext:
        hist = list(self.history) or [rec.telemetry]
        if anomaly is not None:  # describe the arm at the anomaly, not after HOLD zeroed its error
            hist = [t for t in hist if t.timestamp <= anomaly.timestamp] or hist
        latest = hist[-1]
        loads = np.abs(np.array([t.torque_load for t in hist]))
        recent = [t for t in hist if t.timestamp >= latest.timestamp - 2.0]
        bias = None
        target = next((o for o in objects if o.label == "target"), None)
        if target is not None and saliency is not None:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
            region = cv2.dilate(target.mask.astype(np.uint8), k).astype(bool)
            # Measure on the frame where the policy saw the target most clearly (most saliency mass on it):
            # once the gripper hovers above the target it occludes it from the overhead camera.
            best, best_mass = saliency, float(saliency[region].sum())
            for r in self.ring.snapshot()[::10]:
                if r.saliency_u8 is None:
                    continue
                sal = r.saliency(self.size_wh)
                mass = float(sal[region].sum())
                if mass > best_mass:
                    best, best_mass = sal, mass
            perceived = saliency_centroid(best, region)
            if perceived is not None:
                px = np.array(self.homography.pixel_to_workspace(*perceived))
                true = np.array(self.homography.pixel_to_workspace(*centroid(target.mask)))
                bias = tuple(_r(px - true, 1))
        active = [f"mask suppression of {v} (#{k})" for k, v in self.suppressor.labels.items()]
        if self._grasp_offset() != [0.0, 0.0]:
            active.append(f"grasp offset {self._grasp_offset()} mm")
        return DiagnosisContext(
            anomaly=anomaly,
            attributions=attributions,
            latest=latest,
            peak_joint_error=max(t.joint_error for t in hist),
            peak_chunk_variance=max(t.action_chunk_variance for t in hist),
            peak_load=_r(loads.max(axis=0), 1),
            gripper_xy_mm=tuple(self.kin.fk(np.array(latest.qpos_actual))),
            policy_phase=rec.phase,
            saliency_method=self.saliency_method,
            target_grasp_bias_mm=bias,
            active_interventions=active,
            mean_joint_error_2s=float(np.mean([t.joint_error for t in recent])),
            mean_abs_load_2s=_r(np.abs(np.array([t.torque_load for t in recent])).mean(axis=0), 1),
        )

    async def analyze(self, trigger: str, anomaly: AnomalyEvent | None = None) -> AnalysisResult:
        if self._analysis_lock.locked():
            raise ConflictError("An analysis is already running")
        async with self._analysis_lock:
            with self._lock:
                epoch = self._epoch
                if self.state == RUNNING:
                    self._enter_hold("analysing")
                anomaly = anomaly or self.last_anomaly  # a manual press while held still explains the anomaly
                # Demo: an operator's Analyze with nothing detected is treated as the failure itself.
                forced = (anomaly is None and trigger != "auto" and self.s.demo_manual_failure and not self.camera_only
                          and not self._awaiting_start)
            self.hub.publish_event({"type": "analysis_started", "trigger": trigger})
            rec = self.ring.latest()
            if rec is None:
                raise ConflictError("No frames recorded yet")
            # The run as of the failure: SAM and Spark take ~10 s, and the ring keeps overwriting meanwhile.
            run_records = self.ring.snapshot()
            # Under demo staging, explain what the policy saw (the distractor on the goal), not the raw scene.
            frame = rec.policy_frame_bgr() if self.stager.pasting else rec.frame_bgr()
            saliency = rec.saliency(self.size_wh)
            lat: dict[str, float] = {}

            t0 = time.perf_counter()
            clip = None
            if self.s.use_sam_video:
                clip = await asyncio.to_thread(RingBuffer.export_mp4, self.ring.snapshot(self.s.sam_clip_frames), self.s.control_hz)
            objects, sam_source = await self.sam.segment(frame, clip)
            lat["sam_ms"] = round((time.perf_counter() - t0) * 1000)

            deviation = self.reference.score(rec.telemetry.qpos_actual, saliency) if self.reference is not None else None
            attributions = compute_attributions(objects, saliency, self.homography, deviation)
            if forced:
                anomaly = self._forced_anomaly(rec, attributions, epoch)
            ctx = self._diagnosis_context(anomaly, attributions, objects, saliency, rec)
            caption = "GlassBox keyframe | " + ("; ".join(anomaly.reasons) if anomaly else "manual analysis")
            composite = composite_keyframe(frame, saliency, objects, attributions, caption[:140])
            t1 = time.perf_counter()
            diagnosis, diag_source = await self.spark.diagnose(composite, ctx)
            lat["spark_ms"] = round((time.perf_counter() - t1) * 1000)
            if forced and not diagnosis.failure_detected:
                diagnosis = forced_diagnosis(ctx) or diagnosis

            plan, plan_source, plan_status = None, None, "not_needed"
            if diagnosis.intervention_type == "WAYPOINT_BYPASS":
                plan = self.wonder.baseline(self.last_q.copy(), objects, attributions, self.homography)
                plan_source = "local-fallback"
                plan_status = "pending" if self.meta.enabled else "ready"

            aid = uuid.uuid4().hex[:8]
            result = AnalysisResult(
                analysis_id=aid,
                created_at=time.time(),
                trigger=trigger,
                anomaly=anomaly,
                objects=[o.view() for o in objects],
                attributions=attributions,
                diagnosis=diagnosis,
                diagnosis_source=diag_source,
                sam_source=sam_source,
                plan=plan,
                plan_source=plan_source,
                plan_status=plan_status,
                plan_path_px=self._plan_path_px(plan),
                keyframe_url=f"/api/analysis/{aid}/keyframe.jpg",
                latency_ms=lat,
                evidence=ctx.evidence(),
            )
            self.analyses[aid] = AnalysisRecord(
                result, objects, frame, saliency, encode_jpeg(composite, 88), ctx,
                qpos=np.asarray(rec.telemetry.qpos_actual, np.float64),
            )
            with self._lock:
                stale = epoch != self._epoch
                if not stale:
                    self.latest_analysis_id = aid
                    self.status_text = f"diagnosed: {diagnosis.failure_type} -> {diagnosis.intervention_type} (awaiting approval)"
                rerun = stale and self.state == HOLD and self.last_anomaly is not None and self.s.auto_analyze
            if stale:
                # The scene was reset mid-analysis: don't offer a fix for a scene that no longer exists.
                log.info("Discarding analysis %s: the scenario was reset while it ran", aid)
                if rerun:  # an anomaly in the new scene was blocked by this analysis; explain that one instead
                    # Starts after this coroutine returns and releases the analysis lock.
                    asyncio.create_task(self.analyze("auto", self.last_anomaly)).add_done_callback(self._log_future_error)
                return result
            self._publish_analysis(aid, "analysis")
            if plan_status == "pending":
                asyncio.create_task(self._refine_plan(aid))
            asyncio.create_task(self._render_counterfactual(aid))
            self._start_replay(trigger, anomaly, rec.timestamp, aid, objects, run_records)
            return result

    # ================================================================== offline replay
    def _start_replay(self, trigger: str, anomaly: AnomalyEvent | None, t_latest: float, aid: str,
                      objects: list[SegmentedObject], run_records: list[FrameRecord]) -> None:
        if self.replay is None:
            return
        diag = self.analyses[aid].result.diagnosis
        culprit = next((o.label for o in objects if o.mask_id == diag.culprit_mask_id), None)
        try:
            last = run_records[-1].timestamp if run_records else float("-inf")
            records = run_records + [r for r in self.ring.snapshot() if r.timestamp > last]  # + what came since
            run = Run.from_records(
                records, fps=self.s.replay_fps, control_hz=self.s.control_hz,
                window_s=self.s.replay_window_s, tail_s=self.s.replay_tail_s, t_start=self._run_t0,
                t_failure=anomaly.timestamp if anomaly is not None else t_latest, trigger=trigger,
                task=self.s.task_description, reasons=anomaly.reasons if anomaly is not None else [],
                staging=self.stager.info() if self.stager.pasting else None,
                spark={"failure_type": diag.failure_type, "headline": diag.headline_summary, "culprit_label": culprit},
                analysis_id=aid,
            )
        except ValueError as e:
            log.warning("No replay for analysis %s: %s", aid, e)
            return
        log.info("Replaying %d frames (%.1f s) of the run offline", len(run.frames), run.frames[-1].t - run.frames[0].t)
        self.replay.start(run)

    async def _on_breakdown(self, r: ReplayResult) -> None:
        """Wonder's verdict is in: self-correct if it says GlassBox can, otherwise say what a human must do."""
        b = r.breakdown
        rec = self.analyses.get(r.analysis_id) if r.analysis_id else None
        if b is None:
            return
        automatic = b.self_correctable and not b.needs_human and b.intervention_type in AUTOMATIC
        if rec is None:
            r.self_correction, r.self_correction_note = "unavailable", "This run's scene is gone: nothing to correct live."
        elif not automatic:
            r.self_correction = "declined"
            r.self_correction_note = f"A human is needed: {b.human_action or 'see the breakdown'}"
        elif rec.applied or self.state != HOLD:
            r.self_correction = "unavailable"
            r.self_correction_note = "The arm has moved on since this failure (a fix was applied or the scene reset)."
        elif not self.self_correct:
            r.self_correction = "pending"
            r.self_correction_note = f"GlassBox can fix this by itself ({b.intervention_type}): approve to apply."
        else:
            await self.apply_replay_fix(r)

    async def apply_replay_fix(self, r: ReplayResult) -> dict:
        b = r.breakdown
        if b is None or not r.analysis_id or r.analysis_id not in self.analyses:
            raise ConflictError("This replay has no live failure to correct")
        if b.intervention_type not in AUTOMATIC:
            raise ConflictError(f"{b.intervention_type} needs a human: {b.human_action}")
        labels = [b.culprit_label] if b.culprit_label else None
        out = await self.apply_fix(r.analysis_id, override=b.intervention_type, labels=labels,
                                   params=b.intervention_parameters)
        r.self_correction, r.self_correction_note = "applied", out["message"]
        self.hub.publish_event({"type": "replay_update", "replay": json.loads(r.model_dump_json())})
        return out

    def set_self_correct(self, on: bool) -> None:
        self.self_correct = on
        self.hub.publish_event({"type": "settings", "self_correct": on})

    def replay_saved_run(self, rid: str) -> ReplayResult:
        if self.replay is None:
            raise ConflictError("Replay is disabled")
        shown = self.replay.show(rid)
        if shown is not None:
            return shown
        run = self.replay.get_run(rid)
        if run is None:
            raise KeyError(rid)
        return self.replay.start(run, reuse_cached=True)

    def list_replays(self) -> list[dict]:
        return list_runs(self.s.runs_dir)

    # ================================================================== demo staging
    def _schedule_preflight(self) -> None:
        if not self.s.demo_distractor_on_goal or self.camera_only or self.loop is None:
            return

        def go() -> None:
            if self._preflight_task is not None and not self._preflight_task.done():
                self._preflight_task.cancel()
            self._preflight_task = asyncio.create_task(self._preflight())

        self.loop.call_soon_threadsafe(go)

    async def _preflight(self) -> None:
        """Before the run: SAM 3.1 finds the distractor and the goal, and the stager shows the policy one on the other."""
        with self._lock:
            epoch = self._epoch
            self._preflight_pending = True
            self.status_text = "pre-flight: SAM 3.1 is locating the distractor and the goal"
        self.hub.publish_event({"type": "state", "state": self.state, "status": self.status_text})
        note = "pre-flight: no distractor in view, running normally"
        try:
            await asyncio.sleep(0.5)  # a fresh frame of the reset scene
            rec = self.ring.latest()
            if rec is not None:
                objs, source = await asyncio.wait_for(self.sam.segment(rec.frame_bgr()), self.s.preflight_timeout_s)
                note = self._configure_staging(rec.frame_bgr(), objs, epoch) or note
        except asyncio.TimeoutError:
            note = "pre-flight: SAM 3.1 timed out, running without staging"
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - staging is optional; never block the run on it
            log.exception("Pre-flight failed")
            note = f"pre-flight failed ({e}): running without staging"
        finally:
            with self._lock:
                if epoch == self._epoch:
                    self._preflight_pending = False
                    self.detector.reset()
                    latest = self.ring.latest()
                    self._run_t0 = latest.timestamp if latest is not None else 0.0
                    if self.state == RUNNING:
                        self.status_text = note
                    elif self._awaiting_start:
                        self.status_text = f"{note} · {START_PROMPT}"
            self.hub.publish_event({"type": "staging", "staging": self.stager.info()})
            self.hub.publish_event({"type": "state", "state": self.state, "status": self.status_text})

    def _configure_staging(self, frame: np.ndarray, objs: list[SegmentedObject], epoch: int) -> str | None:
        """Arm the stager from one segmentation of the raw scene. Returns a status note, or None if nothing to stage."""
        books = [o for o in objs if o.label == "distractor"]
        goal = next((o for o in objs if o.label == "goal"), None)
        if not books or goal is None:
            return None if not books else "pre-flight: distractor found but no goal tile, running without staging"
        book = max(books, key=lambda o: int(o.mask.sum()))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        near_goal = cv2.dilate(goal.mask.astype(np.uint8), k).astype(bool)
        if (book.mask & near_goal).sum() > 0.3 * book.mask.sum():
            return "pre-flight: the distractor is already on the goal, no staging needed"
        with self._lock:
            if epoch != self._epoch:
                return None
            self.stager.configure(frame, book.mask, centroid(goal.mask), book.label)
        return f"DEMO STAGING: the policy is shown the {book.prompt} on the goal tile"

    def _publish_analysis(self, aid: str, kind: str) -> None:
        self.hub.publish_event({"type": kind, "result": json.loads(self.analyses[aid].result.model_dump_json())})

    def _telemetry_digest(self) -> list[list]:
        hist = list(self.history)
        if not hist:
            return []
        t_end = hist[-1].timestamp
        step = max(1, int(self.s.control_hz / 2))  # 2 Hz
        return [
            [round(t.timestamp - t_end, 2), _r(t.qpos_actual, 1), _r(t.qpos_target, 1), _r(t.torque_load, 0),
             round(t.joint_error, 1), round(t.action_chunk_variance, 1)]
            for t in hist[::step]
        ]

    async def _refine_plan(self, aid: str) -> None:
        rec = self.analyses[aid]
        r = rec.result
        source_code = inspect.getsource(ControlLoop._tick) + "\n" + inspect.getsource(ControlLoop.apply_fix)
        t0 = time.perf_counter()
        plan, plan_source, notes = await self.wonder.plan(
            r.diagnosis, self.last_q.copy(), rec.objects, r.attributions, self.homography,
            self._telemetry_digest(), source_code,
        )
        r.latency_ms["wonder_ms"] = round((time.perf_counter() - t0) * 1000)
        if rec.applied:
            plan.safety_notes += " (Arrived after the baseline plan was already executed.)"
        r.plan, r.plan_source, r.plan_status = plan, plan_source, "ready"
        r.plan_path_px = self._plan_path_px(plan)
        if notes:
            log.info("Wonder plan notes: %s", notes)
        self._publish_analysis(aid, "analysis_update")

    def _plan_path_px(self, plan: RecoveryPlan | None) -> list[tuple[float, float]] | None:
        if plan is None or not plan.waypoints:
            return None
        pts = [self.kin.fk(self.last_q)] + [self.kin.fk(np.array(w.qpos)) for w in plan.waypoints]
        return [[round(float(c), 1) for c in self.homography.workspace_to_pixel(x, y)] for x, y in pts]

    async def _render_counterfactual(self, aid: str) -> None:
        rec = self.analyses[aid]
        diag = rec.result.diagnosis
        suspect = next((o for o in rec.objects if o.mask_id == diag.culprit_mask_id), None)
        failure_type, culprit = cfx.choose_edit(diag.failure_type, rec.objects, suspect) or (diag.failure_type, suspect)

        def publish(cf: Counterfactual) -> None:
            rec.result.counterfactual = cf
            self._publish_analysis(aid, "analysis_update")

        await self.counterfactual(
            failure_type, rec.frame, rec.qpos if rec.qpos is not None else self.last_q.copy(), rec.objects, culprit,
            f"/api/analysis/{aid}/counterfactual.jpg", publish, rec.counterfactual_views.__setitem__,
        )

    # ------------------------------------------------------------------ counterfactuals
    def attenders(self) -> list[tuple[str, str, Callable]]:
        """``(kind, title, attend)`` for the running policy; ``attend(state, frame_bgr) -> (saliency, method, chunk)``."""
        p = self.policy
        if isinstance(p, ACTRunner):
            return [("act", "ACT · live attention", p.attend)]
        if isinstance(p, MockPolicy):
            return [("act", "Mock policy · colour attention", p.attend)]
        if isinstance(p, SmolVLAObserver):
            return [(f"vla_{p.map_kind}", f"SmolVLA · {p.map_kind} attention", p.attend)]
        if isinstance(p, CameraObserver):
            return [("saliency", "Visual saliency · no policy", p.attend)]
        return []

    def _chunk_path_px(self, q0: np.ndarray, chunk: np.ndarray | None, scale: float = 1.0) -> list[list[float]] | None:
        """The gripper path an action chunk predicts, in overhead pixels (from ``q0``), times ``scale``."""
        if chunk is None or len(chunk) == 0:
            return None
        qs = [q0] + list(chunk[:: max(1, len(chunk) // 20)]) + [chunk[-1]]
        return [[round(float(c) * scale, 1) for c in self.homography.workspace_to_pixel(*self.kin.fk(np.asarray(q)))]
                for q in qs]

    async def counterfactual(
        self, failure_type: str, frame: np.ndarray, qpos: np.ndarray, objects: list[SegmentedObject],
        culprit: SegmentedObject | None, url: str, publish: Callable[[Counterfactual], None],
        store: Callable[[str, bytes], None], attenders: list[tuple[str, str, Callable]] | Callable | None = None,
        frame_index: int | None = None,
    ) -> Counterfactual | None:
        """Render the scene without the culprit (Muse Image inside SAM's region), then ask each model where it
        would look, and where it would reach, in that scene at the same joint state. Never raises."""
        plan = await asyncio.to_thread(cfx.plan_edit, failure_type, frame, objects, culprit, self.plate)
        if plan is None:
            return None
        target = next((o for o in objects if o.label == "target"), None)
        target_name = target.prompt if target else "red cube"
        prompt = counterfactual_prompt(
            failure_type, culprit.prompt if culprit else None, image_region(plan.bbox, self.size_wh), target_name,
            erased=plan.pre_erase,
        )
        if prompt is None:
            return None
        cf = Counterfactual(failure_type=failure_type, frame=frame_index, culprit_label=plan.culprit_label,
                            removed=plan.removed, region_polygons=cfx.region_polygons(plan.region))
        publish(cf)
        try:
            t0 = time.perf_counter()
            img, cf.source = await self.images.render(frame, prompt, plan.region, self.plate, plan.culprit_mask, plan.pre_erase)
            cf.latency_ms["image_ms"] = round((time.perf_counter() - t0) * 1000)
            store("original", encode_jpeg(frame, 92))
            store("image", encode_jpeg(img, 92))
            cf.image_url, cf.status = f"{url}?v={int(time.time() * 1000)}", "attending"
            publish(cf)

            h, w = frame.shape[:2]

            def fit(sal):
                return None if sal is None else cv2.resize(sal.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

            if attenders is None:
                attenders = self.attenders()
            elif callable(attenders):  # resolved late, e.g. after a model finishes loading
                attenders = await attenders()
            for kind, title, attend in attenders:
                t1 = time.perf_counter()
                s0, _, c0 = await asyncio.to_thread(attend, qpos, frame)
                s1, _, c1 = await asyncio.to_thread(attend, qpos, img)
                cf.latency_ms[f"{kind}_ms"] = round((time.perf_counter() - t1) * 1000)
                if s0 is None or s1 is None:
                    continue
                s0, s1 = fit(s0), fit(s1)
                cf.maps.append(cfx.compare_maps(kind, title, s0, s1, objects, self.s.saliency_grid))
                if cf.path_before_px is None and c0 is not None and c1 is not None:
                    k = w / self.size_wh[0]  # a recorded run may be at another resolution than the homography
                    cf.path_before_px, cf.path_after_px = self._chunk_path_px(qpos, c0, k), self._chunk_path_px(qpos, c1, k)
                    cf.path_shift_px = cfx.path_shift(cf.path_before_px, cf.path_after_px)
                if len(cf.maps) == 1:  # server-rendered views of the primary model, for the live inspector
                    def norm(s):
                        return s / (float(s.max()) + 1e-9)

                    before = draw_path(heatmap_overlay(frame, norm(s0), 0.5), cf.path_before_px or [], (60, 60, 255))
                    after = draw_path(heatmap_overlay(img, norm(s1), 0.5), cf.path_after_px or [], (80, 255, 80))
                    store("attention_before", encode_jpeg(before, 88))
                    store("attention_after", encode_jpeg(after, 88))
                publish(cf)
            reach = None
            if target is not None and cf.path_before_px and cf.path_after_px:
                u, v = centroid(target.mask)
                reach = tuple(float(np.hypot(p[-1][0] - u, p[-1][1] - v)) for p in (cf.path_before_px, cf.path_after_px))
            cf.verdict = " ".join([
                cfx.verdict(plan.removed, cf.maps[0] if cf.maps else None, plan.culprit_label, cf.path_shift_px,
                            reach, target_name),
                *(cfx.map_sentence(m, plan.culprit_label, plan.removed, target_name) for m in cf.maps[1:]),
            ])
            a = cfx.assess(cf.maps[0] if cf.maps else None, plan.culprit_label, cf.path_shift_px)
            cf.outcome = a.outcome if a else None
            cf.status = "ready"
        except Exception as e:  # noqa: BLE001 - a counterfactual is a bonus, never a crash
            log.exception("Counterfactual failed")
            cf.status, cf.verdict = "failed", f"Counterfactual failed: {e}"
        publish(cf)
        return cf

    # ================================================================== operator commands
    async def apply_fix(
        self, analysis_id: str, override: str | None = None, labels: list[str] | None = None,
        params: dict | None = None,
    ) -> dict:
        """Apply an intervention for an analysis. ``labels``/``params`` come from a replay breakdown, which
        names objects by SAM label (its masks are from other frames than the analysis keyframe)."""
        rec = self.analyses.get(analysis_id)
        if rec is None:
            raise KeyError(analysis_id)
        diag, plan = rec.result.diagnosis, rec.result.plan
        itype = override or diag.intervention_type
        params = {**diag.intervention_parameters, **(params or {})}
        tracked: SegmentedObject | None = None
        with self._lock:
            if self.state == ESTOP:
                raise ConflictError("E-stop is active: resume torque first")
            if itype == "INPAINT_MASK_SUPPRESSION":
                if labels is None and isinstance(params.get("label"), str):
                    labels = [params["label"]]
                if labels:
                    objs = [o for o in rec.objects if o.label in labels]
                else:
                    ids = [int(params["mask_id"])] if "mask_id" in params else []
                    ids += (plan.suppress_mask_ids if plan else []) or ([diag.culprit_mask_id] if diag.culprit_mask_id is not None else [])
                    objs = [o for o in rec.objects if o.mask_id in ids]
                staged = self.stager.pasting and (
                    any(o.label == self.stager.label for o in objs) or (labels is not None and self.stager.label in labels)
                )
                if staged:
                    # The policy saw a staged copy of the distractor: stop staging it (the real one stays erased).
                    self.stager.suppress()
                    objs = [o for o in objs if o.label != self.stager.label]
                elif not objs:  # fall back to the most-attended non-target object
                    shares = {a.mask_id: a.attention_share for a in rec.result.attributions}
                    cands = [o for o in rec.objects if o.label not in NON_DISTRACTOR_LABELS]
                    objs = [max(cands, key=lambda o: shares.get(o.mask_id, 0))] if cands else []
                if not objs and not staged:
                    raise ConflictError("No mask to suppress")
                for obj in objs:
                    self.suppressor.set_mask(obj.mask_id, obj.mask, obj.label)
                    self._tracked[obj.mask_id] = obj
                tracked = objs[0] if objs else None
                names = [f"#{o.mask_id} {o.label}" for o in objs] + ([self.stager.label] if staged else [])
                message = "Suppressing " + ", ".join(names) + " in the policy's view"
            elif itype == "WAYPOINT_BYPASS":
                if plan is None or not plan.waypoints:
                    plan = self.wonder.baseline(self.last_q.copy(), rec.objects, rec.result.attributions, self.homography)
                    rec.result.plan, rec.result.plan_source, rec.result.plan_status = plan, "local-fallback", "ready"
                    rec.result.plan_path_px = self._plan_path_px(plan)
                ticks = self.executor.load(self.last_q.copy(), plan.waypoints, self.s.control_hz)
                message = f"Executing {len(plan.waypoints)}-waypoint bypass ({ticks / self.s.control_hz:.1f} s, {rec.result.plan_source})"
            elif itype == "RETRY_GRASP_OFFSET":
                dx = float(params.get("dx_mm", plan.grasp_offset_mm[0] if plan else 0.0))
                dy = float(params.get("dy_mm", plan.grasp_offset_mm[1] if plan else 0.0))
                if isinstance(self.policy, MockPolicy):
                    self.policy.grasp_offset_mm = np.array([dx, dy])
                else:
                    self.grasp.set(dx, dy)
                message = f"Grasp target shifted by ({dx:+.1f}, {dy:+.1f}) mm"
            else:
                if self.mock and diag.failure_type == "OCCLUSION_OOD":
                    self.robot.clear_operator_interference()
                message = "Workspace cleared by operator; resuming policy"
            if self.policy is not None:
                self.policy.queue.clear()
            self.detector.reset()
            self.last_target, self.last_anomaly = None, None
            self.state = RUNNING
            self.status_text = message
            rec.applied = True
        log.info("Fix applied (%s): %s", itype, message)
        self.hub.publish_event({"type": "fix_applied", "analysis_id": analysis_id, "intervention_type": itype, "message": message})
        self.hub.publish_event({"type": "state", "state": RUNNING, "status": message})
        if tracked is not None and self.meta.enabled and self.s.track_refresh_s > 0 and tracked.source == "sam-3.1":
            if self._track_task is None or self._track_task.done():
                self._track_task = asyncio.create_task(self._track_suppressed())
        return {"ok": True, "intervention_type": itype, "message": message}

    async def _track_suppressed(self) -> None:
        """Keep suppressed masks locked on their objects by re-segmenting them periodically with SAM 3.1."""
        while self.suppressor.active and self._tracked:
            await asyncio.sleep(self.s.track_refresh_s)
            rec = self.ring.latest()
            if rec is None:
                continue
            frame = rec.frame_bgr()
            for mid, obj in list(self._tracked.items()):
                mask = await self.sam.refresh_track(frame, obj)
                if mask is not None and mask.any() and mid in self.suppressor.labels:
                    obj.mask = mask
                    self.suppressor.set_mask(mid, mask, obj.label)

    async def _live_sam_loop(self) -> None:
        """Hardware mode: periodic SAM pass so attention shares (and the hijack trigger) are live.

        Segments what the policy sees (so the shares match its input). With demo staging on and nothing
        staged yet, a distractor that appears mid-run in the raw scene arms the stager."""
        while True:
            await asyncio.sleep(self.s.live_sam_interval_s)
            rec = self.ring.latest()
            if rec is None or self.state != RUNNING or self._preflight_pending:
                continue
            epoch = self._epoch
            objs, source = await self.sam.segment(rec.policy_frame_bgr())
            if source == "local-fallback":
                continue
            with self._lock:
                self._sam_live_objects = objs
            if self.s.demo_distractor_on_goal and not self.stager.active and rec.policy_jpeg is None:
                note = self._configure_staging(rec.frame_bgr(), objs, epoch)
                if note and self.stager.active:
                    with self._lock:
                        self.status_text = note
                    self.hub.publish_event({"type": "staging", "staging": self.stager.info()})

    def clear_interventions(self) -> None:
        with self._lock:
            self.suppressor.clear()
            if self.stager.active:
                self.stager.suppressed = False  # staging is the demo's scene, not an intervention: restore it
            self._tracked.clear()
            self.executor.cancel()
            self.grasp.clear()
            if isinstance(self.policy, MockPolicy):
                self.policy.grasp_offset_mm = np.zeros(2)
            self.status_text = "interventions cleared"
        self.hub.publish_event({"type": "state", "state": self.state, "status": self.status_text})

    def reset(self, scenario: str | None = None) -> None:
        with self._lock:
            self._epoch += 1
            if self.mock:
                self.robot.set_scenario(scenario or self.robot.scenario)
            elif scenario:
                raise ConflictError("Scenarios only exist in --mock-hardware mode")
            if self.policy is not None:
                self.policy.reset()
            self.suppressor.clear()
            self.stager.clear()
            self._preflight_pending = self.s.demo_distractor_on_goal and not self.camera_only
            self._run_t0 = 0.0
            self._tracked.clear()
            self.executor.cancel()
            self.grasp.clear()
            self.detector.reset()
            self.ring.clear()
            self.latest_analysis_id = None  # past analyses stay readable by id; Q&A starts fresh
            self.history.clear()
            self.saliency, self.last_target, self.last_anomaly, self.attn_dev = None, None, None, None
            self.live_objects, self.live_attrs, self.live_views = [], [], []
            if self.state == ESTOP:
                self.robot.resume_torque()
            self._awaiting_start = self.s.start_paused and self.policy is not None and not self.camera_only
            if self._awaiting_start:
                self.state = HOLD
                self.robot.hold()
                self.status_text = f"reset{f' to scenario {scenario}' if scenario else ''}: {START_PROMPT}"
            else:
                self.state = RUNNING
                self.status_text = f"reset{f' to scenario {scenario}' if scenario else ''}: policy running"
        self.hub.publish_event({"type": "reset", "scenario": getattr(self.robot, "scenario", None)})
        self.hub.publish_event({"type": "state", "state": self.state, "status": self.status_text})
        self._schedule_preflight()

    def hold(self) -> None:
        """Operator pause: the arm holds position (and a replay paused for the arm resumes its SmolVLA pass)."""
        with self._lock:
            if self.state == RUNNING:
                self._enter_hold("paused by operator")

    def estop(self) -> None:
        with self._lock:
            self.executor.cancel()
            self.robot.estop()
            self.state, self.status_text = ESTOP, "E-STOP: torque disabled"
            self.last_target = None
        self.hub.publish_event({"type": "state", "state": self.state, "status": self.status_text})

    def resume(self) -> None:
        with self._lock:
            if self.state == ESTOP:
                self.robot.resume_torque()
            if self.state in (ESTOP, HOLD):
                if self.policy is not None:
                    self.policy.queue.clear()
                self.detector.reset()
                self.last_target, self.last_anomaly = None, None
                self.state, self.status_text = RUNNING, "resumed by operator"
                if self._awaiting_start:  # the run starts now, not when the server or the reset did
                    self._awaiting_start, self.status_text = False, "policy started by operator"
                    latest = self.ring.latest()
                    if not self._preflight_pending and latest is not None:
                        self._run_t0 = latest.timestamp
        self.hub.publish_event({"type": "state", "state": self.state, "status": self.status_text})

    async def ask(self, question: str) -> AskResponse:
        rec = self.analyses.get(self.latest_analysis_id) if self.latest_analysis_id else None
        latest = self.ring.latest()
        image, ctx = None, None
        if rec is not None:
            image, ctx = cv2.imdecode(np.frombuffer(rec.keyframe_jpeg, np.uint8), cv2.IMREAD_COLOR), rec.ctx
        elif latest is not None:
            image = latest.frame_bgr()
            with self._lock:
                attrs = list(self.live_attrs)
            ctx = self._diagnosis_context(self.last_anomaly, attrs, self.live_objects, self.saliency, latest)
        answer, source = await self.spark.answer(question, ctx, rec.result.diagnosis if rec else None, image)
        return AskResponse(question=question, answer=answer, source=source)

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "status": self.status_text,
                "phase": self.phase,
                "mock_hardware": self.mock,
                "camera_only": self.camera_only,
                "scenario": getattr(self.robot, "scenario", None),
                "scenarios": SCENARIOS if self.mock else {},
                "policy": self.policy_kind,
                "saliency_method": self.saliency_method,
                "meta_api_enabled": bool(self.meta and self.meta.enabled),
                "models": {
                    "sam": self.s.sam_model, "spark": self.s.spark_model, "wonder": self.s.wonder_model,
                    "image": self.s.image_model, "voice": self.s.voice_model,
                },
                "suppression": [{"mask_id": k, "label": v} for k, v in self.suppressor.labels.items()],
                "staging": self.stager.info(),
                "demo_staging_enabled": self.s.demo_distractor_on_goal,
                "preflight": self._preflight_pending,
                "awaiting_start": self._awaiting_start,
                "self_correct": self.self_correct,
                "replay": {
                    "enabled": self.replay is not None,
                    "latest_id": self.replay.latest_id if self.replay else None,
                    "vla": self.replay.vla_status() if self.replay else "off",
                    "vla_path": self.s.sam3_model if self.s.replay_sam3 else self.s.replay_vla_path,
                    "lens": "SAM 3" if self.s.replay_sam3 else "SmolVLA" if self.s.replay_vla_path else None,
                },
                "recovery_active": self.executor.active,
                "grasp_offset_mm": self._grasp_offset(),
                "latest_analysis_id": self.latest_analysis_id,
                "analysis_running": bool(self._analysis_lock and self._analysis_lock.locked()),
                "fps": round(self.fps, 1),
                "frame_size": list(self.size_wh),
                "thresholds": {
                    "joint_error_deg": self.s.joint_error_threshold_deg,
                    "chunk_variance": self.s.chunk_variance_threshold,
                    "attention_hijack": self.s.attention_hijack_threshold,
                    "attention_deviation": self.deviation_threshold,
                },
            }
