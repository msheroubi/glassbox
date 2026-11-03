"""The 30 Hz runtime: policy -> interventions -> robot, telemetry, anomaly triggers and the XAI pipeline.

Threading model: the control loop runs in a dedicated thread (hardware I/O and the policy are
blocking); the Meta API pipeline runs as coroutines on the FastAPI event loop. Every frame the
thread publishes a packet (telemetry JSON + overhead JPEG) to the websocket ``Hub``. Commands from
the API (apply fix, e-stop, reset, ...) take ``self._lock`` so they never interleave with a tick.

Interventions only touch what the *policy* sees or what the *servos* are sent:

* mask suppression rewrites the overhead image fed to the policy (the raw frame is still recorded),
* a waypoint bypass pre-empts the policy until the clamped trajectory completes,
* a grasp offset shifts the policy's grasp target.
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
from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.config import Settings
from backend.meta_clients.base import MetaClient
from backend.meta_clients.sam_client import SAMClient
from backend.meta_clients.spark_explainer import DiagnosisContext, SparkExplainer
from backend.meta_clients.voice_and_image import CounterfactualRenderer, counterfactual_prompt, image_region
from backend.meta_clients.wonder_planner import WonderPlanner
from backend.robot.calibration import Homography
from backend.robot.policy_hook import ACTRunner, CameraObserver, MockPolicy, PolicyStep
from backend.robot.so101_interface import (
    SCENARIOS, MockSO101, Observation, PlanarKinematics, RealSO101, WebcamSource,
)
from backend.schemas import AnalysisResult, AnomalyEvent, AskResponse, RecoveryPlan, ServoTelemetry
from backend.xai_engine.attribution import (
    NON_DISTRACTOR_LABELS,
    AnomalyDetector,
    SegmentedObject,
    centroid,
    compute_attributions,
    joint_error,
    saliency_centroid,
)
from backend.xai_engine.intervention import GraspOffset, MaskSuppressor, WaypointExecutor, clamp_joints
from backend.xai_engine.ring_buffer import RingBuffer
from backend.xai_engine.visualize import composite_keyframe, encode_jpeg

log = logging.getLogger(__name__)

RUNNING, HOLD, ESTOP, COMPLETE = "RUNNING", "HOLD", "ESTOP", "COMPLETE"
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
    counterfactual_jpeg: bytes | None = None
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

        self.policy: MockPolicy | ACTRunner | CameraObserver | None = None
        self.policy_kind = "none"
        self.suppressor = MaskSuppressor(self.plate, s.tabletop_bgr)
        self.executor = WaypointExecutor()
        self.grasp = GraspOffset()
        self.detector = AnomalyDetector(
            s.control_hz, s.joint_error_threshold_deg, s.joint_error_hold_s, s.chunk_variance_threshold,
            s.attention_hijack_threshold, s.stall_progress_s,
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
        self.sam = SAMClient(self.s, self.meta, self.robot.ground_truth_objects if self.mock else None)
        self.spark = SparkExplainer(self.s, self.meta)
        self.wonder = WonderPlanner(self.s, self.meta, self.kin)
        self.images = CounterfactualRenderer(self.s, self.meta)

        await asyncio.to_thread(self.robot.connect)
        if self.camera_only:
            self.policy = CameraObserver(self.s.overhead_camera_key)
            self.policy_kind = "camera-only"
        elif self.s.policy_path:
            self.policy = await asyncio.to_thread(
                ACTRunner, self.s.policy_path, self.s.policy_device, self.s.overhead_camera_key, self.size_wh,
                self.s.policy_n_action_steps, self.s.task_description,
            )
            self.policy_kind = "act"
        elif self.mock:
            self.policy = MockPolicy(self.kin, self.homography, n_action_steps=10)
            self.policy_kind = "mock"
        else:
            log.warning("No --policy-path given: the arm holds position; streaming and analysis still work")
        self.status_text = (
            "camera-only: observing the workspace (no arm connected)" if self.camera_only
            else "running" if self.policy is not None else "no policy loaded: holding position"
        )
        if not self.meta.enabled:
            log.warning("MODEL_API_KEY not set: all Meta models will use local fallbacks")
        if not self.mock and self.meta.enabled and self.s.live_sam_interval_s > 0:
            self._live_sam_task = asyncio.create_task(self._live_sam_loop())
        self._thread = threading.Thread(target=self._run, name="glassbox-control", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._stop.set()
        for task in (self._track_task, self._live_sam_task):
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
        if self.suppressor.active:
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
            policy_frame = self.suppressor.apply(raw) if self.suppressor.active else raw
            target = self.last_target if self.last_target is not None else obs.qpos.copy()
            variance = self.last_variance
            if self.state == RUNNING:
                if self.executor.active:
                    target, self.recovery_note = self.executor.next()
                    self.phase, variance = "RECOVERY", 0.0
                    if not self.executor.active:
                        self._recovery_finished()
                elif self.policy is not None:
                    step = self._policy_step(obs, policy_frame)
                    target, variance = np.asarray(step.action, np.float64), step.chunk_variance
                    if step.saliency is not None:
                        self.saliency, self.saliency_method = step.saliency, step.saliency_method
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
            )
            self.history.append(tel)

            if self.tick_count % LIVE_ATTRIBUTION_EVERY == 0:
                self._refresh_live_attribution()

            # camera-only: there is no arm to fail, so the triggers stay off and analysis is on demand
            if self.state == RUNNING and not self.executor.active and self.phase != "DONE" and not self.camera_only:
                event = self.detector.update(tel, self.live_attrs)
                if event is not None:
                    self._on_anomaly(event)
            if self.mock and self.state == RUNNING and (self.robot.delivered or self.phase == "DONE"):
                self.state, self.status_text = COMPLETE, "task complete: cube delivered to the bin"
                self.robot.hold()
                self.hub.publish_event({"type": "state", "state": self.state, "status": self.status_text})
            state, phase = self.state, self.phase
            suppression = {
                "active": self.suppressor.active,
                "objects": [{"mask_id": k, "label": v} for k, v in self.suppressor.labels.items()],
            }
            recovery = {"active": self.executor.active, "note": self.recovery_note if self.executor.active else ""}

        jpeg = encode_jpeg(raw, self.s.stream_jpeg_quality)
        rec = self.ring.push(obs.timestamp, jpeg, self.saliency, tel, phase)

        now = time.time()
        if now - self._last_publish < 0.9 / self.s.stream_fps or not self.hub.clients:
            return
        self._last_publish = now
        policy_jpeg = encode_jpeg(policy_frame, self.s.stream_jpeg_quality) if policy_frame is not raw else None
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
                for i, g in enumerate(self.robot.ground_truth_objects())
            ]
        else:
            objs = self._sam_live_objects
        self.live_objects = objs
        self.live_attrs = compute_attributions(objs, self.saliency, self.homography) if objs and self.saliency is not None else []
        shares = {a.mask_id: a.attention_share for a in self.live_attrs}
        self.live_views = [
            {**o.view().model_dump(), "source": o.source, "attention_share": shares.get(o.mask_id, 0.0)} for o in objs
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
        self._enter_hold("anomaly detected")
        self.hub.publish_event({"type": "anomaly", "event": event.model_dump()})
        if self.s.auto_analyze and self.loop is not None:
            fut = asyncio.run_coroutine_threadsafe(self.analyze("auto", event), self.loop)
            fut.add_done_callback(self._log_future_error)

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
            self.hub.publish_event({"type": "analysis_started", "trigger": trigger})
            rec = self.ring.latest()
            if rec is None:
                raise ConflictError("No frames recorded yet")
            frame = rec.frame_bgr()
            saliency = rec.saliency(self.size_wh)
            lat: dict[str, float] = {}

            t0 = time.perf_counter()
            clip = None
            if self.s.use_sam_video:
                clip = await asyncio.to_thread(RingBuffer.export_mp4, self.ring.snapshot(self.s.sam_clip_frames), self.s.control_hz)
            objects, sam_source = await self.sam.segment(frame, clip)
            lat["sam_ms"] = round((time.perf_counter() - t0) * 1000)

            attributions = compute_attributions(objects, saliency, self.homography)
            ctx = self._diagnosis_context(anomaly, attributions, objects, saliency, rec)
            caption = "GlassBox keyframe | " + ("; ".join(anomaly.reasons) if anomaly else "manual analysis")
            composite = composite_keyframe(frame, saliency, objects, attributions, caption[:140])
            t1 = time.perf_counter()
            diagnosis, diag_source = await self.spark.diagnose(composite, ctx)
            lat["spark_ms"] = round((time.perf_counter() - t1) * 1000)

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
                result, objects, frame, saliency, encode_jpeg(composite, 88), ctx
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
            return result

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
        culprit = next((o for o in rec.objects if o.mask_id == diag.culprit_mask_id), None)
        target = next((o for o in rec.objects if o.label == "target"), None)
        prompt = counterfactual_prompt(
            diag,
            culprit.prompt if culprit else None,
            image_region(culprit.bbox, self.size_wh) if culprit else None,
            target.prompt if target else "red cube",
        )
        if prompt is None:
            return
        mask, path = (culprit.mask if culprit else None), self._plan_path_px(rec.result.plan)
        if diag.failure_type == "SPECULAR_GLARE":
            # Don't erase the target itself; show the grasp correction as an arrow instead.
            mask = None
            if culprit is not None:
                u, v = centroid(culprit.mask)
                x, y = self.homography.pixel_to_workspace(u, v)
                dx = float(diag.intervention_parameters.get("dx_mm", 0))
                dy = float(diag.intervention_parameters.get("dy_mm", 0))
                path = [self.homography.workspace_to_pixel(x - dx, y - dy), (u, v)]
        t0 = time.perf_counter()
        img, source = await self.images.render(rec.frame, prompt, mask, self.plate, path)
        rec.counterfactual_jpeg = encode_jpeg(img, 88)
        rec.result.counterfactual_url = f"/api/analysis/{aid}/counterfactual.jpg?v={int(time.time())}"
        rec.result.counterfactual_source = source
        rec.result.latency_ms["image_ms"] = round((time.perf_counter() - t0) * 1000)
        self._publish_analysis(aid, "analysis_update")

    # ================================================================== operator commands
    async def apply_fix(self, analysis_id: str, override: str | None = None) -> dict:
        rec = self.analyses.get(analysis_id)
        if rec is None:
            raise KeyError(analysis_id)
        diag, plan = rec.result.diagnosis, rec.result.plan
        itype = override or diag.intervention_type
        params = diag.intervention_parameters
        tracked: SegmentedObject | None = None
        with self._lock:
            if self.state == ESTOP:
                raise ConflictError("E-stop is active: resume torque first")
            if itype == "INPAINT_MASK_SUPPRESSION":
                ids = [int(params["mask_id"])] if "mask_id" in params else []
                ids += (plan.suppress_mask_ids if plan else []) or ([diag.culprit_mask_id] if diag.culprit_mask_id is not None else [])
                objs = [o for o in rec.objects if o.mask_id in ids]
                if not objs:  # fall back to the most-attended non-target object
                    shares = {a.mask_id: a.attention_share for a in rec.result.attributions}
                    cands = [o for o in rec.objects if o.label not in NON_DISTRACTOR_LABELS]
                    objs = [max(cands, key=lambda o: shares.get(o.mask_id, 0))] if cands else []
                if not objs:
                    raise ConflictError("No mask to suppress")
                for obj in objs:
                    self.suppressor.set_mask(obj.mask_id, obj.mask, obj.label)
                    self._tracked[obj.mask_id] = obj
                tracked = objs[0]
                message = "Suppressing " + ", ".join(f"#{o.mask_id} {o.label}" for o in objs) + " in the policy's view"
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
        """Hardware mode: periodic SAM pass so attention shares (and the hijack trigger) are live."""
        while True:
            await asyncio.sleep(self.s.live_sam_interval_s)
            rec = self.ring.latest()
            if rec is None or self.state != RUNNING:
                continue
            objs, source = await self.sam.segment(rec.frame_bgr())
            if source != "local-fallback":
                with self._lock:
                    self._sam_live_objects = objs

    def clear_interventions(self) -> None:
        with self._lock:
            self.suppressor.clear()
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
            self._tracked.clear()
            self.executor.cancel()
            self.grasp.clear()
            self.detector.reset()
            self.ring.clear()
            self.latest_analysis_id = None  # past analyses stay readable by id; Q&A starts fresh
            self.history.clear()
            self.saliency, self.last_target, self.last_anomaly = None, None, None
            self.live_objects, self.live_attrs, self.live_views = [], [], []
            if self.state == ESTOP:
                self.robot.resume_torque()
            self.state = RUNNING
            self.status_text = f"reset{f' to scenario {scenario}' if scenario else ''}: policy running"
        self.hub.publish_event({"type": "reset", "scenario": getattr(self.robot, "scenario", None)})
        self.hub.publish_event({"type": "state", "state": self.state, "status": self.status_text})

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
                },
            }
