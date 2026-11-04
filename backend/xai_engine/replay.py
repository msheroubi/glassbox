"""Record a failed run and replay it offline through heavier attention models.

The live loop can only afford ACT's own attention. When a run fails, the last seconds of the ring
buffer become a ``Run``: the frames the policy actually saw (after any staging or suppression), the
raw camera frames, joint states, ACT's live saliency and telemetry. It is saved under ``runs/<id>/`` so
it can be replayed again later, e.g. as a fallback during a demo.

``ReplayAnalyzer`` then takes its time over the run, with the arm held:

1. SAM 3.1 segments a few keyframes of the policy's view.
2. ACT's recorded attention becomes a map track (5 Hz).
3. An observe-only lens runs on the recorded frames (it never drives the arm), either
   - SAM 3 (``--replay-sam3``, ``facebook/sam3`` locally): where a perception model finds the target's and
     the goal's phrases. Not attention: a map that puts "red cube" on the red book shows visual ambiguity; or
   - SmolVLA (``--replay-vla``), with the real joint states: one pass gives two maps, its action expert's
     attention and the task text's grounding.
   The failure frame goes first, then a coarse sweep, then the gaps are filled, so the replay sharpens
   progressively and the breakdown doesn't wait for the fine pass.
4. A(label)(t), the attention share of each SAM label over time, and the first attention shift
   (a distractor holding more attention than the target; for SAM 3, more of the phrase's map than the
   object it names) are computed for every map.
5. Counterfactual test on the failure frame (concurrently with 3): Muse Image removes the suspect inside its
   SAM region, and the policy that ran (and SmolVLA, if it is the lens) are asked where they would look
   (and reach) without it.
6. Muse Wonder writes the forensic breakdown, with the counterfactual as evidence, and decides whether
   GlassBox can correct the failure by itself or a human is needed.

Progress is published as websocket events; the browser plays the run back at full frame rate and
interpolates the heatmaps between keyframes.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import cv2
import numpy as np

from backend.config import Settings
from backend.meta_clients.sam_client import SAMClient
from backend.meta_clients.wonder_breakdown import ReplayEvidence, WonderBreakdown
from backend.schemas import (
    Counterfactual, ReplayEvent, ReplayFrameInfo, ReplayKeyframe, ReplayMap, ReplayResult, ReplaySegmentation,
)
from backend.xai_engine import counterfactual as cfx
from backend.xai_engine.attribution import (
    NEAR_DRAW_LIFT, NON_DISTRACTOR_LABELS, Neighbourhoods, SegmentedObject, grid_b64 as _grid_b64, label_shares,
    near_shares, neighbourhoods,
)
from backend.xai_engine.ring_buffer import SALIENCY_DOWNSCALE, FrameRecord
from backend.xai_engine.visualize import draw_objects, encode_jpeg, heatmap_overlay

log = logging.getLogger(__name__)

MAP_INFO = {
    "vla_language": (
        "SmolVLA · task grounding",
        "Where a pretrained vision-language-action model (SmolVLA, observe only, not the policy that ran) grounds "
        "the task's words in the frames the policy saw: which pixels read as 'the red cube' and 'the green tile'.",
    ),
    "vla_action": (
        "SmolVLA · action attention",
        "Where SmolVLA's action expert looks while predicting the next actions from these frames and joint "
        "states: a second, larger policy's view of what matters for acting.",
    ),
    "act": (
        "ACT · live attention",
        "The attention of the ACT policy that actually drove the arm (decoder cross-attention over the camera "
        "features), recorded live.",
    ),
}
# SAM 3 lens maps, "sam3_<concept label>": where SAM 3 finds that concept's phrase. Titled per run's prompts.
SAM3_INFO = (
    "SAM 3 · '{phrase}'",
    "Where a promptable perception model (SAM 3, run locally; not the policy, and not anyone's attention) finds "
    "'{phrase}' in the frames the policy saw: per-pixel probability, dimmed by its confidence the phrase is "
    "present. Mass on another object means the scene is visually ambiguous for that phrase.",
)
ACT_MAP_HZ = 5.0
SHIFT_MIN_SHARE = 0.12  # a distractor needs at least this much attention (inside its mask) before a shift counts
SHIFT_MIN_LIFT = NEAR_DRAW_LIFT  # ... or, measured near it, this many times what a uniform map would put there


# ====================================================================================== recorded runs
@dataclass
class RunFrame:
    t: float
    raw_jpeg: bytes
    policy_jpeg: bytes | None
    qpos: list[float]
    act_saliency_u8: np.ndarray | None
    phase: str
    joint_error: float
    attention_deviation: float
    peak_load: float

    def raw_bgr(self) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.raw_jpeg, np.uint8), cv2.IMREAD_COLOR)

    def policy_bgr(self) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.policy_jpeg or self.raw_jpeg, np.uint8), cv2.IMREAD_COLOR)


@dataclass
class Run:
    run_id: str
    created_at: float
    fps: float
    trigger: str
    task: str
    anomaly_reasons: list[str]
    failure_index: int
    frames: list[RunFrame]
    staging: dict | None = None
    spark: dict | None = None  # Muse Spark's live triage of the same failure
    analysis_id: str | None = None
    path: Path | None = None

    @classmethod
    def from_records(
        cls, records: list[FrameRecord], *, fps: float, control_hz: float, window_s: float, t_start: float,
        tail_s: float = 0.5,
        t_failure: float, trigger: str, task: str, reasons: list[str], staging: dict | None,
        spark: dict | None, analysis_id: str | None,
    ) -> "Run":
        recs = [r for r in records if t_start <= r.timestamp <= t_failure + tail_s]
        recs = [r for r in recs if r.timestamp >= t_failure - window_s]
        if not recs:
            raise ValueError("No recorded frames in the run window")
        step = max(1, int(round(control_hz / fps)))
        picked = recs[::step]
        frames = [
            RunFrame(
                t=r.timestamp, raw_jpeg=r.jpeg, policy_jpeg=r.policy_jpeg, qpos=list(r.telemetry.qpos_actual),
                act_saliency_u8=r.saliency_u8, phase=r.phase, joint_error=r.telemetry.joint_error,
                attention_deviation=r.telemetry.attention_deviation,
                peak_load=max(abs(v) for v in r.telemetry.torque_load[:5]),
            )
            for r in picked
        ]
        failure = int(np.argmin([abs(f.t - t_failure) for f in frames]))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return cls(
            run_id=f"{stamp}-{uuid.uuid4().hex[:4]}", created_at=time.time(), fps=control_hz / step, trigger=trigger,
            task=task, anomaly_reasons=reasons, failure_index=failure, frames=frames, staging=staging, spark=spark,
            analysis_id=analysis_id,
        )

    # ------------------------------------------------------------------ persistence
    def save(self, root: Path) -> Path:
        d = root / self.run_id
        (d / "raw").mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": self.run_id, "created_at": self.created_at, "fps": self.fps, "trigger": self.trigger,
            "task": self.task, "anomaly_reasons": self.anomaly_reasons, "failure_index": self.failure_index,
            "staging": self.staging, "spark": self.spark,
            "frames": [
                {"t": f.t, "qpos": f.qpos, "phase": f.phase, "joint_error": f.joint_error,
                 "attention_deviation": f.attention_deviation, "peak_load": f.peak_load,
                 "policy_view": f.policy_jpeg is not None, "act_saliency": f.act_saliency_u8 is not None}
                for f in self.frames
            ],
        }
        sal = {}
        for i, f in enumerate(self.frames):
            (d / "raw" / f"{i:04d}.jpg").write_bytes(f.raw_jpeg)
            if f.policy_jpeg is not None:
                (d / "policy").mkdir(exist_ok=True)
                (d / "policy" / f"{i:04d}.jpg").write_bytes(f.policy_jpeg)
            if f.act_saliency_u8 is not None:
                sal[f"f{i:04d}"] = f.act_saliency_u8
        if sal:
            np.savez_compressed(d / "act_saliency.npz", **sal)
        (d / "meta.json").write_text(json.dumps(meta))
        self.path = d
        return d

    @classmethod
    def load(cls, d: Path) -> "Run":
        meta = json.loads((d / "meta.json").read_text())
        sal = dict(np.load(d / "act_saliency.npz")) if (d / "act_saliency.npz").exists() else {}
        frames = []
        for i, m in enumerate(meta["frames"]):
            pol = d / "policy" / f"{i:04d}.jpg"
            frames.append(RunFrame(
                t=m["t"], raw_jpeg=(d / "raw" / f"{i:04d}.jpg").read_bytes(),
                policy_jpeg=pol.read_bytes() if pol.exists() else None, qpos=m["qpos"],
                act_saliency_u8=sal.get(f"f{i:04d}"), phase=m["phase"], joint_error=m["joint_error"],
                attention_deviation=m["attention_deviation"], peak_load=m["peak_load"],
            ))
        return cls(
            run_id=meta["run_id"], created_at=meta["created_at"], fps=meta["fps"], trigger=meta["trigger"],
            task=meta["task"], anomaly_reasons=meta["anomaly_reasons"], failure_index=meta["failure_index"],
            frames=frames, staging=meta.get("staging"), spark=meta.get("spark"), analysis_id=None, path=d,
        )


def list_runs(root: Path) -> list[dict]:
    out = []
    for d in sorted(root.glob("*/meta.json"), reverse=True):
        try:
            meta = json.loads(d.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        frames = meta.get("frames", [])
        out.append({
            "run_id": meta["run_id"], "created_at": meta["created_at"], "trigger": meta["trigger"],
            "anomaly_reasons": meta["anomaly_reasons"], "n_frames": len(frames),
            "duration_s": round(frames[-1]["t"] - frames[0]["t"], 1) if frames else 0.0,
            "staged": meta.get("staging") is not None, "analysed": (d.parent / "replay.json").exists(),
        })
    return out


# ====================================================================================== lazy replay lens
class LazyModel:
    """Builds a model on a background thread so startup (and the live arm) never wait for it."""

    def __init__(self, name: str, factory: Callable[[], Any]) -> None:
        self.name = name
        self.model: Any = None
        self.error: str | None = None
        self._ready = threading.Event()
        threading.Thread(target=self._load, args=(factory,), name=f"load-{name}", daemon=True).start()

    def _load(self, factory) -> None:
        try:
            self.model = factory()
        except Exception as e:  # noqa: BLE001 - replay degrades to ACT-only maps
            log.exception("Could not load %s", self.name)
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self._ready.set()

    @property
    def loading(self) -> bool:
        return not self._ready.is_set()

    async def wait(self) -> Any:
        await asyncio.to_thread(self._ready.wait)
        return self.model


# ====================================================================================== analysis helpers
def _bisect_order(n: int) -> list[int]:
    """0..n-1 ordered so every prefix is spread as evenly as possible."""
    if n <= 0:
        return []
    order, seen = [], set()
    step = 1 << max(0, (n - 1).bit_length())
    while step >= 1:
        for i in range(0, n, step):
            if i not in seen:
                seen.add(i)
                order.append(i)
        step //= 2
    return order


def attention_shift(
    series: list[tuple[int, float, dict[str, float]]], lifts: list[dict[str, float]] | None = None,
    ref: str = "target",
) -> tuple[int, float, str, float, float] | None:
    """First keyframe from which a distractor out-draws ``ref`` (the target, or the concept a grounding map
    is prompted with) and still does at the next keyframe.

    With ``lifts`` (one per keyframe) the shares are the attention near each object, and a distractor counts
    once its neighbourhood holds SHIFT_MIN_LIFT times its uniform share (at keyframes where SAM found the target:
    "nothing near a target it didn't find" compares nothing); otherwise they are in-mask shares."""
    def top(sh):
        d = {k: v for k, v in sh.items() if k not in NON_DISTRACTOR_LABELS | {"hand"}}
        return max(d.items(), key=lambda kv: kv[1]) if d else (None, 0.0)

    for j, (frame, t, sh) in enumerate(series):
        lab, v = top(sh)
        tgt = sh.get(ref, 0.0)
        if lab is None or v <= tgt or (lifts and ref not in sh):
            continue
        if (lifts[j].get(lab, 0.0) < SHIFT_MIN_LIFT) if lifts else v < SHIFT_MIN_SHARE:
            continue
        if j + 1 < len(series):
            nxt = series[j + 1][2]
            if nxt.get(lab, 0.0) <= nxt.get(ref, 0.0):
                continue
        return frame, t, lab, v, tgt
    return None


def grounded_label(kind: str) -> str | None:
    """The concept a perception (grounding) map is meant to sit on, or ``None`` for attention maps."""
    return kind.removeprefix("sam3_") if kind.startswith("sam3_") else None


def near_series(m: ReplayMap, last_frame: int) -> tuple[list, list] | None:
    """``(series, lifts)`` of the attention near each object, or ``None`` for keyframes saved without it."""
    kfs = [k for k in m.keyframes if k.frame <= last_frame]
    if not kfs or any(not k.near for k in kfs):
        return None
    return [(k.frame, k.t, k.near) for k in kfs], [k.near_lift for k in kfs]


def _cf_ready(cf: Counterfactual | None) -> bool:
    return cf is not None and cf.status == "ready" and bool(cf.maps)


def counterfactual_facts(cf: Counterfactual) -> dict:
    """The counterfactual test as evidence for Muse Wonder."""
    def one(m):
        out = {"attention_moved": m.moved, "shares_with": m.shares_before, "shares_without": m.shares_after}
        if m.area:
            out.update({
                "near_with": m.near_before, "near_without": m.near_after, "neighbourhood_area": m.area,
                "moved_attention_left_from": m.left_from, "moved_attention_arrived_at": m.arrived_at,
                "peak_with": m.peak_before, "peak_without": m.peak_after,
            })
        return out

    return {
        "removed": cf.removed,
        "culprit_label": cf.culprit_label,
        "edited_frame_by": cf.source,
        "maps": {m.kind: one(m) for m in cf.maps},
        "predicted_reach_shift_px": cf.path_shift_px,
        "outcome_for_the_acting_policy": cf.outcome,
        "verdict": cf.verdict,
    }


# ====================================================================================== analyser
class ReplayAnalyzer:
    def __init__(
        self,
        settings: Settings,
        sam: SAMClient,
        wonder: WonderBreakdown,
        publish: Callable[[dict], None],
        busy: Callable[[], bool],
        vla: LazyModel | None,
        on_breakdown: Callable[[ReplayResult], Awaitable[None]],
        counterfactual: Callable[..., Awaitable[Counterfactual | None]] | None = None,
        live_attenders: Callable[[], list[tuple[str, str, Callable]]] | None = None,
    ) -> None:
        """``counterfactual`` is ``ControlLoop.counterfactual``; ``live_attenders`` the running policy's attend."""
        self.s = settings
        self.sam = sam
        self.wonder = wonder
        self.publish = publish
        self.busy = busy
        self.vla = vla
        self.on_breakdown = on_breakdown
        self.counterfactual = counterfactual
        self.live_attenders = live_attenders
        self._cf_tasks: dict[str, asyncio.Task] = {}
        self._nb_cache: dict[int, tuple[list[SegmentedObject], Neighbourhoods | None]] = {}
        self.runs: dict[str, Run] = {}
        self.results: dict[str, ReplayResult] = {}
        self.latest_id: str | None = None
        self._task: asyncio.Task | None = None
        self.root = settings.runs_dir

    # ------------------------------------------------------------------ public
    def show(self, rid: str) -> ReplayResult | None:
        """Make an analysed (or still analysing) replay the current one again."""
        r = self.results.get(rid)
        if r is not None and rid in self.runs:
            self.latest_id = rid
            self._event("replay_started", r)
        return r

    def start(self, run: Run, reuse_cached: bool = False) -> ReplayResult:
        """Begin analysing ``run`` in the background; a replay still running is stopped first."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self.runs[run.run_id] = run
        for old in list(self.runs)[:-6]:  # keep a handful of runs' frames in memory
            self.runs.pop(old, None)
        cached = self._load_cached(run) if reuse_cached else None
        result = cached or self._new_result(run)
        self.results[run.run_id] = result
        self.latest_id = run.run_id
        self._event("replay_started", result)
        if cached is None:
            self._task = asyncio.create_task(self._analyse(run, result))
            self._task.add_done_callback(self._log_error)
        return result

    def get_run(self, rid: str) -> Run | None:
        run = self.runs.get(rid)
        if run is None and (self.root / rid / "meta.json").exists():
            run = Run.load(self.root / rid)
            self.runs[rid] = run
        return run

    def vla_status(self) -> str:
        if self.vla is None:
            return "off"
        if self.vla.loading:
            return "loading"
        return "error" if self.vla.model is None else "ready"

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _log_error(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            log.error("Replay analysis failed: %r", task.exception())

    def _event(self, kind: str, result: ReplayResult) -> None:
        self.publish({"type": kind, "replay": json.loads(result.model_dump_json())})

    def _progress(self, r: ReplayResult, status: str, progress: float, text: str) -> None:
        r.status, r.progress, r.progress_text = status, round(progress, 3), text
        self.publish({"type": "replay_progress", "replay_id": r.replay_id, "status": status,
                      "progress": r.progress, "text": text})

    def _new_result(self, run: Run) -> ReplayResult:
        t0 = run.frames[run.failure_index].t
        return ReplayResult(
            replay_id=run.run_id, analysis_id=run.analysis_id, created_at=run.created_at, trigger=run.trigger,
            task=run.task, fps=run.fps, n_frames=len(run.frames), failure_frame=run.failure_index,
            frames=[
                ReplayFrameInfo(t=round(f.t - t0, 3), phase=f.phase, state="RUNNING" if i <= run.failure_index else "HOLD",
                                joint_error=f.joint_error, attention_deviation=f.attention_deviation)
                for i, f in enumerate(run.frames)
            ],
            has_policy_view=any(f.policy_jpeg is not None for f in run.frames),
            staging=run.staging,
            self_correction="off",
        )

    def _load_cached(self, run: Run) -> ReplayResult | None:
        if run.path is None or not (run.path / "replay.json").exists():
            return None
        try:
            r = ReplayResult.model_validate_json((run.path / "replay.json").read_text())
        except ValueError:
            return None
        r.analysis_id, r.self_correction = None, "unavailable"
        r.self_correction_note = "Loaded from disk: this run's scene is gone, so there is nothing to correct live."
        return r

    def _save(self, run: Run, r: ReplayResult) -> None:
        if run.path is not None:
            try:
                (run.path / "replay.json").write_text(r.model_dump_json())
            except OSError as e:
                log.warning("Could not save replay %s: %s", run.run_id, e)

    def _objects_for(self, frame: int, segs: dict[int, list[SegmentedObject]]) -> list[SegmentedObject]:
        if not segs:
            return []
        return segs[min(segs, key=lambda k: abs(k - frame))]

    def _neighbourhoods(self, objs: list[SegmentedObject]) -> Neighbourhoods | None:
        hit = self._nb_cache.get(id(objs))
        if hit is None or hit[0] is not objs:  # keyed by identity: one segmentation serves many keyframes
            hit = (objs, neighbourhoods(objs))
            if len(self._nb_cache) > 64:
                self._nb_cache.clear()
            self._nb_cache[id(objs)] = hit
        return hit[1]

    def _add_keyframe(self, m: ReplayMap, run: Run, frame: int, sal: np.ndarray, segs) -> None:
        t = run.frames[frame].t - run.frames[run.failure_index].t
        objs = self._objects_for(frame, segs)
        # A grounding map is a segmentation, not diffuse attention: what lies inside each mask is the measure.
        nearby = objs and grounded_label(m.kind) is None
        near, lift = near_shares(sal, objs, self._neighbourhoods(objs)) if nearby else ({}, {})
        kf = ReplayKeyframe(frame=frame, t=round(t, 3), grid=_grid_b64(sal, self.s.saliency_grid),
                            shares=label_shares(sal, objs), near=near, near_lift=lift)
        m.keyframes = sorted([k for k in m.keyframes if k.frame != frame] + [kf], key=lambda k: k.frame)

    def _phrase(self, label: str) -> str:
        return next((c.prompt for c in self.s.concepts if c.label == label), label)

    def _new_map(self, kind: str, expected: int) -> ReplayMap:
        if (label := grounded_label(kind)) is not None:
            title, desc = (x.format(phrase=self._phrase(label)) for x in SAM3_INFO)
        else:
            title, desc = MAP_INFO[kind]
        gw, gh = self.s.saliency_grid
        return ReplayMap(kind=kind, title=title, description=desc, grid_w=gw, grid_h=gh, expected_keyframes=expected)

    def _recompute_events(self, run: Run, r: ReplayResult) -> None:
        events: list[ReplayEvent] = []
        fail_t = 0.0
        if run.staging:
            events.append(ReplayEvent(
                t=round(run.frames[0].t - run.frames[run.failure_index].t, 2), frame=0, kind="staging",
                label="Demo staging active",
                detail=f"GlassBox showed the policy the {run.staging.get('label', 'distractor')} on the goal "
                       "(it is elsewhere on the table) to reproduce the failure on demand.",
            ))
        prev_phase = None
        for i, f in enumerate(run.frames[: run.failure_index + 1]):
            if f.phase and prev_phase and f.phase != prev_phase and f.phase not in ("ACT", "OBSERVE"):
                events.append(ReplayEvent(t=round(f.t - run.frames[run.failure_index].t, 2), frame=i, kind="phase",
                                          label=f"Phase: {f.phase}", detail=f"{prev_phase} -> {f.phase}"))
            prev_phase = f.phase
        for m in r.maps:
            if (ref := grounded_label(m.kind)) is not None:
                # Only keyframes where SAM 3.1 found the named object: "more than an object it didn't find" is no test.
                hit = attention_shift([(k.frame, k.t, k.shares) for k in m.keyframes
                                       if k.frame <= run.failure_index and ref in k.shares], ref=ref)
                if hit is not None:
                    frame, t, lab, v, tgt = hit
                    phrase = self._phrase(ref)
                    events.append(ReplayEvent(
                        t=round(t, 2), frame=frame, kind="attention_shift", map=m.kind,
                        label=f"SAM 3 reads the {self._phrase(lab)} as '{phrase}'",
                        detail=f"{m.title}: {v:.0%} of the '{phrase}' map lies on the {lab} vs {tgt:.0%} on the "
                               f"{ref} (perception, not the policy's attention)",
                    ))
                continue
            near = near_series(m, run.failure_index)
            if near is not None:
                hit = attention_shift(*near)
                if hit is not None:
                    frame, t, lab, v, tgt = hit
                    lift = next(k.near_lift.get(lab, 0.0) for k in m.keyframes if k.frame == frame)
                    events.append(ReplayEvent(
                        t=round(t, 2), frame=frame, kind="attention_shift", map=m.kind,
                        label=f"Attention shifts to the {lab}",
                        detail=f"{m.title}: {v:.0%} of the attention is near the {lab} ({lift:.1f}x a uniform map) "
                               f"vs {tgt:.0%} near the target",
                    ))
                continue
            hit = attention_shift([(k.frame, k.t, k.shares) for k in m.keyframes if k.frame <= run.failure_index])
            if hit is not None:
                frame, t, lab, v, tgt = hit
                events.append(ReplayEvent(
                    t=round(t, 2), frame=frame, kind="attention_shift", map=m.kind,
                    label=f"Attention shifts to the {lab}",
                    detail=f"{m.title}: A({lab}) = {v:.0%} vs A(target) = {tgt:.0%}",
                ))
        events.append(ReplayEvent(
            t=fail_t, frame=run.failure_index, kind="failure",
            label="Failure detected" if run.trigger == "auto" or run.anomaly_reasons else "Analysis requested",
            detail="; ".join(run.anomaly_reasons) or "operator pressed analyse",
        ))
        r.events = sorted(events, key=lambda e: e.t)

    def _filmstrip(self, run: Run, m: ReplayMap | None, segs) -> np.ndarray | None:
        if m is None or not m.keyframes:
            return None
        kfs = [k for k in m.keyframes if k.frame <= run.failure_index] or m.keyframes
        idx = sorted({round(i * (len(kfs) - 1) / 3) for i in range(4)}) if len(kfs) > 1 else [0]
        tiles = []
        for i in idx:
            k = kfs[i]
            frame = run.frames[k.frame].policy_bgr()
            h, w = frame.shape[:2]
            grid = np.frombuffer(base64.b64decode(k.grid), np.uint8).reshape(m.grid_h, m.grid_w)
            sal = cv2.resize(grid.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_CUBIC)
            img = draw_objects(heatmap_overlay(frame, np.clip(sal, 0, 1), 0.5), self._objects_for(k.frame, segs))
            label = f"t = {k.t:+.1f} s" + ("  (failure)" if k.frame == run.failure_index else "")
            cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
            cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(cv2.resize(img, (640, int(640 * h / w))))
        while len(tiles) < 4:
            tiles.append(np.zeros_like(tiles[0]))
        return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])

    def _evidence(self, run: Run, r: ReplayResult, primary: str, segs) -> ReplayEvidence:
        t0 = run.frames[run.failure_index].t
        step = max(1, int(round(run.fps / 2)))
        fail_objs = self._objects_for(run.failure_index, segs)
        return ReplayEvidence(
            task=run.task, trigger=run.trigger, anomaly_reasons=run.anomaly_reasons, spark=run.spark,
            maps={
                m.kind: {"title": m.title, "description": m.description,
                         **({"grounds": grounded_label(m.kind), "prompt": self._phrase(grounded_label(m.kind))}
                            if grounded_label(m.kind) else {}),
                         "series": [(k.t, k.shares) for k in m.keyframes if k.frame <= run.failure_index],
                         "near_series": [(k.t, k.near, k.near_lift) for k in m.keyframes
                                         if k.frame <= run.failure_index and k.near]}
                for m in r.maps
            },
            primary_map=primary,
            events=[e for e in r.events if e.kind != "staging"],  # the analyst judges what the policy saw
            telemetry=[
                [round(f.t - t0, 2), f.phase, round(f.joint_error, 1), round(f.attention_deviation, 3), round(f.peak_load, 1)]
                for f in run.frames[: run.failure_index + 1: step]
            ],
            labels_at_failure=sorted({o.label for o in fail_objs}),
            duration_s=run.frames[run.failure_index].t - run.frames[0].t,
            extra={"counterfactual_test": counterfactual_facts(r.counterfactual)} if _cf_ready(r.counterfactual) else {},
        )

    async def _segment(self, run: Run, r: ReplayResult) -> dict[int, list[SegmentedObject]]:
        n = len(run.frames)
        k = max(1, self.s.replay_sam_keyframes)
        before = run.failure_index
        picks = sorted({run.failure_index} | {round(i * before / max(1, k - 1)) for i in range(k - 1)})
        sem = asyncio.Semaphore(2)
        done = 0

        async def one(i: int):
            nonlocal done
            async with sem:
                objs, source = await self.sam.segment(run.frames[i].policy_bgr())
            done += 1
            self._progress(r, "segmenting", 0.05 + 0.15 * done / len(picks), f"SAM 3.1: segmented {done}/{len(picks)} keyframes")
            return i, objs, source

        t0 = time.perf_counter()
        results = await asyncio.gather(*(one(i) for i in picks if i < n))
        r.latency_ms["sam_ms"] = round((time.perf_counter() - t0) * 1000)
        segs = {i: objs for i, objs, _ in results if objs}
        t_fail = run.frames[run.failure_index].t
        r.segmentations = [
            ReplaySegmentation(frame=i, t=round(run.frames[i].t - t_fail, 3), objects=[o.view() for o in objs])
            for i, objs in sorted(segs.items())
        ]
        return segs

    async def _analyse(self, run: Run, r: ReplayResult) -> None:
        t_start = time.perf_counter()
        try:
            if run.path is None:
                self._progress(r, "saving", 0.02, "Saving the run")
                await asyncio.to_thread(run.save, self.root)
            self._progress(r, "segmenting", 0.05, "SAM 3.1: segmenting keyframes of what the policy saw")
            segs = await self._segment(run, r)

            # ACT's own attention, recorded live: cheap, so the replay has a map immediately.
            act = self._new_map("act", 0)
            step = max(1, int(round(run.fps / ACT_MAP_HZ)))
            size = None
            last_grid = None
            for i in range(0, len(run.frames), step):
                u8 = run.frames[i].act_saliency_u8
                if u8 is None:
                    continue
                if size is None:
                    h, w = run.frames[i].policy_bgr().shape[:2]
                    size = (w, h)
                sal = cv2.resize(u8, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                grid = _grid_b64(sal, self.s.saliency_grid)
                if grid == last_grid:
                    continue
                last_grid = grid
                self._add_keyframe(act, run, i, sal, segs)
            act.expected_keyframes = len(act.keyframes)
            if act.keyframes:
                r.maps.append(act)
            self._recompute_events(run, r)
            self._event("replay_update", r)
            if self.counterfactual is not None:
                self._cf_tasks[run.run_id] = asyncio.create_task(self._counterfactual(run, r, segs))

            model = None
            if self.vla is not None:
                if self.vla.loading:
                    self._progress(r, "attention", 0.22, f"Waiting for {self.vla.name} to finish loading")
                model = await self.vla.wait()
            breakdown_task: asyncio.Task | None = None
            if model is not None:
                breakdown_task = await self._vla_pass(run, r, model, segs)
            if breakdown_task is None:
                await self._run_breakdown(run, r, segs)
            else:
                await breakdown_task
            cf_task = self._cf_tasks.get(run.run_id)
            if cf_task is not None:
                self._progress(r, "refining", max(r.progress, 0.95), "Counterfactual: finishing the attention comparison")
                await cf_task
            r.latency_ms["replay_total_ms"] = round((time.perf_counter() - t_start) * 1000)
            self._progress(r, "done", 1.0, "Replay analysis complete")
        except asyncio.CancelledError:
            if (t := self._cf_tasks.get(run.run_id)) is not None:
                t.cancel()
            r.progress_text = "Stopped early: a newer run is being analysed"
            r.status = "done" if r.maps else "failed"
            self._event("replay_update", r)
            self._save(run, r)
            raise
        except Exception as e:
            if (t := self._cf_tasks.get(run.run_id)) is not None:
                t.cancel()
            log.exception("Replay %s failed", run.run_id)
            self._progress(r, "failed", r.progress, f"Replay analysis failed: {e}")
            self._event("replay_update", r)
            raise
        finally:
            self._cf_tasks.pop(run.run_id, None)
        self._event("replay_update", r)
        self._save(run, r)

    # ------------------------------------------------------------------ counterfactual
    def _suspect(self, run: Run, r: ReplayResult, objs: list[SegmentedObject]) -> SegmentedObject | None:
        """The object the counterfactual removes: Spark's culprit, else the distractor with the most attention
        at the failure (in any map), else the largest distractor."""
        cands = [o for o in objs if o.label not in NON_DISTRACTOR_LABELS]
        if not cands:
            return None
        named = (run.spark or {}).get("culprit_label")
        shares: dict[str, float] = {}
        for m in r.maps:
            k = min(m.keyframes, key=lambda k: abs(k.frame - run.failure_index), default=None)
            for lab, v in ((k.near or k.shares) if k else {}).items():
                shares[lab] = max(shares.get(lab, 0.0), v)
        return max(cands, key=lambda o: (o.label == named, shares.get(o.label, 0.0), int(o.mask.sum())))

    def _vla_attenders(self, model) -> list[tuple[str, str, Callable]]:
        """The lens's attention maps from one forward pass per frame (it observes only: no chunk). Grounding
        maps (SAM 3) are left out: with the suspect painted out, of course SAM 3 no longer finds it."""
        cache: dict[int, dict[str, np.ndarray]] = {}
        kinds = [k for k, label in model.lens_maps.items() if label is None]

        def make(kind: str):
            def attend(state, frame_bgr):
                key = id(frame_bgr)
                if key not in cache:
                    while self.busy():  # don't starve the arm (see _vla_pass)
                        time.sleep(1.0)
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    cache[key] = model.lens_frame(np.asarray(state, np.float32), rgb)
                return cache[key].get(kind), model.lens_name.lower().replace(" ", ""), None
            return attend

        return [(k, MAP_INFO[k][0], make(k)) for k in kinds]

    async def _counterfactual(self, run: Run, r: ReplayResult, segs) -> Counterfactual | None:
        fail = run.failure_index
        objs = self._objects_for(fail, segs)
        culprit = self._suspect(run, r, objs)
        choice = cfx.choose_edit((run.spark or {}).get("failure_type"), objs, culprit)
        if choice is None:
            return None
        failure_type, culprit = choice
        frame = run.frames[fail].policy_bgr()

        async def attenders():
            # The policy that ran first (its verdict leads and it has a predicted reach), then the lens's maps.
            out = [a for a in (self.live_attenders() if self.live_attenders else []) if a[0] == "act"]
            model = await self.vla.wait() if self.vla is not None else None
            return out + (self._vla_attenders(model) if model is not None else [])

        def store(view: str, jpeg: bytes) -> None:
            if run.path is not None:
                name = "counterfactual.jpg" if view == "image" else f"counterfactual_{view}.jpg"
                (run.path / name).write_bytes(jpeg)

        def publish(cf: Counterfactual) -> None:
            r.counterfactual = cf
            self._event("replay_update", r)

        try:
            return await self.counterfactual(
                failure_type, frame, np.asarray(run.frames[fail].qpos, np.float64), objs, culprit,
                f"/api/replay/{run.run_id}/counterfactual.jpg", publish, store, attenders, fail,
            )
        except Exception:  # noqa: BLE001 - the replay goes on without it
            log.exception("Replay counterfactual failed")
            return None

    async def _vla_pass(self, run: Run, r: ReplayResult, model, segs) -> asyncio.Task | None:
        """The lens (SmolVLA or SAM 3) over keyframes, failure first, then coarse, then fine. Starts the
        breakdown after coarse."""
        fail = run.failure_index
        sam3 = model.lens_name == "SAM 3"
        hz = self.s.replay_sam3_hz if sam3 else self.s.replay_vla_hz
        stride = max(1, int(round(run.fps / max(hz, 0.05))))
        grid_frames = list(range(fail, -1, -stride))[::-1]  # aligned on the failure frame
        order = [grid_frames[i] for i in reversed(_bisect_order(len(grid_frames)))]
        order = [fail] + [f for f in order if f != fail]
        coarse = min(len(order), max(1, self.s.replay_sam3_coarse_keyframes if sam3 else self.s.replay_coarse_keyframes))
        maps = {kind: self._new_map(kind, len(order)) for kind in model.lens_maps}
        r.maps = list(maps.values()) + r.maps
        what = "where it finds " + " / ".join(f"'{self._phrase(l)}'" for l in model.lens_maps.values() if l) \
            if sam3 else "attention"
        breakdown_task: asyncio.Task | None = None
        for n, frame in enumerate(order):
            while self.busy():
                self._progress(r, r.status, r.progress, f"Paused while the arm runs ({model.lens_name} and ACT share the CPU)")
                await asyncio.sleep(1.0)
            phase = "attention" if n < coarse else "refining"
            self._progress(r, phase, 0.22 + 0.68 * n / len(order),
                           f"{model.lens_name}: {what} in keyframe {n + 1}/{len(order)} "
                           f"(t = {run.frames[frame].t - run.frames[fail].t:+.1f} s)")
            f = run.frames[frame]
            rgb = cv2.cvtColor(f.policy_bgr(), cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            out = await asyncio.to_thread(model.lens_frame, np.asarray(f.qpos, np.float32), rgb)
            r.latency_ms["vla_pass_ms"] = round((time.perf_counter() - t0) * 1000)
            for kind, sal in out.items():
                if sal.shape != rgb.shape[:2]:  # the observer works at the live camera size
                    sal = cv2.resize(sal.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                self._add_keyframe(maps[kind], run, frame, sal, segs)
            self._recompute_events(run, r)
            self._event("replay_update", r)
            if n + 1 == coarse:
                breakdown_task = asyncio.create_task(self._run_breakdown(run, r, segs))
        return breakdown_task

    async def _run_breakdown(self, run: Run, r: ReplayResult, segs) -> None:
        kinds = {m.kind: m for m in r.maps if m.keyframes}
        primary = next((k for k in (self.s.replay_primary_map, "vla_language", "vla_action", "act", "sam3_target")
                        if k in kinds), None)
        refining = r.status == "refining"
        if not refining:
            self._progress(r, "breakdown", max(r.progress, 0.5), "Muse Wonder: writing the forensic breakdown")
        cf_task = self._cf_tasks.get(run.run_id)
        if cf_task is not None and not cf_task.done():
            self._progress(r, r.status, r.progress, "Waiting for the counterfactual test (Muse Image + attention)")
            await asyncio.wait({cf_task}, timeout=self.s.replay_counterfactual_wait_s)
        film = await asyncio.to_thread(self._filmstrip, run, kinds.get(primary), segs)
        ev = self._evidence(run, r, primary or "act", segs)
        t0 = time.perf_counter()
        r.breakdown, r.breakdown_source = await self.wonder.breakdown(film, ev)
        r.latency_ms["wonder_ms"] = round((time.perf_counter() - t0) * 1000)
        self._event("replay_update", r)
        await self.on_breakdown(r)
        self._event("replay_update", r)
