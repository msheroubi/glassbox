"""Attribution math that aligns policy saliency with SAM 3.1 object masks, plus anomaly triggers.

Attention-to-Mask Attribution Score for object i with mask M_i and saliency S(u, v):

    A(i) = sum_{(u,v) in M_i} S(u,v) / (sum_{(u,v)} S(u,v) + 1e-6)
"""

from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from backend.robot.calibration import Homography
from backend.xai_engine.attention_reference import grid_share
from backend.schemas import AnomalyEvent, MaskAttribution, SegmentedObjectView, ServoTelemetry

NON_DISTRACTOR_LABELS = {"target", "gripper", "goal"}


@dataclass
class SegmentedObject:
    mask_id: int
    label: str
    prompt: str
    mask: np.ndarray  # (H, W) bool at overhead resolution
    confidence: float | None = None
    source: str = "sam-3.1"

    @property
    def bbox(self) -> list[int]:
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return [0, 0, 0, 0]
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    def polygons(self, epsilon_px: float = 2.0) -> list[list[list[int]]]:
        contours, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in contours:
            if cv2.contourArea(c) < 20:
                continue
            approx = cv2.approxPolyDP(c, epsilon_px, True)
            polys.append(approx.reshape(-1, 2).tolist())
        return polys

    def view(self) -> SegmentedObjectView:
        return SegmentedObjectView(
            mask_id=self.mask_id,
            label=self.label,
            prompt=self.prompt if self.source == "sam-3.1" else f"{self.prompt} ({self.source})",
            confidence=self.confidence,
            bbox=self.bbox,
            polygons=self.polygons(),
        )


def attention_share(saliency: np.ndarray, mask: np.ndarray) -> float:
    return float(saliency[mask].sum() / (saliency.sum() + 1e-6))


def grid_b64(saliency: np.ndarray, grid_wh: tuple[int, int]) -> str:
    small = cv2.resize(saliency.astype(np.float32), grid_wh, interpolation=cv2.INTER_AREA)
    return base64.b64encode(np.clip(small * 255, 0, 255).astype(np.uint8).tobytes()).decode()


def label_shares(saliency: np.ndarray, objects: list[SegmentedObject]) -> dict[str, float]:
    total = float(saliency.sum()) + 1e-6
    unions: dict[str, np.ndarray] = {}
    for o in objects:
        unions[o.label] = o.mask if o.label not in unions else np.logical_or(unions[o.label], o.mask)
    return {lab: round(float(saliency[m].sum()) / total, 4) for lab, m in unions.items()}


# "Near" an object: within this fraction of the frame width of its SAM mask, and closer to it than to any other
# object. Attention maps are coarse (one cell per backbone feature or ViT patch) and diffuse, so the share
# strictly inside a small mask is ~1% even when the hot spot sits right on the object's edge.
NEAR_RADIUS_FRAC = 0.09  # ~58 px at 640 wide: a feature cell or two beyond SAM's tight mask edges
NEAR_WORK_W = 320  # neighbourhoods are computed at this width (shares don't depend on the resolution)
NEAR_DRAW_LIFT = 1.3  # an object "draws" attention when its neighbourhood holds this many times its uniform share


@dataclass
class Neighbourhoods:
    """Each label's neighbourhood (a nearest-object partition of the pixels within reach of some mask)."""

    regions: dict[str, np.ndarray]  # label -> bool at the working resolution
    size_wh: tuple[int, int]

    def fit(self, saliency: np.ndarray) -> np.ndarray:
        """``saliency`` at the working resolution, normalised to sum to 1."""
        s = np.clip(saliency.astype(np.float32), 0, None)
        if s.shape[:2] != (self.size_wh[1], self.size_wh[0]):
            s = cv2.resize(s, self.size_wh, interpolation=cv2.INTER_AREA)
        s = s.astype(np.float64)
        return s / (s.sum() + 1e-12)

    def area(self) -> dict[str, float]:
        """Fraction of the frame each neighbourhood covers: the share a uniform map would put there."""
        return {lab: round(float(r.mean()), 4) for lab, r in self.regions.items()}

    def sums(self, field: np.ndarray) -> dict[str, float]:
        return {lab: round(float(field[r].sum()), 4) for lab, r in self.regions.items()}

    def label_at(self, x: int, y: int) -> str | None:
        return next((lab for lab, r in self.regions.items() if r[y, x]), None)


def neighbourhoods(objects: list[SegmentedObject]) -> Neighbourhoods | None:
    masks = [o for o in objects if o.mask.any()]
    if not masks:
        return None
    h, w = masks[0].mask.shape[:2]
    k = min(1.0, NEAR_WORK_W / w)
    size = (max(1, round(w * k)), max(1, round(h * k)))
    unions: dict[str, np.ndarray] = {}
    for o in masks:
        m = cv2.resize(o.mask.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
        unions[o.label] = m if o.label not in unions else unions[o.label] | m
    labels = list(unions)
    dist = np.stack([cv2.distanceTransform((~unions[lab]).astype(np.uint8), cv2.DIST_L2, 5) for lab in labels])
    nearest, within = dist.argmin(0), dist.min(0) <= NEAR_RADIUS_FRAC * size[0]
    return Neighbourhoods({lab: (nearest == i) & within for i, lab in enumerate(labels)}, size)


def near_shares(
    saliency: np.ndarray, objects: list[SegmentedObject], nb: Neighbourhoods | None = None
) -> tuple[dict[str, float], dict[str, float]]:
    """``(share, lift)`` per label: the attention near each object, and that share over the neighbourhood's
    area (1 = what a uniform map gives it, 2 = twice as concentrated there as anywhere on average)."""
    nb = nb or neighbourhoods(objects)
    if nb is None:
        return {}, {}
    share, area = nb.sums(nb.fit(saliency)), nb.area()
    return share, {lab: round(share[lab] / area[lab], 2) if area[lab] > 0 else 0.0 for lab in share}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


SINGLE_INSTANCE_LABELS = {"target", "gripper", "goal"}  # roles with exactly one object in the scene
DUPLICATE_IOU = 0.6
PRESENCE_PRIOR = 0.15  # bonus for explaining a single-instance role SAM reported (the target is in the scene)

# Appearance vote (Settings.appearance_refs): SAM's "red cube" and "red book" scores are near coin flips on
# the real rig, but the two objects' colours are ~23 LAB units apart and stable across runs and positions.
APPEARANCE_SIGMA = 8.0  # LAB units
APPEARANCE_MAX = 30.0  # farther than this from every reference: some other object or other lighting, no vote
APPEARANCE_WEIGHT = 1.0  # a decisive colour outweighs SAM's confidence gap (which rarely exceeds 0.4)
APPEARANCE_L_WEIGHT = 0.5  # lightness depends on where the object sits under the lamp; hue less so
ALT_CONFIDENCE = 0.3  # SAM score assumed for a reference label SAM didn't return for this object


def appearance(lab: np.ndarray, sat: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Median LAB of the object's saturated pixels: its colour, not its glare or the table at its edges.

    ``lab`` is the frame in OpenCV 8-bit LAB, ``sat`` its HSV saturation."""
    m = mask.astype(bool)
    if m.shape != sat.shape:
        return None
    px = m & (sat > 70)
    if px.sum() < 20:
        return None
    return np.median(lab[px].astype(np.float32), axis=0)


def appearance_votes(colour: np.ndarray | None, refs: dict[str, list[float]] | None) -> dict[str, float]:
    """``P(label | colour)`` over the reference labels; empty when the colour is close to none of them."""
    if colour is None or not refs:
        return {}
    w = np.array([APPEARANCE_L_WEIGHT, 1.0, 1.0], np.float32)
    d = {k: float(np.linalg.norm((colour - np.asarray(v, np.float32)) * w)) for k, v in refs.items()}
    if min(d.values()) > APPEARANCE_MAX:
        return {}
    like = {k: float(np.exp(-0.5 * (x / APPEARANCE_SIGMA) ** 2)) for k, x in d.items()}
    total = sum(like.values())
    return {k: v / total for k, v in like.items()} if total > 1e-12 else {}


def dedupe_objects(
    objects: list[SegmentedObject], frame: np.ndarray | None = None,
    appearance_refs: dict[str, list[float]] | None = None, prompts: dict[str, str] | None = None,
) -> list[SegmentedObject]:
    """Merge masks of one physical object that SAM returned under several concepts, and label each one.

    SAM segments each concept independently, so the red book also comes back as "red cube" and the cube
    as "red book". Overlapping masks become one object, and the labels of contested objects are chosen
    jointly: each single-instance role (one target) is used at most once, and the assignment maximises total
    confidence plus a small bonus per single-instance role it covers. Per-object "highest confidence wins"
    mislabels the cube whenever SAM rates it a slightly better book than a cube.

    With ``frame`` and ``appearance_refs`` ({label: LAB colour}), an object SAM gave one of those labels
    also gets a colour vote for each of them, so the cube is the target even when SAM only called it a book.
    ``prompts`` ({label: noun phrase}) names an object relabelled that way."""
    clusters: list[list[SegmentedObject]] = []
    for o in objects:
        for c in clusters:
            if any(iou(o.mask, p.mask) > DUPLICATE_IOU for p in c):
                c.append(o)
                break
        else:
            clusters.append([o])

    def conf(o: SegmentedObject) -> float:
        return 0.5 if o.confidence is None else float(o.confidence)

    lab = sat = None
    if frame is not None and appearance_refs:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1]
    options: list[dict[str, float]] = []
    for c in clusters:
        opt = {lab_: max(conf(o) for o in c if o.label == lab_) for lab_ in {o.label for o in c}}
        if lab is not None and set(opt) & set(appearance_refs):
            union = np.logical_or.reduce([o.mask for o in c])
            for lab_, p in appearance_votes(appearance(lab, sat, union), appearance_refs).items():
                opt[lab_] = opt.get(lab_, ALT_CONFIDENCE) + APPEARANCE_WEIGHT * p
        options.append(opt)
    choice = [next(iter(opt)) if len(opt) == 1 else None for opt in options]
    contested = [i for i, ch in enumerate(choice) if ch is None]
    taken = {ch for ch in choice if ch in SINGLE_INSTANCE_LABELS}
    best: tuple[float, list[str]] | None = None

    def search(j: int, picked: list[str], used: set[str], score: float) -> None:
        nonlocal best
        if j == len(contested):
            if best is None or score > best[0]:
                best = (score, picked.copy())
            return
        for lab_, sc in options[contested[j]].items():
            if lab_ in SINGLE_INSTANCE_LABELS and lab_ in used:
                continue
            picked.append(lab_)
            single = lab_ in SINGLE_INSTANCE_LABELS
            search(j + 1, picked, used | ({lab_} if single else set()), score + sc + (PRESENCE_PRIOR if single else 0.0))
            picked.pop()

    if contested and len(contested) <= 8:
        search(0, [], taken, 0.0)
    for k, i in enumerate(contested):
        choice[i] = best[1][k] if best is not None else max(options[i], key=options[i].get)

    out = []
    for c, lab_ in zip(clusters, choice):
        same = [o for o in c if o.label == lab_]
        rep = max(same or c, key=conf)
        prompt = rep.prompt if same else (prompts or {}).get(lab_) or next(
            (o.prompt for o in objects if o.label == lab_), lab_)
        mask = np.logical_or.reduce([o.mask for o in c]) if len(c) > 1 else rep.mask
        out.append(SegmentedObject(len(out), lab_, prompt, mask, rep.confidence, rep.source))
    return out


def centroid(mask: np.ndarray) -> tuple[int, int]:
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return 0, 0
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def saliency_centroid(saliency: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float] | None:
    s = saliency if mask is None else saliency * mask
    total = s.sum()
    if total < 1e-6:
        return None
    ys, xs = np.indices(s.shape)
    return float((xs * s).sum() / total), float((ys * s).sum() / total)


def compute_attributions(
    objects: list[SegmentedObject], saliency: np.ndarray | None, homography: Homography, deviation=None
) -> list[MaskAttribution]:
    """``deviation`` (attention_reference.Deviation) adds each mask's expected share and share of unexpected attention."""
    gripper = next((o.mask for o in objects if o.label == "gripper"), None)
    out = []
    for obj in objects:
        u, v = centroid(obj.mask)
        x, y = homography.pixel_to_workspace(u, v)
        out.append(
            MaskAttribution(
                mask_id=obj.mask_id,
                label=obj.label,
                centroid_uv=[u, v],
                workspace_xy_mm=[round(x, 1), round(y, 1)],
                attention_share=round(attention_share(saliency, obj.mask), 4) if saliency is not None else 0.0,
                iou_with_gripper=round(iou(obj.mask, gripper), 4) if gripper is not None and obj.label != "gripper" else 0.0,
                expected_share=round(grid_share(deviation.expected, obj.mask), 4) if deviation is not None else None,
                excess_share=round(grid_share(deviation.excess, obj.mask), 4) if deviation is not None else None,
            )
        )
    return out


def joint_error(qpos_target, qpos_actual) -> float:
    """Servo following error ||q_target - q_actual||_2 over the five body joints (degrees)."""
    d = np.asarray(qpos_target[:5], dtype=np.float64) - np.asarray(qpos_actual[:5], dtype=np.float64)
    return float(np.linalg.norm(d))


STALL_SPEED_DEG_S = 15.0  # a servo is "stalled" only if the body joints are (nearly) not moving


class AnomalyDetector:
    """Debounced triggers for collision/stall, OOD confusion, attention hijacking and no-progress."""

    def __init__(
        self,
        hz: float,
        joint_error_threshold: float,
        joint_error_hold_s: float,
        variance_threshold: float,
        hijack_threshold: float,
        stall_window_s: float,
        disabled: frozenset[str] = frozenset(),
        deviation_threshold: float | None = None,
        deviation_hold_s: float = 1.0,
        hijack_excess_threshold: float = 0.15,
    ) -> None:
        self.disabled = disabled  # any of: servo, variance, hijack, stall, deviation
        self.hz = hz
        self.je_thr = joint_error_threshold
        self.je_hold = max(1, int(joint_error_hold_s * hz))
        self.var_thr = variance_threshold
        self.var_hold = max(1, int(0.4 * hz))
        self.hijack_thr = hijack_threshold
        self.hijack_hold = max(1, int(0.5 * hz))
        self.stall_window = max(2, int(stall_window_s * hz))
        # With an attention reference, "attention left what the demos did" replaces the fixed A(i) rule.
        self.dev_thr = deviation_threshold
        self.hijack_excess_thr = hijack_excess_threshold
        self.dev_hold = max(1, int(deviation_hold_s * hz))
        self.reset()

    def reset(self) -> None:
        self.je_count = self.var_count = self.hijack_count = self.dev_count = 0
        self.qpos_hist: deque[np.ndarray] = deque(maxlen=self.stall_window)
        self.armed = True

    def update(
        self, tel: ServoTelemetry, attributions: list[MaskAttribution] | None = None
    ) -> AnomalyEvent | None:
        self.qpos_hist.append(np.asarray(tel.qpos_actual[:6]))  # incl. wrist and gripper: grasping is progress
        speed = float(np.linalg.norm(np.asarray(tel.qvel[:5])))
        # Tracking lag while the arm moves is normal; a collision shows as large error with no motion.
        self.je_count = self.je_count + 1 if tel.joint_error > self.je_thr and speed < STALL_SPEED_DEG_S else 0
        self.var_count = self.var_count + 1 if tel.action_chunk_variance > self.var_thr else 0
        # A(i), or with an attention reference A(i) minus what the demos attended there at this pose
        # (a habitual hot spot is not a hijack, the same share on something new is).
        def surprise(a: MaskAttribution) -> float:
            return a.attention_share - a.expected_share if a.expected_share is not None else a.attention_share

        max_share, top, hijack_thr = 0.0, None, self.hijack_thr
        if attributions:
            top = max((a for a in attributions if a.label not in NON_DISTRACTOR_LABELS), key=surprise, default=None)
            max_share = surprise(top) if top else 0.0
            if top is not None and top.expected_share is not None:
                hijack_thr = self.hijack_excess_thr
        self.hijack_count = self.hijack_count + 1 if max_share > hijack_thr else 0

        self.dev_count = self.dev_count + 1 if self.dev_thr is not None and tel.attention_deviation > self.dev_thr else 0

        if not self.armed:
            return None
        reasons = []
        if self.dev_count >= self.dev_hold and "deviation" not in self.disabled:
            culprit = max((a for a in attributions or [] if a.excess_share), key=lambda a: a.excess_share, default=None)
            where = f"; {culprit.excess_share:.0%} of it is on '{culprit.label}' #{culprit.mask_id}" if culprit else ""
            reasons.append(
                f"attention deviates from the demos: {tel.attention_deviation:.0%} of it is somewhere the policy "
                f"never looked at this arm pose (> {self.dev_thr:.0%}){where}"
            )
        if self.je_count >= self.je_hold and "servo" not in self.disabled:
            reasons.append(f"servo following error {tel.joint_error:.1f} deg > {self.je_thr} deg while the arm is not moving (collision/stall)")
        if self.var_count >= self.var_hold and "variance" not in self.disabled:
            reasons.append(f"action-chunk variance {tel.action_chunk_variance:.1f} > {self.var_thr} (OOD confusion)")
        if self.hijack_count >= self.hijack_hold and "hijack" not in self.disabled:
            if top.expected_share is not None:
                reasons.append(
                    f"attention on '{top.label}' #{top.mask_id} is A={top.attention_share:.2f} but the demos had "
                    f"{top.expected_share:.2f} there at this pose (+{max_share:.2f} > {hijack_thr})"
                )
            else:
                reasons.append(f"attention share on non-target '{top.label}' #{top.mask_id} A={max_share:.2f} > {hijack_thr}")
        if len(self.qpos_hist) == self.stall_window and "stall" not in self.disabled:
            spread = np.ptp(np.stack(self.qpos_hist), axis=0).max()
            if spread < 4.0:
                reasons.append(f"no task progress for {self.stall_window / self.hz:.0f}s (no joint (incl. gripper) moved more than {spread:.1f} deg)")
        if not reasons:
            return None
        self.armed = False
        return AnomalyEvent(
            timestamp=tel.timestamp,
            reasons=reasons,
            joint_error=round(tel.joint_error, 2),
            chunk_variance=round(tel.action_chunk_variance, 2),
            max_attention_share=round(max_share, 3),
            attention_deviation=round(tel.attention_deviation, 3),
        )
