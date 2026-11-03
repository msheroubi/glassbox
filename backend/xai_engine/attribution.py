"""Attribution math that aligns policy saliency with SAM 3.1 object masks, plus anomaly triggers.

Attention-to-Mask Attribution Score for object i with mask M_i and saliency S(u, v):

    A(i) = sum_{(u,v) in M_i} S(u,v) / (sum_{(u,v)} S(u,v) + 1e-6)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from backend.robot.calibration import Homography
from backend.schemas import AnomalyEvent, MaskAttribution, SegmentedObjectView, ServoTelemetry

NON_DISTRACTOR_LABELS = {"target", "gripper"}


@dataclass
class SegmentedObject:
    mask_id: int
    label: str
    prompt: str
    mask: np.ndarray  # (H, W) bool at overhead resolution
    confidence: float | None = None
    source: str = "sam-3.1"  # or "sim-ground-truth"

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


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


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
    objects: list[SegmentedObject], saliency: np.ndarray | None, homography: Homography
) -> list[MaskAttribution]:
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
            )
        )
    return out


def joint_error(qpos_target, qpos_actual) -> float:
    """Servo following error ||q_target - q_actual||_2 over the five body joints (degrees)."""
    d = np.asarray(qpos_target[:5], dtype=np.float64) - np.asarray(qpos_actual[:5], dtype=np.float64)
    return float(np.linalg.norm(d))


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
    ) -> None:
        self.hz = hz
        self.je_thr = joint_error_threshold
        self.je_hold = max(1, int(joint_error_hold_s * hz))
        self.var_thr = variance_threshold
        self.var_hold = max(1, int(0.4 * hz))
        self.hijack_thr = hijack_threshold
        self.hijack_hold = max(1, int(0.5 * hz))
        self.stall_window = max(2, int(stall_window_s * hz))
        self.reset()

    def reset(self) -> None:
        self.je_count = self.var_count = self.hijack_count = 0
        self.qpos_hist: deque[np.ndarray] = deque(maxlen=self.stall_window)
        self.armed = True

    def update(
        self, tel: ServoTelemetry, attributions: list[MaskAttribution] | None = None
    ) -> AnomalyEvent | None:
        self.qpos_hist.append(np.asarray(tel.qpos_actual[:3]))
        self.je_count = self.je_count + 1 if tel.joint_error > self.je_thr else 0
        self.var_count = self.var_count + 1 if tel.action_chunk_variance > self.var_thr else 0
        max_share, top = 0.0, None
        if attributions:
            top = max((a for a in attributions if a.label not in NON_DISTRACTOR_LABELS), key=lambda a: a.attention_share, default=None)
            max_share = top.attention_share if top else 0.0
        self.hijack_count = self.hijack_count + 1 if max_share > self.hijack_thr else 0

        if not self.armed:
            return None
        reasons = []
        if self.je_count >= self.je_hold:
            reasons.append(f"servo following error {tel.joint_error:.1f} deg > {self.je_thr} deg (collision/stall)")
        if self.var_count >= self.var_hold:
            reasons.append(f"action-chunk variance {tel.action_chunk_variance:.1f} > {self.var_thr} (OOD confusion)")
        if self.hijack_count >= self.hijack_hold:
            reasons.append(f"attention share on non-target '{top.label}' #{top.mask_id} A={max_share:.2f} > {self.hijack_thr}")
        if len(self.qpos_hist) == self.stall_window:
            spread = np.ptp(np.stack(self.qpos_hist), axis=0).max()
            if spread < 4.0:
                reasons.append(f"no task progress for {self.stall_window / self.hz:.0f}s (arm joints moved {spread:.1f} deg)")
        if not reasons:
            return None
        self.armed = False
        return AnomalyEvent(
            timestamp=tel.timestamp,
            reasons=reasons,
            joint_error=round(tel.joint_error, 2),
            chunk_variance=round(tel.action_chunk_variance, 2),
            max_attention_share=round(max_share, 3),
        )
