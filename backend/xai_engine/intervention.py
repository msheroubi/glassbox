"""Live interventions applied by the control loop without retraining the policy.

* ``MaskSuppressor`` - Real-Time Visual Mask Suppression. Pixels of a SAM 3.1-tracked distractor are
  replaced in the *policy's* input frame with the calibrated empty-table background (background
  plate if captured, otherwise Telea inpainting, otherwise the tabletop colour), with a feathered
  edge so no new high-frequency edge appears for the CNN backbone to latch onto.
* ``WaypointExecutor`` - plays back a joint-space bypass trajectory (from Muse Wonder or the local
  planner) with limit clamping and a velocity cap, then hands control back to the policy.
* ``GraspOffset`` - shifts the grasp target by (dx, dy) mm.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.config import JOINT_LIMITS, JOINT_NAMES
from backend.schemas import JointWaypoint

log = logging.getLogger(__name__)


class MaskSuppressor:
    def __init__(self, background_plate: np.ndarray | None, tabletop_bgr: tuple[int, int, int], dilate_px: int = 15) -> None:
        self.plate = background_plate
        self.tabletop_bgr = tabletop_bgr
        self.dilate_px = dilate_px
        self._lock = threading.Lock()
        self._masks: dict[int, np.ndarray] = {}  # mask_id -> bool mask
        self._alpha: np.ndarray | None = None  # feathered union, float32 (h, w, 1) cropped to _roi
        self._roi: tuple[slice, slice] | None = None
        self.labels: dict[int, str] = {}

    @property
    def active(self) -> bool:
        return bool(self._masks)

    def set_mask(self, mask_id: int, mask: np.ndarray, label: str = "") -> None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.dilate_px + 1, 2 * self.dilate_px + 1))
        dilated = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
        with self._lock:
            self._masks[mask_id] = dilated
            self.labels[mask_id] = label
            self._rebuild()

    def clear(self) -> None:
        with self._lock:
            self._masks.clear()
            self.labels.clear()
            self._alpha = None
            self._roi = None

    def _rebuild(self) -> None:
        union = np.zeros_like(next(iter(self._masks.values())), dtype=np.float32)
        for m in self._masks.values():
            union = np.maximum(union, m.astype(np.float32))
        alpha = cv2.GaussianBlur(union, (0, 0), self.dilate_px / 3)
        ys, xs = np.nonzero(alpha > 0.01)
        if len(xs) == 0:
            self._alpha, self._roi = None, None
            return
        # Blend only inside the bounding box of the feathered mask to stay within the 33 ms tick.
        self._roi = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        self._alpha = alpha[self._roi][..., None]

    def union_mask(self) -> np.ndarray | None:
        with self._lock:
            if self._alpha is None:
                return None
            full = np.zeros(next(iter(self._masks.values())).shape, bool)
            full[self._roi] = self._alpha[..., 0] > 0.5
            return full

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        with self._lock:
            alpha, roi = self._alpha, self._roi
        if alpha is None:
            return frame_bgr
        full_region = np.zeros(frame_bgr.shape[:2], bool)
        full_region[roi] = alpha[..., 0] > 0.02
        fill = self._fill(frame_bgr, full_region)
        out = frame_bgr.copy()
        out[roi] = (frame_bgr[roi].astype(np.float32) * (1 - alpha) + fill[roi].astype(np.float32) * alpha).astype(np.uint8)
        return out

    def _fill(self, frame: np.ndarray, region: np.ndarray) -> np.ndarray:
        if self.plate is not None and self.plate.shape == frame.shape:
            return self.plate
        try:
            # Inpaint at quarter resolution to stay inside the 33 ms control budget.
            small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(region.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            painted = cv2.inpaint(small, small_mask, 5, cv2.INPAINT_TELEA)
            return cv2.resize(painted, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        except cv2.error:
            return np.full_like(frame, self.tabletop_bgr)


def clamp_joints(q: np.ndarray) -> np.ndarray:
    lo = np.array([JOINT_LIMITS[n][0] for n in JOINT_NAMES])
    hi = np.array([JOINT_LIMITS[n][1] for n in JOINT_NAMES])
    return np.clip(q, lo, hi)


@dataclass
class WaypointExecutor:
    max_speed_deg_s: float = 60.0
    _targets: list[np.ndarray] = field(default_factory=list)
    _notes: list[str] = field(default_factory=list)
    _i: int = 0

    @property
    def active(self) -> bool:
        return self._i < len(self._targets)

    def load(self, start_q: np.ndarray, waypoints: list[JointWaypoint], hz: float) -> int:
        """Densify waypoints into per-tick targets. Returns the number of control ticks."""
        targets, notes = [], []
        prev = np.asarray(start_q, dtype=np.float64)
        for wp in waypoints:
            goal = clamp_joints(np.asarray(wp.qpos, dtype=np.float64))
            # Respect the requested duration but never exceed the velocity cap.
            min_t = np.abs(goal[:5] - prev[:5]).max() / self.max_speed_deg_s
            n = max(1, int(max(wp.duration_s, min_t) * hz))
            for k in range(1, n + 1):
                a = k / n
                a = a * a * (3 - 2 * a)  # smoothstep: zero velocity at each waypoint
                targets.append(prev + (goal - prev) * a)
                notes.append(wp.note)
            prev = goal
        self._targets, self._notes, self._i = targets, notes, 0
        log.info("Loaded %d waypoints -> %d control ticks", len(waypoints), len(targets))
        return len(targets)

    def next(self) -> tuple[np.ndarray, str]:
        q, note = self._targets[self._i], self._notes[self._i]
        self._i += 1
        return q, note

    def cancel(self) -> None:
        self._targets, self._notes, self._i = [], [], 0


@dataclass
class GraspOffset:
    dx_mm: float = 0.0
    dy_mm: float = 0.0
    applied_at: float = 0.0

    def set(self, dx_mm: float, dy_mm: float) -> None:
        self.dx_mm, self.dy_mm, self.applied_at = float(dx_mm), float(dy_mm), time.time()

    def clear(self) -> None:
        self.dx_mm = self.dy_mm = 0.0

    def pan_delta_deg(self, gripper_xy_mm: tuple[float, float]) -> float:
        """Small-angle conversion of a lateral offset to a shoulder-pan bias for a real ACT policy."""
        x, y = gripper_xy_mm
        r = max(np.hypot(x, y), 80.0)
        tangent = np.array([-y, x]) / r
        return float(np.degrees(np.dot([self.dx_mm, self.dy_mm], tangent) / r))
