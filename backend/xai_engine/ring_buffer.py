"""Circular buffer of the last N control steps: overhead JPEG, policy saliency and servo telemetry.

When an intervention or demo staging edits the policy's input, the edited frame is kept too
(``policy_jpeg``), so a replay can show exactly what the policy saw.

Frames are stored JPEG-encoded (the same bytes streamed to the dashboard) and saliency at quarter
resolution, so 150 steps of 720p video cost ~20 MB instead of ~1 GB.
"""

from __future__ import annotations

import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from backend.schemas import ServoTelemetry

SALIENCY_DOWNSCALE = 4


@dataclass
class FrameRecord:
    index: int
    timestamp: float
    jpeg: bytes
    saliency_u8: np.ndarray | None  # (H/4, W/4) uint8
    telemetry: ServoTelemetry
    phase: str = ""
    policy_jpeg: bytes | None = None  # the policy's input when it differs from the raw frame

    def frame_bgr(self) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)

    def policy_frame_bgr(self) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.policy_jpeg or self.jpeg, np.uint8), cv2.IMREAD_COLOR)

    def saliency(self, size_wh: tuple[int, int]) -> np.ndarray | None:
        if self.saliency_u8 is None:
            return None
        return cv2.resize(self.saliency_u8, size_wh, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


def pack_saliency(saliency: np.ndarray | None) -> np.ndarray | None:
    if saliency is None:
        return None
    h, w = saliency.shape
    small = cv2.resize(saliency, (w // SALIENCY_DOWNSCALE, h // SALIENCY_DOWNSCALE), interpolation=cv2.INTER_AREA)
    return np.clip(small * 255.0, 0, 255).astype(np.uint8)


class RingBuffer:
    def __init__(self, capacity: int = 150) -> None:
        self._buf: deque[FrameRecord] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._next_index = 0

    def push(
        self, timestamp: float, jpeg: bytes, saliency: np.ndarray | None, telemetry: ServoTelemetry, phase: str = "",
        policy_jpeg: bytes | None = None,
    ) -> FrameRecord:
        with self._lock:
            rec = FrameRecord(self._next_index, timestamp, jpeg, pack_saliency(saliency), telemetry, phase, policy_jpeg)
            self._buf.append(rec)
            self._next_index += 1
            return rec

    def snapshot(self, last_n: int | None = None) -> list[FrameRecord]:
        with self._lock:
            items = list(self._buf)
        return items[-last_n:] if last_n else items

    def get(self, index: int) -> FrameRecord | None:
        with self._lock:
            for rec in reversed(self._buf):
                if rec.index == index:
                    return rec
        return None

    def latest(self) -> FrameRecord | None:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    @staticmethod
    def export_mp4(records: list[FrameRecord], fps: float = 30.0) -> bytes:
        """Encode records as an MPEG-4 Part 2 MP4 (a container/codec SAM 3.1 accepts)."""
        if not records:
            raise ValueError("No frames to export")
        first = records[0].frame_bgr()
        h, w = first.shape[:2]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(first)
            for rec in records[1:]:
                writer.write(rec.frame_bgr())
            writer.release()
            return path.read_bytes()
