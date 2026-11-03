"""Rendering helpers: saliency heatmaps, SAM contour overlays and the composite diagnostic keyframe."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from backend.schemas import MaskAttribution

from .attribution import SegmentedObject

LABEL_COLOURS = {  # BGR
    "gripper": (255, 200, 0),
    "target": (80, 220, 80),
    "distractor": (255, 0, 255),
    "obstacle": (0, 165, 255),
    "hand": (0, 255, 255),
}


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def data_url(img: np.ndarray, quality: int = 85) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(encode_jpeg(img, quality)).decode()


def heatmap_overlay(frame: np.ndarray, saliency: np.ndarray | None, alpha: float = 0.45) -> np.ndarray:
    if saliency is None:
        return frame.copy()
    heat = cv2.applyColorMap(np.clip(saliency * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    weight = (np.clip(saliency, 0, 1) * alpha)[..., None]
    return (frame * (1 - weight) + heat * weight).astype(np.uint8)


def draw_objects(img: np.ndarray, objects: list[SegmentedObject], attributions: list[MaskAttribution] | None = None) -> np.ndarray:
    shares = {a.mask_id: a.attention_share for a in attributions or []}
    for obj in objects:
        colour = LABEL_COLOURS.get(obj.label, (255, 255, 255))
        polys = [np.array(p, np.int32) for p in obj.polygons()]
        cv2.polylines(img, polys, True, colour, 3, cv2.LINE_AA)
        x1, y1, _, _ = obj.bbox
        text = f"#{obj.mask_id} {obj.label}"
        if obj.mask_id in shares:
            text += f"  A={shares[obj.mask_id]:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = max(th + 6, y1 - 6)
        cv2.rectangle(img, (x1, y - th - 6), (x1 + tw + 8, y + 4), (0, 0, 0), -1)
        cv2.putText(img, text, (x1 + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return img


def draw_path(img: np.ndarray, points_px: list[tuple[float, float]], colour=(255, 255, 255), ghost: bool = False) -> np.ndarray:
    if len(points_px) < 2:
        return img
    pts = np.array(points_px, np.int32)
    layer = img.copy()
    cv2.polylines(layer, [pts], False, colour, 6 if ghost else 3, cv2.LINE_AA)
    for p in pts[:: max(1, len(pts) // 8)]:
        cv2.circle(layer, tuple(p), 9 if ghost else 5, colour, -1, cv2.LINE_AA)
    cv2.arrowedLine(layer, tuple(pts[-2]), tuple(pts[-1]), colour, 4, cv2.LINE_AA, tipLength=0.6)
    return cv2.addWeighted(layer, 0.6, img, 0.4, 0) if ghost else layer


def composite_keyframe(
    frame: np.ndarray,
    saliency: np.ndarray | None,
    objects: list[SegmentedObject],
    attributions: list[MaskAttribution],
    caption: str = "",
) -> np.ndarray:
    """Raw frame | frame + SAM contours + policy heatmap, side by side, for Muse Spark."""
    annotated = draw_objects(heatmap_overlay(frame, saliency), objects, attributions)
    combo = np.hstack([frame, annotated])
    if caption:
        cv2.rectangle(combo, (0, 0), (combo.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(combo, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # Keep the upload small: Spark's latency budget is ~2 s.
    scale = min(1.0, 1600 / combo.shape[1])
    return cv2.resize(combo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else combo
