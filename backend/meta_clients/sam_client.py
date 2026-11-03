"""SAM 3.1 segmentation through ``POST /v1/responses`` (model ``sam-3.1``).

SAM takes one short noun-phrase concept per request, so concepts are segmented concurrently. Output
is SAM's special-token text grammar (``<0f>0<|box;...|><|mask;...;data=H,W,!payload|>``); it is
parsed with Meta's ``meta-sam-parser`` and each box-local raster is placed into a full-frame mask.

Keyframe mode segments the anomaly frame (~2 s). Clip mode uploads the ring-buffer clip as MP4 via
the Files API and uses SAM's video tracking, keeping the masks from the last frame.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from collections.abc import Callable

import cv2
import numpy as np
from meta_sam_parser import (
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)

from backend.config import SegmentationConcept, Settings
from backend.xai_engine.attribution import SegmentedObject
from backend.xai_engine.visualize import encode_jpeg

from .base import MetaAPIError, MetaClient, user_message

log = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.3

GroundTruthFn = Callable[[], list]  # returns objects with .label, .prompt, .mask


def _place_mask(rec: SegmentationMaskRecord, frame_hw: tuple[int, int]) -> np.ndarray:
    raster = np.frombuffer(decode_mask_to_raster(rec.mask), np.uint8).reshape(rec.mask.height, rec.mask.width)
    b = rec.bounds
    left, top, right, bottom = int(b.left), int(b.top), int(b.right), int(b.bottom)
    full = np.zeros(frame_hw, bool)
    if right <= left or bottom <= top:
        return full
    patch = cv2.resize(raster.astype(np.float32), (right - left, bottom - top), interpolation=cv2.INTER_LINEAR) > 0.5
    full[top:bottom, left:right] = patch[: frame_hw[0] - top, : frame_hw[1] - left]
    return full


class SAMClient:
    def __init__(self, settings: Settings, client: MetaClient, ground_truth: GroundTruthFn | None = None) -> None:
        self.s = settings
        self.client = client
        self.ground_truth = ground_truth

    async def _events(self, body: dict, cache_key: str):
        """Stream events live; record the output text so a later identical request can replay offline."""
        cache_path = self.client.cache_dir / f"sam-{cache_key}.txt"
        lane = {"item_id": "cached", "output_index": 0, "content_index": 0}
        try:
            text = []
            async for ev in self.client.stream_events("/responses", body, self.s.sam_timeout_s):
                if ev.get("type") == "response.output_text.delta":
                    text.append(ev.get("delta", ""))
                if ev.get("type") == "response.completed":
                    cache_path.write_text("".join(text))
                yield ev
        except Exception:
            if not cache_path.exists():
                raise
            log.warning("SAM stream failed; replaying cached segmentation %s", cache_path.name)
            cached = cache_path.read_text()
            if cached:
                yield {"type": "response.output_text.delta", **lane, "delta": cached}
            yield {"type": "response.output_text.done", **lane, "text": cached}
            yield {"type": "response.completed"}

    async def _segment_concept(self, media_part: dict, concept: SegmentationConcept, frame_hw, video: bool):
        for prompt in [concept.prompt, *concept.fallback_prompts]:
            found = await self._segment_prompt(media_part, concept, prompt, frame_hw, video)
            if found:
                return found
        return []

    async def _segment_prompt(self, media_part: dict, concept: SegmentationConcept, prompt: str, frame_hw, video: bool):
        body = {
            "model": self.s.sam_model,
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
            "input": user_message({"type": "input_text", "text": prompt}, media_part),
        }
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:32]
        fmt = video_segmentation_format() if video else image_segmentation_format()
        result = await parse_responses_stream(self._events(body, key), fmt).final_result()
        masks = [r for r in result.records if isinstance(r, SegmentationMaskRecord)]
        if video and masks:  # keep each tracked object's mask from its latest frame
            last = {}
            for r in masks:
                fi = r.frame.frame_index if r.frame else 0
                if r.object_id not in last or fi >= (last[r.object_id].frame.frame_index if last[r.object_id].frame else 0):
                    last[r.object_id] = r
            masks = list(last.values())
        found = concept.model_copy(update={"prompt": prompt})
        return [
            (found, r.confidence, _place_mask(r, frame_hw))
            for r in masks
            if r.confidence is None or r.confidence >= MIN_CONFIDENCE
        ]

    async def segment(
        self, frame_bgr: np.ndarray, clip_mp4: bytes | None = None, concepts: list[SegmentationConcept] | None = None
    ) -> tuple[list[SegmentedObject], str]:
        """Returns ``(objects, source)``. Never raises."""
        concepts = concepts or self.s.concepts
        frame_hw = frame_bgr.shape[:2]
        objects: list[SegmentedObject] = []
        source = "local-fallback"
        if self.client.enabled:
            try:
                video = clip_mp4 is not None
                if video:
                    file_id = await self.client.upload_file(clip_mp4, "glassbox_clip.mp4", "video/mp4")
                    media = {"type": "input_video", "file_id": file_id}
                else:
                    url = "data:image/jpeg;base64," + base64.b64encode(encode_jpeg(frame_bgr, 90)).decode()
                    media = {"type": "input_image", "image_url": url}
                results = await asyncio.gather(
                    *(self._segment_concept(media, c, frame_hw, video) for c in concepts), return_exceptions=True
                )
                errors = [r for r in results if isinstance(r, Exception)]
                if len(errors) == len(results):
                    raise errors[0]
                for err in errors:
                    log.warning("SAM concept failed: %s", err)
                for res in results:
                    if isinstance(res, Exception):
                        continue
                    for concept, conf, mask in res:
                        if mask.any():
                            objects.append(
                                SegmentedObject(len(objects), concept.label, concept.prompt, mask, conf, "sam-3.1")
                            )
                source = "meta-api"
            except (MetaAPIError, Exception) as e:  # noqa: BLE001 - demo must never crash on SAM
                log.warning("SAM 3.1 unavailable (%s); using fallback segmentation", e)
                objects = []
        # Fill concepts SAM did not find (e.g. a stylised simulator arm) from simulator ground truth.
        if self.ground_truth is not None:
            found = {o.label for o in objects}
            for gt in self.ground_truth():
                if gt.label not in found and gt.mask.any():
                    objects.append(SegmentedObject(len(objects), gt.label, gt.prompt, gt.mask, None, "sim-ground-truth"))
        return objects, source

    async def refresh_track(self, frame_bgr: np.ndarray, previous: SegmentedObject) -> np.ndarray | None:
        """Re-segment one concept and return the mask that best overlaps the previous one."""
        concept = SegmentationConcept(label=previous.label, prompt=previous.prompt)
        objs, source = await self.segment(frame_bgr, concepts=[concept])
        candidates = [o for o in objs if o.label == previous.label and (source == "meta-api" or o.source != "sam-3.1")]
        if not candidates:
            return None
        best = max(candidates, key=lambda o: np.logical_and(o.mask, previous.mask).sum() / (o.mask.sum() + 1))
        return best.mask
