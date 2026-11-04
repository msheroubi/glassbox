"""Muse Voice Transcribe (realtime ASR relay) and Muse Image (counterfactual frames).

Voice: the browser streams 16 kHz mono PCM16 to ``/ws/voice``; ``VoiceRelay`` forwards it to
``wss://api.meta.ai/v1/asr/realtime`` (handshake -> binary audio -> ``endStream``) and yields
transcript events back. Push-to-talk mode, with robot vocabulary passed as keywords.

Image: ``POST /v1/images/edits`` renders the scene the policy *should* have seen, guided by SAM's mask
(see ``backend.xai_engine.counterfactual``): the culprit's region is erased locally first, Muse only
cleans the patch up, and its pixels are composited back inside that region alone. Offline fallback:
the local erase (background plate, or inpainting).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from collections.abc import AsyncIterator

import cv2
import numpy as np

from backend.config import Settings
from backend.xai_engine import counterfactual as cf
from backend.xai_engine.visualize import data_url, draw_path

from .base import MetaAPIError, MetaClient

log = logging.getLogger(__name__)

ROBOT_KEYWORDS = ["SO-101", "gripper", "cube", "bin", "analyze", "apply fix", "e-stop", "resume", "attention", "suppress"]


class VoiceRelay:
    """One push-to-talk utterance: ``send_audio`` chunks, then ``finish``; iterate ``events``."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.ws = None

    async def open(self) -> None:
        from websockets.asyncio.client import connect

        if not self.s.meta_api_enabled:
            raise MetaAPIError("MODEL_API_KEY is not configured")
        url = f"{self.s.meta_asr_ws_url}?sessionId={uuid.uuid4()}"
        self.ws = await asyncio.wait_for(connect(url, max_size=2**22, open_timeout=10), timeout=12)
        await self.ws.send(
            json.dumps(
                {
                    "authorization": {"accessToken": f"Bearer {self.s.model_api_key}"},
                    "audioEncoding": "PCM_16KHZ",
                    "model": self.s.voice_model,
                    "mode": "PUSH_TO_TALK",
                    "partialMode": "CUMULATIVE",
                    "keywords": ROBOT_KEYWORDS,
                }
            )
        )
        ack = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=10))
        if ack.get("type") == "error":
            raise MetaAPIError(f"ASR handshake rejected: {ack.get('message')}")
        log.info("Muse Voice session %s open", ack.get("sessionId"))

    async def send_audio(self, pcm16: bytes) -> None:
        await self.ws.send(pcm16)

    async def finish(self) -> None:
        await self.ws.send(json.dumps({"type": "endStream"}))

    async def events(self) -> AsyncIterator[dict]:
        from websockets.exceptions import ConnectionClosed

        try:
            async for msg in self.ws:
                if isinstance(msg, bytes):
                    continue
                ev = json.loads(msg)
                if ev.get("type") in ("transcript", "error"):
                    yield ev
        except ConnectionClosed as e:
            if e.rcvd and e.rcvd.code not in (1000, None):
                yield {"type": "error", "message": f"ASR closed with code {e.rcvd.code}: {e.rcvd.reason}"}

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()


def image_region(bbox: list[int], frame_wh: tuple[int, int]) -> str:
    """Coarse words for where a box sits in the image ("upper left"), so edit prompts are unambiguous."""
    u, v = (bbox[0] + bbox[2]) / 2 / frame_wh[0], (bbox[1] + bbox[3]) / 2 / frame_wh[1]
    row = "upper" if v < 0.4 else "lower" if v > 0.6 else "middle"
    col = "left" if u < 0.4 else "right" if u > 0.6 else "centre"
    return "centre" if (row, col) == ("middle", "centre") else f"{row} {col}"


def counterfactual_prompt(
    failure_type: str, culprit: str | None, where: str | None = None, target: str = "red cube",
    erased: bool = False,
) -> str | None:
    """``culprit`` is the SAM noun phrase of the culprit mask, ``where`` its image region. ``erased``: the culprit
    was already filled in locally, so Muse only has to turn the smudge into clean table (asking it to "remove"
    an object it can't see makes it invent one)."""
    keep = (f"Keep the {target}, the robot arm, the camera viewpoint, lighting, table texture and every other "
            "object exactly unchanged.")
    at = f" in the {where} of the image" if where else ""
    if failure_type in ("ATTENTION_HIJACK", "SERVO_STALL_COLLISION"):
        if erased:
            return (f"Clean up the blurry smudged patch on the table{at} so it looks like plain, bare tabletop matching "
                    "the surrounding surface, lighting and texture. Do not add any object there. Keep everything "
                    "else in the image exactly unchanged.")
        default = "distracting object" if failure_type == "ATTENTION_HIJACK" else "obstacle"
        return f"Remove only the {culprit or default}{at} from the table, showing bare table where it was. {keep}"
    if failure_type == "OCCLUSION_OOD":
        return f"Remove the human hand and arm from the scene so the {target} underneath is fully visible. {keep}"
    if failure_type == "SPECULAR_GLARE":
        return (f"Remove the white specular glare from the {target} so the whole cube is evenly lit, matte red. "
                "Keep the robot arm, camera viewpoint, table and every other object exactly unchanged.")
    return None


class CounterfactualRenderer:
    def __init__(self, settings: Settings, client: MetaClient) -> None:
        self.s = settings
        self.client = client

    async def render(
        self,
        frame_bgr: np.ndarray,
        prompt: str,
        region: np.ndarray | None,
        background_plate: np.ndarray | None,
        culprit_mask: np.ndarray | None = None,
        pre_erase: bool = False,
    ) -> tuple[np.ndarray, str]:
        """Returns ``(image_bgr, source)``, the clean counterfactual frame. Never raises.

        ``region``: the only pixels the edit may change. Muse's output is composited back inside it, so the
        rest is the original frame; ``None`` accepts the whole edited frame. ``culprit_mask``: what must be
        gone; a result where it survives is rejected. ``pre_erase``: fill ``region`` locally before Muse sees
        the frame (an object to remove from bare table), and reject a result with anything but table there."""
        local = cf.erase_local(frame_bgr, region, background_plate) if pre_erase and region is not None else None
        if self.client.enabled:
            body = {
                "model": self.s.image_model,
                "prompt": prompt,
                "images": [{"image_url": data_url(local if local is not None else frame_bgr, 92)}],
                "n": 1,
                "response_format": "b64_json",
                "output_format": "jpeg",
            }
            try:
                data, source = await self.client.post_json("/images/edits", body, self.s.image_timeout_s)
                item = data["data"][0]
                if item.get("b64_json"):
                    raw = base64.b64decode(item["b64_json"])
                else:
                    r = await self.client.http.get(item["url"], timeout=30)
                    r.raise_for_status()
                    raw = r.content
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise MetaAPIError("Muse Image returned undecodable image data")
                img = await asyncio.to_thread(cf.align, img, frame_bgr)
                if region is not None:
                    img = cf.composite(frame_bgr, img, region)
                if culprit_mask is not None and cf.culprit_still_visible(frame_bgr, img, culprit_mask):
                    raise MetaAPIError("the culprit is still visible in Muse Image's edit")
                if local is not None and not cf.looks_like_table(img, region):
                    raise MetaAPIError("Muse Image's edit put something where the culprit was")
                return img, source
            except Exception as e:  # noqa: BLE001
                log.warning("Muse Image counterfactual unusable (%s); using the local one", e)
        if local is not None:
            return local, "local-fallback"
        if region is not None and culprit_mask is not None:
            return cf.erase_local(frame_bgr, region, background_plate), "local-fallback"
        if region is not None:  # glare: inpaint the blown-out pixels on the target
            return cf.remove_glare(frame_bgr, region), "local-fallback"
        return frame_bgr.copy(), "local-fallback"

    @staticmethod
    def annotate(img: np.ndarray, caption: str, path_px=None, color=(80, 255, 80)) -> np.ndarray:
        out = draw_path(img.copy(), path_px, color, ghost=True) if path_px else img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(out, caption, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return out
