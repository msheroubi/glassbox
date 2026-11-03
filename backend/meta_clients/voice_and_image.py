"""Muse Voice Transcribe (realtime ASR relay) and Muse Image (counterfactual frames).

Voice: the browser streams 16 kHz mono PCM16 to ``/ws/voice``; ``VoiceRelay`` forwards it to
``wss://api.meta.ai/v1/asr/realtime`` (handshake -> binary audio -> ``endStream``) and yields
transcript events back. Push-to-talk mode, with robot vocabulary passed as keywords.

Image: ``POST /v1/images/edits`` with the anomaly frame and a "what if the culprit were gone"
instruction, so the operator can see the scene the policy *should* have seen. Offline fallback:
the culprit mask is filled with the calibrated background plate (the same operation the live mask
suppression performs) and the planned recovery path is drawn as a ghost overlay.
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
from backend.schemas import XAIDiagnosisResponse
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
    diag: XAIDiagnosisResponse, culprit: str | None, where: str | None = None, target: str = "red cube"
) -> str | None:
    """``culprit`` is the SAM noun phrase of the culprit mask, ``where`` its image region."""
    keep = (f"Keep the {target}, the robot arm, the camera viewpoint, lighting, table texture and every other "
            "object exactly unchanged.")
    at = f" in the {where} of the image" if where else ""
    if diag.failure_type == "ATTENTION_HIJACK":
        return f"Remove only the {culprit or 'distracting object'}{at} from the table, showing bare table where it was. {keep}"
    if diag.failure_type == "OCCLUSION_OOD":
        return f"Remove the human hand and arm from the scene so the {target} underneath is fully visible. {keep}"
    if diag.failure_type == "SPECULAR_GLARE":
        return (f"Remove the white specular glare from the {target} so the whole cube is evenly lit, matte red. "
                "Keep the robot arm, camera viewpoint, table and every other object exactly unchanged.")
    if diag.failure_type == "SERVO_STALL_COLLISION":
        return f"Remove only the {culprit or 'obstacle'}{at} from the table, showing bare table where it was. {keep}"
    return None


class CounterfactualRenderer:
    def __init__(self, settings: Settings, client: MetaClient) -> None:
        self.s = settings
        self.client = client

    async def render(
        self,
        frame_bgr: np.ndarray,
        prompt: str,
        culprit_mask: np.ndarray | None,
        background_plate: np.ndarray | None,
        path_px: list[tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, str]:
        """Returns ``(image_bgr, source)``. Never raises."""
        if self.client.enabled:
            body = {
                "model": self.s.image_model,
                "prompt": prompt,
                "images": [{"image_url": data_url(frame_bgr, 90)}],
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
                if img.shape[:2] != frame_bgr.shape[:2]:
                    img = cv2.resize(img, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_AREA)
                return self._annotate(img, path_px, "Muse Image counterfactual"), source
            except Exception as e:  # noqa: BLE001
                log.warning("Muse Image unavailable (%s); rendering local counterfactual", e)
        return self._annotate(self._local(frame_bgr, culprit_mask, background_plate), path_px, "local counterfactual"), "local-fallback"

    @staticmethod
    def _local(frame: np.ndarray, mask: np.ndarray | None, plate: np.ndarray | None) -> np.ndarray:
        if mask is None or not mask.any():
            return frame.copy()
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        region = cv2.dilate(mask.astype(np.uint8), k)
        if plate is not None and plate.shape == frame.shape:
            alpha = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 4)[..., None]
            return (frame * (1 - alpha) + plate * alpha).astype(np.uint8)
        return cv2.inpaint(frame, region, 7, cv2.INPAINT_TELEA)

    @staticmethod
    def _annotate(img: np.ndarray, path_px, caption: str) -> np.ndarray:
        out = draw_path(img.copy(), path_px, (80, 255, 80), ghost=True) if path_px else img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(out, caption, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return out
