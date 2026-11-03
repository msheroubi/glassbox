"""Shared async HTTP client for the Meta Model API (https://api.meta.ai/v1).

Network resilience for live demos:
* retries with exponential backoff on 429 / 5xx / transport errors (4xx are not retried),
* a disk cache of successful JSON responses keyed by the exact request body, so a rehearsed request
  replays instantly if the network drops on stage,
* every result is tagged with its source (``meta-api`` | ``cache``) and callers fall back to local
  engines (tagged ``local-fallback``) when both fail - nothing on the hot path raises.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from backend.config import Settings

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class MetaAPIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class MetaClient:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.enabled = settings.meta_api_enabled
        self.cache_dir = Path(settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.http = httpx.AsyncClient(
            base_url=settings.meta_api_base,
            headers={"Authorization": f"Bearer {settings.model_api_key}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    # ------------------------------------------------------------------ cache
    def _cache_path(self, body: dict) -> Path:
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:32]
        return self.cache_dir / f"{body.get('model', 'x')}-{key}.json"

    def _cache_get(self, body: dict) -> dict | None:
        p = self._cache_path(body)
        return json.loads(p.read_text()) if p.exists() else None

    def _cache_put(self, body: dict, data: dict) -> None:
        try:
            self._cache_path(body).write_text(json.dumps(data))
        except OSError as e:
            log.warning("Could not write API cache: %s", e)

    # ------------------------------------------------------------------ requests
    async def post_json(self, path: str, body: dict, timeout: float) -> tuple[dict, str]:
        """POST JSON; returns ``(response, source)`` or raises ``MetaAPIError`` if no cache exists."""
        if not self.enabled:
            raise MetaAPIError("MODEL_API_KEY is not configured")
        last_err: Exception | None = None
        for attempt in range(self.s.http_retries + 1):
            try:
                r = await self.http.post(path, json=body, timeout=timeout)
                if r.status_code == 200:
                    data = r.json()
                    self._cache_put(body, data)
                    return data, "meta-api"
                last_err = MetaAPIError(f"{path} -> HTTP {r.status_code}: {r.text[:400]}", r.status_code)
                if r.status_code not in RETRYABLE_STATUS:
                    break
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_err = MetaAPIError(f"{path} -> {type(e).__name__}: {e}")
            await asyncio.sleep(min(8.0, 0.5 * 2**attempt))
        cached = self._cache_get(body)
        if cached is not None:
            log.warning("Meta API failed (%s); serving cached response", last_err)
            return cached, "cache"
        raise last_err or MetaAPIError(f"{path} failed")

    async def stream_events(self, path: str, body: dict, timeout: float) -> AsyncIterator[dict]:
        """POST with ``stream: true`` and yield parsed server-sent events (Responses API)."""
        if not self.enabled:
            raise MetaAPIError("MODEL_API_KEY is not configured")
        async with self.http.stream("POST", path, json={**body, "stream": True}, timeout=timeout) as r:
            if r.status_code != 200:
                raise MetaAPIError(f"{path} -> HTTP {r.status_code}: {(await r.aread())[:400]!r}", r.status_code)
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    yield json.loads(payload)

    async def upload_file(self, content: bytes, filename: str, mime: str) -> str:
        if not self.enabled:
            raise MetaAPIError("MODEL_API_KEY is not configured")
        r = await self.http.post(
            "/files", files={"file": (filename, content, mime)}, data={"purpose": "user_data"}, timeout=120.0
        )
        if r.status_code != 200:
            raise MetaAPIError(f"/files -> HTTP {r.status_code}: {r.text[:400]}", r.status_code)
        return r.json()["id"]


def output_text(response: dict) -> str:
    """Concatenate ``output_text`` parts of assistant messages (reasoning items are skipped)."""
    parts = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                parts.append(c.get("text", ""))
            elif c.get("type") == "refusal":
                raise MetaAPIError(f"Model refused: {c.get('refusal')}")
    return "".join(parts)


def user_message(*content: dict) -> list[dict]:
    return [{"type": "message", "role": "user", "content": list(content)}]


def json_schema_format(name: str, schema: dict) -> dict:
    return {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": False}}
