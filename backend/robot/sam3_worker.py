"""SAM 3 (``facebook/sam3``) as a worker process: where the model finds each noun phrase in a frame.

Runs in its own interpreter (``.venv-sam3``: transformers >= 5, which SAM 3 needs, on top of the lerobot
environment's torch), so the main process keeps the transformers lerobot was built against.
Imports nothing from ``backend``.

Protocol, over stdin/stdout: each message is a 4-byte big-endian length and a pickle.
- request ``{"rgb": uint8 (H, W, 3), "prompts": [str, ...]}``
- reply ``{"maps": {prompt: {"instances": float32 (h, w), "semantic": float32 (h, w)}},
  "scores": {prompt: [float, ...]}, "presence": {prompt: float}, "ms": float}`` or ``{"error": str}``.

The image goes through the vision encoder once; each prompt then reuses its features.
"""

from __future__ import annotations

import os
import pickle
import struct
import sys
import time

MODEL_ID = os.environ.get("GLASSBOX_SAM3_MODEL", "facebook/sam3")


def _read(stream) -> object | None:
    head = stream.read(4)
    if len(head) < 4:
        return None
    (n,) = struct.unpack(">I", head)
    return pickle.loads(stream.read(n))


def _write(stream, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">I", len(data)) + data)
    stream.flush()


def main() -> None:
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    torch.set_grad_enabled(False)
    if os.environ.get("GLASSBOX_SAM3_THREADS"):
        torch.set_num_threads(int(os.environ["GLASSBOX_SAM3_THREADS"]))
    t0 = time.perf_counter()
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model = Sam3Model.from_pretrained(MODEL_ID).eval()
    text_cache: dict[str, tuple] = {}
    out, inp = sys.stdout.buffer, sys.stdin.buffer
    sys.stdout = sys.stderr  # nothing but replies on the pipe
    print(f"sam3-worker: loaded {MODEL_ID} in {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True)
    _write(out, {"ready": True})

    while (req := _read(inp)) is not None:
        try:
            t = time.perf_counter()
            image = Image.fromarray(req["rgb"])
            pix = processor(images=image, return_tensors="pt")
            vision = model.get_vision_features(pixel_values=pix.pixel_values)
            maps, scores, presence = {}, {}, {}
            for prompt in req["prompts"]:
                if prompt not in text_cache:
                    ti = processor(text=prompt, return_tensors="pt")
                    text_cache[prompt] = (model.get_text_features(input_ids=ti.input_ids, attention_mask=ti.attention_mask),
                                          ti.attention_mask)
                text, mask = text_cache[prompt]
                o = model(vision_embeds=vision, text_embeds=text, attention_mask=mask)
                pres = o.presence_logits.sigmoid().reshape(-1)[0] if o.presence_logits is not None else torch.tensor(1.0)
                s = o.pred_logits[0].sigmoid() * pres  # (Q,) each query's confidence that it is this phrase
                inst = (s[:, None, None] * o.pred_masks[0].sigmoid()).sum(0).clamp(max=1.0)
                sem = o.semantic_seg[0, 0].sigmoid() if o.semantic_seg is not None else inst
                maps[prompt] = {"instances": inst.float().numpy(), "semantic": sem.float().numpy()}
                scores[prompt] = sorted((round(float(v), 3) for v in s if v > 0.05), reverse=True)[:8]
                presence[prompt] = round(float(pres), 3)
            _write(out, {"maps": maps, "scores": scores, "presence": presence,
                         "size_hw": list(req["rgb"].shape[:2]), "ms": round((time.perf_counter() - t) * 1000)})
        except Exception as e:  # noqa: BLE001 - report it and keep serving
            _write(out, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
