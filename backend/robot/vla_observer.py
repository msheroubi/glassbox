"""Observe-only SmolVLA: attention maps of a pretrained vision-language-action model, no arm commands.

SmolVLA (``lerobot/smolvla_base``) feeds a VLM prefix ``[image tokens per camera, task tokens, state]``
to an action expert that denoises a 50-step action chunk. Its attention runs through
``SmolVLMWithExpertModel.eager_attention_forward``, which this wrapper intercepts to read two maps over
the overhead camera's image tokens:

* ``action``: the action expert's queries -> image tokens, averaged over heads, layers, chunk steps and
  the 10 denoising steps. This is where the policy looks when it decides what to do.
* ``language``: the task's tokens -> image tokens inside the VLM prefix. This shows which part of the
  image the model grounds the instruction ("red cube", "green tile") to.

Each map is renormalised over the image tokens first, so attention sinks on text or state tokens don't
wash it out. It is then divided by the same map for a flat grey frame: SmolVLM puts 2-7x the uniform
share on the right-hand token column and the last token whatever the image shows, and the ratio leaves
only what the content adds. Images are padded to a 512x512 square on the top and left, so the token
grid (8x8 for SmolVLM2-500M) is cropped back to the frame's own area before upsampling.

Both maps come out of the same pass. Inference takes about 6 s per pass on CPU, so live it runs on a
worker thread: ``step()`` never blocks, it hands over the newest frame and returns a map whenever a pass
has finished. ``analyze_frame()`` is the blocking call the offline replay analyser uses on recorded frames.
"""

from __future__ import annotations

import logging
import math
import threading
import time

import cv2
import numpy as np

from backend.robot.policy_hook import PolicyStep, normalize_map

log = logging.getLogger(__name__)

# Median first-frame state of devsheroubi/so101metahack: the arm at rest. Camera-only mode has no joint
# readings, and an all-zeros state is far outside anything the model was trained on.
REST_POSE_DEG = np.array([-7.0, -104.3, 94.1, 81.5, 8.5, 0.8], dtype=np.float32)
RESIZE_PAD = 512
MAP_KINDS = ("action", "language")


class SmolVLAObserver:
    lens_name = "SmolVLA"
    lens_maps = {"vla_language": None, "vla_action": None}  # replay map kinds (no grounded concept label)

    def __init__(
        self,
        policy_path: str,
        device: str,
        camera_key: str,
        vla_camera_key: str,
        overhead_size_wh: tuple[int, int],
        task: str,
        map_kind: str = "action",
    ) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if map_kind not in MAP_KINDS:
            raise ValueError(f"map_kind must be 'action' or 'language', not {map_kind!r}")
        self.torch = torch
        self.device = torch.device(device)
        t0 = time.perf_counter()
        self.policy = SmolVLAPolicy.from_pretrained(policy_path)
        self.policy.config.device = device
        self.policy.to(self.device).eval()
        self.preprocessor, _ = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.camera_key = camera_key
        self.vla_key = f"observation.images.{vla_camera_key}"
        image_keys = list(self.policy.config.image_features)
        if self.vla_key not in image_keys:
            raise ValueError(f"{self.vla_key} not in SmolVLA image features {image_keys}")
        # Only the overhead camera is fed; prepare_images drops the missing ones, so it is the first image.
        self.size_wh = overhead_size_wh
        self.task = task
        self.map_kind = map_kind
        self.method = f"smolvla_{map_kind}_attention"
        self.chunk_size = self.policy.config.chunk_size

        vwe = self.policy.model.vlm_with_expert
        self._n_img = 0
        self._prefix_pass = False
        self._acc: dict[str, np.ndarray] = {}
        self._n_acc: dict[str, int] = {}
        self._lang_mask = None
        # frame (h, w) -> grey-frame token attention per map kind
        self._baseline: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self._infer_lock = threading.Lock()  # the hooks share state: one pass at a time

        orig_embed, orig_forward, orig_attn = vwe.embed_image, vwe.forward, vwe.eager_attention_forward

        def embed_image(image):
            out = orig_embed(image)
            self._n_img = out.shape[1]
            return out

        def forward(*args, **kwargs):
            self._prefix_pass = bool(kwargs.get("fill_kv_cache"))
            return orig_forward(*args, **kwargs)

        def attn(mask, batch_size, head_dim, q, k, v):
            self._capture(mask, head_dim, q, k)
            return orig_attn(mask, batch_size, head_dim, q, k, v)

        vwe.embed_image, vwe.forward, vwe.eager_attention_forward = embed_image, forward, attn

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self._result: np.ndarray | None = None
        self.last_chunk: np.ndarray | None = None
        self.last_latency_s = 0.0
        self._thread = threading.Thread(target=self._worker, name="glassbox-smolvla", daemon=True)
        self._thread.start()
        self.queue: list = []  # the control loop clears this on resets; nothing is ever queued
        log.info("Loaded SmolVLA from %s in %.0fs (observe only, %s attention)", policy_path, time.perf_counter() - t0, map_kind)

    # ------------------------------------------------------------------ attention capture
    def _capture(self, mask, head_dim, q, k) -> None:
        n = self._n_img
        if not n:
            return
        torch = self.torch
        if self._prefix_pass:  # VLM prefix: the task's tokens -> image tokens
            if self._lang_mask is None:
                return
            kind = "language"
            rows = torch.nonzero(self._lang_mask[0]).flatten() + n
            qq, m = q[:, rows].float(), mask[:, rows]
        else:  # action expert denoising step: every action query -> image tokens
            if q.shape[1] != self.chunk_size:
                return
            kind = "action"
            qq, m = q.float(), mask
        groups = q.shape[2] // k.shape[2]
        kk = k.repeat_interleave(groups, dim=2).float()  # matches the expand/reshape in eager_attention_forward
        att = torch.einsum("blhd,bshd->bhls", qq, kk) * head_dim**-0.5
        att = att.masked_fill(~m[:, None], torch.finfo(att.dtype).min).softmax(-1)
        img = att[0, :, :, :n]
        img = img / img.sum(-1, keepdim=True).clamp_min(1e-9)
        grid = img.mean((0, 1)).cpu().numpy()
        self._acc[kind] = grid + self._acc.get(kind, 0.0)
        self._n_acc[kind] = self._n_acc.get(kind, 0) + 1

    def _token_map_to_frame(self, tokens: np.ndarray, frame_hw: tuple[int, int]) -> np.ndarray:
        side = int(round(math.sqrt(len(tokens))))
        grid = tokens.reshape(side, side)  # pixel-shuffled SigLIP patches, row-major
        h, w = frame_hw
        ratio = max(w / RESIZE_PAD, h / RESIZE_PAD)
        rh, rw = int(h / ratio), int(w / ratio)
        sq = cv2.resize(grid.astype(np.float32), (RESIZE_PAD, RESIZE_PAD), interpolation=cv2.INTER_CUBIC)
        crop = sq[RESIZE_PAD - rh :, RESIZE_PAD - rw :]  # padding is on the top and left
        return normalize_map(crop, self.size_wh)

    # ------------------------------------------------------------------ inference
    def _token_attention(self, state: np.ndarray, rgb: np.ndarray) -> dict[str, np.ndarray]:
        from lerobot.policies.utils import prepare_observation_for_inference

        obs = {"observation.state": state.astype(np.float32), self.vla_key: rgb}
        batch = prepare_observation_for_inference(obs, self.device, self.task, "so_follower")
        batch = self.preprocessor(batch)
        self._lang_mask = batch.get("observation.language.attention_mask")
        self._acc, self._n_acc = {}, {}
        with self.torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
        self.last_chunk = chunk[0].float().cpu().numpy()
        return {kind: acc / self._n_acc[kind] for kind, acc in self._acc.items()}

    def _infer(self, state: np.ndarray, rgb: np.ndarray) -> dict[str, np.ndarray]:
        with self._infer_lock:
            shape = rgb.shape[:2]
            if shape not in self._baseline:
                # Attention on a flat grey frame of the same size: what the model attends to by position alone
                # (the right-hand column and the last token get 2-7x uniform on every frame).
                self._baseline[shape] = self._token_attention(state, np.full_like(rgb, 128))
            tokens = self._token_attention(state, rgb)
            base = self._baseline[shape]
            return {
                kind: self._token_map_to_frame(t / np.maximum(base[kind], 1e-6), shape)
                for kind, t in tokens.items() if kind in base
            }

    def analyze_frame(self, state: np.ndarray, rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Blocking: both attention maps ("action", "language") for one recorded RGB frame and joint state."""
        q = np.asarray(state, np.float32)
        return self._infer(REST_POSE_DEG if not np.any(q) else q, rgb)

    def lens_frame(self, state: np.ndarray, rgb: np.ndarray) -> dict[str, np.ndarray]:
        """The replay lens interface (as ``Sam3Observer``): both maps keyed by replay map kind."""
        return {f"vla_{kind}": sal for kind, sal in self.analyze_frame(state, rgb).items()}

    def attend(self, state: np.ndarray, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, str, None]:
        """Blocking, observe only: this observer's map for an (edited) overhead frame."""
        sal = self.analyze_frame(state, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).get(self.map_kind)
        return sal, self.method, None

    def _worker(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                continue
            t0 = time.perf_counter()
            try:
                sal = self._infer(*job).get(self.map_kind)
            except Exception:  # noqa: BLE001 - keep streaming; the map just stops updating
                log.exception("SmolVLA inference failed")
                time.sleep(2.0)
                continue
            self.last_latency_s = time.perf_counter() - t0
            log.debug("SmolVLA pass %.2fs", self.last_latency_s)
            with self._lock:
                self._result = sal

    def reset(self) -> None:
        with self._lock:
            self._result = None

    def step(self, state: np.ndarray, images: dict[str, np.ndarray]) -> PolicyStep:
        rgb = images.get(self.camera_key)
        q = np.asarray(state, np.float32)
        model_state = REST_POSE_DEG if not np.any(q) else q
        with self._lock:
            if rgb is not None:
                self._pending = (model_state, rgb.copy())
                self._wake.set()
            saliency, self._result = self._result, None
        return PolicyStep(action=np.asarray(state, np.float64).copy(), chunk_variance=0.0,
                          saliency=saliency, saliency_method=self.method)
