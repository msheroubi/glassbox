"""Policy wrappers that expose spatial saliency and action-chunk variance.

``ACTRunner`` drives a trained LeRobot ``ACTPolicy``. It registers PyTorch forward hooks on

* ``policy.model.backbone`` (torchvision ``IntermediateLayerGetter`` -> ``{"feature_map": layer4}``),
  called once per camera in ``config.image_features`` order, and
* ``policy.model.decoder.layers[-1].multihead_attn`` (``nn.MultiheadAttention``), whose second
  output is the head-averaged cross-attention ``(B, chunk_size, n_encoder_tokens)``.

Encoder tokens are ``[latent, (robot_state), (env_state), cam0 h*w, cam1 h*w, ...]``, so the
overhead camera's slice of the cross-attention reshapes to an ``(h, w)`` map that is upsampled to
the overhead frame and normalised to ``[0, 1]``. If cross-attention is unavailable the channel L2
norm of the layer4 activations is used instead.

``ACTRunner`` also has ``attend(state, frame_bgr) -> (saliency, method, chunk)``: an observe-only pass with
no side effects on the running policy (queue, variance tracker), used to ask "where would it look, and
where would it go, in this edited frame?" for counterfactuals.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PolicyStep:
    action: np.ndarray  # (6,) joint targets, degrees (gripper 0-100)
    chunk_variance: float  # mean per-joint variance across overlapping chunk predictions (deg^2)
    saliency: np.ndarray | None  # (H, W) float32 in [0, 1] at overhead resolution, None if unchanged
    saliency_method: str = ""
    replanned: bool = False


def normalize_map(m: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    m = m.astype(np.float32)
    m = cv2.resize(m, size_wh, interpolation=cv2.INTER_CUBIC)
    m -= m.min()
    peak = m.max()
    return m / peak if peak > 1e-8 else np.zeros_like(m)


@dataclass
class ChunkVarianceTracker:
    """Variance across every predicted chunk that covers the current control step.

    Requires overlapping chunks: ``n_action_steps < chunk_size`` or observe-only passes between replans.
    """

    max_chunks: int = 8
    horizon: int | None = None  # ignore predictions made more than this many steps ago (far-ahead guesses always disagree)
    chunks: deque = field(default_factory=lambda: deque(maxlen=8))

    def add(self, start_step: int, chunk: np.ndarray) -> None:
        self.chunks.append((start_step, chunk))

    def variance(self, step: int) -> float:
        preds = [c[step - s] for s, c in self.chunks if 0 <= step - s < min(len(c), self.horizon or len(c))]
        if len(preds) < 2:
            return 0.0
        return float(np.var(np.stack(preds), axis=0)[:5].mean())  # body joints only

    def reset(self) -> None:
        self.chunks.clear()


class ACTRunner:
    def __init__(
        self,
        policy_path: str,
        device: str,
        overhead_key: str,
        overhead_size_wh: tuple[int, int],
        n_action_steps: int,
        task: str,
        observe_every: int = 0,
    ) -> None:
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.torch = torch
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(policy_path)
        self.policy.to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.task = task
        self.overhead_key = f"observation.images.{overhead_key}"
        self.overhead_size_wh = overhead_size_wh
        self.image_keys = list(self.policy.config.image_features)
        if self.overhead_key not in self.image_keys:
            raise ValueError(f"{self.overhead_key} not in policy image features {self.image_keys}")
        self.input_shapes = {k: tuple(f.shape) for k, f in self.policy.config.image_features.items()}
        self.chunk_size = self.policy.config.chunk_size
        self.n_action_steps = min(n_action_steps, self.chunk_size)
        self.n_1d_tokens = 1 + int(self.policy.config.robot_state_feature is not None) + int(
            self.policy.config.env_state_feature is not None
        )

        self._feature_maps: list = []
        self._cross_attn = None
        model = self.policy.model
        model.register_forward_pre_hook(lambda *_: self._feature_maps.clear())
        model.backbone.register_forward_hook(lambda _m, _i, out: self._feature_maps.append(out["feature_map"].detach()))
        model.decoder.layers[-1].multihead_attn.register_forward_hook(self._capture_attn)

        # Observe passes only make sense between replans.
        self.observe_every = observe_every if 0 < observe_every < self.n_action_steps else 0
        self.variance = ChunkVarianceTracker(horizon=2 * (self.observe_every or self.n_action_steps))
        self.queue: deque[np.ndarray] = deque()
        self.step_idx = 0
        self._infer_lock = threading.Lock()  # the hooks share state: one forward pass at a time
        self._last_images: dict[str, np.ndarray] = {}
        log.info(
            "Loaded ACT policy from %s (chunk=%d, executing %d, observing every %s)",
            policy_path, self.chunk_size, self.n_action_steps, self.observe_every or "replan",
        )

    def _capture_attn(self, _module, _inputs, output) -> None:
        weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        self._cross_attn = weights.detach() if weights is not None else None

    def reset(self) -> None:
        self.queue.clear()
        self.variance.reset()
        self.step_idx = 0

    def _observation(self, state: np.ndarray, images: dict[str, np.ndarray]) -> dict:
        from lerobot.policies.utils import prepare_observation_for_inference

        obs = {"observation.state": state.astype(np.float32)}
        for key in self.image_keys:
            cam = key.removeprefix("observation.images.")
            _, h, w = self.input_shapes[key]
            img = images[cam]
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            obs[key] = img
        return prepare_observation_for_inference(obs, self.device, self.task, "so_follower")

    def _saliency(self) -> tuple[np.ndarray | None, str]:
        cam_idx = self.image_keys.index(self.overhead_key)
        if cam_idx >= len(self._feature_maps):
            return None, ""
        fmap = self._feature_maps[cam_idx][0]  # (C, h, w)
        h, w = fmap.shape[-2:]
        if self._cross_attn is not None:
            offset = self.n_1d_tokens + sum(
                self._feature_maps[i].shape[-2] * self._feature_maps[i].shape[-1] for i in range(cam_idx)
            )
            attn = self._cross_attn[0, :, offset : offset + h * w]  # (chunk_size, h*w)
            if attn.shape[-1] == h * w:
                m = attn.mean(0).reshape(h, w).float().cpu().numpy()
                return normalize_map(m, self.overhead_size_wh), "act_cross_attention"
        m = fmap.float().norm(dim=0).cpu().numpy()
        return normalize_map(m, self.overhead_size_wh), "resnet_layer4_activation"

    def _forward(self, state: np.ndarray, images: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray | None, str]:
        """One forward pass: ``(chunk, saliency, method)``. Hold ``_infer_lock``."""
        batch = self.preprocessor(self._observation(state, images))
        with self.torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim), normalised
            chunk = self.postprocessor(chunk)
        return chunk[0].float().cpu().numpy(), *self._saliency()

    def attend(self, state: np.ndarray, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, str, np.ndarray]:
        """Observe only: saliency and predicted chunk for ``frame_bgr`` as the overhead view at joint state
        ``state``. Other cameras get their latest live frames (black if the policy hasn't run yet)."""
        cam = self.overhead_key.removeprefix("observation.images.")
        images = dict(self._last_images)
        images[cam] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for key in self.image_keys:
            other = key.removeprefix("observation.images.")
            if other not in images:
                _, h, w = self.input_shapes[key]
                images[other] = np.zeros((h, w, 3), np.uint8)
        with self._infer_lock:
            chunk, saliency, method = self._forward(np.asarray(state, np.float32), images)
        return saliency, method, chunk

    def step(self, state: np.ndarray, images: dict[str, np.ndarray]) -> PolicyStep:
        saliency, method, replanned = None, "", False
        self._last_images = images
        if not self.queue:
            with self._infer_lock:
                chunk, saliency, method = self._forward(state, images)
            self.variance.add(self.step_idx, chunk)
            self.queue.extend(chunk[: self.n_action_steps])
            replanned = True
        elif self.observe_every and self.step_idx % self.observe_every == 0:
            with self._infer_lock:  # observe only: the executing chunk is left untouched
                chunk, saliency, method = self._forward(state, images)
            self.variance.add(self.step_idx, chunk)
        action = self.queue.popleft()
        var = self.variance.variance(self.step_idx)
        self.step_idx += 1
        return PolicyStep(action=action, chunk_variance=var, saliency=saliency, saliency_method=method, replanned=replanned)
