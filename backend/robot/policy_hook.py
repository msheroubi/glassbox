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

``MockPolicy`` is the stand-in for ``--mock-hardware``: an image-driven pick-and-place controller
whose "attention" is a colour-saliency map of the frame it is fed. Because it only sees pixels,
mask suppression changes its behaviour for the same causal reason it changes a real ACT policy's.
"""

from __future__ import annotations

import logging
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

    Requires ``n_action_steps < chunk_size`` so that consecutive chunks overlap.
    """

    max_chunks: int = 8
    chunks: deque = field(default_factory=lambda: deque(maxlen=8))

    def add(self, start_step: int, chunk: np.ndarray) -> None:
        self.chunks.append((start_step, chunk))

    def variance(self, step: int) -> float:
        preds = [c[step - s] for s, c in self.chunks if 0 <= step - s < len(c)]
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

        self.variance = ChunkVarianceTracker()
        self.queue: deque[np.ndarray] = deque()
        self.step_idx = 0
        log.info("Loaded ACT policy from %s (chunk=%d, executing %d)", policy_path, self.chunk_size, self.n_action_steps)

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

    def step(self, state: np.ndarray, images: dict[str, np.ndarray]) -> PolicyStep:
        saliency, method, replanned = None, "", False
        if not self.queue:
            batch = self.preprocessor(self._observation(state, images))
            with self.torch.inference_mode():
                chunk = self.policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim), normalised
                chunk = self.postprocessor(chunk)
            chunk = chunk[0].float().cpu().numpy()
            self.variance.add(self.step_idx, chunk)
            self.queue.extend(chunk[: self.n_action_steps])
            saliency, method = self._saliency()
            replanned = True
        action = self.queue.popleft()
        var = self.variance.variance(self.step_idx)
        self.step_idx += 1
        return PolicyStep(action=action, chunk_variance=var, saliency=saliency, saliency_method=method, replanned=replanned)


def red_saliency(frame_bgr: np.ndarray) -> np.ndarray:
    """Colour-opponent saliency for saturated red, the mock policy's learned visual cue."""
    f = frame_bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    s = np.clip(r - np.maximum(g, b) - 40.0, 0, None) / 160.0
    s = cv2.GaussianBlur(s, (0, 0), 6)
    return (s / max(float(s.max()), 0.5)).astype(np.float32)  # absolute floor: no red => no saliency


class MockPolicy:
    """Image-conditioned pick-and-place controller standing in for ACT when no checkpoint exists.

    Every ``n_action_steps`` it re-plans: it samples target hypotheses from the red-saliency blobs of
    the frame it receives (ACT's attention behaves similarly when two objects match the cue), plans
    a short Cartesian chunk toward each, and executes the hypothesis mean. Disagreement between
    hypotheses is reported as chunk variance.
    """

    PHASES = ("REACH", "GRASP", "LIFT", "CARRY", "RELEASE", "DONE")

    def __init__(self, kinematics, homography, n_action_steps: int = 10, seed: int = 0) -> None:
        self.kin = kinematics
        self.hom = homography
        self.n_action_steps = n_action_steps
        self.rng = np.random.default_rng(seed)
        self.grasp_offset_mm = np.zeros(2)
        self.reset()

    def reset(self) -> None:
        self.phase = "REACH"
        self.phase_t = 0
        self.queue: deque[np.ndarray] = deque()
        self.hyp_variance = 0.0
        self.grasp_offset_mm = np.zeros(2)
        self.memory: tuple[np.ndarray, float] | None = None  # last clearly seen goal (xy, saliency mass)

    def _blobs(self, sal: np.ndarray) -> list[tuple[np.ndarray, float]]:
        mask = (sal > 0.35).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
        out = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 150:
                continue
            weight = float(sal[labels == i].sum())
            out.append((np.array(self.hom.pixel_to_workspace(*cents[i])), weight))
        return out

    def _plan_chunk(self, start_xy: np.ndarray, goal_xy: np.ndarray, length: int) -> np.ndarray:
        steps = []
        for k in range(1, length + 1):
            a = min(1.0, k * 6.0 / max(np.linalg.norm(goal_xy - start_xy), 1e-6))
            steps.append(start_xy + (goal_xy - start_xy) * a)
        return np.array(steps)

    def step(self, state: np.ndarray, frame_bgr: np.ndarray, world) -> PolicyStep:
        """``world`` exposes ``bin_xy_mm`` and ``holding`` (proprioceptive grasp signal)."""
        sal = red_saliency(frame_bgr)
        gripper_xy = np.array(self.kin.fk(state))
        replanned = False
        if not self.queue:
            replanned = True
            self.phase_t += 1
            goal, hyps = self._goal(sal, gripper_xy, world)
            horizon = self.n_action_steps * 3
            chunks = [self._plan_chunk(gripper_xy, h, horizon) for h in hyps]
            xy_chunk = np.mean(chunks, axis=0) if chunks else self._plan_chunk(gripper_xy, goal, horizon)
            chunk = np.array([self._joints(xy) for xy in xy_chunk])[: self.n_action_steps]
            # Ramp wrist and gripper to the phase posture across the chunk (demonstrations are smooth).
            ramp = np.minimum(1.0, np.arange(1, len(chunk) + 1) / len(chunk))[:, None]
            chunk[:, 3:] = state[3:] + (chunk[:, 3:] - state[3:]) * ramp
            self.queue.extend(chunk)
            # Variance across hypotheses' joint trajectories stands in for ensemble disagreement.
            self.hyp_variance = 0.0
            if len(chunks) > 1:
                per_hyp = np.array([[self.kin.ik(*xy) for xy in c] for c in chunks])
                self.hyp_variance = float(per_hyp.var(axis=0).mean())
        action = self.queue.popleft()
        return PolicyStep(
            action=action,
            chunk_variance=self.hyp_variance,
            saliency=sal,
            saliency_method="mock_colour_saliency",
            replanned=replanned,
        )

    def _goal(self, sal, gripper_xy, world) -> tuple[np.ndarray, list[np.ndarray]]:
        if self.phase == "REACH":
            blobs = self._blobs(sal)
            mass = sum(w for _, w in blobs)
            # Near the goal the gripper hides the object from the overhead camera; like ACT's action chunk,
            # keep heading for where it was last seen clearly instead of chasing the uncovered sliver.
            if self.memory is not None and np.linalg.norm(self.memory[0] - gripper_xy) < 20.0 and mass < 0.3 * self.memory[1]:
                goal = self.memory[0] + self.grasp_offset_mm
                if np.linalg.norm(goal - gripper_xy) < 8.0:
                    self.phase, self.phase_t = "GRASP", 0
                return goal, [goal]
            if not blobs:
                return gripper_xy, []
            weights = np.array([w for _, w in blobs])
            idx = self.rng.choice(len(blobs), size=6, p=weights / weights.sum())
            seen = [blobs[i][0] + self.rng.normal(0, 2.0, 2) for i in idx]
            if len({int(i) for i in idx}) == 1:
                self.memory = (np.mean(seen, axis=0), mass)
            hyps = [h + self.grasp_offset_mm for h in seen]
            goal = np.mean(hyps, axis=0)
            if np.linalg.norm(goal - gripper_xy) < 8.0 and len({int(i) for i in idx}) == 1:
                self.phase, self.phase_t = "GRASP", 0
            return goal, hyps
        if self.phase == "GRASP":
            if self.phase_t >= 3:
                self.phase, self.phase_t = ("LIFT" if world.holding else "REACH"), 0
            return gripper_xy, []
        if self.phase == "LIFT":
            if self.phase_t >= 2:
                self.phase, self.phase_t = "CARRY", 0
            return gripper_xy, []
        if self.phase == "CARRY":
            goal = np.array(world.bin_xy_mm)
            if np.linalg.norm(goal - gripper_xy) < 10.0:
                self.phase, self.phase_t = "RELEASE", 0
            return goal, []
        if self.phase == "RELEASE":
            if self.phase_t >= 3:
                self.phase = "DONE"
            return gripper_xy, []
        return gripper_xy, []

    def _joints(self, xy: np.ndarray) -> np.ndarray:
        pan, lift, elbow = self.kin.ik(*xy)
        wrist = {"GRASP": 35.0, "LIFT": -10.0, "RELEASE": 20.0}.get(self.phase, 0.0)
        gripper = {"GRASP": 2.0, "LIFT": 2.0, "CARRY": 2.0}.get(self.phase, 60.0)
        return np.array([pan, lift, elbow, wrist, 0.0, gripper])


class CameraObserver:
    """Camera-only stand-in for a policy (``--webcam``): holds the arm still and exposes a
    bottom-up visual-saliency map (spectral residual, Hou & Zhang 2007) of the overhead frame.

    There is no learned policy in this mode, so the map shows what is visually conspicuous rather than
    what a policy attends to. SAM 3.1 attribution and Muse Spark explanations still run on top of it.
    """

    METHOD = "spectral-residual (camera-only, no policy)"

    def __init__(self, camera_key: str, every_n: int = 3) -> None:
        self.camera_key = camera_key
        self.every_n = every_n
        self.reset()

    def reset(self) -> None:
        self.queue: deque[np.ndarray] = deque()
        self._n = 0
        self._ema: np.ndarray | None = None

    def _spectral_residual(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(cv2.resize(rgb, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        spec = np.fft.fft2(gray.astype(np.float32))
        log_amp = np.log1p(np.abs(spec))
        residual = log_amp - cv2.blur(log_amp, (3, 3))
        sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spec)))) ** 2
        return cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 2.5)

    def step(self, state: np.ndarray, images: dict[str, np.ndarray]) -> PolicyStep:
        self._n += 1
        rgb = images.get(self.camera_key)
        saliency = None
        if rgb is not None and self._n % self.every_n == 1:
            sal = self._spectral_residual(rgb)
            sal /= max(float(sal.max()), 1e-8)
            self._ema = sal if self._ema is None else 0.6 * self._ema + 0.4 * sal  # damp webcam flicker
            saliency = cv2.resize(self._ema, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return PolicyStep(action=np.asarray(state, dtype=np.float64).copy(), chunk_variance=0.0,
                          saliency=saliency, saliency_method=self.METHOD)
