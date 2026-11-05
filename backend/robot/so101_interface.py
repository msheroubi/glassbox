"""SO-101 hardware: the LeRobot follower arm with the overhead (and optional wrist) camera.

The control loop uses ``connect() / disconnect() / read() -> Observation / send(qpos_target) / hold() /
estop()``. ``PlanarKinematics`` is the top-down approximation used to place the arm in the table frame.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.config import JOINT_NAMES, Settings

log = logging.getLogger(__name__)

STS3215_STEPS_PER_REV = 4096


@dataclass
class Observation:
    timestamp: float
    qpos: np.ndarray  # (6,) degrees, gripper 0-100
    qvel: np.ndarray  # (6,) deg/s
    load: np.ndarray  # (6,) percent of rated load, signed
    overhead_bgr: np.ndarray  # (H, W, 3) uint8
    images_rgb: dict[str, np.ndarray] = field(default_factory=dict)  # camera key -> RGB, for the policy


class RealSO101:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.robot = None
        self._last_qpos = np.zeros(6)
        self._lock = threading.Lock()

    def connect(self) -> None:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        s = self.s
        cameras = {
            s.overhead_camera_key: OpenCVCameraConfig(
                index_or_path=s.overhead_camera_index, fps=s.camera_fps, width=s.overhead_width, height=s.overhead_height
            )
        }
        if s.wrist_camera_key and s.wrist_camera_index is not None:
            cameras[s.wrist_camera_key] = OpenCVCameraConfig(
                index_or_path=s.wrist_camera_index, fps=s.camera_fps, width=s.wrist_width, height=s.wrist_height
            )
        cfg = SO101FollowerConfig(
            port=s.robot_port,
            id=s.robot_id,
            cameras=cameras,
            use_degrees=True,
            max_relative_target=_relative_caps(s),
        )
        self.robot = SO101Follower(cfg)
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            log.warning("SO-101 '%s' has no matching calibration; run lerobot-calibrate first", s.robot_id)
        log.info("Connected SO-101 on %s with cameras %s", s.robot_port, list(cameras))

    def calibrated_limits(self) -> dict[str, tuple[float, float]] | None:
        """Each joint's range from the arm's calibration file, in the units actions use.

        lerobot-calibrate records how far each joint actually travels and writes it to the servos' position
        limits, so this is the range the motors will reach anyway (and what the training demos were recorded in)."""
        from lerobot.motors import MotorNormMode

        bus = self.robot.bus
        if not self.robot.is_calibrated:
            return None
        limits = {}
        for m in JOINT_NAMES:
            cal, mode = bus.calibration[m], bus.motors[m].norm_mode
            if mode is MotorNormMode.DEGREES:  # lerobot: (raw - mid) * 360 / max_res
                half = (cal.range_max - cal.range_min) / 2 * 360 / (bus.model_resolution_table[bus.motors[m].model] - 1)
                limits[m] = (-half, half)
            elif mode is MotorNormMode.RANGE_0_100:
                limits[m] = (0.0, 100.0)
            else:
                limits[m] = (-100.0, 100.0)
        return limits

    def disconnect(self) -> None:
        if self.robot is not None and self.robot.is_connected:
            self.robot.disconnect()

    def read(self) -> Observation:
        with self._lock:
            obs = self.robot.get_observation()
            load = self.robot.bus.sync_read("Present_Load", normalize=False)
            vel = self.robot.bus.sync_read("Present_Velocity", normalize=False)
        qpos = np.array([obs[f"{m}.pos"] for m in JOINT_NAMES], dtype=np.float64)
        self._last_qpos = qpos
        images = {k: obs[k] for k in self.robot.cameras}
        return Observation(
            timestamp=time.time(),
            qpos=qpos,
            # Present_Load is 0.1 % of rated torque; Present_Velocity is encoder steps/s.
            load=np.array([load[m] / 10.0 for m in JOINT_NAMES]),
            qvel=np.array([vel[m] * 360.0 / STS3215_STEPS_PER_REV for m in JOINT_NAMES]),
            overhead_bgr=cv2.cvtColor(images[self.s.overhead_camera_key], cv2.COLOR_RGB2BGR),
            images_rgb=images,
        )

    def send(self, qpos_target: np.ndarray) -> np.ndarray:
        with self._lock:
            sent = self.robot.send_action({f"{m}.pos": float(v) for m, v in zip(JOINT_NAMES, qpos_target)})
        return np.array([sent[f"{m}.pos"] for m in JOINT_NAMES])

    def hold(self) -> None:
        self.send(self._last_qpos)

    def estop(self) -> None:
        with self._lock:
            self.robot.bus.disable_torque()
        log.warning("E-STOP: torque disabled on all SO-101 servos")

    def resume_torque(self) -> None:
        with self._lock:
            self.robot.bus.enable_torque()


def _relative_caps(s: Settings) -> dict[str, float] | None:
    """LeRobot's max_relative_target per motor; an uncapped joint gets an unreachable cap (LeRobot needs every key)."""
    if s.max_relative_target_deg is None and s.max_relative_target_gripper is None:
        return None
    caps = {m: s.max_relative_target_deg for m in JOINT_NAMES}
    caps["gripper"] = s.max_relative_target_gripper
    return {m: 1e6 if c is None else float(c) for m, c in caps.items()}


class PlanarKinematics:
    """Top-down SO-101 approximation: pan about the base, two-link reach in the vertical plane."""

    L1, L2 = 170.0, 170.0  # upper arm, forearm+wrist (mm)
    DROP = 90.0  # shoulder height above the grasp plane (mm)
    R_MIN, R_MAX = 60.0, 325.0

    def ik(self, x: float, y: float) -> tuple[float, float, float]:
        pan = math.degrees(math.atan2(y, x))
        r = float(np.clip(math.hypot(x, y), self.R_MIN, self.R_MAX))
        d = math.hypot(r, self.DROP)
        cos_e = np.clip((d * d - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2), -1.0, 1.0)
        t2 = math.acos(cos_e)
        alpha = math.atan2(r, self.DROP)
        t1 = alpha - math.atan2(self.L2 * math.sin(t2), self.L1 + self.L2 * math.cos(t2))
        return pan, math.degrees(t1) - 45.0, math.degrees(t2) - 90.0

    def fk(self, q: np.ndarray) -> tuple[float, float]:
        pan, lift, elbow = q[0], q[1], q[2]
        t2 = math.radians(elbow + 90.0)
        t1 = math.radians(lift + 45.0)
        alpha = t1 + math.atan2(self.L2 * math.sin(t2), self.L1 + self.L2 * math.cos(t2))
        d = math.sqrt(self.L1**2 + self.L2**2 + 2 * self.L1 * self.L2 * math.cos(t2))
        r = d * math.sin(alpha)
        return r * math.cos(math.radians(pan)), r * math.sin(math.radians(pan))
