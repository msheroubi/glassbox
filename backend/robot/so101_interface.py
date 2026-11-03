"""SO-101 hardware abstraction: the real LeRobot follower arm and a physics-lite mock for demos.

Both expose the same small surface used by the control loop:

    connect() / disconnect() / read() -> Observation / send(qpos_target) / hold() / estop()

``MockSO101`` renders a synthetic overhead view of the tabletop, simulates joint tracking, collisions
with obstacles (which stall the servos and spike ``torque_load``), grasping and placing, and replays
one of several repeatable failure scenarios. It also exposes ground-truth object masks so the XAI
pipeline can run fully offline when no Meta API key is configured.
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
from backend.robot.calibration import Homography

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


@dataclass
class GroundTruthObject:
    label: str
    prompt: str
    mask: np.ndarray  # (H, W) bool


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
            max_relative_target=s.max_relative_target_deg,
        )
        self.robot = SO101Follower(cfg)
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            log.warning("SO-101 '%s' has no matching calibration; run lerobot-calibrate first", s.robot_id)
        log.info("Connected SO-101 on %s with cameras %s", s.robot_port, list(cameras))

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


class WebcamSource:
    """Camera-only mode: a USB webcam stands in for the overhead camera and no arm is connected.

    Frames are grabbed on a background thread and ``read()`` returns the newest one, so a slow or
    unplugged camera never stalls the control loop (a "no signal" frame is shown instead). Joint state
    is all zeros and motion commands are ignored.
    """

    STALE_S = 2.0

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.W, self.H = settings.overhead_width, settings.overhead_height
        self.device = settings.overhead_camera_index
        self.cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._frame_t = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.native_size: tuple[int, int] | None = None

    def connect(self) -> None:
        dev = int(self.device) if str(self.device).isdigit() else self.device
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2) if isinstance(dev, int) else cv2.VideoCapture(dev)
        if not cap.isOpened():
            cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam {self.device!r}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        cap.set(cv2.CAP_PROP_FPS, self.s.camera_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.native_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._thread = threading.Thread(target=self._grab, name="glassbox-webcam", daemon=True)
        self._thread.start()
        log.info("Webcam %r opened at %sx%s (stream %sx%s)", self.device, *self.native_size, self.W, self.H)

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        """Centre-crop to the stream aspect ratio, then resize to the stream size."""
        h, w = frame.shape[:2]
        target = self.W / self.H
        if w / h > target:
            cw = int(round(h * target))
            frame = frame[:, (w - cw) // 2 : (w - cw) // 2 + cw]
        elif w / h < target:
            ch = int(round(w / target))
            frame = frame[(h - ch) // 2 : (h - ch) // 2 + ch]
        if frame.shape[1] != self.W or frame.shape[0] != self.H:
            frame = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return frame

    def _grab(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            frame = self._fit(frame)
            with self._lock:
                self._frame, self._frame_t = frame, time.time()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)
        if self.cap is not None:
            self.cap.release()

    def read(self) -> Observation:
        time.sleep(0.005)  # the grabber thread paces the stream; don't spin on an unchanged frame
        with self._lock:
            frame, t = self._frame, self._frame_t
        now = time.time()
        if frame is None or now - t > self.STALE_S:
            frame = np.zeros((self.H, self.W, 3), np.uint8)
            cv2.putText(frame, f"no signal from webcam {self.device}", (30, self.H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (80, 80, 255), 2, cv2.LINE_AA)
        zeros = np.zeros(6)
        return Observation(
            timestamp=now, qpos=zeros, qvel=zeros.copy(), load=zeros.copy(), overhead_bgr=frame,
            images_rgb={self.s.overhead_camera_key: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)},
        )

    def send(self, qpos_target: np.ndarray) -> np.ndarray:
        return np.asarray(qpos_target, dtype=np.float64)

    def hold(self) -> None:
        pass

    def estop(self) -> None:
        log.warning("E-STOP pressed in camera-only mode (no arm connected)")

    def resume_torque(self) -> None:
        pass


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


SCENARIOS = {
    "nominal": "Clean table: the policy picks the red cube and drops it in the bin.",
    "attention_hijack": "A large red ball shares the policy's colour cue and hijacks its attention.",
    "servo_stall": "A cardboard box sits on the straight-line reach path; the gripper jams against it.",
    "occlusion": "An operator's hand reaches over and occludes the cube mid-reach.",
    "specular_glare": "A specular highlight washes out most of the cube, biasing the grasp point.",
}

HOME_Q = np.array([0.0, *PlanarKinematics().ik(120.0, 0.0)[1:], 0.0, 0.0, 60.0])


class MockSO101:
    """Deterministic simulated SO-101 + overhead camera for ``--mock-hardware`` demos."""

    CUBE_MM = 50.0
    GRASP_TOL_MM = 10.0

    def __init__(self, settings: Settings, homography: Homography, scenario: str = "attention_hijack") -> None:
        self.s = settings
        self.hom = homography
        self.kin = PlanarKinematics()
        self.W, self.H = settings.overhead_width, settings.overhead_height
        self.px_per_mm = 1.0 / np.linalg.norm(
            np.subtract(self.hom.pixel_to_workspace(0, 0), self.hom.pixel_to_workspace(1, 0))
        )
        self._lock = threading.Lock()
        self._table = self._render_table()
        self.set_scenario(scenario)

    # ------------------------------------------------------------------ scenario state
    def set_scenario(self, scenario: str) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")
        with self._lock:
            self.scenario = scenario
            self.t0 = time.time()
            self.q = HOME_Q.copy()
            self.q_target = HOME_Q.copy()
            self.qvel = np.zeros(6)
            self.load = np.zeros(6)
            self.cube_xy = np.array([240.0, -60.0])
            self.bin_xy_mm = (150.0, 215.0)
            self.holding = False
            self.delivered = False
            self.torque_enabled = True
            self.blocked = False
            self.operator_cleared = False
            self.ball_xy = np.array([228.0, 112.0]) if scenario == "attention_hijack" else None
            self.box = (158.0, 202.0, -46.0, -8.0) if scenario == "servo_stall" else None  # x0, x1, y0, y1
            self._last = time.time()

    def clear_operator_interference(self) -> None:
        """Operator acknowledges the occlusion and withdraws their hand."""
        self.operator_cleared = True

    def _hand_xy(self, t: float) -> np.ndarray | None:
        if self.scenario != "occlusion":
            return None
        if self.operator_cleared:
            return None
        start, rest = np.array([250.0, -420.0]), self.cube_xy + np.array([0.0, -12.0])
        if t < 0.15:
            return None
        a = min(1.0, (t - 0.15) / 0.5)
        return start + (rest - start) * (a * a * (3 - 2 * a))

    # ------------------------------------------------------------------ hardware surface
    def connect(self) -> None:
        log.info("Mock SO-101 connected (scenario=%s)", self.scenario)

    def disconnect(self) -> None:
        pass

    def send(self, qpos_target: np.ndarray) -> np.ndarray:
        with self._lock:
            self.q_target = np.asarray(qpos_target, dtype=np.float64).copy()
            return self.q_target.copy()

    def hold(self) -> None:
        with self._lock:
            self.q_target = self.q.copy()

    def estop(self) -> None:
        with self._lock:
            self.torque_enabled = False
            self.q_target = self.q.copy()
        log.warning("MOCK E-STOP: torque disabled")

    def resume_torque(self) -> None:
        with self._lock:
            self.torque_enabled = True

    def read(self) -> Observation:
        with self._lock:
            now = time.time()
            self._physics(min(now - self._last, 0.1))
            self._last = now
            frame = self._render(now - self.t0)
            return Observation(
                timestamp=now,
                qpos=self.q.copy(),
                qvel=self.qvel.copy(),
                load=self.load.copy(),
                overhead_bgr=frame,
                images_rgb={self.s.overhead_camera_key: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)},
            )

    # ------------------------------------------------------------------ physics
    def gripper_xy(self, q: np.ndarray | None = None) -> np.ndarray:
        return np.array(self.kin.fk(self.q if q is None else q))

    def _in_box(self, xy: np.ndarray, margin: float = 12.0) -> bool:
        if self.box is None:
            return False
        x0, x1, y0, y1 = self.box
        return x0 - margin <= xy[0] <= x1 + margin and y0 - margin <= xy[1] <= y1 + margin

    def _physics(self, dt: float) -> None:
        if not self.torque_enabled:
            self.qvel[:] = 0
            self.load[:] = 0
            return
        prev = self.q.copy()
        err = self.q_target - self.q
        step = np.clip(err * (1 - math.exp(-dt / 0.08)), -150 * dt, 150 * dt)
        candidate = self.q + step
        # Sweep the step so a late tick (dt up to 0.1 s, ~50 mm of travel) can't tunnel through the box.
        self.blocked = not self._in_box(self.gripper_xy()) and any(
            self._in_box(self.gripper_xy(self.q + step * f)) for f in (0.25, 0.5, 0.75, 1.0)
        )
        if self.blocked:
            candidate[:3] = self.q[:3]  # gripper jammed against the obstacle: arm joints stall
        self.q = candidate
        self.qvel = (self.q - prev) / max(dt, 1e-3)
        following = np.abs(self.q_target - self.q)
        noise = np.random.default_rng().normal(0, 0.6, 6)
        self.load = np.sign(err) * np.clip(4.0 + 3.2 * following, 0, 100) + noise
        if self.holding:
            self.load[5] += 28.0

        gxy = self.gripper_xy()
        closing = self.q[5] < 10.0 and prev[5] >= 10.0
        if closing and self.q[3] > 25.0 and not self.holding:
            if np.linalg.norm(gxy - self.cube_xy) < self.GRASP_TOL_MM:
                self.holding = True
        if self.holding:
            self.cube_xy = gxy.copy()
            if self.q[5] > 40.0:
                self.holding = False
                bx, by = self.bin_xy_mm
                self.delivered = abs(gxy[0] - bx) < 55 and abs(gxy[1] - by) < 70

    # ------------------------------------------------------------------ rendering
    def _poly_px(self, pts_mm) -> np.ndarray:
        return np.array([self.hom.workspace_to_pixel(x, y) for x, y in pts_mm], dtype=np.int32)

    def _render_table(self) -> np.ndarray:
        rng = np.random.default_rng(7)
        base = np.full((self.H, self.W, 3), self.s.tabletop_bgr, np.float32)
        grain = cv2.GaussianBlur(rng.normal(0, 9, (self.H, self.W)).astype(np.float32), (0, 0), sigmaX=18, sigmaY=2)
        base += grain[..., None]
        yy = np.linspace(-1, 1, self.H)[:, None]
        xx = np.linspace(-1, 1, self.W)[None, :]
        base *= (1.0 - 0.12 * (xx**2 + yy**2))[..., None]
        return np.clip(base, 0, 255).astype(np.uint8)

    def background_plate(self) -> np.ndarray:
        return self._table.copy()

    def _rect_mm(self, cx, cy, half) -> list[tuple[float, float]]:
        return [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]

    def _draw_objects(self, img: np.ndarray, t: float) -> None:
        # bin
        bx, by = self.bin_xy_mm
        bin_poly = self._poly_px([(bx - 55, by - 70), (bx + 55, by - 70), (bx + 55, by + 70), (bx - 55, by + 70)])
        cv2.fillPoly(img, [bin_poly], (150, 95, 40))
        cv2.polylines(img, [bin_poly], True, (90, 55, 20), 6)
        # obstacle box
        if self.box is not None:
            x0, x1, y0, y1 = self.box
            poly = self._poly_px([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            cv2.fillPoly(img, [poly], (70, 125, 170))
            cv2.polylines(img, [poly], True, (40, 80, 120), 4)
            cv2.line(img, tuple(poly[0]), tuple(poly[2]), (55, 100, 140), 2)
        # distractor ball
        if self.ball_xy is not None:
            c = tuple(int(v) for v in self.hom.workspace_to_pixel(*self.ball_xy))
            r = int(34 * self.px_per_mm)
            cv2.circle(img, c, r, (30, 25, 215), -1, cv2.LINE_AA)
            cv2.circle(img, (c[0] - r // 3, c[1] - r // 3), r // 4, (120, 120, 255), -1, cv2.LINE_AA)
        # target cube (drawn after bin so it is visible once delivered)
        cube = self._poly_px(self._rect_mm(*self.cube_xy, self.CUBE_MM / 2))
        cv2.fillPoly(img, [cube], (35, 35, 200))
        cv2.polylines(img, [cube], True, (20, 20, 140), 2, cv2.LINE_AA)
        if self.scenario == "specular_glare" and not self.holding and not self.delivered:
            gx, gy = self.cube_xy
            glare = self._poly_px([(gx - 25, gy - 25), (gx + 25, gy - 25), (gx + 25, gy + 10), (gx - 25, gy + 10)])
            overlay = img.copy()
            cv2.fillPoly(overlay, [glare], (250, 250, 250))
            img[:] = cv2.addWeighted(overlay, 0.92, img, 0.08, 0)

    def _draw_arm(self, img: np.ndarray) -> None:
        base = np.array(self.hom.workspace_to_pixel(0, 0))
        gxy = self.gripper_xy()
        g = np.array(self.hom.workspace_to_pixel(*gxy))
        elbow = base + (g - base) * 0.48
        shade = (45, 45, 45) if self.torque_enabled else (60, 60, 140)
        cv2.circle(img, tuple(base.astype(int)), int(48 * self.px_per_mm), (40, 40, 40), -1, cv2.LINE_AA)
        cv2.line(img, tuple(base.astype(int)), tuple(elbow.astype(int)), shade, int(34 * self.px_per_mm), cv2.LINE_AA)
        cv2.line(img, tuple(elbow.astype(int)), tuple(g.astype(int)), (70, 70, 70), int(26 * self.px_per_mm), cv2.LINE_AA)
        cv2.circle(img, tuple(elbow.astype(int)), int(20 * self.px_per_mm), (25, 25, 25), -1, cv2.LINE_AA)
        # gripper fingers, perpendicular to the arm, spread with the gripper opening
        d = (g - base) / (np.linalg.norm(g - base) + 1e-6)
        perp = np.array([-d[1], d[0]])
        spread = (8 + 0.35 * self.q[5]) * self.px_per_mm
        tip_scale = 1.0 - 0.15 * max(0.0, self.q[3]) / 40.0  # wrist flex: tilting down looks shorter
        for sgn in (-1, 1):
            p0 = g + perp * spread * sgn
            p1 = p0 + d * 30 * self.px_per_mm * tip_scale
            cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), (20, 20, 20), int(9 * self.px_per_mm), cv2.LINE_AA)
        cv2.circle(img, tuple(g.astype(int)), int(16 * self.px_per_mm), (90, 90, 90), -1, cv2.LINE_AA)

    def _draw_hand(self, img: np.ndarray, hxy: np.ndarray) -> None:
        skin = (120, 160, 215)
        palm = np.array(self.hom.workspace_to_pixel(*hxy)).astype(int)
        wrist = np.array(self.hom.workspace_to_pixel(hxy[0], hxy[1] - 260)).astype(int)
        cv2.line(img, tuple(wrist), tuple(palm), skin, int(80 * self.px_per_mm), cv2.LINE_AA)
        cv2.ellipse(img, tuple(palm), (int(52 * self.px_per_mm), int(46 * self.px_per_mm)), 0, 0, 360, skin, -1, cv2.LINE_AA)
        for ang in (-40, -14, 12, 38):  # fingers point toward image-left, away from the wrist
            a = math.radians(ang)
            tip = palm + (np.array([-math.cos(a), math.sin(a)]) * 95 * self.px_per_mm).astype(int)
            cv2.line(img, tuple(palm), tuple(tip), skin, int(19 * self.px_per_mm), cv2.LINE_AA)

    def _render(self, t: float) -> np.ndarray:
        img = self._table.copy()
        self._draw_objects(img, t)
        self._draw_arm(img)
        hxy = self._hand_xy(t)
        if hxy is not None:
            self._draw_hand(img, hxy)
        noise = np.random.default_rng().integers(-3, 4, img.shape, dtype=np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ ground truth for offline SAM
    def ground_truth_objects(self) -> list[GroundTruthObject]:
        concepts = {c.label: c.prompt for c in self.s.concepts}
        t = time.time() - self.t0
        objs: list[GroundTruthObject] = []

        def blank():
            return np.zeros((self.H, self.W), np.uint8)

        with self._lock:
            m = blank()
            g = np.array(self.hom.workspace_to_pixel(*self.gripper_xy())).astype(int)
            cv2.circle(m, tuple(g), int(42 * self.px_per_mm), 1, -1)
            objs.append(GroundTruthObject("gripper", concepts.get("gripper", "gripper"), m.astype(bool)))
            m = blank()
            cv2.fillPoly(m, [self._poly_px(self._rect_mm(*self.cube_xy, self.CUBE_MM / 2))], 1)
            objs.append(GroundTruthObject("target", concepts.get("target", "cube"), m.astype(bool)))
            if self.ball_xy is not None:
                m = blank()
                c = tuple(int(v) for v in self.hom.workspace_to_pixel(*self.ball_xy))
                cv2.circle(m, c, int(34 * self.px_per_mm), 1, -1)
                objs.append(GroundTruthObject("distractor", concepts.get("distractor", "ball"), m.astype(bool)))
            if self.box is not None:
                m = blank()
                x0, x1, y0, y1 = self.box
                cv2.fillPoly(m, [self._poly_px([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])], 1)
                objs.append(GroundTruthObject("obstacle", concepts.get("obstacle", "box"), m.astype(bool)))
            hxy = self._hand_xy(t)
            if hxy is not None:
                hand_img = np.zeros((self.H, self.W, 3), np.uint8)
                self._draw_hand(hand_img, hxy)
                objs.append(GroundTruthObject("hand", concepts.get("hand", "hand"), hand_img.any(axis=2)))
        # Occlusion: visible target pixels exclude the hand.
        hands = [o.mask for o in objs if o.label == "hand"]
        if hands:
            for o in objs:
                if o.label == "target":
                    o.mask = o.mask & ~hands[0]
        return objs
