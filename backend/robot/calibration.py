"""4-point homography between overhead-camera pixels (u, v) and the tabletop plane (X, Y) in mm.

Workspace frame: origin at the centre of the SO-101 base, +X pointing away from the robot (up the
image), +Y to the robot's left (image left). Run this module as a script to calibrate a real camera:

    python -m backend.robot.calibration --camera 0 --world "0,320;0,-320;360,-320;360,320"

Click the four reference marks in the same order as the ``--world`` list, press ``b`` with an empty
table to store a clean background plate (used by mask suppression), then ``s`` to save.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Homography:
    H: np.ndarray  # 3x3, pixels -> mm
    image_size: tuple[int, int]  # (width, height) the calibration was made at

    @classmethod
    def from_points(
        cls, pixels: list[tuple[float, float]], world_mm: list[tuple[float, float]], image_size: tuple[int, int]
    ) -> Homography:
        if len(pixels) != 4 or len(world_mm) != 4:
            raise ValueError("Exactly four point correspondences are required")
        H = cv2.getPerspectiveTransform(np.float32(pixels), np.float32(world_mm))
        return cls(H=H, image_size=image_size)

    @classmethod
    def default(cls, width: int, height: int) -> Homography:
        """Nominal overhead view: robot base at bottom-centre, 0.5 mm/px at 1280x720."""
        s = 0.5 * 1280 / width
        pixels = [(0, height), (width, height), (width, 0), (0, 0)]
        world = [(0, width / 2 * s), (0, -width / 2 * s), (height * s, -width / 2 * s), (height * s, width / 2 * s)]
        return cls.from_points(pixels, world, (width, height))

    @classmethod
    def load_or_default(cls, path: Path, width: int, height: int) -> Homography:
        if path.exists():
            data = json.loads(path.read_text())
            hom = cls(H=np.array(data["H"], dtype=np.float64), image_size=tuple(data["image_size"]))
            return hom.rescaled(width, height)
        return cls.default(width, height)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"H": self.H.tolist(), "image_size": list(self.image_size)}, indent=2))

    def rescaled(self, width: int, height: int) -> Homography:
        """Re-express the homography for a different capture resolution."""
        if (width, height) == tuple(self.image_size):
            return self
        sx, sy = self.image_size[0] / width, self.image_size[1] / height
        S = np.diag([sx, sy, 1.0])
        return Homography(H=self.H @ S, image_size=(width, height))

    def pixel_to_workspace(self, u: float, v: float) -> tuple[float, float]:
        p = self.H @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def workspace_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        p = np.linalg.inv(self.H) @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def table(self) -> dict:
        """Compact calibration table handed to Muse Wonder."""
        w, h = self.image_size
        corners = {
            name: [round(c, 1) for c in self.pixel_to_workspace(u, v)]
            for name, (u, v) in {
                "top_left_px": (0, 0),
                "top_right_px": (w, 0),
                "bottom_left_px": (0, h),
                "bottom_right_px": (w, h),
                "centre_px": (w / 2, h / 2),
            }.items()
        }
        return {
            "frame": "origin at SO-101 base centre; +X away from robot (image up); +Y robot-left (image left); mm",
            "image_size_px": [w, h],
            "H_pixel_to_mm": np.round(self.H, 6).tolist(),
            "corner_samples_mm": corners,
        }


def _calibrate_cli() -> None:
    from backend.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default=str(settings.overhead_camera_index))
    parser.add_argument("--width", type=int, default=settings.overhead_width)
    parser.add_argument("--height", type=int, default=settings.overhead_height)
    parser.add_argument(
        "--world",
        default="0,320;0,-320;360,-320;360,320",
        help="Four X,Y workspace points in mm, ';'-separated, in click order",
    )
    args = parser.parse_args()
    world = [tuple(float(c) for c in p.split(",")) for p in args.world.split(";")]

    cap = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    clicks: list[tuple[int, int]] = []
    cv2.namedWindow("calibrate")
    cv2.setMouseCallback(
        "calibrate", lambda ev, x, y, *_: clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4 else None
    )
    frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"Could not read camera {args.camera}")
        view = frame.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(view, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(view, f"{i + 1}: {world[i]}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        msg = f"click point {len(clicks) + 1} -> {world[len(clicks)]}" if len(clicks) < 4 else "s=save  r=reset"
        cv2.putText(view, msg + "  b=background plate  q=quit", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("calibrate", view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            clicks.clear()
        if key == ord("b"):
            settings.background_plate_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(settings.background_plate_path), frame)
            print(f"Saved background plate to {settings.background_plate_path}")
        if key == ord("s") and len(clicks) == 4:
            hom = Homography.from_points(clicks, world, (frame.shape[1], frame.shape[0]))
            hom.save(settings.homography_path)
            print(f"Saved homography to {settings.homography_path}")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _calibrate_cli()
