"""Muse Wonder (``muse-wonder-1.0``) recovery planning plus a local geometric planner.

Wonder gets the whole picture: 10 s of servo telemetry, the pixel->mm calibration table, every SAM
object in pixels and millimetres, the diagnosis, the joint limits, the arm's kinematic model and the
control-loop source that will execute its plan. It returns a ``RecoveryPlan`` of joint waypoints plus
an explanatory Python routine.

Nothing Wonder returns is executed blindly. Waypoints are clamped to joint limits, densified along
the same joint-space interpolation the executor uses, forward-simulated with the planar kinematic
model and checked against every obstacle footprint and the reach envelope. The Python routine is
shown to the operator for audit only; it is never ``exec``'d. If the plan fails validation (or the
API is unavailable) the deterministic visibility-graph planner below provides the bypass.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass

import numpy as np

from backend.config import JOINT_LIMITS, JOINT_NAMES, Settings
from backend.robot.calibration import Homography
from backend.robot.so101_interface import PlanarKinematics
from backend.schemas import JointWaypoint, MaskAttribution, RecoveryPlan, XAIDiagnosisResponse
from backend.xai_engine.attribution import SegmentedObject
from backend.xai_engine.intervention import clamp_joints

from .base import MetaClient, json_schema_format, output_text, user_message
from .spark_explainer import INTERVENTIONS, _parse_json

log = logging.getLogger(__name__)

PHYSICAL_MARGIN_MM = 12.5  # gripper half-width: closer than this to an obstacle footprint is contact
VIA_CLEARANCE_MM = 32.0
REACH_SPEED_MM_S = 90.0

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "strategy": {"type": "string", "enum": INTERVENTIONS},
        "waypoints": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "qpos": {"type": "array", "items": {"type": "number"}, "minItems": 6, "maxItems": 6},
                    "duration_s": {"type": "number"},
                    "note": {"type": "string"},
                },
                "required": ["qpos", "duration_s", "note"],
            },
        },
        "suppress_mask_ids": {"type": "array", "items": {"type": "integer"}},
        "grasp_offset_mm": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "python_routine": {"type": "string"},
        "safety_notes": {"type": "string"},
    },
    "required": [
        "summary", "strategy", "waypoints", "suppress_mask_ids", "grasp_offset_mm", "python_routine", "safety_notes"
    ],
}

KINEMATICS_BRIEF = f"""Joint order: {JOINT_NAMES}; degrees, gripper 0 (closed) .. 100 (open).
Joint limits: {json.dumps(JOINT_LIMITS)}.
Top-down planar model used to validate your plan (Python):
    L1 = L2 = 170.0  # mm; DROP = 90.0 (shoulder height above grasp plane); reach r in [60, 325] mm
    def ik(x, y):
        pan = degrees(atan2(y, x)); r = clip(hypot(x, y), 60, 325); d = hypot(r, DROP)
        t2 = acos(clip((d*d - L1**2 - L2**2) / (2*L1*L2), -1, 1)); alpha = atan2(r, DROP)
        t1 = alpha - atan2(L2*sin(t2), L1 + L2*cos(t2))
        return pan, degrees(t1) - 45.0, degrees(t2) - 90.0   # shoulder_pan, shoulder_lift, elbow_flex
    def fk(q):  # inverse of ik for (x, y) of the gripper
        t2 = radians(q[2] + 90); t1 = radians(q[1] + 45)
        alpha = t1 + atan2(L2*sin(t2), L1 + L2*cos(t2)); d = sqrt(L1**2 + L2**2 + 2*L1*L2*cos(t2))
        r = d*sin(alpha); return r*cos(radians(q[0])), r*sin(radians(q[0]))
wrist_flex 0 = level, wrist_roll 0, keep the gripper open (60) while bypassing.
The executor interpolates LINEARLY IN JOINT SPACE between waypoints (smoothstep timing, <= 60 deg/s),
so check the swept gripper path, not only the waypoints. Obstacle footprints must be cleared by at
least {PHYSICAL_MARGIN_MM} mm everywhere along the path (the gripper is ~25 mm wide)."""


@dataclass
class Rect:
    """Axis-aligned obstacle footprint in workspace millimetres."""

    x0: float
    x1: float
    y0: float
    y1: float
    mask_id: int = -1

    def distance(self, p: np.ndarray) -> float:
        dx = max(self.x0 - p[0], 0.0, p[0] - self.x1)
        dy = max(self.y0 - p[1], 0.0, p[1] - self.y1)
        return float(np.hypot(dx, dy))

    def expanded_corners(self, m: float) -> list[np.ndarray]:
        return [
            np.array(c)
            for c in [(self.x0 - m, self.y0 - m), (self.x1 + m, self.y0 - m), (self.x1 + m, self.y1 + m), (self.x0 - m, self.y1 + m)]
        ]

    def as_list(self) -> list[float]:
        return [round(float(v), 1) for v in (self.x0, self.x1, self.y0, self.y1)]


def obstacle_rect_mm(obj: SegmentedObject, homography: Homography) -> Rect:
    pts = [homography.pixel_to_workspace(u, v) for poly in obj.polygons(1.0) for u, v in poly]
    if not pts:
        x0, y0, x1, y1 = obj.bbox
        pts = [homography.pixel_to_workspace(u, v) for u, v in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
    a = np.array(pts)
    return Rect(a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max(), obj.mask_id)


def swept_path(start_q: np.ndarray, waypoints: list[JointWaypoint], kin: PlanarKinematics, n: int = 24) -> list[np.ndarray]:
    path, prev = [], np.asarray(start_q, np.float64)
    for wp in waypoints:
        goal = clamp_joints(np.asarray(wp.qpos, np.float64))
        for a in np.linspace(0, 1, n):
            path.append(np.array(kin.fk(prev + (goal - prev) * a)))
        prev = goal
    return path


def validate_plan(
    plan: RecoveryPlan,
    start_q: np.ndarray,
    obstacles: list[Rect],
    goal_xy: np.ndarray | None,
    kin: PlanarKinematics,
) -> list[str]:
    """Returns a list of problems; empty means the plan is safe to execute."""
    problems = []
    if not plan.waypoints:
        return ["plan has no waypoints"]
    if len(plan.waypoints) > 12:
        problems.append("more than 12 waypoints")
    for i, wp in enumerate(plan.waypoints):
        clamped = clamp_joints(np.asarray(wp.qpos, np.float64))
        if np.abs(clamped - wp.qpos).max() > 1e-6:
            problems.append(f"waypoint {i} exceeds joint limits (clamped)")
        if not 0.05 <= wp.duration_s <= 15:
            problems.append(f"waypoint {i} duration {wp.duration_s}s out of range")
    path = swept_path(start_q, plan.waypoints, kin)
    start_xy = path[0]
    for rect in obstacles:
        allowed = min(PHYSICAL_MARGIN_MM, rect.distance(start_xy) - 0.5)  # may start at the contact margin
        worst = min(rect.distance(p) for p in path)
        if worst < allowed:
            problems.append(f"path passes {worst:.1f} mm from obstacle #{rect.mask_id} (needs {PHYSICAL_MARGIN_MM})")
    radii = [np.hypot(*p) for p in path]
    if min(radii) < PlanarKinematics.R_MIN - 1 or max(radii) > PlanarKinematics.R_MAX + 1:
        problems.append("path leaves the reachable annulus")
    if goal_xy is not None and np.linalg.norm(path[-1] - goal_xy) > 30.0:
        problems.append(f"plan ends {np.linalg.norm(path[-1] - goal_xy):.0f} mm from the target")
    return problems


def _segment_clear(a: np.ndarray, b: np.ndarray, obstacles: list[Rect], kin: PlanarKinematics, start_xy: np.ndarray) -> bool:
    qa, qb = np.array(kin.ik(*a)), np.array(kin.ik(*b))
    for t in np.linspace(0, 1, 30):
        p = np.array(kin.fk(qa + (qb - qa) * t))
        for r in obstacles:
            if r.distance(p) < min(PHYSICAL_MARGIN_MM + 3.0, r.distance(start_xy) - 0.5):
                return False
        if not PlanarKinematics.R_MIN <= np.hypot(*p) <= PlanarKinematics.R_MAX:
            return False
    return True


def local_bypass_plan(
    start_q: np.ndarray, obstacles: list[Rect], goal_xy: np.ndarray, kin: PlanarKinematics, note: str = ""
) -> RecoveryPlan:
    """Back off from the contact, then route around obstacle corners to a pre-grasp above the target."""
    start_xy = np.array(kin.fk(start_q))
    if obstacles:
        nearest = min(obstacles, key=lambda r: r.distance(start_xy))
        closest = np.array([np.clip(start_xy[0], nearest.x0, nearest.x1), np.clip(start_xy[1], nearest.y0, nearest.y1)])
        away = start_xy - closest
        away = away / (np.linalg.norm(away) + 1e-6)
        retreat = start_xy + away * 35.0
    else:
        retreat = start_xy
    corners = [c for r in obstacles for c in r.expanded_corners(VIA_CLEARANCE_MM)]
    best, best_len = None, np.inf
    for k in range(0, 3):
        for seq in itertools.permutations(corners, k):
            pts = [retreat, *seq, goal_xy]
            length = sum(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:]))
            if length >= best_len:
                continue
            if all(_segment_clear(a, b, obstacles, kin, start_xy) for a, b in zip(pts, pts[1:])):
                best, best_len = pts, length
    if best is None:
        best = [retreat, goal_xy]
        note += " No collision-free corner route found; using direct path (operator review required)."
    waypoints = []
    prev = start_xy
    labels = ["back off from contact"] + [f"via point {i + 1} around obstacle" for i in range(len(best) - 2)] + ["pre-grasp above target"]
    for xy, label in zip(best, labels):
        pan, lift, elbow = kin.ik(*xy)
        dur = max(0.4, float(np.linalg.norm(xy - prev)) / REACH_SPEED_MM_S)
        waypoints.append(JointWaypoint(qpos=[round(pan, 2), round(lift, 2), round(elbow, 2), 0.0, 0.0, 60.0], duration_s=round(dur, 2), note=label))
        prev = xy
    route = " -> ".join(f"({p[0]:.0f}, {p[1]:.0f})" for p in [start_xy, *best])
    return RecoveryPlan(
        summary=f"Back off 35 mm, route around the obstacle footprint with {VIA_CLEARANCE_MM:.0f} mm clearance, then hand control back to the policy at the target.",
        strategy="WAYPOINT_BYPASS",
        waypoints=waypoints,
        suppress_mask_ids=[],
        grasp_offset_mm=[0.0, 0.0],
        python_routine=(
            "# Local visibility-graph planner (GlassBox fallback)\n"
            f"# gripper route (mm): {route}\n"
            "for wp in plan.waypoints:\n"
            "    q = clamp_joints(wp.qpos)\n"
            "    executor.move_to(q, duration=wp.duration_s)  # joint-space smoothstep, <= 60 deg/s\n"
            "policy.resume()  # ACT re-plans from the pre-grasp pose\n"
        ),
        safety_notes=("Validated by forward simulation of the swept joint-space path against every obstacle "
                      f"footprint (>= {PHYSICAL_MARGIN_MM} mm) and the reach envelope." + note).strip(),
    )


class WonderPlanner:
    def __init__(self, settings: Settings, client: MetaClient, kinematics: PlanarKinematics) -> None:
        self.s = settings
        self.client = client
        self.kin = kinematics

    def _geometry(self, objects, attributions, homography) -> tuple[list[Rect], np.ndarray | None]:
        obstacles = [obstacle_rect_mm(o, homography) for o in objects if o.label == "obstacle"]
        target = next((a for a in attributions if a.label == "target"), None)
        return obstacles, (np.array(target.workspace_xy_mm) if target else None)

    def baseline(
        self, start_q: np.ndarray, objects: list[SegmentedObject], attributions: list[MaskAttribution], homography: Homography
    ) -> RecoveryPlan:
        """Instant local plan, available while Muse Wonder is still reasoning."""
        obstacles, goal_xy = self._geometry(objects, attributions, homography)
        if goal_xy is None:
            return local_bypass_plan(start_q, obstacles, np.array(self.kin.fk(start_q)), self.kin, " Target not segmented.")
        return local_bypass_plan(start_q, obstacles, goal_xy, self.kin)

    async def plan(
        self,
        diagnosis: XAIDiagnosisResponse,
        start_q: np.ndarray,
        objects: list[SegmentedObject],
        attributions: list[MaskAttribution],
        homography: Homography,
        telemetry_history: list[dict],
        control_source: str,
    ) -> tuple[RecoveryPlan, str, list[str]]:
        """Returns ``(plan, source, validation_notes)``. Never raises; falls back to the local planner."""
        obstacles, goal_xy = self._geometry(objects, attributions, homography)
        fallback = self.baseline(start_q, objects, attributions, homography)
        if not self.client.enabled:
            return fallback, "local-fallback", []

        prompt = "\n\n".join(
            [
                "You are the recovery planner for a LeRobot SO-101 arm whose imitation-learning policy (ACT) just "
                f"failed at the task '{self.s.task_description}'. Plan a short joint-space recovery that gets the "
                "gripper, open, to a pre-grasp pose above the target WITHOUT touching any obstacle, after which the "
                "policy resumes. Respond with JSON matching the schema; strategy should be WAYPOINT_BYPASS. "
                "python_routine documents the routine for the operator (it is reviewed, never executed).",
                f"Diagnosis (Muse Spark): {diagnosis.model_dump_json()}",
                f"Current joint state (deg): {[round(v, 2) for v in start_q]}; gripper at "
                f"{[round(v, 1) for v in self.kin.fk(start_q)]} mm.",
                "Objects (SAM 3.1 masks; pixel centroid, workspace centroid and attention share): "
                + json.dumps([a.model_dump() for a in attributions]),
                "Obstacle footprints [x0, x1, y0, y1] mm: "
                + json.dumps({f"#{r.mask_id}": r.as_list() for r in obstacles}),
                f"Target pre-grasp xy (mm): {None if goal_xy is None else [round(v, 1) for v in goal_xy]}",
                "Camera calibration: " + json.dumps(homography.table()),
                KINEMATICS_BRIEF,
                "Servo telemetry, last 10 s (t s, qpos_actual, qpos_target, load %, joint_error, chunk_variance): "
                + json.dumps(telemetry_history),
                "Control loop that will execute the plan:\n```python\n" + control_source + "\n```",
                "A planner baseline (already collision-checked). Improve on it if you can (shorter path, more "
                "clearance, better timing); either way write your OWN summary, python_routine and safety_notes "
                "explaining the recovery in terms of the telemetry and the obstacle geometry: "
                + fallback.model_dump_json(),
            ]
        )
        body = {
            "model": self.s.wonder_model,
            "input": user_message({"type": "input_text", "text": prompt}),
            "text": json_schema_format("recovery_plan", PLAN_SCHEMA),
            "reasoning": {"effort": self.s.wonder_reasoning_effort},
        }
        try:
            data, source = await self.client.post_json("/responses", body, self.s.wonder_timeout_s)
            raw = _parse_json(output_text(data))
            raw.setdefault("suppress_mask_ids", [])
            raw.setdefault("grasp_offset_mm", [0.0, 0.0])
            plan = RecoveryPlan.model_validate(raw)
        except Exception as e:  # noqa: BLE001
            log.warning("Muse Wonder unavailable (%s); using local bypass planner", e)
            return fallback, "local-fallback", [f"Muse Wonder unavailable: {e}"]

        problems = validate_plan(plan, start_q, obstacles, goal_xy, self.kin)
        if problems:
            log.warning("Rejected Wonder plan: %s", problems)
            fallback.safety_notes += " Muse Wonder's plan was rejected by validation: " + "; ".join(problems)
            fallback.python_routine = (
                "# Muse Wonder routine (REJECTED by GlassBox validation, shown for audit)\n"
                + plan.python_routine + "\n\n" + fallback.python_routine
            )
            return fallback, "local-fallback", problems
        for wp in plan.waypoints:
            wp.qpos = [round(float(v), 2) for v in clamp_joints(np.asarray(wp.qpos, np.float64))]
        plan.strategy = "WAYPOINT_BYPASS"
        notes = []
        if len(plan.waypoints) == len(fallback.waypoints) and all(
            np.allclose(a.qpos, b.qpos, atol=0.5) for a, b in zip(plan.waypoints, fallback.waypoints)
        ):
            notes.append("Muse Wonder endorsed the local planner's waypoints unchanged")
            if "GlassBox fallback" in plan.python_routine:
                plan.python_routine = "# Muse Wonder: endorsed the local planner's route\n" + plan.python_routine
        return plan, source, notes
