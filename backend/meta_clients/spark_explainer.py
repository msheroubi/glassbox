"""Muse Spark (``muse-spark-1.3``) failure diagnosis and operator Q&A.

Spark receives the composite keyframe (raw frame | SAM contours + policy saliency heatmap) plus the
numbers behind it (attention shares, servo telemetry, anomaly reasons) and returns a structured
``XAIDiagnosisResponse`` via ``text.format = json_schema``. If the API is unreachable and no cached
response exists, a deterministic rule engine over the same evidence produces the diagnosis so the
demo never stalls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np

from backend.config import Settings
from backend.schemas import AnomalyEvent, MaskAttribution, ServoTelemetry, XAIDiagnosisResponse
from backend.xai_engine.attribution import NON_DISTRACTOR_LABELS
from backend.xai_engine.visualize import data_url

from .base import MetaAPIError, MetaClient, json_schema_format, output_text, user_message

log = logging.getLogger(__name__)

FAILURE_TYPES = ["ATTENTION_HIJACK", "SERVO_STALL_COLLISION", "OCCLUSION_OOD", "SPECULAR_GLARE", "NONE"]
INTERVENTIONS = ["INPAINT_MASK_SUPPRESSION", "WAYPOINT_BYPASS", "RETRY_GRASP_OFFSET", "NONE"]

DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "failure_detected": {"type": "boolean"},
        "failure_type": {"type": "string", "enum": FAILURE_TYPES},
        "culprit_mask_id": {"type": ["integer", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "headline_summary": {"type": "string", "description": "One sentence for the operator, max 20 words."},
        "technical_root_cause": {"type": "string", "description": "2-4 sentences citing the numeric evidence."},
        "intervention_type": {"type": "string", "enum": INTERVENTIONS},
        "intervention_parameters": {
            "type": "object",
            "additionalProperties": {"type": ["number", "integer", "string"]},
        },
    },
    "required": [
        "failure_detected",
        "failure_type",
        "culprit_mask_id",
        "confidence",
        "headline_summary",
        "technical_root_cause",
        "intervention_type",
        "intervention_parameters",
    ],
}

SYSTEM_BRIEF = """You are the failure-analysis module of GlassBox-SO101, an explainability layer for a
LeRobot SO-101 arm running an imitation-learning policy (ACT) on the task: "{task}".

The image is a side-by-side composite from the overhead camera at the moment of the anomaly:
LEFT = raw camera frame. RIGHT = the same frame with SAM 3.1 segmentation contours labelled
"#<mask_id> <label> A=<attention share>" and the policy's spatial saliency as a jet heatmap
(red = where the policy is looking; saliency method: {saliency_method}).

Attention-to-Mask Attribution: A(i) = sum of saliency inside mask i / total saliency. A distractor
with A > {hijack_thr} means the policy's visual attention has been hijacked.

Failure taxonomy and the matching intervention:
- ATTENTION_HIJACK: a non-target object captures the policy's attention -> INPAINT_MASK_SUPPRESSION
  (parameters: mask_id of the distractor to erase from the policy's input, dilate_px).
- SERVO_STALL_COLLISION: following error and torque load spike while the gripper is pressed against
  an obstacle -> WAYPOINT_BYPASS (parameters: obstacle_mask_id, clearance_mm).
- OCCLUSION_OOD: the target is hidden (e.g. by a human hand) or the scene is out of distribution;
  high action-chunk variance -> NONE (the operator must clear the scene; parameters: blocking_mask_id).
- SPECULAR_GLARE: glare washes out part of the target so the perceived grasp point is offset ->
  RETRY_GRASP_OFFSET (parameters: dx_mm, dy_mm = correction to add to the grasp target, workspace frame).
  Signature: "no task progress" with low joint error and low action-chunk variance while the
  gripper keeps closing over the target and missing; target_grasp_bias_mm (saliency-weighted centre the
  policy perceives minus the SAM mask centroid of the target, from the clearest recent frame) has
  magnitude > 8 mm. Use dx_mm, dy_mm = -target_grasp_bias_mm.
- NONE: no failure.

Notes: the gripper mask naturally carries high attention while it hovers over the target; that is not
a hijack. Brief joint-error spikes during grasp attempts are normal; a collision shows sustained error
(joint_error_deg_mean_2s) with high load (mean_abs_load_pct_2s) while the gripper is against an
obstacle (an obstacle mask with iou_with_gripper > 0 is in contact with the gripper). Servo values are
taken at the anomaly, just before the arm was put on HOLD.

Workspace frame: millimetres, origin at the robot base, +X away from the base (image up), +Y to the
image left. Respond with JSON only, following the schema. culprit_mask_id must be one of the listed
mask ids or null. Cite the numbers (A values, joint error, load, variance) in technical_root_cause.

Evidence:
{evidence}"""


@dataclass
class DiagnosisContext:
    anomaly: AnomalyEvent | None
    attributions: list[MaskAttribution]
    latest: ServoTelemetry  # at the anomaly (the last tick before HOLD), not after: HOLD zeroes the error
    peak_joint_error: float
    peak_chunk_variance: float
    peak_load: list[float]
    gripper_xy_mm: tuple[float, float]
    policy_phase: str
    saliency_method: str
    # Saliency-weighted centre of the target minus its mask centroid, in mm (perceived grasp bias).
    target_grasp_bias_mm: tuple[float, float] | None = None
    active_interventions: list[str] = field(default_factory=list)
    # Means over the 2 s before the anomaly: separate a sustained stall from brief spikes.
    mean_joint_error_2s: float = 0.0
    mean_abs_load_2s: list[float] = field(default_factory=list)

    def evidence(self) -> dict:
        return {
            "anomaly_reasons": self.anomaly.reasons if self.anomaly else ["manual analysis request"],
            "objects": [a.model_dump() for a in self.attributions],
            "servo": {
                "qpos_actual_deg": [round(v, 1) for v in self.latest.qpos_actual],
                "qpos_target_deg": [round(v, 1) for v in self.latest.qpos_target],
                "torque_load_pct": [round(v, 1) for v in self.latest.torque_load],
                "joint_error_deg_at_anomaly": round(self.latest.joint_error, 2),
                "joint_error_deg_mean_2s": round(self.mean_joint_error_2s, 2),
                "joint_error_deg_peak_10s": round(self.peak_joint_error, 2),
                "mean_abs_load_pct_2s": [round(v, 1) for v in self.mean_abs_load_2s],
                "peak_abs_load_pct_10s": [round(v, 1) for v in self.peak_load],
                "action_chunk_variance_at_anomaly": round(self.latest.action_chunk_variance, 2),
                "action_chunk_variance_peak_10s": round(self.peak_chunk_variance, 2),
            },
            "gripper_xy_mm": [round(v, 1) for v in self.gripper_xy_mm],
            "policy_phase": self.policy_phase,
            "target_grasp_bias_mm": None
            if self.target_grasp_bias_mm is None
            else [round(v, 1) for v in self.target_grasp_bias_mm],
            "active_interventions": self.active_interventions,
        }


class SparkExplainer:
    def __init__(self, settings: Settings, client: MetaClient) -> None:
        self.s = settings
        self.client = client

    def _prompt(self, ctx: DiagnosisContext) -> str:
        return SYSTEM_BRIEF.format(
            task=self.s.task_description,
            saliency_method=ctx.saliency_method or "unknown",
            hijack_thr=self.s.attention_hijack_threshold,
            evidence=json.dumps(ctx.evidence(), indent=1),
        )

    async def diagnose(self, composite_bgr: np.ndarray, ctx: DiagnosisContext) -> tuple[XAIDiagnosisResponse, str]:
        """Returns ``(diagnosis, source)``. Never raises."""
        if self.client.enabled:
            body = {
                "model": self.s.spark_model,
                "input": user_message(
                    {"type": "input_text", "text": self._prompt(ctx)},
                    {"type": "input_image", "image_url": data_url(composite_bgr, 85)},
                ),
                "text": json_schema_format("xai_diagnosis", DIAGNOSIS_SCHEMA),
                "reasoning": {"effort": self.s.spark_reasoning_effort},
            }
            try:
                data, source = await self.client.post_json("/responses", body, self.s.spark_timeout_s)
                diag = self._validate(_parse_json(output_text(data)), ctx)
                return diag, source
            except Exception as e:  # noqa: BLE001 - fall back to the rule engine on any failure
                log.warning("Muse Spark diagnosis unavailable (%s); using local rules", e)
        return local_diagnosis(ctx, self.s.attention_hijack_threshold, self.s.joint_error_threshold_deg), "local-fallback"

    @staticmethod
    def _validate(raw: dict, ctx: DiagnosisContext) -> XAIDiagnosisResponse:
        params = raw.get("intervention_parameters") or {}
        raw["intervention_parameters"] = {
            str(k): v for k, v in params.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)
        }
        raw["confidence"] = float(np.clip(float(raw.get("confidence", 0.5)), 0.0, 1.0))
        diag = XAIDiagnosisResponse.model_validate(raw)
        ids = {a.mask_id for a in ctx.attributions}
        if diag.culprit_mask_id is not None and diag.culprit_mask_id not in ids:
            log.warning("Spark named unknown mask id %s; clearing it", diag.culprit_mask_id)
            diag.culprit_mask_id = None
        if diag.intervention_type == "INPAINT_MASK_SUPPRESSION":
            mid = diag.intervention_parameters.get("mask_id", diag.culprit_mask_id)
            if not isinstance(mid, (int, float)) or int(mid) not in ids:
                mid = diag.culprit_mask_id
            if mid is None:
                raise MetaAPIError("Mask suppression proposed without a valid mask id")
            diag.intervention_parameters["mask_id"] = int(mid)
        return diag

    async def answer(self, question: str, ctx: DiagnosisContext | None, last: XAIDiagnosisResponse | None,
                     image_bgr: np.ndarray | None) -> tuple[str, str]:
        """Free-form operator question about the robot's behaviour. Returns ``(answer, source)``."""
        if self.client.enabled:
            parts = [
                "You are GlassBox-SO101's explainability assistant for a LeRobot SO-101 arm running an ACT policy "
                f"on the task '{self.s.task_description}'. Answer the operator in at most 4 sentences, plainly, "
                "citing the evidence numbers where relevant. The image (if any) is raw overhead frame | SAM contours "
                "with attention shares A and the policy saliency heatmap.",
                f"Latest diagnosis: {last.model_dump_json() if last else 'none yet'}",
                f"Evidence: {json.dumps(ctx.evidence()) if ctx else 'none'}",
                f"Operator question: {question}",
            ]
            content = [{"type": "input_text", "text": "\n\n".join(parts)}]
            if image_bgr is not None:
                content.append({"type": "input_image", "image_url": data_url(image_bgr, 80)})
            body = {"model": self.s.spark_model, "input": user_message(*content), "reasoning": {"effort": "low"}}
            try:
                data, source = await self.client.post_json("/responses", body, self.s.spark_timeout_s)
                text = output_text(data).strip()
                if text:
                    return text, source
            except Exception as e:  # noqa: BLE001
                log.warning("Muse Spark Q&A unavailable (%s)", e)
        return local_answer(question, ctx, last), "local-fallback"


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise MetaAPIError(f"No JSON object in model output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def local_diagnosis(ctx: DiagnosisContext, hijack_thr: float, je_thr: float) -> XAIDiagnosisResponse:
    """Deterministic rules over the same evidence Spark sees. Order matters: most specific first."""
    attrs = ctx.attributions
    by_label = {}
    for a in attrs:
        by_label.setdefault(a.label, []).append(a)
    gx, gy = ctx.gripper_xy_mm
    peak_load = max((abs(v) for v in ctx.peak_load[:5]), default=0.0)

    hands = by_label.get("hand", [])
    if hands:
        h = max(hands, key=lambda a: a.attention_share)
        target = (by_label.get("target") or [None])[0]
        hidden = f" while the visible target holds only A={target.attention_share:.2f}" if target else ""
        return XAIDiagnosisResponse(
            failure_detected=True,
            failure_type="OCCLUSION_OOD",
            culprit_mask_id=h.mask_id,
            confidence=0.8,
            headline_summary="A human hand is covering the target; the policy has lost sight of it.",
            technical_root_cause=(
                f"Mask #{h.mask_id} (hand) overlaps the workspace at {h.workspace_xy_mm} mm{hidden}. "
                f"Action-chunk variance peaked at {ctx.peak_chunk_variance:.1f}, consistent with an "
                "out-of-distribution scene; erasing the hand cannot reveal the occluded target."
            ),
            intervention_type="NONE",
            intervention_parameters={"blocking_mask_id": h.mask_id, "action": "operator_clear_workspace"},
        )

    distractors = [a for a in attrs if a.label not in NON_DISTRACTOR_LABELS | {"hand"}]
    top = max(distractors, key=lambda a: a.attention_share, default=None)
    if top is not None and top.attention_share > hijack_thr:
        target = (by_label.get("target") or [None])[0]
        tgt = f" versus A={target.attention_share:.2f} on the target" if target else ""
        return XAIDiagnosisResponse(
            failure_detected=True,
            failure_type="ATTENTION_HIJACK",
            culprit_mask_id=top.mask_id,
            confidence=float(np.clip(0.55 + top.attention_share / 2, 0, 0.97)),
            headline_summary=f"The policy is staring at the {top.label} (#{top.mask_id}) instead of the target.",
            technical_root_cause=(
                f"Attention-to-mask attribution gives A={top.attention_share:.2f} on mask #{top.mask_id} "
                f"({top.label}){tgt}, above the {hijack_thr} hijack threshold. Action-chunk variance peaked at "
                f"{ctx.peak_chunk_variance:.1f}: the policy is split between two look-alike goals and hovers."
            ),
            intervention_type="INPAINT_MASK_SUPPRESSION",
            intervention_parameters={"mask_id": top.mask_id, "dilate_px": 15},
        )

    if ctx.peak_joint_error > je_thr or peak_load > 45.0:
        obstacles = by_label.get("obstacle", [])
        culprit = min(
            obstacles, key=lambda a: np.hypot(a.workspace_xy_mm[0] - gx, a.workspace_xy_mm[1] - gy), default=None
        )
        where = f"against mask #{culprit.mask_id} (obstacle) at {culprit.workspace_xy_mm} mm" if culprit else ""
        return XAIDiagnosisResponse(
            failure_detected=True,
            failure_type="SERVO_STALL_COLLISION",
            culprit_mask_id=culprit.mask_id if culprit else None,
            confidence=0.85 if culprit else 0.6,
            headline_summary="The arm is jammed against an obstacle on its reach path.",
            technical_root_cause=(
                f"Servo following error peaked at {ctx.peak_joint_error:.1f} deg (threshold {je_thr}) with "
                f"peak load {peak_load:.0f}% while the gripper sits at ({gx:.0f}, {gy:.0f}) mm {where}. "
                "The policy keeps commanding the straight-line reach because it never saw this obstacle in training."
            ),
            intervention_type="WAYPOINT_BYPASS",
            intervention_parameters={
                "obstacle_mask_id": culprit.mask_id if culprit else -1,
                "clearance_mm": 30,
            },
        )

    bias = ctx.target_grasp_bias_mm
    if bias is not None and np.hypot(*bias) > 8.0:
        target = (by_label.get("target") or [None])[0]
        return XAIDiagnosisResponse(
            failure_detected=True,
            failure_type="SPECULAR_GLARE",
            culprit_mask_id=target.mask_id if target else None,
            confidence=0.7,
            headline_summary="Glare hides part of the target, so the policy keeps grasping off-centre.",
            technical_root_cause=(
                f"The saliency-weighted grasp point sits ({bias[0]:.1f}, {bias[1]:.1f}) mm from the SAM mask "
                "centroid of the target: the washed-out region carries no saliency, so every grasp lands on the "
                f"edge and misses. Joint error stays low ({ctx.peak_joint_error:.1f} deg), ruling out a collision."
            ),
            intervention_type="RETRY_GRASP_OFFSET",
            intervention_parameters={"dx_mm": round(-bias[0], 1), "dy_mm": round(-bias[1], 1)},
        )

    if ctx.peak_chunk_variance > 25.0:
        return XAIDiagnosisResponse(
            failure_detected=True,
            failure_type="OCCLUSION_OOD",
            culprit_mask_id=None,
            confidence=0.5,
            headline_summary="The policy is uncertain about this scene and its action predictions disagree.",
            technical_root_cause=(
                f"Action-chunk variance peaked at {ctx.peak_chunk_variance:.1f} without a dominant distractor, "
                "collision signature or grasp bias, suggesting an out-of-distribution observation."
            ),
            intervention_type="NONE",
            intervention_parameters={},
        )

    return XAIDiagnosisResponse(
        failure_detected=False,
        failure_type="NONE",
        culprit_mask_id=None,
        confidence=0.6,
        headline_summary="No failure signature: the policy is attending to the target and tracking well.",
        technical_root_cause=(
            f"Joint error {ctx.latest.joint_error:.1f} deg, chunk variance {ctx.latest.action_chunk_variance:.1f}, "
            f"largest distractor share {top.attention_share if top else 0:.2f}."
        ),
        intervention_type="NONE",
        intervention_parameters={},
    )


def local_answer(question: str, ctx: DiagnosisContext | None, last: XAIDiagnosisResponse | None) -> str:
    if last is None and ctx is None:
        return "No analysis has run yet. Press 'Trigger Failure Analysis' or say 'analyze' and I will explain what the policy is doing."
    lines = []
    if last is not None:
        lines.append(f"{last.headline_summary} {last.technical_root_cause}")
        if last.intervention_type != "NONE":
            lines.append(f"Recommended fix: {last.intervention_type.replace('_', ' ').lower()}.")
    if ctx is not None:
        top = max(ctx.attributions, key=lambda a: a.attention_share, default=None)
        if top is not None:
            lines.append(f"Right now the most-attended object is #{top.mask_id} {top.label} with A={top.attention_share:.2f}.")
    lines.append("(Offline answer: Muse Spark is unreachable.)")
    return " ".join(lines)
