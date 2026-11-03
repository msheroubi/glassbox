"""Pydantic v2 data contracts shared by the backend modules, the Meta clients and the frontend."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

FailureType = Literal[
    "ATTENTION_HIJACK", "SERVO_STALL_COLLISION", "OCCLUSION_OOD", "SPECULAR_GLARE", "NONE"
]
InterventionType = Literal[
    "INPAINT_MASK_SUPPRESSION", "WAYPOINT_BYPASS", "RETRY_GRASP_OFFSET", "NONE"
]
ResultSource = Literal["meta-api", "cache", "local-fallback"]


class ServoTelemetry(BaseModel):
    timestamp: float
    qpos_actual: List[float] = Field(..., min_length=6, max_length=6)
    qpos_target: List[float] = Field(..., min_length=6, max_length=6)
    qvel: List[float] = Field(default_factory=lambda: [0.0] * 6, min_length=6, max_length=6)
    torque_load: List[float] = Field(..., min_length=6, max_length=6)
    action_chunk_variance: float
    joint_error: float = 0.0
    attention_deviation: float = 0.0  # fraction of the policy's attention outside what the demos attended at this pose


class MaskAttribution(BaseModel):
    mask_id: int
    label: str
    centroid_uv: List[int]
    workspace_xy_mm: List[float]
    attention_share: float = Field(
        ..., description="0.0 to 1.0 fraction of policy saliency inside mask"
    )
    iou_with_gripper: float
    expected_share: float | None = Field(
        None, description="Share of attention inside this mask in the demos at the nearest arm poses"
    )
    excess_share: float | None = Field(
        None, description="Fraction of the *unexpected* attention (vs the demos) that lies inside this mask"
    )


class XAIDiagnosisResponse(BaseModel):
    failure_detected: bool
    failure_type: FailureType
    culprit_mask_id: int | None
    confidence: float
    headline_summary: str
    technical_root_cause: str
    intervention_type: InterventionType
    intervention_parameters: Dict[str, float | int | str]


class JointWaypoint(BaseModel):
    qpos: List[float] = Field(..., min_length=6, max_length=6, description="Joint targets, degrees (gripper 0-100)")
    duration_s: float = Field(..., description="Time to interpolate from the previous waypoint")
    note: str


class RecoveryPlan(BaseModel):
    """Muse Wonder's recovery output. Waypoints are executed after clamping; the routine is for audit."""

    summary: str
    strategy: InterventionType
    waypoints: List[JointWaypoint]
    suppress_mask_ids: List[int]
    grasp_offset_mm: List[float] = Field(..., min_length=2, max_length=2)
    python_routine: str
    safety_notes: str


class AnomalyEvent(BaseModel):
    timestamp: float
    reasons: List[str]
    joint_error: float
    chunk_variance: float
    max_attention_share: float = 0.0
    attention_deviation: float = 0.0


class SegmentedObjectView(BaseModel):
    """What the browser needs to draw one SAM object."""

    mask_id: int
    label: str
    prompt: str
    confidence: float | None
    bbox: List[int]  # x1, y1, x2, y2 (exclusive)
    polygons: List[List[List[int]]]


class AnalysisResult(BaseModel):
    analysis_id: str
    created_at: float
    trigger: Literal["manual", "auto", "voice"]
    anomaly: AnomalyEvent | None
    objects: List[SegmentedObjectView]
    attributions: List[MaskAttribution]
    diagnosis: XAIDiagnosisResponse
    diagnosis_source: ResultSource
    sam_source: ResultSource
    plan: RecoveryPlan | None = None
    plan_source: ResultSource | None = None
    plan_status: Literal["not_needed", "pending", "ready", "failed"] = "not_needed"
    plan_path_px: List[List[float]] | None = None  # gripper path in overhead pixels: start + one per waypoint
    counterfactual_url: str | None = None
    counterfactual_source: ResultSource | None = None
    keyframe_url: str
    latency_ms: Dict[str, float] = Field(default_factory=dict)
    evidence: Dict[str, Any] = Field(default_factory=dict)  # the numbers Muse Spark was given


class ApplyFixRequest(BaseModel):
    analysis_id: str
    intervention_type: InterventionType | None = None  # override the diagnosed intervention


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    source: ResultSource
