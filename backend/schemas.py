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


class CounterfactualMap(BaseModel):
    """One model's attention on the real frame vs the edited frame, at the same joint state."""

    kind: str  # act | vla_action | vla_language | the live policy's saliency method
    title: str
    grid_w: int
    grid_h: int
    grid_before: str  # base64 uint8, grid_w * grid_h, row-major
    grid_after: str
    shares_before: Dict[str, float]  # SAM label -> attention share, same masks for both frames
    shares_after: Dict[str, float]
    moved: float  # total-variation distance between the two attention maps (0 = identical, 1 = disjoint)
    # Near each object (its neighbourhood, see attribution.neighbourhoods): diffuse maps put ~1% strictly inside
    # a small mask even when their hot spot is right next to it, so the test also looks around the masks.
    near_before: Dict[str, float] = Field(default_factory=dict)  # label -> share of attention near it
    near_after: Dict[str, float] = Field(default_factory=dict)
    area: Dict[str, float] = Field(default_factory=dict)  # label -> its neighbourhood's fraction of the frame
    # Where the change happened: of the attention that moved, the fraction that left / arrived near each label.
    # A small total move can be concentrated on the culprit (left_from >> area): the culprit pulled that attention.
    left_from: Dict[str, float] = Field(default_factory=dict)
    arrived_at: Dict[str, float] = Field(default_factory=dict)
    peak_before: str | None = None  # label whose neighbourhood holds the map's hottest point (None: nowhere near one)
    peak_after: str | None = None


class Counterfactual(BaseModel):
    """The scene without the culprit (Muse Image, confined to SAM's mask) and the policy's attention on it."""

    status: Literal["rendering", "attending", "ready", "failed"] = "rendering"
    failure_type: FailureType
    frame: int | None = None  # replay frame index (None for a live analysis)
    culprit_label: str | None = None
    removed: str  # what was edited out, in words: "the red book"
    image_url: str | None = None
    source: ResultSource | None = None
    region_polygons: List[List[List[int]]] = Field(default_factory=list)  # the only pixels the edit changed
    maps: List[CounterfactualMap] = Field(default_factory=list)
    path_before_px: List[List[float]] | None = None  # the policy's predicted gripper path, overhead pixels
    path_after_px: List[List[float]] | None = None
    path_shift_px: float | None = None  # how far the predicted reach's end point moves
    verdict: str = ""
    # What the primary model's test says about the culprit (counterfactual.outcome): "implicated" | "confirmed"
    # (its reach moved too) | "partial" | "ruled_out"; None until the maps are compared.
    outcome: str | None = None
    latency_ms: Dict[str, float] = Field(default_factory=dict)


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
    counterfactual: Counterfactual | None = None
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


# ---------------------------------------------------------------------------------------- replay forensics
FailureStage = Literal["PERCEPTION", "DECISION", "ACTUATION", "NONE"]
ReplayStatus = Literal["saving", "segmenting", "attention", "breakdown", "refining", "done", "failed"]


class ReplayFrameInfo(BaseModel):
    t: float  # seconds relative to the failure frame (<= 0)
    phase: str
    state: str
    joint_error: float
    attention_deviation: float


class ReplayKeyframe(BaseModel):
    frame: int  # index into the run's frames
    t: float
    grid: str  # base64 uint8, grid_w * grid_h, row-major
    shares: Dict[str, float]  # SAM label -> share of this map's attention inside that label's masks
    near: Dict[str, float] = Field(default_factory=dict)  # SAM label -> share near it (its neighbourhood)
    near_lift: Dict[str, float] = Field(default_factory=dict)  # near share / neighbourhood area (1 = uniform)


class ReplayMap(BaseModel):
    kind: str  # act | vla_language | vla_action (SmolVLA lens) | sam3_target | sam3_goal (SAM 3 lens)
    title: str
    description: str
    grid_w: int
    grid_h: int
    expected_keyframes: int
    keyframes: List[ReplayKeyframe] = Field(default_factory=list)


class ReplaySegmentation(BaseModel):
    frame: int
    t: float
    objects: List[SegmentedObjectView]


class ReplayEvent(BaseModel):
    t: float
    frame: int
    kind: Literal["attention_shift", "failure", "anomaly", "phase", "staging"]
    label: str
    detail: str
    map: str | None = None


class TimelineEntry(BaseModel):
    t: float = Field(..., description="Seconds relative to the failure (negative = before it)")
    event: str
    evidence: str


class FailureBreakdown(BaseModel):
    """Muse Wonder's forensic breakdown of one recorded run."""

    headline: str
    failure_stage: FailureStage
    culprit_label: str | None
    perception_analysis: str
    root_cause: str
    timeline: List[TimelineEntry]
    counterfactual: str
    self_correctable: bool
    needs_human: bool
    human_action: str
    intervention_type: InterventionType
    intervention_parameters: Dict[str, float | int | str] = Field(default_factory=dict)
    confidence: float
    operator_explanation: str


class ReplayResult(BaseModel):
    replay_id: str
    analysis_id: str | None  # the live analysis this run belongs to (None for a run loaded from disk)
    created_at: float
    trigger: str
    task: str
    fps: float
    n_frames: int
    failure_frame: int
    frames: List[ReplayFrameInfo]
    has_policy_view: bool
    staging: Dict[str, Any] | None = None  # demo staging that was active during the run
    status: ReplayStatus = "saving"
    progress: float = 0.0
    progress_text: str = ""
    maps: List[ReplayMap] = Field(default_factory=list)
    segmentations: List[ReplaySegmentation] = Field(default_factory=list)
    events: List[ReplayEvent] = Field(default_factory=list)
    breakdown: FailureBreakdown | None = None
    breakdown_source: ResultSource | None = None
    self_correction: Literal["off", "pending", "applied", "declined", "unavailable"] = "off"
    self_correction_note: str = ""
    counterfactual: Counterfactual | None = None
    latency_ms: Dict[str, float] = Field(default_factory=dict)


class SettingsPatch(BaseModel):
    self_correct: bool | None = None
