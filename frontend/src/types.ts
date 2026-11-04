// Mirrors backend/schemas.py and the JSON emitted by backend/control_loop.py.

export type RunState = "RUNNING" | "HOLD" | "ESTOP" | "COMPLETE";
export type FailureType = "ATTENTION_HIJACK" | "SERVO_STALL_COLLISION" | "OCCLUSION_OOD" | "SPECULAR_GLARE" | "NONE";
export type InterventionType = "INPAINT_MASK_SUPPRESSION" | "WAYPOINT_BYPASS" | "RETRY_GRASP_OFFSET" | "NONE";
export type ResultSource = "meta-api" | "cache" | "local-fallback";

export const INTERVENTIONS: InterventionType[] = [
  "INPAINT_MASK_SUPPRESSION",
  "WAYPOINT_BYPASS",
  "RETRY_GRASP_OFFSET",
  "NONE",
];

export interface ServoTelemetry {
  timestamp: number;
  qpos_actual: number[];
  qpos_target: number[];
  qvel: number[];
  torque_load: number[];
  action_chunk_variance: number;
  joint_error: number;
}

export interface MaskAttribution {
  mask_id: number;
  label: string;
  centroid_uv: number[];
  workspace_xy_mm: number[];
  attention_share: number;
  iou_with_gripper: number;
}

export interface XAIDiagnosisResponse {
  failure_detected: boolean;
  failure_type: FailureType;
  culprit_mask_id: number | null;
  confidence: number;
  headline_summary: string;
  technical_root_cause: string;
  intervention_type: InterventionType;
  intervention_parameters: Record<string, number | string>;
}

export interface JointWaypoint {
  qpos: number[];
  duration_s: number;
  note: string;
}

export interface RecoveryPlan {
  summary: string;
  strategy: InterventionType;
  waypoints: JointWaypoint[];
  suppress_mask_ids: number[];
  grasp_offset_mm: number[];
  python_routine: string;
  safety_notes: string;
}

export interface AnomalyEvent {
  timestamp: number;
  reasons: string[];
  joint_error: number;
  chunk_variance: number;
  max_attention_share: number;
}

export interface SegmentedObjectView {
  mask_id: number;
  label: string;
  prompt: string;
  confidence: number | null;
  bbox: number[];
  polygons: number[][][];
}

export interface LiveObject extends SegmentedObjectView {
  source: string;
  attention_share: number;
}

export interface AnalysisResult {
  analysis_id: string;
  created_at: number;
  trigger: "manual" | "auto" | "voice";
  anomaly: AnomalyEvent | null;
  objects: SegmentedObjectView[];
  attributions: MaskAttribution[];
  diagnosis: XAIDiagnosisResponse;
  diagnosis_source: ResultSource;
  sam_source: ResultSource;
  plan: RecoveryPlan | null;
  plan_source: ResultSource | null;
  plan_status: "not_needed" | "pending" | "ready" | "failed";
  plan_path_px: number[][] | null;
  counterfactual: Counterfactual | null;
  keyframe_url: string;
  latency_ms: Record<string, number>;
  evidence: Record<string, unknown>;
}

/** One model's attention on the failure frame with and without the culprit (same masks, same joint state). */
export interface CounterfactualMap {
  kind: string;
  title: string;
  grid_w: number;
  grid_h: number;
  grid_before: string;
  grid_after: string;
  shares_before: Record<string, number>;
  shares_after: Record<string, number>;
  moved: number;
  near_before?: Record<string, number>;
  near_after?: Record<string, number>;
  area?: Record<string, number>; // each label's neighbourhood as a fraction of the frame
  left_from?: Record<string, number>; // of the attention that moved: fraction that left each label's neighbourhood
  arrived_at?: Record<string, number>;
  peak_before?: string | null;
  peak_after?: string | null;
}

/** Muse Image removes the culprit inside its SAM region; the policies are asked where they'd look without it. */
export interface Counterfactual {
  status: "rendering" | "attending" | "ready" | "failed";
  failure_type: string;
  frame: number | null;
  culprit_label: string | null;
  removed: string;
  image_url: string | null;
  source: ResultSource | null;
  region_polygons: number[][][];
  maps: CounterfactualMap[];
  path_before_px: number[][] | null;
  path_after_px: number[][] | null;
  path_shift_px: number | null;
  verdict: string;
  outcome?: "confirmed" | "implicated" | "partial" | "ruled_out" | null;
  latency_ms: Record<string, number>;
}

export interface RuntimeStatus {
  state: RunState;
  status: string;
  phase: string;
  mock_hardware: boolean;
  camera_only: boolean;
  scenario: string | null;
  scenarios: Record<string, string>;
  policy: string;
  saliency_method: string;
  meta_api_enabled: boolean;
  models: Record<string, string>;
  suppression: { mask_id: number; label: string }[];
  recovery_active: boolean;
  grasp_offset_mm: number[];
  latest_analysis_id: string | null;
  analysis_running: boolean;
  fps: number;
  frame_size: [number, number];
  thresholds: { joint_error_deg: number; chunk_variance: number; attention_hijack: number };
  staging: StagingInfo | null;
  demo_staging_enabled: boolean;
  preflight: boolean;
  awaiting_start: boolean;
  self_correct: boolean;
  replay: { enabled: boolean; latest_id: string | null; vla: "off" | "loading" | "error" | "ready"; vla_path: string | null; lens?: string | null };
}

/** Demo staging: the distractor is cut from where it is and pasted onto the goal in the policy's input. */
export interface StagingInfo {
  label: string;
  suppressed: boolean;
  from_bbox: number[];
  to_bbox: number[];
}

export interface SaliencyGrid {
  w: number;
  h: number;
  method: string;
  data: string; // base64 uint8, row-major
}

export interface FrameMeta {
  type: "frame";
  index: number;
  t: number;
  state: RunState;
  phase: string;
  status: string;
  scenario: string | null;
  fps: number;
  telemetry: ServoTelemetry;
  saliency: SaliencyGrid | null;
  suppression: { active: boolean; objects: { mask_id: number; label: string }[] };
  recovery: { active: boolean; note: string };
  grasp_offset_mm: number[];
  objects: LiveObject[];
  view: "policy" | "raw";
}

export interface HistoryPoint {
  index: number;
  t: number;
  joint_error: number;
  chunk_variance: number;
  max_load: number;
  phase?: string; // present in /api/buffer snapshots
}

export interface LogEntry {
  id: number;
  t: number;
  kind: "info" | "anomaly" | "analysis" | "fix" | "error" | "voice";
  text: string;
}

export interface AskResponse {
  question: string;
  answer: string;
  source: ResultSource;
}

// ------------------------------------------------------------------ offline replay forensics
export type FailureStage = "PERCEPTION" | "DECISION" | "ACTUATION" | "NONE";
export type ReplayStatus = "saving" | "segmenting" | "attention" | "breakdown" | "refining" | "done" | "failed";
export type SelfCorrection = "off" | "pending" | "applied" | "declined" | "unavailable";
export type MapKind = "vla_language" | "vla_action" | "act" | "sam3_target" | "sam3_goal" | string;

export interface ReplayFrameInfo {
  t: number; // seconds relative to the failure frame
  phase: string;
  state: string;
  joint_error: number;
  attention_deviation: number;
}

export interface ReplayKeyframe {
  frame: number;
  t: number;
  grid: string; // base64 uint8, grid_w * grid_h, row-major
  shares: Record<string, number>; // share of the map inside each label's SAM masks
  near?: Record<string, number>; // share near each label (its neighbourhood); absent in older replays
  near_lift?: Record<string, number>; // near share / neighbourhood area (1 = what a uniform map gives it)
}

export interface ReplayMap {
  kind: MapKind;
  title: string;
  description: string;
  grid_w: number;
  grid_h: number;
  expected_keyframes: number;
  keyframes: ReplayKeyframe[];
}

export interface ReplaySegmentation {
  frame: number;
  t: number;
  objects: SegmentedObjectView[];
}

export interface ReplayEvent {
  t: number;
  frame: number;
  kind: "attention_shift" | "failure" | "anomaly" | "phase" | "staging";
  label: string;
  detail: string;
  map: string | null;
}

export interface TimelineEntry {
  t: number;
  event: string;
  evidence: string;
}

export interface FailureBreakdown {
  headline: string;
  failure_stage: FailureStage;
  culprit_label: string | null;
  perception_analysis: string;
  root_cause: string;
  timeline: TimelineEntry[];
  counterfactual: string;
  self_correctable: boolean;
  needs_human: boolean;
  human_action: string;
  intervention_type: InterventionType;
  intervention_parameters: Record<string, number | string>;
  confidence: number;
  operator_explanation: string;
}

export interface ReplayResult {
  replay_id: string;
  analysis_id: string | null;
  created_at: number;
  trigger: string;
  task: string;
  fps: number;
  n_frames: number;
  failure_frame: number;
  frames: ReplayFrameInfo[];
  has_policy_view: boolean;
  staging: StagingInfo | null;
  status: ReplayStatus;
  progress: number;
  progress_text: string;
  maps: ReplayMap[];
  segmentations: ReplaySegmentation[];
  events: ReplayEvent[];
  breakdown: FailureBreakdown | null;
  breakdown_source: ResultSource | null;
  self_correction: SelfCorrection;
  self_correction_note: string;
  counterfactual: Counterfactual | null;
  latency_ms: Record<string, number>;
}

export interface SavedRun {
  run_id: string;
  created_at: number;
  trigger: string;
  anomaly_reasons: string[];
  n_frames: number;
  duration_s: number;
  staged: boolean;
  analysed: boolean;
}
