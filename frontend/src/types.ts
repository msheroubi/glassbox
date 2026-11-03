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
  counterfactual_url: string | null;
  counterfactual_source: ResultSource | null;
  keyframe_url: string;
  latency_ms: Record<string, number>;
  evidence: Record<string, unknown>;
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
