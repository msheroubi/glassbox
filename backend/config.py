"""Runtime configuration for GlassBox-SO101.

Every field can be overridden with an environment variable prefixed ``GLASSBOX_`` (for example
``GLASSBOX_ROBOT_PORT=/dev/ttyACM1``) or from a ``.env`` file in the repository root. The Meta Model
API key is read from ``MODEL_API_KEY`` (the variable name used in Meta's own docs).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Conservative SO-101 joint limits (degrees for body joints, 0-100 for the gripper), matching
# LeRobot's `use_degrees=True` convention. Used to clamp every recovery waypoint before execution.
JOINT_LIMITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (0.0, 100.0),
}


class SegmentationConcept(BaseModel):
    """One SAM 3.1 concept prompt. SAM accepts exactly one short noun phrase per request."""

    label: str  # role used by the attribution engine: gripper | target | obstacle | hand | distractor
    prompt: str  # noun phrase sent to SAM
    fallback_prompts: list[str] = []  # tried in order when the prompt finds nothing


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GLASSBOX_", env_file=str(REPO_ROOT / ".env"), extra="ignore"
    )

    # --- Meta Model API -------------------------------------------------------------------------
    model_api_key: str = Field(
        default="", validation_alias=AliasChoices("MODEL_API_KEY", "GLASSBOX_MODEL_API_KEY")
    )
    meta_api_base: str = "https://api.meta.ai/v1"
    meta_asr_ws_url: str = "wss://api.meta.ai/v1/asr/realtime"
    sam_model: str = "sam-3.1"
    spark_model: str = "muse-spark-1.3"
    wonder_model: str = "muse-wonder-1.0"
    image_model: str = "muse-image-1.0"
    voice_model: str = "muse-voice-transcribe-1.0"
    spark_reasoning_effort: str = "low"
    wonder_reasoning_effort: str = "high"
    spark_timeout_s: float = 30.0
    sam_timeout_s: float = 30.0
    wonder_timeout_s: float = 180.0
    image_timeout_s: float = 90.0
    http_retries: int = 2
    cache_dir: Path = REPO_ROOT / ".cache" / "meta_api"

    # --- Hardware -------------------------------------------------------------------------------
    mock_hardware: bool = False
    mock_scenario: str = "attention_hijack"
    # Camera-only (--webcam): stream a USB webcam as the overhead view with no arm; SAM 3.1 / Muse explain the scene.
    camera_only: bool = False
    robot_port: str = "/dev/ttyACM0"
    robot_id: str = "kenai_follower"
    serial_baudrate: int = 1_000_000
    # Camera keys must match the keys the ACT policy was trained with (dataset `observation.images.*`).
    overhead_camera_key: str = "scene"
    overhead_camera_index: int | str = 0
    overhead_width: int = 1280
    overhead_height: int = 720
    wrist_camera_key: str | None = "wrist"
    wrist_camera_index: int | str | None = 2
    wrist_width: int = 640
    wrist_height: int = 480
    camera_fps: int = 30
    max_relative_target_deg: float | None = 10.0

    # --- Policy ---------------------------------------------------------------------------------
    policy_path: str | None = None  # local dir or HF repo id of a trained LeRobot ACT checkpoint
    policy_device: str = "cpu"
    # Executing fewer steps than the chunk size gives overlapping chunks, which is what the
    # action-chunk variance (OOD signal) is computed from.
    policy_n_action_steps: int = 20
    task_description: str = "Pick up the red cube and place it in the bin."

    # --- Calibration ----------------------------------------------------------------------------
    homography_path: Path = REPO_ROOT / "calibration" / "homography.json"
    background_plate_path: Path = REPO_ROOT / "calibration" / "background.png"
    tabletop_bgr: tuple[int, int, int] = (150, 172, 196)  # fallback fill colour for suppression

    # --- XAI engine -----------------------------------------------------------------------------
    control_hz: float = 30.0
    ring_buffer_frames: int = 300  # 10 s at 30 Hz; the scrubber can rewind this far
    sam_clip_frames: int = 90
    joint_error_threshold_deg: float = 12.0
    joint_error_hold_s: float = 0.5
    chunk_variance_threshold: float = 25.0
    attention_hijack_threshold: float = 0.45
    stall_progress_s: float = 4.0
    auto_analyze: bool = True
    use_sam_video: bool = False  # True: upload the 90-frame clip; False: segment the anomaly keyframe
    live_sam_interval_s: float = 4.0  # real robot: background SAM pass feeding the live hijack trigger (0 = off)
    track_refresh_s: float = 3.0  # re-segment a suppressed object this often so the mask follows it (0 = off)
    concepts: list[SegmentationConcept] = [
        SegmentationConcept(label="gripper", prompt="robotic arm gripper"),
        SegmentationConcept(label="target", prompt="red cube"),
        SegmentationConcept(label="distractor", prompt="red ball"),
        SegmentationConcept(label="obstacle", prompt="cardboard box"),
        SegmentationConcept(label="hand", prompt="human hand"),
    ]

    # --- Server ---------------------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    stream_jpeg_quality: int = 70
    stream_fps: float = 15.0  # browser frame rate; the control loop and ring buffer still run at control_hz
    saliency_grid: tuple[int, int] = (64, 36)  # (W, H) of the heatmap sent to the browser

    @property
    def meta_api_enabled(self) -> bool:
        return bool(self.model_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
