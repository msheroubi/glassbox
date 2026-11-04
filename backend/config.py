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
    robot_id: str = "kowalski_follower"
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
    # Steps executed from each predicted chunk (the whole chunk, as in lerobot-record, runs most smoothly).
    policy_n_action_steps: int = 100
    # Re-run the policy every N steps without changing the executing chunk ("observe" passes): refreshes
    # the attention map and gives the overlapping predictions the chunk-variance OOD signal needs. 0 = off.
    policy_observe_every: int = 20
    # Camera-only: show a pretrained VLA's attention instead of spectral-residual saliency (no arm commands).
    vla_path: str | None = None  # e.g. lerobot/smolvla_base
    vla_camera_key: str = "camera1"  # the VLA's name for the overhead camera
    vla_attention: str = "action"  # "action": action expert -> image; "language": task tokens -> image
    task_description: str = "Pick up the red cube and place it on the green hex tile."

    # --- Calibration ----------------------------------------------------------------------------
    homography_path: Path = REPO_ROOT / "calibration" / "homography.json"
    background_plate_path: Path = REPO_ROOT / "calibration" / "background.png"
    tabletop_bgr: tuple[int, int, int] = (150, 172, 196)  # fallback fill colour for suppression

    # --- XAI engine -----------------------------------------------------------------------------
    control_hz: float = 30.0
    ring_buffer_frames: int = 900  # 30 s at 30 Hz (~15 KB a frame): the scrubber and the replay recorder can rewind this far
    sam_clip_frames: int = 90
    joint_error_threshold_deg: float = 12.0
    joint_error_hold_s: float = 0.5
    chunk_variance_threshold: float = 25.0
    attention_hijack_threshold: float = 0.45  # fixed A(i) rule: only used when there is no attention reference
    # Attention-deviation trigger: attention compared with the demos' attention at the same arm pose
    # (calibration/attention_reference.npz, built by scripts/build_attention_reference.py).
    attention_reference_path: Path = REPO_ROOT / "calibration" / "attention_reference.npz"
    # With a reference, the per-object hijack test is A(i) minus the share the demos had there at this pose.
    attention_hijack_excess_threshold: float = 0.15
    attention_deviation_threshold: float | None = None  # None = the threshold calibrated into the reference file
    attention_deviation_hold_s: float = 1.0  # sustained over two observe passes, so one noisy map can't stop the arm
    stall_progress_s: float = 4.0
    disabled_triggers: list[str] = []  # servo | variance | hijack | stall: detection off, arm keeps running
    auto_analyze: bool = True
    use_sam_video: bool = False  # True: upload the 90-frame clip; False: segment the anomaly keyframe
    live_sam_interval_s: float = 4.0  # real robot: background SAM pass feeding the live hijack trigger (0 = off)
    track_refresh_s: float = 3.0  # re-segment a suppressed object this often so the mask follows it (0 = off)
    concepts: list[SegmentationConcept] = [
        SegmentationConcept(label="gripper", prompt="robotic arm gripper"),
        SegmentationConcept(label="target", prompt="red cube"),
        SegmentationConcept(label="distractor", prompt="red book", fallback_prompts=["red notebook", "small red book"]),
        SegmentationConcept(label="obstacle", prompt="cardboard box"),
        SegmentationConcept(label="goal", prompt="green hex tile"),
        SegmentationConcept(label="hand", prompt="human hand"),
    ]

    # SAM scores the cube as a "red book" (and the book as a "red cube") almost as confidently as the right
    # way round, so on the real camera each red object's colour inside its mask also votes for its label.
    # Median OpenCV 8-bit LAB of the saturated pixels, measured on this rig's overhead camera
    # (scripts/measure_appearance.py <run> re-measures them, e.g. under a venue's lighting).
    # GLASSBOX_APPEARANCE_REFS='{}' turns the vote off.
    appearance_refs: dict[str, list[float]] = {"target": [156.0, 189.0, 155.0], "distractor": [169.0, 182.0, 134.0]}

    # --- Offline replay (the failure, re-analysed with heavier attention models) -----------------
    replay_enabled: bool = True
    runs_dir: Path = REPO_ROOT / "runs"
    replay_window_s: float = 15.0  # seconds of the run before the failure that are replayed
    replay_tail_s: float = 2.0  # and after it: the arm stopping (recorded while SAM and Spark run)
    replay_fps: float = 15.0  # frames kept per second of run (playback is interpolated in the browser)
    replay_sam_keyframes: int = 4  # SAM 3.1 segmentations across the replay (masks for the A(label) curves)
    replay_vla_path: str | None = None  # e.g. lerobot/smolvla_base: observe-only attention lens for replays
    replay_vla_hz: float = 1.0  # SmolVLA keyframes per second of run (~6 s each on CPU)
    replay_coarse_keyframes: int = 5  # Muse Wonder starts once this many SmolVLA keyframes are done
    # SAM 3 (facebook/sam3, run locally) as the replay lens instead of SmolVLA: where a perception model finds
    # the target and goal phrases in each keyframe. Not the policy's attention; it shows visual ambiguity.
    replay_sam3: bool = False
    sam3_model: str = "facebook/sam3"
    sam3_python: Path = REPO_ROOT / ".venv-sam3" / "bin" / "python"  # transformers >= 5 lives only there
    sam3_lens_labels: list[str] = ["target", "goal"]  # concepts whose prompts SAM 3 grounds
    replay_sam3_hz: float = 0.5  # SAM 3 keyframes per second of run (~20-25 s each on CPU)
    replay_sam3_coarse_keyframes: int = 3  # Muse Wonder starts once this many SAM 3 keyframes are done
    replay_primary_map: str = "vla_language"  # the map shown in Wonder's filmstrip (falls back if missing)
    replay_counterfactual_wait_s: float = 30.0  # Wonder waits this long for the counterfactual test
    # Self-correction: apply Wonder's intervention automatically when it says no human is needed.
    self_correct: bool = False

    # --- Demo staging ---------------------------------------------------------------------------
    # Show the policy the distractor ON the goal tile wherever it really is (the overfit ACT policy then
    # fails reliably). Configured by a SAM pre-flight scan before each run; labelled in the UI and the replay.
    demo_distractor_on_goal: bool = False
    # A manual Analyze (button or voice) with no anomaly on record is treated as a failure: the arm holds and the
    # run is diagnosed, replayed and self-corrected like a detected one. Labelled "DEMO" in the anomaly reasons.
    demo_manual_failure: bool = False
    # Wait in HOLD for the operator's Start (the Resume button) after launch and after each Reset, instead of
    # running the policy as soon as it is loaded. Lets the cameras, SAM pre-flight and SmolVLA load first.
    start_paused: bool = False
    preflight_timeout_s: float = 20.0

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
