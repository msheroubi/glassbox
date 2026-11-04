# GlassBox-SO101

**Robot policies rarely fail because they decide badly. They fail because they see the scene wrong.**
GlassBox is an explainability and recovery layer for a LeRobot **SO-101** arm running an ACT policy
from an overhead webcam. It shows where the policy is looking, pinpoints the object that throws its
perception off, has Meta's models explain the failure, and fixes it without retraining when no human
is needed.

1. **Live.** ACT drives the arm. GlassBox records every frame with ACT's own attention and the servo
   telemetry, and holds the arm when it detects an anomaly.
2. **Replay.** The seconds before the failure are saved and replayed offline, where there is time for
   heavier models: SAM 3.1 segments the scene, and **SAM 3**, run locally (too slow to run live on a CPU), is
   asked frame by frame where it finds "red cube" and "green hex tile", a perception lens next to ACT's
   attention (`--replay-vla` uses SmolVLA's attention instead). The browser interpolates
   between attention keyframes, so playback is smooth.
3. **Breakdown.** **Muse Wonder** gets the filmstrip, the attention share of every object over time and
   the telemetry. It decides whether the failure started in *perception*, *decision* or *actuation*,
   names the culprit, and says whether GlassBox can fix it itself.
4. **Self-correct.** If it can (e.g. erase the distractor from the policy's camera input), GlassBox
   applies the fix and resumes. If not, it says what a person must do.

| Meta model | Used for |
|---|---|
| **SAM 3.1** (`sam-3.1`) | Segments the scene by concept ("red cube", "red book", "robot arm", "human hand", ...) |
| **Muse Spark** (`muse-spark-1.3`) | Fast live triage from one keyframe (failure type, culprit, fix) and Q&A |
| **Muse Wonder** (`muse-wonder-1.0`) | Forensic breakdown of the replayed failure; joint-space bypass plans for collisions |
| **Muse Image** | Counterfactual "what the policy should have seen" rendering |
| **Muse Voice Transcribe** | Push-to-talk commands and questions |

Every Meta call has a local fallback, so the whole stack runs without an API key. Every result is
tagged with its source: `meta-api`, `cache` or `local-fallback`.

## Quick start

```bash
pip install -r requirements.txt            # Python 3.10+
cp .env.example .env                       # optional: put MODEL_API_KEY=... in .env
scripts/start_mock.sh attention_hijack     # builds the UI on first run, then http://localhost:8000
```

| Script | What it runs |
|---|---|
| `scripts/start_mock.sh [scenario]` | Simulated SO-101 + overhead camera (`--mock-hardware`). Scenarios: `nominal`, `attention_hijack`, `servo_stall`, `occlusion`, `specular_glare` |
| `scripts/start_webcam.sh [index]` | **Camera-only**: a real USB webcam as the overhead view, no arm (see below) |
| `scripts/start_vla.sh [index]` | Camera-only, with the heatmap taken from a pretrained **SmolVLA**'s attention. Observe only: its actions are never sent (`pip install "lerobot[smolvla]"`) |
| `scripts/start_demo.sh` | **The demo**: real SO-101 + ACT live, SAM 3 failure replay (`LENS=smolvla` for SmolVLA), demo staging and self-correction (see [docs/08-demo-runbook.md](docs/08-demo-runbook.md)) |
| `scripts/start_robot.sh <policy>` | Real SO-101 driven by a trained LeRobot ACT checkpoint (`pip install "lerobot[feetech]"`) |
| `scripts/start_frontend_dev.sh` | Vite dev server on :5173 with hot reload, proxying to the backend on :8000 |

Set `PYTHON=/path/to/python` to choose the interpreter and `PORT=` to change the port. Extra
arguments go straight to `python -m backend.main`; run it with `--help` for the full list.

## How it works

```
 overhead cam ──► control loop (30 Hz thread) ──► policy (ACT / mock / camera observer)
                    │  saliency map + action-chunk variance + servo telemetry
                    ├─► ring buffer (10 s) ──► /ws/telemetry ──► React console
                    └─► anomaly detector ──► HOLD ──► analysis pipeline (asyncio)
                                                       SAM 3.1 masks
                                                       A(i) attention share per mask
                                                       Muse Spark diagnosis
                                                       Muse Wonder plan / Muse Image counterfactual
                                                     ◄── "1-Click Apply Muse Fix"
```

**A(i) attribution.** Saliency comes from the ACT decoder's cross-attention over the overhead
camera's tokens, with the layer-4 backbone activation norm as a fallback, via forward hooks in
`backend/robot/policy_hook.py`. For each SAM mask *i*, `A(i)` is the share of total saliency mass
that falls inside the mask.

**Anomaly triggers** (`backend/xai_engine/attribution.py`):
- Joint following error above 12° for 0.5 s.
- Action-chunk variance above 25 deg² across overlapping ACT chunks, an out-of-distribution signal.
- Attention deviation: the policy's attention leaves what the demos attended at the same arm pose (`calibration/attention_reference.npz`); without that file, a non-target object with `A > 0.45`.
- No task progress for 4 s.

A trigger pauses the arm (HOLD) and starts the analysis automatically, unless you pass
`--no-auto-analyze`. Thresholds live in `backend/config.py`.

**Failure replay** (`backend/xai_engine/replay.py`). On a failure, the last 12 s of the ring buffer
are saved to `runs/<id>/` at 15 fps (raw frames, the policy's input frames, ACT saliency, telemetry)
and re-analysed in the background, streaming progress to the console:
1. SAM 3.1 segments 4 keyframes across the run.
2. ACT's recorded attention is attributed per object at 5 Hz.
3. With `--replay-sam3`, SAM 3 (`facebook/sam3`, in `.venv-sam3`) maps where it finds the target's and
   the goal's phrases: perception, not attention. When its "red cube" map lands on the book, the scene
   is ambiguous for the task's own words. With `--replay-vla lerobot/smolvla_base` instead, SmolVLA
   computes *task grounding* and *action attention* maps. Keyframes are computed coarse-to-fine
   (bisection order), so the curves are usable after a few and sharpen as more arrive. This pauses
   while the arm runs, so the policy keeps the CPU.
4. An **attention shift** is the first keyframe where a distractor holds at least 12% of a map
   and more than the target, and still does at the next keyframe.
5. Muse Wonder writes the breakdown once the first lens keyframes are in (3 for SAM 3, 5 for SmolVLA). With
   `--self-correct`, its automatic interventions are applied immediately.

**Demo staging** (`--demo-distractor-on-goal`). The demo's ACT policy fails reliably only when the red
book lies on the green tile. With this flag, a SAM pre-flight scan before each run finds the book
wherever it is, and every frame the policy sees has the book pasted onto the goal (and erased from where
it really is). The arm still occludes the pasted book. This is labelled **DEMO STAGING** everywhere in
the UI and in the replay, and the Raw camera view shows the real scene. Suppressing the book turns the
staging off as well.

**Interventions** (`backend/xai_engine/intervention.py`):
- `INPAINT_MASK_SUPPRESSION`: Real-Time Visual Mask Suppression. Every frame, the culprit's pixels
  are erased from the *policy's* input. The fill is the calibrated empty-table plate, or Telea
  inpainting if there is no plate. The mask is re-segmented every 3 s so it follows the object.
- `WAYPOINT_BYPASS`: plays back a clamped, velocity-capped joint trajectory from Muse Wonder. The
  local planner covers it immediately while Wonder refines the plan.
- `RETRY_GRASP_OFFSET`: shifts the grasp point, e.g. away from specular glare.

## The console

Two tabs share the inspector on the right:
- **Failure replay** opens by itself when a failure is recorded and plays at 0.5×:
  - The policy's view with the attention heatmap, interpolated between keyframes, and SAM masks
    tagged with each object's attention share (`◀ HIJACK` when a distractor beats the target).
  - The map selector: ACT live, and the lens's maps (SAM 3 · 'red cube' / 'green hex tile', or SmolVLA
    grounding / action). On a SAM 3 map the tag reads `◀ READ AS 'red cube'` when a distractor holds more
    of the phrase's map than the object it names.
  - The attention-share timeline, with phase bands, events and a draggable playhead.
  - Muse Wonder's breakdown: the perception → decision → actuation stage strip, the culprit, a
    clickable timeline, and the self-correction.
  - Keyboard: Space plays and pauses, ←/→ step a frame, Shift+←/→ jump between keyframes. Saved
    runs can be reopened from the dropdown, or at `/#replay`.
- **Live** is the real-time view below.

- **Overhead view**: live frames with a saliency heatmap, SAM masks tagged `#id label A=…`
  (red when hijacking), suppressed masks drawn dashed, and the recovery path. Toggle between the
  **Policy** view (after suppression) and the **Raw** camera view.
- **Telemetry scrubber**: joint error, chunk variance and peak servo load with anomaly markers.
  Click the chart to rewind the 10 s ring buffer frame by frame.
- **Muse XAI Inspector**:
  - **Trigger Failure Analysis**, then the diagnosis, attribution bars, keyframe and counterfactual.
  - **1-Click Apply Muse Fix**, with an option to override the intervention.
  - The recovery plan.
  - Typed or hold-to-talk questions. Voice commands: "analyze", "apply the fix", "resume",
    "reset", "emergency stop".
- **E-STOP**: disables torque immediately.

## Camera-only mode (USB webcam, no arm)

```bash
scripts/start_webcam.sh 2        # or: python -m backend.main --webcam /dev/video2
```

This streams a real webcam through the same pipeline. Because no policy or arm is connected:
- The heatmap is bottom-up **spectral-residual saliency**, i.e. what is visually conspicuous, not
  what a policy attends to.
- The joint and servo triggers are off. Analysis runs on demand.

SAM 3.1 still segments the scene (a background pass runs every 4 s), Muse Spark still diagnoses it
and answers questions, and mask suppression can still be previewed in the Policy view.

The default concepts target the SO-101 pick-and-place scene. Pass your own with roles as
`role=phrase`; the roles are `target`, `gripper`, `obstacle` and `hand`, and any other label is
treated as a potential distractor:

```bash
scripts/start_webcam.sh 2 --concepts "target=red cube,gripper=robot arm,coffee mug,keyboard"
```

To find your camera index, open each one, e.g. with `python -c "import cv2; print(cv2.VideoCapture(2).read()[0])"`.

`scripts/start_vla.sh 2` swaps the spectral-residual heatmap for the attention of `lerobot/smolvla_base`
over the overhead frame. You can show the action expert's attention (the default) or the task text's
(`--vla-attention language`). Nothing it predicts reaches an arm. A pass takes ~6 s on CPU, so the map
refreshes about every 6 s. See [docs/06-running.md](docs/06-running.md).

## Real hardware

1. Calibrate the table-to-pixel homography and capture an empty-table background plate (used by
   mask suppression):
   ```bash
   python -m backend.robot.calibration --camera 0 --world "0,320;0,-320;360,-320;360,320"
   ```
2. Start the policy. `--robot-port` defaults to `/dev/ttyACM0`. The camera keys must match the
   ones the policy was trained with (`overhead_camera_key`, default `scene`):
   ```bash
   scripts/start_robot.sh outputs/train/act_so101/checkpoints/last/pretrained_model
   ```
With no `--policy-path`, the arm holds position while streaming and analysis keep working.

## API

| Endpoint | |
|---|---|
| `GET /api/status` | Run state, models, thresholds, active interventions |
| `POST /api/analyze` | Run the full pipeline on the latest frames and return `AnalysisResult` |
| `GET /api/analysis/{id}` · `/latest` · `/{id}/keyframe.jpg` · `/{id}/counterfactual.jpg` | Results |
| `POST /api/apply_fix` `{analysis_id, intervention_type?}` | Apply the recommended (or overridden) intervention |
| `POST /api/clear_interventions` · `/api/reset` · `/api/estop` · `/api/resume` | Run control |
| `POST /api/hold` · `POST /api/settings` `{self_correct}` | Pause the arm (lets the replay finish); toggle self-correction |
| `GET /api/replays` · `/api/replay/{id}` · `/latest` · `/{id}/frame/{i}.jpg?view=policy\|raw` | Saved failure runs, their replay analysis and frames |
| `POST /api/replay/{id}/open` · `/api/replay/{id}/apply` | Reload a saved run (re-analysing if needed); apply Wonder's fix |
| `POST /api/ask` `{question}` | Muse Spark Q&A grounded in the latest analysis |
| `GET /api/buffer` · `/api/buffer/{index}.jpg?overlay=true` | Ring-buffer timeline and frames |
| `WS /ws/telemetry` | JSON frame metadata, then a binary JPEG for each frame, plus events (`replay_started`, `replay_progress`, `replay_update`, `staging`, ...). Send `{"view": "policy" \| "raw"}` to choose the view |
| `WS /ws/voice` | Push-to-talk PCM16 16 kHz audio relayed to Muse Voice Transcribe |

## Configuration

Every field in `backend/config.py` can be set from the environment or `.env` with a `GLASSBOX_`
prefix, e.g. `GLASSBOX_POLICY_DEVICE=cuda` or `GLASSBOX_STREAM_FPS=15`. The API key is read from
`MODEL_API_KEY`. Meta API responses are cached under `.cache/meta_api/`.

## Layout

```
backend/
  main.py               FastAPI app, websockets, voice commands, CLI
  control_loop.py       30 Hz control thread, live attribution, analysis pipeline, interventions
  config.py, schemas.py Settings and Pydantic models shared with the frontend
  robot/                SO-101 (LeRobot), simulator, webcam source, policy hooks, calibration
  xai_engine/           attribution + anomaly detector, interventions + demo staging, ring buffer,
                        failure replay (replay.py), rendering
  meta_clients/         SAM 3.1, Muse Spark (triage), Muse Wonder (breakdown, plans), Muse Image + Voice
frontend/src/
  components/           OverheadCanvas, TelemetryScrubber (live); ReplayPlayer, AttentionTimeline,
                        ReplayBreakdown (replay); MuseXAIInspector
  viz.ts, replayMath.ts drawing helpers; keyframe interpolation
  useTelemetry.ts       websocket stream + throttled React state
  useVoice.ts           push-to-talk (AudioWorklet -> PCM16)
scripts/                startup scripts
```

## Notes and limitations

- SAM 3 and SmolVLA are *lenses*, not the policy that failed: larger pretrained models asked where the
  task's objects are in the frames ACT saw. SAM 3's maps are segmentations, not attention. ACT's own attention is always shown next to it. When the two
  agree that the book pulls attention off the cube, the case is strong. When they disagree, say so.
- The 12% attention-shift threshold was tuned on the simulator. Check it against your own runs: SmolVLA
  maps are more diffuse than ACT's.

- SAM 3.1 returns raster masks. Polygons sent to the browser are traced from them.
- In mock mode, simulator ground truth fills in objects SAM misses; these are tagged
  `sim-ground-truth`.
- Muse Wonder's plan is shown as an auditable routine. Only its joint waypoints are executed,
  after clamping and velocity limiting. No generated code runs.
- Without an ACT checkpoint the real-arm mode only streams and analyses. Mock mode uses an
  image-driven stand-in policy, so suppression changes its behaviour for the same pixel-level
  reason it would change ACT's.
