# GlassBox-SO101

**Robot policies rarely fail because they decide badly. They fail because they see the scene wrong.**
GlassBox is an explainability and recovery layer for a LeRobot **SO-101** arm running an ACT policy
from an overhead webcam. It shows where the policy is looking, pinpoints the object that throws its
perception off, has Meta's models explain the failure, and fixes it without retraining when no human
is needed.

1. **Live.** ACT drives the arm. GlassBox records every frame with ACT's own attention and the servo
   telemetry, and holds the arm when it detects an anomaly.
2. **Replay.** The seconds before the failure are saved and replayed offline, where there is time for
   heavier models: SAM 3.1 segments the scene, and **SAM 3**, run locally (too slow to run live on a CPU),
   is asked where it finds "red cube" and "green hex tile" in the failure frame, a perception lens next to
   ACT's attention. The browser interpolates between attention keyframes, so playback is smooth.
3. **Breakdown.** **Muse Wonder** gets the filmstrip, the attention share of every object over time and
   the telemetry. It decides whether the failure started in *perception*, *decision* or *actuation*,
   names the culprit, and says whether GlassBox can fix it itself.
   It then proposes one experiment of its own (remove an object at a recorded moment, with a predicted
   effect); GlassBox runs it, and Wonder's final breakdown says whether its hypothesis held.
4. **Self-correct.** If it can (e.g. erase the distractor from the policy's camera input), GlassBox
   applies the fix and resumes. If not, it says what a person must do.

| Meta model | Used for |
|---|---|
| **SAM 3.1** (`sam-3.1`) | Segments the scene by concept ("red cube", "red book", "robot arm", "human hand", ...) |
| **SAM 3** (`facebook/sam3`, local) | Replay lens: where a perception model finds the task's noun phrases |
| **Muse Spark** (`muse-spark-1.3`) | Fast live triage from one keyframe (failure type, culprit, fix) and Q&A |
| **Muse Wonder** (`muse-wonder-1.0`) | Forensic breakdown of the replayed failure; joint-space bypass plans for collisions |
| **Muse Image** | Counterfactual "what the policy should have seen" rendering |
| **Muse Voice Transcribe** | Push-to-talk commands and questions |

Every Meta API call has a local fallback, so the stack still runs without an API key. Every result is
tagged with its source: `meta-api`, `cache` or `local-fallback`.

## Setup

```bash
# 1. The main environment (Python 3.10, the one LeRobot runs in)
pip install "lerobot[feetech]"             # the SO-101 driver and the ACT policy (brings torch)
pip install -r requirements.txt
cp .env.example .env                       # put MODEL_API_KEY=... in .env

# 2. SAM 3 needs transformers >= 5 and LeRobot is pinned to 4.x, so it gets its own venv on top of the
#    main environment's torch. facebook/sam3 is gated on Hugging Face: accept its licence and log in.
python -m venv --system-site-packages .venv-sam3
.venv-sam3/bin/pip install "transformers>=5" accelerate pillow
.venv-sam3/bin/hf auth login

# 3. Optional: calibrate the table-to-pixel homography and capture an empty-table background plate (the
#    plate is the fill for mask suppression; without it, suppression inpaints)
python -m backend.robot.calibration --camera 2 --world "0,320;0,-320;360,-320;360,320"
```

The console (React + Vite in `frontend/`) is built into `frontend/dist` on the first start, so `npm` is
needed once.

## Running the demo

```bash
PYTHON=~/miniconda3/envs/lerobot/bin/python scripts/start_demo.sh
```

Then open http://localhost:8000 (saved failures: http://localhost:8000/#replay). The script checks for
the arm and `.venv-sam3`, then starts `python -m backend.main` with this rig's defaults. Each one is an
environment variable:

| Variable | Default | |
|---|---|---|
| `PYTHON` | `python` | The main environment's interpreter |
| `POLICY` | `../models/last2/pretrained_model` | ACT checkpoint (local dir or HF repo id) |
| `ROBOT_PORT` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46082778-if00` | Follower arm (`ttyACM` numbers swap on re-plug) |
| `ROBOT_ID` | `kowalski_follower` | LeRobot calibration id |
| `CAMERA` | `2` | Overhead webcam (captured at 640x360, no wrist camera) |
| `PORT` | `8000` | |
| `STAGING` | `1` | Show the policy the red book on the goal wherever it is (`0` if you place it there yourself) |
| `AUTOSTART` | `0` | `0` waits for **▶ Start policy** after launch and each Reset |
| `FORCE_FAILURE` | `1` | A manual Analyze counts as a failure even if nothing was detected |
| `SELF_CORRECT` | `1` | Apply Wonder's fix automatically (`0` shows an Apply button) |

Extra arguments go straight to `python -m backend.main` (`--help` lists them). Every other setting in
`backend/config.py` can be set from the environment or `.env` with a `GLASSBOX_` prefix, e.g.
`GLASSBOX_POLICY_DEVICE=cuda` or `GLASSBOX_AUTO_ANALYZE=true`. The API key is read from
`MODEL_API_KEY`. Meta API responses are cached under `.cache/meta_api/`.

## How it works

```
 overhead cam ──► control loop (30 Hz thread) ──► ACT policy ──► SO-101
                    │  saliency map + action-chunk variance + servo telemetry
                    ├─► ring buffer (30 s) ──► /ws/telemetry ──► React console
                    └─► anomaly detector ──► HOLD ──► analysis pipeline (asyncio)
                                                       SAM 3.1 masks
                                                       A(i) attention share per mask
                                                       Muse Spark diagnosis
                                                       Muse Wonder plan / Muse Image counterfactual
                                                     ──► failure replay (SAM 3 lens, Muse Wonder breakdown)
                                                     ◄── "1-Click Apply Muse Fix" / self-correction
```

**A(i) attribution.** Saliency comes from the ACT decoder's cross-attention over the overhead
camera's tokens, with the layer-4 backbone activation norm as a fallback, via forward hooks in
`backend/robot/policy_hook.py`. For each SAM mask *i*, `A(i)` is the share of total saliency mass
that falls inside the mask.

**Anomaly triggers** (`backend/xai_engine/attribution.py`):
- Joint following error above 12° for 0.5 s.
- Action-chunk variance above 25 deg² across overlapping ACT chunks, an out-of-distribution signal.
- Attention deviation: the policy's attention leaves what the demos attended at the same arm pose
  (`calibration/attention_reference.npz`); without that file, a non-target object with `A > 0.45`.
- No task progress for 4 s.

A trigger pauses the arm (HOLD) and waits: the analysis, and with it every Meta API call, starts only when
you press **Analyze** (`GLASSBOX_AUTO_ANALYZE=true` starts it at once). Thresholds live in `backend/config.py`.

**Failure replay** (`backend/xai_engine/replay.py`). On a failure, the 15 s before it (and 2 s after) are
saved to `runs/<id>/` at 15 fps (raw frames, the policy's input frames, ACT saliency, telemetry) and
re-analysed in the background, streaming progress to the console:
1. SAM 3.1 segments 4 keyframes across the run.
2. ACT's recorded attention is attributed per object at 5 Hz.
3. SAM 3 maps where it finds the target's and the goal's phrases in the failure keyframe (~25 s on CPU):
   perception, not attention. When its "red cube" map lands on the book, the scene is ambiguous for the
   task's own words. The breakdown doesn't wait for it.
4. An **attention shift** is the first keyframe where a distractor holds at least 12% of a map
   and more than the target, and still does at the next keyframe.
5. A counterfactual test: Muse Image removes the suspect from the failure frame and ACT is asked where
   it would look and reach without it.
6. Muse Wonder writes the breakdown, tests one hypothesis of its own, and finalises it. With
   `SELF_CORRECT=1`, its automatic interventions are applied immediately.

**Demo staging** (`STAGING=1`). The demo's ACT policy fails reliably only when the red book lies on the
green tile. With staging on, a SAM pre-flight scan before each run finds the book wherever it is, and
every frame the policy sees has the book pasted onto the goal (and erased from where it really is). The
arm still occludes the pasted book. This is labelled **DEMO STAGING** everywhere in the UI and in the
replay, and the Raw camera view shows the real scene. Suppressing the book turns the staging off as well.

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
  - The map selector: ACT live, and SAM 3 · 'red cube' / 'green hex tile'. On a SAM 3 map the tag reads
    `◀ READ AS 'red cube'` when a distractor holds more of the phrase's map than the object it names.
  - The attention-share timeline, with phase bands, events and a draggable playhead.
  - Muse Wonder's breakdown: the perception → decision → actuation stage strip, the culprit, a
    clickable timeline, and the self-correction.
  - Keyboard: Space plays and pauses, ←/→ step a frame, Shift+←/→ jump between keyframes. Saved
    runs can be reopened from the dropdown, or at `/#replay`.
- **Live** is the real-time view:
  - **Overhead view**: live frames with a saliency heatmap, SAM masks tagged `#id label A=…`
    (red when hijacking), suppressed masks drawn dashed, and the recovery path. Toggle between the
    **Policy** view (after suppression and staging) and the **Raw** camera view.
  - **Telemetry scrubber**: joint error, chunk variance and peak servo load with anomaly markers.
    Click the chart to rewind the ring buffer frame by frame.
- **Muse XAI Inspector**:
  - **Trigger Failure Analysis**, then the diagnosis, attribution bars, keyframe and counterfactual.
  - **1-Click Apply Muse Fix**, with an option to override the intervention.
  - The recovery plan.
  - Typed or hold-to-talk questions. Voice commands: "analyze", "apply the fix", "resume",
    "reset", "emergency stop".
- **E-STOP**: disables torque immediately.

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
| `GET /api/replay/{id}/counterfactual.jpg` · `/{id}/check.jpg` `?view=image\|original\|attention_before\|attention_after` | The first counterfactual test and Wonder's own follow-up test |
| `POST /api/replay/{id}/open` · `/api/replay/{id}/apply` | Reload a saved run (re-analysing if needed); apply Wonder's fix |
| `POST /api/ask` `{question}` | Muse Spark Q&A grounded in the latest analysis |
| `GET /api/buffer` · `/api/buffer/{index}.jpg?overlay=true` | Ring-buffer timeline and frames |
| `WS /ws/telemetry` | JSON frame metadata, then a binary JPEG for each frame, plus events (`replay_started`, `replay_progress`, `replay_update`, `staging`, ...). Send `{"view": "policy" \| "raw"}` to choose the view |
| `WS /ws/voice` | Push-to-talk PCM16 16 kHz audio relayed to Muse Voice Transcribe |

## Layout

```
backend/
  main.py               FastAPI app, websockets, voice commands, CLI
  control_loop.py       30 Hz control thread, live attribution, analysis pipeline, interventions
  config.py, schemas.py Settings and Pydantic models shared with the frontend
  robot/                SO-101 (LeRobot), ACT attention hooks, SAM 3 lens worker, calibration
  xai_engine/           attribution + anomaly detector, interventions + demo staging, ring buffer,
                        failure replay (replay.py), rendering
  meta_clients/         SAM 3.1, Muse Spark (triage), Muse Wonder (breakdown, plans), Muse Image + Voice
frontend/src/
  components/           OverheadCanvas, TelemetryScrubber (live); ReplayPlayer, AttentionTimeline,
                        ReplayBreakdown (replay); MuseXAIInspector
  viz.ts, replayMath.ts drawing helpers; keyframe interpolation
  useTelemetry.ts       websocket stream + throttled React state
  useVoice.ts           push-to-talk (AudioWorklet -> PCM16)
calibration/            attention_reference.npz (ACT's attention over its demos); homography + plate
scripts/start_demo.sh   the demo launcher
```

## Notes and limitations

- SAM 3 is a *lens*, not the policy that failed: a larger pretrained model asked where the task's
  objects are in the frame ACT saw. Its maps are segmentations, not attention, and ACT's own attention
  is always shown next to them. When the two agree that the book pulls attention off the cube, the case
  is strong. When they disagree, the breakdown says so.
- SAM 3.1 returns raster masks. Polygons sent to the browser are traced from them.
- Muse Wonder's plan is shown as an auditable routine. Only its joint waypoints are executed,
  after clamping to the arm's calibrated range and velocity limiting. No generated code runs.
- Without an ACT checkpoint the arm holds position while streaming and analysis keep working.
