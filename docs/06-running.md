# 06 · Running GlassBox

## Requirements

- Python 3.10+.
- Node 18+ and npm. Only needed to build the console once; the backend then serves it.
- Optional: a Meta Model API key. Without one, every model runs on its local fallback, and the UI
  shows `local-fallback` chips.
- Optional: an SO-101 follower arm with a Feetech bus, an overhead USB camera, and a trained
  LeRobot ACT checkpoint.

## Install

```bash
git clone <this repo> glassbox-so101 && cd glassbox-so101
pip install -r requirements.txt            # server + XAI engine (mock and webcam modes)
pip install "lerobot[feetech]"             # only for the real arm + ACT policy (brings torch)
cp .env.example .env                       # then set MODEL_API_KEY=... in .env
```

`.env` is gitignored. Never commit it. The key is read by `backend/config.py` and is only ever
sent to `api.meta.ai`.

The first run of any `scripts/start_*.sh` builds the console (`frontend/dist`) if npm is available.

## Five ways to run

All five serve the console and API on **http://localhost:8000**. Set `PORT=8010` to change the
port and `PYTHON=/path/to/python` to choose the interpreter.

### 1. Simulator: no hardware needed

```bash
scripts/start_mock.sh attention_hijack     # or: nominal | servo_stall | occlusion | specular_glare
```

This runs a simulated SO-101 with a rendered overhead camera and a pixel-driven pick-and-place
policy. Each scenario fails in a different, physically meaningful way. The **Scenario** dropdown in
the console switches between them live.

| Scenario | What goes wrong | Expected diagnosis → fix |
|---|---|---|
| `nominal` | Nothing: the cube lands on the green hex tile | None |
| `attention_hijack` | A red ball shares the policy's colour cue | `ATTENTION_HIJACK` → erase the ball from the policy's view |
| `servo_stall` | A box sits on the reach path and the gripper jams | `SERVO_STALL_COLLISION` → Wonder/local bypass trajectory |
| `occlusion` | An operator's hand covers the cube | `OCCLUSION_OOD` → operator clears, policy resumes |
| `specular_glare` | A highlight biases the grasp point | `SPECULAR_GLARE` → grasp offset |

### 2. Camera only: your real desk, no arm

```bash
scripts/start_webcam.sh 2                  # camera index (or /dev/video2)
scripts/start_webcam.sh 2 --concepts "target=red cube,gripper=robot arm,coffee mug,keyboard"
```

This streams a real USB webcam through the same SAM 3.1, A(i) and Muse Spark pipeline. With no
policy or servos connected:
- saliency is spectral-residual visual saliency, and the UI says so;
- the anomaly triggers are off;
- **Analyze** and the chat are on demand.

It is a good way to check that SAM finds your objects before connecting the arm.

To find the camera index:

```bash
python -c "import cv2; [print(i, cv2.VideoCapture(i).read()[0]) for i in range(5)]"
```

### 3. Camera only with a pretrained VLA's attention (SmolVLA, observe only)

```bash
pip install "lerobot[smolvla]"             # num2words + accelerate on top of lerobot
scripts/start_vla.sh 2                     # action-expert attention (default)
scripts/start_vla.sh 2 --vla-attention language
VLA=path/to/finetuned_smolvla scripts/start_vla.sh 2
```

This mode is the same as camera-only mode, except that the heatmap comes from a real
vision-language-action model instead of spectral-residual saliency. `lerobot/smolvla_base` (about 1 GB,
downloaded on first run) is fed the overhead frame, the `--task` text and the arm's rest pose. **Its actions are never sent anywhere**: no arm is connected in this mode, and the
predicted chunk is only kept for inspection.

| `--vla-attention` | Map |
|---|---|
| `action` | Where the action expert looks while it denoises the 50-step action chunk: attention from its queries to the image tokens, averaged over layers, heads and the 10 denoising steps |
| `language` | Where the instruction's tokens look inside the VLM, i.e. what the model grounds the task text to |

- Both maps are divided by the same map for a flat grey frame. SmolVLM puts several times the uniform share
  on the right-hand token column whatever the image shows, and the ratio removes that.
- The map is 8x8 tokens, so it is coarse.
- On CPU a pass takes about 6 s. It runs on a worker thread, so the stream stays at 30 fps and the heatmap
  refreshes every ~6 s. Set `GLASSBOX_POLICY_DEVICE=cuda` if you have a GPU.
- The base model has not been fine-tuned on this task. The action map follows the cube, the tile, and
  red decoys placed near either one. The language map is weaker and often settles on the gripper.

### 4. Real SO-101 with an ACT policy

```bash
# once: table-to-pixel homography + empty-table background plate (used by mask suppression)
python -m backend.robot.calibration --camera 0 --world "0,320;0,-320;360,-320;360,320"

scripts/start_robot.sh models/last/pretrained_model --robot-port /dev/ttyACM0 --camera /dev/video2
```

- The policy's camera keys must match the ones it was trained with: `GLASSBOX_OVERHEAD_CAMERA_KEY`
  (default `scene`) and `GLASSBOX_WRIST_CAMERA_KEY` (default `wrist`).
- `--camera` takes an index or a path such as `/dev/video2`. `start_robot.sh` captures it at 640x360 and
  turns the wrist camera off by default. Override with `--camera-size WxH` and `--wrist-camera INDEX_OR_PATH`.
  LeRobot rejects a capture size the camera can't deliver (`failed to set capture_width`), so match the
  size your policy was trained at and your webcam supports.
- Without `--policy-path`, the arm holds position while streaming and analysis still work.

### Failure demos on the real arm

Each one is set up physically while the policy runs. The HOLD status line names the trigger that fired.

| Set-up | Trigger that should fire | Expected diagnosis → fix |
|---|---|---|
| A second red object (the decoy) on the green tile | attention deviation (unexpected attention on the decoy), variance or stall | `ATTENTION_HIJACK` → erase the decoy from the policy's view; the arm then runs on a clean tile |
| A hand over the cube mid-reach | variance | `OCCLUSION_OOD` → take the hand away, then **Apply** to resume |
| A box on the reach path | servo (error while the arm is stuck) | `SERVO_STALL_COLLISION` → Wonder bypass waypoints |

SAM can only attribute attention to things it segments, so name the decoy as its own concept. Any role other
than `target`, `gripper`, `goal` and `hand` counts as a distractor:

```bash
scripts/start_robot.sh models/last/pretrained_model --camera /dev/video2 --robot-id kowalski_follower \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46082778-if00 \
  --concepts "target=red cube,gripper=robot arm,goal=green hex tile,decoy=red cup,hand=human hand,obstacle=cardboard box"
```

Use a phrase that matches the decoy but not the cube (`red cup`, `red ball`, `red marker`). If both match
`red cube`, the decoy is labelled `target` and the hijack can't be attributed. Run
`scripts/start_webcam.sh 2 --concepts ...` first to check the masks. Capture the background plate
(`calibration.py` above) so suppression shows the empty tile rather than an inpainted smear.

### 5. The demo: ACT live, SAM 3 failure replay, self-correction

```bash
pip install "lerobot[feetech,smolvla]"
# SAM 3 runs locally in its own venv (it needs transformers 5; lerobot is pinned to 4.x). It shares the
# lerobot env's torch, numpy and OpenCV. facebook/sam3 is gated: request access on Hugging Face first,
# then `hf auth login`.
"$(python -c 'import sys; print(sys.executable)')" -m venv --system-site-packages .venv-sam3
.venv-sam3/bin/pip install "transformers>=5.5,<6"
scripts/start_demo.sh                       # LENS=smolvla / AUTOSTART=1 / STAGING=0 / FORCE_FAILURE=0 / SELF_CORRECT=0 / CAMERA=... / POLICY=... to change
```

This is mode 4 with the defaults set for this rig: `../models/last/pretrained_model`, the follower at
`/dev/serial/by-id/...5A46082778-if00` (`kowalski_follower`), and overhead camera 2 at 640x360. It adds:
- `--replay-sam3`: failures are recorded and replayed offline. Next to ACT's attention, SAM 3 shows where it
  finds "red cube" and "green hex tile" in each keyframe (see [04 §6](04-explainability.md#6-failure-replay-a-second-slower-look-at-the-run)).
  It takes about 25 s per keyframe on a CPU and ~3.5 GB of RAM. `LENS=smolvla` uses `--replay-vla
  lerobot/smolvla_base` instead (SmolVLA's attention, ~6 s per keyframe), and `LENS=none` leaves ACT's map only;
- `--demo-distractor-on-goal`: **demo staging**. The policy is shown the red book on the green tile
  wherever the book really is, so the overfit ACT checkpoint fails on cue. It is labelled in the UI.
  Use `STAGING=0` when you put the book on the tile yourself;
- `--start-paused`: the policy doesn't run until you press **▶ Start policy** in the console, after launch
  and after each **↺ Reset**. Meanwhile, the arm holds still while ACT, the cameras, the SAM pre-flight scan and the replay lens load.
  A line under the controls shows what is still loading. Use `AUTOSTART=1` to run at once;
- `--demo-manual-failure`: pressing **Analyze** (or saying "analyze") when nothing has been detected counts
  as a failure. The arm holds, and the run is diagnosed, replayed and self-corrected like a detected failure.
  The anomaly reasons start with `DEMO: failure triggered by the operator`. If Spark finds no failure, the
  distractor holding the most attention is diagnosed, and the counterfactual test still judges it. Use `FORCE_FAILURE=0` to turn it off;
- `--self-correct`: Muse Wonder's fix is applied automatically when it says no human is needed.

The default SAM concepts already name the book (`distractor` = "red book", with "red notebook" and
"small red book" as fallbacks), the cube (`target`), the tile (`goal`), the arm and a hand. Check them
once with `scripts/start_webcam.sh 2`. The walkthrough is in [08](08-demo-runbook.md).

SAM rates the cube as a "red book" almost as confidently as a "red cube", and the other way round.
So on the real camera, each red object's colour also votes for its label: the book is pinker
(LAB b* ≈ 134 vs ≈ 155 for the cube). The reference colours (`appearance_refs` in `backend/config.py`)
were measured on this rig. Under different lighting, such as at the venue, run the arm once with both
objects in view and re-measure:
`PYTHONPATH=. python scripts/measure_appearance.py runs/<id>`. It prints a
`GLASSBOX_APPEARANCE_REFS=...` line for `.env`. When an object's colour is far from both references,
the vote abstains and SAM's own labels stand.

Each failure is saved under `runs/<id>/` (roughly 10–15 MB at 640x360). Saved runs can be replayed with no arm
connected: `scripts/start_mock.sh` or `start_webcam.sh` and the run dropdown on the replay tab, or
`POST /api/replay/<id>/open`.

## Command-line options

`python -m backend.main [options]`, which the scripts wrap:

| Option | |
|---|---|
| `--mock-hardware` / `--scenario NAME` | Simulator mode and starting scenario |
| `--webcam INDEX` | Camera-only mode |
| `--vla PATH`, `--vla-attention action\|language` | With `--webcam`: SmolVLA checkpoint whose attention is shown (observe only) and which map |
| `--policy-path PATH` | ACT checkpoint (local dir or HF repo id) |
| `--robot-port`, `--robot-id` | SO-101 serial port and calibration id |
| `--camera`, `--camera-size`, `--wrist-camera` | Real arm: overhead camera (index or path), capture size, wrist camera or `none` |
| `--concepts "role=phrase,…"` | SAM 3.1 concepts. Roles: `target`, `gripper`, `obstacle`, `goal`, `hand`; other labels are potential distractors |
| `--task "…"` | Task description given to Muse Spark |
| `--replay-sam3` | Replay failures through SAM 3 (`facebook/sam3`, local, in `.venv-sam3`): where it finds the target's and goal's phrases. Perception, not attention |
| `--replay-vla PATH` | Replay failures through this VLA's attention offline (e.g. `lerobot/smolvla_base`); observe only. Alternative to `--replay-sam3` |
| `--no-replay` | Live triage only: don't record and replay failed runs |
| `--demo-distractor-on-goal` | Demo staging: show the policy the distractor on the goal tile wherever it is (labelled in the UI) |
| `--start-paused` | Wait in HOLD for **▶ Start policy** after launch and after each Reset, instead of running the policy as soon as it loads |
| `--demo-manual-failure` | Demo: a manual Analyze with no anomaly on record is treated as a failure (held, replayed, self-corrected; labelled `DEMO` in the reasons) |
| `--self-correct` | Apply Muse Wonder's intervention automatically when it says no human is needed (also a switch in the console) |
| `--no-auto-analyze` | Detect anomalies and HOLD, but wait for the **Analyze** button |
| `--disable-triggers LIST` | Turn off anomaly triggers (`servo`, `variance`, `deviation`, `hijack`, `stall`) so the arm keeps running. The HOLD status line and the backend log (`Anomaly: ...`) name the trigger that fired |
| `--host`, `--port`, `--log-level` | Server |

## Configuration

Every field of `backend/config.py: Settings` can be set in `.env` or the environment with a
`GLASSBOX_` prefix. Useful ones:

| Variable | Default | |
|---|---|---|
| `GLASSBOX_SPARK_REASONING_EFFORT` | `low` | Muse Spark effort, for diagnosis latency |
| `GLASSBOX_WONDER_REASONING_EFFORT` | `high` | Muse Wonder effort |
| `GLASSBOX_SPARK_TIMEOUT_S` | `30` | Per-attempt Spark timeout before retry, then local fallback |
| `GLASSBOX_JOINT_ERROR_THRESHOLD_DEG` | `12` | Servo-error trigger |
| `GLASSBOX_CHUNK_VARIANCE_THRESHOLD` | `25` | Action-chunk variance trigger |
| `GLASSBOX_ATTENTION_HIJACK_THRESHOLD` | `0.45` | `A(i)` hijack trigger when there is no attention reference (with one, the trigger is A(i) minus the demos' share, see `GLASSBOX_ATTENTION_HIJACK_EXCESS_THRESHOLD`, default `0.15`) |
| `GLASSBOX_ATTENTION_DEVIATION_THRESHOLD` | calibrated | Fraction of unexpected attention that triggers (default: the value in `attention_reference.npz`) |
| `GLASSBOX_USE_SAM_VIDEO` | `false` | Segment the 90-frame clip instead of the keyframe |
| `GLASSBOX_LIVE_SAM_INTERVAL_S` | `4` | Hardware: background SAM pass for the live hijack trigger |
| `GLASSBOX_TRACK_REFRESH_S` | `3` | Re-segment suppressed objects this often |
| `GLASSBOX_STREAM_FPS` | `15` | Browser frame rate (the control loop stays at 30 Hz) |
| `GLASSBOX_POLICY_DEVICE` | `cpu` | `cuda` / `mps` for ACT |
| `GLASSBOX_POLICY_N_ACTION_STEPS` | `100` | Steps executed from each ACT chunk (the whole chunk, as in `lerobot-record`) |
| `GLASSBOX_POLICY_OBSERVE_EVERY` | `20` | Re-run ACT every N steps without acting on it, to refresh the attention map and the chunk-variance signal (`0` = off) |
| `GLASSBOX_REPLAY_WINDOW_S` | `12` | Seconds of run before the failure that are recorded and replayed |
| `GLASSBOX_REPLAY_VLA_HZ` | `1` | SmolVLA keyframes per second of run (about 6 s each on CPU) |
| `GLASSBOX_REPLAY_COARSE_KEYFRAMES` | `5` | Muse Wonder starts once this many SmolVLA keyframes are done |
| `GLASSBOX_REPLAY_SAM3_HZ` | `0.5` | SAM 3 keyframes per second of run (about 25 s each on CPU) |
| `GLASSBOX_REPLAY_SAM3_COARSE_KEYFRAMES` | `3` | Muse Wonder starts once this many SAM 3 keyframes are done |
| `GLASSBOX_SAM3_PYTHON` | `.venv-sam3/bin/python` | Interpreter the SAM 3 worker runs in (transformers >= 5) |
| `GLASSBOX_SAM3_MODEL` | `facebook/sam3` | SAM 3 checkpoint |
| `GLASSBOX_REPLAY_PRIMARY_MAP` | `vla_language` | Map in Wonder's filmstrip: `vla_language`, `vla_action`, `act` or `sam3_target` (falls back in that order) |
| `GLASSBOX_RUNS_DIR` | `runs/` | Where failed runs are saved |
| `GLASSBOX_VLA_CAMERA_KEY` | `camera1` | SmolVLA image input the overhead frame is fed as |

## Frontend development

```bash
scripts/start_mock.sh &                    # backend on :8000
scripts/start_frontend_dev.sh              # Vite on :5173 with hot reload, proxying /api and /ws
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Chips say `local-fallback` | No `MODEL_API_KEY`, or the API is unreachable. Check the server log for `401` or timeouts |
| Chips say `cache` | The request matched one replayed from `.cache/meta_api/`. Delete that folder to force live calls |
| Webcam shows "no signal" | Wrong index, or another app holds the camera. Try the index probe above |
| SAM doesn't find the arm on a real camera | Camera-only mode tries "robot arm" then "robotic arm". With `--concepts`, try both phrases |
| Replay stuck at "refining" | The replay lens pauses while the arm runs. Press **❚❚ Hold** (or E-STOP) to let it finish |
| No staging badge after Reset | The pre-flight SAM scan didn't find the book or the tile. Check the concepts with `start_webcam.sh`; the status line says what it found |
| `409: An analysis is already running` | Wait for it. Only one analysis runs at a time |
| Port already in use | Stop the old server with `pkill -f backend.main` or `fuser -k 8000/tcp` |
| Muse Voice button errors | The browser needs microphone permission, on `localhost` or HTTPS. The typed command box goes through the same path |
