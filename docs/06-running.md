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

## Three ways to run

All three serve the console and API on **http://localhost:8000**. Set `PORT=8010` to change the
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
| `nominal` | Nothing: the cube goes in the bin | None |
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

### 3. Real SO-101 with an ACT policy

```bash
# once: table-to-pixel homography + empty-table background plate (used by mask suppression)
python -m backend.robot.calibration --camera 0 --world "0,320;0,-320;360,-320;360,320"

scripts/start_robot.sh outputs/train/act_so101/checkpoints/last/pretrained_model --robot-port /dev/ttyACM0
```

- The policy's camera keys must match the ones it was trained with: `GLASSBOX_OVERHEAD_CAMERA_KEY`
  (default `scene`) and `GLASSBOX_WRIST_CAMERA_KEY` (default `wrist`).
- Without `--policy-path`, the arm holds position while streaming and analysis still work.

## Command-line options

`python -m backend.main [options]`, which the scripts wrap:

| Option | |
|---|---|
| `--mock-hardware` / `--scenario NAME` | Simulator mode and starting scenario |
| `--webcam INDEX` | Camera-only mode |
| `--policy-path PATH` | ACT checkpoint (local dir or HF repo id) |
| `--robot-port`, `--robot-id` | SO-101 serial port and calibration id |
| `--concepts "role=phrase,…"` | SAM 3.1 concepts. Roles: `target`, `gripper`, `obstacle`, `hand`; other labels are potential distractors |
| `--task "…"` | Task description given to Muse Spark |
| `--no-auto-analyze` | Detect anomalies and HOLD, but wait for the **Analyze** button |
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
| `GLASSBOX_ATTENTION_HIJACK_THRESHOLD` | `0.45` | `A(i)` hijack trigger |
| `GLASSBOX_USE_SAM_VIDEO` | `false` | Segment the 90-frame clip instead of the keyframe |
| `GLASSBOX_LIVE_SAM_INTERVAL_S` | `4` | Hardware: background SAM pass for the live hijack trigger |
| `GLASSBOX_TRACK_REFRESH_S` | `3` | Re-segment suppressed objects this often |
| `GLASSBOX_STREAM_FPS` | `15` | Browser frame rate (the control loop stays at 30 Hz) |
| `GLASSBOX_POLICY_DEVICE` | `cpu` | `cuda` / `mps` for ACT |

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
| `409: An analysis is already running` | Wait for it. Only one analysis runs at a time |
| Port already in use | Stop the old server with `pkill -f backend.main` or `fuser -k 8000/tcp` |
| Muse Voice button errors | The browser needs microphone permission, on `localhost` or HTTPS. The typed command box goes through the same path |
