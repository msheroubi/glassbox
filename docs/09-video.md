# 09 · Explainer video

**[`video/glassbox_explainer.mp4`](video/glassbox_explainer.mp4)**: 1920×1080, 30 fps, 3 min 28 s,
with burnt-in captions and no audio track.

The timed narration script is in **[`video/narration.md`](video/narration.md)**. It is generated
with the video, so its timestamps match the captions exactly. Read it as a voice-over, or let the
captions carry the video on their own.

## What is real in it

Every model output on screen is a **recorded response from the Meta Model API**, captured from
the running system:

- the SAM 3.1 masks and confidences;
- every Muse Spark diagnosis, root cause and answer;
- Muse Wonder's waypoints, routine and safety notes;
- the Muse Image counterfactual;
- the latencies in the pipeline rows.

Nothing is typed into the video by hand. The renderer reads the same `AnalysisResult` JSON that
the console receives.

Only the pacing is edited. The arm waits on **HOLD** while the models think (10–40 s), and those
stretches are fast-forwarded. A `FAST-FORWARD N×` tag and the real HOLD duration are shown on
screen whenever that happens.

## Scenes

| Time | Scene | What it shows | Models on screen |
|---|---|---|---|
| 0:00 | Title | The question: can a robot say why it failed and how to fix it | |
| 0:07 | 01 · Learned policies fail silently | The SO-101 task, and why retraining is the usual answer | |
| 0:19 | 02 · How GlassBox works | The 30 Hz loop with its four triggers. HOLD, then the analysis pipeline, then the validated fix | all five |
| 0:35 | 03 · Five Meta models, five jobs | One card per model, with its role and real latency | all five |
| 0:49 | 04 · Attention hijack | A red ball steals 58 % of the policy's attention. Spark diagnoses it at 0.92, Muse Image renders the counterfactual, a spoken question gets a grounded answer, and the 1-click fix erases the ball from the policy's input | SAM 3.1, Spark, Image, Voice |
| 1:31 | 05 · Servo stall | A box blocks the reach. Telemetry spikes and Spark returns SERVO_STALL_COLLISION at 0.93. Wonder's verified 3-waypoint bypass and its routine follow, then the recovery | SAM 3.1, Spark, Wonder, Image |
| 2:06 | 06 · Specular glare | A 16.2 mm perceived-grasp bias, measured against SAM's mask centroid. Spark's offset fix makes the pick | SAM 3.1, Spark |
| 2:28 | 07 · Occlusion | A hand covers the cube. Spark returns OCCLUSION_OOD with intervention NONE: knowing when not to act | SAM 3.1, Spark |
| 2:43 | 08 · Real desk | The overhead USB webcam over the real SO-101: SAM masks "robotic arm" and "red cube", and Spark's reading of the scene | SAM 3.1, Spark |
| 3:01 | 09 · Built to run on a real robot | The 30 Hz loop, the triggers, schema validation, swept-path checks, offline fallbacks and the replay buffer | |
| 3:17 | Close | Who it is for, and the five models | all five |

## Regenerating it

There are two steps: record real sessions, then render.

### 1. Capture

Start a server with a valid `MODEL_API_KEY` (see [06](06-running.md)). Then record each scenario.
The capture script drives the public API: reset, wait for the anomaly and the automatic analysis,
ask a question, apply the fix, and record the recovery.

```bash
PORT=8020 scripts/start_mock.sh nominal &        # wait for http://localhost:8020/api/status
python docs/video/capture.py --base http://localhost:8020 \
    nominal attention_hijack servo_stall specular_glare occlusion

PORT=8021 scripts/start_webcam.sh 2 &            # optional: the real-desk segment
python docs/video/capture.py --base http://localhost:8021 --webcam
```

Sessions land in `.cache/video_capture/<name>/`:

| File | Contents |
|---|---|
| `frames/` | Every websocket frame as JPEG |
| `frames.jsonl` | Per-frame telemetry, saliency and live masks |
| `events.jsonl` | Anomaly, analysis and fix events with timestamps |
| `analysis.json` | The final `AnalysisResult` |
| `keyframe.jpg` | The composite image Spark saw |
| `counterfactual.jpg` | The Muse Image output |
| `ask.json` | The question asked, and Spark's answer |

### 2. Render

```bash
PYTHONPATH=. python docs/video/make_video.py                  # full video + narration.md (a few minutes)
PYTHONPATH=. python docs/video/make_video.py --preview stall 24   # one frame -> /tmp/glassbox_preview.png
PYTHONPATH=. python docs/video/make_video.py --only hijack stall --out /tmp/part.mp4
```

The renderer needs `ffmpeg` with libx264, plus Pillow, OpenCV and NumPy (all in the project
environment). Scene timings, captions and layouts are in `build_scenes()` in `make_video.py`.
Each scene maps its own clock onto the recorded session through a piecewise `Clip`, which is
where the fast-forwarding happens.
