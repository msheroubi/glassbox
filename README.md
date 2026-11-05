# GlassBox

GlassBox is a diagnostic harness for robot policies, built around **Muse Wonder**, **SAM** and
**Muse Image**. It diagnoses perceptual failures and explains them. When a policy fails, GlassBox gives
Muse Wonder the evidence it needs to work out why. It also gives Wonder tools to test its own hypotheses,
so it can raise its confidence in the diagnosis before it commits to one.

We run it on a LeRobot **SO-101** arm with an ACT pick-and-place policy, filmed by an overhead camera.

Everything was built/trained for this hackathon from scratch.

## How it works

**1. What matters in the scene.** GlassBox records two internal representations of every frame:
- **ACT's attention.** The policy's decoder cross-attention over the camera image, read with forward hooks.
- **SAM 3's perception maps** (`facebook/sam3`, run locally). These show where a perception model finds
  the task's own phrases ("red cube", "green hex tile").

**SAM 3.1** segments the scene into objects, and GlassBox measures each object's share of the attention
over time. This shows which object pulled the policy's attention, and when it happened. It also shows
whether the scene is ambiguous to a perception model at all, for example when "red cube" lights up a red
book.

**2. Counterfactual tests.** SAM 3.1 masks the suspect object, and **Muse Image** removes it from the
frame the policy saw. The policy runs again on the edited frame. If its attention and its reach change,
the object is causally involved, not just correlated.

**3. Diagnosis.** **Muse Wonder** receives the filmstrip, the attention timelines, the counterfactual
result and the servo telemetry. It works out whether the failure started in perception, decision or
actuation, and names the culprit. Before its final answer, Wonder proposes one hypothesis and an
experiment to test it: which object to remove, at which moment, and the effect it predicts. GlassBox runs
that experiment and gives Wonder the result. Wonder's final breakdown then says whether the hypothesis held.

If the fix needs no human (for example, erasing the distractor from the policy's camera input in real
time), GlassBox applies it and resumes the run.

| Meta model | Role |
|---|---|
| **Muse Wonder** | Forensic diagnosis, its own hypothesis tests, recovery plans |
| **SAM 3.1** | Segments the scene by concept: the masks for attention shares and counterfactual edits |
| **SAM 3** (local) | Perception maps of the task's phrases, alongside the policy's attention |
| **Muse Image** | Counterfactual frames with an object removed |
| **Muse Spark** | Fast live triage and Q&A |
| **Muse Voice** | Push-to-talk commands and questions |

## Setup

```bash
pip install "lerobot[feetech]" -r requirements.txt   # Python 3.10; LeRobot brings torch
cp .env.example .env                                 # MODEL_API_KEY=...

# SAM 3 needs transformers >= 5 (LeRobot pins 4.x), so it runs as a worker in its own venv.
# facebook/sam3 is gated on Hugging Face: accept its licence first.
python -m venv --system-site-packages .venv-sam3
.venv-sam3/bin/pip install "transformers>=5" accelerate pillow
.venv-sam3/bin/hf auth login
```

Without an API key, every Meta call falls back to a local stand-in.

## Running

```bash
PYTHON=~/miniconda3/envs/lerobot/bin/python scripts/start_demo.sh
```

Open http://localhost:8000. Saved failures can be replayed at `/#replay`. The script builds the console
on first run (this needs `npm`). It is configured with environment variables:

| Variable | Default | |
|---|---|---|
| `POLICY` | `../models/last2/pretrained_model` | ACT checkpoint |
| `ROBOT_PORT` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46082778-if00` | Follower arm |
| `ROBOT_ID` | `kowalski_follower` | LeRobot calibration id |
| `CAMERA` | `2` | Overhead webcam |
| `STAGING` | `1` | Show the policy the red book on the goal tile (labelled **DEMO STAGING** in the UI) |
| `AUTOSTART` | `0` | Wait for **▶ Start policy** |
| `FORCE_FAILURE` | `1` | A manual Analyze counts as a failure |
| `SELF_CORRECT` | `1` | Apply Wonder's fix automatically |

Any field in `backend/config.py` can also be set as a `GLASSBOX_*` environment variable.

## Layout

```
backend/
  control_loop.py   30 Hz control loop, anomaly detection, live analysis, interventions
  xai_engine/       attention attribution, failure replay, counterfactuals, interventions
  meta_clients/     SAM 3.1, Muse Wonder, Muse Image, Muse Spark, Muse Voice
  robot/            SO-101 driver, ACT attention hooks, SAM 3 worker
frontend/           React console: live view, failure replay, Wonder's breakdown
scripts/start_demo.sh
```
