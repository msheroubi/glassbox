# GlassBox-SO101 documentation

GlassBox gives a learned robot policy something it has never had: the ability to **explain why it
failed and recover on its own**, live, with no retraining. Five Meta models work together:
SAM 3.1, Muse Spark, Muse Wonder, Muse Image and Muse Voice Transcribe.

| | |
|---|---|
| [01 · Overview](01-overview.md) | The problem, who it helps, and what GlassBox makes possible |
| [02 · Architecture](02-architecture.md) | Control loop, analysis pipeline, threading, data flow |
| [03 · How we use Meta's models](03-meta-models.md) | What each model does, what we send it, and how its output is checked |
| [04 · Explainability](04-explainability.md) | Policy saliency, A(i) attention-to-mask attribution, anomaly triggers, the offline failure replay with SAM 3 or SmolVLA |
| [05 · Recovery and safety](05-recovery.md) | Mask suppression, waypoint bypass, grasp offset, validation, fallbacks |
| [06 · Running it](06-running.md) | Install, mock demo, webcam mode, real SO-101, frontend dev, troubleshooting |
| [07 · API reference](07-api.md) | REST and websocket endpoints |
| [08 · Live demo runbook](08-demo-runbook.md) | The 5-minute real-arm demo (ACT fails, SAM 3 replay, Wonder breakdown, self-correction), step by step |
| [09 · Explainer video](09-video.md) | The explainer video, the timed narration script, and how to regenerate it |

**30-second version**

```bash
pip install -r requirements.txt
cp .env.example .env                 # add MODEL_API_KEY=... (optional: everything has a local fallback)
scripts/start_mock.sh attention_hijack
# open http://localhost:8000 and watch the robot get distracted, diagnose itself, and recover

scripts/start_demo.sh                # the real SO-101 demo (see 08)
```
