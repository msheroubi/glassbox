# 07 · API reference

The server is FastAPI (`backend/main.py`), with interactive docs at **http://localhost:8000/docs**.
Request and response models are in `backend/schemas.py`, mirrored in `frontend/src/types.ts`.

## REST

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/status` | | Run state, mode, models, thresholds, active interventions, `analysis_running`, fps |
| POST | `/api/analyze` | | `AnalysisResult`. Runs SAM, A(i), Spark (and Wonder/Image async) on the latest frames. `409` if one is already running |
| GET | `/api/analysis/latest` | | `AnalysisResult` |
| GET | `/api/analysis/{id}` | | `AnalysisResult` |
| GET | `/api/analysis/{id}/keyframe.jpg` | | The composite image Spark saw |
| GET | `/api/analysis/{id}/counterfactual.jpg` | | The Muse Image counterfactual (`404` until ready) |
| POST | `/api/analysis/{id}/counterfactual` | | Re-render the counterfactual |
| POST | `/api/apply_fix` | `{"analysis_id", "intervention_type"?}` | `{ok, intervention_type, message}`. `409` during E-STOP |
| POST | `/api/clear_interventions` | | Status. Stops suppression, trajectories and grasp offsets |
| POST | `/api/reset` | `{"scenario"?}` | Status. Restarts the scenario (mock) or clears the session |
| POST | `/api/estop` | | Status. Torque off |
| POST | `/api/resume` | | Status. Torque on, policy resumes |
| POST | `/api/ask` | `{"question"}` | `{question, answer, source}`. Muse Spark, grounded in the latest evidence |
| GET | `/api/buffer` | | Ring-buffer timeline: per-frame `index, t, phase, joint_error, chunk_variance, max_load` |
| GET | `/api/buffer/{index}.jpg?overlay=true` | | A past frame with its saliency heatmap |
| GET | `/api/buffer/{index}/telemetry` | | That frame's `ServoTelemetry` |

## `AnalysisResult`

```jsonc
{
  "analysis_id": "407ef275", "created_at": 1793..., "trigger": "auto",
  "anomaly": {"reasons": ["action-chunk variance 81.1 > 25.0 (OOD confusion)"], "joint_error": 0.0, ...},
  "objects": [{"mask_id": 1, "label": "distractor", "prompt": "red ball", "confidence": 0.9, "bbox": [...], "polygons": [...]}],
  "attributions": [{"mask_id": 1, "label": "distractor", "attention_share": 0.5775, "workspace_xy_mm": [235.5, 120.2], ...}],
  "diagnosis": {"failure_type": "ATTENTION_HIJACK", "culprit_mask_id": 1, "confidence": 0.92,
                "headline_summary": "...", "technical_root_cause": "...",
                "intervention_type": "INPAINT_MASK_SUPPRESSION", "intervention_parameters": {"mask_id": 1, "dilate_px": 5}},
  "diagnosis_source": "meta-api", "sam_source": "meta-api",
  "plan": null, "plan_source": null, "plan_status": "not_needed",   // WAYPOINT_BYPASS: "pending" -> "ready"
  "plan_path_px": null,
  "counterfactual_url": "/api/analysis/407ef275/counterfactual.jpg", "counterfactual_source": "meta-api",
  "keyframe_url": "/api/analysis/407ef275/keyframe.jpg",
  "latency_ms": {"sam": 1759, "spark": 6612, "image": 13314},
  "evidence": { /* the exact JSON Muse Spark was given */ }
}
```

## WebSocket `/ws/telemetry`

When a client connects, the server sends `{"type": "status"}`, plus the latest `analysis` if
there is one. After that it streams:

- **Frames**: a JSON header `{"type": "frame", …}` immediately followed by **one binary JPEG**. The
  header carries:
  - `index`, `t`, `state`, `phase`, `status`, `scenario`, `fps`;
  - `telemetry` (`ServoTelemetry`);
  - `saliency` (`{w: 64, h: 36, method, data: base64 uint8}`);
  - `objects` (live SAM/ground-truth polygons);
  - `suppression`, `recovery`, `grasp_offset_mm`.

  Frames are coalesced per client at `stream_fps`.
- **Events** (JSON only):

| `type` | When |
|---|---|
| `anomaly` | A trigger fired and the arm is in HOLD (`AnomalyEvent`) |
| `analysis_started` | The pipeline began (`trigger`: auto, manual, voice) |
| `analysis` | Diagnosis ready (`result`: `AnalysisResult`) |
| `analysis_update` | Wonder's plan or Muse Image's counterfactual arrived (same `result`, updated) |
| `fix_applied` | `analysis_id`, `intervention_type`, `message` |
| `state` | `state` (RUNNING, HOLD, ESTOP, COMPLETE) and `status` text |
| `reset` | Scenario restarted |

Client to server: `{"view": "policy"}` (default; the frame the policy sees, with suppression
applied) or `{"view": "raw"}`.

## WebSocket `/ws/voice`

| Client sends | Server replies |
|---|---|
| `{"type": "start"}` | Opens a Muse Voice Transcribe session, then replies `{"type": "listening"}` |
| binary PCM16, 16 kHz mono | `{"type": "transcript", "text", "final"}` as partials arrive |
| `{"type": "stop"}` | Flushes. Then one of `{"type": "command", "command", "message"}` or `{"type": "answer", "question", "answer", "source"}`, followed by `{"type": "done"}` |
| `{"type": "text", "text"}` | Typed fallback through the same command path |

## Scripting example

```python
import httpx
c = httpx.Client(base_url="http://localhost:8000", timeout=120)
c.post("/api/reset", json={"scenario": "attention_hijack"})
res = c.post("/api/analyze").json()
print(res["diagnosis"]["headline_summary"], res["latency_ms"])
print(c.post("/api/ask", json={"question": "Why did it stop?"}).json()["answer"])
c.post("/api/apply_fix", json={"analysis_id": res["analysis_id"]})
```

`docs/video/capture.py` is a complete client: it records sessions over the websocket for the
explainer video.
