# 02 · Architecture

```
                         ┌───────────────────────── 30 Hz control thread ──────────────────────────┐
 overhead camera ──────► │ read obs ─► mask suppressor ─► policy (ACT / mock / camera observer)   │
 servo bus (Feetech) ◄── │            (policy input only)     │ action chunk, saliency, variance  │
                         │                                    ▼                                   │
                         │   waypoint executor ─► clamp_joints ─► robot.send()                    │
                         │   telemetry ─► ring buffer (300 frames = 10 s) ─► Hub ─► /ws/telemetry │
                         │   live A(i) every 3 ticks ─► AnomalyDetector ──► HOLD                  │
                         └──────────────────────────────────────────┬─────────────────────────────┘
                                                                    │ run_coroutine_threadsafe
                         ┌──────────────── asyncio analysis pipeline (FastAPI loop) ───────────────┐
                         │ SAM 3.1 (one request per concept, concurrent) ─► masks                 │
                         │ A(i) attribution + grasp-bias estimate ─► composite keyframe           │
                         │ Muse Spark (image + evidence JSON, json_schema) ─► diagnosis           │
                         │   ├─ WAYPOINT_BYPASS: local planner now, Muse Wonder refines async     │
                         │   └─ Muse Image counterfactual (async)                                 │
                         │ operator: 1-Click Apply Muse Fix ─► intervention ─► RUNNING            │
                         │ SAM 3.1 re-tracks suppressed masks every 3 s                           │
                         └─────────────────────────────────────────────────────────────────────────┘
                                                   ▲  ▲
 React console (Vite, TypeScript) ── REST /api/* ──┘  └── /ws/voice ◄─► Muse Voice Transcribe (wss)
```

## Two worlds, one lock

- **The control thread** (`backend/control_loop.py: ControlLoop._run/_tick`) runs at a fixed
  30 Hz. Hardware I/O, the PyTorch policy, the suppressor and the trajectory executor are all
  blocking and latency-sensitive, so they live here and never wait on the network.
- **The analysis pipeline** (`ControlLoop.analyze`) runs as coroutines on FastAPI's event loop.
  It makes every Meta API call, so a 10 s Muse Spark call never delays a servo command.
- They meet in two places.
  - `self._lock` guards state changes. Every operator command (apply fix, reset, E-STOP, resume)
    takes it, so a command never lands halfway through a tick.
  - The **Hub** passes data from the thread to the websocket. It keeps one latest-frame slot and
    an event queue per browser client, so a slow browser gets *coalesced* frames instead of a
    growing backlog. Frames go out at 15 fps by default (`stream_fps`) while the loop keeps
    running at 30 Hz.

## One tick

`ControlLoop._tick()`:

1. `robot.read()` runs outside the lock, because a camera read can take a whole frame period. It
   returns the overhead frame, joint positions, velocities and servo loads.
2. If suppression is active, the policy gets `suppressor.apply(raw)`. The raw frame is still what
   is recorded.
3. If a recovery trajectory is active, the next waypoint target wins. Otherwise the policy steps
   and returns its action, `saliency` and `chunk_variance`.
4. `clamp_joints()` limits the target to the joint limits, then `robot.send()` sends it.
5. A `ServoTelemetry` record is built: target and actual qpos, qvel, load, joint error and chunk
   variance.
6. Every 3 ticks, live `A(i)` is recomputed from the latest masks: SAM's background pass on
   hardware, simulator ground truth in mock mode.
7. `AnomalyDetector.update()` runs. If it fires, the arm goes to HOLD and `analyze("auto")` is
   scheduled on the event loop.
8. The frame is JPEG-encoded once into the ring buffer, and a frame packet is published:
   telemetry, a 64×36 saliency grid, object polygons and intervention state.

## The analysis pipeline

`ControlLoop.analyze()`:

| Step | What happens | Typical latency |
|---|---|---|
| Freeze | HOLD the arm, take the newest ring-buffer frame and its saliency | under 1 ms |
| Segment | SAM 3.1, one concurrent streaming request per concept (gripper, target, distractor, obstacle, hand), parsed with `meta-sam-parser` | 1 to 2 s |
| Attribute | `A(i)` per mask, workspace millimetres via the homography, IoU with the gripper, and a grasp-bias estimate from the clearest recent frame | under 10 ms |
| Composite | Side-by-side keyframe: raw on the left; SAM contours labelled `#id label A=…` plus the jet heatmap on the right | under 10 ms |
| Diagnose | Muse Spark gets the keyframe and the evidence JSON, and returns a schema-validated `XAIDiagnosisResponse` | 2 to 10 s |
| Plan *(bypass only)* | The local visibility-graph planner answers immediately. Muse Wonder refines it asynchronously, and its plan is validated before replacing the baseline | 0 s, then 30 to 120 s |
| Counterfactual | Muse Image edits the anomaly frame to show "what if the culprit were gone" (async) | 5 to 30 s |

The browser receives an `analysis` event as soon as the diagnosis is ready. `analysis_update`
events follow when Wonder or Muse Image finish, so the operator never waits on the slowest model.

## Data contracts

`backend/schemas.py` holds the Pydantic models, mirrored in `frontend/src/types.ts`:
- `ServoTelemetry`
- `MaskAttribution`
- `XAIDiagnosisResponse`
- `RecoveryPlan`
- `AnomalyEvent`
- `AnalysisResult`

The same JSON schema that validates Spark's output is sent to Spark as `text.format`, so the model
is constrained to exactly what the code can execute.

## Hardware abstraction

`backend/robot/so101_interface.py` exposes one small surface: `connect / read / send / hold /
estop`. Three implementations sit behind it.

| Mode | Robot | Policy | Saliency |
|---|---|---|---|
| `--policy-path …` | `RealSO101` (LeRobot `SO101Follower`, Feetech bus, OpenCV camera) | `ACTRunner` | ACT decoder cross-attention, with a layer-4 activation-norm fallback |
| `--mock-hardware` | `MockSO101`: rendered overhead view, joint tracking, collisions that stall and spike load, grasping, five scenarios | `MockPolicy`, a pixel-driven pick-and-place controller | Colour saliency of the frame it is fed |
| `--webcam N` | `WebcamSource`: background grab thread, newest frame only, "no signal" if stale | `CameraObserver` | Spectral-residual visual saliency |

`MockPolicy` sees only pixels. Mask suppression therefore changes its behaviour for the same
causal reason it changes a real ACT policy: the distractor really is gone from its input.

## Resilience

Every Meta call goes through `backend/meta_clients/base.py: MetaClient`:
- exponential-backoff retries on 429, 5xx and transport errors;
- a disk cache keyed by the exact request body (`.cache/meta_api/`), so a rehearsed request
  replays instantly if the network drops;
- a source tag on every result: `meta-api`, `cache` or `local-fallback`.

Each model also has a deterministic local fallback:
- simulator ground-truth masks;
- a rule-based diagnosis over the same evidence;
- the geometric bypass planner;
- background-plate inpainting.

None of these calls on the hot path can raise, and the console shows which source answered.

## Frontend

React, TypeScript and Vite, in `frontend/src/`.

- `useTelemetry.ts` receives a JSON frame header followed by a binary JPEG. It decodes the image
  off the main thread with `createImageBitmap` and flushes React state at about 2.5 Hz. It skips
  decoding when the tab is hidden.
- `OverheadCanvas.tsx` draws the frame on a canvas with these overlays:
  - the saliency heatmap, cached per grid;
  - SAM polygons with `A(i)` tags;
  - suppressed masks, dashed;
  - the recovery path.
- `TelemetryScrubber.tsx` plots joint error, chunk variance and peak load (peak-preserving
  downsample to 160 points). Clicking it rewinds the ring buffer.
- `MuseXAIInspector.tsx` shows:
  - run state and controls;
  - the diagnosis with source chips and latency;
  - the fix button;
  - attribution bars against the hijack threshold;
  - the keyframe and the counterfactual;
  - the Wonder plan table and the audited routine;
  - chat with push-to-talk.
- `useVoice.ts` captures audio with an AudioWorklet and streams PCM16 at 16 kHz over `/ws/voice`.
