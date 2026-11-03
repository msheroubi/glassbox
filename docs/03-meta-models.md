# 03 · How we use Meta's models

GlassBox treats Meta's models as a team. Each one does a job the others can't, and their outputs
feed each other:

```
SAM 3.1 masks ──► A(i) numbers ──► Muse Spark diagnosis ──► Muse Wonder plan ──► validated waypoints
     │                                   │      │
     │                                   │      └──► Muse Image counterfactual ("what it should have seen")
     └── re-tracks the suppressed object every 3 s while the fix runs
Muse Voice Transcribe ──► spoken commands / questions ──► Muse Spark answers grounded in the same evidence
```

All calls go to `https://api.meta.ai/v1` with `MODEL_API_KEY`, through the shared client in
`backend/meta_clients/base.py` (retries, request-keyed disk cache, source tagging).

---

## SAM 3.1: turning pixels into named objects, so attention becomes measurable
`backend/meta_clients/sam_client.py`

**What we send.** One `POST /v1/responses` per *concept*, with `model: "sam-3.1"`, a short noun
phrase ("red cube", "robot arm", "red ball", "cardboard box", "human hand") and the frame as
`input_image`. We ask for `include_confidence` and `one_bit` mask encoding. All concepts are
requested **concurrently** (`asyncio.gather`) and streamed.

**What we do with it.**
- SAM answers in its special-token grammar
  (`<0f>0<|box;…|><|mask;…;data=H,W,!payload|>`). It is parsed with Meta's `meta-sam-parser`, and
  each box-local raster is placed into a full-frame boolean mask.
- **Concepts map to roles.** `--concepts "target=red cube,gripper=robot arm,coffee mug"` tells
  GlassBox which mask is the goal and which are potential distractors. The scene vocabulary is
  open, and no detector needs training.
- **Attribution.** Each mask is intersected with the policy's saliency map to get `A(i)`. This is
  the core trick that lets a vision model explain a *different* neural network's behaviour (see
  [04](04-explainability.md)).
- **Tracking during a fix.** While Real-Time Visual Mask Suppression runs, the suppressed object
  is re-segmented every 3 s. The best-overlapping mask replaces the old one, so the erasure stays
  locked on the object even if it moves.
- **Live hijack trigger.** On real hardware a background SAM pass runs every 4 s, so the
  attention-hijack trigger works on live footage.
- **Clip mode** (`GLASSBOX_USE_SAM_VIDEO=true`). The 90-frame ring-buffer clip is exported as
  MP4, uploaded through the Files API, and segmented with SAM's video tracking.
- **Fallback prompts.** A concept can carry `fallback_prompts`, tried in order when the main
  phrase finds nothing. On the real overhead webcam SAM 3.1 never matches "robotic arm gripper",
  and whether "robot arm" or "robotic arm" finds the SO-101 (confidence 0.52 to 0.68) depends on
  the lighting and pose. Camera-only mode therefore asks for "robot arm", then "robotic arm".

**Fallback.** Simulator ground-truth masks, tagged `sim-ground-truth`. The disk cache also replays
any previously seen frame.

---

## Muse Spark: a structured, evidence-citing failure diagnosis
`backend/meta_clients/spark_explainer.py`

**What we send.** `POST /v1/responses` with `model: "muse-spark-1.3"` and two inputs.

1. **A composite keyframe image**: on the left, the raw camera frame; on the right, the same frame
   with SAM contours labelled `#id label A=0.58` and the policy's saliency as a jet heatmap. Spark
   *sees* where the policy was looking.
2. **The numbers behind the picture**, as JSON:
   - anomaly reasons and per-object `A(i)`;
   - workspace position in mm;
   - IoU with the gripper;
   - current and peak joint error;
   - peak servo load per joint;
   - action-chunk variance;
   - the gripper position;
   - the policy phase;
   - a perceived-vs-true grasp bias in mm;
   - interventions already active.

The prompt carries a compact failure taxonomy that maps each failure type to its intervention.
The output is constrained with `text.format = json_schema` to `XAIDiagnosisResponse`:

```json
{
 "failure_type": "ATTENTION_HIJACK", "culprit_mask_id": 1, "confidence": 0.92,
 "headline_summary": "Policy attention hijacked by red distractor, ignoring target cube.",
 "technical_root_cause": "Distractor mask #1 holds A=0.5775 exceeding 0.45 hijack threshold, while target #0 holds only A=0.3456. Servo shows no stall with joint error 0.0 deg now and low loads <14.4%, and grasp bias [-0.5,-0.5] mm rules out glare. High action-chunk variance 81.1 >25.0 reflects OOD confusion from the hijack during REACH.",
 "intervention_type": "INPAINT_MASK_SUPPRESSION",
 "intervention_parameters": {"mask_id": 1, "dilate_px": 5}
}
```
*(A real response captured from the demo.)*

Note that Spark doesn't just name the failure. It **rules out the alternatives** using the
evidence ("no stall … rules out glare"), and it returns parameters the control loop executes
directly: `mask_id` selects which SAM mask to erase.

**Grounded Q&A.** The same evidence backs `POST /api/ask` and spoken questions, so "Why did the
robot stop?" gets an answer that cites the numbers:

> "It stopped because its attention was hijacked by the red ball [#1] with A=0.58, while the true
> target cube [#0] only got A=0.35 … It's not a motor stall — joint error is 0.0 deg now with low
> loads."

**Safety net.** `_validate()` drops parameters that aren't scalars and checks that
`culprit_mask_id` refers to a real mask. If the API is unavailable and nothing is cached,
`local_diagnosis()` runs a deterministic rule engine over the *same* evidence, so the demo never
stalls.

---

## Muse Wonder: physical reasoning for recovery motions
`backend/meta_clients/wonder_planner.py`

When Spark diagnoses `SERVO_STALL_COLLISION`, the arm needs to move *around* something. We give
Muse Wonder (`muse-wonder-1.0`, `reasoning.effort: high`) the whole physical picture:
- Spark's diagnosis;
- the current joint state and gripper position;
- every SAM object in pixels **and** millimetres, plus obstacle footprints as rectangles;
- the pixel-to-mm calibration table;
- joint limits and the arm's planar kinematic model, given as Python source;
- **10 s of servo telemetry at 2 Hz**: qpos, targets, load, joint error and variance;
- **the control-loop source code that will execute its plan** (`inspect.getsource` of `_tick` and
  `apply_fix`);
- a baseline plan from our local planner that it is asked to improve on.

It returns a `RecoveryPlan` (JSON schema): joint waypoints with durations and notes, a summary,
safety notes, and a Python routine that explains the recovery.

**Nothing Wonder returns runs blindly.** `validate_plan()` does four things:
1. Clamps every waypoint to the joint limits.
2. Densifies the path with the same joint-space interpolation the executor uses.
3. Forward-simulates it with the kinematic model.
4. Checks every sample against each obstacle footprint (with a 12.5 mm gripper margin) and the
   reach envelope.

A plan that fails is rejected. It stays visible to the operator, marked *REJECTED*, and the
validated local plan runs instead. The Python routine is **shown for audit and never `exec`'d**.

**Latency hiding.** Wonder reasons for tens of seconds, so the local visibility-graph planner
produces a collision-checked plan **instantly**. The operator can apply it right away, and
Wonder's refined plan replaces it in the UI when it arrives (`analysis_update`).

---

## Muse Image: the counterfactual ("what the policy should have seen")
`backend/meta_clients/voice_and_image.py: CounterfactualRenderer`

After a diagnosis, the anomaly frame is sent to `POST /v1/images/edits` with an instruction built
from the diagnosis and SAM:
- the culprit's own noun phrase;
- its coarse location ("upper left");
- an explicit list of what to keep unchanged.

| Failure | Edit instruction |
|---|---|
| Attention hijack | "Remove only the red ball in the upper left of the image from the table, showing bare table where it was. Keep the red cube, the robot arm, …" |
| Occlusion | "Remove the human hand and arm from the scene so the red cube underneath is fully visible …" |
| Specular glare | "Remove the white specular glare from the red cube so the whole cube is evenly lit, matte red …" |
| Collision | "Remove only the cardboard box …" |

The result is overlaid with the planned recovery path as a ghost line. It is a visual explanation
for people: "this is the scene your policy was trained on; this one thing was different." It is
also a preview of what Real-Time Mask Suppression will feed the policy.

**Fallback.** The culprit mask is filled with the calibrated empty-table background plate, the
same operation the live suppressor performs, or Telea inpainting.

---

## Muse Voice Transcribe: hands-free operation
`backend/meta_clients/voice_and_image.py: VoiceRelay`, `frontend/src/useVoice.ts`

The browser captures microphone audio with an AudioWorklet and streams **PCM16 at 16 kHz mono**
over `/ws/voice`. The backend relays it to `wss://api.meta.ai/v1/asr/realtime` in `PUSH_TO_TALK`
mode, with `CUMULATIVE` partials and **robot vocabulary as keywords** (SO-101, gripper, cube,
e-stop …). The keywords bias recognition towards the words that matter on a robot bench.

Final transcripts go through `voice_command()`:

| Say | Does |
|---|---|
| "emergency stop", "freeze" | E-STOP: torque disabled immediately |
| "analyze", "what went wrong?" | Runs the full pipeline |
| "apply the fix" | Applies the latest diagnosis |
| "resume", "carry on" | Resumes the policy |
| "reset" | Restarts the scenario |
| anything else | A question for Muse Spark, answered from the latest evidence |

An operator whose hands are on the robot can stop it, ask why it failed, and approve the
recovery without touching a keyboard.

---

## Source tags

Every result in the UI carries a chip showing which source answered:
- `meta-api` (green): a live call to the Meta Model API;
- `cache` (blue): replayed from `.cache/meta_api/` because the network failed;
- `local-fallback` (amber): a deterministic local engine.

Both the source and the latency (`sam 1759 ms · spark 6612 ms · image 13314 ms`) are shown
everywhere, so the operator always knows where an explanation came from.
