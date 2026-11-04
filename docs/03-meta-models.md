# 03 · How we use Meta's models

GlassBox treats Meta's models as a team. Each one does a job the others can't, and their outputs
feed each other:

```
SAM 3.1 masks ──► A(i) numbers ──► Muse Spark diagnosis ──► Muse Wonder plan ──► validated waypoints
     │                                   │      │
     │                                   │      └──► Muse Image counterfactual test ("without the culprit, would it look elsewhere?")
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

## Muse Wonder: the forensic breakdown of a replayed failure

`backend/meta_clients/wonder_breakdown.py`

Spark triages from one keyframe in seconds. Once a failed run has been replayed offline (see
[04 §6](04-explainability.md#6-failure-replay-a-second-slower-look-at-the-run)), Wonder gets the whole
story:
- a filmstrip of four keyframes of what the policy saw, with SAM contours and the attention heatmap;
- the attention share of every object over time, for every map (ACT live, plus the replay lens: SAM 3's
  "red cube" / "green hex tile" maps, labelled as perception rather than attention, or SmolVLA's two maps);
- the detected attention shifts, the servo telemetry and Spark's triage.

It returns a `FailureBreakdown` (JSON schema enforced):
- `failure_stage`: `PERCEPTION`, `DECISION` or `ACTUATION`. The brief states the thesis under test and
  tells Wonder to call perception only when the attention evidence supports it;
- the culprit's SAM label, a perception analysis and a root cause citing the numbers;
- a timeline (each entry seeks the replay when clicked) and a counterfactual;
- `self_correctable`, `needs_human` and `human_action`, plus the intervention and its parameters.

`_consistent()` keeps the flags coherent with the intervention whatever the model writes: a `NONE`
intervention can never be self-correctable. If Wonder rejects the image, the request is retried
text-only. Without the API, `local_breakdown()` applies deterministic rules to the same evidence.

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

## Muse Image: the counterfactual test ("would it have looked elsewhere without it?")
`backend/xai_engine/counterfactual.py`, `ControlLoop.counterfactual`, `CounterfactualRenderer`

Correlation (the attention overlaps the book) isn't causation. So GlassBox removes the suspect from
the failure frame and asks the policy again: **same joint state, edited frame**. Then it compares
where the policy looks and where its predicted reach goes.

**What gets removed** (`choose_edit`): a distractor in the scene is always the suspect, whatever
the diagnosis guessed first. From one frame, Spark often calls the book's sheen "glare on the
target", and repainting glare tests nothing. A hand is removed for occlusion. Glare is only removed
from the target, and only when there is no distractor.

**Where** (`plan_edit`, `removal_region`): Muse Image's edit endpoint takes no mask, and a free-form
"remove the book" redraws the frame (or the book). So the region comes from SAM:
1. Grow SAM's mask over the parts SAM missed but that belong to the object in the picture (dark
   side faces, edges, contact shadow), never into the target, goal or arm.
2. `erase_local`: fill the region with the empty-table plate (or Telea inpainting), so the object
   is gone before Muse sees the frame. Muse is only asked to make the patch look like table.
3. `align` + `composite`: undo Muse's small shift or scale (ECC), then paste its pixels back
   **inside the feathered region only**. Everything else is the original frame, pixel for pixel.
4. Validate: if the culprit's colour survives (`culprit_still_visible`) or the patch doesn't match
   the table around it (`looks_like_table`), use the local erase instead.

**Asking again.** Every available attention model (ACT live, and SmolVLA's two maps when it is the replay
lens) attends the edited frame. SAM 3's maps are left out: without the suspect, it trivially stops finding
it. `compare_maps` reports, all measured in the pre-edit masks:
- how much attention **moved** (total-variation distance);
- each label's share inside its mask and **near** it (its neighbourhood, see
  [04 §2](04-explainability.md#2-ai-attention-to-mask-attribution)) before and after;
- the **flow**: of the attention that moved, the fraction that left each neighbourhood (`left_from`) and
  arrived in it (`arrived_at`), each compared with the neighbourhood's area (`area`);
- the label nearest the map's hottest point before and after (`peak_before`, `peak_after`).

ACT's predicted chunk is projected as a reach path (red: with, green: without). `assess` gives the
`outcome`, judged on the acting policy's map (ACT when it drove the arm):
- **confirmed**: at least 25 % moved, or the reach shifted ≥ 15 px;
- **implicated**: less moved, but it moved *from around the suspect* (≥ 20 % of the moved attention
  left its neighbourhood, at ≥ 2.5× that neighbourhood's area), or the hottest spot sat next to the
  suspect and left with it;
- **ruled out**: under 15 % moved, the reach ends < 15 px from where it did, and nothing concentrated
  left the suspect or the peak;
- **partial**: anything else.

Only the in-mask share would miss a hijack: coarse maps put ~1 % inside a small mask even when their
hot spot hugs it. A reach that is no closer to the cube without the suspect doesn't clear it either:
it is one re-plan from the failure pose, after the policy has already committed.

This goes to Muse Wonder as `counterfactual_test` and **outranks the attention correlation**: only a
ruled-out suspect is cleared, even if it got attention.

In the console, the counterfactual card wipes between the frame as the policy saw it and the edited
frame (dashed: the SAM region), or between the two attention maps.

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
