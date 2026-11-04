# 04 · Explainability: saliency, A(i), anomaly triggers, failure replay

GlassBox's explanations come from **measurements of the running policy**. A language model's
guess about what probably happened is not enough. The measurements are cheap enough to compute at
30 Hz, and they are what Muse Spark reasons over.

## 1. Policy saliency: where is the policy looking?

`backend/robot/policy_hook.py: ACTRunner`

LeRobot's ACT policy encodes each camera image with a ResNet backbone, flattens the layer-4
feature map into tokens, and lets a transformer decoder cross-attend from its action queries to
those tokens. GlassBox registers PyTorch forward hooks:

```python
model.backbone.register_forward_hook(...)                          # {"feature_map": layer4}, once per camera
model.decoder.layers[-1].multihead_attn.register_forward_hook(...)  # (B, chunk_size, n_encoder_tokens)
```

Encoder tokens are laid out as `[latent, robot_state?, env_state?, cam0 h×w, cam1 h×w, …]`. The
overhead camera's slice of the head-averaged cross-attention, averaged over the action chunk,
reshapes into an `(h, w)` map. That map is upsampled to the frame and normalised to `[0, 1]`:
**it is the policy's own attention over the overhead image** (`act_cross_attention`). If attention
weights aren't returned, the channel L2 norm of the layer-4 activations is used instead
(`resnet_layer4_activation`).

No retraining or model surgery is needed, and the hooks only read tensors that are already being
computed.

Two other saliency sources cover the other modes. Both report their method, and that method is
passed to Muse Spark:
- **`MockPolicy`** (simulator) uses colour-opponent saliency on saturated red, which is its learned
  cue. It behaves like ACT when two objects match the cue: it samples target hypotheses from the
  red blobs and executes their mean.
- **`CameraObserver`** (webcam only, no policy) uses spectral-residual saliency (Hou & Zhang): FFT
  log-amplitude minus its local average, inverse-transformed, at 128×72 with EMA smoothing. It
  costs about 4 ms per update. It shows what stands out *visually*, not what a policy attends to,
  and the UI says so.

## 2. A(i): attention-to-mask attribution

`backend/xai_engine/attribution.py`

Given a saliency map `S(u,v)` and the SAM 3.1 mask `M_i` of object *i*:

```
          Σ_{(u,v) ∈ M_i} S(u,v)
A(i) = ───────────────────────────
         Σ_{(u,v)} S(u,v)  + 1e-6
```

`A(i)` is the share of the policy's total visual attention that lands on object *i*. It is:
- **comparable across objects and frames**, because it is normalised by total mass;
- **grounded in named objects**, because SAM supplies the masks from free-text concepts;
- **cheap**: one masked sum per object, recomputed live every 3 ticks.

For every object GlassBox also reports:
- the pixel centroid;
- the **workspace position in mm**, via a 4-point homography (`backend/robot/calibration.py`);
- the IoU with the gripper mask.

**Attention near an object.** Policy maps are coarse (one cell per backbone feature or ViT patch) and
diffuse, so `A(i)` for a small object is ~1 % even when the map's hot spot sits right on its edge. GlassBox
therefore also measures the attention **near** each object (`near_shares`): every pixel within
`NEAR_RADIUS_FRAC` (9 %) of the frame width of some mask belongs to the closest object's neighbourhood.
- **near share**: the attention in that neighbourhood;
- **lift**: the near share over the neighbourhood's area. 1 is what a uniform map gives it, 4 means the
  attention is four times as dense there as on average.

An object **draws** attention when its lift is at least `NEAR_DRAW_LIFT` (1.3). The replay, the
counterfactual test and Muse Wonder use near shares. The live hijack trigger still uses `A(i)`.

Masks labelled `target`, `gripper` and `goal` (the green hex tile) are never treated as distractors. The gripper
naturally draws attention while it hovers over the target.

**Grasp bias.** For glare failures we also measure where the policy *thinks* the target is: the
saliency-weighted centroid inside a dilated target mask, minus SAM's centroid, converted to mm. It
is taken from the recent frame with the most attention mass on the target, because once the
gripper hovers above the cube it hides it from the overhead camera. A bias of more than 8 mm with
"no progress" is the signature of a specular highlight pulling the grasp point. The negated bias
becomes the `RETRY_GRASP_OFFSET` correction.

## 3. Anomaly triggers

`AnomalyDetector.update()` runs every tick. Each trigger is debounced so a single noisy frame
can't stop the arm.

| Trigger | Condition (defaults in `backend/config.py`) | Catches |
|---|---|---|
| Servo following error | `‖q_target − q_actual‖₂` over the 5 body joints > **12°** for **0.5 s** while the arm is not moving (< 15°/s) | Collisions, stalls, jams |
| Action-chunk variance (off on the real arm while observe passes run: it also fires on clean demos) | Mean per-joint variance across the ACT chunk predictions (within 2 replan intervals) that cover the current step > **25 deg²** for 0.4 s | Out-of-distribution confusion: the policy keeps changing its mind |
| Attention deviation | More than the calibrated fraction (`calibration/attention_reference.npz`) of the policy's attention lies outside what the demos attended at the same arm pose, for **1 s** | Anything that pulls the policy's attention somewhere new: decoys, clutter, a hand |
| Attention hijack | A non-target object whose attention share exceeds what the demos had there at this pose by **> 0.15** for 0.5 s (**A > 0.45** outright when there is no attention reference) | Distractors that steal the policy's attention |
| No progress | No joint (incl. the gripper) moves more than 4° over **4 s** | Silent failures: hovering, repeated missed grasps |

**Why attention deviation, not "is it on the target".** A fixed rule breaks as soon as the policy has habits:
the trained ACT checkpoint spends attention on an image corner, on the arm and on the goal tile, and that is
fine. What matters is whether the attention differs from what it looked like in the demonstrations *at the
same arm pose*. `scripts/build_attention_reference.py` replays the demos through the policy and stores
(arm state, attention map). At run time `AttentionReference` looks up the demos at the 12 nearest poses
(one per demo) and computes

    excess = relu(blur(live) − max_demos(blur(demo)))        deviation = Σ excess   (live sums to 1)

so the score is the share of attention that is somewhere no demo ever looked. `excess` also says where:
each SAM object gets an `excess_share` (how much of the unexpected attention is inside it) next to its
`expected_share` (the demos' attention there). That names the culprit and feeds Spark. Build and calibrate once:

```bash
PYTHONPATH=. python scripts/build_attention_reference.py --policy models/last/pretrained_model --dataset <so101 dataset dir>
PYTHONPATH=. python scripts/calibrate_attention_reference.py   # leave-one-episode-out threshold
```

**Why chunk variance.** ACT predicts a chunk of future actions every few steps, and consecutive
chunks overlap (`n_action_steps < chunk_size`). When the policy is confident, every chunk that
covers step *t* predicts about the same joint angles. When it is confused, they disagree. The
variance of those overlapping predictions (`ChunkVarianceTracker`) is a free, model-intrinsic
uncertainty signal, with no ensemble or extra forward passes.

When a trigger fires:
1. The arm goes to **HOLD**: the servos hold position and the trajectory is cancelled.
2. An `anomaly` event, with readable reasons, goes to the console.
3. `analyze("auto")` runs: SAM, then `A(i)`, then the keyframe, then Spark, then the plan and the
   counterfactual test: the suspect removed by Muse Image inside its SAM region, and the policy asked
   again (see [03](03-meta-models.md)).

The detector disarms after firing and re-arms after a fix, a resume or a reset.

## 4. The evidence package

`backend/meta_clients/spark_explainer.py: DiagnosisContext.evidence()` is the exact JSON Muse
Spark receives alongside the composite keyframe. The console shows it under **"Evidence sent to
Muse Spark"**, so every claim in the diagnosis can be checked against the numbers.

```json
{
 "anomaly_reasons": ["action-chunk variance 81.1 > 25.0 (OOD confusion)"],
 "objects": [
  {"mask_id": 0, "label": "target", "workspace_xy_mm": [240.0, -60.0], "attention_share": 0.3456, "iou_with_gripper": 0.0},
  {"mask_id": 1, "label": "distractor", "workspace_xy_mm": [235.5, 120.2], "attention_share": 0.5775, "iou_with_gripper": 0.0}
 ],
 "servo": {"joint_error_deg_at_anomaly": 0.0, "joint_error_deg_mean_2s": 0.4, "joint_error_deg_peak_10s": 3.2,
           "mean_abs_load_pct_2s": [ ... ], "peak_abs_load_pct_10s": [ ... ],
           "action_chunk_variance_at_anomaly": 81.1, "action_chunk_variance_peak_10s": 81.1},
 "gripper_xy_mm": [151.2, 2.9],
 "policy_phase": "REACH",
 "target_grasp_bias_mm": [-0.5, -0.5],
 "active_interventions": []
}
```
*(Abridged from the captured attention-hijack run.)*

The servo values are taken **at the anomaly**, from the last tick before HOLD, because HOLD sets
the target to the actual position and zeroes the error. The 2 s means separate a sustained stall
from the brief error spikes of a normal grasp. The 10 s peaks come from the ring buffer.

## 5. The ring buffer and scrubber

`backend/xai_engine/ring_buffer.py` keeps the last **300 control steps (10 s)**. Each step stores:
- the JPEG frame, using the same bytes that were streamed;
- the saliency at quarter resolution;
- the full `ServoTelemetry`;
- the policy phase.

That costs about 20 MB instead of about 1 GB of raw frames. The console's telemetry scrubber plots
joint error, chunk variance and peak load with anomaly markers. Clicking it rewinds to any frame,
which is served with its saliency overlay from `/api/buffer/{index}.jpg?overlay=true`, so you can
replay the exact moment attention drifted.

## 6. Failure replay: a second, slower look at the run

`backend/xai_engine/replay.py`, `backend/robot/sam3_observer.py` (+ `sam3_worker.py`), `backend/robot/vla_observer.py`,
`backend/meta_clients/wonder_breakdown.py`

The live loop can only afford ACT's own attention at 30 Hz. A second, observe-only **lens** (SAM 3, about
25 s per frame on a CPU, or SmolVLA, about 6 s) is too slow to run live. So when a run fails, GlassBox
**records the run and replays it offline**:

1. **Record.** `replay_window_s` (15 s) of the ring buffer before the failure and `replay_tail_s` (2 s)
   after it are written to `runs/<id>/` at 15 fps. The ring is snapshotted when the failure fires, so
   the ~10 s SAM and Spark take don't overwrite the start of the run. Each frame stores the policy's input (after any staging or suppression), the raw camera frame,
   ACT's saliency, the joint state and the telemetry. Runs stay on disk, so any of them can be replayed
   later (`POST /api/replay/{id}/open`, or the run dropdown in the console).
2. **Segment.** SAM 3.1 segments `replay_sam_keyframes` (4) frames of the policy's view. Each attention
   keyframe uses the nearest segmentation for its masks.
3. **ACT map.** ACT's recorded attention becomes a keyframe track at 5 Hz, available within seconds.
4. **Lens maps** (observe only; one lens per backend).
   - **SAM 3** (`--replay-sam3`, the demo default). `facebook/sam3` runs locally in its own interpreter
     (`.venv-sam3`: it needs transformers 5, while lerobot and SmolVLA are pinned to 4.x), as a niced worker
     process. Each keyframe goes through the vision encoder once. The target's and the goal's phrases
     ("red cube", "green hex tile") then each yield a map, `sam3_target` and `sam3_goal`. Each map is the
     semantic head's per-pixel probability for the phrase, scaled by SAM 3's presence score and not
     renormalised, so a phrase it can't find stays dark. These are **perception maps, not attention**: they
     show where a strong segmenter reads the phrase, not where any policy looks. Their shares are in-mask
     (a segmentation hugs its object; "near" is for diffuse attention).
   - **SmolVLA** (`--replay-vla lerobot/smolvla_base`). It runs on the recorded frames with their real
     joint states. One pass yields two maps:
     - *task grounding*: the task's tokens attending to image tokens;
     - *action attention*: the action expert's queries attending to image tokens.

   Keyframes go in bisection order: the failure frame first, then a coarse sweep, then the gaps. The sweep
   runs at `replay_sam3_hz` (0.5 Hz) for SAM 3 and `replay_vla_hz` (1 Hz) for SmolVLA. The breakdown starts
   once the first `replay_sam3_coarse_keyframes` (3) or `replay_coarse_keyframes` (5) keyframes are done.
   The fine pass then sharpens the playback. It pauses while the arm runs so the policy keeps the CPU
   (**Hold** in the console lets it finish).
5. **Shares and events.** For every map and keyframe, `A(label)` and the near share and lift are computed
   as in §2, over all masks with that label. The first **attention shift** is the first keyframe where all
   of these hold:
   - SAM found the target;
   - a distractor has more attention near it than the target has;
   - the distractor's lift is at least 1.3;
   - it still does at the next keyframe.

   Replays saved before near shares fall back to `A(label)` with `SHIFT_MIN_SHARE` (12 %). For a SAM 3 map
   the same rule runs on in-mask shares, against the object the phrase names: "SAM 3 reads the red book
   as 'red cube'" marks the first keyframe where the book holds more of the "red cube" map than the cube
   does (at least 12 %, and still at the next keyframe). Staging and phase changes are added as events.
6. **Breakdown.** Muse Wonder receives:
   - a filmstrip of 4 keyframes of the primary map with the SAM contours;
   - every map's share and near-share series, and the events;
   - which map is the acting policy's (ACT drove the arm; the lens observes only and can corroborate a
     suspect but not clear it; SAM 3's maps are labelled as perception, not attention);
   - the counterfactual test with its flow and `outcome` ([03](03-meta-models.md)).
   - the telemetry, and Spark's live triage.

   It answers **where the failure started**:
   - `PERCEPTION`: it saw the scene wrong;
   - `DECISION`: it saw it right and still acted wrong;
   - `ACTUATION`: the arm could not execute.

   It also names the culprit and says whether one of GlassBox's interventions fixes the cause
   (`self_correctable`) or a person must act (`needs_human`, `human_action`). Without the API, the same
   evidence goes through deterministic rules (`local_breakdown`).
7. **Self-correction.** If the intervention is automatic and no human is needed, it is applied at once
   (`--self-correct`, or the switch in the console). Otherwise the console shows **Apply self-correction
   and resume**.

**Smooth playback.** The browser downloads every frame and plays them back at 0.25–1×. Heatmaps are
linearly interpolated between the two keyframes around the playhead (u8 grids, per pixel). The timeline
draws the share curves through the keyframes: attention near each object by default, or inside the
masks (the **near / in mask** toggle). The HUD says which keyframe or interpolation you are
looking at, so nothing is presented as more measured than it is.

**How to read the lens next to ACT.** Neither SAM 3 nor SmolVLA is the policy that failed. Each is a larger
pretrained model used as a *perception lens*: it is asked where the task's objects are in the very frames
ACT saw.
- *SAM 3's "red cube" map lands on the book as much as on the cube.* The scene is visually ambiguous for
  the task's own words: a strong segmenter confuses the two. On the demo's failure run, 51 % of the "red
  cube" map lies on the book and 44 % on the cube, which supports a perception failure.
- *ACT's own map shows the same pull.* That is direct evidence about the policy that failed.
- *The two disagree.* That is worth saying out loud: it is what an honest interpretability tool shows.
- *A clean SAM 3 map does not clear the book.* A model trained on billions of masks can tell them apart;
  an ACT trained on a few dozen demos may not.

SAM 3 maps are left out of the counterfactual test: with the suspect painted out, of course SAM 3 no
longer finds it there.

**Demo staging** (`--demo-distractor-on-goal`, `DistractorStager` in `backend/xai_engine/intervention.py`).
The demo's ACT checkpoint fails reliably only when the red book sits on the green tile.
- Before each run, a SAM pre-flight scan locates the book and the tile.
- Every frame, the book's pixels are cut from the pre-flight frame, erased where they really are, and
  pasted feathered onto the tile in the policy's input.
- Where the live frame differs from the pre-flight frame at the tile (the arm passing over it), the paste
  is skipped, so the arm still occludes the staged book.
- Suppressing the book turns staging off too.

Staging is labelled in the live view, the replay, the breakdown evidence and the saved run. The Raw camera
view always shows the real table.

**Operator-triggered failure** (`--demo-manual-failure`, on in `start_demo.sh`). A manual **Analyze** with no
anomaly on record becomes the failure: `ControlLoop._forced_anomaly` records an anomaly at the press, with
the reasons `DEMO: failure triggered by the operator` and the top distractor's attention share. The arm
holds, and the run then goes through the same diagnosis, replay, counterfactual test and self-correction
as a detected failure. If Spark still finds no failure, `forced_diagnosis` names the distractor holding the
most attention as the suspect (confidence 0.5). The counterfactual test can then rule it out.
