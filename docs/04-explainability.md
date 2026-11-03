# 04 · Explainability: saliency, A(i), anomaly triggers

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
| Action-chunk variance | Mean per-joint variance across the ACT chunk predictions (within 2 replan intervals) that cover the current step > **25 deg²** for 0.4 s | Out-of-distribution confusion: the policy keeps changing its mind |
| Attention deviation | More than the calibrated fraction (`calibration/attention_reference.npz`) of the policy's attention lies outside what the demos attended at the same arm pose, for **1 s** | Anything that pulls the policy's attention somewhere new: decoys, clutter, a hand |
| Attention hijack (fallback) | A non-target object with **A > 0.45** for 0.5 s. Only used when there is no attention reference | Distractors that steal the policy's attention |
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
   counterfactual (see [02](02-architecture.md)).

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
