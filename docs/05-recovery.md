# 05 · Recovery: from diagnosis to a safe fix

GlassBox does more than explain a failure. Each failure type maps to an intervention that changes
what the policy sees or does, and **a human approves every one**.

```
anomaly ─► HOLD (servos hold position) ─► diagnosis ─► [operator: 1-Click Apply Muse Fix] ─► RUNNING
                                                             │
          INPAINT_MASK_SUPPRESSION ── erase the culprit from the policy's view (SAM keeps it tracked)
          WAYPOINT_BYPASS ─────────── run a validated joint trajectory around the obstacle, then hand back
          RETRY_GRASP_OFFSET ──────── shift the grasp target by the measured bias
          NONE (occlusion) ────────── the operator removes the obstruction, then the policy resumes
```

| Failure (Spark's `failure_type`) | Intervention | Parameters Spark returns |
|---|---|---|
| `ATTENTION_HIJACK` | `INPAINT_MASK_SUPPRESSION` | `mask_id`, `dilate_px` |
| `SERVO_STALL_COLLISION` | `WAYPOINT_BYPASS` | Muse Wonder / local planner waypoints |
| `SPECULAR_GLARE` | `RETRY_GRASP_OFFSET` | `dx_mm`, `dy_mm` |
| `OCCLUSION_OOD` | `NONE` (the operator clears the workspace) | none |
| `NONE` | `NONE` | none |

## Safety model

- **HOLD on anomaly.** `_enter_hold()` cancels any running trajectory and commands the servos to
  hold their current position. The policy keeps no momentum.
- **The operator approves.** Automatic analysis runs, but nothing moves until **1-Click Apply Muse
  Fix** (or the voice command "apply the fix"). `POST /api/apply_fix` also accepts an
  `intervention_type` override.
- **E-STOP** (button, `POST /api/estop`, or saying "emergency stop") disables torque at once. A
  fix cannot be applied while E-STOP is active.
- **Every command is clamped.** All targets, from the policy, a recovery trajectory or a voice
  command, go through `clamp_joints()` to the SO-101 joint limits. The real arm also uses LeRobot's
  `max_relative_target` (10° per step).
- **Stale work is discarded.** `reset()` bumps an epoch counter. A control tick whose camera read
  straddled the reset is dropped, so the fresh policy never plans from the previous scene's pose.
  An analysis still running when the scene is reset is stored but not offered as a fix: a
  diagnosis of a scene that no longer exists can't be applied to the new one. If the new scene has
  already raised its own anomaly, that anomaly is analysed next.

## Real-Time Visual Mask Suppression

`backend/xai_engine/intervention.py: MaskSuppressor`

The policy failed because of something it *saw*, so the fix edits what it sees. **The robot's
camera input is edited live, every control tick, before the policy runs.**

1. The culprit's SAM mask (chosen by Spark's `mask_id`) is **dilated** by 15 px with an elliptical
   kernel. This also covers the shadow and the antialiased edge.
2. The union of suppressed masks is **feathered** with a Gaussian (σ = dilate/3) into an alpha
   matte, so the policy never sees a hard seam, which would itself be out of distribution.
3. Each tick the masked region is blended with a **fill**, in this order of preference:
   - the **empty-table background plate** captured at calibration, giving a pixel-exact table;
   - **Telea inpainting** at quarter resolution, to fit within the 33 ms budget;
   - the flat tabletop colour.
4. The blend is computed only inside the matte's bounding box (ROI) to keep the tick fast.

The **raw** frame is still recorded and shown in the operator's "raw" view. Only the policy's input
changes. The console's "policy" view shows exactly what the policy now sees.

**SAM keeps the erasure locked on.** While suppression is active, `_track_suppressed()`
re-segments each suppressed object every 3 s with SAM 3.1 and replaces its mask with the
best-overlapping match, so the erasure follows the object if it moves.

Why it works: ACT's attention is a function of its input pixels. Remove the distractor's pixels
and the cross-attention returns to the target. In the simulator the effect is causal: `MockPolicy`
only sees pixels, and after suppression it picks the cube.

## Waypoint bypass (collisions and stalls)

`backend/meta_clients/wonder_planner.py`, `backend/xai_engine/intervention.py: WaypointExecutor`

1. **Instant baseline (local visibility-graph planner).**
   - SAM's obstacle mask is mapped to a millimetre footprint through the homography.
   - The gripper backs off 35 mm from the contact point.
   - The shortest route through the obstacle's corners, expanded by 32 mm, to a pre-grasp above
     the target is found by trying up to two corners in every order.
   - Every segment is collision-checked through the kinematics.
   - Inverse kinematics converts each via point to joint angles.
2. **Muse Wonder refinement (async).** Wonder gets the diagnosis, the geometry, the calibration,
   the kinematics source, 10 s of telemetry, the executing control-loop source and the baseline
   (see [03](03-meta-models.md)). It returns its own `RecoveryPlan`.
3. **Validation, always.** `validate_plan()` checks either plan:
   - joint limits;
   - a densified, forward-simulated swept path;
   - at least 12.5 mm from every obstacle footprint, along the whole path;
   - the reach envelope.

   A Wonder plan that fails is shown as **REJECTED**, with the reasons, and the baseline stays in
   charge.
4. **Execution.** `WaypointExecutor.load()` densifies the waypoints into per-tick targets:
   - smoothstep interpolation, so velocity is zero at each waypoint;
   - the requested duration, stretched if needed so no joint exceeds **60°/s**.

   When the trajectory finishes, the policy's action queue is flushed and ACT re-plans from the
   pre-grasp pose.

Wonder's `python_routine` is **shown for audit and never executed**.

## Grasp offset (specular glare)

A highlight on the cube pulls the policy's attention, and its grasp, away from the true centre.
GlassBox measures the bias (see [04](04-explainability.md#2-ai-attention-to-mask-attribution)), and
the fix shifts the grasp target by the opposite amount:
- in the simulator, through `MockPolicy.grasp_offset_mm`;
- on a real ACT policy, through `GraspOffset.pan_delta_deg()`, a small-angle conversion of the
  lateral millimetre offset into a shoulder-pan bias added to the policy's actions.

## Clear workspace (occlusion)

When a human hand covers the target, the right fix is a person, not an algorithm. The diagnosis
names the occluder. The operator removes it and confirms, and the policy resumes from a clean
action queue.

## Fallbacks for every model

| Model step | If the API is down and nothing is cached |
|---|---|
| SAM 3.1 masks | Simulator ground truth (mock mode); on hardware the analysis runs without masks |
| Muse Spark diagnosis | `local_diagnosis()`, a rule engine over the same evidence JSON |
| Muse Wonder plan | The validated local visibility-graph plan |
| Muse Image counterfactual | Background-plate or Telea fill of the culprit mask, with a ghost recovery path |
| Muse Voice | The typed command box, which uses the same `voice_command()` path |
