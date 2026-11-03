# 01 · Overview

## The problem

Low-cost arms like the **LeRobot SO-101** (around $250 in parts) have put imitation learning on
hobbyist desks, in classrooms, in research labs and on small-factory benches. You record a few
dozen demonstrations, train an ACT policy, and the arm picks up a cube.

Then something changes in the scene and the policy fails. Examples:
- a red ball shows up next to the red cube;
- a box sits on the reach path;
- a hand passes over the workspace;
- the desk lamp puts a highlight on the cube.

The policy has no way to say what went wrong. The operator sees a robot that hesitates, jams or
grabs at nothing. Today the usual fixes are to:
- record more demonstrations and retrain, which takes hours;
- guess at the cause;
- give up on the deployment.

**End-to-end visuomotor policies are black boxes, and that is what stops them leaving the lab.**
When an operator cannot see why a robot failed, they cannot trust it, fix it quickly, or safely
leave it running.

## What GlassBox does

GlassBox sits between the camera, the policy and the servos. It adds three capabilities the
robot didn't have before.

1. **It sees what the policy sees.** It hooks the ACT policy's own cross-attention and turns it
   into a live heatmap. SAM 3.1 segments the scene into named objects. The two are combined into a
   per-object number, `A(i)`: the share of the policy's attention that falls on object *i*.
2. **It explains its own failures.** Four triggers watch for trouble:
   - servo following error;
   - action-chunk disagreement;
   - attention hijack;
   - no progress.

   When one fires, the arm holds and the evidence goes to Muse Spark: an annotated keyframe plus
   the telemetry numbers. Muse Spark returns a structured diagnosis, for example:

   > Attention hijack: the policy is looking at the red ball (A = 0.58), not the cube (A = 0.35).
   > Erase mask #1 from its input.

   Muse Image renders the counterfactual: the scene the policy *should* have seen. You can ask
   follow-up questions by typing or by voice, through Muse Voice Transcribe.
3. **It recovers without retraining.** One click applies the fix live.
   - **Real-Time Visual Mask Suppression** erases the distractor from the policy's input, frame
     by frame, while SAM keeps the mask locked on the object.
   - A **waypoint bypass** planned by Muse Wonder steers the arm around an obstacle. Every
     waypoint is validated against the arm's kinematics first.
   - A **grasp offset** corrects a glare-biased grasp point.

   The policy then carries on with the task.

The whole loop runs in about 2 to 10 seconds instead of a retraining cycle measured in hours. The
operator stays in control: nothing runs until they press **1-Click Apply Muse Fix**, and E-STOP is
always one button (or one spoken word) away.

## Who it helps

- **Makers, students and educators** on SO-101 and similar arms. They get a visual answer to
  "why did my policy do that?" with no ML-debugging background needed. It turns a frustrating
  failure into a lesson on how learned policies perceive the world.
- **Robot-learning researchers.** They get a live, quantitative attribution signal (`A(i)` per
  object) and a recorded 10-second ring buffer at every failure. Distribution-shift bugs become
  reproducible, inspectable events instead of anecdotes.
- **Small manufacturers and lab-automation teams** trying low-cost learned manipulation on real
  benches, where clutter, people and lighting changes are normal. A bench operator can keep the
  cell running through everyday disturbances without a robotics engineer on call.
- **Anyone deploying learned policies near people.** The system shows what the robot is paying
  attention to, holds automatically when something looks wrong, and asks a human before it
  changes behaviour.

## What becomes possible

| Before | With GlassBox |
|---|---|
| The policy fails silently or erratically | An anomaly trigger holds the arm within 0.5 s and starts an analysis |
| "Something distracted it, probably?" | `A(red ball) = 0.58 > 0.45`, with the culprit's mask outlined in red and a cited root cause |
| Retrain with more demonstrations (hours) | Erase the distractor from the policy's input (one click, takes effect the next frame) |
| Jammed against a box, so manual teleop | A collision-checked bypass from Muse Wonder, then the policy resumes |
| Debugging needs an ML engineer | Ask "why did it stop?" in plain language, by text or voice |
| Failure footage is lost | A 10 s ring buffer you can scrub frame by frame, with saliency and telemetry |

## It works on a real desk today

`scripts/start_webcam.sh 2` points GlassBox at a real overhead USB webcam over a real SO-101
workstation. On that footage:
- SAM 3.1 segments the **robot arm** and the **red cube**.
- The live saliency heatmap tracks what stands out visually.
- Muse Spark explains the scene.

In a measured run, one analysis took about 11 s end to end (SAM 1.2 s, Spark 9.5 s), both served
by `meta-api`. With a trained ACT checkpoint, `scripts/start_robot.sh <checkpoint>` runs the full
loop on the physical arm: saliency is read from the policy's real cross-attention, and the
interventions drive the real servos.

The simulator in `--mock-hardware` mode has five repeatable failure scenarios, so the full
detect, explain and recover loop can be shown anywhere, offline included.

## Problem-statement fit

GlassBox brings Meta's models to a physical system. A **machine gets a new capability**:
introspecting and correcting its own learned behaviour. It also gives operators a **better way to
understand and control** a robot that runs an opaque neural policy. It grew out of our own SO-101
workstation, where the frustration of "it worked yesterday, why is it grabbing the wrong thing
today?" was the starting point.
