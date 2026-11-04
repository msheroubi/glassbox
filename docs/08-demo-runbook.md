# 08 · Live demo runbook (about 5 minutes)

**The story:** a robot policy fails, and the cause is perception, not decision-making. GlassBox shows
*what it was looking at*, *what in the scene threw it off*, has Muse Wonder explain it, and fixes it
without retraining when no human is needed.

The demo uses the real SO-101, the ACT checkpoint in `models/last/pretrained_model`, the red cube, the
green hex tile and a **red mini pocket book** (the distractor).

## Before you go on stage

- [ ] `MODEL_API_KEY` is in `.env` and the server log shows no `401`.
- [ ] Calibrate once: the homography and an **empty-table background plate**
      (`python -m backend.robot.calibration --camera 2 ...`). Erasing the book then shows the real
      empty table instead of an inpainted smear, and the staged book is erased cleanly from where it lies.
- [ ] Check the masks: `scripts/start_webcam.sh 2` with the cube, tile and book on the table. SAM 3.1 must
      label the book `distractor`, the cube `target` and the tile `goal`.
- [ ] `scripts/start_demo.sh` is running, and http://localhost:8000 is open in a full-screen browser.
      Wait until the inspector chip reads **SAM 3: ready** (the first load downloads ~3.5 GB; `.venv-sam3`
      must exist, see [06 §5](06-running.md)). The arm
      stays on **READY** until you press **▶ Start policy**, so nothing moves while it loads.
- [ ] **Rehearse twice, end to end.** This warms the Meta API cache (`.cache/meta_api/`): if the venue
      network drops, the same requests replay with a blue `cache` chip. It also leaves saved runs in
      `runs/` as a fallback.
- [ ] Arm at rest, cube on its usual start spot (the policy is overfit to it), book somewhere on the
      table but **not** on the tile.
- [ ] Backup: the video ([09](09-video.md)), and `http://localhost:8000/#replay` with the best rehearsal
      run selected in the dropdown.

## The run

| # | Do | They see | Say |
|---|---|---|---|
| 1 | *(book off the table)* **↺ Reset**, wait for **READY** (pre-flight done, SAM 3 `ready`), then **▶ Start policy** | ACT picks the cube and places it on the tile. The live heatmap is ACT's own attention | "An imitation-learned policy, trained on a few dozen demos. It works until it doesn't, and when it fails it doesn't tell you why." |
| 2 | Put the cube back, put the book anywhere on the table, **↺ Reset**, then **▶ Start policy** once the staging badge shows | A "PRE-FLIGHT: SAM 3.1 is scanning the scene" badge, then a magenta **DEMO STAGING** badge and **READY** | Be upfront: "This policy only fails on cue when the book is on the tile, so for the demo GlassBox shows it the book on the tile. That's labelled, and the raw camera is always one click away." |
| 3 | Watch | The arm heads off course and the attention pulls onto the book. Then **HOLD** and *analysis started* | "GlassBox reads the policy's attention every tick. When it drifts somewhere the demos never looked, it stops the arm." |
| 4 | — | The console switches to **Failure replay** by itself and plays the last seconds at 0.5×, starting with ACT's map | "Now we take our time. The run was recorded, and we replay it offline with models too heavy to run live on this laptop." |
| 5 | Point at the progress bar and map buttons | SAM masks appear, then **SAM 3 · 'red cube'** keyframes arrive (failure frame first, then coarse to fine) | "SAM 3 is a much larger perception model, running locally. We ask it where 'the red cube' is in the exact frames our policy saw, and it finds it on the book about as much as on the cube: the scene really is ambiguous. That's not the policy's attention; that's ACT's map, one click over." |
| 6 | Drag the timeline playhead to the red ◀ HIJACK moment | The book's share crosses the cube's; the mask tag reads `distractor 41% ◀ HIJACK` | "Here's the moment. Attention leaves the cube and lands on the book. The model thinks it sees the goal where the book is." |
| 7 | Point at the breakdown card | Muse Wonder's verdict: the stage strip **PERCEPTION ✕ → decision, actuation not reached**, culprit, confidence, timeline | "Muse Wonder gets the filmstrip, every attention curve and the servo telemetry, and says where the failure started. Perception, not decision." |
| 8 | — | Green **✓ Self-corrected, no human needed**. The arm resumes with the book erased from its input and completes the task | "No retraining. GlassBox edits what the policy sees, live, and the policy does what it was trained to do." |
| 9 | Click **Raw camera** in the replay, then Live → Policy/Raw | The real table vs what the policy saw | "Everything we changed in its input is visible and auditable." |
| 10 | *(optional)* Put a hand over the cube mid-run | HOLD, and the breakdown says **✋ Needs a human**: "Move your hand out of the workspace" | "And when the fix isn't something software should do, it says so." |
| 11 | Say *"emergency stop"* or click **E-STOP** | Torque off | "Safety always wins." |

**Timing.** ACT's map is ready within seconds. Each SAM 3 keyframe takes about 25 s on CPU, and Wonder
starts after 3 of them, so the breakdown lands about 90 s after the failure (`LENS=smolvla`: ~6 s per
keyframe, Wonder after 5, about 40–60 s). Fill the gap with beats 5
and 6, which is exactly the time to show the replay. With **Self-correct** off (switch in the inspector),
step 8 becomes an **⚡ Apply self-correction and resume** button, which works well if you want to pause
for questions.

**Between attempts.** Press E-STOP, move the arm to rest and the cube to its start spot, then
**↺ Reset**. Reset re-enables torque, clears the interventions and re-runs the pre-flight scan. The arm then
waits for **▶ Start policy** (`start_demo.sh` launches with `--start-paused`; `AUTOSTART=1` turns it off).

## If something goes wrong

| Problem | Recovery |
|---|---|
| No DEMO STAGING badge after Reset | Pre-flight didn't find the book or the tile (the status line says which). Move the book into clearer view and press Reset again, or put the book on the tile physically |
| The arm succeeds anyway | Put the book physically on the tile, as in the original failure, and Reset. Staging is only the convenience path |
| No HOLD: the arm keeps going without tripping a trigger | Press **Analyze** (or say "analyze"). With `FORCE_FAILURE=1` (the default) the press counts as the failure: the arm holds, the reasons start with `DEMO: failure triggered by the operator`, and the replay, breakdown and self-correction run as in steps 4–8. If Spark finds no failure on its own, the distractor holding the most attention is diagnosed, and the counterfactual test then checks whether it really steered the policy |
| The breakdown is slow | Keep scrubbing the replay: ACT's map and the SAM tags tell the story on their own. Chips show each model's latency |
| Replay stuck at "refining" | Expected while the arm runs (the replay lens yields the CPU). **❚❚ Hold** lets it finish |
| The API is down | Chips turn `cache` or `local-fallback`. Say so: "this is the offline fallback, same evidence". Self-correction still works |
| The arm or USB misbehaves | E-STOP, then open `/#replay` and pick a rehearsal run from the dropdown. The whole replay story works from a saved run |
| Anything else | Play the video ([09](09-video.md)) |
