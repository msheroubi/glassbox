# 08 · Live demo runbook (about 4 minutes)

A rehearsed path through the console that shows every model doing its job. Each beat lists what
to click, what the audience sees, and one line to say.

## Before you go on stage

- [ ] `MODEL_API_KEY` is in `.env` and the server log shows no `401`.
- [ ] **Warm the cache.** Run each scenario once (Scenario dropdown, then wait for the diagnosis).
      If the venue Wi-Fi drops, the same requests replay from `.cache/meta_api/` with a blue
      `cache` chip instead of failing.
- [ ] `scripts/start_mock.sh attention_hijack` is running, and the console is open at
      http://localhost:8000 in a full-screen browser.
- [ ] Optional real-desk segment: a second server is running with
      `PORT=8001 scripts/start_webcam.sh 2`, open in another tab.
- [ ] The microphone permission is granted for the page (click the mic once).
- [ ] The explainer video ([09](09-video.md)) is ready offline as a backup.

## The run

| # | Do | They see | Say |
|---|---|---|---|
| 1 | Scenario → `nominal` | The arm picks the red cube and places it on the green hex tile. A heatmap shows where the policy looks | "A $250 arm running an imitation-learned policy. It works until it doesn't, and when it fails you get no explanation." |
| 2 | Scenario → `attention_hijack` | The arm drifts towards a red ball. The `A(i)` bar for the ball crosses the 0.45 line, the arm freezes in **HOLD**, and *analysis started* appears | "GlassBox reads the policy's own attention every tick. SAM 3.1 turns the scene into named objects, so we can measure that 58 % of its attention is on the ball and 35 % on the cube." |
| 3 | Wait about 8 s | Spark's diagnosis appears: **ATTENTION_HIJACK**, culprit #1, 0.92, with the root cause citing the numbers and green `meta-api` chips with latencies | "Muse Spark sees the heatmap and the numbers, rules out a stall and glare, and returns a structured fix: erase mask #1." |
| 4 | Point to the counterfactual panel | Muse Image's version of the scene without the ball | "Muse Image shows the scene the policy *should* have seen." |
| 5 | Hold the mic and ask: *"Why did the robot stop instead of picking up the cube?"* | The transcript streams in, then a Spark answer that quotes A = 0.58 vs 0.35 | "Hands-free, through Muse Voice Transcribe, with answers grounded in the same evidence." |
| 6 | **1-Click Apply Muse Fix** | Switch to the *policy* view: the ball is gone from the robot's input, the attention returns to the cube, and the arm completes the task | "We don't retrain anything. We edit what the policy sees, live, at 30 Hz, and SAM keeps the mask locked on." |
| 7 | Scenario → `servo_stall` | The gripper jams on a box. Joint error and load spike on the scrubber, then HOLD | "A different failure. This time the physics is the problem." |
| 8 | Wait for the diagnosis | **SERVO_STALL_COLLISION** with a waypoint plan from the local planner, marked *validated*. Muse Wonder's plan replaces it when it arrives | "Muse Wonder gets the kinematics, 10 s of servo telemetry and the control-loop source that will execute its plan. Every waypoint is forward-simulated against the obstacle before it can run." |
| 9 | Apply the fix | The ghost path is drawn, the arm backs off, routes around the box and hands control back. The cube lands on the green hex tile | "Recovered, not restarted." |
| 10 | Click the telemetry scrubber at the spike | The console rewinds to the frame of the collision, with its heatmap | "Every second of the last ten is replayable." |
| 11 | *(optional)* The webcam tab, then **Analyze** | A real desk and the real SO-101, with SAM masks and Spark's reading of the scene | "Same pipeline, on my actual workstation." |
| 12 | Say *"emergency stop"* | E-STOP, torque off | "And safety always wins." |

## If something goes wrong

| Problem | Recovery |
|---|---|
| Spark is slow (over 15 s) | Keep talking over the A(i) bars. The diagnosis lands, and the `latency_ms` chip shows why |
| The API is down | Chips turn `cache` or `local-fallback`. Say so: "this is the offline fallback, same evidence." The fix still works |
| Wonder is still thinking at step 9 | Apply anyway. The local plan is validated and runs at once |
| The mic fails | Type the same question in the chat box; it goes through the same command path |
| Anything else | Play the video ([09](09-video.md)) |
