"""GlassBox-SO101 FastAPI server.

    python -m backend.main --mock-hardware --scenario attention_hijack
    python -m backend.main --policy-path outputs/train/act_so101/checkpoints/last/pretrained_model
    python -m backend.main --webcam 2                       # camera only: overhead USB webcam, no arm

Websockets:
    /ws/telemetry  server -> client: JSON {"type": "frame", ...} immediately followed by one binary
                   JPEG of the overhead view; plus JSON events (anomaly, analysis, analysis_update,
                   fix_applied, state, reset). client -> server: {"view": "policy" | "raw"}.
    /ws/voice      client -> server: {"type": "start"}, binary PCM16 16 kHz mono chunks, {"type": "stop"};
                   or {"type": "text", "text": ...} for a typed command.
                   server -> client: {"type": "listening" | "transcript" | "command" | "answer" | "error" | "done", ...};
                   "done" closes each start/stop or text turn (after any command/answer it triggered).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import REPO_ROOT, SegmentationConcept, Settings, get_settings
from backend.control_loop import ConflictError, ControlLoop
from backend.meta_clients.voice_and_image import VoiceRelay
from backend.schemas import AnalysisResult, ApplyFixRequest, AskRequest, AskResponse
from backend.xai_engine.visualize import encode_jpeg, heatmap_overlay

log = logging.getLogger("glassbox")

FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


class ScenarioRequest(BaseModel):
    scenario: str | None = None


def create_app(settings: Settings) -> FastAPI:
    runtime = ControlLoop(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await runtime.start()
        log.info("GlassBox-SO101 ready on http://%s:%d (mock=%s, camera_only=%s)", settings.host, settings.port,
                 settings.mock_hardware, settings.camera_only)
        try:
            yield
        finally:
            await runtime.stop()

    app = FastAPI(title="GlassBox-SO101", version="1.0.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def _guard(fn, *args):
        try:
            return fn(*args)
        except ConflictError as e:
            raise HTTPException(409, str(e)) from e

    # ------------------------------------------------------------------ REST
    @app.get("/api/status")
    def status() -> dict:
        return runtime.status()

    @app.post("/api/analyze", response_model=AnalysisResult)
    async def analyze() -> AnalysisResult:
        try:
            return await runtime.analyze("manual")
        except ConflictError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/api/analysis/latest", response_model=AnalysisResult)
    def latest_analysis() -> AnalysisResult:
        if runtime.latest_analysis_id is None:
            raise HTTPException(404, "No analysis yet")
        return runtime.analyses[runtime.latest_analysis_id].result

    @app.get("/api/analysis/{aid}", response_model=AnalysisResult)
    def get_analysis(aid: str) -> AnalysisResult:
        if aid not in runtime.analyses:
            raise HTTPException(404, "Unknown analysis")
        return runtime.analyses[aid].result

    @app.get("/api/analysis/{aid}/keyframe.jpg")
    def keyframe(aid: str) -> Response:
        rec = runtime.analyses.get(aid)
        if rec is None:
            raise HTTPException(404, "Unknown analysis")
        return Response(rec.keyframe_jpeg, media_type="image/jpeg")

    @app.get("/api/analysis/{aid}/counterfactual.jpg")
    def counterfactual(aid: str) -> Response:
        rec = runtime.analyses.get(aid)
        if rec is None or rec.counterfactual_jpeg is None:
            raise HTTPException(404, "No counterfactual for this analysis")
        return Response(rec.counterfactual_jpeg, media_type="image/jpeg")

    @app.post("/api/analysis/{aid}/counterfactual")
    async def regenerate_counterfactual(aid: str) -> dict:
        if aid not in runtime.analyses:
            raise HTTPException(404, "Unknown analysis")
        asyncio.create_task(runtime._render_counterfactual(aid))
        return {"ok": True}

    @app.post("/api/apply_fix")
    async def apply_fix(req: ApplyFixRequest) -> dict:
        try:
            return await runtime.apply_fix(req.analysis_id, req.intervention_type)
        except KeyError as e:
            raise HTTPException(404, "Unknown analysis") from e
        except ConflictError as e:
            raise HTTPException(409, str(e)) from e

    @app.post("/api/clear_interventions")
    def clear_interventions() -> dict:
        runtime.clear_interventions()
        return runtime.status()

    @app.post("/api/reset")
    def reset(req: ScenarioRequest | None = None) -> dict:
        scenario = req.scenario if req else None
        try:
            _guard(runtime.reset, scenario)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return runtime.status()

    @app.post("/api/estop")
    def estop() -> dict:
        runtime.estop()
        return runtime.status()

    @app.post("/api/resume")
    def resume() -> dict:
        runtime.resume()
        return runtime.status()

    @app.post("/api/ask", response_model=AskResponse)
    async def ask(req: AskRequest) -> AskResponse:
        return await runtime.ask(req.question)

    @app.get("/api/buffer")
    def buffer() -> dict:
        recs = runtime.ring.snapshot()
        return {
            "frames": [
                {
                    "index": r.index,
                    "t": r.timestamp,
                    "phase": r.phase,
                    "joint_error": r.telemetry.joint_error,
                    "chunk_variance": r.telemetry.action_chunk_variance,
                    "max_load": max(abs(v) for v in r.telemetry.torque_load[:5]),
                }
                for r in recs
            ]
        }

    @app.get("/api/buffer/{index}.jpg")
    def buffer_frame(index: int, overlay: bool = True) -> Response:
        rec = runtime.ring.get(index)
        if rec is None:
            raise HTTPException(404, "Frame no longer in the ring buffer")
        if not overlay or rec.saliency_u8 is None:
            return Response(rec.jpeg, media_type="image/jpeg")
        frame = rec.frame_bgr()
        img = heatmap_overlay(frame, rec.saliency((frame.shape[1], frame.shape[0])))
        return Response(encode_jpeg(img, 80), media_type="image/jpeg")

    @app.get("/api/buffer/{index}/telemetry")
    def buffer_telemetry(index: int) -> dict:
        rec = runtime.ring.get(index)
        if rec is None:
            raise HTTPException(404, "Frame no longer in the ring buffer")
        return {"index": rec.index, "phase": rec.phase, "telemetry": rec.telemetry.model_dump()}

    # ------------------------------------------------------------------ websockets
    @app.websocket("/ws/telemetry")
    async def ws_telemetry(ws: WebSocket) -> None:
        await ws.accept()
        client = runtime.hub.subscribe()
        await ws.send_text(json.dumps({"type": "status", "status": runtime.status()}))
        if runtime.latest_analysis_id:
            res = runtime.analyses[runtime.latest_analysis_id].result
            await ws.send_text(json.dumps({"type": "analysis", "result": json.loads(res.model_dump_json())}))

        async def receive() -> None:
            while True:
                msg = await ws.receive_text()
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if data.get("view") in ("policy", "raw"):
                    client.view = data["view"]

        receiver = asyncio.create_task(receive())
        try:
            while not receiver.done():
                try:
                    await asyncio.wait_for(client.wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                client.wake.clear()
                while client.events:
                    await ws.send_text(json.dumps(client.events.popleft()))
                if client.frame is not None:
                    meta, raw_jpeg, policy_jpeg = client.frame
                    client.frame = None
                    use_policy = client.view == "policy" and policy_jpeg is not None
                    await ws.send_text(json.dumps({**meta, "view": "policy" if use_policy else "raw"}))
                    await ws.send_bytes(policy_jpeg if use_policy else raw_jpeg)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            receiver.cancel()
            runtime.hub.unsubscribe(client)

    @app.websocket("/ws/voice")
    async def ws_voice(ws: WebSocket) -> None:
        await ws.accept()
        relay: VoiceRelay | None = None
        reader: asyncio.Task | None = None

        async def forward(r: VoiceRelay) -> None:
            final_text = ""
            async for ev in r.events():
                if ev["type"] == "error":
                    await ws.send_text(json.dumps({"type": "error", "message": ev.get("message", "ASR error")}))
                    continue
                text = ev.get("transcript", "")
                await ws.send_text(json.dumps({"type": "transcript", "text": text, "final": bool(ev.get("final"))}))
                if ev.get("final") and text.strip():
                    final_text = (final_text + " " + text).strip()
            if final_text:
                await handle_utterance(final_text)

        async def handle_utterance(text: str) -> None:
            reply = await voice_command(runtime, text)
            await ws.send_text(json.dumps(reply))

        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if relay is not None:
                        await relay.send_audio(msg["bytes"])
                    continue
                data = json.loads(msg.get("text") or "{}")
                if data.get("type") == "start":
                    if relay is not None:
                        await relay.close()
                    relay = VoiceRelay(settings)
                    try:
                        await relay.open()
                    except Exception as e:  # noqa: BLE001
                        relay = None
                        await ws.send_text(json.dumps({"type": "error", "message": f"Muse Voice unavailable: {e}"}))
                        continue
                    reader = asyncio.create_task(forward(relay))
                    await ws.send_text(json.dumps({"type": "listening"}))
                elif data.get("type") == "stop" and relay is not None:
                    await relay.finish()
                    if reader is not None:
                        try:
                            await asyncio.wait_for(reader, timeout=15)
                        except asyncio.TimeoutError:
                            await ws.send_text(json.dumps({"type": "error", "message": "Transcription timed out"}))
                    await relay.close()
                    relay, reader = None, None
                    await ws.send_text(json.dumps({"type": "done"}))
                elif data.get("type") == "text":  # typed fallback goes through the same command path
                    await handle_utterance(str(data.get("text", "")))
                    await ws.send_text(json.dumps({"type": "done"}))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if reader is not None:
                reader.cancel()
            if relay is not None:
                await relay.close()

    # ------------------------------------------------------------------ static frontend
    if FRONTEND_DIST.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(FRONTEND_DIST / "index.html")

        @app.get("/{name}")
        def root_file(name: str) -> FileResponse:  # public/ files copied to the dist root (audio worklet, icons)
            path = FRONTEND_DIST / name
            if not path.is_file() or path.parent != FRONTEND_DIST:
                raise HTTPException(404, "Not found")
            return FileResponse(path)

    return app


async def voice_command(runtime: ControlLoop, text: str) -> dict:
    """Map an utterance to a robot command; anything else is a question for Muse Spark."""
    t = text.lower()
    try:
        if any(w in t for w in ("emergency", "e-stop", "estop", "stop the robot", "freeze")):
            runtime.estop()
            return {"type": "command", "text": text, "command": "estop", "message": "E-stop engaged."}
        if "apply" in t and "fix" in t:
            if runtime.latest_analysis_id is None:
                return {"type": "command", "text": text, "command": "apply_fix", "message": "There is no diagnosis to apply yet."}
            res = await runtime.apply_fix(runtime.latest_analysis_id)
            return {"type": "command", "text": text, "command": "apply_fix", "message": res["message"]}
        if any(w in t for w in ("analyze", "analyse", "diagnose", "what went wrong", "what's wrong")):
            result = await runtime.analyze("voice")
            return {"type": "command", "text": text, "command": "analyze", "message": result.diagnosis.headline_summary}
        if any(w in t for w in ("resume", "continue", "carry on")):
            runtime.resume()
            return {"type": "command", "text": text, "command": "resume", "message": "Resumed."}
        if t.strip(" .!") in ("reset", "restart", "start over"):
            runtime.reset()
            return {"type": "command", "text": text, "command": "reset", "message": "Scenario reset."}
    except ConflictError as e:
        return {"type": "error", "message": str(e)}
    answer = await runtime.ask(text)
    return {"type": "answer", **answer.model_dump()}


def main() -> None:
    parser = argparse.ArgumentParser(description="GlassBox-SO101 XAI and failure-recovery server")
    parser.add_argument("--mock-hardware", action="store_true", help="simulated SO-101 + overhead camera (no hardware)")
    parser.add_argument("--scenario", default=None, help="mock scenario: nominal | attention_hijack | servo_stall | occlusion | specular_glare")
    parser.add_argument("--policy-path", default=None, help="trained LeRobot ACT checkpoint (local dir or HF repo id)")
    parser.add_argument("--robot-port", default=None)
    parser.add_argument("--robot-id", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--webcam", default=None, metavar="INDEX",
                        help="camera-only mode: use this USB webcam (index or /dev/videoN) as the overhead view, no arm")
    parser.add_argument("--concepts", default=None,
                        help='comma-separated SAM 3.1 noun phrases, optionally role=phrase '
                             '(roles: target, gripper, obstacle, hand), e.g. "target=red cube,gripper=robot arm,coffee mug"')
    parser.add_argument("--task", default=None, help="task / scene description given to Muse Spark")
    parser.add_argument("--no-auto-analyze", action="store_true", help="detect anomalies but wait for the button")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    overrides = {
        "mock_hardware": args.mock_hardware or None,
        "mock_scenario": args.scenario,
        "policy_path": args.policy_path,
        "robot_port": args.robot_port,
        "robot_id": args.robot_id,
        "host": args.host,
        "port": args.port,
        "auto_analyze": False if args.no_auto_analyze else None,
    }
    if args.webcam is not None:
        overrides["camera_only"] = True
        overrides["overhead_camera_index"] = int(args.webcam) if args.webcam.isdigit() else args.webcam
        overrides["task_description"] = (
            f"{args.task or get_settings().task_description} (Camera-only session: GlassBox sees the overhead webcam "
            "but is not connected to the arm, so there is no joint or servo telemetry.)"
        )
        # On real overhead footage SAM 3.1 never matches "robotic arm gripper", and which of "robot arm" /
        # "robotic arm" finds the SO-101 depends on lighting and pose, so try both.
        arm = {"prompt": "robot arm", "fallback_prompts": ["robotic arm"]}
        overrides["concepts"] = [c.model_copy(update=arm) if c.label == "gripper" else c for c in get_settings().concepts]
    elif args.task:
        overrides["task_description"] = args.task
    if args.concepts:
        items = [c.strip() for c in args.concepts.split(",") if c.strip()]
        roles = [c.split("=", 1) if "=" in c else (c, c) for c in items]
        overrides["concepts"] = [SegmentationConcept(label=r.strip(), prompt=n.strip()) for r, n in roles]
    settings = get_settings().model_copy(update={k: v for k, v in overrides.items() if v is not None})

    import uvicorn

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
