"""Record real GlassBox sessions for the explainer video.

Drives a running backend through its public API (reset -> wait for the anomaly and the automatic
analysis -> 1-click fix -> recovery) while recording every websocket frame (JPEG + telemetry meta)
and the analysis artefacts (result JSON, keyframe, counterfactual).

    python docs/video/capture.py --base http://localhost:8020 attention_hijack servo_stall
    python docs/video/capture.py --base http://localhost:8021 --webcam      # camera-only session

Output: .cache/video_capture/<scenario>/{frames/*.jpg, frames.jsonl, events.jsonl, analysis.json, ...}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from websockets.asyncio.client import connect

OUT = Path(__file__).resolve().parents[2] / ".cache" / "video_capture"


class Recorder:
    def __init__(self, base: str, name: str) -> None:
        self.base = base
        self.ws_url = base.replace("http", "ws", 1) + "/ws/telemetry"
        self.dir = OUT / name
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        for f in (self.dir / "frames").glob("*.jpg"):
            f.unlink()
        self.frames = open(self.dir / "frames.jsonl", "w")
        self.events = open(self.dir / "events.jsonl", "w")
        self.n = 0
        self.t0 = time.time()
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.state = ""

    def mark(self, label: str, **extra) -> None:
        ev = {"type": "_mark", "label": label, "t_rel": round(time.time() - self.t0, 3), **extra}
        self.events.write(json.dumps(ev) + "\n")
        self.events.flush()
        print(f"  [{ev['t_rel']:6.1f}s] {label}", flush=True)

    async def run(self, view: str = "policy") -> None:
        async with connect(self.ws_url, max_size=2**24) as ws:
            await ws.send(json.dumps({"view": view}))
            meta = None
            async for msg in ws:
                t_rel = round(time.time() - self.t0, 3)
                if isinstance(msg, bytes):
                    if meta is not None:
                        (self.dir / "frames" / f"{self.n:05d}.jpg").write_bytes(msg)
                        self.frames.write(json.dumps({"n": self.n, "t_rel": t_rel, **meta}) + "\n")
                        self.n += 1
                        meta = None
                    continue
                data = json.loads(msg)
                if data.get("type") == "frame":
                    meta = data
                    self.state = data.get("state", "")
                    continue
                data["t_rel"] = t_rel
                self.events.write(json.dumps(data) + "\n")
                self.events.flush()
                await self.queue.put(data)

    async def wait_event(self, kinds: set[str], timeout: float, pred=lambda e: True) -> dict | None:
        end = time.time() + timeout
        while time.time() < end:
            try:
                ev = await asyncio.wait_for(self.queue.get(), timeout=max(0.1, end - time.time()))
            except asyncio.TimeoutError:
                return None
            if ev.get("type") in kinds and pred(ev):
                return ev
        return None

    def close(self) -> None:
        self.frames.close()
        self.events.close()


async def save_analysis(http: httpx.AsyncClient, rec: Recorder, aid: str) -> dict:
    res = (await http.get(f"/api/analysis/{aid}")).json()
    (rec.dir / "analysis.json").write_text(json.dumps(res, indent=1))
    kf = await http.get(f"/api/analysis/{aid}/keyframe.jpg")
    if kf.status_code == 200:
        (rec.dir / "keyframe.jpg").write_bytes(kf.content)
    cf = await http.get(f"/api/analysis/{aid}/counterfactual.jpg")
    if cf.status_code == 200:
        (rec.dir / "counterfactual.jpg").write_bytes(cf.content)
    return res


async def capture_scenario(base: str, scenario: str, question: str | None) -> None:
    print(f"== {scenario}", flush=True)
    async with httpx.AsyncClient(base_url=base, timeout=200) as http:
        for _ in range(240):  # never reset under an in-flight analysis
            if not (await http.get("/api/status")).json().get("analysis_running"):
                break
            await asyncio.sleep(1)
        await http.post("/api/clear_interventions")
        t_reset = time.time()
        await http.post("/api/reset", json={"scenario": scenario})
        rec = Recorder(base, scenario)
        task = asyncio.create_task(rec.run())
        try:
            rec.mark("reset")
            if scenario == "nominal":
                await asyncio.sleep(1)
                end = time.time() + 40
                while time.time() < end and rec.state != "COMPLETE":
                    await asyncio.sleep(0.5)
                rec.mark("complete")
                await asyncio.sleep(2)
                return
            ev = None
            end = time.time() + 90
            while ev is None and time.time() < end and rec.state != "COMPLETE":
                ev = await rec.wait_event({"analysis"}, 1.0, lambda e: e["result"]["created_at"] >= t_reset)
            if ev is None and rec.state == "COMPLETE":
                raise RuntimeError(f"{scenario} completed without an anomaly")
            if ev is None:
                rec.mark("no analysis; triggering manually")
                res = (await http.post("/api/analyze")).json()
                aid = res["analysis_id"]
            else:
                aid = ev["result"]["analysis_id"]
            rec.mark("analysis", analysis_id=aid)
            # Wait for the counterfactual (and Wonder's plan, which can take minutes) before applying.
            need_plan = (await http.get(f"/api/analysis/{aid}")).json()["plan_status"] == "pending"
            end = time.time() + 200
            while time.time() < end:
                res = (await http.get(f"/api/analysis/{aid}")).json()
                if res.get("counterfactual_url") and (not need_plan or res["plan_status"] != "pending"):
                    break
                await asyncio.sleep(1)
            rec.mark("analysis_complete", latency=res.get("latency_ms"))
            if question:
                ans = (await http.post("/api/ask", json={"question": question})).json()
                (rec.dir / "ask.json").write_text(json.dumps(ans, indent=1))
                rec.mark("asked")
            await save_analysis(http, rec, aid)
            fix = (await http.post("/api/apply_fix", json={"analysis_id": aid})).json()
            rec.mark("fix_applied", message=fix.get("message"))
            await asyncio.sleep(1.5)  # let the state flip from HOLD to RUNNING in the frame stream
            end = time.time() + 45
            while time.time() < end and rec.state not in ("COMPLETE", "HOLD"):
                await asyncio.sleep(0.5)
            rec.mark(f"end_state_{rec.state}")
            await asyncio.sleep(2)
        finally:
            task.cancel()
            rec.close()
            await http.post("/api/clear_interventions")


async def capture_webcam(base: str, seconds: float) -> None:
    print("== webcam", flush=True)
    async with httpx.AsyncClient(base_url=base, timeout=120) as http:
        rec = Recorder(base, "webcam")
        task = asyncio.create_task(rec.run(view="raw"))
        try:
            rec.mark("start")
            await asyncio.sleep(seconds / 2)
            res = (await http.post("/api/analyze")).json()
            rec.mark("analysis", analysis_id=res["analysis_id"])
            await asyncio.sleep(seconds / 2)
            ans = (await http.post("/api/ask", json={"question": "What is on the workstation and where is the arm relative to the cube?"})).json()
            (rec.dir / "ask.json").write_text(json.dumps(ans, indent=1))
            await save_analysis(http, rec, res["analysis_id"])
            rec.mark("done")
        finally:
            task.cancel()
            rec.close()


QUESTIONS = {
    "attention_hijack": "Why did the robot stop instead of picking up the cube?",
    "servo_stall": "Is it safe to resume, and what will the arm do?",
}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8020")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--seconds", type=float, default=24)
    ap.add_argument("scenarios", nargs="*")
    a = ap.parse_args()
    if a.webcam:
        await capture_webcam(a.base, a.seconds)
    for s in a.scenarios:
        for attempt in range(3):
            try:
                await capture_scenario(a.base, s, QUESTIONS.get(s))
                break
            except RuntimeError as e:
                print(f"  retrying: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
