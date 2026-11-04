"""SAM 3 as the replay's perception lens: where a promptable segmenter finds the task's noun phrases.

This is not the policy's attention. SAM 3 (``facebook/sam3``) is asked "where is the red cube?" and "where
is the green hex tile?" in each keyframe the policy saw, and its semantic-segmentation head answers with a
per-pixel probability. When the "red cube" map lights up the red book, the scene is visually ambiguous for
that phrase: evidence for a perception failure, independent of how ACT happened to weigh it.

SAM 3 needs transformers >= 5, while SmolVLA and lerobot are pinned to 4.x, so the model runs in its own
interpreter (``.venv-sam3``, see docs/06-running.md) as ``backend/robot/sam3_worker.py``, niced so it gives
way to the control loop. One frame costs ~20-25 s on CPU (the 1008x1008 vision encoder; each phrase adds ~2 s).
"""

from __future__ import annotations

import logging
import os
import pickle
import struct
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

WORKER = Path(__file__).with_name("sam3_worker.py")


class Sam3Observer:
    """Blocking ``lens_frame()`` for the replay analyser; one frame at a time over the worker's pipe."""

    lens_name = "SAM 3"

    def __init__(self, python: str, prompts: dict[str, str], model_id: str = "facebook/sam3",
                 threads: int | None = None, timeout_s: float = 600.0) -> None:
        """``prompts``: concept label ("target", "goal") -> noun phrase ("red cube", "green hex tile")."""
        if not Path(python).exists():
            raise FileNotFoundError(f"SAM 3 interpreter {python} not found (create .venv-sam3, docs/06-running.md)")
        self.prompts = dict(prompts)
        # kind -> the concept label it grounds: the map should sit on that object's mask.
        self.lens_maps = {f"sam3_{label}": label for label in self.prompts}
        self.timeout_s = timeout_s
        self.last_info: dict = {}
        env = dict(os.environ, GLASSBOX_SAM3_MODEL=model_id,
                   GLASSBOX_SAM3_THREADS=str(threads or max(1, (os.cpu_count() or 4) - 4)))
        t0 = time.perf_counter()
        self._proc = subprocess.Popen(
            [python, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
            preexec_fn=lambda: os.nice(10),  # the arm's control loop comes first
        )
        self._lock = threading.Lock()
        ready = self._recv()
        if not (isinstance(ready, dict) and ready.get("ready")):
            self.close()
            raise RuntimeError(f"SAM 3 worker failed to start (exit {self._proc.poll()}); see its stderr above")
        log.info("SAM 3 lens ready in %.0fs (%s)", time.perf_counter() - t0, ", ".join(self.prompts.values()))

    # ------------------------------------------------------------------ pipe
    def _send(self, obj) -> None:
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self._proc.stdin.write(struct.pack(">I", len(data)) + data)
        self._proc.stdin.flush()

    def _recv(self):
        head = self._proc.stdout.read(4)
        if len(head) < 4:
            raise RuntimeError(f"SAM 3 worker exited (code {self._proc.poll()})")
        (n,) = struct.unpack(">I", head)
        return pickle.loads(self._proc.stdout.read(n))

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=5)

    # ------------------------------------------------------------------ inference
    def lens_frame(self, state: np.ndarray, rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Blocking: one grounding map per prompt for an RGB frame, at the frame's size, in [0, 1].

        The map is the semantic head's per-pixel probability for the phrase, scaled by SAM 3's presence score
        (its confidence the phrase is in the image at all), and not renormalised: a phrase SAM 3 can't find
        stays dark instead of having its noise stretched to full brightness."""
        del state  # perception only: the arm's joints don't enter
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError(f"SAM 3 worker is not running (exit {self._proc.returncode})")
            self._send({"rgb": np.ascontiguousarray(rgb), "prompts": list(self.prompts.values())})
            reply = self._recv()
        if "error" in reply:
            raise RuntimeError(f"SAM 3: {reply['error']}")
        h, w = rgb.shape[:2]
        out = {}
        for label, phrase in self.prompts.items():
            sem = reply["maps"][phrase]["semantic"]  # the instance mix leaks onto the gripper; semantic doesn't
            sal = cv2.resize(sem.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)  # square input: stretch back
            out[f"sam3_{label}"] = np.clip(sal * reply["presence"][phrase], 0.0, 1.0)
        self.last_info = {"presence": reply["presence"], "scores": reply["scores"], "ms": reply["ms"]}
        return out
