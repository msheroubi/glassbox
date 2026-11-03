"""Build the nominal-attention bank used by the attention-deviation trigger.

Replays the demonstration episodes through the ACT policy, and at every ``--stride``-th frame records
(arm state, the policy's overhead-camera attention map). At run time ``AttentionReference`` looks up the
demos' attention at the nearest arm poses and flags attention that goes somewhere it never went there.

    PYTHONPATH=. python scripts/build_attention_reference.py --policy models/last/pretrained_model \
        --dataset ~/.cache/huggingface/lerobot/devsheroubi/so101metahack

Decoding uses ffmpeg (OpenCV cannot read the dataset's AV1 video). Writes calibration/attention_reference.npz.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import REPO_ROOT
from backend.xai_engine.attention_reference import GRID_HW, to_grid

log = logging.getLogger("build_attention_reference")


def decode(video: Path, start_s: float, n: int, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(video), "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=180, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=8, help="record every Nth frame")
    ap.add_argument("--episodes", type=int, default=0, help="limit to the first N episodes (0 = all)")
    ap.add_argument("--camera-key", default="scene")
    ap.add_argument("--size", default="640x360")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "calibration" / "attention_reference.npz")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from backend.robot.policy_hook import ACTRunner

    w, h = (int(v) for v in args.size.split("x"))
    runner = ACTRunner(args.policy, "cpu", args.camera_key, (w, h), 100, "Pick up the red cube and place it on the green hex tile.")
    ds = args.dataset
    eps = pd.read_parquet(next((ds / "meta" / "episodes").glob("chunk-*/file-*.parquet")))
    data = pd.concat([pd.read_parquet(p) for p in sorted((ds / "data").glob("chunk-*/file-*.parquet"))])
    if args.episodes:
        eps = eps.iloc[: args.episodes]

    states, maps, ep_ids, frames_i, progress = [], [], [], [], []
    t0 = time.time()
    for _, e in eps.iterrows():
        ei, n = int(e["episode_index"]), int(e["length"])
        video = ds / "videos" / f"observation.images.{args.camera_key}" / (
            f"chunk-{int(e[f'videos/observation.images.{args.camera_key}/chunk_index']):03d}"
            f"/file-{int(e[f'videos/observation.images.{args.camera_key}/file_index']):03d}.mp4")
        frames = decode(video, float(e[f"videos/observation.images.{args.camera_key}/from_timestamp"]), n, (w, h))
        st = np.stack(data[data.episode_index == ei]["observation.state"].to_numpy()).astype(np.float32)
        for i in range(0, min(n, len(frames), len(st)), args.stride):
            runner.reset()
            runner._predict(st[i], {args.camera_key: frames[i]})
            sal, _ = runner._saliency()
            if sal is None:
                continue
            states.append(st[i]); maps.append(to_grid(sal)); ep_ids.append(ei); frames_i.append(i); progress.append(i / n)
        log.info("episode %d done (%d samples, %.0fs elapsed)", ei, len(states), time.time() - t0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, states=np.stack(states), maps=np.stack(maps).astype(np.float16),
                        episodes=np.array(ep_ids), frames=np.array(frames_i), progress=np.array(progress, np.float32),
                        grid_hw=np.array(GRID_HW), threshold=np.float32("nan"))
    log.info("wrote %s (%d samples)", args.out, len(states))


if __name__ == "__main__":
    main()
