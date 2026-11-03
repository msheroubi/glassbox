"""Attention-deviation: does the policy look somewhere it never looked at this arm pose in the demos?

Checking "is the attention on the target?" is a fixed rule that the policy's own habits break (e.g. a
learned hot spot in an image corner). Instead, ``AttentionReference`` holds the policy's attention maps
on the demonstration episodes (built by ``scripts/build_attention_reference.py``), looks up the demos
at the nearest arm poses, and measures how much of the live attention falls outside what those demos
ever attended to:

    excess = relu(blur(live) - max_over_neighbour_demos(blur(demo)))        score = sum(excess)

``live`` is a distribution (sums to 1), so the score is the fraction of attention that is "unexpected"
(0 = nothing out of the ordinary, 1 = all of it). The excess map also says *where*, so it can be matched
against SAM masks to name the culprit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

GRID_HW = (18, 32)  # attention is compared on a coarse grid: the policy's own map is only 12x20
N_DEMOS = 12  # nearest demonstration episodes used for the envelope
BLUR_SIGMA = 1.0  # grid cells: tolerate small shifts of the same attention
MARGIN = 0.004  # per-cell slack above the demo envelope


def to_grid(saliency: np.ndarray) -> np.ndarray:
    """Saliency map at any resolution -> (18, 32) float32 distribution summing to 1."""
    g = cv2.resize(saliency.astype(np.float32), (GRID_HW[1], GRID_HW[0]), interpolation=cv2.INTER_AREA)
    g = np.clip(g, 0, None)
    total = float(g.sum())
    return g / total if total > 1e-8 else np.full(GRID_HW, 1.0 / (GRID_HW[0] * GRID_HW[1]), np.float32)


def _blur(g: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(g, (0, 0), BLUR_SIGMA)


def grid_share(grid: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of a grid distribution lying inside a full-resolution boolean mask."""
    cover = cv2.resize(mask.astype(np.float32), (GRID_HW[1], GRID_HW[0]), interpolation=cv2.INTER_AREA)
    return float((grid * cover).sum() / (grid.sum() + 1e-9))


@dataclass
class Deviation:
    score: float  # fraction of live attention outside the demos' envelope, 0..1
    excess: np.ndarray  # (18, 32) unexpected attention
    expected: np.ndarray  # (18, 32) mean demo attention at this pose
    pose_dist: float  # distance to the nearest demo pose (in per-joint standard deviations)


class AttentionReference:
    def __init__(self, states: np.ndarray, maps: np.ndarray, episodes: np.ndarray, threshold: float | None) -> None:
        order = np.argsort(episodes, kind="stable")
        self.states = states[order].astype(np.float32)
        self.maps = maps[order].astype(np.float32)
        self.episodes = episodes[order]
        self.threshold = threshold
        self.scale = np.maximum(self.states.std(axis=0), 1.0)
        self.feat = self.states / self.scale
        self.blurred = np.stack([_blur(m) for m in self.maps])
        _, self.starts = np.unique(self.episodes, return_index=True)
        self.ep_ids = self.episodes[self.starts]

    @classmethod
    def load(cls, path: Path) -> "AttentionReference":
        z = np.load(path)
        thr = float(z["threshold"])
        return cls(z["states"], z["maps"], z["episodes"], None if np.isnan(thr) else thr)

    def save_threshold(self, path: Path, threshold: float) -> None:
        z = dict(np.load(path))
        z["threshold"] = np.float32(threshold)
        np.savez_compressed(path, **z)

    def score(self, qpos: np.ndarray, saliency: np.ndarray, exclude_episode: int | None = None) -> Deviation:
        live = to_grid(saliency)
        q = np.asarray(qpos[:6], np.float32) / self.scale
        d = np.linalg.norm(self.feat - q, axis=1)
        best_idx, best_d = [], []
        bounds = list(self.starts) + [len(d)]
        for ep, a, b in zip(self.ep_ids, bounds[:-1], bounds[1:]):
            if exclude_episode is not None and ep == exclude_episode:
                continue
            j = a + int(np.argmin(d[a:b]))
            best_idx.append(j)
            best_d.append(d[j])
        keep = np.argsort(best_d)[:N_DEMOS]
        idx = [best_idx[i] for i in keep]
        envelope = self.blurred[idx].max(axis=0) + MARGIN
        excess = np.clip(_blur(live) - envelope, 0, None)
        # blur spreads mass slightly, so normalise by the blurred total to keep the score in 0..1
        score = float(excess.sum() / max(float(_blur(live).sum()), 1e-9))
        return Deviation(score, excess, self.maps[idx].mean(axis=0), float(np.min(best_d)))

    def calibrate(self, quantile: float = 0.995, slack: float = 1.3, floor: float = 0.1) -> dict[str, float]:
        """Leave-one-episode-out scores over the bank itself -> a threshold above normal run-to-run variation."""
        scores = np.array([
            self.score(self.states[i], self.maps[i], exclude_episode=int(self.episodes[i])).score
            for i in range(0, len(self.maps), 2)
        ])
        thr = max(floor, slack * float(np.quantile(scores, quantile)))
        return {"median": float(np.median(scores)), "p95": float(np.quantile(scores, 0.95)),
                "p995": float(np.quantile(scores, quantile)), "max": float(scores.max()), "threshold": thr}
