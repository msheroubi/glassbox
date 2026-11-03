"""Calibrate the attention-deviation threshold of calibration/attention_reference.npz.

Scores every demo sample against the *other* demos (leave-one-episode-out), so the threshold sits above
the normal episode-to-episode variation of the policy's attention. Run after build_attention_reference.py.

    PYTHONPATH=. python scripts/calibrate_attention_reference.py [--path calibration/attention_reference.npz]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.config import REPO_ROOT
from backend.xai_engine.attention_reference import AttentionReference


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=REPO_ROOT / "calibration" / "attention_reference.npz")
    ap.add_argument("--quantile", type=float, default=0.995)
    ap.add_argument("--slack", type=float, default=1.3, help="threshold = slack x the quantile of normal scores")
    args = ap.parse_args()
    ref = AttentionReference.load(args.path)
    stats = ref.calibrate(args.quantile, args.slack)
    ref.save_threshold(args.path, stats["threshold"])
    print({k: round(v, 3) for k, v in stats.items()})
    print(f"saved threshold {stats['threshold']:.3f} to {args.path}")


if __name__ == "__main__":
    main()
