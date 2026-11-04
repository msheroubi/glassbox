"""Re-measure the cube's and the book's colours (Settings.appearance_refs) from a saved, replayed run.

    PYTHONPATH=. python scripts/measure_appearance.py runs/<run_id>

Uses the run's SAM segmentations (replay.json) on its raw frames. Check the labels it prints are right (both
objects in view, the cube where you placed it), then put the medians in .env, e.g.
    GLASSBOX_APPEARANCE_REFS='{"target": [156, 189, 155], "distractor": [169, 182, 134]}'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from backend.xai_engine.attribution import appearance


def main(run_dir: Path) -> None:
    r = json.loads((run_dir / "replay.json").read_text())
    seen: dict[str, list[np.ndarray]] = {}
    for s in r["segmentations"]:
        im = cv2.imread(str(run_dir / "raw" / f"{s['frame']:04d}.jpg"))
        if im is None:
            continue
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        sat = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1]
        for o in s["objects"]:
            if o["label"] not in ("target", "distractor"):
                continue
            mask = np.zeros(im.shape[:2], np.uint8)
            for p in o["polygons"]:
                cv2.fillPoly(mask, [np.array(p, np.int32)], 1)
            colour = appearance(lab, sat, mask.astype(bool))
            if colour is not None:
                seen.setdefault(o["label"], []).append(colour)
                print(f"frame {s['frame']:4d}  {o['label']:10s} bbox {o['bbox']}  LAB {colour.round().tolist()}")
    refs = {k: np.median(v, axis=0).round().tolist() for k, v in seen.items()}
    print("\nGLASSBOX_APPEARANCE_REFS='" + json.dumps(refs) + "'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
