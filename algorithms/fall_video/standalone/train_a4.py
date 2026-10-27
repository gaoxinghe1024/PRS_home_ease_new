# -*- coding: utf-8 -*-
"""
Train and save the A4 hybrid ensemble weights (one-off script)
==========================================

Trains A4 on the complete unified pool (URFD + MCF + Le2i = 223 sequences)
and pickles the trained base learners + meta-learner to standalone/a4_weights.pkl.
main.py then loads a4_weights.pkl directly, with no retraining each time.

Usage:
  python train_a4.py
"""
import os
import sys
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(ROOT, "00_common"),
           os.path.join(ROOT, "02_pose_cnn"),
           os.path.join(ROOT, "01_detect_yolo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# Important: import a4_core from the same folder, so the pickled class paths are a4_core.*,
# exactly what main.py uses when loading (it imports a4_core too).
from a4_core import HybridAlgorithm, make_windows

from windowize import make_windows as _mw  # noqa: F401  (only ensures the window constants match; a4_core's are used)
from splits import unified_pool
from pose_extract import extract_skeleton_sequence

SRC_FPS = {"urfd": 30.0, "mc": 30.0, "le2i": 25.0}


def _gt_intervals(dataset, seq):
    import urfd_loader, mc_loader, le2i_loader
    if dataset == "urfd":
        return urfd_loader.fall_intervals_s(seq)
    if dataset == "mc":
        return mc_loader.fall_intervals_s(seq)
    return le2i_loader.fall_intervals_s(seq)


def build_train_items():
    items = []
    for d, s, *_ in unified_pool():
        dd = extract_skeleton_sequence(d, s)   # read the skeleton cache (real YOLO+P1 output)
        kpts = dd["kpts"]
        fidx = dd["frame_idx"]
        fps = SRC_FPS[d]
        wins, t0s, t1s = make_windows(kpts)
        if len(t0s) == 0:
            continue
        t0_s = fidx[t0s].astype(np.float64) / fps
        iv = _gt_intervals(d, s)
        items.append((kpts, t0_s, iv))
    return items


def main():
    print("[1/2] building training data (reading the skeleton cache) ...", flush=True)
    items = build_train_items()
    print(f"      {len(items)} sequences", flush=True)

    print("[2/2] training the A4 hybrid ensemble ...", flush=True)
    alg = HybridAlgorithm()
    alg.fit(items)

    out = os.path.join(HERE, "a4_weights.pkl")
    with open(out, "wb") as f:
        pickle.dump({
            "bases": alg.bases,
            "meta": alg.meta,
        }, f)
    print(f"[done] A4 weights saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
