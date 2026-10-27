# -*- coding: utf-8 -*-
"""
train_weights.py - re-export the A4 weights in the CURRENT Python environment
================================================================
Why is this needed?
  The A4 base learners include sklearn / xgboost models, and unpickling them is sensitive to the versions
  of these two libraries. standalone/a4_weights.pkl was exported in another environment (sklearn 1.5.1 /
  xgboost 2.1.1) and may fail to load in a newer one (for example sklearn 1.9 /
  xgboost 3.x).

  So A4 is trained once more in the CURRENT environment and exported to systems/a4_weights.pkl;
  engine.py loads that file first (and falls back to the one in standalone/ only if it is missing).

The training data is the skeleton cache already produced by "YOLO detection + our P1 pose model",
so detection does not have to run again (which saves a lot of time); model structure and hyper-parameters are exactly those of A4.

Usage:
  python train_weights.py
"""
import os
import sys
import time
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))               # systems/
ROOT = os.path.abspath(os.path.join(HERE, ".."))                 # project root
STANDALONE = os.path.join(ROOT, "standalone")
for _p in (STANDALONE,
           os.path.join(ROOT, "00_common"),
           os.path.join(ROOT, "02_pose_cnn"),
           os.path.join(ROOT, "01_detect_yolo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# Same class definitions as main.py / engine.py (pickle class paths are always a4_core.*)
from a4_core import HybridAlgorithm, make_windows
from splits import unified_pool
from pose_extract import extract_skeleton_sequence

SRC_FPS = {"urfd": 30.0, "mc": 30.0, "le2i": 25.0}
OUT_PATH = os.path.join(HERE, "a4_weights.pkl")


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
        dd = extract_skeleton_sequence(d, s)      # read the skeleton cache
        kpts = dd["kpts"]
        fidx = dd["frame_idx"]
        fps = SRC_FPS[d]
        wins, t0s, t1s = make_windows(kpts)
        if len(t0s) == 0:
            continue
        t0_s = fidx[t0s].astype(np.float64) / fps
        items.append((kpts, t0_s, _gt_intervals(d, s)))
    return items


def main():
    t0 = time.time()
    print("[1/2] building training data (reading the YOLO+P1 skeleton cache) ...", flush=True)
    items = build_train_items()
    print(f"      {len(items)} sequences in {time.time()-t0:.1f}s", flush=True)

    print("[2/2] training the A4 hybrid ensemble (7 base learners, OOF stacking + XGBoost meta-learner) ...",
          flush=True)
    alg = HybridAlgorithm()
    alg.fit(items)

    with open(OUT_PATH, "wb") as f:
        pickle.dump({"bases": alg.bases, "meta": alg.meta}, f)

    # Read it back immediately to make sure this environment can load it
    with open(OUT_PATH, "rb") as f:
        back = pickle.load(f)
    print(f"[done] weights saved -> {OUT_PATH}")
    print(f"       {len(back['bases'])} base learners, meta-learner "
          f"{'trained' if back['meta'] is not None else 'none'}, "
          f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
