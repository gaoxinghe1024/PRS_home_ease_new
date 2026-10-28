# -*- coding: utf-8 -*-
"""Diagnostic: decay of single-feature discriminability under noise degradation.

Exp.2 shows `jerk_max` alone carries about a quarter of the averaged RF/XGB importance,
which predicts that once the jerk estimate is corrupted by noise, performance drops faster
than for a model with dispersed feature reliance.

This script trains no model at all. It computes each feature's **window-level AUROC**
over the whole window pool before and after degradation, quantifying which physical
quantities noise damages most - the mechanism behind A2's performance decay in Exp.4.

Usage: python _diag_noise_feat.py [--resample 60] [--filt butter5]
"""
import os
import sys
import json
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "00_common"))

import corrupt
import dataset
import features as F
import metrics as M
import windowize

WATCH = ["jerk_max", "mag_ptp", "peak_dev", "mag_std", "mag_rms", "e_10_25Hz", "freefall_ratio"]
CONDS = [("clean", None, 0.0),
         ("noise0.10", "noise", 0.10),
         ("noise0.20", "noise", 0.20),
         ("noise0.30", "noise", 0.30),
         ("drop0.40", "drop", 0.40),
         ("segment0.40", "segment", 0.40)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample", default=60)
    ap.add_argument("--filt", default="butter5")
    a = ap.parse_args()
    cfg = {"resample": float(a.resample), "filt": a.filt, "win_s": 2.0, "step_s": 0.25}

    sm = dataset.load_all(cfg)
    wins, ys = [], []
    for s, d in sm.items():
        if len(d["wins"]) == 0:
            continue
        wins.append(d["wins"])
        ys.append(windowize.window_labels(d["t0_s"], d["intervals"], cfg["win_s"]))
    wins = np.concatenate(wins)
    y = np.concatenate(ys)
    names = F.feature_names()
    idx = {n: i for i, n in enumerate(names)}
    print(f"total windows {len(wins)}  positive {int(y.sum())} ({y.mean()*100:.1f}%)  channels {wins.shape[-1]}")

    table = {}
    for cname, kind, level in CONDS:
        grid = corrupt.corrupt(wins, kind, level, seed=1234)
        X = F.extract(grid, cfg["resample"])
        row = {}
        for k in WATCH:
            row[k] = M.auc_score(X[:, idx[k]], y)
        table[cname] = row
        print(f"  [{cname:12s}] " + "  ".join(f"{k}={row[k]:.4f}" for k in WATCH), flush=True)

    base = table["clean"]
    print("\n--- AUROC retention relative to the clean baseline ---")
    print("feature".ljust(15) + "".join(f"{c[0]:>13s}" for c in CONDS))
    for k in WATCH:
        cells = []
        for cname, _, _ in CONDS:
            v = table[cname][k]
            b = base[k]
            cells.append(f"{v/b*100:12.1f}%" if abs(b) > 1e-9 else "         n/a")
        print(k.ljust(15) + "".join(cells))

    out = {"cfg": cfg, "cond": {c[0]: {"kind": c[1], "level": c[2]} for c in CONDS},
           "auroc": table}
    p = os.path.join(ROOT, "results", "diag_noise_feat.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
