# -*- coding: utf-8 -*-
"""Experiment 5 (supplementary): preprocessing sensitivity -- which stage's gain is
algorithm-dependent.

Motivation: Experiment 1 picked the best preprocessing (butter5 low-pass) using a fixed
A2 downstream model. Applying that config to the learning-free rule A1, however,
drops A1's sequence-level AUROC from 0.78 to 0.65 and event recall to 0.07 -- because
butter5 clips the impact peak (peak retention only 0.41), while A1's criteria are
**absolute g thresholds** (impact > 2 g, free fall < 0.6 g).

The conclusion points at an engineering fact: **preprocessing and algorithm are not
two separable stages**. This experiment quantifies that coupling by placing A1 (fixed
thresholds), A2 (learned feature model) and A3 (end-to-end network) on the same set of
preprocessing configurations and comparing their response curves.

A1 runs every configuration; A2 runs every configuration (cross-checking
Experiment 1); A3 runs only none / butter10 / butter5 (temporal networks are costly).

Usage: python exp05_preproc_sensitivity.py [--jobs 4] [--a3-epochs 20]
Output -> results/exp05_sensitivity.json
"""
import os
import sys
import json
import argparse
import functools
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for sub in ("00_common", "03_alg_rule/src", "04_alg_ml/src", "05_alg_temporal/src"):
    sys.path.insert(0, os.path.join(ROOT, *sub.split("/")))

import evaluate
import preprocess
import windowize
from exp03_alg_compare import build_factory

CFGS = [
    ("none", {"resample": 60.0, "filt": "none"}, True),
    ("savgol", {"resample": 60.0, "filt": "savgol"}, False),
    ("butter20", {"resample": 60.0, "filt": "butter20"}, False),
    ("butter10", {"resample": 60.0, "filt": "butter10"}, True),
    ("butter5", {"resample": 60.0, "filt": "butter5"}, True),
    ("butter3", {"resample": 60.0, "filt": "butter3"}, False),
    ("raw", {"resample": "raw", "filt": "none"}, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a3-epochs", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    out = {"configs": {}, "algs": {}}
    for name, c, with_a3 in CFGS:
        cfg = dict(c, win_s=windowize.WIN_S, step_s=windowize.STEP_S)
        row = {}
        for alg in (["a1", "a2"] + (["a3"] if with_a3 else [])):
            res = evaluate.run_cv(build_factory(alg, cfg, 2026, a.a3_epochs),
                                  cfg, n_jobs=(a.jobs if alg == "a3" else 1))
            row[alg] = {
                "auroc_seq": res["auroc_seq"],
                "win_recall": res["window"]["recall"],
                "seq_recall": res["sequence"]["recall"],
                "ev_recall": res["event"]["event_recall"],
                "ev_prec": res["event"]["event_precision"],
                "ev_acc": res["event"]["event_accuracy"],
            }
        out["configs"][name] = row
        out["algs"][name] = cfg
        line = "  ".join(f"{k}: AUROC={v['auroc_seq']:.4f} ev_recall={v['ev_recall']:.3f}"
                         for k, v in row.items())
        print(f"[{name:9s}] {line}", flush=True)

    p = os.path.join(ROOT, "results", "exp05_sensitivity.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    print(f"\nresults -> {p}")


if __name__ == "__main__":
    main()
