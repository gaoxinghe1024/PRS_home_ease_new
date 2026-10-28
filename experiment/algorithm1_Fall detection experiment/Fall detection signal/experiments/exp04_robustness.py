# -*- coding: utf-8 -*-
"""Experiment 4: robustness to sensor degradation (noise / packet loss / block loss).

Motivation: the sensor path has only URFD, so a cross-dataset (LODO) generalisation
test is impossible. Following the "robustness / behaviour under missing-modality
conditions" requirement, robustness is instead probed by **signal degradation**: the
model trains on clean data and degradation is injected into the test set.

Degradation types (see 00_common/corrupt.py):
  noise   sigma in {0.10, 0.20, 0.30} g -- device thermal noise / loose mounting
  drop    in {0.20, 0.40}               -- random packet loss (Bluetooth dropouts,
                                           sampling gaps); linearly interpolated
  segment in {0.20, 0.40}               -- contiguous block loss (bad contact);
                                           linearly interpolated

Protocol: stratified 5-fold CV (seed 2026), parallel over (algorithm x condition x fold).
Coverage: A1/A2 run all 8 conditions; A3/A4 run only the 4 key conditions
(clean + the strongest level of each degradation type).
Results are bucketed by **(algorithm, condition)** -- fold results from different
algorithms are never pooled together.
Reports sequence-level AUROC / window Recall / event Recall and their decay relative
to the clean baseline.

Usage: python exp04_robustness.py [--algs a1,a2,a3,a4] [--epochs 20] [--jobs 8]
Output -> results/exp04_robustness.json
"""
import os
import sys
import json
import time
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for sub in ("00_common", "03_alg_rule/src", "04_alg_ml/src",
            "05_alg_temporal/src", "06_alg_hybrid/src"):
    sys.path.insert(0, os.path.join(ROOT, *sub.split("/")))

import corrupt
import dataset
import evaluate
import splits
import windowize
from exp03_alg_compare import build_factory, cfg_from_exp01

CONDITIONS = {
    "clean": (None, 0.0),
    "noise0.10": ("noise", 0.10), "noise0.20": ("noise", 0.20), "noise0.30": ("noise", 0.30),
    "drop0.20": ("drop", 0.20), "drop0.40": ("drop", 0.40),
    "segment0.20": ("segment", 0.20), "segment0.40": ("segment", 0.40),
}
HEAVY = ["clean", "noise0.20", "drop0.40", "segment0.40"]      # expensive algorithms (A3/A4) only run these


def _worker(payload):
    alg_name, cond, (kind, level), cfg, train_seqs, test_seqs, epochs, seed = payload
    evaluate._ensure_paths()
    os.environ["FALL_ACC_TREE_JOBS"] = "2"      # task-level parallelism, so cap per-task threads
    try:
        import torch
        torch.set_num_threads(2)
    except Exception:
        pass
    sm = dataset.load_all(cfg)
    inner = build_factory(alg_name, cfg, seed, epochs)()
    alg = corrupt.CorruptedAlgorithm(inner, kind, level, seed=seed)
    res = evaluate.eval_algorithm(alg, sm, train_seqs, test_seqs, cfg["win_s"], cfg)
    return alg_name, cond, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algs", default="a1,a2,a3,a4")
    ap.add_argument("--conds", default=None, help="comma-separated subset of degradation conditions (default: all)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--resample", default=None)
    ap.add_argument("--filt", default=None)
    ap.add_argument("--out", default=None, help="output filename (default exp04_robustness.json)")
    a = ap.parse_args()

    subset = None if a.conds is None else set(x.strip() for x in a.conds.split(","))
    conds_all = {k: v for k, v in CONDITIONS.items() if subset is None or k in subset}

    c0 = cfg_from_exp01()
    if a.resample is not None:
        c0["resample"] = float(a.resample) if a.resample != "raw" else "raw"
    if a.filt is not None:
        c0["filt"] = a.filt
    cfg = {"resample": c0["resample"], "filt": c0["filt"],
           "win_s": windowize.WIN_S, "step_s": windowize.STEP_S}
    algs = [x.strip() for x in a.algs.split(",") if x.strip()]
    folds = splits.stratified_folds(5, a.seed)
    pool = list(dataset.load_all(cfg).keys())
    print(f"preprocessing config: {cfg}\nalgs: {algs}\nconditions: {list(conds_all)}\n", flush=True)

    jobs = []
    for name in algs:
        # Cheap algorithms (A1/A2) get the full 8-condition curve; expensive ones
        # (A3/A4) only get the key conditions
        conds = conds_all if name in ("a1", "a2") else {
            k: v for k, v in conds_all.items() if k in HEAVY}
        if not conds:
            continue
        for cond, (kind, level) in conds.items():
            for te in folds:
                te = list(te)
                tr = [s for s in pool if s not in set(te)]
                jobs.append((name, cond, (kind, level), cfg, tr, te, a.epochs, a.seed))
    print(f"total tasks: {len(jobs)}", flush=True)

    t0 = time.time()
    from joblib import Parallel, delayed
    raw = Parallel(n_jobs=a.jobs, backend="loky", verbose=5)(
        delayed(_worker)(j) for j in jobs)

    # Bucket by (algorithm, condition) -- the algorithm name is essential, otherwise
    # fold results from different algorithms would be pooled together
    bucket = {}
    for alg_name, cond, res in raw:
        bucket.setdefault((alg_name, cond), []).append(res)

    out = {"cfg": cfg, "seed": a.seed, "algorithms": {}, "elapsed_s": None}
    for name in algs:
        per = {}
        for cond in CONDITIONS:
            key = (name, cond)
            if key not in bucket:
                continue
            agg = evaluate._agg_folds(bucket[key])
            per[cond] = {
                "auroc_seq": agg["auroc_seq"],
                "win_acc": agg["window"]["acc"], "win_recall": agg["window"]["recall"],
                "win_f1": agg["window"]["f1"],
                "seq_acc": agg["sequence"]["acc"], "seq_recall": agg["sequence"]["recall"],
                "seq_prec": agg["sequence"]["precision"],
                "ev_recall": agg["event"]["event_recall"], "ev_acc": agg["event"]["event_accuracy"],
                "ev_prec": agg["event"]["event_precision"], "ev_f1": agg["event"]["event_f1"],
                "fp_per_hour": agg["fp_per_hour"], "delay_mean_s": agg["delay_mean_s"],
                "n_folds": len(bucket[key]),
            }
            print(f"  {name:3s} {cond:12s} AUROC={agg['auroc_seq']:.4f} "
                  f"win_Rec={agg['window']['recall']:.3f} seq_Rec={agg['sequence']['recall']:.3f} "
                  f"ev_Rec={agg['event']['event_recall']:.3f} ev_Acc={agg['event']['event_accuracy']:.3f}",
                  flush=True)
        out["algorithms"][name] = per

    print("\n===== sequence AUROC relative to clean (%) =====", flush=True)
    for name in algs:
        per = out["algorithms"].get(name) or {}
        if "clean" not in per:
            continue
        b = per["clean"]["auroc_seq"]
        cells = []
        for cond in CONDITIONS:
            if cond in per:
                cells.append(f"{cond}={per[cond]['auroc_seq']/b*100:.1f}%")
        print(f"  {name}: " + "  ".join(cells), flush=True)

    out["elapsed_s"] = round(time.time() - t0, 1)
    p = os.path.join(ROOT, "results", a.out or "exp04_robustness.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nresults -> {p}  (elapsed {out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
