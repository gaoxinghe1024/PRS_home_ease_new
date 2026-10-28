# -*- coding: utf-8 -*-
"""Experiment 2: comparison of accelerometer "feature extraction / feature selection" methods.

Single-variable principle: preprocessing is taken from the best Experiment 1
configuration (read automatically from the exp01 results; overridable via CLI flags),
the downstream model is fixed (A2: RF+XGB), and only the feature stage changes.

Three comparisons:
  E2a feature-group ablation: each of the 5 groups alone, pairwise combinations,
      and the full set
      motion (motion intensity) / impact (impact & free fall) / orient (posture)
      / freq (spectral) / calm (post-impact inactivity)
  E2b feature selection: all / variance filter / mutual-information top-K /
      RF-importance top-K / RFE / L1 regularisation
  E2c single-feature discriminability (descriptive): window-level AUROC of each
      feature over the whole pool, ranked to show which features carry most information

Usage: python exp02_features.py [--resample 60] [--filt butter10]
Output -> results/exp02_features.json
"""
import os
import sys
import json
import argparse
import functools
import itertools
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "00_common"))
sys.path.insert(0, os.path.join(ROOT, "04_alg_ml", "src"))

import dataset
import evaluate
import features as F
import metrics as M
import preprocess
import splits
import windowize
from ml_algorithm import MLAlgorithm

FAMILIES = ["impact", "calm", "motion", "orient", "freq"]
SELECTIONS = ["all", "var", "mi", "imp", "rfe", "l1"]


def best_preproc_from_exp01():
    valid = set(preprocess.FILTER_CHOICES)
    p = os.path.join(ROOT, "results", "exp01_preproc.json")
    if not os.path.isfile(p):
        return {"resample": 60.0, "filt": "butter10"}
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    cand = []
    for k, v in d.get("E1a_resample", {}).items():
        if v.get("auroc_seq") is None:
            continue
        cand.append((v["auroc_seq"], {"resample": float(k) if k != "raw" else "raw",
                                      "filt": "butter10"}))
    for k, v in d.get("E1b_filter", {}).items():
        if v.get("auroc_seq") is None or k not in valid:
            continue
        cand.append((v["auroc_seq"], {"resample": 60.0, "filt": k}))
    if not cand:
        return {"resample": 60.0, "filt": "butter10"}
    return max(cand, key=lambda x: x[0])[1]


def run(cfg, groups, selection, k, tag, jobs=1):
    fs = cfg["resample"] if cfg["resample"] != "raw" else 62.5
    res = evaluate.run_cv(functools.partial(
        MLAlgorithm, fs=float(fs), seed=0, groups=groups, selection=selection, k=k),
        cfg=cfg, n_jobs=jobs)
    print(f"  [{tag}] seq_AUROC={res['auroc_seq']:.4f}  win_acc={res['window']['acc']:.3f} "
          f"recall={res['window']['recall']:.3f}  seq_recall={res['sequence']['recall']:.3f} "
          f"event_recall={res['event']['event_recall']:.3f} acc={res['event']['event_accuracy']:.3f}",
          flush=True)
    return res


def slim(r):
    return {
        "auroc_seq": r.get("auroc_seq"),
        "window": {k: r["window"].get(k) for k in ("acc", "precision", "recall", "f1", "auroc")},
        "sequence": {k: r["sequence"].get(k) for k in ("acc", "precision", "recall", "f1")},
        "event": r.get("event"), "event_raw": r.get("event_raw"),
        "delay_mean_s": r.get("delay_mean_s"), "fp_per_hour": r.get("fp_per_hour"),
        "n_feat": None, "per_fold_auroc": r.get("per_fold_auroc"),
    }


def univariate_ranking(cfg):
    """Window-level AUROC of each feature over the whole pool (descriptive, not predictive)."""
    sm = dataset.load_all(cfg)
    X, y = [], []
    for s, d in sm.items():
        if len(d["wins"]) == 0:
            continue
        X.append(F.extract(d["wins"], d["fs"]))
        y.append(windowize.window_labels(d["t0_s"], d["intervals"], cfg["win_s"]))
    X = np.concatenate(X); y = np.concatenate(y)
    names = F.feature_names()
    auc = [M.auc_score(X[:, j], y) for j in range(X.shape[1])]
    order = np.argsort([-(a if a == a else 0) for a in auc])
    return [{"feature": names[j], "auroc": float(auc[j])} for j in order]


def importance_table(cfg):
    """Train once on all features and take the top-15 averaged RF/XGB importances."""
    sm = dataset.load_all(cfg)
    d = dataset.pack(sm, list(sm.keys()))
    fs = cfg["resample"] if cfg["resample"] != "raw" else 62.5
    m = MLAlgorithm(fs=float(fs), seed=0, selection="all")
    m.fit(d)
    imp = m.feature_importance()
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:15]
    return [{"feature": k, "importance": float(v)} for k, v in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample", default=None)
    ap.add_argument("--filt", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()

    cfg0 = best_preproc_from_exp01()
    if a.resample is not None:
        cfg0["resample"] = float(a.resample) if a.resample != "raw" else "raw"
    if a.filt is not None:
        cfg0["filt"] = a.filt
    cfg = {"resample": cfg0["resample"], "filt": cfg0["filt"],
           "win_s": windowize.WIN_S, "step_s": windowize.STEP_S}
    print(f"preprocessing config: {cfg}\n", flush=True)

    out = {"cfg": cfg, "E2a_families": {}, "E2b_selection": {}, "E2c_univariate": [], "E2d_importance": []}

    print("=== E2a feature-group ablation (single + pairwise + full) ===", flush=True)
    combos = []
    for f in FAMILIES:
        combos.append((f, (f,)))
    for c in itertools.combinations(FAMILIES, 2):
        combos.append(("+".join(c), c))
    combos.append(("all5", tuple(FAMILIES)))
    for tag, g in combos:
        res = run(cfg, g, "all", 20, tag, a.jobs)
        s = slim(res)
        s["n_feat"] = len(F.feature_names(g))
        out["E2a_families"][tag] = s

    print("\n=== E2b feature selection (input = full feature set) ===", flush=True)
    for sel in SELECTIONS:
        tag = sel
        res = run(cfg, tuple(FAMILIES), sel, 20, tag, a.jobs)
        s = slim(res)
        s["desc"] = F.SELECTION_DESC[sel]
        out["E2b_selection"][tag] = s

    print("\n=== E2c single-feature discriminability (pool-wide window AUROC, top 15) ===", flush=True)
    ur = univariate_ranking(cfg)
    out["E2c_univariate"] = ur
    for r in ur[:15]:
        print(f"  {r['feature']:20s} AUROC={r['auroc']:.4f}")

    print("\n=== E2d full-model feature importance, top 15 ===", flush=True)
    imp = importance_table(cfg)
    out["E2d_importance"] = imp
    for r in imp[:15]:
        print(f"  {r['feature']:20s} imp={r['importance']:.4f}")

    p = os.path.join(ROOT, "results", "exp02_features.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nresults -> {p}")


if __name__ == "__main__":
    main()
