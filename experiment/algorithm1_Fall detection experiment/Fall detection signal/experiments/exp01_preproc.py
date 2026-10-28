# -*- coding: utf-8 -*-
"""Experiment 1: comparison of accelerometer "filtering + preprocessing" methods.

Single-variable principle: the downstream model (A2 features + RF/XGB, all features,
no selection) and the evaluation protocol are fixed; only the preprocessing stage changes.

Two comparisons:
  E1a resampling strategy: raw (no resampling, index treated as uniform) / 50Hz / 60Hz / 100Hz
  E1b filter method      : none / moving average / median / Butterworth low-pass
                           3, 5, 10, 20Hz / Savitzky-Golay
                           (every candidate preserves gravity; high-pass and band-pass
                           are excluded, see the design rationale in preprocess.py)

Evaluation:
  (1) downstream metrics: sequence-level AUROC (primary), window Acc/Recall,
      sequence Acc, event Recall
  (2) signal-level metrics (classifier-independent, directly characterising filter quality):
        peak retention   = filtered impact peak / unfiltered impact peak (closer to 1
                           is better; <1 means peak clipping)
        quiet noise ratio= std of filtered / std of unfiltered inside quiet windows
                           (smaller is better, i.e. noise suppression)
        phase lag        = sample lag (ms) at the cross-correlation peak between the
                           filtered and unfiltered signal. Every filter here is
                           zero-phase, so this is identically 0 -- the metric honestly
                           reflects "offline zero-phase != real-time streamable"

Usage: python exp01_preproc.py
Output -> results/exp01_preproc.json plus console tables
"""
import os
import sys
import json
import functools
import warnings

import argparse
import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "00_common"))
sys.path.insert(0, os.path.join(ROOT, "04_alg_ml", "src"))

import acc_loader
import dataset
import evaluate
import preprocess
import windowize
from ml_algorithm import MLAlgorithm

RESAMPLE_SET = ["raw", 50.0, 60.0, 100.0]
FILTER_SET = preprocess.FILTER_CHOICES      # none/ma/median/butter5/butter10/butter20/savgol


# ---------------------------------------------------------------------------
# Signal-level metrics
# ---------------------------------------------------------------------------
def _mag(seq, resample, filt):
    s = preprocess.build_signal(seq, resample=resample, filt=filt)
    return s


def signal_quality(resample, filt, seqs):
    """Peak retention / quiet noise ratio / phase lag (ms)."""
    peaks, ratios, lags = [], [], []
    for seq in seqs:
        a = _mag(seq, resample, filt)
        b = _mag(seq, resample, "none")
        n = min(len(a["mag"]), len(b["mag"]))
        if n < 30:
            continue
        ma, mb = a["mag"][:n], b["mag"][:n]
        # Phase lag (cross-correlation peak)
        x = ma - ma.mean(); y = mb - mb.mean()
        if x.std() > 1e-9 and y.std() > 1e-9:
            c = np.correlate(x, y, mode="full")
            lag = int(np.argmax(c) - (len(y) - 1))
            lags.append(lag / max(a["fs"], 1e-6) * 1000.0)
        if seq.startswith("fall"):
            pkw = float(np.max(np.abs(mb - 1.0)))
            pkf = float(np.max(np.abs(ma - 1.0)))
            if pkw > 0.5:
                peaks.append(pkf / pkw)
        # Quiet noise: the 20% of windows with the lowest sliding-window std
        wins, _ = windowize.make_windows(np.stack([ma, ma, ma, ma], 1), a["fs"])
        winsb, _ = windowize.make_windows(np.stack([mb, mb, mb, mb], 1), b["fs"])
        m = min(len(wins), len(winsb))
        if m >= 5:
            std_f = np.array([np.std(w[:, 3]) for w in wins[:m]])
            std_b = np.array([np.std(w[:, 3]) for w in winsb[:m]])
            k = max(1, int(0.2 * m))
            idx = np.argsort(std_b)[:k]
            if std_b[idx].mean() > 1e-9:
                ratios.append(float(std_f[idx].mean() / std_b[idx].mean()))
    return {
        "peak_retention": float(np.median(peaks)) if peaks else None,
        "quiet_noise_ratio": float(np.median(ratios)) if ratios else None,
        "phase_lag_ms": float(np.median(lags)) if lags else None,
    }


# ---------------------------------------------------------------------------
# Downstream evaluation
# ---------------------------------------------------------------------------
def downstream(cfg, tag, jobs=5):
    """Run 5-fold CV with a fixed A2 downstream model (folds in parallel)."""
    fs = cfg["resample"] if cfg["resample"] != "raw" else 62.5
    res = evaluate.run_cv(functools.partial(
        MLAlgorithm, fs=float(fs), seed=0,
        groups=("motion", "impact", "orient", "freq", "calm"), selection="all"),
        cfg=cfg, n_jobs=jobs)
    print(f"  [{tag}] seq_AUROC={res['auroc_seq']:.4f}  win_acc={res['window']['acc']:.3f} "
          f"recall={res['window']['recall']:.3f}  seq_acc={res['sequence']['acc']:.3f} "
          f"recall={res['sequence']['recall']:.3f}  event_recall={res['event']['event_recall']:.3f} "
          f"prec={res['event']['event_precision']:.3f} acc={res['event']['event_accuracy']:.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()
    falls, adls = acc_loader.list_sequences()
    seqs = falls + adls
    out = {"E1a_resample": {}, "E1b_filter": {}, "signal_quality": {}}

    print("=== E1a resampling strategy (filter fixed to butter10 low-pass, gravity kept) ===", flush=True)
    for rs in RESAMPLE_SET:
        cfg = {"resample": rs, "filt": "butter10", "win_s": windowize.WIN_S,
               "step_s": windowize.STEP_S}
        tag = f"resample={rs}"
        res = downstream(cfg, tag, args.jobs)
        sq = signal_quality(rs, "butter10", seqs)
        out["E1a_resample"][str(rs)] = {"tag": tag, "downstream": res, "signal": sq}
        print(f"        signal: peak_retention={sq['peak_retention']:.3f} "
              f"quiet_noise_ratio={sq['quiet_noise_ratio']:.3f} phase_lag={sq['phase_lag_ms']:.1f}ms", flush=True)

    print("\n=== E1b filter method (resampling fixed to 60Hz) ===", flush=True)
    for ft in FILTER_SET:
        cfg = {"resample": 60.0, "filt": ft, "win_s": windowize.WIN_S,
               "step_s": windowize.STEP_S}
        tag = f"filt={ft}"
        res = downstream(cfg, tag, args.jobs)
        sq = signal_quality(60.0, ft, seqs)
        out["E1b_filter"][ft] = {"tag": tag, "desc": preprocess.describe(ft),
                                 "downstream": res, "signal": sq}
        print(f"        signal: peak_retention={sq['peak_retention']:.3f} "
              f"quiet_noise_ratio={sq['quiet_noise_ratio']:.3f} phase_lag={sq['phase_lag_ms']:.1f}ms", flush=True)

    # Keep only the serialisable key fields
    def slim(d):
        r = d.get("downstream", {})
        return {
            "auroc_seq": r.get("auroc_seq"),
            "window": {k: r.get("window", {}).get(k) for k in ("acc", "precision", "recall", "f1", "auroc")},
            "sequence": {k: r.get("sequence", {}).get(k) for k in ("acc", "precision", "recall", "f1")},
            "event": r.get("event"), "event_calib": r.get("event_calib"),
            "delay_mean_s": r.get("delay_mean_s"), "fp_per_hour": r.get("fp_per_hour"),
            "per_fold_auroc": r.get("per_fold_auroc"),
            "signal": d.get("signal"), "desc": d.get("desc", ""),
        }

    out2 = {
        "E1a_resample": {k: slim(v) for k, v in out["E1a_resample"].items()},
        "E1b_filter": {k: slim(v) for k, v in out["E1b_filter"].items()},
    }
    p = os.path.join(ROOT, "results", "exp01_preproc.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out2, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nresults -> {p}")


if __name__ == "__main__":
    main()
