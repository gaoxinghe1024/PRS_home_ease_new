"""A1/A2/A3/A4 comparison under the shared evaluation protocol."""

import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
COMMON = os.path.join(ROOT, "00_common")
sys.path.insert(0, COMMON)
import splits
import metrics
from metrics import (
    seq_auroc,
    window_level_metrics,
    window_auroc,
    sequence_level_metrics,
    event_level_metrics,
    sequence_metrics,
    calibrate_threshold,
)
from pose_extract import extract_skeleton_sequence
from windowize import make_windows, WIN_FRAMES, FPS

SRC_FPS = {"urfd": 30.0, "mc": 30.0, "le2i": 25.0}


def load_algorithm(name):
    if name == "a1":
        sys.path.insert(0, os.path.join(ROOT, "03_alg_rule", "src"))
        from rule_algorithm import RuleAlgorithm

        return RuleAlgorithm()
    if name == "a2":
        sys.path.insert(0, os.path.join(ROOT, "04_alg_ml", "src"))
        from ml_algorithm import MLAlgorithm

        return MLAlgorithm()
    if name == "a3":
        sys.path.insert(0, os.path.join(ROOT, "05_alg_temporal", "src"))
        from temporal_algorithm import TemporalAlgorithm

        return TemporalAlgorithm()
    if name == "a4":
        sys.path.insert(0, os.path.join(ROOT, "06_alg_hybrid", "src"))
        from hybrid_algorithm import HybridAlgorithm

        return HybridAlgorithm()
    raise ValueError(f"unknown algorithm {name}")


def load_seq(dataset, seq):
    d = extract_skeleton_sequence(dataset, seq)
    kpts = d["kpts"]
    fidx = d["frame_idx"]
    fps = SRC_FPS[dataset]
    import urfd_loader, mc_loader, le2i_loader

    if dataset == "urfd":
        iv = urfd_loader.fall_intervals_s(seq)
    elif dataset == "mc":
        iv = mc_loader.fall_intervals_s(seq)
    else:
        iv = le2i_loader.fall_intervals_s(seq)
    wins, t0s, t1s = make_windows(kpts)
    t0_s = fidx[t0s].astype(np.float64) / fps if len(t0s) else np.array([])
    return (kpts, t0_s, iv, len(iv) > 0)


def _win_labels_sec(t0_s, iv, n_windows):
    labels = np.zeros(n_windows, np.float32)
    win_len_s = WIN_FRAMES / FPS
    for j, w_start in enumerate(t0_s):
        w_end = w_start + win_len_s
        for a, b in iv:
            if w_start <= b and w_end >= a:
                labels[j] = 1.0
                break
    return labels


def _total_dur_s(kpts, t0_s, iv):
    if len(kpts):
        return (
            float(t0_s[-1] + WIN_FRAMES / FPS) if len(t0_s) else float(len(kpts) / FPS)
        )
    return 0.0


def run_cv(alg, folds, seq_info, verbose=True):
    cache = {s: load_seq(s[0], s[1]) for s in seq_info}
    win_probs_all, win_labels_all = ([], [])
    seq_scores_all, seq_labels_all = ([], [])
    seq_results_all = []
    for fold_i, test_groups in enumerate(folds, 1):
        test_seqs = [s for s in seq_info if splits.group_key(*s) in test_groups]
        train_seqs = [s for s in seq_info if splits.group_key(*s) not in test_groups]
        alg.fit([cache[s][:3] for s in train_seqs])
        tr_probs, tr_t0s, tr_ivs = ([], [], [])
        for s in train_seqs:
            kpts, t0_s, iv, hf = cache[s]
            wp = alg.predict(kpts)
            if len(wp) == 0:
                continue
            tr_probs.append(wp)
            tr_t0s.append(t0_s)
            tr_ivs.append(iv)
        thr, thr_info = calibrate_threshold(
            tr_probs, None, tr_ivs, tr_t0s, WIN_FRAMES / FPS, min_prec=0.75
        )
        for s in test_seqs:
            kpts, t0_s, iv, hf = cache[s]
            wp = alg.predict(kpts)
            if len(wp) == 0:
                continue
            wlab = _win_labels_sec(t0_s, iv, len(wp))
            win_probs_all.append(wp)
            win_labels_all.append(wlab)
            seq_scores_all.append(float(np.max(wp)))
            seq_labels_all.append(1 if hf else 0)
            sr = sequence_metrics(
                wp, thr, t0_s, WIN_FRAMES / FPS, iv, _total_dur_s(kpts, t0_s, iv)
            )
            seq_results_all.append(sr)
    wp = np.concatenate(win_probs_all) if win_probs_all else np.array([])
    wl = np.concatenate(win_labels_all) if win_labels_all else np.array([])
    seq_scores_all = np.asarray(seq_scores_all)
    seq_labels_all = np.asarray(seq_labels_all, np.int64)
    window_met = window_level_metrics(wp, wl, thr=0.5)
    window_met["auroc"] = window_auroc(wp, wl) if len(wp) else None
    seq_met = sequence_level_metrics(seq_scores_all, seq_labels_all, thr=0.5)
    seq_met["auroc"] = seq_auroc(seq_scores_all, seq_labels_all)
    event_met = event_level_metrics(seq_results_all)
    return {
        "auroc_seq": seq_met["auroc"],
        "window": window_met,
        "sequence": seq_met,
        "event": event_met,
        "n_seq": len(seq_scores_all),
        "n_win": int(len(wp)),
    }


def run_lodo(alg, seq_info, verbose=True):
    present = [d for d in ("urfd", "mc", "le2i") if any(s[0] == d for s in seq_info)]
    if len(present) < 2:
        print("  LODO skipped: needs at least two datasets", flush=True)
        return {}
    cache = {s: load_seq(s[0], s[1]) for s in seq_info}
    out = {}
    for ds in present:
        test = [s for s in seq_info if s[0] == ds]
        train = [s for s in seq_info if s[0] != ds]
        alg.fit([cache[s][:3] for s in train])
        scores, labels = ([], [])
        for s in test:
            kpts, t0_s, iv, hf = cache[s]
            wp = alg.predict(kpts)
            scores.append(float(np.max(wp)) if len(wp) else 0.0)
            labels.append(1 if hf else 0)
        out[ds] = {"auroc": seq_auroc(scores, labels), "n": len(scores)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", default=None, help="a1/a2/a3/a4; defaults to all")
    a = ap.parse_args()
    pool = splits.unified_pool()
    seq_info = [(d, s) for d, s, *_ in pool]
    folds = splits.group_5fold()
    algs = [a.alg] if a.alg else ["a1", "a2", "a3", "a4"]
    results = {}
    for name in algs:
        print(f"\n===== {name} =====", flush=True)
        alg = load_algorithm(name)
        cv = run_cv(alg, folds, seq_info)
        lodo = run_lodo(alg, seq_info)
        results[name] = {"cv": cv, "lodo": lodo}
        w, s, e = (cv["window"], cv["sequence"], cv["event"])
        print(f"  sequence-level AUROC={cv['auroc_seq']:.4f}", flush=True)
        print(
            f"  window-level  acc={w['acc']:.4f} prec={w['precision']:.4f} recall={w['recall']:.4f} f1={w['f1']:.4f} auroc={w['auroc']:.4f}",
            flush=True,
        )
        print(
            f"  sequence-level  acc={s['acc']:.4f} prec={s['precision']:.4f} recall={s['recall']:.4f} f1={s['f1']:.4f}",
            flush=True,
        )
        print(
            f"  event-level  acc={e['event_accuracy']:.4f} prec={e['event_precision']:.4f} recall={e['event_recall']:.4f} f1={e['event_f1']:.4f}",
            flush=True,
        )
        if lodo:
            print(
                "  LODO: " + ", ".join(f"{k}={v['auroc']:.3f}" for k, v in lodo.items()),
                flush=True,
            )
    out_path = os.path.join(HERE, "exp01_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, allow_nan=True)
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
