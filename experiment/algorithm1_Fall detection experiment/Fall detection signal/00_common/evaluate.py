# -*- coding: utf-8 -*-
"""Unified evaluator (K-fold CV / repeated K-fold / robustness).

Every experiment shares this evaluation protocol so that results are comparable
across algorithms and configurations:

  primary   sequence-level AUROC (score = max window probability in the sequence;
            threshold-free)
  three levels  window / sequence / event, each with Accuracy and Recall
  event     Recall / Precision / F1 after threshold calibration (train: prec>=0.75
            -> max recall), plus event metrics at a fixed 0.5 threshold as a
            transparent reference
  other     detection delay, FP-per-hour
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dataset
import metrics as M
import splits
import windowize

ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _ensure_paths():
    """Let joblib worker processes import the algorithm modules (sys.path is not inherited)."""
    for sub in ("00_common", "03_alg_rule/src", "04_alg_ml/src",
                "05_alg_temporal/src", "06_alg_hybrid/src"):
        p = os.path.join(ROOT, *sub.split("/"))
        if p not in sys.path:
            sys.path.insert(0, p)


def _fold_job(payload):
    make_alg, cfg, train_seqs, test_seqs, parallel = payload
    _ensure_paths()
    # Cap per-process threads under fold-level multiprocessing to avoid CPU
    # oversubscription; use all cores when running sequentially.
    # FALL_ACC_TREE_JOBS_FIXED pins an explicit value (useful when several
    # experiments run at the same time and the total thread count must be bounded).
    fixed = os.environ.get("FALL_ACC_TREE_JOBS_FIXED")
    os.environ["FALL_ACC_TREE_JOBS"] = fixed or ("2" if parallel else "-1")
    try:
        import torch
        torch.set_num_threads(2)
    except Exception:
        pass
    sm = dataset.load_all(cfg)
    return eval_algorithm(make_alg(), sm, train_seqs, test_seqs, cfg["win_s"], cfg)


def eval_algorithm(alg, seqs_map, train_seqs, test_seqs, win_s, cfg=None):
    """Evaluate one fold: fit(train) -> predict per sequence -> aggregate the three levels."""
    tr = dataset.pack(seqs_map, train_seqs)
    alg.fit(tr)

    # Threshold calibration on the training set (precision >= 0.75 -> max recall)
    tr_probs, tr_t0s, tr_ivs = [], [], []
    for s in train_seqs:
        d = seqs_map[s]
        if len(d["wins"]) == 0:
            continue
        tr_probs.append(alg.predict(d["wins"]))
        tr_t0s.append(d["t0_s"])
        tr_ivs.append(d["intervals"])
    thr, thr_info = M.calibrate_threshold(tr_probs, tr_ivs, tr_t0s, win_s, min_prec=0.75)

    win_p, win_l = [], []
    seq_s, seq_l = [], []
    seq_res_raw, seq_res_cal = [], []
    for s in test_seqs:
        d = seqs_map[s]
        if len(d["wins"]) == 0:
            continue
        p = alg.predict(d["wins"])
        lab = windowize.window_labels(d["t0_s"], d["intervals"], win_s)
        win_p.append(p); win_l.append(lab)
        seq_s.append(float(np.max(p)) if len(p) else 0.0)
        seq_l.append(1 if d["has_fall"] else 0)
        seq_res_raw.append(M.sequence_metrics(p, 0.5, d["t0_s"], win_s, d["intervals"], d["dur_s"]))
        seq_res_cal.append(M.sequence_metrics(p, thr, d["t0_s"], win_s, d["intervals"], d["dur_s"]))

    wp = np.concatenate(win_p) if win_p else np.array([])
    wl = np.concatenate(win_l) if win_l else np.array([])
    ss = np.asarray(seq_s, np.float64)
    sl = np.asarray(seq_l, np.int64)

    win_met = M.window_level_metrics(wp, wl, thr=0.5)
    win_met["auroc"] = M.auc_score(wp, wl) if len(wp) else None
    seq_met = M.sequence_level_metrics(ss, sl, thr=0.5)
    seq_met["auroc"] = M.auc_score(ss, sl)
    # Primary event-level protocol = fixed 0.5 threshold (no calibration, no optimistic bias)
    ev_fixed = M.event_level_metrics(seq_res_raw)
    # Reference event-level protocol = train-fold calibrated threshold (calibrated on
    # train-fold samples, therefore optimistic; reference only)
    ev_cal = M.event_level_metrics(seq_res_cal)

    delays = [r["delay_s"] for r in seq_res_raw if r["delay_s"] is not None]
    return {
        "thr": thr, "thr_info": thr_info,
        "auroc_seq": seq_met["auroc"],
        "window": win_met, "sequence": seq_met,
        "event": ev_fixed, "event_calib": ev_cal,
        "n_seq": int(len(ss)), "n_win": int(len(wp)),
        "delay_mean_s": float(np.mean(delays)) if delays else None,
        "fp_per_hour": float(np.mean([r["fp_per_hour"] for r in seq_res_raw])) if seq_res_raw else None,
        "seq_scores": ss, "seq_labels": sl,
    }


def _agg_folds(fold_outs):
    """Aggregate folds: AUROC is recomputed on pooled test sequences (more stable); the three levels are pooled too."""
    ss = np.concatenate([f["seq_scores"] for f in fold_outs])
    sl = np.concatenate([f["seq_labels"] for f in fold_outs])

    def mean_of(key, sub):
        vals = [f[sub][key] for f in fold_outs if f[sub].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def mean_event(key):
        vals = [f["event"][key] for f in fold_outs if f["event"].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def mean_event_cal(key):
        vals = [f["event_calib"][key] for f in fold_outs if f["event_calib"].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def mean_win(key):
        vals = [f["window"][key] for f in fold_outs if f["window"].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    win_acc = mean_win("acc"); win_rec = mean_win("recall")
    return {
        "auroc_seq": M.auc_score(ss, sl),
        "auroc_seq_fold_mean": float(np.mean([f["auroc_seq"] for f in fold_outs
                                              if f["auroc_seq"] is not None and not np.isnan(f["auroc_seq"])]))
        if any(f["auroc_seq"] is not None and not np.isnan(f["auroc_seq"]) for f in fold_outs) else None,
        "window": {k: mean_win(k) for k in ("acc", "precision", "recall", "f1", "auroc")},
        "sequence": {k: mean_of(k, "sequence") for k in ("acc", "precision", "recall", "f1")},
        "event": {k: mean_event(k) for k in ("event_recall", "event_precision", "event_f1", "event_accuracy")},
        "event_calib": {k: mean_event_cal(k) for k in ("event_recall", "event_precision", "event_f1", "event_accuracy")},
        "delay_mean_s": float(np.mean([f["delay_mean_s"] for f in fold_outs
                                       if f["delay_mean_s"] is not None])),
        "fp_per_hour": float(np.mean([f["fp_per_hour"] for f in fold_outs
                                      if f["fp_per_hour"] is not None])),
        "n_seq": int(len(ss)), "n_win": int(sum(f["n_win"] for f in fold_outs)),
        "thr_mean": float(np.mean([f["thr"] for f in fold_outs])),
    }


def run_cv(make_alg, cfg=None, folds=None, n_jobs=1):
    """Single K-fold CV. make_alg: () -> new algorithm instance (fresh per fold, no state leakage).

    With n_jobs > 1 each fold runs in its own joblib process, which markedly shortens
    the runtime of the temporal-network experiments.
    """
    cfg = dict(dataset.DEFAULT_CFG if cfg is None else cfg)
    folds = splits.stratified_folds() if folds is None else folds
    pool_seqs = list(dataset.load_all(cfg).keys())
    par = bool(n_jobs and n_jobs > 1 and len(folds) > 1)
    jobs = []
    for test_seqs in folds:
        test_seqs = list(test_seqs)
        train_seqs = [s for s in pool_seqs if s not in set(test_seqs)]
        jobs.append((make_alg, cfg, train_seqs, test_seqs, par))
    if par:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fold_job)(j) for j in jobs)
    else:
        outs = [_fold_job(j) for j in jobs]
    res = _agg_folds(outs)
    res["per_fold_auroc"] = [f["auroc_seq"] for f in outs]
    return res


def run_cv_repeated(make_alg, cfg=None, k=splits.K_FOLDS, seeds=splits.SEEDS, n_jobs=1):
    """Multi-seed repeated K-fold: returns {seed: result} plus mean/std of the key metrics."""
    cfg = dict(dataset.DEFAULT_CFG if cfg is None else cfg)
    per_seed = {}
    for s in seeds:
        per_seed[s] = run_cv(make_alg, cfg, folds=splits.stratified_folds(k, s), n_jobs=n_jobs)
    keys = {
        "auroc_seq": [r["auroc_seq"] for r in per_seed.values()],
        "win_acc": [r["window"]["acc"] for r in per_seed.values()],
        "win_recall": [r["window"]["recall"] for r in per_seed.values()],
        "seq_acc": [r["sequence"]["acc"] for r in per_seed.values()],
        "seq_recall": [r["sequence"]["recall"] for r in per_seed.values()],
        "ev_recall": [r["event"]["event_recall"] for r in per_seed.values()],
        "ev_acc": [r["event"]["event_accuracy"] for r in per_seed.values()],
    }
    summary = {k2: {"mean": float(np.mean(v2)), "std": float(np.std(v2))}
               for k2, v2 in keys.items()}
    return {"per_seed": per_seed, "summary": summary}
