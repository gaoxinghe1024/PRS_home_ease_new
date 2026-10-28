# -*- coding: utf-8 -*-
"""Evaluation metrics (sensor path, identical conventions to the video path).

Event semantics:
  window probability > threshold -> hit window; hits closer than MERGE_GAP_WIN
  windows are merged into one event.
  Alarm time = end time of the earliest hit window of the event.

Metrics:
  - event level  : Event Recall / Precision / F1 / Accuracy (GT interval coverage)
  - sequence level: fall sequence detected / ADL sequence no false alarm;
                    AUROC (score = max window probability in the sequence) -- PRIMARY
  - window level : one sample per sliding window
  - Detection delay / FP-per-hour
Threshold calibration: on the validation set, pick the threshold with maximum
recall subject to precision >= 0.75.
"""
import numpy as np

from windowize import MERGE_GAP_WIN


# ---------------------------------------------------------------------------
# Event aggregation
# ---------------------------------------------------------------------------
def event_aggregate(win_probs, thr, merge_gap=MERGE_GAP_WIN):
    """Window probabilities -> event list [(start_win_idx, end_win_idx)]."""
    hits = np.where(np.asarray(win_probs) > thr)[0]
    if len(hits) == 0:
        return []
    events = []
    s = e = hits[0]
    for h in hits[1:]:
        if h - e <= merge_gap + 1:
            e = h
        else:
            events.append((s, e))
            s = e = h
    events.append((s, e))
    return events


def _overlap(a0, a1, b0, b1):
    return a0 <= b1 and a1 >= b0


def sequence_metrics(win_probs, thr, t0_s, win_len_s, fall_intervals_s, total_dur_s):
    """Event-level metrics for one sequence (same inputs as the video path)."""
    ev = event_aggregate(win_probs, thr)
    ev_times = []
    for s_i, e_i in ev:
        ev_times.append((float(t0_s[s_i]), float(t0_s[e_i]) + win_len_s))

    gt_list = list(fall_intervals_s)
    hit_gt = set()
    tp = fp = 0
    delays = []
    for t0e, t1e in ev_times:
        covered = [g for g in gt_list if _overlap(t0e, t1e, g[0], g[1])]
        if covered:
            tp += 1
            for g in covered:
                hit_gt.add(gt_list.index(g))
                delays.append(t1e - g[0])
        else:
            fp += 1
    recall = len(hit_gt) / len(gt_list) if gt_list else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "detected": tp > 0,
        "recall": recall, "precision": precision, "f1": f1,
        "delay_s": float(min(delays)) if delays else None,
        "n_events": len(ev), "fp_per_hour": fp / max(total_dur_s / 3600.0, 1e-9),
        "tp": tp, "fp": fp, "n_gt": len(gt_list),
    }


# ---------------------------------------------------------------------------
# AUROC (Mann-Whitney U; no sklearn dependency, handles ties)
# ---------------------------------------------------------------------------
def auc_score(scores, labels):
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, np.int64)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s_all = np.concatenate([pos, neg])
    order = np.argsort(s_all, kind="mergesort")
    ranks = np.empty(len(order), np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    sorted_s = s_all[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def calibrate_threshold(all_probs, all_intervals, all_t0s, win_len_s,
                        min_prec=0.75, grid=None):
    """Validation-set threshold calibration: max recall subject to precision >= min_prec."""
    probs = np.concatenate(all_probs) if all_probs else np.array([])
    if grid is None:
        grid = np.unique(np.round(np.arange(0.05, 0.96, 0.01), 3))
    best = (0.5, {"recall": -1.0, "precision": 0.0, "fallback": True})
    for thr in grid:
        tot_gt = tot_ev = tot_fp = 0
        for p, t0s, gi in zip(all_probs, all_t0s, all_intervals):
            ev = event_aggregate(p, thr)
            for s_i, e_i in ev:
                t0e, t1e = float(t0s[s_i]), float(t0s[e_i]) + win_len_s
                if any(_overlap(t0e, t1e, g[0], g[1]) for g in gi):
                    tot_ev += 1
                else:
                    tot_fp += 1
            tot_gt += len(gi)
        rec = tot_ev / tot_gt if tot_gt else 0.0
        prec = tot_ev / (tot_ev + tot_fp) if (tot_ev + tot_fp) else 1.0
        if prec >= min_prec and rec > best[1]["recall"]:
            best = (float(thr), {"recall": rec, "precision": prec, "fallback": False})
    return best


# ---------------------------------------------------------------------------
# Binary metrics at the three levels
# ---------------------------------------------------------------------------
def _bin_metrics(score, y, thr):
    pred = (np.asarray(score, np.float64) > thr).astype(np.int64)
    y = np.asarray(y, np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / len(y) if len(y) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "n": int(len(y)), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def window_level_metrics(win_probs, win_labels, thr=0.5):
    return _bin_metrics(win_probs, win_labels, thr)


def sequence_level_metrics(seq_scores, seq_labels, thr=0.5):
    return _bin_metrics(seq_scores, seq_labels, thr)


def event_level_metrics(seq_results):
    """Event-level aggregate: recall=Sum(tp)/Sum(n_gt); precision=Sum(tp)/(Sum(tp)+Sum(fp))."""
    tot_tp = sum(r.get("tp", 0) for r in seq_results)
    tot_fp = sum(r.get("fp", 0) for r in seq_results)
    tot_gt = sum(r.get("n_gt", 0) for r in seq_results)
    if tot_gt == 0:
        return {"event_recall": None, "event_precision": None, "event_f1": None,
                "event_accuracy": None, "tp": tot_tp, "fp": tot_fp, "n_gt": tot_gt}
    rec = tot_tp / tot_gt
    prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"event_recall": float(rec), "event_precision": float(prec),
            "event_f1": float(f1), "event_accuracy": float(tot_tp / (tot_gt + tot_fp)),
            "tp": tot_tp, "fp": tot_fp, "n_gt": tot_gt}


def prf(y, score, thr=0.5):
    m = _bin_metrics(score, y, thr)
    return m
