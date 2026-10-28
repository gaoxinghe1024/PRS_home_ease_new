# -*- coding: utf-8 -*-
"""Window-, sequence- and event-level metrics with threshold calibration."""

import numpy as np

from windowize import MERGE_GAP_WIN


def event_aggregate(win_probs, thr, merge_gap=MERGE_GAP_WIN):
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
    ev = event_aggregate(win_probs, thr)
    ev_times = []
    for s_i, e_i in ev:
        t_start = float(t0_s[s_i])
        t_alert = float(t0_s[e_i] + win_len_s)
        ev_times.append((t_start, t_alert))

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
            for g in covered:
                delays.append(t1e - g[0])
        else:
            fp += 1
    recall = len(hit_gt) / len(gt_list) if gt_list else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    detected = tp > 0
    fp_per_hour = fp / max(total_dur_s / 3600.0, 1e-9)
    return {
        "detected": detected,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "delay_s": float(min(delays)) if delays else None,
        "n_events": len(ev),
        "fp_per_hour": fp_per_hour,
        "tp": tp,
        "fp": fp,
        "n_gt": len(gt_list),
    }


def seq_auroc(scores, labels):
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, np.int64)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), np.float64)
    s_all = np.concatenate([pos, neg])
    ranks[order] = np.arange(1, len(order) + 1)
    sorted_s = s_all[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def calibrate_threshold(
    all_probs, all_labels, all_intervals, all_t0s, win_len_s, min_prec=0.75, grid=None
):
    probs = np.concatenate(all_probs) if all_probs else np.array([])
    if grid is None:
        grid = np.unique(np.round(np.arange(0.05, 0.96, 0.01), 3))
    best = (0.5, {"recall": -1, "precision": 0.0})
    for thr in grid:
        rs, ps, tot_gt, tot_ev, tot_fp = [], [], 0, 0, 0
        for p, t0s, gi in zip(all_probs, all_t0s, all_intervals):
            ev = event_aggregate(p, thr)
            ev_times = [
                (float(t0s[s_i]), float(t0s[e_i]) + win_len_s) for s_i, e_i in ev
            ]
            hit = 0
            tp = fp = 0
            for t0e, t1e in ev_times:
                cov = [g for g in gi if _overlap(t0e, t1e, g[0], g[1])]
                if cov:
                    tp += 1
                    hit += len(cov)
                else:
                    fp += 1
            rs.append(hit / len(gi) if gi else 1.0)
            ps.append(tp / (tp + fp) if (tp + fp) else 1.0)
            tot_gt += len(gi)
            tot_ev += tp
            tot_fp += fp
        rec = tot_ev / tot_gt if tot_gt else 0.0
        prec = tot_ev / (tot_ev + tot_fp) if (tot_ev + tot_fp) else 1.0
        if prec >= min_prec and rec > best[1]["recall"]:
            best = (float(thr), {"recall": rec, "precision": prec})
    best[1]["fallback"] = best[1]["recall"] < 0
    return best


def pool_metrics(seq_results):
    n = len(seq_results)
    det = [r["detected"] for r in seq_results]
    rec = [r["recall"] for r in seq_results]
    prec = [r["precision"] for r in seq_results]
    f1 = [r["f1"] for r in seq_results]
    delays = [r["delay_s"] for r in seq_results if r["delay_s"] is not None]
    fph = [r["fp_per_hour"] for r in seq_results]
    return {
        "n_sequences": n,
        "accuracy": float(np.mean(det)),
        "recall_mean": float(np.mean(rec)),
        "precision_mean": float(np.mean(prec)),
        "f1_mean": float(np.mean(f1)),
        "delay_mean_s": float(np.mean(delays)) if delays else None,
        "fp_per_hour": float(np.sum(fph) / max(n, 1)),
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def window_level_metrics(win_probs, win_labels, thr=0.5):
    p = np.asarray(win_probs, np.float64)
    y = np.asarray(win_labels, np.int64)
    if len(p) == 0:
        return {
            "acc": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "n": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
    pred = (p > thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "n": len(y),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def window_auroc(win_probs, win_labels):
    return seq_auroc(win_probs, np.asarray(win_labels, np.int64))


def sequence_level_metrics(seq_scores, seq_labels, thr=0.5):
    s = np.asarray(seq_scores, np.float64)
    y = np.asarray(seq_labels, np.int64)
    pred = (s > thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "n": len(y),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def event_level_metrics(seq_results):
    tot_tp = sum(r.get("tp", 0) for r in seq_results)
    tot_fp = sum(r.get("fp", 0) for r in seq_results)
    tot_gt = sum(r.get("n_gt", 0) for r in seq_results)
    if tot_gt == 0:
        return {
            "event_recall": None,
            "event_precision": None,
            "event_f1": None,
            "event_accuracy": None,
            "tp": tot_tp,
            "fp": tot_fp,
            "n_gt": tot_gt,
        }
    rec = tot_tp / tot_gt
    prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = tot_tp / (tot_gt + tot_fp)
    return {
        "event_recall": float(rec),
        "event_precision": float(prec),
        "event_f1": float(f1),
        "event_accuracy": float(acc),
        "tp": tot_tp,
        "fp": tot_fp,
        "n_gt": tot_gt,
    }
