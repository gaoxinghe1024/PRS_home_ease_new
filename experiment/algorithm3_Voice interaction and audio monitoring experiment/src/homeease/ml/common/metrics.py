"""Metrics with explicit denominators.

Every function returns counts alongside rates so that reports can show sample
sizes; rates are ``None`` (not 0) when the denominator is empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


def _div(a: float, b: float) -> float | None:
    return None if b == 0 else float(a) / float(b)


def binary_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    yt, yp = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = None if (p is None or r is None or (p + r) == 0) else 2 * p * r / (p + r)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f1, "fpr": _div(fp, fp + tn), "n": len(yt), "positives": int(yt.sum())}


def multiclass_report(y_true: Sequence[str], y_pred: Sequence[str], labels: list[str]) -> dict[str, Any]:
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b in zip(y_true, y_pred):
        if a in idx and b in idx:
            cm[idx[a], idx[b]] += 1
    per = {}
    f1s = []
    for i, l in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p, r = _div(tp, tp + fp), _div(tp, tp + fn)
        f1 = None if (p is None or r is None or (p + r) == 0) else 2 * p * r / (p + r)
        per[l] = {"precision": p, "recall": r, "f1": f1, "support": int(cm[i, :].sum())}
        if cm[i, :].sum() > 0:
            f1s.append(f1 or 0.0)
    acc = _div(int(np.trace(cm)), int(cm.sum()))
    return {"labels": labels, "confusion": cm.tolist(), "per_class": per, "macro_f1": float(np.mean(f1s)) if f1s else None, "accuracy": acc, "n": int(cm.sum())}


def pr_auc(y_true: Sequence[int], score: Sequence[float]) -> float | None:
    from sklearn.metrics import average_precision_score

    yt = np.asarray(y_true)
    if yt.sum() == 0 or yt.sum() == len(yt):
        return None
    return float(average_precision_score(yt, np.asarray(score)))


def roc_auc(y_true: Sequence[int], score: Sequence[float]) -> float | None:
    from sklearn.metrics import roc_auc_score

    yt = np.asarray(y_true)
    if yt.sum() == 0 or yt.sum() == len(yt):
        return None
    return float(roc_auc_score(yt, np.asarray(score)))


def calibration(y_true: Sequence[int], prob: Sequence[float], n_bins: int = 10) -> dict[str, Any]:
    """Expected calibration error with equal-width bins, plus Brier score."""
    yt, p = np.asarray(y_true, dtype=float), np.clip(np.asarray(prob, dtype=float), 0, 1)
    if len(yt) == 0:
        return {"ece": None, "brier": None, "bins": [], "n": 0, "binning": f"{n_bins} equal-width"}
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        sel = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1]))
        if not sel.any():
            continue
        conf, acc = float(p[sel].mean()), float(yt[sel].mean())
        ece += sel.mean() * abs(conf - acc)
        bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(sel.sum()), "confidence": conf, "accuracy": acc})
    return {"ece": float(ece), "brier": float(np.mean((p - yt) ** 2)), "bins": bins, "n": int(len(yt)), "binning": f"{n_bins} equal-width"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def grouped_bootstrap(groups: Sequence[str], stat_fn: Any, n_boot: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    """Percentile CI resampling whole groups (participants/recordings), never windows."""
    ug = sorted(set(groups))
    if len(ug) < 3:
        return None
    rng = np.random.default_rng(seed)
    garr = np.asarray(groups)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([np.where(garr == g)[0] for g in pick])
        v = stat_fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return None
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ------------------------------------------------------------- event level
@dataclass
class Alert:
    t_ms: float  # time the alert/decision was issued
    level: str  # PENDING | EMERGENCY
    anchor_ms: float | None = None


def match_events(truth: list[dict[str, Any]], alerts: list[Alert], pre_ms: float = 1000.0, post_ms: float = 30000.0) -> dict[str, Any]:
    """One-to-one matching of alerts to annotated falls.

    An alert matches a fall when it is issued within [onset - pre, onset + post].
    Several alerts for one fall count once (highest level kept); unmatched alerts are
    false alerts. Returns per-fall outcomes so that every fall is accounted for.
    """
    alerts = sorted(alerts, key=lambda a: a.t_ms)
    used = [False] * len(alerts)
    per_fall = []
    for f in truth:
        onset = float(f["onset_ms"])
        lo, hi = onset - pre_ms, onset + post_ms
        best_level, first_t, emerg_t = None, None, None
        for i, a in enumerate(alerts):
            if lo <= a.t_ms <= hi:
                used[i] = True
                if first_t is None:
                    first_t = a.t_ms
                if a.level == "EMERGENCY":
                    best_level = "EMERGENCY"
                    emerg_t = a.t_ms if emerg_t is None else emerg_t
                elif best_level is None:
                    best_level = a.level
        per_fall.append({
            "onset_ms": onset, "matched_level": best_level, "first_alert_latency_s": None if first_t is None else (first_t - onset) / 1000.0,
            "emergency_latency_s": None if emerg_t is None else (emerg_t - onset) / 1000.0,
        })
    false_alerts = [{"t_ms": a.t_ms, "level": a.level} for i, a in enumerate(alerts) if not used[i]]
    return {"per_fall": per_fall, "false_alerts": false_alerts}


def summarize_event_level(rows: list[dict[str, Any]], hours: float | None = None, continuous_exposure: bool = False) -> dict[str, Any]:
    """rows: one per recording with keys group, n_falls, per_fall[], false_alerts[], kind."""
    falls = [pf for r in rows for pf in r["per_fall"]]
    n = len(falls)
    any_hit = sum(1 for f in falls if f["matched_level"] in ("PENDING", "EMERGENCY"))
    emerg_hit = sum(1 for f in falls if f["matched_level"] == "EMERGENCY")
    fe = sum(1 for r in rows for a in r["false_alerts"] if a["level"] == "EMERGENCY")
    fp_pending = sum(1 for r in rows for a in r["false_alerts"] if a["level"] == "PENDING")
    total_emerg_alerts = emerg_hit + fe
    neg_units = [r for r in rows if r["n_falls"] == 0]
    neg_with_emerg = sum(1 for r in neg_units if any(a["level"] == "EMERGENCY" for a in r["false_alerts"]))
    neg_with_any = sum(1 for r in neg_units if r["false_alerts"])
    lat = [f["first_alert_latency_s"] for f in falls if f["first_alert_latency_s"] is not None]
    elat = [f["emergency_latency_s"] for f in falls if f["emergency_latency_s"] is not None]
    out = {
        "falls": n,
        "recall_any_alert": _div(any_hit, n),
        "recall_any_alert_ci95": wilson(any_hit, n),
        "recall_emergency_only": _div(emerg_hit, n),
        "recall_emergency_only_ci95": wilson(emerg_hit, n),
        "missed_any": n - any_hit,
        "not_emergency": n - emerg_hit,
        "emergency_alerts_total": total_emerg_alerts,
        "emergency_precision": _div(emerg_hit, total_emerg_alerts),
        "false_emergencies": fe,
        "false_pending": fp_pending,
        "negative_units": len(neg_units),
        "event_fpr_emergency_per_negative_recording": _div(neg_with_emerg, len(neg_units)),
        "event_fpr_any_alert_per_negative_recording": _div(neg_with_any, len(neg_units)),
        "fpr_unit": "one non-fall recording = one negative unit",
        "latency_first_alert_s": _lat(lat),
        "latency_emergency_s": _lat(elat),
    }
    if hours and hours > 0:
        out["monitored_hours"] = hours
        out["false_emergency_per_hour"] = fe / hours
        out["false_emergency_per_day"] = (fe / hours) * 24 if continuous_exposure else None
        if not continuous_exposure:
            out["false_emergency_per_day_note"] = "unavailable: short clips are not continuous daily exposure"
    return out


def _lat(v: list[float]) -> dict[str, Any] | None:
    if not v:
        return None
    a = np.asarray(v)
    return {"n": int(len(a)), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "within_30s": float(np.mean(a <= 30.0))}


# ------------------------------------------------------------- counting
def counting_metrics(pred: Sequence[int], truth: Sequence[int]) -> dict[str, Any]:
    p, t = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    n = len(t)
    if n == 0:
        return {"sessions": 0}
    err = np.abs(p - t)
    nz = t > 0
    rel = np.where(nz, err / np.where(nz, t, 1), np.nan)
    return {
        "sessions": int(n),
        "mae": float(err.mean()),
        "mape": float(np.nanmean(rel)) if nz.any() else None,
        "mape_excluded_zero_sessions": int((~nz).sum()),
        "exact_count_accuracy": float(np.mean(err == 0)),
        "share_relative_error_le_5pct": float(np.mean(rel[nz] <= 0.05)) if nz.any() else None,
        "total_abs_relative_error": float(err.sum() / t.sum()) if t.sum() > 0 else None,
        "zero_rep_sessions": int((~nz).sum()),
        "false_counts_in_zero_sessions": int(p[~nz].sum()) if (~nz).any() else 0,
        "per_session_relative_error": [None if not np.isfinite(x) else float(x) for x in rel],
    }


def match_cycles(pred_ends: Iterable[float], truth: list[dict[str, Any]], tol_ms: float = 1500.0) -> dict[str, Any]:
    pe = sorted(pred_ends)
    used = [False] * len(pe)
    tp, offsets = 0, []
    for c in truth:
        end = float(c["end_ms"])
        best, bd = -1, 1e18
        for i, x in enumerate(pe):
            if not used[i] and abs(x - end) <= tol_ms and abs(x - end) < bd:
                best, bd = i, abs(x - end)
        if best >= 0:
            used[best] = True
            tp += 1
            offsets.append((pe[best] - end) / 1000.0)
    return {
        "true_cycles": len(truth), "pred_cycles": len(pe), "matched": tp, "missed": len(truth) - tp, "spurious": len(pe) - tp,
        "cycle_recall": _div(tp, len(truth)), "cycle_precision": _div(tp, len(pe)),
        "timing_offset_s_mean": float(np.mean(offsets)) if offsets else None,
    }
