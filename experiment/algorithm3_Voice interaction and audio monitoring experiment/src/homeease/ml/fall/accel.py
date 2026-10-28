"""Accelerometer branch: window features and the transparent rule scorer.

Windows are uniform 50 Hz, causal (the window ends at the evaluation time),
expressed in g. With ``includes_gravity`` the static magnitude is ~1 g and the
dynamic component is |SV-1|; with linear acceleration the dynamic component
is |a| directly. Wrist/phone motion only describes the device, not the whole
body, so post-impact "low activity" is reported as device stillness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from homeease.core.config import AccelRuleCfg
from homeease.schemas.evidence import ScoreSemantics

FS = 50.0

BASE_FEATURES = [
    "peak_sv", "peak_dyn", "jerk_max", "pre_min_sv", "pre_std", "win_mean_dyn", "win_std", "win_energy",
    "win_range",
]
POST_FEATURES = [
    "post_std", "post_mean_dyn", "still_ratio", "post_obs_s", "orient_change_deg", "post_jerk_mean",
]
ALL_FEATURES = BASE_FEATURES + POST_FEATURES


@dataclass
class AccelWindow:
    t: np.ndarray  # (n,) ms
    axes: np.ndarray  # (n,3) g, NaN where masked
    mask: np.ndarray  # (n,) int8
    includes_gravity: bool
    clock_uncertainty_ms: float | None = None
    carriage: str = "unknown"
    rate_hz: float | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class ScoreResult:
    valid: bool
    score: float
    candidate: bool
    peak_time_ms: float | None
    features: dict[str, Any]
    reasons: list[str]
    quality: float
    flags: list[str]
    contradicts: bool = False


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return float("nan")
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def window_features(
    w: AccelWindow,
    post_start_s: float,
    post_min_s: float,
    post_window_s: float,
    peak_search: tuple[float, float] | None = None,
    saturation_g: float = 7.8,
) -> dict[str, Any]:
    """Compute impact and post-impact features. Returns {'valid': False} when no
    searchable impact region with enough valid samples exists."""
    t, ax, m = w.t, w.axes, w.mask.astype(bool)
    n = len(t)
    out: dict[str, Any] = {"valid": False}
    if n < 10 or m.sum() < 10:
        out["reason"] = "insufficient_samples"
        return out
    sv = np.sqrt(np.sum(ax**2, axis=1))
    dyn = np.abs(sv - 1.0) if w.includes_gravity else sv
    t_end = t[-1]
    lo, hi = t[0] + 500.0, t_end - post_min_s * 1000.0
    if peak_search is not None:
        lo, hi = max(lo, peak_search[0]), min(hi, peak_search[1])
    search = (t >= lo) & (t <= hi) & m
    if not search.any():
        out["reason"] = "no_search_region"
        return out
    key = sv if w.includes_gravity else dyn
    idx = np.where(search)[0]
    ip = int(idx[np.nanargmax(key[idx])])
    peak_t = float(t[ip])
    pre = (t >= peak_t - 1500.0) & (t < peak_t - 100.0) & m
    post = (t >= peak_t + post_start_s * 1000.0) & (t <= peak_t + (post_start_s + post_window_s) * 1000.0) & m
    near = (t >= peak_t - 500.0) & (t <= peak_t + 200.0) & m
    dsv = np.abs(np.diff(sv)) * FS
    dsv = np.concatenate([[0.0], dsv])
    dsv[~m] = np.nan
    valid_sv = sv[m]
    out.update(
        valid=True,
        peak_index=ip,
        peak_time_ms=peak_t,
        peak_sv=float(sv[ip]),
        peak_dyn=float(dyn[ip]),
        jerk_max=float(np.nanmax(dsv[near])) if near.any() else 0.0,
        pre_min_sv=float(np.min(sv[pre])) if pre.any() else float(sv[ip]),
        pre_std=float(np.std(sv[pre])) if pre.sum() > 2 else 0.0,
        win_mean_dyn=float(np.mean(dyn[m])),
        win_std=float(np.std(valid_sv)),
        win_energy=float(np.mean(dyn[m] ** 2)),
        win_range=float(np.max(valid_sv) - np.min(valid_sv)),
        post_std=float(np.std(sv[post])) if post.sum() > 2 else float("nan"),
        post_mean_dyn=float(np.mean(dyn[post])) if post.any() else float("nan"),
        still_ratio=float(np.mean(dyn[post] < 0.1)) if post.any() else float("nan"),
        post_obs_s=float(post.sum() / FS),
        orient_change_deg=(
            _angle_deg(np.mean(ax[pre], axis=0), np.mean(ax[post], axis=0))
            if (w.includes_gravity and pre.any() and post.any())
            else float("nan")
        ),
        post_jerk_mean=float(np.nanmean(dsv[post])) if post.any() else float("nan"),
        valid_fraction=float(m.mean()),
        saturated=bool(np.nanmax(np.abs(ax[m])) >= saturation_g),
    )
    return out


def feature_vector(feats: dict[str, Any], names: list[str]) -> np.ndarray:
    v = np.array([float(feats.get(k, np.nan)) for k in names], dtype=np.float64)
    return v


def quality_of(w: AccelWindow, feats: dict[str, Any], target_hz: float, max_unc_ms: float) -> tuple[float, list[str]]:
    flags = list(w.flags)
    q = float(np.mean(w.mask)) if len(w.mask) else 0.0
    if q < 0.95:
        flags.append("gap_in_window")
    if w.rate_hz is not None and w.rate_hz < 0.6 * target_hz:
        q *= max(0.3, w.rate_hz / target_hz)
        flags.append("sampling_rate_low")
    if w.clock_uncertainty_ms is None or w.clock_uncertainty_ms > max_unc_ms:
        q *= 0.5
        flags.append("timestamp_untrusted")
    carriage_factor = {"confirmed": 1.0, "likely": 1.0, "unknown": 0.75, "not_carried": 0.2}.get(w.carriage, 0.75)
    q *= carriage_factor
    if w.carriage in ("unknown", "not_carried"):
        flags.append(f"phone_carriage_{w.carriage}")
    if feats.get("saturated"):
        flags.append("saturated")
    return float(np.clip(q, 0.0, 1.0)), sorted(set(flags))


class AccelRuleScorer:
    """Impact followed by device stillness. The score is a heuristic in [0,1]."""

    channel = "accel"
    model_id = "accel_rule"
    model_version = "1.0"
    score_semantics = ScoreSemantics.heuristic_score
    calibration_id = None

    def __init__(self, cfg: AccelRuleCfg, target_hz: float = FS, max_unc_ms: float = 150.0, saturation_g: float = 7.8):
        self.cfg = cfg
        self.target_hz = target_hz
        self.max_unc_ms = max_unc_ms
        self.saturation_g = saturation_g
        self.strong_threshold = cfg.strong_score
        self.candidate_threshold = 0.35
        self.window_ms = cfg.window_s * 1000.0
        self.step_ms = float(cfg.step_ms)
        self.min_post_s = cfg.post_min_s

    def score(self, w: AccelWindow, peak_search: tuple[float, float] | None = None, truncated: bool = False) -> ScoreResult:
        c = self.cfg
        post_min = 0.4 if truncated else c.post_min_s
        f = window_features(w, c.post_start_s, post_min, c.post_window_s, peak_search, self.saturation_g)
        if not f.get("valid"):
            return ScoreResult(False, 0.0, False, None, f, [f.get("reason", "invalid_window")], 0.0, list(w.flags))
        q, flags = quality_of(w, f, self.target_hz, self.max_unc_ms)
        reasons: list[str] = []
        peak = f["peak_sv"] if w.includes_gravity else 1.0 + f["peak_dyn"]
        impact = peak >= c.impact_threshold_g
        reasons.append("impact_detected" if impact else "no_impact")
        post_std = f["post_std"]
        enough_post = f["post_obs_s"] >= (0.35 if truncated else 0.8)
        if truncated:
            reasons.append("truncated_post_window")
        if not enough_post or not np.isfinite(post_std):
            reasons.append("insufficient_post_window")
            still = False
            s_term = 0.0
        else:
            still = post_std <= c.still_std_g
            reasons.append("post_impact_low_activity" if still else "post_impact_activity")
            s_term = float(np.clip(1.0 - post_std / (2.0 * c.still_std_g), 0.0, 1.0))
        i_term = float(np.clip((peak - 1.2) / 1.8, 0.0, 1.0))
        score = float(np.sqrt(i_term * s_term))
        candidate = bool(impact and still)
        return ScoreResult(True, score, candidate, f["peak_time_ms"], f, reasons, q, flags)
