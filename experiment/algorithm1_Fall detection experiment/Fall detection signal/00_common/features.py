# -*- coding: utf-8 -*-
"""Accelerometer feature extraction and feature selection.

Shared backbone for Experiment 2 (feature extraction / selection comparison).

Feature groups, split by physical meaning so that group-wise ablation is possible:
  motion : motion intensity -- magnitude statistics, per-axis activity, inter-axis correlation
  impact : impact / free-fall -- peak deviation, free fall, jerk, impact rise slope
           (matches "impact peak magnitude" in the literature)
  orient : posture / orientation -- gravity direction estimate, half-window direction flip angle
           (captures "did the body end up lying down")
  freq   : spectral -- dominant frequency, spectral centroid, spectral entropy, band energies
  calm   : post-impact inactivity -- activity ratio of the second half of the window

Per-window input is [win_n, 4] = (ax, ay, az, mag), evenly spaced and already filtered.
"""
import os
import numpy as np


def tree_jobs():
    """Tree-model thread count: all cores when sequential (-1), capped at 2 under fold-level multiprocessing."""
    try:
        return int(os.environ.get("FALL_ACC_TREE_JOBS", "-1"))
    except ValueError:
        return -1


G = 1.0                    # gravity baseline (g)
FREEFALL_TH = 0.6          # free-fall threshold (g)
DEFAULT_GROUPS = ("motion", "impact", "orient", "freq", "calm")
GROUP_DESC = {
    "motion": "motion intensity (magnitude stats / per-axis activity / inter-axis correlation)",
    "impact": "impact & free-fall (peak deviation / free fall / jerk / rise slope)",
    "orient": "posture & orientation (gravity direction / half-window flip angle)",
    "freq": "spectral (dominant freq / centroid / entropy / band energy)",
    "calm": "post-impact inactivity (second-half activity ratio)",
}


# ---------------------------------------------------------------------------
# Single-window features
# ---------------------------------------------------------------------------
def _motion(w, fs):
    ax, ay, az, mag = w[:, 0], w[:, 1], w[:, 2], w[:, 3]
    f = [
        float(np.mean(mag)), float(np.std(mag)), float(np.max(mag)),
        float(np.min(mag)), float(np.ptp(mag)), float(np.sqrt(np.mean(mag ** 2))),
        float(np.mean(np.abs([ax, ay, az]))),                 # SMA
        float(np.std(ax)), float(np.std(ay)), float(np.std(az)),
    ]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(np.stack([ax, ay, az]))
    c = np.nan_to_num(c, nan=0.0)
    f += [float(c[0, 1]), float(c[0, 2]), float(c[1, 2])]
    return f


def _impact(w, fs):
    mag = w[:, 3]
    dt = 1.0 / max(fs, 1e-6)
    d = np.diff(mag) / dt
    peak_dev = float(np.max(np.abs(mag - G)))
    min_mag = float(np.min(mag))
    free_ratio = float(np.mean(mag < FREEFALL_TH))
    jerk_max = float(np.max(np.abs(d))) if len(d) else 0.0
    i_pk = int(np.argmax(np.abs(mag - G)))
    rise = float((mag[i_pk] - mag[0]) / max(i_pk * dt, dt))    # impact rise slope (g/s)
    return [peak_dev, min_mag, free_ratio, jerk_max, rise,
            float(np.max(np.abs(w[:, 0]))), float(np.max(np.abs(w[:, 1]))),
            float(np.max(np.abs(w[:, 2])))]


def _orient(w, fs):
    h = len(w) // 2
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.zeros_like(v)
    m1 = unit(np.median(w[:h, :3], axis=0))
    m2 = unit(np.median(w[h:, :3], axis=0))
    cos = float(np.clip(np.dot(m1, m2), -1.0, 1.0))
    return [float(m1[0]), float(m1[1]), float(m1[2]),
            float(m2[0]), float(m2[1]), float(m2[2]),
            cos, float(np.degrees(np.arccos(cos)))]


def _freq(w, fs):
    mag = w[:, 3] - np.mean(w[:, 3])
    n = len(mag)
    if n < 8:
        return [0.0] * 7
    win = np.hanning(n)
    sp = np.abs(np.fft.rfft(mag * win))
    fr = np.fft.rfftfreq(n, d=1.0 / fs)
    p = sp ** 2
    tot = p.sum() + 1e-12
    pn = p / tot
    peak_f = float(fr[int(np.argmax(sp[1:]) + 1)]) if len(sp) > 1 else 0.0
    centroid = float((fr * pn).sum())
    entropy = float(-(pn[pn > 0] * np.log(pn[pn > 0])).sum() / np.log(len(pn) + 1e-12))
    bw = float(np.sqrt(((fr - centroid) ** 2 * pn).sum()))
    def band(a, b):
        m = (fr >= a) & (fr < b)
        return float(pn[m].sum())
    return [peak_f, centroid, entropy, bw, band(0, 3), band(3, 10), band(10, 25)]


def _calm(w, fs):
    h = len(w) // 2
    mag = w[:, 3]
    s_all = float(np.std(mag)) + 1e-9
    s_tail = float(np.std(mag[h:]))
    e = mag ** 2
    e_all = float(e.mean()) + 1e-9
    e_tail = float(e[h:].mean())
    sma = np.abs(w[:, :3]).sum(axis=1)
    sma_all = float(sma.mean()) + 1e-9
    sma_tail = float(sma[h:].mean())
    return [s_tail / s_all, e_tail / e_all, sma_tail / sma_all]


_FUNCS = {"motion": _motion, "impact": _impact, "orient": _orient,
          "freq": _freq, "calm": _calm}
_COUNTS = {"motion": 13, "impact": 8, "orient": 8, "freq": 7, "calm": 3}


def feature_names(groups=DEFAULT_GROUPS):
    names = []
    for g in groups:
        if g == "motion":
            names += ["mag_mean", "mag_std", "mag_max", "mag_min", "mag_ptp", "mag_rms",
                      "sma", "std_ax", "std_ay", "std_az", "corr_xy", "corr_xz", "corr_yz"]
        elif g == "impact":
            names += ["peak_dev", "min_mag", "freefall_ratio", "jerk_max", "rise_slope",
                      "max_abs_ax", "max_abs_ay", "max_abs_az"]
        elif g == "orient":
            names += ["dir1_x", "dir1_y", "dir1_z", "dir2_x", "dir2_y", "dir2_z",
                      "half_cos", "half_angle_deg"]
        elif g == "freq":
            names += ["peak_freq", "spec_centroid", "spec_entropy", "spec_bw",
                      "e_0_3Hz", "e_3_10Hz", "e_10_25Hz"]
        elif g == "calm":
            names += ["tail_std_ratio", "tail_energy_ratio", "tail_sma_ratio"]
    return names


def window_feature(w, fs, groups=DEFAULT_GROUPS):
    """Single window [win_n,4] -> feature vector [D]."""
    out = []
    for g in groups:
        out += _FUNCS[g](w, fs)
    return np.asarray(out, np.float64)


def extract(windows, fs, groups=DEFAULT_GROUPS):
    """[N,win_n,4] -> X [N,D] (nan/inf set to 0)."""
    if len(windows) == 0:
        return np.zeros((0, len(feature_names(groups))), np.float64)
    X = np.stack([window_feature(w, fs, groups) for w in windows])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Feature selection (compared in Experiment 2)
# ---------------------------------------------------------------------------
SELECTION_CHOICES = ["all", "var", "mi", "imp", "rfe", "l1"]
SELECTION_DESC = {
    "all": "all features (no selection)",
    "var": "variance filter (VarianceThreshold=0.01)",
    "mi": "mutual information top-K (SelectKBest / mutual_info)",
    "imp": "random-forest importance top-K",
    "rfe": "recursive feature elimination (RFE + random forest)",
    "l1": "L1-regularised logistic regression (SelectFromModel)",
}


def select_features(X, y, method="all", k=20, seed=0):
    """Return (X_sel, kept_idx, name).

    Every selector is fitted inside the training fold only, so test information
    never leaks into feature selection.
    """
    from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                           mutual_info_classif, RFE, SelectFromModel)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    n_feat = X.shape[1]
    k = int(min(k, n_feat))
    if method == "all" or n_feat <= 1:
        idx = np.arange(n_feat)
    elif method == "var":
        sel = VarianceThreshold(threshold=0.01).fit(X)
        idx = np.where(sel.get_support())[0]
    elif method == "mi":
        sel = SelectKBest(lambda a, b: mutual_info_classif(a, b, random_state=seed), k=k).fit(X, y)
        idx = np.where(sel.get_support())[0]
    elif method == "imp":
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed,
                                    n_jobs=tree_jobs(), class_weight="balanced").fit(X, y)
        idx = np.argsort(rf.feature_importances_)[::-1][:k]
        idx = np.sort(idx)
    elif method == "rfe":
        rf = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=seed, n_jobs=tree_jobs())
        sel = RFE(rf, n_features_to_select=k, step=0.2).fit(X, y)
        idx = np.where(sel.support_)[0]
    elif method == "l1":
        lr = LogisticRegression(penalty="l1", C=0.5, solver="liblinear",
                                max_iter=2000, class_weight="balanced").fit(
            (X - X.mean(0)) / (X.std(0) + 1e-9), y)
        sel = SelectFromModel(lr, prefit=True)
        idx = np.where(sel.get_support())[0]
        if len(idx) == 0:
            idx = np.argsort(np.abs(lr.coef_[0]))[::-1][:k]
            idx = np.sort(idx)
    else:
        raise ValueError(f"unknown feature selection method {method}")
    if len(idx) == 0:
        idx = np.arange(n_feat)
    return X[:, idx], idx, SELECTION_DESC.get(method, method)
