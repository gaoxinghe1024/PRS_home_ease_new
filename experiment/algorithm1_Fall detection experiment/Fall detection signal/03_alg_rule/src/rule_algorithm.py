# -*- coding: utf-8 -*-
"""A1 threshold rule algorithm (learning-free baseline).

Idea: a fall leaves three interpretable physical imprints on the tri-axial
accelerometer. Each imprint is mapped to a 0-1 sub-score by a fixed threshold, and
a weighted sum yields a continuous per-window fall score.

Three sub-scores (matching "impact peak magnitude / motion intensity /
post-impact inactivity" in the literature):
  s_impact   impact peak     -- max(| |a| - 1g |) in the window, linear over [2g, 6g]
  s_active   motion intensity-- std(|a|) in the window, linear over [0.2, 1.0]
  s_calm     post-impact rest-- std of second half / std of whole window, reversed
               over [1.2, 0.3]: lying still after the fall -> small ratio -> high
               score; steady walking -> ratio ~ 1 -> low score

window score = 0.50*s_impact + 0.20*s_active + 0.30*s_calm

Why absolute g thresholds instead of data-driven ones: acceleration is expressed in
g, and the physical scale of an impact (typically 3-10 g for a fall) is stable
across devices and mounting positions, so the rule transfers. Normalised-coordinate
thresholds on the video path, by contrast, depend on body scale and camera view --
one reason the video-path rule baseline (AUROC 0.55) is far weaker than the
sensor-path rule baseline.

Interface matches A2/A3/A4: fit(train) / predict(wins).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "00_common"))

G = 1.0
IMPACT_LO, IMPACT_HI = 2.0, 6.0
ACTIVE_LO, ACTIVE_HI = 0.20, 1.00
CALM_HI_R, CALM_LO_R = 1.20, 0.30           # ratio 1.20 -> score 0, ratio 0.30 -> score 1
W_IMPACT, W_ACTIVE, W_CALM = 0.50, 0.20, 0.30


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def window_sub_scores(w):
    """Single window -> (s_impact, s_active, s_calm). w: [win_n,4] (ax,ay,az,mag)."""
    mag = np.asarray(w)[:, 3]
    peak_dev = float(np.max(np.abs(mag - G)))
    std_all = float(np.std(mag))
    h = len(mag) // 2
    std_tail = float(np.std(mag[h:])) if h > 0 else std_all
    tail_ratio = std_tail / (std_all + 1e-9)

    s_impact = _clip01((peak_dev - IMPACT_LO) / (IMPACT_HI - IMPACT_LO))
    s_active = _clip01((std_all - ACTIVE_LO) / (ACTIVE_HI - ACTIVE_LO))
    s_calm = _clip01((CALM_HI_R - tail_ratio) / (CALM_HI_R - CALM_LO_R))
    return s_impact, s_active, s_calm


def window_score(w):
    si, sa, sc = window_sub_scores(w)
    return W_IMPACT * si + W_ACTIVE * sa + W_CALM * sc


class RuleAlgorithm:
    """A1 training-free threshold rule (fixed physical thresholds, no fitted parameters)."""

    name = "A1_threshold_rule"

    def __init__(self, fs=60.0, **kw):
        self.fs = fs

    def fit(self, train):
        # No learning: only record fs, keeping the A2/A3/A4 interface identical
        self.fs = train.get("fs", self.fs)
        return self

    def predict(self, wins):
        wins = np.asarray(wins)
        if len(wins) == 0:
            return np.zeros(0, np.float64)
        return np.array([window_score(w) for w in wins], np.float64)

    # Exposed so the report can ablate individual sub-scores
    def predict_subscores(self, wins):
        S = np.array([window_sub_scores(w) for w in wins], np.float64)
        return {"impact": S[:, 0], "active": S[:, 1], "calm": S[:, 2]}
