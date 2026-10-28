# -*- coding: utf-8 -*-
"""39 statistical features (5 groups) - a compact version numerically identical to `00_common/features.py`.

Input: one window [win_n, 4] = (ax, ay, az, mag), evenly spaced and filtered.
Groups and sizes:
  motion 13: magnitude statistics (6) + SMA (1) + per-axis standard deviation (3) + inter-axis correlation (3)
  impact  8: peak deviation / minimum magnitude / free-fall ratio / jerk / impact rise slope + per-axis absolute peak (3)
  orient  8: gravity direction unit vectors of the first/second half window (6) + half-window direction cosine + flip angle (degrees)
  freq    7: dominant frequency / spectral centroid / spectral entropy / spectral bandwidth + energy share of three bands
  calm    3: second-half / whole-window ratios of standard deviation, energy and SMA (post-impact rest)
"""
import numpy as np

from . import config as C

G = C.G
FREEFALL_TH = C.FREEFALL_TH

GROUPS = C.FEATURE_GROUPS
GROUP_DESC = {
    "motion": "Motion intensity (magnitude statistics / per-axis activity / inter-axis correlation)",
    "impact": "Impact and free fall (peak deviation / free fall / jerk / rise slope)",
    "orient": "Posture orientation (gravity direction / in-window flip angle)",
    "freq": "Frequency domain (dominant frequency / spectral centroid / spectral entropy / band energy)",
    "calm": "Post-impact rest (activity share of the second half window)",
}


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

_NAMES = {
    "motion": ["mag_mean", "mag_std", "mag_max", "mag_min", "mag_ptp", "mag_rms",
               "sma", "std_ax", "std_ay", "std_az", "corr_xy", "corr_xz", "corr_yz"],
    "impact": ["peak_dev", "min_mag", "freefall_ratio", "jerk_max", "rise_slope",
               "max_abs_ax", "max_abs_ay", "max_abs_az"],
    "orient": ["dir1_x", "dir1_y", "dir1_z", "dir2_x", "dir2_y", "dir2_z",
               "half_cos", "half_angle_deg"],
    "freq": ["peak_freq", "spec_centroid", "spec_entropy", "spec_bw",
             "e_0_3Hz", "e_3_10Hz", "e_10_25Hz"],
    "calm": ["tail_std_ratio", "tail_energy_ratio", "tail_sma_ratio"],
}


def feature_names(groups=GROUPS):
    names = []
    for g in groups:
        names += _NAMES[g]
    return names


def window_feature(w, fs, groups=GROUPS):
    """One window [win_n,4] -> feature vector [D]."""
    out = []
    for g in groups:
        out += _FUNCS[g](w, fs)
    return np.asarray(out, np.float64)


def extract(windows, fs, groups=GROUPS):
    """[N,win_n,4] -> X [N,D] (nan/inf set to 0)."""
    if len(windows) == 0:
        return np.zeros((0, len(feature_names(groups))), np.float64)
    X = np.stack([window_feature(w, fs, groups) for w in windows])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
