# -*- coding: utf-8 -*-
"""Accelerometer preprocessing (filtering + resampling).

Shared backbone for Experiment 1 (filter / preprocessing comparison). It turns the
raw irregularly-sampled tri-axial signal into an evenly-spaced, denoised signal
used by every downstream algorithm.

Pipeline (fixed order; single-variable comparisons swap exactly one stage):
      raw (t_ms, ax, ay, az)
        |
        +-[1 resample]  linear interpolation onto a fixed rate
        |               (or 'raw' = keep index order, treat as evenly spaced)
        |
        +-[2 filter]    moving average / median / Butterworth low-pass (3/5/10/20 Hz)
                        / Savitzky-Golay / none
                        (all gravity-preserving, zero-phase -- see rationale below)
       -> canonical signal (t_s, ax, ay, az, mag)
"""
import numpy as np
from scipy import signal as sps

import acc_loader

# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
RESAMPLE_CHOICES = ["raw", 50.0, 60.0, 100.0]
RAW_FS = 62.5      # nominal dataset rate (median interval 16 ms across sequences)

def _uniform_grid(t_s, fs):
    return np.arange(t_s[0], t_s[-1] + 1e-12, 1.0 / fs)


def resample_signal(t_s, sig, fs):
    """Linear interpolation onto a fixed rate fs. sig: [N] or [N,C]."""
    grid = _uniform_grid(t_s, fs)
    if sig.ndim == 1:
        return grid, np.interp(grid, t_s, sig)
    cols = [np.interp(grid, t_s, sig[:, c]) for c in range(sig.shape[1])]
    return grid, np.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
# Design choice: every candidate filter preserves the gravity DC component.
# High-pass / band-pass are deliberately excluded: the absolute magnitude is the
# physical basis of the impact-peak and free-fall (|a| -> 0) criteria. Removing
# gravity destroys the 1 g baseline, so "3-10 g impact" and "|a| < 0.6 g weightless"
# lose their absolute reference, and the physical thresholds of rule algorithm A1
# become meaningless.
FILTER_CHOICES = ["none", "ma", "median", "butter5", "butter10", "butter20", "savgol"]
BUTTER_CUTOFF = {"butter3": 3.0, "butter5": 5.0, "butter10": 10.0, "butter20": 20.0}


def _sos_filtfilt(sos, x):
    """Zero-phase filtering with padlen tightened for short signals.

    Note: every filter here is zero-phase (non-causal; symmetric kernel / filtfilt),
    so no phase lag is introduced -- fine for offline analysis, but NOT usable for
    real-time streaming inference. A deployed system must switch to a causal
    implementation and accept a fixed group delay.
    """
    n = len(x)
    if n < 12:
        return x.copy()
    padlen = min(3 * (2 * sos.shape[0] + 1), n - 1)
    return sps.sosfiltfilt(sos, x, padlen=padlen)


def apply_filter(x, method, fs, k=5):
    """Filter a single- or multi-channel signal. x: [N] or [N,C]."""
    if method == "none":
        return x
    x2 = x if x.ndim == 2 else x[:, None]
    out = np.empty_like(x2)
    if method == "ma":                       # moving average (symmetric kernel, zero-phase)
        w = np.ones(k) / k
        for c in range(x2.shape[1]):
            out[:, c] = np.convolve(x2[:, c], w, mode="same")
    elif method == "median":                 # median filter (symmetric window, zero-phase, kills spikes)
        for c in range(x2.shape[1]):
            out[:, c] = sps.medfilt(x2[:, c], kernel_size=k if k % 2 else k + 1)
    elif method in BUTTER_CUTOFF:            # 4th-order Butterworth low-pass (gravity kept, zero-phase)
        fc = min(BUTTER_CUTOFF[method], 0.45 * fs)
        sos = sps.butter(4, fc, btype="low", fs=fs, output="sos")
        for c in range(x2.shape[1]):
            out[:, c] = _sos_filtfilt(sos, x2[:, c])
    elif method == "savgol":                 # Savitzky-Golay (polynomial smoothing, shape preserving)
        wl = k if k % 2 else k + 1
        wl = min(wl, (len(x2) // 2) * 2 - 1)
        wl = max(wl, 5)
        for c in range(x2.shape[1]):
            out[:, c] = sps.savgol_filter(x2[:, c], window_length=wl, polyorder=2)
    else:
        raise ValueError(f"unknown filter method {method}")
    return out if x.ndim == 2 else out[:, 0]


# ---------------------------------------------------------------------------
# Single entry point
# ---------------------------------------------------------------------------
def build_signal(seq, resample="raw", filt="none", fs_target=60.0, k=5,
                 force_mag=True):
    """Raw CSV -> canonical signal.

    Args:
      resample : 'raw' (no resampling, index treated as uniform) or a target rate
      filt     : one of FILTER_CHOICES
      fs_target: target rate used when resample != 'raw'
      k        : kernel length for moving average / median / SG

    Returns a dict:
      t_s [M] / ax,ay,az [M] / mag [M] / fs / seq / n_raw / fs_nom
    """
    raw = acc_loader.read_acc(seq)
    t_s, ax, ay, az = raw["t_s"], raw["ax"], raw["ay"], raw["az"]

    if resample == "raw":
        # No timestamp-based resampling: index is treated as uniform at the nominal
        # 62.5 Hz rate (timestamp jitter ignored). This makes window timestamps drift
        # against wall-clock time -- exactly the effect this comparison quantifies.
        fs = RAW_FS
        t_out = np.arange(len(ax), dtype=np.float64) / fs
        sig = np.stack([ax, ay, az], axis=1)
    else:
        fs = float(resample if resample is not True else fs_target)
        t_out, sig = resample_signal(t_s, np.stack([ax, ay, az], axis=1), fs)

    sig = apply_filter(sig, filt, fs, k=k)
    axo, ayo, azo = sig[:, 0], sig[:, 1], sig[:, 2]
    if force_mag:
        mag = np.sqrt(axo ** 2 + ayo ** 2 + azo ** 2)
    else:
        mag = sig[:, 3] if sig.shape[1] > 3 else np.sqrt(axo ** 2 + ayo ** 2 + azo ** 2)
    return {"seq": seq, "t_s": t_out, "ax": axo, "ay": ayo, "az": azo, "mag": mag,
            "fs": float(fs), "n_raw": raw["n"], "fs_nom": raw["fs_nom"]}


def describe(name):
    """Human-readable method description (written into experiment reports)."""
    return {
        "raw": "no resampling (index treated as uniform @62.5Hz)",
        "none": "no filtering",
        "ma": "moving average (k=5, symmetric kernel, zero-phase)",
        "median": "median filter (k=5, symmetric window, zero-phase)",
        "butter3": "Butterworth 4th-order low-pass 3Hz (gravity kept, zero-phase)",
        "butter5": "Butterworth 4th-order low-pass 5Hz (gravity kept, zero-phase)",
        "butter10": "Butterworth 4th-order low-pass 10Hz (gravity kept, zero-phase)",
        "butter20": "Butterworth 4th-order low-pass 20Hz (gravity kept, zero-phase)",
        "savgol": "Savitzky-Golay (win=11, poly=2)",
    }.get(str(name), str(name))


if __name__ == "__main__":
    for cfg in [("raw", "none"), (60.0, "none"), (60.0, "butter10"), (60.0, "savgol")]:
        d = build_signal("fall-01", resample=cfg[0], filt=cfg[1])
        print(f"resample={cfg[0]} filt={cfg[1]:9s} n={len(d['t_s']):4d} fs={d['fs']:6.2f} "
              f"max|a-1g|={np.max(np.abs(d['mag']-1)):.2f}")
