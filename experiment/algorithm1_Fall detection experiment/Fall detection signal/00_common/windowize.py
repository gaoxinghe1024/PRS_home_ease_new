# -*- coding: utf-8 -*-
"""Windowing (accelerometer signal -> window samples).

Protocol:
  Window 2.0 s, stride 0.25 s.
  The sampling rate comes from preprocessing (default: resampled to 60 Hz -> 120
  samples per window, 15 samples per stride).
  Why a 0.25 s stride: a fall impact lasts 0.3-0.6 s, and a coarser stride can skip
  the impact peak entirely. At 0.25 s, any 0.5 s event is fully covered by at
  least two windows.

Window labels (same convention as the video path, any-hit semantics):
  window [t0, t0+win] overlaps any GT fall interval -> positive window.
"""
import numpy as np

WIN_S = 2.0
STEP_S = 0.25
MERGE_GAP_WIN = 2          # event merging: hits closer than 2 windows (0.5 s) belong to one event


def n_samples(fs, sec):
    return max(1, int(round(fs * sec)))


def make_windows(sig, fs, win_s=WIN_S, step_s=STEP_S):
    """sig [T,C] or [T] -> windows [N,win_n,C], t0_idx [N]."""
    sig = np.asarray(sig)
    if sig.ndim == 1:
        sig = sig[:, None]
    win_n, step_n = n_samples(fs, win_s), n_samples(fs, step_s)
    T = len(sig)
    if T < win_n:
        return np.zeros((0, win_n, sig.shape[1]), np.float32), []
    t0s = list(range(0, T - win_n + 1, step_n))
    wins = np.stack([sig[i:i + win_n] for i in t0s]).astype(np.float32)
    return wins, t0s


def window_start_times(t_s, t0_idx, win_s=WIN_S, fs=None):
    """Window start times in seconds."""
    return np.asarray([float(t_s[i]) for i in t0_idx], np.float64)


def window_labels(t0_s, intervals_s, win_s=WIN_S):
    """Window labels: window [t0, t0+win] overlapping any GT interval -> 1."""
    lab = np.zeros(len(t0_s), np.float32)
    for j, a in enumerate(t0_s):
        b = a + win_s
        for g0, g1 in intervals_s:
            if a <= g1 and b >= g0:
                lab[j] = 1.0
                break
    return lab


def total_duration_s(t_s):
    return float(t_s[-1] - t_s[0]) if len(t_s) else 0.0
