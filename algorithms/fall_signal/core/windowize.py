# -*- coding: utf-8 -*-
"""Sliding windows and event aggregation (best combination: 2.0 s windows, 0.25 s step, merge gap of 2 windows).

Physical reason for the 0.25 s step: a fall impact lasts 0.3-0.6 s, and a 0.25 s step guarantees that any
0.5 s event is fully covered by at least 2 windows, so the impact peak is never missed.
"""
import numpy as np

from . import config as C


def n_samples(fs, sec):
    return max(1, int(round(fs * sec)))


def make_windows(sig, fs, win_s=C.WIN_S, step_s=C.STEP_S):
    """[T,C] or [T] -> windows [N,win_n,C], t0_idx [N] (window start **sample index**)."""
    sig = np.asarray(sig)
    if sig.ndim == 1:
        sig = sig[:, None]
    win_n, step_n = n_samples(fs, win_s), n_samples(fs, step_s)
    T = len(sig)
    if T < win_n:
        return np.zeros((0, win_n, sig.shape[1]), np.float32), np.array([], np.int64)
    t0s = list(range(0, T - win_n + 1, step_n))
    wins = np.stack([sig[i:i + win_n] for i in t0s]).astype(np.float32)
    return wins, np.asarray(t0s, np.int64)


def windows_from_grid(t, sig, fs, win_s=C.WIN_S, step_s=C.STEP_S):
    """Cut windows by absolute time from the absolute time grid t and the signal sig.

    Returns (wins [N,win_n,C], t_start[N], t_end[N]), all times in absolute seconds.
    Used for streaming: window end points fall exactly on integer multiples of step_s.
    """
    sig = np.asarray(sig)
    if sig.ndim == 1:
        sig = sig[:, None]
    win_n = n_samples(fs, win_s)
    step_n = n_samples(fs, step_s)
    if len(sig) < win_n:
        return np.zeros((0, win_n, sig.shape[1]), np.float32), np.array([]), np.array([])
    starts = list(range(0, len(sig) - win_n + 1, step_n))
    wins = np.stack([sig[i:i + win_n] for i in starts]).astype(np.float32)
    t_start = np.asarray([t[i] for i in starts], np.float64)
    return wins, t_start, t_start + win_s


def merge_hits(hit_t_end, win_s=C.WIN_S, gap_win=C.MERGE_GAP_WIN):
    """Aggregate hit windows (sorted by end time) into events.

    Rules (same meaning as MERGE_GAP_WIN in 00_common/windowize.py):
      hit windows whose **index gap** is <= gap_win belong to the same event;
      event interval = [start of the first hit window, end of the last hit window];
      alarm time = **end of the earliest hit window** (= the earliest possible alarm).
    Returns list[dict] with the fields t_win_start / t_win_end / t_alarm / n_windows.
    """
    hit = np.sort(np.asarray(hit_t_end, np.float64))
    if len(hit) == 0:
        return []
    tol = 1e-6
    events, cur = [], [hit[0]]
    for x in hit[1:]:
        if (x - cur[-1]) <= gap_win * C.STEP_S + tol:
            cur.append(x)
        else:
            events.append(cur)
            cur = [x]
    events.append(cur)
    out = []
    for ev in events:
        t_first_end = float(ev[0])
        t_last_end = float(ev[-1])
        out.append({
            "t_win_start": t_first_end - win_s,
            "t_win_end": t_last_end,
            "t_alarm": t_first_end,          # alarm time = end of the earliest hit window
            "n_windows": int(len(ev)),
        })
    return out
