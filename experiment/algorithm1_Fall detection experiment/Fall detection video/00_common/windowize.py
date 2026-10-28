# -*- coding: utf-8 -*-
"""Skeleton sequence to sliding-window conversion (2 s window, 0.5 s step)."""

import numpy as np

FPS = 15.0
WIN_FRAMES = 30  # 2s
STEP_FRAMES = 8
MERGE_GAP_WIN = 2


def make_windows(kpts, win=WIN_FRAMES, step=STEP_FRAMES):
    T = len(kpts)
    if T < win:
        return np.zeros((0, win, 17, 3), np.float32), [], []
    t0s = list(range(0, T - win + 1, step))
    wins = np.stack([kpts[i : i + win] for i in t0s]).astype(np.float32)
    t1s = [i + win for i in t0s]
    return wins, t0s, t1s


def window_labels(t0_idx, t1_idx, fall_intervals, fps=FPS):
    labels = np.zeros(len(t0_idx), np.float32)
    for a, b in fall_intervals:
        for j, (i0, i1) in enumerate(zip(t0_idx, t1_idx)):
            if i0 <= b and i1 - 1 >= a:
                labels[j] = 1.0
    return labels


def frame_time_s(frame_idx, fps=FPS):
    return np.asarray(frame_idx, np.float64) / fps
