# -*- coding: utf-8 -*-
"""A1: geometric rule baseline over skeleton windows."""

import numpy as np

from windowize import make_windows, WIN_FRAMES, STEP_FRAMES

SH_L, SH_R = 5, 6
HIP_L, HIP_R = 11, 12
NOSE = 0


def frame_fall_score(kpts):
    T = len(kpts)
    score = np.zeros(T, np.float32)

    def mid(i, j):
        a, b = kpts[:, i], kpts[:, j]
        conf = np.minimum(a[:, 2], b[:, 2])
        m = (a[:, :2] + b[:, :2]) / 2.0
        return m, conf

    sh, shc = mid(SH_L, SH_R)
    hp, hpc = mid(HIP_L, HIP_R)
    conf = np.minimum(shc, hpc)

    v = sh - hp  # [T,2]
    norm = np.linalg.norm(v, axis=1) + 1e-6
    cos_ang = -v[:, 1] / norm  # v·(0,-1)/|v| = -vy/|v|
    cos_ang = np.clip(cos_ang, -1, 1)
    angle = np.arccos(cos_ang)  # [0,π]
    sin_ang = np.sin(angle)

    hip_y = hp[:, 1]
    descent = np.zeros(T, np.float32)
    descent[1:] = hip_y[1:] - hip_y[:-1]

    visible = conf > 0.3
    speed = np.clip(descent / 0.15, 0.0, 1.0)
    s = 0.80 * sin_ang + 0.20 * speed
    score[visible] = s[visible]
    return score


class RuleAlgorithm:

    name = "A1_rule"

    def fit(self, train_data):
        return self

    def predict(self, kpts):
        fs = frame_fall_score(kpts)
        wins, t0s, t1s = make_windows(kpts)
        if len(wins) == 0:
            return np.zeros(0, np.float32)
        out = np.zeros(len(wins), np.float32)
        for j, (i0, i1) in enumerate(zip(t0s, t1s)):
            seg = fs[i0:i1]
            if len(seg):
                out[j] = np.percentile(seg, 90)
        return out
