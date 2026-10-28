# -*- coding: utf-8 -*-
"""A2: kinematic feature model over skeleton windows."""

import numpy as np

from windowize import make_windows, window_labels, WIN_FRAMES, STEP_FRAMES

NOSE = 0
SH_L, SH_R = 5, 6
HIP_L, HIP_R = 11, 12


def _mid(kpts, i, j):
    a, b = kpts[:, i], kpts[:, j]
    c = np.minimum(a[:, 2], b[:, 2])
    return (a[:, :2] + b[:, :2]) / 2.0, c


def extract_window_features(win):
    k = win
    sh, shc = _mid(k, SH_L, SH_R)
    hp, hpc = _mid(k, HIP_L, HIP_R)
    nose = k[:, NOSE]

    v = sh - hp
    norm = np.linalg.norm(v, axis=1) + 1e-6
    cos_ang = np.clip(-v[:, 1] / norm, -1, 1)
    angle = np.arccos(cos_ang)
    sin_ang = np.sin(angle)

    hip_y, hip_conf = hp[:, 1], hpc
    nose_y, nose_conf = nose[:, 1], nose[:, 2]
    sh_y = sh[:, 1]

    def grad(y):
        return np.diff(y, prepend=y[:1])

    hip_vy = grad(hip_y)
    hip_acc = grad(hip_vy)

    f = []
    f += [np.max(sin_ang), np.mean(sin_ang), np.std(sin_ang)]
    trunk = norm
    f += [np.min(trunk), np.mean(trunk)]
    f += [np.min(hip_y), np.max(hip_y), np.mean(hip_y), hip_y[-1] - hip_y[0]]
    f += [np.max(np.abs(hip_vy)), np.max(np.abs(hip_acc))]
    f += [np.min(nose_y), np.max(np.abs(grad(nose_y)))]
    f += [np.min(sh_y), np.max(np.abs(grad(sh_y)))]
    xs = k[:, :, 0][k[:, :, 2] > 0.3]
    ys = k[:, :, 1][k[:, :, 2] > 0.3]
    if len(xs) > 2:
        hh = np.percentile(ys, 95) - np.percentile(ys, 5)
        ww = np.percentile(xs, 95) - np.percentile(xs, 5)
        f += [hh / (ww + 1e-6), ww, hh]
    else:
        f += [1.0, 0.0, 0.0]
    return np.asarray(f, np.float32)


def build_dataset(items, pos_windows=None):
    X, y = [], []
    for kpts, t0_s, iv in items:
        wins, t0s, t1s = make_windows(kpts)
        if len(wins) == 0:
            continue
        labels = np.zeros(len(wins), np.float32)
        win_len_s = WIN_FRAMES / 15.0  # 2s @15fps
        for j, w_start in enumerate(t0_s):
            w_end = w_start + win_len_s
            for a, b in iv:
                if w_start <= b and w_end >= a:
                    labels[j] = 1.0
                    break
        feats = np.stack([extract_window_features(w) for w in wins])
        X.append(feats)
        y.append(labels)
    X = np.concatenate(X, 0).astype(np.float32)
    y = np.concatenate(y, 0).astype(np.float32)
    return X, y


class MLAlgorithm:

    name = "A2_ml"

    def __init__(self):
        self.rf = None
        self.xgb = None

    def fit(self, train_items):
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb

        X, y = build_dataset(train_items)
        if len(X) == 0 or y.sum() == 0 or (y == 1).all():
            self._const = float(y.mean()) if len(y) else 0.0
            return self
        self._const = None
        self.rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=2026,
        )
        self.xgb = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=2026,
            n_jobs=-1,
        )
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        return self

    def predict(self, kpts):
        wins, t0s, t1s = make_windows(kpts)
        if len(wins) == 0:
            return np.zeros(0, np.float32)
        if self._const is not None:
            return np.full(len(wins), self._const, np.float32)
        feats = np.stack([extract_window_features(w) for w in wins])
        p1 = self.rf.predict_proba(feats)[:, 1]
        p2 = self.xgb.predict_proba(feats)[:, 1]
        return 0.5 * p1 + 0.5 * p2
