# -*- coding: utf-8 -*-
"""Sensor degradation injection on window tensors (for robustness experiments).

The sensor path has no second dataset for cross-domain (LODO) testing, so signal
degradation stands in for a generalisation test:

  noise   white Gaussian noise: device thermal noise / loose-mount jitter, sigma in g
  drop    random packet loss: Bluetooth dropouts / sampling gaps; lost samples are
          linearly interpolated
  segment contiguous block loss: a whole stretch missing (occlusion, bad contact);
          interpolated within the block

Degradation is applied to TEST windows only; the training set stays clean, matching
the real scenario "trained on clean data, deployed on degraded signals".
"""
import numpy as np


def add_noise(wins, sigma, rng):
    w = np.asarray(wins, np.float32).copy()
    w[..., :3] += rng.normal(0.0, sigma, size=w[..., :3].shape).astype(np.float32)
    w[..., 3] = np.sqrt((w[..., :3] ** 2).sum(-1))
    return w


def drop_samples(wins, rate, rng):
    """Drop a `rate` fraction of samples at random (per-axis mask, then per-channel linear interpolation)."""
    w = np.asarray(wins, np.float32).copy()
    n, T, C = w.shape
    for i in range(n):
        k = int(round(rate * T))
        if k <= 0:
            continue
        idx = rng.choice(T, size=k, replace=False)
        idx = np.sort(idx)
        keep = np.setdiff1d(np.arange(T), idx)
        if len(keep) < 2:
            continue
        for c in range(C):
            w[i, idx, c] = np.interp(idx, keep, w[i, keep, c])
    w[..., 3] = np.sqrt((w[..., :3] ** 2).sum(-1))
    return w


def drop_segments(wins, rate, rng, max_seg=3):
    """Contiguous block loss: total loss ratio ~ rate, split into at most max_seg blocks."""
    w = np.asarray(wins, np.float32).copy()
    n, T, C = w.shape
    total = int(round(rate * T))
    if total <= 0:
        return w
    for i in range(n):
        left = total
        mask = np.zeros(T, bool)
        for _ in range(max_seg):
            if left <= 0:
                break
            L = int(rng.integers(1, max(2, left // max_seg + 1)))
            L = min(L, left)
            s = int(rng.integers(0, max(1, T - L)))
            mask[s:s + L] = True
            left -= L
        keep = np.where(~mask)[0]
        if len(keep) < 2:
            continue
        for c in range(C):
            w[i, mask, c] = np.interp(np.where(mask)[0], keep, w[i, keep, c])
    w[..., 3] = np.sqrt((w[..., :3] ** 2).sum(-1))
    return w


def corrupt(wins, kind=None, level=0.0, seed=0):
    if kind is None or level <= 0:
        return np.asarray(wins, np.float32)
    rng = np.random.default_rng(seed)
    if kind == "noise":
        return add_noise(wins, level, rng)
    if kind == "drop":
        return drop_samples(wins, level, rng)
    if kind == "segment":
        return drop_segments(wins, level, rng)
    raise ValueError(kind)


class CorruptedAlgorithm:
    """Wrap any algorithm: inject degradation before predict (fit still uses clean data)."""

    def __init__(self, alg, kind, level, seed=0):
        self.alg = alg
        self.kind = kind
        self.level = level
        self.seed = seed

    def fit(self, train):
        self.alg.fit(train)
        return self

    def predict(self, wins):
        return self.alg.predict(corrupt(wins, self.kind, self.level, self.seed))
