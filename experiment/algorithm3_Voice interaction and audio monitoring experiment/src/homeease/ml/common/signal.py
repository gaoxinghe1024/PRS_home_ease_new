"""Causal signal processing shared by live inference and offline training.

Offline helpers are thin wrappers that push a whole recording through the
same streaming classes, so live and offline outputs are identical by
construction (verified in tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

G = 9.80665


def to_g(values: np.ndarray, unit: str) -> np.ndarray:
    if unit == "g":
        return values.astype(np.float64)
    if unit == "m_s2":
        return values.astype(np.float64) / G
    raise ValueError(f"unknown unit {unit}")


@dataclass
class UniformChunk:
    t: np.ndarray  # (n,) ms on the uniform grid
    x: np.ndarray  # (n, c) values; NaN where masked
    mask: np.ndarray  # (n,) 1 valid, 0 gap


class StreamingResampler:
    """Linear-interpolation resampler onto a uniform grid, strictly causal.

    A grid point is emitted only once a raw sample at or after it exists.
    Gaps longer than ``max_interp_ms`` are not interpolated: grid points inside
    them are emitted with mask 0 and NaN values, so long gaps stay visible.
    """

    def __init__(self, fs_hz: float, n_channels: int, max_interp_ms: float = 120.0) -> None:
        self.fs = float(fs_hz)
        self.dt = 1000.0 / self.fs
        self.c = n_channels
        self.max_interp_ms = float(max_interp_ms)
        self._last_t: float | None = None
        self._last_x: np.ndarray | None = None
        self._next_grid: float | None = None

    def reset(self) -> None:
        self._last_t = None
        self._last_x = None
        self._next_grid = None

    @property
    def next_grid_time(self) -> float | None:
        return self._next_grid

    def push(self, t: np.ndarray, x: np.ndarray) -> UniformChunk:
        t = np.asarray(t, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64).reshape(len(t), self.c)
        out_t: list[float] = []
        out_x: list[np.ndarray] = []
        out_m: list[int] = []
        for i in range(len(t)):
            ti, xi = float(t[i]), x[i]
            if self._last_t is None:
                # Start the grid at the first sample (aligned to the grid step).
                self._next_grid = float(np.ceil(ti / self.dt) * self.dt)
                if self._next_grid - ti < 1e-9:
                    out_t.append(self._next_grid)
                    out_x.append(xi.copy())
                    out_m.append(1)
                    self._next_grid += self.dt
                self._last_t, self._last_x = ti, xi.copy()
                continue
            if ti <= self._last_t:
                # Duplicates / out-of-order samples are dropped (handled upstream).
                continue
            gap = ti - self._last_t
            assert self._next_grid is not None and self._last_x is not None
            while self._next_grid <= ti + 1e-9:
                g = self._next_grid
                if gap <= self.max_interp_ms:
                    w = (g - self._last_t) / gap
                    out_x.append(self._last_x + w * (xi - self._last_x))
                    out_m.append(1)
                else:
                    if abs(g - ti) < 1e-6:
                        out_x.append(xi.copy())
                        out_m.append(1)
                    else:
                        out_x.append(np.full(self.c, np.nan))
                        out_m.append(0)
                out_t.append(g)
                self._next_grid += self.dt
            self._last_t, self._last_x = ti, xi.copy()
        if out_t:
            return UniformChunk(np.asarray(out_t), np.vstack(out_x), np.asarray(out_m, dtype=np.int8))
        return UniformChunk(np.zeros(0), np.zeros((0, self.c)), np.zeros(0, dtype=np.int8))


class StreamingLowpass:
    """Causal Butterworth low-pass (second-order sections) with persistent state.

    The filter state is re-initialised at the first valid sample after a gap so
    that gaps do not produce long transients.
    """

    def __init__(self, fs_hz: float, cutoff_hz: float, n_channels: int, order: int = 4) -> None:
        nyq = fs_hz / 2.0
        wn = min(cutoff_hz / nyq, 0.99)
        self.sos = sps.butter(order, wn, btype="low", output="sos")
        self.zi0 = sps.sosfilt_zi(self.sos)  # (n_sections, 2)
        self.c = n_channels
        self.zi: list[np.ndarray] | None = None

    def reset(self) -> None:
        self.zi = None

    def push(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Filter a chunk. Contiguous valid runs are filtered in one call, which is
        numerically identical to sample-by-sample processing with carried state."""
        x = np.asarray(x, dtype=np.float64)
        mask = np.asarray(mask).astype(bool)
        y = np.full_like(x, np.nan)
        i, n = 0, len(x)
        while i < n:
            if not mask[i]:
                self.zi = None
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            if self.zi is None:
                self.zi = [self.zi0 * x[i, ch] for ch in range(self.c)]
            for ch in range(self.c):
                y[i:j, ch], self.zi[ch] = sps.sosfilt(self.sos, x[i:j, ch], zi=self.zi[ch])
            i = j
        return y


def resample_uniform(t: np.ndarray, x: np.ndarray, fs_hz: float, max_interp_ms: float = 120.0) -> UniformChunk:
    """Offline resampling through the streaming implementation."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    r = StreamingResampler(fs_hz, x.shape[1], max_interp_ms)
    return r.push(t, x)


def causal_lowpass(x: np.ndarray, mask: np.ndarray, fs_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    f = StreamingLowpass(fs_hz, cutoff_hz, x.shape[1], order)
    return f.push(x, mask)


def magnitude(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nansum(np.asarray(x) ** 2, axis=-1)) * np.where(np.isnan(np.asarray(x)).any(axis=-1), np.nan, 1.0)


class RingBuffer:
    """Fixed-capacity ring buffer of rows (time-ordered appends)."""

    def __init__(self, capacity: int, width: int) -> None:
        self.cap = int(capacity)
        self.w = int(width)
        self.buf = np.full((self.cap, self.w), np.nan)
        self.n = 0
        self.head = 0  # next write position

    def append(self, rows: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=np.float64).reshape(-1, self.w)
        for r in rows[-self.cap :]:
            self.buf[self.head] = r
            self.head = (self.head + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def view(self) -> np.ndarray:
        if self.n < self.cap:
            return self.buf[: self.n].copy()
        return np.vstack([self.buf[self.head :], self.buf[: self.head]])

    def last(self) -> np.ndarray | None:
        if self.n == 0:
            return None
        return self.buf[(self.head - 1) % self.cap].copy()

    def clear(self) -> None:
        self.n = 0
        self.head = 0
        self.buf[:] = np.nan
