# -*- coding: utf-8 -*-
"""Preprocessing of the best combination: resample to 60 Hz + 4th-order Butterworth low-pass at 5 Hz (zero phase, gravity kept).

Only the **best path** is kept, without the 6 candidate filters of the experiments; deployment needs no ablation switches.
The implementation matches the `resample=60.0, filt="butter5"` branch of `00_common/preprocess.py` line by line,
so the experimental conclusions and the deployed system share exactly the same numerical behaviour.
"""
import numpy as np
from scipy import signal as sps

from . import config as C


# ---------------------------------------------------------------------------
# Filter kernel
# ---------------------------------------------------------------------------
def butter5_sos(fs):
    """SOS coefficients of the 4th-order Butterworth 5 Hz low-pass (cut-off protected against the fs limit)."""
    fc = min(C.BUTTER_CUTOFF, 0.45 * fs)
    return sps.butter(C.BUTTER_ORDER, fc, btype="low", fs=fs, output="sos")


def _sos_filtfilt(sos, x):
    """Zero-phase filtering (filtfilt); padlen shrinks automatically for short signals.

    Note: filtfilt is non-causal; offline / whole-recording analysis has no phase lag.
    Real-time streaming runs the same operation on a **finite sliding buffer**, an engineering approximation
    that adds a look-ahead of about half a window (capped by the buffer length in the system).
    """
    n = len(x)
    if n < 12:
        return x.copy()
    padlen = min(3 * (2 * sos.shape[0] + 1), n - 1)
    return sps.sosfiltfilt(sos, x, padlen=padlen)


def apply_butter5(sig, fs):
    """butter5 zero-phase low-pass on an [N] or [N,C] signal."""
    x2 = sig if sig.ndim == 2 else sig[:, None]
    sos = butter5_sos(fs)
    out = np.empty_like(x2, dtype=np.float64)
    for c in range(x2.shape[1]):
        out[:, c] = _sos_filtfilt(sos, x2[:, c])
    return out if sig.ndim == 2 else out[:, 0]


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def uniform_grid(t_s, fs, anchor=None):
    """Fixed-rate time grid.

    With anchor=None the grid starts at t_s[0] (offline, whole recording);
    with an anchor the grid points align to absolute times k/fs (streaming, so repeated evaluations share one grid).
    """
    if anchor is None:
        return np.arange(t_s[0], t_s[-1] + 1e-12, 1.0 / fs)
    k0 = int(np.floor(anchor * fs))
    return np.arange(k0 / fs, t_s[-1] + 1e-9, 1.0 / fs)


def resample_signal(t_s, sig, fs, anchor=None):
    """Linear interpolation over the timestamps to a fixed rate fs. sig: [N] or [N,C]."""
    grid = uniform_grid(t_s, fs, anchor)
    if len(grid) == 0:
        grid = np.array([t_s[-1]], dtype=np.float64)
    if sig.ndim == 1:
        return grid, np.interp(grid, t_s, sig)
    cols = [np.interp(grid, t_s, sig[:, c]) for c in range(sig.shape[1])]
    return grid, np.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# Normalized entry point
# ---------------------------------------------------------------------------
def normalize(t_raw, a_raw, fs=C.RESAMPLE_FS, anchor=None):
    """Raw (t, [N,3]) -> normalized signal.

    Returns a dict:
      t [M] / ax,ay,az [M] / mag [M] / fs
    """
    t_raw = np.asarray(t_raw, np.float64)
    a_raw = np.asarray(a_raw, np.float64)
    if a_raw.ndim == 1:
        a_raw = a_raw[:, None]
    order = np.argsort(t_raw)
    t_raw, a_raw = t_raw[order], a_raw[order]

    t, sig = resample_signal(t_raw, a_raw, fs, anchor=anchor)
    sig = apply_butter5(sig, fs)
    ax, ay, az = sig[:, 0], sig[:, 1], sig[:, 2]
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    return {"t": t, "ax": ax, "ay": ay, "az": az, "mag": mag, "fs": float(fs)}


def read_acc_csv(path, has_header=False):
    """Read a three-axis acceleration CSV.

    Two column orders are supported (detected automatically):
      URFD native:    t_ms, |a|, ax, ay, az   - 5 columns, first column in milliseconds
      generic upload: t, ax, ay, az or t_ms, ax, ay, az - 4 columns, first column is time
    Returns (t_s, a_raw[N,3]) with time converted to seconds and starting at 0.
    """
    import csv
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.reader(f):
            if not r:
                continue
            try:
                vals = [float(x) for x in r[:6] if str(x).strip() != ""]
            except ValueError:
                continue  # skip headers / malformed rows
            rows.append(vals)
    if not rows:
        raise ValueError("The CSV is empty or cannot be parsed")
    arr = np.asarray([r[:5] + [np.nan] * (5 - len(r)) for r in rows], np.float64)
    if arr.shape[1] >= 5:                       # URFD: t_ms, |a|, ax, ay, az
        t_ms, ax, ay, az = arr[:, 0], arr[:, 2], arr[:, 3], arr[:, 4]
        t_s = (t_ms - t_ms[0]) / 1000.0
    elif arr.shape[1] == 4:                     # t, ax, ay, az
        t0 = arr[:, 0]
        t_s = t0 - t0[0]
        if np.nanmedian(np.diff(t_s)) > 1.0:    # looks like milliseconds
            t_s = t_s / 1000.0
        ax, ay, az = arr[:, 1], arr[:, 2], arr[:, 3]
    else:
        raise ValueError(f"Unrecognized number of columns: {arr.shape[1]}")
    a = np.stack([ax, ay, az], axis=1)
    m = np.isfinite(a).all(axis=1) & np.isfinite(t_s)
    return t_s[m], a[m]
