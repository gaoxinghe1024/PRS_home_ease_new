# -*- coding: utf-8 -*-
"""URFD tri-axial accelerometer loading and fall-interval alignment.

CSV columns: `t_ms, |a|, ax, ay, az` (acceleration in g, time in ms, origin synced to video)
  - `|a|` is the vector magnitude; it is recomputed here for internal consistency.
  - Sampling is irregular (nominal ~16 ms, jitter 5-37 ms); median effective rate 62.5 Hz.

Frame alignment (important):
  `{seq}-data.csv` maps `frame@30fps <-> t_ms` (frame=1 <-> t_ms~0), i.e. the
  accelerometer timeline and the video frame timeline share the same origin:
      frame f  ->  seconds t = f / 30

Fall interval definition (identical to the video path):
      first frame with label=0 (fall onset) -> first frame with label=1 + 30 frames (+1 s after impact)
  Intervals are clipped to the accelerometer recording range (some sequences stop early, e.g. fall-10).

Sequence naming: fall-01..30 (fall) / adl-01..40 (activities of daily living, no GT interval).
"""
import os
import glob
import numpy as np
import pandas as pd

import paths

SRC_FPS = 30.0            # video/annotation frame rate, used for frame <-> second conversion
TAIL_FRAMES = 30          # +1 s after impact (same convention as the video path)
_G = 1.0                  # gravity baseline (g)


def list_sequences():
    """Return two ordered lists (falls, adls)."""
    d = paths.acc_csv_dir()
    falls = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(d, "fall-*-acc.csv")))
    adls = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(d, "adl-*-acc.csv")))
    return falls, adls


def read_acc(seq):
    """Read raw tri-axial data.

    Returns a dict:
      t_s  : [N] seconds relative to sequence start
      ax/ay/az : [N] tri-axial acceleration (g)
      mag  : [N] magnitude (g), recomputed instead of using the CSV column
      fs_nom : nominal sampling rate (1 / median interval)
    """
    p = paths.acc_csv(seq)
    df = pd.read_csv(p, header=None, names=["t_ms", "mag_csv", "ax", "ay", "az"])
    df = df.sort_values("t_ms").reset_index(drop=True)
    t_ms = df["t_ms"].to_numpy(np.float64)
    ax = df["ax"].to_numpy(np.float64)
    ay = df["ay"].to_numpy(np.float64)
    az = df["az"].to_numpy(np.float64)
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    t_s = (t_ms - t_ms[0]) / 1000.0
    dt = np.diff(t_s)
    dt = dt[dt > 0]
    fs_nom = 1.0 / np.median(dt) if len(dt) else 0.0
    return {"seq": seq, "t_s": t_s, "ax": ax, "ay": ay, "az": az, "mag": mag,
            "fs_nom": float(fs_nom), "n": len(df)}


def _load_gt():
    """Frame-level fall annotations as {seq: DataFrame(frame,label)}."""
    df = pd.read_csv(paths.falls_gt_csv(), header=None,
                     names=["seq", "frame", "label", "hwr", "mmr", "bbo",
                            "stdxz", "hhmax", "H", "D", "p40"])
    out = {}
    for seq, g in df.groupby("seq"):
        out[seq] = g[["frame", "label"]].sort_values("frame").reset_index(drop=True)
    return out


_GT_CACHE = None


def fall_frames(seq):
    """Fall intervals in frame numbers (endpoints included) as [(f0, f1)]; ADL returns []."""
    global _GT_CACHE
    if _GT_CACHE is None:
        _GT_CACHE = _load_gt()
    if seq not in _GT_CACHE:
        return []
    if not seq.startswith("fall"):
        return []
    g = _GT_CACHE[seq]
    l0 = g[g.label == 0]["frame"]
    l1 = g[g.label == 1]["frame"]
    f0 = int(l0.iloc[0]) if len(l0) else int(g["frame"].iloc[0])
    f1 = int(l1.iloc[0]) + TAIL_FRAMES if len(l1) else int(g["frame"].max()) + TAIL_FRAMES
    return [(f0, f1)]


def fall_intervals_s(seq, clip_to=None):
    """Fall intervals in seconds. clip_to=(0,dur) clips to the recording range.

    ADL sequences return [] (all negative samples).
    """
    ivs = [(a / SRC_FPS, b / SRC_FPS) for a, b in fall_frames(seq)]
    if clip_to is not None:
        lo, hi = clip_to
        out = []
        for a, b in ivs:
            a2, b2 = max(a, lo), min(b, hi)
            if b2 > a2:
                out.append((a2, b2))
        ivs = out
    return ivs


def duration_s(seq):
    d = read_acc(seq)
    return float(d["t_s"][-1])


def has_fall(seq):
    return len(fall_intervals_s(seq)) > 0


if __name__ == "__main__":
    falls, adls = list_sequences()
    print(f"falls {len(falls)} / adl {len(adls)}")
    for s in falls[:3] + adls[:3]:
        d = read_acc(s)
        print(f"{s}: n={d['n']:4d} dur={duration_s(s):5.2f}s fs~{d['fs_nom']:5.1f}Hz "
              f"GT={[(round(a,2), round(b,2)) for a, b in fall_intervals_s(s, (0, duration_s(s)))]}")
