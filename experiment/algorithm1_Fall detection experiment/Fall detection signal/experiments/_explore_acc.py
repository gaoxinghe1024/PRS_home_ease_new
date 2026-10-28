# -*- coding: utf-8 -*-
"""URFD accelerometer data exploration (one-off script, used to fix the design and parameters).

Reports: sampling characteristics / sequence sizes / fall-vs-ADL separability statistics.
"""
import os
import glob
import numpy as np
import pandas as pd

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "urfd", "csv")
CSV = os.path.abspath(CSV)


def read_acc(seq):
    p = os.path.join(CSV, f"{seq}-acc.csv")
    df = pd.read_csv(p, header=None, names=["t", "mag", "ax", "ay", "az"])
    return df


def main():
    falls = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(CSV, "fall-*-acc.csv")))
    adls = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(CSV, "adl-*-acc.csv")))
    print(f"fall sequences {len(falls)}  adl sequences {len(adls)}  total {len(falls)+len(adls)}")

    dts, durs, nrows = [], [], []
    stats = {"fall": [], "adl": []}
    for seq in falls + adls:
        df = read_acc(seq)
        t = df["t"].values.astype(float)
        n = len(df)
        dur = (t.max() - t.min()) / 1000.0
        dt = np.diff(t) / 1000.0
        dt = dt[dt > 0]
        dts.append(dt)
        durs.append(dur)
        nrows.append(n)
        mag = df["mag"].values.astype(float)
        g = mag - np.median(mag)
        # Impact peak (largest deviation from baseline)
        peak = np.max(np.abs(g))
        # Motion intensity (std)
        inten = np.std(mag)
        # Tail rest: std of the last 1 s
        tail_n = max(int(1.0 / max(np.median(dt), 1e-3)), 5)
        tail_std = np.std(mag[-tail_n:]) if n > tail_n else np.nan
        key = "fall" if seq.startswith("fall") else "adl"
        stats[key].append((seq, n, dur, peak, inten, tail_std))

    alldt = np.concatenate(dts)
    print(f"\nsampling interval dt(ms): median={np.median(alldt)*1000:.2f} mean={alldt.mean()*1000:.2f} "
          f"p5={np.percentile(alldt,5)*1000:.2f} p95={np.percentile(alldt,95)*1000:.2f}")
    print(f"=> equivalent sampling rate ~ {1.0/np.median(alldt):.1f} Hz (median)")
    print(f"sequence duration(s): median={np.median(durs):.2f} range=[{np.min(durs):.2f},{np.max(durs):.2f}]")
    print(f"samples per sequence: median={int(np.median(nrows))} range=[{min(nrows)},{max(nrows)}]")

    for k in ("fall", "adl"):
        arr = np.array([[a[1], a[2], a[3], a[4], a[5]] for a in stats[k]], float)
        m = np.nanmedian(arr, axis=0)
        print(f"\n[{k}] n={len(arr)}  median samples={m[0]:.0f} median duration={m[1]:.2f}s "
              f"median |da| peak={m[2]:.2f} median intensity std={m[3]:.3f} median tail std={m[4]:.3f}")
        print(f"      peak range [{np.nanmin(arr[:,2]):.2f},{np.nanmax(arr[:,2]):.2f}]  "
              f"intensity range [{np.nanmin(arr[:,3]):.3f},{np.nanmax(arr[:,3]):.3f}]")

    # Simple separability: sweep a peak threshold
    pk = np.array([a[3] for a in stats["fall"]] + [a[3] for a in stats["adl"]])
    y = np.array([1] * len(stats["fall"]) + [0] * len(stats["adl"]))
    best = (0, 0)
    for thr in np.arange(0.2, 3.0, 0.05):
        pred = pk > thr
        tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        rec = tp / (tp + fn) if tp + fn else 0
        prec = tp / (tp + fp) if tp + fp else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        if f1 > best[1]:
            best = (thr, f1, rec, prec)
    print(f"\nsingle-feature (peak) best threshold={best[0]:.2f} F1={best[1]:.3f} R={best[2]:.3f} P={best[3]:.3f}")


if __name__ == "__main__":
    main()
