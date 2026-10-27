# -*- coding: utf-8 -*-
"""Train and save the **best combination** model (RF400 + XGB400 probability average).

Data: the parent project's `../data/urfd/csv` (read-only, not copied).
Labels (identical to `00_common/acc_loader.py`):
    fall interval = [first frame with label=0, first frame with label=1 + 30 frames], converted to seconds at 30 fps
    and clipped to the acceleration recording; ADL sequences have no fall interval (all negative).
    A window is positive <=> its [t0, t0+2s] overlaps any fall interval.

Note: **all 70 sequences** are used for training here (deployment uses all data, which is standard practice).
The generalization figures used by the system (sequence-level AUROC 0.9853 +/- 0.0010 etc.) come from the 3-seed x stratified
5-fold cross-validation of `experiments/exp03_alg.py` in the algorithm project, not from a hold-out of this script.

Run:
    python scripts/train_best_model.py
"""
import os
import sys
import csv
import json
import time
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from core import config as C          # noqa: E402
from core import preprocess as P      # noqa: E402
from core import windowize as W       # noqa: E402
from core.model import BestFallModel  # noqa: E402

DATA_DIR = os.environ.get("FALL_SYSTEM_DATA_DIR") or \
    os.path.abspath(os.path.join(ROOT, "..", "data", "urfd", "csv"))
GT_CSV = os.path.join(DATA_DIR, "urfall-cam0-falls.csv")
SRC_FPS = 30.0
TAIL_FRAMES = 30
MODEL_PATH = os.path.join(ROOT, "model", "best_model.pkl")
META_PATH = os.path.join(ROOT, "model", "meta.json")


def _load_gt():
    """{seq: (f0, f1)} fall frame intervals; ADL sequences are not listed."""
    out = {}
    if not os.path.isfile(GT_CSV):
        return out
    rows = {}
    with open(GT_CSV, "r", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) < 3:
                continue
            try:
                rows.setdefault(r[0], []).append((int(float(r[1])), int(float(r[2]))))
            except ValueError:
                continue
    for seq, fr in rows.items():
        fr.sort()
        l0 = [f for f, lab in fr if lab == 0]
        l1 = [f for f, lab in fr if lab == 1]
        f0 = l0[0] if l0 else fr[0][0]
        f1 = (l1[0] + TAIL_FRAMES) if l1 else (fr[-1][0] + TAIL_FRAMES)
        out[seq] = (int(f0), int(f1))
    return out


def fall_intervals_s(seq, gt, dur):
    if not str(seq).startswith("fall") or seq not in gt:
        return []
    f0, f1 = gt[seq]
    a, b = f0 / SRC_FPS, f1 / SRC_FPS
    a, b = max(a, 0.0), min(b, dur)
    return [(a, b)] if b > a else []


def main():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"Data folder not found: {DATA_DIR}\nSet the environment variable FALL_SYSTEM_DATA_DIR")

    gt = _load_gt()
    falls = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(DATA_DIR, "fall-*-acc.csv")))
    adls = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(DATA_DIR, "adl-*-acc.csv")))
    seqs = falls + adls
    print(f"[data] {len(falls)} falls / {len(adls)} daily activities, {len(seqs)} sequences in total")

    all_wins, all_lab = [], []
    t0 = time.time()
    for i, seq in enumerate(seqs, 1):
        t_raw, a_raw = P.read_acc_csv(os.path.join(DATA_DIR, f"{seq}-acc.csv"))
        dur = float(t_raw[-1] - t_raw[0])
        ivs = fall_intervals_s(seq, gt, dur)
        norm = P.normalize(t_raw, a_raw, fs=C.RESAMPLE_FS)
        sig = np.stack([norm["ax"], norm["ay"], norm["az"], norm["mag"]], axis=1)
        wins, t0_idx = W.make_windows(sig, C.RESAMPLE_FS)
        if len(wins) == 0:
            continue
        starts = np.asarray([norm["t"][j] for j in t0_idx], np.float64)
        lab = np.zeros(len(starts), np.float32)
        for j, a in enumerate(starts):
            b = a + C.WIN_S
            for g0, g1 in ivs:
                if a <= g1 and b >= g0:
                    lab[j] = 1.0
                    break
        all_wins.append(wins)
        all_lab.append(lab)
        if i % 10 == 0 or i == len(seqs):
            print(f"  ... {i}/{len(seqs)}  {seq}  windows={len(wins)}  positive={int(lab.sum())}")

    wins = np.concatenate(all_wins)
    lab = np.concatenate(all_lab)
    print(f"[features/windows] {len(wins)} windows in total, {int(lab.sum())} positive ({lab.mean()*100:.1f}%), "
          f"took {time.time()-t0:.1f}s")

    print("[training] RF(400) + XGB(400) ...")
    t1 = time.time()
    model = BestFallModel(fs=C.RESAMPLE_FS, seed=C.TRAIN_SEED).fit(wins, lab)
    print(f"[training] done in {time.time()-t1:.1f}s")

    model.save(MODEL_PATH)
    meta = dict(model.meta)
    meta.update({
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": DATA_DIR,
        "n_sequences": len(seqs),
        "n_fall_seq": len(falls), "n_adl_seq": len(adls),
        "reference_metrics": {
            "note": "from the 3-seed x stratified 5-fold CV of algorithms/experiments/exp03_alg.py",
            "auroc_seq_mean": 0.9852777777777777,
            "auroc_seq_std": 0.0010393492741038967,
            "event_accuracy": 0.8261904761904763,
            "event_recall": 0.9666666666666668,
        },
    })
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[saved] {MODEL_PATH}\n[saved] {META_PATH}")


if __name__ == "__main__":
    main()
