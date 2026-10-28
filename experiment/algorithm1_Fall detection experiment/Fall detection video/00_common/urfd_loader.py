# -*- coding: utf-8 -*-
"""URFD frame reader and frame-level fall labels."""

import os
import glob
import numpy as np
import pandas as pd

import paths


def list_sequences():
    root = paths.urfd_root()
    falls = [
        f"fall-{i:02d}"
        for i in range(1, 31)
        if os.path.isdir(os.path.join(root, "frames", f"fall-{i:02d}-cam0-rgb"))
    ]
    adls = [
        f"adl-{i:02d}"
        for i in range(1, 41)
        if os.path.isdir(os.path.join(root, "frames", f"adl-{i:02d}-cam0-rgb"))
    ]
    return falls, adls


def frames_dir(seq):
    return os.path.join(paths.urfd_root(), "frames", f"{seq}-cam0-rgb")


def frame_files(seq):
    fs = sorted(glob.glob(os.path.join(frames_dir(seq), "*.png")))
    return fs


def load_fall_gt():
    path = os.path.join(paths.urfd_root(), "csv", "urfall-cam0-falls.csv")
    df = pd.read_csv(
        path,
        header=None,
        names=[
            "seq",
            "frame",
            "label",
            "hwr",
            "mmr",
            "bbo",
            "stdxz",
            "hhmax",
            "H",
            "D",
            "p40",
        ],
    )
    out = {}
    for seq, g in df.groupby("seq"):
        out[seq] = g[["frame", "label"]].sort_values("frame").reset_index(drop=True)
    return out


def fall_intervals_s(seq, fps_src=30.0):
    if not seq.startswith("fall"):
        return []
    gt = load_fall_gt()
    g = gt[seq]
    l0 = g[g.label == 0]["frame"]
    l1 = g[g.label == 1]["frame"]
    f0 = int(l0.iloc[0]) if len(l0) else int(g["frame"].iloc[0])
    f1 = int(l1.iloc[0]) + 30 if len(l1) else int(g["frame"].max())
    return [(f0 / fps_src, f1 / fps_src)]


def video_source(seq):
    return frames_dir(seq)
