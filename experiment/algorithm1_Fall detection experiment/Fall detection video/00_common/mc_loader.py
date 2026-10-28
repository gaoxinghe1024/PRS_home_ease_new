# -*- coding: utf-8 -*-
"""Multiple Cameras Fall reader, single-channel protocol (cam1)."""

import os
import numpy as np
import pandas as pd

import paths

CAM = 1
FPS = 30.0


def _root():
    base = paths.mc_root()
    inner = os.path.join(base, "dataset", "dataset")
    if os.path.isdir(inner):
        return inner, os.path.join(base, "data_tuple3.csv")
    return base, os.path.join(base, "data_tuple3.csv")


def list_sequences():
    chutes, csv_path = _root()
    df = pd.read_csv(csv_path)
    chutes_in_csv = set(df.chute.dropna().astype(int).unique())
    out = []
    for d in sorted(os.listdir(chutes)):
        if d.startswith("chute") and os.path.isdir(os.path.join(chutes, d)):
            n = int(d.replace("chute", ""))
            if n in chutes_in_csv and os.path.exists(video_path(d)):
                out.append(d)
    return out


def video_path(seq):
    chutes, _ = _root()
    return os.path.join(chutes, seq, f"cam{CAM}.avi")


def fall_intervals_s(seq):
    chutes, csv_path = _root()
    df = pd.read_csv(csv_path)
    cid = int(seq.replace("chute", ""))
    g = df[(df.chute == cid) & (df.cam == CAM)]
    return [
        (float(r.start) / FPS, float(r.end) / FPS)
        for r in g.itertuples()
        if r.label == 1
    ]


def adl_spans_s(seq):
    chutes, csv_path = _root()
    df = pd.read_csv(csv_path)
    cid = int(seq.replace("chute", ""))
    g = df[(df.chute == cid) & (df.cam == CAM)]
    return [
        (float(r.start) / FPS, float(r.end) / FPS)
        for r in g.itertuples()
        if r.label == 0
    ]


def has_fall(seq):
    return len(fall_intervals_s(seq)) > 0
