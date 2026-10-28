# -*- coding: utf-8 -*-
"""Le2i / ImViA reader for the four labelled scenes."""

import os
import numpy as np

import paths

FPS = 25.0
SCENES = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]
UNLABELED_SCENES = ["Office", "Lecture_room"]


def _root():
    return paths.le2i_root()


def _video_dir(scene):
    root = _root()
    cands = [
        os.path.join(root, scene, scene, "Videos"),
        os.path.join(root, scene, "Lecture room"),
        os.path.join(root, scene, scene),
    ]
    for c in cands:
        if os.path.isdir(c) and any(f.lower().endswith(".avi") for f in os.listdir(c)):
            return c
    return None


def _anno_dir(scene):
    root = _root()
    cands = [
        os.path.join(root, scene, scene, "Annotation_files"),
        os.path.join(root, scene, scene, "Annotations_files"),
        os.path.join(root, scene, "Annotation_files"),
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    return None


def list_videos():
    out = []
    for scene in SCENES:
        vdir = _video_dir(scene)
        if not vdir:
            continue
        for f in sorted(os.listdir(vdir)):
            if f.lower().endswith(".avi"):
                i = f.split("(")[1].split(")")[0]
                out.append((f"{scene}_video{i}", os.path.join(vdir, f)))
    return out


def anno_path(seq_name):
    scene = "_".join(seq_name.split("_")[:-1])
    i = seq_name.rsplit("video", 1)[1]
    return os.path.join(_anno_dir(scene), f"video ({i}).txt")


def _parse_anno(seq_name):
    p = anno_path(seq_name)
    with open(p, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    f0, f1 = 0, 0
    if len(lines) >= 2 and "," not in lines[0] and "," not in lines[1]:
        f0, f1 = int(float(lines[0])), int(float(lines[1]))
        lines = lines[2:]
    boxes = {}
    for ln in lines:
        parts = [int(float(x)) for x in ln.split(",")]
        if len(parts) >= 6:
            fr, _pid, h, w, cx, cy = parts[:6]
            if h > 0 and w > 0:
                boxes[fr] = (h, w, cx, cy)
    return f0, f1, boxes


def fall_intervals_s(seq_name):
    f0, f1, _ = _parse_anno(seq_name)
    if 0 < f0 <= f1:
        return [(f0 / FPS, f1 / FPS)]
    return []


def gt_boxes(seq_name):
    _, _, boxes = _parse_anno(seq_name)
    out = {}
    for fr, (h, w, cx, cy) in boxes.items():
        x1 = max(0.0, (cx - w / 2) / 320.0)
        y1 = max(0.0, (cy - h / 2) / 240.0)
        x2 = min(1.0, (cx + w / 2) / 320.0)
        y2 = min(1.0, (cy + h / 2) / 240.0)
        if x2 > x1 and y2 > y1:
            out[fr] = [x1, y1, x2, y2]
    return out


def scene_of(seq_name):
    return seq_name.rsplit("_video", 1)[0]
