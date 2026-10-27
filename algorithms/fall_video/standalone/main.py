# -*- coding: utf-8 -*-
"""
main.py - complete fall detection pipeline (self-contained, end to end + visualization)
============================================================

This is a FULLY SELF-CONTAINED test entry: it imports no project module,
and the weights of all three stages sit next to this file (the standalone/ folder) and are loaded directly.

  Stage 1 person detection : yolov8n-pose.onnx (pre-trained YOLOv8n, box head only)
  Stage 2 pose estimation  : pose_best.pt (our own small P1 U-Net, trained from scratch)
  Stage 3 temporal decision: a4_weights.pkl (A4 hybrid ensemble, OOF stacking + XGBoost)

Pipeline (live inference throughout, no cache is read):
  video frame -> YOLO person bbox -> P1 pose skeleton -> A4 temporal decision -> fall event

Dependencies (third-party packages only): numpy, cv2, torch, sklearn, xgboost, pandas

Usage:
  python main.py --demo urfd fall-01            # live visualization window
  python main.py --demo le2i Home_01_video1
  python main.py --demo mc chute01
  python main.py --demo urfd fall-01 --save out.mp4

Notes:
  - In the visualization window, Space pauses/resumes and ESC quits.
  - Data path: the script finds the project's data/ folder (two levels above this file).
"""
import os
import sys
import math
import glob
import pickle
import argparse

# ---- dependency check: a clear message when an IDE uses the wrong interpreter ----
_MISSING = []
for _mod in ("numpy", "cv2", "torch", "sklearn", "xgboost", "pandas"):
    try:
        __import__(_mod)
    except Exception:
        _MISSING.append(_mod)
if _MISSING:
    sys.stderr.write(
        "\n[missing dependencies] missing: %s\nRun it with a Python that has them installed:\n"
        "  python main.py --demo urfd fall-01\n\n" % ", ".join(_MISSING))
    sys.exit(1)

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared A4 core (imported from the same folder, so no cross-folder path issues)
from a4_core import (make_windows, extract_window_features, HybridAlgorithm,
                     KinematicBase, TemporalBase, WIN_FRAMES, STEP_FRAMES, IN_DIM)

# ===========================================================================
# Paths
# ===========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))            # standalone/
PROJECT = os.path.abspath(os.path.join(HERE, ".."))          # project root
YOLO_ONNX = os.path.join(HERE, "yolov8n-pose.onnx")
POSE_CKPT = os.path.join(HERE, "pose_best.pt")
A4_WEIGHTS = os.path.join(HERE, "a4_weights.pkl")
URFD_ROOT = os.path.join(PROJECT, "data", "urfd")

SRC_FPS = {"urfd": 30.0, "mc": 30.0, "le2i": 25.0}
SAMPLE_FPS = 15.0

# ===========================================================================
# Data sources and ground-truth lookup (inline, for the real datasets only)
# ===========================================================================
def _kaggle_dir(slug):
    home = os.path.expanduser("~")
    for base in (os.path.join(home, ".cache", "kagglehub", "datasets"),
                 os.path.join(PROJECT, "00_common", "cache", "kaggle")):
        for c in glob.glob(os.path.join(base, "**", slug), recursive=True):
            if os.path.isdir(c) and len(os.listdir(c)):
                v = os.path.join(c, "versions")
                if os.path.isdir(v):
                    vers = sorted(os.listdir(v))
                    if vers:
                        return os.path.join(v, vers[-1])
                return c
    return None


def _urfd_frames_dir(seq):
    return os.path.join(URFD_ROOT, "frames", f"{seq}-cam0-rgb")


def _urfd_fall_intervals_s(seq):
    if not seq.startswith("fall"):
        return []
    path = os.path.join(URFD_ROOT, "csv", "urfall-cam0-falls.csv")
    df = pd.read_csv(path, header=None,
                     names=["seq", "frame", "label", "hwr", "mmr", "bbo",
                            "stdxz", "hhmax", "H", "D", "p40"])
    g = df[df.seq == seq]
    l0 = g[g.label == 0]["frame"]
    l1 = g[g.label == 1]["frame"]
    f0 = int(l0.iloc[0]) if len(l0) else int(g["frame"].iloc[0])
    f1 = int(l1.iloc[0]) + 30 if len(l1) else int(g["frame"].max())
    return [(f0 / 30.0, f1 / 30.0)]


def _le2i_list_videos():
    root = _kaggle_dir("falldataset-imvia")
    out = []
    for scene in ("Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"):
        vdir = None
        for cand in (os.path.join(root, scene, scene, "Videos"),
                     os.path.join(root, scene, "Lecture room"),
                     os.path.join(root, scene, scene)):
            if os.path.isdir(cand) and any(f.lower().endswith(".avi") for f in os.listdir(cand)):
                vdir = cand
                break
        if not vdir:
            continue
        for f in sorted(os.listdir(vdir)):
            if f.lower().endswith(".avi"):
                i = f.split("(")[1].split(")")[0]
                out.append((f"{scene}_video{i}", os.path.join(vdir, f)))
    return out


def _le2i_fall_intervals_s(seq_name):
    scene = "_".join(seq_name.split("_")[:-1])
    i = seq_name.rsplit("video", 1)[1]
    root = _kaggle_dir("falldataset-imvia")
    ap = None
    for cand in (os.path.join(root, scene, scene, "Annotation_files"),
                 os.path.join(root, scene, scene, "Annotations_files"),
                 os.path.join(root, scene, "Annotation_files")):
        p = os.path.join(cand, f"video ({i}).txt")
        if os.path.isfile(p):
            ap = p
            break
    if not ap:
        return []
    with open(ap, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2 or "," in lines[0] or "," in lines[1]:
        return []
    f0, f1 = int(float(lines[0])), int(float(lines[1]))
    if 0 < f0 <= f1:
        return [(f0 / 25.0, f1 / 25.0)]
    return []


def _mc_video_path(seq):
    root = _kaggle_dir("multiple-cameras-fall-dataset")
    inner = os.path.join(root, "dataset", "dataset")
    chutes = inner if os.path.isdir(inner) else root
    return os.path.join(chutes, seq, "cam1.avi")


def _mc_fall_intervals_s(seq):
    root = _kaggle_dir("multiple-cameras-fall-dataset")
    csv_path = os.path.join(root, "data_tuple3.csv")
    df = pd.read_csv(csv_path)
    cid = int(seq.replace("chute", ""))
    g = df[(df.chute == cid) & (df.cam == 1)]
    return [(float(r.start) / 30.0, float(r.end) / 30.0)
            for r in g.itertuples() if r.label == 1]


def _source_of(dataset, seq):
    if dataset == "urfd":
        return _urfd_frames_dir(seq), 30.0
    if dataset == "mc":
        # MCF videos are actually 120 fps (high-speed recording), while the annotations use 30 fps.
        # The sampling step uses the real fps so the sampling density is correct and inference is not too slow.
        return _mc_video_path(seq), 120.0
    if dataset == "le2i":
        for name, path in _le2i_list_videos():
            if name == seq:
                return path, 25.0
    raise ValueError(f"Unknown sequence {dataset}/{seq}")


def _gt_intervals(dataset, seq):
    if dataset == "urfd":
        return _urfd_fall_intervals_s(seq)
    if dataset == "mc":
        return _mc_fall_intervals_s(seq)
    return _le2i_fall_intervals_s(seq)


def _iter_frames(dataset, seq, sample_fps=15.0):
    src, src_fps = _source_of(dataset, seq)
    step = max(1, int(round(src_fps / sample_fps))) if src_fps else 2
    if os.path.isdir(src):
        files = sorted(f for f in os.listdir(src)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
        for i, fn in enumerate(files):
            if i % step == 0:
                img = cv2.imread(os.path.join(src, fn), cv2.IMREAD_COLOR)
                if img is not None:
                    yield i, img
    else:
        cap = cv2.VideoCapture(src)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield i, frame
            i += 1
        cap.release()


# ===========================================================================
# Stage 1: YOLO person detection (live ONNX inference, box head only)
# ===========================================================================
class YOLOPersonDetector:
    def __init__(self, model_path=YOLO_ONNX, conf=0.35, iou=0.45):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf, self.iou = conf, iou

    def detect_frame(self, img):
        h, w = img.shape[:2]
        r = min(640 / h, 640 / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        px, py = (640 - nw) // 2, (640 - nh) // 2
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]  # [56,8400]
        scores = out[4]
        keep = scores > self.conf
        if not keep.any():
            return None, 0.0
        boxes = out[:4, keep].T
        ss = scores[keep]
        kpts_conf = out[5::3, keep].max(0) if out.shape[0] >= 56 else ss
        final = 0.5 * ss + 0.5 * kpts_conf
        xy = boxes[:, :2] - boxes[:, 2:] / 2
        xywh = np.concatenate([xy, boxes[:, 2:]], 1)
        nms = cv2.dnn.NMSBoxes(xywh.tolist(), ss.tolist(), self.conf, self.iou)
        idx = np.array(nms).ravel() if len(nms) else np.arange(len(ss))
        best = idx[np.argmax(final[idx])]
        cx, cy, bw, bh = boxes[best]
        x1 = (cx - bw / 2 - px) / r; y1 = (cy - bh / 2 - py) / r
        x2 = (cx + bw / 2 - px) / r; y2 = (cy + bh / 2 - py) / r
        box = np.clip(np.array([x1 / w, y1 / h, x2 / w, y2 / h], np.float32), 0, 1)
        return box, float(ss[best])


# ===========================================================================
# Stage 2: our small P1 U-Net pose estimation (weights trained from scratch, live inference)
# ===========================================================================
IMG_H, IMG_W = 256, 192
OUT_H, OUT_W = 64, 48
NUM_KPTS = 17
BBOX_EXPAND = 1.25


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True))


class SmallUNet(nn.Module):
    def __init__(self, c_in=3, width=(32, 64, 128, 256)):
        super().__init__()
        w1, w2, w3, w4 = width
        self.enc1 = nn.Sequential(ConvBNReLU(c_in, w1), ConvBNReLU(w1, w1))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(ConvBNReLU(w1, w2), ConvBNReLU(w2, w2))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(ConvBNReLU(w2, w3), ConvBNReLU(w3, w3))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(ConvBNReLU(w3, w4), ConvBNReLU(w4, w4))
        self.pool4 = nn.MaxPool2d(2)
        self.mid = ConvBNReLU(w4, w4)
        self.up3 = nn.ConvTranspose2d(w4, w3, 2, 2)
        self.dec3 = nn.Sequential(ConvBNReLU(w3 + w4, w3), ConvBNReLU(w3, w3))
        self.up2 = nn.ConvTranspose2d(w3, w2, 2, 2)
        self.dec2 = nn.Sequential(ConvBNReLU(w2 + w3, w2), ConvBNReLU(w2, w2))
        self.head = nn.Conv2d(w2, NUM_KPTS, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        m = self.mid(self.pool4(e4))
        d3 = self.dec3(torch.cat([self.up3(m), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e3], dim=1))
        return [self.head(d2)]


def _affine_matrix(center, scale, rot_deg, out_w, out_h):
    cv = math.cos(math.radians(rot_deg))
    sv = math.sin(math.radians(rot_deg))
    sx = scale * cv; sy = scale * cv
    shear_x = -scale * sv; shear_y = scale * sv
    tx = out_w / 2.0 - sx * center[0] - shear_x * center[1]
    ty = out_h / 2.0 - sy * center[1] - shear_y * center[0]
    return np.asarray([[sx, shear_x, tx], [shear_y, sy, ty]], dtype=np.float64)


class PoseExtractor:
    """Our P1 pose estimator (live inference, no cache)."""
    def __init__(self, ckpt_path=POSE_CKPT, device=None):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = SmallUNet()
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extract(self, img, bbox_px):
        """img BGR; bbox_px [x1,y1,x2,y2] in pixels -> (kpts [17,2], conf [17])."""
        x1, y1, x2, y2 = [float(v) for v in bbox_px]
        bbox = np.asarray([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)], np.float32)
        cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
        side = max(bbox[2], bbox[3]) * BBOX_EXPAND
        scale = IMG_W / side
        M = _affine_matrix((cx, cy), scale, 0.0, IMG_W, IMG_H)
        crop = cv2.warpAffine(img, M, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        Minv = cv2.invertAffineTransform(M)
        t = torch.from_numpy(np.ascontiguousarray(crop[:, :, ::-1]))
        t = t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            hm = self.model(t.to(self.device))[-1][0]   # [17,64,48]
        conf, idx = hm.view(NUM_KPTS, -1).max(-1)
        xs = (idx % hm.shape[2]).float() * (IMG_W / hm.shape[2])
        ys = (idx // hm.shape[2]).float() * (IMG_H / hm.shape[1])
        pts_in = torch.stack([xs, ys], -1).cpu().numpy().astype(np.float64)
        pts_orig = pts_in @ Minv[:, :2].T + Minv[:, 2]
        return pts_orig.astype(np.float32), conf.cpu().numpy()


# ===========================================================================
# Stage 3: A4 hybrid ensemble (trained weights loaded, live inference)
# ===========================================================================
class A4Algorithm:
    """Load the pickled weights (a4_weights.pkl) for live inference."""
    def __init__(self, weights_path=A4_WEIGHTS, device=None):
        with open(weights_path, "rb") as f:
            data = pickle.load(f)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bases = data["bases"]
        self.meta = data["meta"]
        for b in self.bases:
            if isinstance(b, TemporalBase):
                b.net.to(self.device).eval()

    def predict(self, kpts):
        preds = []
        for b in self.bases:
            if isinstance(b, KinematicBase):
                preds.append(b.predict(kpts))
            else:
                preds.append(b.predict(kpts))
        if len(preds[0]) == 0:
            return preds[0]
        X = np.stack(preds, axis=1).astype(np.float32)
        if self.meta is None:
            return np.mean(X, axis=1).astype(np.float32)
        return self.meta.predict_proba(X)[:, 1].astype(np.float32)


# ===========================================================================
# Visualization
# ===========================================================================
COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
              (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
EDGE_COLORS = [(255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0), (255, 0, 0),
               (0, 255, 0), (255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 0),
               (255, 0, 0), (0, 255, 0), (255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 0)]


def draw_skeleton(img, kpts, conf, box_norm=None, thr=0.3):
    H, W = img.shape[:2]
    if box_norm is not None:
        x1, y1, x2, y2 = box_norm
        cv2.rectangle(img, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)),
                      (255, 128, 0), 1)
    pts = []
    for j in range(17):
        x, y = int(kpts[j][0]), int(kpts[j][1])
        pts.append((x, y))
        if conf[j] > thr:
            color = (0, 0, 255) if conf[j] < 0.5 else (0, 255, 255)
            cv2.circle(img, (x, y), 3, color, -1)
    for ei, (i0, i1) in enumerate(COCO_EDGES):
        if conf[i0] > thr and conf[i1] > thr:
            cv2.line(img, pts[i0], pts[i1], EDGE_COLORS[ei], 2)
    return img


# ===========================================================================
# Main flow
# ===========================================================================
def run_demo(dataset, seq, save_path=None, headless=False):
    print(f"[loading models] YOLO + P1 pose + A4 ...", flush=True)
    det = YOLOPersonDetector()
    pose = PoseExtractor()
    alg = A4Algorithm()

    gt = _gt_intervals(dataset, seq)
    print(f"[{dataset}/{seq}] ground-truth fall interval (s)={gt}", flush=True)
    print("[visualization] Space=pause/resume  ESC=quit", flush=True)

    win_buf = []
    kpts_hist = []
    writer = None
    paused = False
    frame_i = 0

    for fidx, frame in _iter_frames(dataset, seq, SAMPLE_FPS):
        H, W = frame.shape[:2]

        # ① YOLO
        box_norm, score = det.detect_frame(frame)
        box_px = None
        if box_norm is not None and score > 0:
            x1, y1, x2, y2 = box_norm
            box_px = [x1 * W, y1 * H, x2 * W, y2 * H]
            # Stage 2: pose
            pts, conf = pose.extract(frame, box_px)
            kpts_norm = np.stack([pts[:, 0] / W, pts[:, 1] / H, conf], axis=1).astype(np.float32)
            kpts_hist.append(kpts_norm)
        else:
            kpts_hist.append(np.zeros((17, 3), np.float32))

        # Stage 3: A4 sliding-window temporal decision
        cur_prob = 0.0
        if len(kpts_hist) >= WIN_FRAMES:
            kwin = np.stack(kpts_hist[-WIN_FRAMES:]).astype(np.float32)
            wp = alg.predict(kwin)
            cur_prob = float(wp[0]) if len(wp) else 0.0
            win_buf.append(cur_prob)

        # Visualization
        vis = frame.copy()
        if box_px is not None:
            draw_skeleton(vis, pts, conf, box_norm)
        t_s = fidx / SRC_FPS[dataset]
        label = "FALL" if cur_prob > 0.5 else "ADL"
        bar_color = (0, 0, 255) if cur_prob > 0.5 else (0, 180, 0)
        cv2.rectangle(vis, (0, 0), (W, 34), (40, 40, 40), -1)
        cv2.putText(vis, f"{dataset}/{seq}  t={t_s:.1f}s  P(fall)={cur_prob:.2f}  {label}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bar_color, 2)
        if len(win_buf) > 1:
            curve_h = 60
            cv2.rectangle(vis, (0, H - curve_h), (W, H), (20, 20, 20), -1)
            xs = np.linspace(0, W, len(win_buf))
            ys = (1 - np.asarray(win_buf)) * (curve_h - 10) + (H - curve_h + 5)
            pts_c = np.stack([xs, ys], axis=1).astype(np.int32)
            cv2.polylines(vis, [pts_c], False, (0, 255, 255), 1)
            cv2.putText(vis, "window P(fall)", (4, H - curve_h + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if save_path and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, SAMPLE_FPS, (W, H))
        if writer is not None:
            writer.write(vis)

        if not headless:
            cv2.imshow("Fall Detection - End-to-End", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 32:
                paused = not paused
                while paused:
                    k = cv2.waitKey(30) & 0xFF
                    if k in (27, 32):
                        paused = False
                        break

        frame_i += 1

    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        print(f"[saved] visualization video -> {save_path}")
    if win_buf:
        print(f"\n[{dataset}/{seq}] processed {frame_i} frames, maximum window probability={max(win_buf):.3f}")
    print(f"ground-truth fall interval (s)={gt}")
    return win_buf, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", nargs=2, metavar=("DATASET", "SEQ"),
                    help="end-to-end inference + visualization: --demo urfd fall-01")
    ap.add_argument("--save", default=None, help="path to save the visualization video (e.g. out.mp4)")
    ap.add_argument("--headless", action="store_true", help="no window (use with --save)")
    ap.add_argument("--list", action="store_true", help="list the available sequences and exit")
    a = ap.parse_args()

    if a.list:
        print("Available sequences (--demo <dataset> <sequence>):")
        print("  urfd :", " ".join(
            sorted(os.path.basename(d).split("-")[0]
                   for d in glob.glob(os.path.join(URFD_ROOT, "frames", "*-cam0-rgb")))))
        for name, _ in _le2i_list_videos():
            print("  le2i :", name)
        root = _kaggle_dir("multiple-cameras-fall-dataset")
        if root:
            chutes = sorted(os.path.basename(p) for p in
                            glob.glob(os.path.join(root, "dataset", "dataset", "chute*"))
                            if os.path.isdir(p))
            print("  mc   :", " ".join(chutes))
        return

    if a.demo:
        dataset, seq = a.demo
    else:
        # Without arguments run urfd fall-01 by default, so pressing Run shows the predictions live
        dataset, seq = "urfd", "fall-01"
        print("[default demo] no sequence given, using urfd/fall-01. "
              "Use --demo <dataset> <sequence> to choose another one.")
    run_demo(dataset, seq, save_path=a.save, headless=a.headless)


if __name__ == "__main__":
    main()
