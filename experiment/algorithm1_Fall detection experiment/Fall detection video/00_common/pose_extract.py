"""Build the skeleton cache from the detection cache."""

import os
import sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import urfd_root

DET_CACHE = os.path.join(HERE, "cache", "detections")
SKEL_CACHE = os.path.join(HERE, "cache", "skeletons")
POSE_CNN_DIR = os.path.abspath(os.path.join(HERE, "..", "02_pose_cnn"))
YOLO_DIR = os.path.abspath(os.path.join(HERE, "..", "01_detect_yolo"))


def _load_det(dataset, seq):
    path = os.path.join(DET_CACHE, dataset, f"{seq}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing detection cache {path}; run the 01_detect_yolo batch first"
        )
    d = np.load(path)
    return (d["boxes"], d["scores"], d["frame_idx"], float(d["fps"]))


def _iter_frames(dataset, seq, frame_idx):
    sys.path.insert(0, os.path.join(HERE, "..", "01_detect_yolo"))
    src = _source_of(dataset, seq)
    if os.path.isdir(src):
        files = sorted(
            (
                f
                for f in os.listdir(src)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            )
        )
        for fi in frame_idx:
            yield cv2.imread(os.path.join(src, files[fi]), cv2.IMREAD_COLOR)
    else:
        cap = cv2.VideoCapture(src)
        need = set((int(i) for i in frame_idx))
        i, emitted = (0, 0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i in need:
                yield frame
                emitted += 1
                if emitted == len(need):
                    break
            i += 1
        cap.release()


def _source_of(dataset, seq):
    import urfd_loader, mc_loader, le2i_loader

    if dataset == "urfd":
        return urfd_loader.frames_dir(seq)
    if dataset == "mc":
        return mc_loader.video_path(seq)
    if dataset == "le2i":
        return dict(le2i_loader.list_videos())[seq]
    raise ValueError(dataset)


def extract_skeleton_sequence(dataset, seq, ckpt_path=None, force=False, use_gpu=True):
    arch_tag = "self-pose"
    sdir = os.path.join(SKEL_CACHE, dataset)
    os.makedirs(sdir, exist_ok=True)
    spath = os.path.join(sdir, f"{seq}_{arch_tag}.npz")
    if os.path.exists(spath) and (not force):
        d = np.load(spath)
        return {k: d[k] for k in ("kpts", "boxes", "scores", "frame_idx", "fps")}
    boxes, scores, fidx, fps = _load_det(dataset, seq)
    sys.path.insert(0, POSE_CNN_DIR)
    from export_infer import PoseCNNExtractor

    ext = (
        PoseCNNExtractor()
        if ckpt_path is None
        else PoseCNNExtractor(ckpt_path=ckpt_path)
    )
    if not use_gpu:
        ext.device = __import__("torch").device("cpu")
        ext.model.to(ext.device)
    H = W = None
    kpts = np.zeros((len(fidx), 17, 3), np.float32)
    for t, (fi, img) in enumerate(zip(fidx, _iter_frames(dataset, seq, fidx))):
        if img is None:
            continue
        if H is None:
            H, W = img.shape[:2]
        b = boxes[t]
        if scores[t] <= 0 or b[2] - b[0] <= 0:
            continue
        bb = [b[0] * W, b[1] * H, b[2] * W, b[3] * H]
        pts, conf = ext.extract(img, bb)
        kpts[t, :, 0] = pts[:, 0] / W
        kpts[t, :, 1] = pts[:, 1] / H
        kpts[t, :, 2] = conf
    np.savez_compressed(
        spath,
        kpts=kpts,
        boxes=boxes,
        scores=scores,
        frame_idx=fidx,
        fps=np.float32(fps),
    )
    return dict(kpts=kpts, boxes=boxes, scores=scores, frame_idx=fidx, fps=fps)


def main(dataset, limit=None, force=False):
    sys.path.insert(0, HERE)
    import urfd_loader, mc_loader, le2i_loader

    if dataset == "urfd":
        falls, adls = urfd_loader.list_sequences()
        seqs = falls + adls
    elif dataset == "mc":
        seqs = mc_loader.list_sequences()
    elif dataset == "le2i":
        seqs = [s for s, _ in le2i_loader.list_videos()]
    else:
        raise SystemExit(f"unknown dataset {dataset}")
    if limit:
        seqs = seqs[:limit]
    import time

    for i, seq in enumerate(seqs, 1):
        t0 = time.time()
        d = extract_skeleton_sequence(dataset, seq, force=force)
        valid = float((d["kpts"][:, :, 2] > 0.3).mean())
        print(
            f"  [{dataset} {i}/{len(seqs)}] {seq} T={len(d['frame_idx'])} keypoint valid rate{valid:.1%} {time.time() - t0:.0f}s",
            flush=True,
        )
    print(f"[{dataset}] skeleton extraction done")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["urfd", "mc", "le2i"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.dataset, a.limit, a.force)
