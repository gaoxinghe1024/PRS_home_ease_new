"""Stage 1: YOLOv8n-pose ONNX person detection, writes the detection cache."""

import os
import sys
import time
import glob
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "00_common"))
MODEL = os.path.join(COMMON, "models", "yolov8n-pose.onnx")
DET_CACHE = os.path.join(COMMON, "cache", "detections")
INPUT_SIZE = 640
CONF_THR = 0.35
IOU_THR = 0.45


def _letterbox(img, new_shape=INPUT_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = (int(round(h * r)), int(round(w * r)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = ((new_shape - nw) // 2, (new_shape - nh) // 2)
    out = np.full((new_shape, new_shape, 3), color, np.uint8)
    out[py : py + nh, px : px + nw] = resized
    return (out, r, px, py)


class YOLOPersonDetector:

    def __init__(self, model_path=MODEL, conf=CONF_THR, iou=IOU_THR):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf, self.iou = (conf, iou)

    def detect_frame(self, img):
        h, w = img.shape[:2]
        lb, r, px, py = _letterbox(img)
        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]
        out = out[0]
        scores = out[4]
        keep = scores > self.conf
        if not keep.any():
            return (None, 0.0)
        boxes_cxcywh = out[:4, keep].T
        ss = scores[keep]
        kpts_conf = out[5::3, keep].max(0) if out.shape[0] >= 56 else ss
        final = 0.5 * ss + 0.5 * kpts_conf
        xy = boxes_cxcywh[:, :2] - boxes_cxcywh[:, 2:] / 2
        xywh = np.concatenate([xy, boxes_cxcywh[:, 2:]], 1)
        nms_idx = cv2.dnn.NMSBoxes(xywh.tolist(), ss.tolist(), self.conf, self.iou)
        idx = np.array(nms_idx).ravel() if len(nms_idx) else np.arange(len(ss))
        best = idx[np.argmax(final[idx])]
        cx, cy, bw, bh = boxes_cxcywh[best]
        x1 = (cx - bw / 2 - px) / r
        y1 = (cy - bh / 2 - py) / r
        x2 = (cx + bw / 2 - px) / r
        y2 = (cy + bh / 2 - py) / r
        box = np.array([x1 / w, y1 / h, x2 / w, y2 / h], np.float32)
        box = np.clip(box, 0, 1)
        return (box, float(ss[best]))

    def _iter_source(self, source, src_fps, sample_fps):
        if os.path.isdir(source):
            files = sorted(
                (
                    f
                    for f in os.listdir(source)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                )
            )
            if not files:
                raise FileNotFoundError(f"frame directory is empty: {source}")
            step = max(1, int(round(src_fps / sample_fps))) if src_fps else 2
            for i, fn in enumerate(files):
                if i % step == 0:
                    img = cv2.imread(os.path.join(source, fn), cv2.IMREAD_COLOR)
                    if img is not None:
                        yield (i, img)
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise FileNotFoundError(source)
            if src_fps is None:
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(round(src_fps / sample_fps)))
            i, ok = (0, True)
            while ok:
                ok, frame = cap.read()
                if ok and i % step == 0:
                    yield (i, frame)
                i += 1
            cap.release()

    def video_detections(self, source, sample_fps=15.0, src_fps=None, max_frames=None):
        if src_fps is None and os.path.isdir(source):
            src_fps = 30.0
        boxes, scores, fidx = ([], [], [])
        for i, frame in self._iter_source(source, src_fps, sample_fps):
            if max_frames and i >= max_frames:
                break
            b, s = self.detect_frame(frame)
            boxes.append(b if b is not None else np.zeros(4, np.float32))
            scores.append(s)
            fidx.append(i)
        return dict(
            boxes=np.asarray(boxes, np.float32),
            scores=np.asarray(scores, np.float32),
            frame_idx=np.asarray(fidx, np.int64),
            fps=float(sample_fps),
        )


def _detect_video_cached(det, dataset, seq, source, sample_fps=15.0, src_fps=None):
    cdir = os.path.join(DET_CACHE, dataset)
    os.makedirs(cdir, exist_ok=True)
    cpath = os.path.join(cdir, f"{seq}.npz")
    if os.path.exists(cpath):
        d = np.load(cpath)
        return dict(
            boxes=d["boxes"],
            scores=d["scores"],
            frame_idx=d["frame_idx"],
            fps=float(d["fps"]),
        )
    t0 = time.time()
    d = det.video_detections(source, sample_fps, src_fps)
    np.savez_compressed(
        cpath,
        boxes=d["boxes"],
        scores=d["scores"],
        frame_idx=d["frame_idx"],
        fps=d["fps"],
    )
    det_rate = float((d["scores"] > 0).mean())
    print(
        f"  [{dataset}/{seq}] {len(d['frame_idx'])} frames, detection rate{det_rate:.1%} {time.time() - t0:.0f}s",
        flush=True,
    )
    return d


def main(dataset, limit=None):
    sys.path.insert(0, COMMON)
    from paths import urfd_root
    import urfd_loader, mc_loader, le2i_loader

    det = YOLOPersonDetector()
    if dataset == "urfd":
        falls, adls = urfd_loader.list_sequences()
        items = [(s, urfd_loader.frames_dir(s), 30.0) for s in falls + adls]
    elif dataset == "mc":
        items = [(s, mc_loader.video_path(s), 30.0) for s in mc_loader.list_sequences()]
    elif dataset == "le2i":
        items = [(s, src, 25.0) for s, src in le2i_loader.list_videos()]
    else:
        raise SystemExit(f"unknown dataset {dataset}")
    if limit:
        items = items[:limit]
    for seq, src, fps in items:
        _detect_video_cached(det, dataset, seq, src, 15.0, fps)
    print(f"[{dataset}] all done")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["urfd", "mc", "le2i"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.dataset, a.limit)
