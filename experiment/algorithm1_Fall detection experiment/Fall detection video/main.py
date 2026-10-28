"""End-to-end demo: detection, pose estimation, A4 decision and visualization."""

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "00_common")
sys.path.insert(0, COMMON)
sys.path.insert(0, os.path.join(HERE, "01_detect_yolo"))
sys.path.insert(0, os.path.join(HERE, "02_pose_cnn"))
sys.path.insert(0, os.path.join(HERE, "06_alg_hybrid", "src"))
_MISSING = []
for _mod in ("numpy", "cv2", "torch"):
    try:
        __import__(_mod)
    except Exception:
        _MISSING.append(_mod)
for _mod in ("sklearn", "xgboost", "pandas"):
    try:
        __import__(_mod)
    except Exception:
        _MISSING.append(_mod)
if _MISSING:
    sys.stderr.write(
        "\n[missing dependencies] the current Python interpreter lacks: %s\n"
        "Run it with the Anaconda environment, e.g.:\n"
        "  E:/anaconda3/python.exe main.py --demo urfd fall-01\n"
        "or switch the IDE interpreter to E:\\anaconda3\\python.exe (3.12.13).\n\n"
        % ", ".join(_MISSING)
    )
    sys.exit(1)
import cv2
import numpy as np
import splits
from windowize import make_windows, WIN_FRAMES, FPS

SRC_FPS = {"urfd": 30.0, "mc": 30.0, "le2i": 25.0}
SAMPLE_FPS = 15.0
COCO_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]
EDGE_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 0),
]


def load_best_algorithm():
    from hybrid_algorithm import HybridAlgorithm

    return HybridAlgorithm()


def load_detector():
    from yolo_person import YOLOPersonDetector

    return YOLOPersonDetector()


def load_pose_extractor():
    from export_infer import PoseCNNExtractor

    return PoseCNNExtractor()


def _source_of(dataset, seq):
    import urfd_loader, mc_loader, le2i_loader

    if dataset == "urfd":
        return (urfd_loader.frames_dir(seq), 30.0)
    if dataset == "mc":
        return (mc_loader.video_path(seq), 30.0)
    return (dict(le2i_loader.list_videos())[seq], 25.0)


def _gt_intervals(dataset, seq):
    import urfd_loader, mc_loader, le2i_loader

    if dataset == "urfd":
        return urfd_loader.fall_intervals_s(seq)
    if dataset == "mc":
        return mc_loader.fall_intervals_s(seq)
    return le2i_loader.fall_intervals_s(seq)


def _iter_frames(dataset, seq, sample_fps=15.0):
    src, src_fps = _source_of(dataset, seq)
    step = max(1, int(round(src_fps / sample_fps))) if src_fps else 2
    if os.path.isdir(src):
        files = sorted(
            (
                f
                for f in os.listdir(src)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            )
        )
        for i, fn in enumerate(files):
            if i % step == 0:
                img = cv2.imread(os.path.join(src, fn), cv2.IMREAD_COLOR)
                if img is not None:
                    yield (i, img)
    else:
        cap = cv2.VideoCapture(src)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield (i, frame)
            i += 1
        cap.release()


def _fit_on_pool(alg, dataset, seq):
    pool = splits.unified_pool()
    items = []
    for d, s, *_ in pool:
        if d == dataset and s == seq:
            continue
        from pose_extract import extract_skeleton_sequence

        dd = extract_skeleton_sequence(d, s)
        kpts = dd["kpts"]
        fidx = dd["frame_idx"]
        fps = SRC_FPS[d]
        wins, t0s, t1s = make_windows(kpts)
        t0_s = fidx[t0s].astype(np.float64) / fps if len(t0s) else np.array([])
        import urfd_loader, mc_loader, le2i_loader

        if d == "urfd":
            iv = urfd_loader.fall_intervals_s(s)
        elif d == "mc":
            iv = mc_loader.fall_intervals_s(s)
        else:
            iv = le2i_loader.fall_intervals_s(s)
        items.append((kpts, t0_s, iv))
    alg.fit(items)
    return alg


def draw_skeleton(img, kpts, conf, box=None, thr=0.3):
    H, W = img.shape[:2]
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(
            img,
            (int(x1 * W), int(y1 * H)),
            (int(x2 * W), int(y2 * H)),
            (255, 128, 0),
            1,
        )
    pts = []
    for j in range(17):
        x = int(kpts[j][0])
        y = int(kpts[j][1])
        pts.append((x, y))
        c = conf[j]
        if c > thr:
            color = (0, 0, 255) if c < 0.5 else (0, 255, 255)
            cv2.circle(img, (x, y), 3, color, -1)
    for ei, (i0, i1) in enumerate(COCO_EDGES):
        if conf[i0] > thr and conf[i1] > thr:
            cv2.line(img, pts[i0], pts[i1], EDGE_COLORS[ei], 2)
    return img


def run_demo(dataset, seq, save_path=None, fit_pool=True, headless=False):
    det = load_detector()
    pose = load_pose_extractor()
    alg = load_best_algorithm()
    if fit_pool:
        print(
            f"[fit A4] training on the remaining sequences of the unified pool ...",
            flush=True,
        )
        _fit_on_pool(alg, dataset, seq)
    else:
        print(
            "[note] A4 not fitted (--no-fit); falls back to averaging the base learners"
        )
    gt = _gt_intervals(dataset, seq)
    print(f"[{dataset}/{seq}] GT fall intervals (s)={gt}")
    win_buf = []
    win_t0_buf = []
    kpts_hist = []
    writer = None
    paused = False
    frame_i = 0
    print("[viewer] space=pause/resume  ESC=quit", flush=True)
    for fidx, frame in _iter_frames(dataset, seq, SAMPLE_FPS):
        H, W = frame.shape[:2]
        box_norm, score = det.detect_frame(frame)
        box_px = None
        if box_norm is not None and score > 0:
            x1, y1, x2, y2 = box_norm
            box_px = [x1 * W, y1 * H, x2 * W, y2 * H]
            pts, conf = pose.extract(frame, box_px)
            kpts_norm = np.stack([pts[:, 0] / W, pts[:, 1] / H, conf], axis=1).astype(
                np.float32
            )
            kpts_hist.append(kpts_norm)
        else:
            kpts_hist.append(np.zeros((17, 3), np.float32))
        cur_prob = 0.0
        if len(kpts_hist) >= WIN_FRAMES:
            kwin = np.stack(kpts_hist[-WIN_FRAMES:]).astype(np.float32)
            wp = alg.predict(kwin)
            cur_prob = float(wp[0]) if len(wp) else 0.0
            win_buf.append(cur_prob)
            win_t0_buf.append(fidx - WIN_FRAMES)
        vis = frame.copy()
        if box_px is not None:
            draw_skeleton(vis, pts, conf, box_norm)
        t_s = fidx / SRC_FPS[dataset]
        label = "FALL" if cur_prob > 0.5 else "ADL"
        bar_color = (0, 0, 255) if cur_prob > 0.5 else (0, 180, 0)
        cv2.rectangle(vis, (0, 0), (W, 34), (40, 40, 40), -1)
        cv2.putText(
            vis,
            f"{dataset}/{seq}  t={t_s:.1f}s  P(fall)={cur_prob:.2f}  {label}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            bar_color,
            2,
        )
        if len(win_buf) > 1:
            curve_h = 60
            cv2.rectangle(vis, (0, H - curve_h), (W, H), (20, 20, 20), -1)
            xs = np.linspace(0, W, len(win_buf))
            ys = (1 - np.asarray(win_buf)) * (curve_h - 10) + (H - curve_h + 5)
            pts_c = np.stack([xs, ys], axis=1).astype(np.int32)
            cv2.polylines(vis, [pts_c], False, (0, 255, 255), 1)
            cv2.line(
                vis,
                (0, int((1 - 0.5) * (curve_h - 10) + (H - curve_h + 5))),
                (W, int((1 - 0.5) * (curve_h - 10) + (H - curve_h + 5))),
                (0, 0, 255),
                1,
            )
            cv2.putText(
                vis,
                "window P(fall)",
                (4, H - curve_h + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )
        if save_path and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, SAMPLE_FPS, (W, H))
        if not headless:
            cv2.imshow("Fall Detection - End-to-End", vis)
            key = cv2.waitKey(1) & 255
            if key == 27:
                break
            elif key == 32:
                paused = not paused
                while paused:
                    k = cv2.waitKey(30) & 255
                    if k == 32:
                        paused = False
                    elif k == 27:
                        paused = False
                        break
        frame_i += 1
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        print(f"[save] visualization video -> {save_path}")
    print(
        f"\n[{dataset}/{seq}] processed {frame_i} frames, max window probability={max(win_buf):.3f}"
        if win_buf
        else "no windows"
    )
    print(f"GT fall intervals (s)={gt}")
    return (win_buf, win_t0_buf, gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--demo",
        nargs=2,
        metavar=("DATASET", "SEQ"),
        help="end-to-end inference + visualization: --demo urfd fall-01",
    )
    ap.add_argument(
        "--save",
        default=None,
        help="path to save the visualization video (e.g. out.mp4)",
    )
    ap.add_argument(
        "--no-fit",
        action="store_true",
        help="skip A4 fitting (use the default fallback)",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="headless mode (no imshow; combine with --save)",
    )
    a = ap.parse_args()
    if not a.demo:
        print("examples:")
        print("  python main.py --demo urfd fall-01")
        print("  python main.py --demo le2i Home_01_video1 --save out.mp4")
        print("  python main.py --demo mc chute01")
        return
    dataset, seq = a.demo
    run_demo(dataset, seq, save_path=a.save, fit_pool=not a.no_fit, headless=a.headless)


if __name__ == "__main__":
    main()
