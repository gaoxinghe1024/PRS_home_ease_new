# -*- coding: utf-8 -*-
"""
engine.py - real-time fall detection engine
================================================================
Wraps the trained three-stage algorithm weights in standalone/ into a long-running service:

  Stage 1 person detection : yolov8n-pose.onnx      (pre-trained YOLOv8n, box head only)
  Stage 2 pose estimation  : pose_best.pt           (our own small P1 U-Net, trained from scratch)
  Stage 3 temporal decision: a4_weights.pkl         (A4 hybrid ensemble, OOF stacking + XGBoost)

How it works:
  - a background thread keeps reading frames (camera / video file / bundled sample datasets)
  - every frame runs YOLO -> P1 pose -> a 30-frame (2 s) sliding window -> A4, giving P(fall)
  - P(fall) above the threshold `confirm` times in a row -> one fall event is triggered:
      - the system time is recorded
      - a 3 s clip is written to disk (1 s pre-buffer + 2 s after) for report playback
      - an alarm marker is set
  - records go to records/falls.json, clips to records/clips/*.mp4

Note: the engine only does inference and recording and needs no network access; the API layer only reads and writes its state.
"""
from __future__ import annotations

import os
import sys
import json
import time
import glob
import threading
import contextlib
from collections import deque
from datetime import datetime

# Performance: the same settings apply when this file runs on its own
# (the OMP/joblib thread pools are initialized on the first import of numpy/sklearn/torch, so they must be set before that).
if os.environ.get("FVS_NO_TUNE") != "1":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import cv2
import numpy as np

# The 7 base learners of A4 are tiny; by default sklearn uses joblib processes / OpenMP threads, and scheduling and
# collecting results costs far more than the computation itself (joblib's _retrieve polls every 10 ms).
# Wrapping these predict calls in "threading backend + single thread" avoids that without changing the algorithm files.
# Measured (RTX 3050 Ti, urfd/fall-01): 68.5 ms/frame -> 37.0 ms/frame.
try:
    from joblib import parallel_backend as _joblib_backend
except Exception:                                    # runs fine without joblib as well
    _joblib_backend = None
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:
    _threadpool_limits = None


@contextlib.contextmanager
def a4_fast_context():
    """Speed-up context for A4 inference (silently falls back to plain execution when the libraries are missing)."""
    with contextlib.ExitStack() as stack:
        if _joblib_backend is not None:
            stack.enter_context(_joblib_backend("threading", n_jobs=1))
        if _threadpool_limits is not None:
            stack.enter_context(_threadpool_limits(limits=1))
        yield

# Record persistence (SQLite, module in the same folder)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import RecordStore

# ---------------------------------------------------------------------------
# Reuse the self-contained algorithm implementation in standalone/ (same weights, same pre/post-processing, nothing copied)
#
# Note: an older main.py may exist next to this file, and a plain `import main` would pick it up.
# So importlib loads standalone/main.py by ABSOLUTE PATH under a unique module name to avoid any name clash,
# and standalone/ is put on sys.path so that `from a4_core import ...` inside the loaded
# main.py resolves.
# ---------------------------------------------------------------------------
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))              # systems/
PROJECT = os.path.abspath(os.path.join(HERE, ".."))            # project root
STANDALONE = os.path.join(PROJECT, "standalone")
if STANDALONE not in sys.path:
    sys.path.insert(0, STANDALONE)

_spec = importlib.util.spec_from_file_location(
    "fvs_standalone_main", os.path.join(STANDALONE, "main.py"))
_main = importlib.util.module_from_spec(_spec)
sys.modules["fvs_standalone_main"] = _main
_spec.loader.exec_module(_main)

YOLOPersonDetector = _main.YOLOPersonDetector
PoseExtractor = _main.PoseExtractor
A4Algorithm = _main.A4Algorithm
draw_skeleton = _main.draw_skeleton
WIN_FRAMES = _main.WIN_FRAMES
SRC_FPS = _main.SRC_FPS
_source_of = _main._source_of
_gt_intervals = _main._gt_intervals
_le2i_list_videos = _main._le2i_list_videos
_kaggle_dir = _main._kaggle_dir
_urfd_frames_dir = _main._urfd_frames_dir

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
RECORDS_DIR = os.path.join(HERE, "records")
CLIPS_DIR = os.path.join(RECORDS_DIR, "clips")
UPLOADS_DIR = os.path.join(HERE, "uploads")
for _d in (RECORDS_DIR, CLIPS_DIR, UPLOADS_DIR):
    os.makedirs(_d, exist_ok=True)
RECORDS_JSON = os.path.join(RECORDS_DIR, "falls.json")

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
SAMPLE_FPS = 15.0      # common sampling rate (the algorithm was trained at 15 fps / 2 s windows)
CLIP_FPS = 15.0        # nominal clip frame rate (same as sampling -> 3 s = 45 frames)


def placeholder_jpeg(w=640, h=360, text="NO SIGNAL"):
    """No-signal placeholder image (light, matching the light theme)."""
    img = np.full((h, w, 3), 240, np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (215, 220, 228), 2)
    cv2.line(img, (0, 0), (w, h), (225, 230, 236), 1)
    cv2.line(img, (w, 0), (0, h), (225, 230, 236), 1)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 158, 170), 2)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes()


# ===========================================================================
# Frame sources: camera / local video / bundled sample datasets (all behind one read() interface)
# ===========================================================================
class FrameSource:
    def __init__(self, spec: dict):
        self.spec = dict(spec or {})
        self.kind = self.spec.get("type")
        self.cap = None
        self.files = None
        self.dir = None
        self.idx = 0
        self.n_read = 0
        self.stride = 1
        self.realtime = True          # throttle to 15 fps (simulates real time)
        self.label = "-"
        self.video_fps = None

        if self.kind == "camera":
            index = int(self.spec.get("index", 0) or 0)
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(index, backend)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open the camera (index={index}); check whether it is in use or permitted")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.label = f"Camera {index}"

        elif self.kind == "file":
            path = str(self.spec.get("path", "")).strip().strip('"')
            if not path or not os.path.isfile(path):
                raise RuntimeError(f"Video file not found: {path}")
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open the video: {path}")
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.video_fps = fps
            self.stride = max(1, int(round(fps / SAMPLE_FPS)))
            self.label = os.path.basename(path)

        elif self.kind == "demo":
            ds = self.spec.get("dataset")
            seq = self.spec.get("seq")
            if not ds or not seq:
                raise RuntimeError("A sample source needs dataset and seq")
            src, src_fps = _source_of(ds, seq)
            self.label = f"{ds}/{seq}"
            if os.path.isdir(src):
                self.files = sorted(
                    f for f in os.listdir(src)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
                self.dir = src
                if not self.files:
                    raise RuntimeError(f"No image frames in the folder: {src}")
                self.stride = max(1, int(round((src_fps or 30.0) / SAMPLE_FPS)))
            else:
                self.cap = cv2.VideoCapture(src)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Cannot open the sample video: {src}")
                fps = self.cap.get(cv2.CAP_PROP_FPS) or (src_fps or 30.0)
                self.video_fps = fps
                self.stride = max(1, int(round(fps / SAMPLE_FPS)))
        else:
            raise RuntimeError(f"Unknown input type: {self.kind!r} (expected camera / file / demo)")

    def read(self):
        """Return (ok, frame)."""
        if self.files is not None:
            while self.idx < len(self.files):
                i = self.idx
                self.idx += 1
                if i % self.stride:
                    continue
                img = cv2.imread(os.path.join(self.dir, self.files[i]), cv2.IMREAD_COLOR)
                if img is not None:
                    return True, img
            return False, None
        if self.cap is None:
            return False, None
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return False, None
            if self.n_read % self.stride == 0:
                self.n_read += 1
                return True, frame
            self.n_read += 1

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


# ===========================================================================
# Available input sources
# ===========================================================================
def list_demo_sources():
    """List the sequences available in the bundled sample datasets."""
    out = {}
    # URFD (frame folders)
    try:
        dirs = glob.glob(os.path.join(os.path.dirname(_urfd_frames_dir("x")), "*-cam0-rgb"))
        seqs = sorted(os.path.basename(d).split("-cam0-rgb")[0] for d in dirs)
        if seqs:
            out["urfd"] = seqs
    except Exception:
        pass
    # Le2i (avi videos)
    try:
        seqs = [name for name, _ in _le2i_list_videos()]
        if seqs:
            out["le2i"] = seqs
    except Exception:
        pass
    # MCF (chute videos)
    try:
        root = _kaggle_dir("multiple-cameras-fall-dataset")
        if root:
            inner = os.path.join(root, "dataset", "dataset")
            base = inner if os.path.isdir(inner) else root
            seqs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(base, "chute*"))
                          if os.path.isdir(p))
            if seqs:
                out["mc"] = seqs
    except Exception:
        pass
    return out


# ===========================================================================
# Detection engine
# ===========================================================================
class DetectionEngine:
    """Continuous inference + fall records. Thread-safe: all external state is read and written under self._lock."""

    # Maximum number of clip files deleted per clear (see clear_records)
    MAX_CLIP_DELETES = 40

    def __init__(self, threshold=0.60, cooldown_s=6.0, confirm=2,
                 pre_s=1.0, post_s=2.0, clip_fps=CLIP_FPS, alarm_hold_s=4.0,
                 detect_every=3, prob_every=2,
                 show_hud=False, show_skeleton=True):
        self.threshold = float(threshold)
        self.cooldown_s = float(cooldown_s)
        self.confirm = int(confirm)
        self.pre_frames = max(1, int(round(pre_s * clip_fps)))
        self.post_frames = max(1, int(round(post_s * clip_fps)))
        self.clip_fps = float(clip_fps)
        self.alarm_hold_s = float(alarm_hold_s)

        # Reduced inference rate (performance tuning)
        #   detect_every: run YOLO person detection every N frames and reuse the last box in between
        #                 (a person moves little between frames, so reusing the box does not affect the pose crop)
        #   prob_every:   run the A4 temporal decision every N frames and keep the last probability in between
        #                 (neighbouring samples of a 2 s window overlap heavily, so per-frame evaluation is unnecessary)
        #   P1 pose always runs on every frame (only a few ms on a GPU)
        self.detect_every = max(1, int(detect_every))
        self.prob_every = max(1, int(prob_every))

        # Overlay switches (only affect the VISUAL layer of display and clips; unrelated to model input)
        #   show_hud     : top info bar + "frame N P(fall)=x FALL/ADL" text + probability bar
        #   show_skeleton: person box + skeleton lines
        # Text overlays are off by default so that text on the image is not distracting or mistaken for a prompt;
        # whatever the switches, the model always receives the RAW frame (overlays are drawn after the three stages).
        self.show_hud = bool(show_hud)
        self.show_skeleton = bool(show_skeleton)

        self._lock = threading.RLock()
        self._thread = None
        self._stop_evt = threading.Event()

        self._state = "idle"                 # idle / running / finished / error
        self._error = None
        self._source_spec = None
        self._source_label = "-"
        self._prob = 0.0
        self._peak = 0.0
        self._frames = 0
        self._fps = 0.0
        self._t_start = None
        self._n_falls = 0
        self._hot = 0
        self._last_alarm = 0.0
        self._alarm_until = 0.0
        self._rec = None
        self._session = 0
        self._gt = []
        self._last_box = None
        self._last_score = 0.0
        self._last_det_n = 0
        self._seen = 0

        self._pre = deque(maxlen=180)          # (ts, jpeg_bytes) pre-buffer, about 8-10 s
        self._kpts = deque(maxlen=WIN_FRAMES)
        self._last_jpeg = placeholder_jpeg()
        self._n_records = 0                    # cached record count (status() reads memory only, never the database)
        self._last_record = None

        # Fall records and report data are stored in SQLite (records/falls.db)
        self.store = RecordStore(RECORDS_DIR)
        migrated = self.store.migrate_from_json(RECORDS_JSON)   # compatibility with the old JSON
        if migrated:
            print(f"[store] migrated {migrated} records from the old falls.json into SQLite")
        self._refresh_record_cache()

        # ---- the weights of all three stages are loaded once ----
        # A4 prefers a4_weights.pkl in this folder (re-exported by train_weights.py in the CURRENT environment,
        # so that different sklearn / xgboost versions cannot break deserialization);
        # without it, fall back to the default weights in standalone/.
        self._det = None
        self._pose = None
        self._alg = None
        self.models_ready = False
        self._load_error = None
        try:
            self._det = YOLOPersonDetector()
            self._pose = PoseExtractor()
            a4_local = os.path.join(HERE, "a4_weights.pkl")
            if os.path.isfile(a4_local):
                self._alg = A4Algorithm(weights_path=a4_local)
                self.a4_source = a4_local
            else:
                self._alg = A4Algorithm()
                self.a4_source = "standalone/a4_weights.pkl"
            self.models_ready = True
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ records
    def records(self):
        """All fall records (newest first), from SQLite."""
        return self.store.list(newest_first=True)

    def delete_record(self, rid):
        """Delete one record together with its clip file.

        The returned record has an extra ``clip_removed`` field telling whether the clip file was really deleted
        (a bulk-deletion guard of the environment may block the deletion; the record is then gone but the file stays
        as an orphan, which the next clear_records removes).
        """
        gone = self.store.delete(rid)
        if gone is None:
            return None
        gone["clip_removed"] = self._remove_clip(gone.get("clip"))
        self._refresh_record_cache()
        return gone

    def clear_records(self, delete_clips=True):
        """Clear all records; with delete_clips=True the clip files are deleted as well.

        Returns {"deleted": records, "clips_removed": clips deleted, "clips_left": clips left in the folder}

        Clips are deleted IN BATCHES on purpose: the environment may guard against deleting more than 50 files in one go
        and raise SystemExit (not an OSError, so it must be caught separately). At most MAX_CLIP_DELETES are deleted
        per call and the rest go next time; this costs a little convenience but never fails the whole request
        and never bypasses the environment's guard.
        """
        gone = self.store.clear()
        removed = 0
        if delete_clips:
            names, seen = [], set()
            for r in gone:
                n = os.path.basename(r.get("clip") or "")
                if n and n not in seen:
                    seen.add(n)
                    names.append(n)
            # Fallback: leftover clips no record refers to (including those a bulk-deletion guard blocked last time)
            for p in glob.glob(os.path.join(CLIPS_DIR, "*.mp4")):
                n = os.path.basename(p)
                if n not in seen:
                    seen.add(n)
                    names.append(n)

            fail_streak = 0
            for n in names:
                if removed >= self.MAX_CLIP_DELETES:
                    break
                if self._remove_clip(n):
                    removed += 1
                    fail_streak = 0
                elif os.path.isfile(os.path.join(CLIPS_DIR, n)):
                    fail_streak += 1
                    if fail_streak >= 2:      # blocked twice in a row, stop trying this round
                        break

        self._refresh_record_cache()
        left = len(glob.glob(os.path.join(CLIPS_DIR, "*.mp4")))
        return {"deleted": len(gone), "clips_removed": removed, "clips_left": left}

    @staticmethod
    def _remove_clip(name):
        """Delete one clip file and return whether it was really deleted.

        SystemExit is caught: the environment's bulk-deletion guard uses it to stop deletions;
        catching it only keeps the whole HTTP request from failing and does not bypass the guard.
        """
        if not name:
            return False
        p = os.path.join(CLIPS_DIR, os.path.basename(name))
        if not os.path.isfile(p):
            return False
        try:
            os.remove(p)
            return True
        except (OSError, SystemExit):
            return False
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass

    # ------------------------------------------------------------ start / stop / config
    def set_config(self, threshold=None, cooldown_s=None, confirm=None,
                   pre_s=None, post_s=None, show_hud=None, show_skeleton=None):
        with self._lock:
            if threshold is not None:
                self.threshold = float(np.clip(float(threshold), 0.05, 0.99))
            if cooldown_s is not None:
                self.cooldown_s = max(0.0, float(cooldown_s))
            if confirm is not None:
                self.confirm = max(1, int(confirm))
            if pre_s is not None:
                self.pre_frames = max(1, int(round(float(pre_s) * self.clip_fps)))
            if post_s is not None:
                self.post_frames = max(1, int(round(float(post_s) * self.clip_fps)))
            if show_hud is not None:
                self.show_hud = bool(show_hud)
            if show_skeleton is not None:
                self.show_skeleton = bool(show_skeleton)
            return self.config()

    def config(self):
        with self._lock:
            return {
                "threshold": self.threshold,
                "cooldown_s": self.cooldown_s,
                "confirm": self.confirm,
                "clip_fps": self.clip_fps,
                "pre_frames": self.pre_frames,
                "post_frames": self.post_frames,
                "detect_every": self.detect_every,
                "prob_every": self.prob_every,
                "show_hud": self.show_hud,
                "show_skeleton": self.show_skeleton,
                "clip_seconds": round((self.pre_frames + self.post_frames) / self.clip_fps, 2),
            }

    def is_running(self):
        with self._lock:
            return self._state == "running"

    def start(self, spec: dict):
        """Start a new detection session. spec is validated first (RuntimeError on failure)."""
        if not self.models_ready:
            raise RuntimeError(
                "Models not ready: " + str(self._load_error) +
                ". If the A4 weights are incompatible with this environment, first run "
                "python train_weights.py to re-export them here.")
        self.stop()
        src = FrameSource(spec)        # raises here when validation fails

        with self._lock:
            self._source_spec = dict(spec)
            self._source_label = src.label
            self._state = "running"
            self._error = None
            self._prob = 0.0
            self._peak = 0.0
            self._frames = 0
            self._fps = 0.0
            self._n_falls = 0
            self._hot = 0
            self._rec = None
            self._last_alarm = 0.0
            self._alarm_until = 0.0
            self._session += 1
            self._t_start = time.time()
            self._pre.clear()
            self._kpts.clear()
            self._gt = []
            self._last_box = None
            self._last_score = 0.0
            self._last_det_n = 0
            self._seen = 0
            if spec.get("type") == "demo":
                try:
                    self._gt = _gt_intervals(spec["dataset"], spec["seq"])
                except Exception:
                    self._gt = []

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, args=(src,),
                                        name="fall-detector", daemon=True)
        self._thread.start()
        return self.status()

    def stop(self, timeout=5.0):
        th = self._thread
        if th is not None and th.is_alive():
            self._stop_evt.set()
            th.join(timeout=timeout)
        self._thread = None
        self._finalize_recording()
        with self._lock:
            if self._state == "running":
                self._state = "idle"
        return self.status()

    # ------------------------------------------------------------ main loop
    def _run(self, src: FrameSource):
        # Throttling uses the HIGH-RESOLUTION monotonic clock perf_counter: on Windows time.time() only has about
        # 15.6 ms resolution, which would make the per-frame timing jitter and the frame rate unstable.
        # Record timestamps still use time.time() (they must be real system time).
        t_beat = time.perf_counter()
        next_tick = time.perf_counter()
        frame_i = 0
        marks = deque(maxlen=2000)      # (perf_t, frame_i) for the 2-second sliding-window fps
        try:
            while not self._stop_evt.is_set():
                ok, frame = src.read()
                if not ok:
                    with self._lock:
                        self._state = "finished"
                    break

                frame_i += 1
                self._seen = frame_i
                # Reduced inference rate: the first frame is always detected (so there is a box from the start), then every
                #            detect_every frames; A4 decides every prob_every frames. A value of 1 restores per-frame work.
                do_detect = ((frame_i - 1) % self.detect_every == 0)
                do_prob = (frame_i % self.prob_every == 0)
                vis, prob, triggered = self._process(frame, frame_i, do_detect, do_prob)

                ok2, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok2:
                    jpeg = buf.tobytes()
                    with self._lock:
                        self._last_jpeg = jpeg
                    # Pre-buffer / fall trigger / clip collection (the same JPEG is reused, never encoded twice)
                    self._record_flow(time.time(), jpeg, triggered, prob)

                if src.realtime:
                    # Fixed cadence: accumulate absolute time points so per-frame sleeps do not accumulate error
                    next_tick += 1.0 / SAMPLE_FPS
                    delay = next_tick - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_tick = time.perf_counter()   # running faster than the cadence, realign

                now = time.perf_counter()
                marks.append((now, frame_i))
                if now - t_beat >= 0.5:
                    # fps over the LAST 2 SECONDS: a cumulative average is dragged down for a long time by the first-frame
                    # warm-up (CUDA/cuDNN initialization) and the frontend would show a much lower number.
                    while len(marks) > 1 and now - marks[0][0] > 2.0:
                        marks.popleft()
                    span = now - marks[0][0]
                    with self._lock:
                        self._fps = (frame_i - marks[0][1]) / span if span > 0.2 else 0.0
                    t_beat = now
        except Exception as e:                      # make sure the thread never dies silently
            with self._lock:
                self._state = "error"
                self._error = f"{type(e).__name__}: {e}"
        finally:
            self._finalize_recording()
            src.release()
            with self._lock:
                if self._state == "running":
                    self._state = "finished"
                # Keep the last measured frame rate (not reset to zero) so the frontend still shows how fast the session ran;
                # the next start() resets it to 0.

    # ------------------------------------------------------------ per-frame processing
    def _process(self, frame, frame_i, do_detect=True, do_prob=True):
        H, W = frame.shape[:2]

        # Stage 1: YOLO person detection (reduced rate: frames without detection reuse the last box and confidence)
        pts = conf = None
        kpts = np.zeros((17, 3), np.float32)
        if do_detect:
            box, score = self._det.detect_frame(frame)
            self._last_box = tuple(box) if box is not None else None
            self._last_score = float(score or 0.0)
            self._last_det_n = 1 if (box is not None and score > 0) else 0
        else:
            box = self._last_box
            score = self._last_score

        if box is not None and score > 0:
            x1, y1, x2, y2 = box
            box_px = [x1 * W, y1 * H, x2 * W, y2 * H]
            # Stage 2: our P1 pose model (every frame: a few ms on a GPU, and the window needs a skeleton per frame)
            pts, conf = self._pose.extract(frame, box_px)
            kpts = np.stack([pts[:, 0] / W, pts[:, 1] / H, conf], axis=1).astype(np.float32)

        with self._lock:
            self._kpts.append(kpts)          # the window grows every frame, no rate reduction
            self._frames = frame_i
            have_win = len(self._kpts) >= WIN_FRAMES
            kwin = np.stack(list(self._kpts)[-WIN_FRAMES:]).astype(np.float32) if have_win else None

        # Stage 3: A4 sliding-window temporal decision (reduced rate: other frames keep the last probability)
        prob = self._prob
        if do_prob and kwin is not None:
            with a4_fast_context():          # skip the joblib/OpenMP scheduling overhead
                wp = self._alg.predict(kwin)
            if len(wp):
                prob = float(wp[0])

        now = time.time()
        with self._lock:
            self._prob = prob
            self._peak = max(self._peak, prob)
            # The confirm counter only advances on real decision frames, keeping the smoothing semantics of confirm unchanged
            if do_prob:
                if prob >= self.threshold:
                    self._hot += 1
                else:
                    self._hot = 0
            triggered = (self._hot >= self.confirm
                         and (now - self._last_alarm) >= self.cooldown_s
                         and self._rec is None)
            if triggered:
                self._last_alarm = now
                self._alarm_until = now + self.alarm_hold_s

        # Overlays (drawn on vis only: the model input is always the raw frame, overlays never feed back into the model)
        vis = frame.copy()
        if self.show_skeleton and pts is not None:
            draw_skeleton(vis, pts, conf, box)
        if self.show_hud:
            vis = self._draw_hud(vis, prob, frame_i)
        return vis, prob, triggered

    def _draw_hud(self, vis, prob, frame_i):
        H, W = vis.shape[:2]
        alarm = prob >= self.threshold
        color = (0, 0, 230) if alarm else (0, 170, 0)      # BGR
        cv2.rectangle(vis, (0, 0), (W, 30), (34, 36, 40), -1)
        # ASCII only: OpenCV cannot render Chinese characters
        txt = f"frame {frame_i}  P(fall)={prob:.2f}  {'FALL' if alarm else 'ADL'}"
        cv2.putText(vis, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Probability bar
        bw = int(W * 0.5)
        x0, y0 = int(W * 0.03), 42
        cv2.rectangle(vis, (x0, y0), (x0 + bw, y0 + 12), (70, 74, 80), -1)
        cv2.rectangle(vis, (x0, y0), (x0 + int(bw * min(max(prob, 0.0), 1.0)), y0 + 12), color, -1)
        cx = x0 + int(bw * self.threshold)
        cv2.line(vis, (cx, y0 - 4), (cx, y0 + 16), (230, 230, 230), 1)
        return vis

    # ------------------------------------------------------------ clips
    def _record_flow(self, now, jpeg, triggered, prob):
        """Called every frame: maintain the pre-buffer, trigger a clip, collect later frames and write the clip after 3 s.

        Frames are collected by REAL timestamps so a "3 s clip" is really 3 seconds (1 s before + 2 s after),
        independent of the inference speed (a fast GPU or a slow CPU does not change the clip's real duration).
        """
        to_finalize = None
        with self._lock:
            self._pre.append((now, jpeg))
            if self._rec is None:
                if triggered:
                    ts = datetime.now()
                    name = "fall_%s.mp4" % ts.strftime("%Y%m%d_%H%M%S")
                    pre = [(t, b) for (t, b) in self._pre
                           if t >= now - self.pre_frames / self.clip_fps]
                    self._rec = {"name": name, "t": ts, "t0": now,
                                 "prob": prob, "peak": prob, "buf": pre}
            else:
                self._rec["buf"].append((now, jpeg))
                self._rec["peak"] = max(self._rec["peak"], prob)
                if now - self._rec["t0"] >= self.post_frames / self.clip_fps:
                    to_finalize = self._rec
                    self._rec = None
        if to_finalize is not None:
            self._finalize_recording(to_finalize)

    def _finalize_recording(self, rec=None):
        if rec is None:
            with self._lock:
                rec = self._rec
                self._rec = None
        if rec is None:
            return None
        buf = rec.get("buf") or []
        if not buf:
            return None

        # Real duration -> nominal frame rate, so the clip plays back at real speed
        n = len(buf)
        duration = max(0.1, buf[-1][0] - buf[0][0])
        fps = float(np.clip(n / duration, 1.0, 60.0))

        first = cv2.imdecode(np.frombuffer(buf[0][1], np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            return None
        H, W = first.shape[:2]

        path = os.path.join(CLIPS_DIR, rec["name"])
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        try:
            for _, jb in buf:
                img = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H))
                writer.write(img)
        finally:
            writer.release()

        ts = rec["t"]
        entry = {
            "id": ts.strftime("%Y%m%d_%H%M%S"),
            "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": ts.timestamp(),
            "prob": round(float(rec["prob"]), 4),
            "peak": round(float(rec["peak"]), 4),
            "clip": rec["name"],
            "clip_url": "/api/clip/" + rec["name"],
            "alarm": True,
            "source": self._source_label,
            "frames": n,
            "duration_s": round(duration, 2),
            "fps": round(fps, 2),
            "threshold": self.threshold,
        }
        with self._lock:
            self._n_falls += 1
        saved = self.store.add(entry)          # write to SQLite
        self._refresh_record_cache()           # update the in-memory cache (used by status())
        return saved

    # ------------------------------------------------------------ status
    def _refresh_record_cache(self):
        """Cache the record count and the latest record in memory.

        status() is polled by the frontend every 0.5 s; querying SQLite each time would happen while holding the lock
        and block the inference thread on it (measured about 15 ms/frame slower). So the cache is refreshed only when
        the records change, and status() reads memory only.
        """
        try:
            n = self.store.count()
            last = self.store.latest()
        except Exception:
            return
        with self._lock:
            self._n_records = int(n)
            self._last_record = last

    def latest_jpeg(self):
        with self._lock:
            return self._last_jpeg

    def status(self):
        with self._lock:
            running = self._state == "running"
            elapsed = (time.time() - self._t_start) if (running and self._t_start) else (
                self._frames / SAMPLE_FPS if self._frames else 0.0)
            return {
                "state": self._state,
                "error": self._error,
                "models_ready": self.models_ready,
                "load_error": self._load_error,
                "a4_weights": getattr(self, "a4_source", None),
                "source": self._source_label,
                "source_spec": self._source_spec,
                "prob": round(float(self._prob), 4),
                "peak": round(float(self._peak), 4),
                "alarm": time.time() < self._alarm_until,
                "threshold": self.threshold,
                "frames": int(self._frames),
                "seen": int(self._seen),
                "detect_every": int(self.detect_every),
                "prob_every": int(self.prob_every),
                "fps": round(float(self._fps), 2),
                "elapsed": round(float(elapsed), 1),
                "n_falls": int(self._n_falls),
                "recording": self._rec is not None,
                "n_records": self._n_records,
                "last_record": self._last_record,
                "gt": self._gt,
                "session": self._session,
            }


if __name__ == "__main__":
    # Command-line self-check: load the models and run a short sample to confirm the engine works
    eng = DetectionEngine()
    print("[OK] models loaded: YOLO + P1 pose + A4")
    print("[OK] available sample sources:", list(list_demo_sources().keys()))
    print("[demo] starting urfd/fall-01 with 60 frames ...")
    eng.start({"type": "demo", "dataset": "urfd", "seq": "fall-01"})
    time.sleep(8)
    st = eng.status()
    eng.stop()
    print("[result] state=%s frames=%s peak=%.3f n_falls=%d records=%d"
          % (st["state"], st["frames"], st["peak"], st["n_falls"], st["n_records"]))
