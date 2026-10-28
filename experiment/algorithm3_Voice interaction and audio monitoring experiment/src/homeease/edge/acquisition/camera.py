"""Local camera acquisition with adaptive processing rate.

A grabber thread keeps only the newest frame (no stale driver buffers); a
processing thread runs pose inference at the rate requested by the pipeline
(base rate while idle, higher during candidate events and exercise sessions).
Read failures, timeouts and reconnects are recorded and surface as coverage
reasons. Visual checks never depend on the phone being in motion.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

import numpy as np

from homeease.core.clock import Clock
from homeease.core.config import AppConfig
from homeease.core.util import get_logger
from homeease.datasets.recording import DetectionFrame
from homeease.edge.acquisition.replay import frame_to_obs, score_scale_for
from homeease.edge.tracking import TargetTracker
from homeease.ml.pose.backends import PoseBackend, PoseBackendError, create_backend

log = get_logger(__name__)


class CameraWorker:
    def __init__(
        self,
        cfg: AppConfig,
        clock: Clock,
        pipeline: Any,
        subject_id: str,
        device_id: str = "camera-01",
        backend: PoseBackend | None = None,
        on_frame: Callable[[np.ndarray, Any, Any], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.clock = clock
        self.pipeline = pipeline
        self.subject_id = subject_id
        self.device_id = device_id
        self.session_id = f"{device_id}-{int(time.time())}"
        self.backend = backend
        self.on_frame = on_frame
        self.tracker = TargetTracker(cfg.acquisition.zone, min_score=cfg.acquisition.pose.min_keypoint_score)
        self.stop_evt = threading.Event()
        self._latest: tuple[float, np.ndarray] | None = None
        self._latest_lock = threading.Lock()
        self.cap: Any = None
        self.status: dict[str, Any] = {"state": "starting", "error": None, "reconnects": 0, "frames_read": 0, "frames_processed": 0, "read_failures": 0, "fps": 0.0, "inference_ms": None}
        self._proc_times: deque[float] = deque(maxlen=30)
        self._inf_ms: deque[float] = deque(maxlen=30)
        self.preview_jpeg: bytes | None = None
        self.recorder: Any = None  # ResearchRecorder while an explicit research recording runs
        self._threads: list[threading.Thread] = []
        self.is_file = False
        self.file_fps = 30.0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.backend is None:
            try:
                self.backend = create_backend(self.cfg.acquisition.pose)
                self.pipeline.set_camera_status(model_status="ok", backend=self.backend.name)
            except (PoseBackendError, Exception) as e:  # model unavailable must be visible, not silent
                log.error("pose backend unavailable: %s", e)
                self.pipeline.set_camera_status(model_status="unavailable", backend=None)
                self.status["error"] = f"visual model unavailable: {e}"
        for fn, name in ((self._grab_loop, "camera-grab"), (self._process_loop, "camera-proc")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self.stop_evt.set()
        for t in self._threads:
            t.join(timeout=3)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:  # pragma: no cover
                pass
        if self.backend is not None:
            self.backend.close()

    # --------------------------------------------------------------- grabber
    def _open(self) -> bool:
        import cv2

        src = self.cfg.acquisition.camera.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        try:
            if isinstance(src, int):
                cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION) if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.VideoCapture(src)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.acquisition.camera.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.acquisition.camera.height)
                self.is_file = False
            else:
                cap = cv2.VideoCapture(str(src))
                self.is_file = not str(src).lower().startswith(("rtsp://", "http://", "https://"))
                if self.is_file:
                    self.file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if not cap.isOpened():
                cap.release()
                return False
            self.cap = cap
            return True
        except Exception as e:  # pragma: no cover
            self.status["error"] = str(e)
            return False

    def _grab_loop(self) -> None:
        backoff = list(self.cfg.acquisition.camera.reconnect_backoff_ms)
        attempt = 0
        while not self.stop_evt.is_set():
            if self.cap is None:
                if not self._open():
                    self.status.update(state="offline", error="camera could not be opened")
                    self.pipeline.set_camera_status(camera_error="camera_open_failed")
                    wait = backoff[min(attempt, len(backoff) - 1)] / 1000.0
                    attempt += 1
                    self.status["reconnects"] += 1
                    self.stop_evt.wait(wait)
                    continue
                attempt = 0
                self.status.update(state="online", error=None)
                self.pipeline.set_camera_status(camera_error=None)
            t0 = time.perf_counter()
            ok, frame = self.cap.read()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if not ok or frame is None:
                self.status["read_failures"] += 1
                if self.is_file:
                    self.status.update(state="ended", error="video file ended")
                    self.pipeline.set_camera_status(camera_error="video_file_ended")
                    self.stop_evt.wait(1.0)
                    continue
                self.status.update(state="offline", error="frame read failed")
                self.pipeline.set_camera_status(camera_error="camera_read_failed")
                try:
                    self.cap.release()
                finally:
                    self.cap = None
                continue
            if dt_ms > self.cfg.acquisition.camera.read_timeout_ms:
                self.status["slow_reads"] = self.status.get("slow_reads", 0) + 1
            self.status["frames_read"] += 1
            with self._latest_lock:
                self._latest = (self.clock.mono_ms(), frame)
            if self.is_file:
                time.sleep(max(0.0, 1.0 / self.file_fps - dt_ms / 1000.0))

    # ------------------------------------------------------------ processing
    def _process_loop(self) -> None:
        last_proc = 0.0
        last_frame_t = -1.0
        while not self.stop_evt.is_set():
            now = self.clock.mono_ms()
            fps = float(self.pipeline.desired_fps(now))
            period = 1000.0 / max(fps, 0.5)
            if now - last_proc < period:
                self.stop_evt.wait(min(0.02, (period - (now - last_proc)) / 1000.0))
                continue
            with self._latest_lock:
                latest = self._latest
            if latest is None or latest[0] == last_frame_t:
                self.stop_evt.wait(0.01)
                continue
            t_cap, frame = latest
            last_frame_t = t_cap
            last_proc = now
            h, w = frame.shape[:2]
            if self.backend is None:
                self.pipeline.set_camera_status(model_status="unavailable")
                continue
            try:
                persons, inf_ms = self.backend.infer(frame, t_cap)
                model_ok = True
            except Exception as e:  # pragma: no cover - backend failure must be visible
                log.exception("pose inference failed")
                persons, inf_ms, model_ok = [], 0.0, False
                self.status["error"] = f"inference error: {e}"
            self._inf_ms.append(inf_ms)
            self._proc_times.append(now)
            if len(self._proc_times) > 2:
                span = (self._proc_times[-1] - self._proc_times[0]) / 1000.0
                self.status["fps"] = round((len(self._proc_times) - 1) / span, 2) if span > 0 else 0.0
            self.status["inference_ms"] = round(float(np.mean(self._inf_ms)), 1)
            self.status["frames_processed"] += 1
            backlog = inf_ms > self.cfg.quality.max_model_latency_ms
            self.pipeline.set_camera_status(
                model_status="ok" if model_ok else "error", fps=self.status["fps"], backlog=backlog, camera_error=None,
                backend=self.backend.name, inference_ms=self.status["inference_ms"],
            )
            df = DetectionFrame(t_cap, self.status["frames_processed"], w, h, persons, self.backend.name, inf_ms)
            rec = self.recorder
            if rec is not None:
                try:
                    rec.on_frame(df, frame)
                except Exception:  # pragma: no cover - recording must never stop monitoring
                    log.exception("research recorder failed on a frame")
            obs = frame_to_obs(df, self.tracker, t_cap, subject_id=self.subject_id, device_id=self.device_id, session_id=self.session_id, source_mode="live", score_scale=score_scale_for(self.cfg, self.backend.name))
            self.pipeline.ingest_pose(obs)
            if self.on_frame is not None:
                try:
                    self.on_frame(frame, obs, df)
                except Exception:  # pragma: no cover
                    log.exception("on_frame hook failed")
