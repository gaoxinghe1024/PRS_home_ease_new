"""Pose estimation backends producing COCO-17 normalised keypoints.

* MediaPipe Pose Landmarker (BlazePose GHUM, single-stage detector + landmark
  model, 33 points mapped to COCO-17; Apache-2.0). On macOS with mediapipe 1.x
  the CPU path of the detector requires a GPU service, so the GPU (Metal)
  delegate with SRGBA input is the default.
* RTMPose (top-down: YOLOX person detector + RTMPose SimCC head; Apache-2.0)
  through rtmlib + onnxruntime.

Downstream code never depends on which backend produced the keypoints.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from homeease.core.config import PoseCfg
from homeease.core.paths import resolve
from homeease.core.util import sha256_file
from homeease.ml.pose.layout import MEDIAPIPE_EXTRAS, MEDIAPIPE_TO_COCO


class PoseBackendError(RuntimeError):
    pass


class PoseBackend(Protocol):
    name: str
    version: str

    def infer(self, bgr: np.ndarray, t_ms: float) -> tuple[list[np.ndarray], float]: ...

    def close(self) -> None: ...

    def info(self) -> dict[str, Any]: ...


class MediaPipeBackend:
    def __init__(self, model_path: str | Path, delegate: str = "gpu", num_poses: int = 3) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as bo

        p = resolve(str(model_path))
        if p is None or not p.is_file():
            raise PoseBackendError(f"MediaPipe model not found: {model_path} (run `homeease models fetch`)")
        self._mp = mp
        self.model_path = p
        self.delegate = delegate
        d = bo.BaseOptions.Delegate.GPU if delegate == "gpu" else bo.BaseOptions.Delegate.CPU
        opts = vision.PoseLandmarkerOptions(
            base_options=bo.BaseOptions(model_asset_path=str(p), delegate=d),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lm = vision.PoseLandmarker.create_from_options(opts)
        self.name = f"mediapipe_{p.stem.replace('pose_landmarker_', '')}"
        self.version = mp.__version__
        self._last_ts = -1
        self._sha = None

    def infer(self, bgr: np.ndarray, t_ms: float) -> tuple[list[np.ndarray], float]:
        import cv2

        t0 = time.perf_counter()
        ts = max(int(t_ms), self._last_ts + 1)
        self._last_ts = ts
        if self.delegate == "gpu":
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGBA, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA))
        else:
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(img, ts)
        persons = []
        for lms in res.pose_landmarks or []:
            kp = np.full((17, 3), np.nan)
            for ci, mi in enumerate(MEDIAPIPE_TO_COCO):
                l = lms[mi]
                vis = getattr(l, "visibility", None)
                score = float(vis) if vis is not None else 1.0
                kp[ci] = [float(l.x), float(l.y), min(max(score, 0.0), 1.0)]
            persons.append(kp)
        return persons, (time.perf_counter() - t0) * 1000.0

    def close(self) -> None:
        try:
            self.lm.close()
        except Exception:  # pragma: no cover
            pass

    def info(self) -> dict[str, Any]:
        if self._sha is None:
            self._sha = sha256_file(self.model_path)
        return {"backend": self.name, "framework": f"mediapipe {self.version}", "model": str(self.model_path.name), "sha256": self._sha, "delegate": self.delegate, "license": "Apache-2.0", "extras": list(MEDIAPIPE_EXTRAS)}


class RTMPoseBackend:
    def __init__(self, det_path: str | Path, pose_path: str | Path, device: str = "cpu") -> None:
        try:
            from rtmlib import Body
        except ImportError as e:  # pragma: no cover
            raise PoseBackendError("rtmlib/onnxruntime not installed: uv sync --extra pose-rtm") from e
        det, pose = resolve(str(det_path)), resolve(str(pose_path))
        for p in (det, pose):
            if p is None or not p.is_file():
                raise PoseBackendError(f"RTMPose model file missing: {p} (run `homeease models fetch --rtmpose`)")
        self.det_path, self.pose_path = det, pose
        self.body = Body(det=str(det), det_input_size=(416, 416), pose=str(pose), pose_input_size=(192, 256), backend="onnxruntime", device=device)
        self.name = "rtmpose_s"
        self.version = "rtmlib"
        self._sha: dict[str, str] | None = None

    def infer(self, bgr: np.ndarray, t_ms: float) -> tuple[list[np.ndarray], float]:
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        kps, scores = self.body(bgr)
        persons = []
        for k, sc in zip(kps, scores):
            kp = np.full((17, 3), np.nan)
            kp[:, 0] = k[:17, 0] / w
            kp[:, 1] = k[:17, 1] / h
            kp[:, 2] = np.clip(sc[:17], 0.0, 1.0)
            persons.append(kp)
        return persons, (time.perf_counter() - t0) * 1000.0

    def close(self) -> None:
        pass

    def info(self) -> dict[str, Any]:
        if self._sha is None:
            self._sha = {"det": sha256_file(self.det_path), "pose": sha256_file(self.pose_path)}
        return {"backend": self.name, "framework": "rtmlib+onnxruntime", "det_model": self.det_path.name, "pose_model": self.pose_path.name, "sha256": self._sha, "license": "Apache-2.0"}


def create_backend(cfg: PoseCfg, name: str | None = None) -> PoseBackend:
    name = name or cfg.backend
    if name == "mediapipe":
        return MediaPipeBackend(cfg.mediapipe_model, cfg.mediapipe_delegate, cfg.max_people)
    if name.startswith("mediapipe_"):
        variant = name.split("_", 1)[1]
        return MediaPipeBackend(f"models/pose/mediapipe/pose_landmarker_{variant}.task", cfg.mediapipe_delegate, cfg.max_people)
    if name in ("rtmpose", "rtmpose_s"):
        return RTMPoseBackend(cfg.rtmpose_det, cfg.rtmpose_pose)
    raise PoseBackendError(f"unknown pose backend {name}")
