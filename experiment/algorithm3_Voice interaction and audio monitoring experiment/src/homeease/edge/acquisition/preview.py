"""Technical-page preview: the camera or replayed video frame with the pose skeleton drawn on it.

* Live mode: frames from the node's camera.
* Replay: the recording's source video, when it exists.
  * UR Fall: the RGB half of the cam0 video, the same pixels the detections were computed on.
  * Own captures: `video.mp4`, when consent allowed storing it.
  * Synthetic recordings have no video; their skeletons are drawn on a dark canvas.

Every detected person is drawn in grey. The monitored person is drawn on top, coloured by
posture: green upright, orange bending or descending, red lying. Frames are rendered in memory
for the node's own browser (loopback only) and are never written to disk.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from homeease.core.paths import project_root
from homeease.datasets.recording import DetectionFrame, Recording
from homeease.ml.pose.layout import L_HIP, L_SH, R_HIP, R_SH, SKELETON_EDGES

# BGR colours (OpenCV)
POSTURE_BGR = {"upright": (70, 170, 40), "bending": (0, 150, 255), "lying": (40, 40, 225), "unknown": (170, 170, 170)}
OTHER_BGR = (150, 150, 150)


def posture_of(kp: np.ndarray | None, width: float = 1.0, height: float = 1.0, min_score: float = 0.3) -> tuple[str, float | None]:
    """Coarse posture from the torso angle to the vertical (shoulder midpoint → hip midpoint, in
    pixels): upright < 30°, bending 30–60°, lying ≥ 60°. Display only; detection uses its own features."""
    if kp is None:
        return "unknown", None
    pts = []
    for a, b in ((L_SH, R_SH), (L_HIP, R_HIP)):
        cand = [kp[i, :2] for i in (a, b) if np.isfinite(kp[i, 0]) and kp[i, 2] >= min_score]
        if not cand:
            return "unknown", None
        pts.append(np.mean(cand, axis=0))
    dx, dy = (pts[1][0] - pts[0][0]) * width, (pts[1][1] - pts[0][1]) * height
    if abs(dx) + abs(dy) < 1e-6:
        return "unknown", None
    angle = float(np.degrees(np.arctan2(abs(dx), abs(dy))))  # 0° = vertical torso
    return ("upright" if angle < 30 else "bending" if angle < 60 else "lying"), round(angle, 1)


def resolve_video(rec: Recording) -> tuple[Path, tuple[float, float, float, float]] | None:
    """Source video of a replayed recording and the crop (normalised x0, y0, x1, y1) that the
    stored detections refer to."""
    own = rec.path / "video.mp4"
    if own.is_file():
        return own, (0.0, 0.0, 1.0, 1.0)
    m = rec.manifest
    if m.get("dataset") == "ur_fall":
        p = project_root() / "data" / "raw" / "ur_fall" / "data" / f"{m.get('session_id')}-cam0.mp4"
        if p.is_file():
            return p, (0.5, 0.0, 1.0, 1.0)  # depth on the left half, RGB on the right half
    return None


def draw_skeleton(img: np.ndarray, kp: np.ndarray, color: tuple[int, int, int], thickness: int, min_score: float) -> None:
    import cv2

    h, w = img.shape[:2]

    def pt(i: int) -> tuple[int, int] | None:
        k = kp[i]
        if not np.isfinite(k[0]) or k[2] < min_score:
            return None
        return int(k[0] * w), int(k[1] * h)

    for a, b in SKELETON_EDGES:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)
    for i in range(len(kp)):
        p = pt(i)
        if p:
            cv2.circle(img, p, max(2, thickness), color, -1, cv2.LINE_AA)
            cv2.circle(img, p, max(2, thickness), (255, 255, 255), 1, cv2.LINE_AA)


class FrameRenderer:
    """Reads (and caches) frames of one source video and draws skeletons on them."""

    def __init__(self, video: tuple[Path, tuple[float, float, float, float]] | None, out_width: int = 480) -> None:
        self.video = video
        self.out_width = out_width
        self.lock = threading.Lock()
        self._cap: Any = None
        self._pos = -1
        self._last: tuple[Any, bytes] | None = None

    def close(self) -> None:
        with self.lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def _frame(self, index: int) -> np.ndarray | None:
        import cv2

        if self.video is None:
            return None
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.video[0]))
            self._pos = -1
        if not (0 <= index - self._pos <= 30):  # backwards or far ahead: seek; otherwise read forward
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._pos = index - 1
        img = None
        while self._pos < index:
            ok, img = self._cap.read()
            if not ok:
                return None
            self._pos += 1
        if img is None:
            return None
        h, w = img.shape[:2]
        x0, y0, x1, y1 = self.video[1]
        return np.ascontiguousarray(img[int(y0 * h): int(y1 * h), int(x0 * w): int(x1 * w)])

    def render(self, det: DetectionFrame | None, target_kp: np.ndarray | None, posture: str, caption: str, min_score: float = 0.3) -> bytes | None:
        import cv2

        if det is None:
            return None
        key = (det.frame_index, det.t_ms, posture, caption)
        with self.lock:
            if self._last is not None and self._last[0] == key:
                return self._last[1]
            img = self._frame(det.frame_index)
            aspect = det.height / max(det.width, 1)
            out_h = int(self.out_width * aspect)
            img = np.full((out_h, self.out_width, 3), 28, np.uint8) if img is None else cv2.resize(img, (self.out_width, out_h))
            for p in det.persons:
                draw_skeleton(img, p, OTHER_BGR, 1, 0.2)
            if target_kp is not None:
                draw_skeleton(img, target_kp, POSTURE_BGR.get(posture, POSTURE_BGR["unknown"]), 3, min_score)
            cv2.rectangle(img, (0, out_h - 24), (self.out_width, out_h), (0, 0, 0), -1)
            cv2.putText(img, caption, (8, out_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return None
            self._last = (key, buf.tobytes())
            return self._last[1]
