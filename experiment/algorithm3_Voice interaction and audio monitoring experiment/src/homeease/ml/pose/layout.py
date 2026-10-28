"""Unified 2D keypoint layout (COCO-17) and conversions from backend layouts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from homeease.schemas.observations import COCO17, LAYOUT_VERSION

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
CORE = [L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK]

# MediaPipe BlazePose 33 -> COCO-17
MEDIAPIPE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
MEDIAPIPE_EXTRAS = {
    "left_heel": 29, "right_heel": 30, "left_foot_index": 31, "right_foot_index": 32,
    "left_index": 19, "right_index": 20,
}

SKELETON_EDGES = [
    (L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
    (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANK),
    (R_HIP, R_KNEE), (R_KNEE, R_ANK), (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
]


@dataclass
class PoseObs:
    """Internal pose observation (numpy) for one frame of the target."""

    t_ms: float
    kp: np.ndarray  # (17,3) normalised x,y,score; NaN when missing
    width: int
    height: int
    frame_index: int = 0
    track_id: int | None = None
    target_visible: bool = True
    target_conf: float = 1.0
    person_count: int = 1
    flags: list[str] = field(default_factory=list)
    backend: str = "unknown"
    device_id: str = "camera-01"
    session_id: str = "cam-session"
    source_mode: str = "live"
    inference_ms: float | None = None
    extras: dict[str, list[float]] | None = None
    bbox: list[float] | None = None

    @property
    def layout(self) -> str:
        return LAYOUT_VERSION


def empty_kp() -> np.ndarray:
    return np.full((17, 3), np.nan)


def kp_from_list(kps: list[list[float] | None]) -> np.ndarray:
    a = empty_kp()
    for i, k in enumerate(kps[:17]):
        if k is not None:
            a[i] = k
    return a


def kp_to_list(kp: np.ndarray, ndigits: int = 4) -> list[list[float] | None]:
    out: list[list[float] | None] = []
    for row in kp:
        if np.isnan(row).any():
            out.append(None)
        else:
            out.append([round(float(row[0]), ndigits), round(float(row[1]), ndigits), round(float(min(max(row[2], 0.0), 1.0)), 3)])
    return out


def bbox_from_kp(kp: np.ndarray, min_score: float = 0.3) -> list[float] | None:
    v = (~np.isnan(kp[:, 0])) & (kp[:, 2] >= min_score)
    if v.sum() < 3:
        return None
    xs, ys = kp[v, 0], kp[v, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


__all__ = ["COCO17", "LAYOUT_VERSION", "PoseObs", "CORE", "SKELETON_EDGES", "MEDIAPIPE_TO_COCO"]
