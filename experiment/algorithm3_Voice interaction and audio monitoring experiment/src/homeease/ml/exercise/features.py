"""Sit-to-stand pose features (reusing the shared PoseObs stream).

``thigh_vertical`` = (knee_y - hip_y) / torso_length works for side and front
views: the thigh is near horizontal (small value) when seated and vertical
(large value) when standing. Torso length is a session-level upright reference.
"""

from __future__ import annotations

import numpy as np

from homeease.ml.pose.layout import L_ANK, L_HIP, L_KNEE, L_SH, R_ANK, R_HIP, R_KNEE, R_SH, PoseObs

PHASES = ["SITTING", "RISING", "STANDING", "DESCENDING"]
UNKNOWN = "UNKNOWN"
PHASE_IDX = {p: i for i, p in enumerate(PHASES)}

EX_FEATURES = ["thigh_vertical", "hip_rel", "knee_angle", "torso_deg", "hip_vel_tl_s", "hip_y_tl", "knee_vel_tl_s"]


def _pt(kp: np.ndarray, i: int, ms: float) -> np.ndarray | None:
    if np.isnan(kp[i, 0]) or kp[i, 2] < ms:
        return None
    return kp[i, :2]


def _avg(kp: np.ndarray, idx: tuple[int, int], ms: float, w: int, h: int) -> np.ndarray | None:
    pts = [p for p in (_pt(kp, i, ms) for i in idx) if p is not None]
    if not pts:
        return None
    a = np.mean(pts, axis=0)
    return np.array([a[0] * w, a[1] * h])


def _knee_angle(kp: np.ndarray, ms: float, w: int, h: int) -> float:
    angs = []
    for hi, ki, ai in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        p = [_pt(kp, i, ms) for i in (hi, ki, ai)]
        if any(x is None for x in p):
            continue
        a, b, c = (np.array([x[0] * w, x[1] * h]) for x in p)  # type: ignore[index]
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            angs.append(np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1, 1))))
    return float(np.mean(angs)) if angs else float("nan")


class ExerciseFeatureExtractor:
    """Causal per-frame features with a session-level scale and sit/stand references."""

    def __init__(self, min_score: float = 0.3) -> None:
        self.ms = min_score
        self.tl_ref: float | None = None
        self._tl_hist: list[float] = []
        self.sit_hip_y: float | None = None
        self.stand_hip_y: float | None = None
        self._prev: tuple[float, float, float] | None = None  # t, hip_y, knee_y
        self._vel_hist: list[float] = []
        self._kvel_hist: list[float] = []

    def reset_motion(self) -> None:
        self._prev = None
        self._vel_hist.clear()
        self._kvel_hist.clear()

    def __call__(self, obs: PoseObs) -> dict[str, float] | None:
        if not obs.target_visible:
            self.reset_motion()
            return None
        kp, w, h = obs.kp, obs.width, obs.height
        hip = _avg(kp, (L_HIP, R_HIP), self.ms, w, h)
        knee = _avg(kp, (L_KNEE, R_KNEE), self.ms, w, h)
        sh = _avg(kp, (L_SH, R_SH), self.ms, w, h)
        if hip is None or knee is None or sh is None:
            self.reset_motion()
            return None
        tl = float(np.hypot(*(sh - hip)))
        torso = float(np.degrees(np.arctan2(abs(sh[0] - hip[0]), -(sh[1] - hip[1]))))
        if torso < 35 and tl > 5:
            self._tl_hist.append(tl)
            self._tl_hist = self._tl_hist[-90:]
            self.tl_ref = float(np.median(self._tl_hist))
        ref = self.tl_ref or tl
        if ref < 5:
            return None
        tv = float((knee[1] - hip[1]) / ref)
        vel = 0.0
        kvel = 0.0
        if self._prev is not None and obs.t_ms > self._prev[0]:
            dt = (obs.t_ms - self._prev[0]) / 1000.0
            if dt < 0.5:
                vel = (hip[1] - self._prev[1]) / ref / dt
                kvel = (knee[1] - self._prev[2]) / ref / dt
        self._prev = (obs.t_ms, float(hip[1]), float(knee[1]))
        self._vel_hist.append(vel)
        self._vel_hist = self._vel_hist[-3:]
        self._kvel_hist.append(kvel)
        self._kvel_hist = self._kvel_hist[-3:]
        hip_rel = float("nan")
        if self.sit_hip_y is not None and self.stand_hip_y is not None and abs(self.sit_hip_y - self.stand_hip_y) > 1:
            hip_rel = float((self.sit_hip_y - hip[1]) / (self.sit_hip_y - self.stand_hip_y))
        return {
            "t": obs.t_ms,
            "thigh_vertical": tv,
            "hip_rel": hip_rel,
            "knee_angle": _knee_angle(kp, self.ms, w, h),
            "torso_deg": torso,
            "hip_vel_tl_s": float(np.mean(self._vel_hist)),
            "hip_y_tl": float(hip[1] / ref),
            "knee_vel_tl_s": float(np.mean(self._kvel_hist)),
            "hip_y_px": float(hip[1]),
        }

    def note_phase(self, phase: str, feats: dict[str, float]) -> None:
        """Update sit/stand references from confirmed phases (never from labels)."""
        y = feats.get("hip_y_px")
        if y is None:
            return
        if phase == "SITTING":
            self.sit_hip_y = y if self.sit_hip_y is None else 0.9 * self.sit_hip_y + 0.1 * y
        elif phase == "STANDING":
            self.stand_hip_y = y if self.stand_hip_y is None else 0.9 * self.stand_hip_y + 0.1 * y


def feature_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([float(feats.get(k, np.nan)) for k in EX_FEATURES], dtype=np.float64)
