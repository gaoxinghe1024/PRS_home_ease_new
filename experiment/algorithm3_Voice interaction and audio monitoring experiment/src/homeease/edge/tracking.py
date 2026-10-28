"""Target association for the single monitored older adult.

Visitors may appear in the frame. A pose is only attributed to the subject
when the association is unambiguous; another person standing normally must
never count as evidence that the subject is safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from homeease.ml.pose.layout import bbox_from_kp


@dataclass
class Detection:
    kp: np.ndarray  # (17,3) normalised
    bbox: list[float]
    score: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


@dataclass
class Track:
    track_id: int
    bbox: list[float]
    center: tuple[float, float]
    last_t: float
    hits: int = 1


@dataclass
class TargetResult:
    detection: Detection | None
    track_id: int | None
    person_count: int
    confidence: float
    flags: list[str] = field(default_factory=list)


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class TargetTracker:
    def __init__(self, zone: list[float] | None = None, max_missing_ms: float = 1500.0, min_score: float = 0.3) -> None:
        self.zone = zone or [0.0, 0.0, 1.0, 1.0]
        self.max_missing_ms = max_missing_ms
        self.min_score = min_score
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.target_id: int | None = None
        self.target_last_center: tuple[float, float] | None = None
        self.target_last_t: float | None = None
        self.pinned: bool = False

    def set_target(self, track_id: int | None) -> None:
        self.target_id = track_id
        self.pinned = track_id is not None

    def _in_zone(self, c: tuple[float, float]) -> bool:
        z = self.zone
        return z[0] <= c[0] <= z[2] and z[1] <= c[1] <= z[3]

    def detections_from_kps(self, kps: list[np.ndarray]) -> list[Detection]:
        dets = []
        for kp in kps:
            bb = bbox_from_kp(kp, self.min_score)
            if bb is None:
                continue
            v = (~np.isnan(kp[:, 0])) & (kp[:, 2] >= self.min_score)
            dets.append(Detection(kp, bb, float(np.mean(kp[v, 2])) if v.any() else 0.0))
        return dets

    def update(self, dets: list[Detection], t: float) -> TargetResult:
        # expire tracks
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_t > self.max_missing_ms]:
            self.tracks.pop(tid)
        dets = [d for d in dets if self._in_zone(d.center)]
        assigned: dict[int, Detection] = {}
        used: set[int] = set()
        pairs = []
        for tid, tr in self.tracks.items():
            for j, d in enumerate(dets):
                ov = iou(tr.bbox, d.bbox)
                dist = float(np.hypot(tr.center[0] - d.center[0], tr.center[1] - d.center[1]))
                if ov >= 0.15 or dist < 0.2:
                    pairs.append((ov - dist, tid, j))
        for _, tid, j in sorted(pairs, reverse=True):
            if tid in assigned or j in used:
                continue
            assigned[tid] = dets[j]
            used.add(j)
        for tid, d in assigned.items():
            tr = self.tracks[tid]
            tr.bbox, tr.center, tr.last_t, tr.hits = d.bbox, d.center, t, tr.hits + 1
        for j, d in enumerate(dets):
            if j not in used:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, d.bbox, d.center, t)
                assigned[tid] = d
        person_count = len(dets)
        flags: list[str] = []
        if person_count == 0:
            return TargetResult(None, None, 0, 0.0, ["person_not_visible"])
        if self.target_id is not None and self.target_id in assigned:
            d = assigned[self.target_id]
            tr = self.tracks[self.target_id]
            conf = min(1.0, 0.5 + tr.hits / 20.0)
            others = [o for k, o in assigned.items() if k != self.target_id]
            if any(iou(d.bbox, o.bbox) > 0.4 for o in others):
                flags.append("target_ambiguous")
                conf *= 0.4
            elif others:
                conf *= 0.9
            self.target_last_center, self.target_last_t = d.center, t
            return TargetResult(d, self.target_id, person_count, conf, flags)
        # target lost or not yet chosen
        if self.pinned and self.target_id is not None and self.target_id in self.tracks:
            return TargetResult(None, None, person_count, 0.0, ["person_not_visible"])
        if len(assigned) == 1:
            tid, d = next(iter(assigned.items()))
            switched = (
                self.target_id is not None
                and self.target_last_t is not None
                and t - self.target_last_t < 3000.0
                and self.target_last_center is not None
                and float(np.hypot(d.center[0] - self.target_last_center[0], d.center[1] - self.target_last_center[1])) > 0.3
            )
            self.target_id = tid
            self.target_last_center, self.target_last_t = d.center, t
            conf = 0.5 if switched else min(1.0, 0.5 + self.tracks[tid].hits / 20.0)
            if switched:
                flags.append("target_switched")
            return TargetResult(d, tid, person_count, conf, flags)
        # several people and no continuous target: do not guess
        return TargetResult(None, None, person_count, 0.0, ["target_ambiguous"])
