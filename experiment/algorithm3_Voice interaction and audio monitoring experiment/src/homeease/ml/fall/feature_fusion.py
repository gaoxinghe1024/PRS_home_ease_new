"""Feature-level fusion engine (research path, slide 13 "B").

A trained temporal model reads the aligned multimodal window (accelerometer
summaries + pose channels, each with an explicit mask) and outputs a fall
probability. The product mapping is the SAME policy as late fusion: EMERGENCY
requires p >= strong AND both modalities valid in the window; otherwise a
candidate yields PENDING. Missing modalities are masked, never imputed as normal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from homeease.core.config import AppConfig
from homeease.core.util import new_id
from homeease.ml.fall.models import apply_platt, load_model, predict_any
from homeease.ml.fall.windows import fused_seq
from homeease.schemas.evidence import DecisionLevel, FusionDecision


class FeatureFusionEngine:
    def __init__(self, cfg: AppConfig, accel: Any, vision: Any, subject_id: str, room_id: str | None, source_mode: str, package: str | Path) -> None:
        self.cfg = cfg
        self.accel, self.vision = accel, vision
        self.subject_id, self.room_id, self.source_mode = subject_id, room_id, source_mode
        self.model, self.meta = load_model(Path(package))
        if self.meta["task"] != "fall_feature_fusion":
            raise ValueError("package is not a feature-fusion model")
        self.backend = "feature_fusion"
        self.cal = self.meta.get("calibration")
        th = self.meta["thresholds"]
        self.th_c, self.th_s = float(th["candidate"]), float(th["strong"])
        self.window_ms = float(self.meta["input"]["window_ms"])
        self.step_ms = float(self.meta["input"].get("step_ms", 200.0))
        self.T = int(round(self.window_ms / 1000.0 * 15))
        self.next_eval: float | None = None
        self._episode: dict[str, Any] | None = None
        self._emitted: list[float] = []
        self.settle_ms = 400.0
        self.cases: list[Any] = []
        self.history: list[dict[str, Any]] = []

    def on_evidence(self, ev: Any, now: float) -> None:  # channel rule evidence is informational here
        return None

    def _latest(self) -> float | None:
        ts = [t for t in (self.accel.latest_t, self.vision.latest_t) if t is not None]
        return max(ts) if ts else None

    def _valid(self, aw: Any, vw: Any) -> tuple[bool, bool]:
        q = self.cfg.quality
        a_ok = aw is not None and len(aw.t) > 0 and float(aw.mask.mean()) >= 0.8 and aw.clock_uncertainty_ms is not None and aw.clock_uncertainty_ms <= q.max_clock_uncertainty_ms
        v_ok = False
        if vw is not None and len(vw.mask):
            m = vw.mask.astype(bool)
            tc = float(np.nanmean(vw.col("target_conf")[m])) if m.any() else 0.0
            amb = float(np.nan_to_num(vw.col("ambiguous"), nan=0.0).mean())
            v_ok = float(m.mean()) >= 0.6 and tc >= 0.6 and amb <= 0.2
        return a_ok, v_ok

    def tick(self, now: float, final: bool = False) -> list[FusionDecision]:
        out: list[FusionDecision] = []
        latest = self._latest()
        if latest is None:
            return out
        if self.next_eval is None:
            self.next_eval = latest
        while self.next_eval <= latest + 1e-6:
            t = self.next_eval
            self.next_eval += self.step_ms
            aw = self.accel.window(t, self.window_ms) if self.accel.latest_t is not None and self.accel.latest_t >= t - self.window_ms else None
            vw = self.vision.window(t, self.window_ms) if self.vision.latest_t is not None and self.vision.latest_t >= t - self.window_ms else None
            x = fused_seq(aw, vw, self.T)[None]
            p = float(apply_platt(predict_any(self.model, self.meta["kind"], x), self.cal)[0])
            a_ok, v_ok = self._valid(aw, vw)
            if p >= self.th_c:
                anchor = self._anchor(aw, vw, t)
                if any(abs(anchor - e) < 2500.0 for e in self._emitted):
                    continue
                if self._episode is None:
                    self._episode = {"first": t, "p": p, "anchor": anchor, "a_ok": a_ok, "v_ok": v_ok}
                elif p > self._episode["p"]:
                    self._episode.update(p=p, anchor=anchor, a_ok=a_ok, v_ok=v_ok)
            if self._episode is not None and (t - self._episode["first"] >= self.settle_ms or final):
                out.append(self._decide(self._episode, now))
                self._emitted.append(self._episode["anchor"])
                self._episode = None
        if final and self._episode is not None:
            out.append(self._decide(self._episode, now))
            self._episode = None
        return out

    def _anchor(self, aw: Any, vw: Any, t: float) -> float:
        if aw is not None and len(aw.t):
            sv = np.sqrt(np.nansum(np.nan_to_num(aw.axes) ** 2, axis=1))
            return float(aw.t[int(np.argmax(sv))])
        return t - 1000.0

    def _decide(self, ep: dict[str, Any], now: float) -> FusionDecision:
        a = self.accel.assess(ep["anchor"], now)
        v = self.vision.assess(ep["anchor"], now)
        level = DecisionLevel.EMERGENCY if (ep["p"] >= self.th_s and ep["a_ok"] and ep["v_ok"]) else DecisionLevel.PENDING
        reasons = ["feature_fusion_strong" if level == DecisionLevel.EMERGENCY else "feature_fusion_candidate"]
        if not ep["a_ok"]:
            reasons.append("accel_missing")
        if not ep["v_ok"]:
            reasons.append("vision_missing")
        d = FusionDecision(
            decision_id=f"{new_id('ff_')}:0", subject_id=self.subject_id, level=level, rejected=False, decided_at_ms=now,
            anchor_time_ms=ep["anchor"], onset_estimate_ms=ep["anchor"] - 1000.0, fusion_backend="feature_fusion",
            fused_score=round(ep["p"], 4), gate={"p": ep["p"], "accel_valid": ep["a_ok"], "vision_valid": ep["v_ok"], "strong_threshold": self.th_s},
            reason_codes=reasons, evidences=[a, v], source_mode=self.source_mode, room_id=self.room_id,
        )
        self.history.append({"t": now, "level": level.value, "p": ep["p"]})
        return d

    def snapshot(self) -> list[dict[str, Any]]:
        return [] if self._episode is None else [{"case_id": "feature_fusion_episode", "anchor_ms": self._episode["anchor"], "deadline_ms": None, "channels": {"fused": {"candidate": True, "score": self._episode["p"]}}, "decided": None}]


def build_feature_fusion_engine(cfg: AppConfig, accel: Any, vision: Any, subject_id: str, room_id: str | None, source_mode: str, package: str | None) -> FeatureFusionEngine:
    if package is None:
        raise ValueError("feature fusion package path required")
    return FeatureFusionEngine(cfg, accel, vision, subject_id, room_id, source_mode, package)
