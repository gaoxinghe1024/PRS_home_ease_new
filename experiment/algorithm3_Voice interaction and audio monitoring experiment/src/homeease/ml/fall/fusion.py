"""Decision-level (late) fusion with an explicit dual-channel emergency gate.

Backends
--------
late_rule_basic     Baseline: EMERGENCY if both channels flag a candidate, PENDING if one does.
                    No quality, time, target or contradiction checks (kept to measure their value).
late_quality_gated  Default product path. Quality-weighted fused score for reporting plus a hard
                    gate: EMERGENCY only when both channels are valid, strong, time-consistent,
                    target-consistent and clock-trusted. A missing channel is never treated as
                    evidence of normality.
late_trained        Logistic stacker over per-channel outputs (trained on validation data); the
                    same dual-channel gate still applies to EMERGENCY.

Product mapping (proposal Table 6-1 / prompt 7.4):
  both strong + consistent              -> EMERGENCY
  one candidate, other missing/untrusted/
  weak/unable to clearly refute         -> PENDING
  candidate clearly refuted by a valid,
  target-consistent vision channel      -> NORMAL (rejected candidate, technical record only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from homeease.core.config import AppConfig
from homeease.core.util import new_id
from homeease.schemas.evidence import DecisionLevel, Evidence, FusionDecision, LEVEL_RANK


@dataclass
class FusionCase:
    case_id: str
    anchor_ms: float
    created_ms: float
    deadline_ms: float
    ev: dict[str, Evidence] = field(default_factory=dict)
    decided: DecisionLevel | None = None
    decisions: int = 0
    open_until_ms: float = 0.0
    last_decision: FusionDecision | None = None


def _valid(e: Evidence | None, min_q: float, max_unc: float) -> bool:
    if e is None or not e.available:
        return False
    if e.quality_score < min_q:
        return False
    if e.clock_uncertainty_ms is not None and e.clock_uncertainty_ms > max_unc:
        return False
    if e.channel == "accel" and e.clock_uncertainty_ms is None:
        return False
    return True


def _target_ok(v: Evidence | None) -> bool:
    if v is None:
        return False
    if "target_ambiguous" in v.quality_flags or "target_switched" in v.quality_flags:
        return False
    return (v.target_confidence or 0.0) >= 0.6


class TrainedFusionModel:
    """Logistic stacker; features are channel outputs (never raw labels)."""

    FEATURES = ["a_avail", "a_score", "a_q", "a_cand", "v_avail", "v_score", "v_q", "v_cand", "v_contra", "dt_s"]

    def __init__(self, coef: list[float], intercept: float, th_emergency: float, th_pending: float, model_id: str, version: str):
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.th_emergency = th_emergency
        self.th_pending = th_pending
        self.model_id = model_id
        self.version = version

    @staticmethod
    def featurize(a: Evidence | None, v: Evidence | None) -> np.ndarray:
        def g(e: Evidence | None) -> list[float]:
            if e is None or not e.available:
                return [0.0, 0.0, 0.0, 0.0]
            return [1.0, e.score, e.quality_score, 1.0 if e.candidate else 0.0]

        dt = 0.0
        if a and v and a.peak_time_ms is not None and v.peak_time_ms is not None:
            dt = abs(a.peak_time_ms - v.peak_time_ms) / 1000.0
        contra = 1.0 if (v is not None and v.contradicts) else 0.0
        return np.asarray(g(a) + g(v)[:4] + [contra, min(dt, 5.0)], dtype=float)

    def predict(self, a: Evidence | None, v: Evidence | None) -> float:
        z = float(self.featurize(a, v) @ self.coef + self.intercept)
        return float(1.0 / (1.0 + np.exp(-z)))


class LateFusionEngine:
    def __init__(
        self,
        cfg: AppConfig,
        accel: Any,
        vision: Any,
        subject_id: str,
        room_id: str | None,
        source_mode: str,
        backend: str | None = None,
        trained: TrainedFusionModel | None = None,
    ) -> None:
        self.cfg = cfg
        self.accel = accel
        self.vision = vision
        self.subject_id = subject_id
        self.room_id = room_id
        self.source_mode = source_mode
        self.backend = backend or cfg.fall.fusion.backend
        if self.backend == "feature_fusion":
            raise ValueError("feature_fusion is handled by FeatureFusionEngine, not LateFusionEngine")
        self.trained = trained
        if self.backend == "late_trained" and trained is None:
            raise ValueError("late_trained backend requires a trained fusion model package")
        f = cfg.fall.fusion
        self.min_q = f.min_quality
        self.max_unc = cfg.quality.max_clock_uncertainty_ms
        self.tol = cfg.quality.fusion_time_tolerance_ms
        self.wait_ms = f.fusion_wait_ms
        self.pending_min = f.pending_min_score
        self.weights = f.weights
        self.assoc_ms = 2500.0
        self.upgrade_window_ms = 12_000.0
        self.cases: list[FusionCase] = []
        self.history: list[dict[str, Any]] = []

    # --------------------------------------------------------------- intake
    def on_evidence(self, ev: Evidence, now: float) -> None:
        if not ev.candidate:
            return
        t = ev.peak_time_ms if ev.peak_time_ms is not None else ev.time_end_ms
        for case in self.cases:
            if abs(case.anchor_ms - t) <= self.assoc_ms:
                prev = case.ev.get(ev.channel)
                if prev is None or not prev.candidate or ev.score > prev.score:
                    case.ev[ev.channel] = ev
                return
        case = FusionCase(
            case_id=new_id("fc_"), anchor_ms=t, created_ms=now, deadline_ms=now + self.wait_ms, ev={ev.channel: ev}
        )
        self.cases.append(case)

    # ----------------------------------------------------------------- tick
    def tick(self, now: float, final: bool = False) -> list[FusionDecision]:
        out: list[FusionDecision] = []
        for case in list(self.cases):
            both = all(ch in case.ev and case.ev[ch].candidate for ch in ("accel", "vision"))
            if case.decided is None:
                if both or now >= case.deadline_ms or final:
                    d = self._decide(case, now)
                    case.decided = d.level
                    case.decisions += 1
                    case.last_decision = d
                    case.open_until_ms = now + self.upgrade_window_ms
                    out.append(d)
            elif case.decided != DecisionLevel.EMERGENCY and now <= case.open_until_ms:
                # Upgrade path: a late candidate from the missing channel may raise the level.
                if both and case.decisions < 3:
                    d = self._decide(case, now)
                    if LEVEL_RANK[d.level] > LEVEL_RANK[case.decided]:
                        case.decided = d.level
                        case.decisions += 1
                        case.last_decision = d
                        out.append(d)
            if case.decided is not None and (now > case.open_until_ms or case.decided == DecisionLevel.EMERGENCY):
                self.cases.remove(case)
        return out

    # --------------------------------------------------------------- decide
    def _fill(self, case: FusionCase, now: float) -> tuple[Evidence, Evidence]:
        a = case.ev.get("accel")
        v = case.ev.get("vision")
        if a is None:
            a = self.accel.assess(case.anchor_ms, now, self.tol)
        if v is None:
            v = self.vision.assess(case.anchor_ms, now)
        return a, v

    def _fused_score(self, a: Evidence, v: Evidence) -> float | None:
        num = den = 0.0
        for e in (a, v):
            if e.available and e.quality_score > 0:
                w = self.weights.get(e.channel, 1.0) * e.quality_score
                num += w * (e.score if not e.contradicts else 0.0)
                den += w
        return None if den == 0 else num / den

    def _decide(self, case: FusionCase, now: float) -> FusionDecision:
        a, v = self._fill(case, now)
        gate: dict[str, Any] = {}
        reasons: list[str] = []
        a_valid = _valid(a, self.min_q, self.max_unc)
        v_valid = _valid(v, self.min_q, self.max_unc)
        a_strong = a.candidate and a.strong
        v_strong = v.candidate and v.strong
        dt = None
        if a.peak_time_ms is not None and v.peak_time_ms is not None and a.candidate and v.candidate:
            dt = abs(a.peak_time_ms - v.peak_time_ms)
        time_ok = dt is not None and dt <= self.tol
        target_ok = _target_ok(v)
        gate.update(
            accel_valid=a_valid, vision_valid=v_valid, accel_strong=a_strong, vision_strong=v_strong,
            accel_candidate=a.candidate, vision_candidate=v.candidate, dt_ms=dt, time_ok=time_ok,
            target_ok=target_ok, vision_contradicts=v.contradicts,
        )
        fused = self._fused_score(a, v)
        level = DecisionLevel.NORMAL
        rejected = False

        if self.backend == "late_rule_basic":
            if a.candidate and v.candidate:
                level = DecisionLevel.EMERGENCY
                reasons.append("both_channels_candidate")
            elif a.candidate or v.candidate:
                level = DecisionLevel.PENDING
                reasons.append("single_channel_candidate")
            else:
                rejected = True
                reasons.append("no_candidate")
        else:
            dual_gate = a_valid and v_valid and a_strong and v_strong and time_ok and target_ok
            if self.backend == "late_trained":
                assert self.trained is not None
                p = self.trained.predict(a, v)
                gate["p_trained"] = round(p, 4)
                fused = p
                if dual_gate and p >= self.trained.th_emergency:
                    level = DecisionLevel.EMERGENCY
                    reasons.append("trained_fusion_emergency")
                elif p >= self.trained.th_pending and (a.candidate or v.candidate):
                    level = DecisionLevel.PENDING
                    reasons.append("trained_fusion_pending")
                else:
                    rejected = True
                    reasons.append("trained_fusion_below_threshold")
            elif dual_gate:
                level = DecisionLevel.EMERGENCY
                reasons.append("aligned_motion_and_posture_evidence")
            else:
                has_cand = a.candidate or v.candidate
                vision_refutes = (
                    a.candidate and not v.candidate and v.contradicts and v_valid and target_ok
                )
                if vision_refutes:
                    rejected = True
                    reasons.append("motion_candidate_refuted_by_vision")
                elif has_cand:
                    level = DecisionLevel.PENDING
                else:
                    rejected = True
                    reasons.append("no_candidate")
            # explain why EMERGENCY was not reached
            if level != DecisionLevel.EMERGENCY:
                if not a.available:
                    reasons.append("accel_missing")
                elif not a_valid:
                    reasons.append("accel_untrusted_or_low_quality")
                elif a.candidate and not a_strong:
                    reasons.append("accel_weak")
                elif not a.candidate:
                    reasons.append("accel_no_impact")
                if not v.available:
                    reasons.append("vision_missing")
                elif not v_valid:
                    reasons.append("vision_low_quality")
                elif not target_ok:
                    reasons.append("target_not_confirmed")
                elif v.candidate and not v_strong:
                    reasons.append("vision_weak")
                elif not v.candidate:
                    reasons.append("vision_no_fall_posture")
                if a.candidate and v.candidate and not time_ok:
                    reasons.append("time_inconsistent")
        onset = None
        vd = v.features_summary.get("descent_start_ms") if v.candidate else None
        if vd is not None:
            onset = float(vd)
        elif a.candidate and a.peak_time_ms is not None:
            onset = a.peak_time_ms - 1000.0
        decision = FusionDecision(
            decision_id=f"{case.case_id}:{case.decisions}",
            subject_id=self.subject_id,
            level=level,
            rejected=rejected,
            decided_at_ms=now,
            anchor_time_ms=case.anchor_ms,
            onset_estimate_ms=onset,
            fusion_backend=self.backend,
            fused_score=None if fused is None else round(float(fused), 4),
            gate=gate,
            reason_codes=sorted(set(reasons)),
            evidences=[a, v],
            source_mode=self.source_mode,
            room_id=self.room_id,
        )
        self.history.append({"t": now, "decision_id": decision.decision_id, "level": level.value, "reasons": decision.reason_codes})
        self.history = self.history[-200:]
        return decision

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "case_id": c.case_id,
                "anchor_ms": c.anchor_ms,
                "deadline_ms": c.deadline_ms,
                "channels": {k: {"candidate": e.candidate, "score": e.score, "q": e.quality_score} for k, e in c.ev.items()},
                "decided": c.decided.value if c.decided else None,
            }
            for c in self.cases
        ]
