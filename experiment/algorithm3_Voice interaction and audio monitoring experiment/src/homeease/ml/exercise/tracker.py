"""Exercise session tracker: phases, counting, quality pauses and interruptions.

Session status values (persisted by the backend):
CREATED, RUNNING, PAUSED, COMPLETED, ABORTED, INTERRUPTED. Inside RUNNING the
tracker reports a sub-state: waiting_for_sitting | counting | paused_quality.
Manual pauses (user) and automatic pauses (recognition unavailable) are
recorded separately. Unknown periods never produce counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from homeease.core.config import ExerciseCfg
from homeease.ml.exercise.counter import CounterEvent, RepCounter
from homeease.ml.exercise.features import UNKNOWN, ExerciseFeatureExtractor
from homeease.ml.exercise.phase import PhaseClassifier, RulePhaseClassifier
from homeease.ml.pose.layout import PoseObs


@dataclass
class Interruption:
    kind: str  # manual_pause | auto_pause_quality | fall_suspected
    start_ms: float
    end_ms: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "start_ms": self.start_ms, "end_ms": self.end_ms, "reason": self.reason}


@dataclass
class ExerciseUpdate:
    session_id: str
    t_ms: float
    count: int
    sub_state: str
    phase: str
    events: list[dict[str, Any]] = field(default_factory=list)
    interruption_started: Interruption | None = None
    interruption_ended: Interruption | None = None


class ExerciseTracker:
    def __init__(
        self,
        cfg: ExerciseCfg,
        session_id: str,
        target_reps: int,
        definition: str | None = None,
        classifier: PhaseClassifier | None = None,
        min_score: float = 0.3,
    ) -> None:
        self.cfg = cfg
        self.session_id = session_id
        self.target = target_reps
        self.definition = definition or cfg.repetition_definition
        self.features = ExerciseFeatureExtractor(min_score)
        self.classifier: PhaseClassifier = classifier or RulePhaseClassifier(cfg)
        self.counter = RepCounter(cfg, self.definition)
        self.sub_state = "waiting_for_sitting"
        self.manual_paused = False
        self.phase = UNKNOWN
        self.interruptions: list[Interruption] = []
        self._open_auto: Interruption | None = None
        self._open_manual: Interruption | None = None
        self._invalid_since: float | None = None
        self._valid_since: float | None = None
        self.frames = 0
        self.valid_frames = 0
        self.start_ms: float | None = None
        self.last_ms: float | None = None
        self.paused_manual_ms = 0.0
        self.paused_auto_ms = 0.0
        self.phase_trace: list[tuple[float, str]] = []

    # --------------------------------------------------------------- control
    def pause(self, t: float, reason: str = "user") -> Interruption | None:
        if self.manual_paused:
            return None
        self.manual_paused = True
        self.counter.abort_partial(t, "manual_pause")
        self.classifier.reset()
        self._open_manual = Interruption("manual_pause", t, reason=reason)
        self.interruptions.append(self._open_manual)
        return self._open_manual

    def resume(self, t: float) -> Interruption | None:
        if not self.manual_paused:
            return None
        self.manual_paused = False
        it = self._open_manual
        if it is not None:
            it.end_ms = t
            self.paused_manual_ms += t - it.start_ms
        self._open_manual = None
        self.sub_state = "waiting_for_sitting"
        self.features.reset_motion()
        return it

    def suspend_for_fall(self, t: float) -> Interruption:
        it = Interruption("fall_suspected", t, reason="fall_event")
        self.counter.abort_partial(t, "fall_suspected")
        self.manual_paused = True
        self._open_manual = it
        self.interruptions.append(it)
        return it

    def close(self, t: float) -> None:
        for it in (self._open_auto, self._open_manual):
            if it is not None and it.end_ms is None:
                it.end_ms = t
                if it.kind == "auto_pause_quality":
                    self.paused_auto_ms += t - it.start_ms
                else:
                    self.paused_manual_ms += t - it.start_ms
        self.counter.abort_partial(t, "session_end")

    # ------------------------------------------------------------------ step
    def push(self, obs: PoseObs) -> ExerciseUpdate:
        t = obs.t_ms
        if self.start_ms is None:
            self.start_ms = t
        self.last_ms = t
        self.frames += 1
        upd = ExerciseUpdate(self.session_id, t, self.counter.count, self.sub_state, self.phase)
        if self.manual_paused:
            upd.sub_state = "paused_manual"
            return upd
        feats = self.features(obs)
        valid = feats is not None and obs.target_visible and "target_ambiguous" not in obs.flags
        cfg = self.cfg
        if not valid:
            self._valid_since = None
            if self._invalid_since is None:
                self._invalid_since = t
            if self._open_auto is None and t - self._invalid_since >= cfg.quality_pause_s * 1000.0:
                self._open_auto = Interruption("auto_pause_quality", self._invalid_since, reason="keypoints_insufficient_or_out_of_view")
                self.interruptions.append(self._open_auto)
                for ev in self.counter.abort_partial(t, "visibility_lost"):
                    upd.events.append({"kind": ev.kind, "t_ms": ev.t_ms, "detail": ev.detail, "cycle": ev.cycle.to_dict() if ev.cycle else None})
                self.classifier.reset()
                self.sub_state = "paused_quality"
                upd.interruption_started = self._open_auto
            phase, _ = self.classifier.step(None)
            self.phase = UNKNOWN
            if self._open_auto is None:
                for ev in self.counter.step(UNKNOWN, t):
                    upd.events.append(self._ev(ev))
            upd.sub_state, upd.phase, upd.count = self.sub_state, self.phase, self.counter.count
            return upd
        self.valid_frames += 1
        self._invalid_since = None
        if self._valid_since is None:
            self._valid_since = t
        if self._open_auto is not None:
            if t - self._valid_since < cfg.resume_stable_s * 1000.0:
                upd.sub_state = "paused_quality"
                return upd
            self._open_auto.end_ms = t
            self.paused_auto_ms += t - self._open_auto.start_ms
            upd.interruption_ended = self._open_auto
            self._open_auto = None
            self.sub_state = "waiting_for_sitting"
        assert feats is not None
        phase, _ = self.classifier.step(feats)
        self.phase = phase
        self.phase_trace.append((t, phase))
        if len(self.phase_trace) > 2000:
            self.phase_trace = self.phase_trace[-2000:]
        for ev in self.counter.step(phase, t):
            upd.events.append(self._ev(ev))
        if self.counter.stable in ("SITTING", "STANDING"):
            self.features.note_phase(self.counter.stable, feats)
        if self.counter.state in ("WAIT_SIT",):
            self.sub_state = "waiting_for_sitting"
        else:
            self.sub_state = "counting"
        upd.sub_state, upd.phase, upd.count = self.sub_state, self.phase, self.counter.count
        return upd

    @staticmethod
    def _ev(ev: CounterEvent) -> dict[str, Any]:
        return {"kind": ev.kind, "t_ms": ev.t_ms, "detail": ev.detail, "cycle": ev.cycle.to_dict() if ev.cycle else None}

    def summary(self) -> dict[str, Any]:
        total = (self.last_ms - self.start_ms) if (self.start_ms is not None and self.last_ms is not None) else 0.0
        return {
            "count": self.counter.count,
            "target": self.target,
            "definition": self.definition,
            "counter_version": self.counter.version,
            "phase_model": getattr(self.classifier, "model_id", "?") + "@" + getattr(self.classifier, "model_version", "?"),
            "frames": self.frames,
            "valid_frame_ratio": (self.valid_frames / self.frames) if self.frames else 0.0,
            "total_s": total / 1000.0,
            "paused_manual_s": self.paused_manual_ms / 1000.0,
            "paused_auto_s": self.paused_auto_ms / 1000.0,
            "effective_s": max(0.0, (total - self.paused_manual_ms - self.paused_auto_ms) / 1000.0),
            "cycles": [c.to_dict() for c in self.counter.cycles],
            "interruptions": [i.to_dict() for i in self.interruptions],
            "target_reached": self.counter.count >= self.target,
        }
