"""Repetition counting.

``RepCounter`` (default) counts a repetition only after the full cycle
SITTING -> RISING -> STANDING -> DESCENDING -> SITTING is observed with
debounced (duration-confirmed) phases and plausible cycle duration.
``stand_complete`` is an explicitly separate definition: count at confirmed
standing, re-arm only after a confirmed sit. Results always carry the definition.

``NaiveThresholdCounter`` is the E4 baseline: one threshold, no hysteresis or
duration checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from homeease.core.config import ExerciseCfg
from homeease.ml.exercise.features import UNKNOWN


@dataclass
class Cycle:
    index: int
    start_ms: float
    stand_ms: float | None = None
    end_ms: float | None = None
    valid: bool = False
    reject_reason: str | None = None
    definition: str = "full_cycle"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "start_ms": self.start_ms, "stand_ms": self.stand_ms, "end_ms": self.end_ms,
            "valid": self.valid, "reject_reason": self.reject_reason, "definition": self.definition,
        }


@dataclass
class CounterEvent:
    kind: str  # "count" | "reject" | "armed" | "abort"
    t_ms: float
    cycle: Cycle | None = None
    detail: str = ""


class RepCounter:
    version = "fsm-1.0"

    def __init__(self, cfg: ExerciseCfg, definition: str | None = None) -> None:
        self.cfg = cfg
        self.definition = definition or cfg.repetition_definition
        if self.definition not in ("full_cycle", "stand_complete"):
            raise ValueError(self.definition)
        self.state = "WAIT_SIT"  # WAIT_SIT, SAT, RISE, STAND, DESC, REARM (stand_complete only)
        self.count = 0
        self.cycles: list[Cycle] = []
        self._cur: Cycle | None = None
        self._cand: str | None = None
        self._cand_since: float | None = None
        self.stable: str = UNKNOWN
        self._unknown_since: float | None = None
        self._last_t: float | None = None

    # -------------------------------------------------------------- helpers
    def _debounce(self, phase: str, t: float) -> str | None:
        """Return a newly confirmed stable phase, or None."""
        if phase == UNKNOWN:
            return None
        if phase != self._cand:
            self._cand, self._cand_since = phase, t
        if self._cand_since is not None and t - self._cand_since >= self.cfg.min_phase_s * 1000.0 and phase != self.stable:
            self.stable = phase
            return phase
        return None

    def abort_partial(self, t: float, reason: str) -> list[CounterEvent]:
        ev: list[CounterEvent] = []
        if self._cur is not None:
            self._cur.reject_reason = reason
            self._cur.end_ms = t
            self.cycles.append(self._cur)
            ev.append(CounterEvent("abort", t, self._cur, reason))
            self._cur = None
        self.state = "WAIT_SIT"
        self.stable = UNKNOWN
        self._cand = None
        self._cand_since = None
        return ev

    def _new_cycle(self, t: float) -> None:
        self._cur = Cycle(index=len(self.cycles), start_ms=t, definition=self.definition)

    def _close(self, t: float, valid: bool, reason: str | None) -> CounterEvent:
        assert self._cur is not None
        c = self._cur
        c.end_ms = t
        c.valid = valid
        c.reject_reason = reason
        self.cycles.append(c)
        self._cur = None
        if valid:
            self.count += 1
            return CounterEvent("count", t, c)
        return CounterEvent("reject", t, c, reason or "")

    # ----------------------------------------------------------------- step
    def step(self, phase: str, t: float) -> list[CounterEvent]:
        cfg = self.cfg
        events: list[CounterEvent] = []
        self._last_t = t
        if phase == UNKNOWN:
            if self._unknown_since is None:
                self._unknown_since = t
            elif t - self._unknown_since > cfg.max_unknown_s * 1000.0 and self.state != "WAIT_SIT":
                events += self.abort_partial(t, "unknown_gap")
            return events
        self._unknown_since = None
        if self._cur is not None and t - self._cur.start_ms > cfg.max_cycle_s * 1000.0:
            events += self.abort_partial(t, "cycle_timeout")
        new = self._debounce(phase, t)
        if new is None:
            return events
        s = self.state
        if s == "REARM":  # stand_complete: re-arm only after a confirmed sit
            if new == "SITTING":
                self.state = "SAT"
                events.append(CounterEvent("armed", t))
        elif s == "WAIT_SIT":
            if new == "SITTING":
                self.state = "SAT"
                events.append(CounterEvent("armed", t))
        elif s == "SAT":
            if new in ("RISING", "STANDING"):
                self._new_cycle(t)
                self.state = "RISE"
                if new == "STANDING":
                    self._on_stand(t, events)
        elif s == "RISE":
            if new == "STANDING":
                self._on_stand(t, events)
            elif new == "SITTING":
                self._close(t, False, "half_rise")
                self.state = "SAT"
        elif s == "STAND":
            if new in ("DESCENDING", "SITTING"):
                self.state = "DESC"
                if new == "SITTING":
                    self._on_sit_after_stand(t, events)
        elif s == "DESC":
            if new == "SITTING":
                self._on_sit_after_stand(t, events)
            elif new == "STANDING":
                self.state = "STAND"
        return events

    def _on_stand(self, t: float, events: list[CounterEvent]) -> None:
        assert self._cur is not None
        self._cur.stand_ms = t
        self.state = "STAND"
        if self.definition == "stand_complete":
            dur = t - self._cur.start_ms
            ok = dur >= self.cfg.min_cycle_s * 1000.0 * 0.4
            ev = self._close(t, ok, None if ok else "too_fast")
            events.append(ev)
            self.state = "REARM"

    def _on_sit_after_stand(self, t: float, events: list[CounterEvent]) -> None:
        if self.definition == "stand_complete":
            self.state = "SAT"
            return
        if self._cur is None:
            self.state = "SAT"
            return
        dur = t - self._cur.start_ms
        if dur < self.cfg.min_cycle_s * 1000.0:
            events.append(self._close(t, False, "too_fast"))
        else:
            events.append(self._close(t, True, None))
        self.state = "SAT"

    def snapshot(self) -> dict[str, Any]:
        return {"state": self.state, "stable_phase": self.stable, "count": self.count, "definition": self.definition}


@dataclass
class NaiveThresholdCounter:
    """Baseline: count each upward crossing of thigh_vertical through one threshold."""

    threshold: float = 0.55
    count: int = 0
    _above: bool | None = None
    crossings: list[float] = field(default_factory=list)
    definition: str = "naive_threshold"
    version: str = "naive-1.0"

    def step_value(self, tv: float | None, t: float) -> bool:
        if tv is None:
            return False
        above = tv >= self.threshold
        counted = False
        if self._above is not None and above and not self._above:
            self.count += 1
            self.crossings.append(t)
            counted = True
        self._above = above
        return counted
