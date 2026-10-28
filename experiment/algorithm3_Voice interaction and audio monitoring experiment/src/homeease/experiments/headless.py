"""Headless replay through the complete backend (events, outbox, timers, exercise,
coverage) on a virtual clock. Used by `homeease replay`, scenario acceptance
tests and system-level replay evaluation.

The virtual clock follows the recording time axis, so measured latencies are
recording-time latencies; this does not measure real network delivery,
energy use or long-term stability.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select

from homeease.backend.services.bootstrap import ensure_contact, ensure_subject, ensure_user
from homeease.backend.services.runtime import Runtime
from homeease.backend.storage.db import Database
from homeease.backend.storage.models import Event, EventTransition, ExerciseSession, HealthInterval, Notification
from homeease.core.clock import VirtualClock, iso
from homeease.core.config import AppConfig
from homeease.datasets.recording import Recording, load_recording
from homeease.edge.acquisition.replay import ReplayFeeder

NODE_BASE = 1_000_000.0


@dataclass
class HeadlessResult:
    recording_id: str
    decisions: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    notifications: list[dict[str, Any]] = field(default_factory=list)
    exercise: list[dict[str, Any]] = field(default_factory=list)
    coverage: list[dict[str, Any]] = field(default_factory=list)
    final_coverage: dict[str, Any] = field(default_factory=dict)
    ingest_flags: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class HeadlessSystem:
    """A Runtime on a VirtualClock with a private SQLite database."""

    def __init__(self, cfg: AppConfig, db_url: str | None = None, contacts: int = 2, source_mode: str = "replay") -> None:
        cfg = cfg.model_copy(deep=True)
        cfg.notifications.test_sink.enabled = True
        # Replayed / synthetic inputs are never labelled as live events.
        cfg.runtime.mode = source_mode if source_mode in ("replay", "synthetic") else "replay"
        self.cfg = cfg
        self._tmp = None
        if db_url is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="homeease_headless_")
            db_url = f"sqlite:///{self._tmp.name}/headless.db"
        self.db = Database(db_url)
        self.db.migrate()
        ensure_subject(self.db, cfg)
        self.user_ids = []
        for i in range(contacts):
            uid, _ = ensure_user(self.db, f"cg{i + 1}", "caregiver", f"Caregiver {i + 1}", "test-password-123")
            ensure_contact(self.db, cfg, f"Caregiver {i + 1}", uid, i + 1)
            self.user_ids.append(uid)
        self.clock = VirtualClock(NODE_BASE)
        self.rt = Runtime(cfg, self.db, clock=self.clock, headless=True)
        self.rt.broadcaster.sink = []
        self.rt.recover()

    def step(self, now: float, out: Any = None) -> None:
        self.clock.set(max(now, self.clock.mono_ms()))
        if out is not None:
            self.rt.handle(out)
        self.rt.notifier.dispatch_due()
        self.rt.events.run_due_tasks()

    def advance(self, seconds: float, tick_ms: float = 100.0) -> None:
        t = self.clock.mono_ms()
        end = t + seconds * 1000.0
        while t < end:
            t += tick_ms
            out = self.rt.pipeline.tick(t)
            self.step(t, out)

    def close(self) -> None:
        self.db.dispose()
        if self._tmp is not None:
            self._tmp.cleanup()


def run_recording(
    system: HeadlessSystem,
    rec: Recording,
    extra_s: float = 8.0,
    after_s: float = 0.0,
    on_tick: Any = None,
) -> HeadlessResult:
    rt = system.rt
    res = HeadlessResult(rec.recording_id)

    def start_ex(ex: dict[str, Any]) -> None:
        rt.exercise.start(rt.subject_id, int(ex.get("target", 10)))

    feeder = ReplayFeeder(rec, system.cfg, rt.pipeline, rt.ingestor, node_base_ms=NODE_BASE, on_exercise_start=start_ex)
    for out in feeder.iter_ticks(extra_ms=extra_s * 1000.0):
        for d in out.decisions:
            res.decisions.append({"t_rec_ms": round(out.now - NODE_BASE, 1), "id": d.decision_id, "level": d.level.value, "rejected": d.rejected, "reasons": d.reason_codes, "gate": d.gate, "anchor_rec_ms": round(d.anchor_time_ms - NODE_BASE, 1)})
        if out.coverage is not None and out.coverage.changed:
            res.coverage.append({"t_rec_ms": round(out.now - NODE_BASE, 1), "status": out.coverage.status, "reasons": out.coverage.reasons})
        system.step(out.now, out)
        if on_tick is not None:
            on_tick(out)
    if after_s > 0:
        system.advance(after_s)
    res.final_coverage = {"status": rt.coverage.current.get("status"), "reasons": rt.coverage.current.get("reasons")}
    res.ingest_flags = sorted({f for l in feeder.ingest_log for f in l.get("flags", [])})
    collect(system, res)
    return res


def collect(system: HeadlessSystem, res: HeadlessResult) -> None:
    with system.db.session() as s:
        for e in s.execute(select(Event).order_by(Event.classified_at)).scalars():
            res.events.append({
                "id": e.id, "origin": e.origin, "level": e.decision_level, "max_level": e.max_level, "status": e.lifecycle_status,
                "outcome": e.outcome, "source_mode": e.source_mode, "classified_at": iso(e.classified_at), "candidate_at": iso(e.candidate_at),
                "escalation_step": e.escalation_step, "summary": e.summary,
                "transitions": [t.action for t in s.execute(select(EventTransition).where(EventTransition.event_id == e.id).order_by(EventTransition.seq)).scalars()],
            })
        for n in s.execute(select(Notification).order_by(Notification.id)).scalars():
            res.notifications.append({"id": n.id, "event_id": n.event_id, "type": n.notification_type, "step": n.escalation_step, "channel": n.channel, "status": n.status, "contact": n.recipient_contact_id, "queued_at": iso(n.queued_at), "delivered_at": iso(n.delivered_at), "is_test": n.is_test})
        for x in s.execute(select(ExerciseSession)).scalars():
            res.exercise.append({"id": x.id, "status": x.status, "count": x.count_valid, "target": x.target_reps, "sub_state": x.sub_state, "definition": x.repetition_definition})
    res.messages = [{"type": m["type"], "payload_keys": sorted(m["payload"].keys())} for m in (system.rt.broadcaster.sink or [])[-50:]]


def replay_headless(cfg: AppConfig, recording: str | Path, extra_s: float = 8.0, after_s: float = 0.0, db_url: str | None = None) -> HeadlessResult:
    rec = load_recording(recording)
    sysm = HeadlessSystem(cfg, db_url, source_mode=rec.source_mode)
    try:
        return run_recording(sysm, rec, extra_s, after_s)
    finally:
        sysm.close()
