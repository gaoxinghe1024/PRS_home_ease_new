"""Exercise session persistence and control."""

from __future__ import annotations

from typing import Any, Protocol

from sqlalchemy import select

from homeease.backend.services.realtime import Broadcaster
from homeease.backend.storage.db import Database
from homeease.backend.storage.models import ExerciseCycle, ExerciseInterruption, ExerciseSession
from homeease.core.clock import Clock, iso
from homeease.core.config import AppConfig
from homeease.core.util import new_id
from homeease.ml.exercise.tracker import ExerciseUpdate


class TrackerControl(Protocol):
    def start_tracker(self, session_id: str, target: int, definition: str) -> None: ...

    def pause_tracker(self, session_id: str) -> None: ...

    def resume_tracker(self, session_id: str) -> None: ...

    def stop_tracker(self, session_id: str) -> dict[str, Any] | None: ...

    def tracker_mono_ms(self) -> float: ...


class ExerciseError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def session_to_dict(x: ExerciseSession) -> dict[str, Any]:
    return {
        "id": x.id, "subject_id": x.subject_id, "activity": x.activity, "target_reps": x.target_reps, "status": x.status,
        "sub_state": x.sub_state, "repetition_definition": x.repetition_definition, "count_valid": x.count_valid,
        "started_at": iso(x.started_at), "ended_at": iso(x.ended_at), "total_s": round(x.total_s, 1),
        "effective_s": round(x.effective_s, 1), "paused_manual_s": round(x.paused_manual_s, 1), "paused_auto_s": round(x.paused_auto_s, 1),
        "phase_model": x.phase_model, "counter_version": x.counter_version, "quality": x.quality, "target_reached": x.target_reached,
        "source_mode": x.source_mode, "version": x.version,
    }


class ExerciseService:
    def __init__(self, db: Database, cfg: AppConfig, clock: Clock, broadcaster: Broadcaster, control: TrackerControl, source_mode: str) -> None:
        self.db = db
        self.cfg = cfg
        self.clock = clock
        self.b = broadcaster
        self.control = control
        self.source_mode = source_mode
        self.active_id: str | None = None
        self._last_published: tuple[int, str] | None = None

    def start(self, subject_id: str, target: int, activity: str = "sit_to_stand") -> dict[str, Any]:
        if activity != "sit_to_stand":
            raise ExerciseError("unsupported_activity", "only sit_to_stand is available in the MVP")
        if not (1 <= target <= 100):
            raise ExerciseError("invalid_target", "target must be between 1 and 100")
        now = self.clock.utc_now()
        with self.db.tx() as s:
            active = s.execute(select(ExerciseSession).where(ExerciseSession.subject_id == subject_id, ExerciseSession.status.in_(["RUNNING", "PAUSED"]))).scalars().first()
            if active is not None:
                raise ExerciseError("session_active", "a session is already running", 409)
            x = ExerciseSession(
                id=new_id("ex_"), subject_id=subject_id, activity=activity, target_reps=target, status="RUNNING",
                sub_state="waiting_for_sitting", repetition_definition=self.cfg.exercise.repetition_definition, started_at=now,
                created_at=now, updated_at=now, source_mode=self.source_mode, counter_version="fsm-1.0",
            )
            s.add(x)
            s.flush()
            d = session_to_dict(x)
        self.control.start_tracker(d["id"], target, d["repetition_definition"])
        self.active_id = d["id"]
        self.b.publish("exercise_update", {"session": d, "event": "started"}, subject_id, audience="all", durable=True)
        return d

    def action(self, session_id: str, action: str) -> dict[str, Any]:
        now = self.clock.utc_now()
        with self.db.tx() as s:
            x = s.get(ExerciseSession, session_id)
            if x is None:
                raise ExerciseError("not_found", "session not found", 404)
            if x.status not in ("RUNNING", "PAUSED"):
                raise ExerciseError("invalid_transition", f"session is {x.status}", 409)
            if action == "pause":
                if x.status == "PAUSED":
                    return session_to_dict(x)
                self.control.pause_tracker(session_id)
                x.status = "PAUSED"
                s.add(ExerciseInterruption(session_id=x.id, kind="manual_pause", reason="user", start_at=now))
            elif action == "resume":
                if x.status == "RUNNING":
                    return session_to_dict(x)
                self.control.resume_tracker(session_id)
                x.status = "RUNNING"
                x.sub_state = "waiting_for_sitting"
                it = s.execute(select(ExerciseInterruption).where(ExerciseInterruption.session_id == x.id, ExerciseInterruption.kind.in_(["manual_pause", "fall_suspected"]), ExerciseInterruption.end_at.is_(None))).scalars().first()
                if it is not None:
                    it.end_at = now
            elif action in ("end", "abort"):
                summary = self.control.stop_tracker(session_id)
                self._finalize(s, x, summary, "COMPLETED" if action == "end" else "ABORTED")
                self.active_id = None
            else:
                raise ExerciseError("invalid_action", f"unknown action {action}")
            x.version += 1
            x.updated_at = now
            d = session_to_dict(x)
        self.b.publish("exercise_update", {"session": d, "event": action}, x.subject_id, audience="all", durable=True)
        return d

    def _finalize(self, s: Any, x: ExerciseSession, summary: dict[str, Any] | None, status: str) -> None:
        now = self.clock.utc_now()
        x.status = status
        x.ended_at = now
        x.sub_state = None
        if summary:
            x.count_valid = int(summary["count"])
            x.total_s = float(summary["total_s"])
            x.effective_s = float(summary["effective_s"])
            x.paused_manual_s = float(summary["paused_manual_s"])
            x.paused_auto_s = float(summary["paused_auto_s"])
            x.phase_model = summary.get("phase_model")
            x.counter_version = summary.get("counter_version")
            x.target_reached = bool(summary["count"] >= x.target_reps)
            x.quality = {"valid_frame_ratio": round(summary.get("valid_frame_ratio", 0.0), 3), "frames": summary.get("frames", 0)}
            s.query(ExerciseCycle).filter(ExerciseCycle.session_id == x.id).delete()
            for c in summary.get("cycles", []):
                s.add(ExerciseCycle(
                    session_id=x.id, idx=c["index"], definition=c["definition"], valid=c["valid"], reject_reason=c["reject_reason"],
                    cycle_start_at=self.clock.mono_to_utc(c["start_ms"]) if c["start_ms"] is not None else None,
                    stand_reached_at=self.clock.mono_to_utc(c["stand_ms"]) if c["stand_ms"] is not None else None,
                    cycle_end_at=self.clock.mono_to_utc(c["end_ms"]) if c["end_ms"] is not None else None,
                    duration_s=((c["end_ms"] - c["start_ms"]) / 1000.0) if (c["end_ms"] is not None and c["start_ms"] is not None) else None,
                ))
            for it in summary.get("interruptions", []):
                if it["kind"] == "auto_pause_quality":
                    s.add(ExerciseInterruption(
                        session_id=x.id, kind=it["kind"], reason=it["reason"], start_at=self.clock.mono_to_utc(it["start_ms"]),
                        end_at=self.clock.mono_to_utc(it["end_ms"]) if it["end_ms"] is not None else now,
                    ))
            for it in s.execute(select(ExerciseInterruption).where(ExerciseInterruption.session_id == x.id, ExerciseInterruption.end_at.is_(None))).scalars():
                it.end_at = now
        else:
            x.target_reached = x.count_valid >= x.target_reps

    def apply_update(self, upd: ExerciseUpdate) -> None:
        now = self.clock.utc_now()
        changed_struct = any(e["kind"] in ("count", "reject", "abort") for e in upd.events) or upd.interruption_started or upd.interruption_ended
        with self.db.tx() as s:
            x = s.get(ExerciseSession, upd.session_id)
            if x is None or x.status not in ("RUNNING", "PAUSED"):
                return
            x.count_valid = upd.count
            if x.status == "RUNNING":
                x.sub_state = upd.sub_state
            if upd.interruption_started is not None:
                it = upd.interruption_started
                s.add(ExerciseInterruption(session_id=x.id, kind=it.kind, reason=it.reason, start_at=self.clock.mono_to_utc(it.start_ms)))
                if it.kind == "fall_suspected":
                    x.status = "PAUSED"
            if upd.interruption_ended is not None:
                row = s.execute(select(ExerciseInterruption).where(ExerciseInterruption.session_id == x.id, ExerciseInterruption.kind == upd.interruption_ended.kind, ExerciseInterruption.end_at.is_(None))).scalars().first()
                if row is not None:
                    row.end_at = now
            x.updated_at = now
            d = session_to_dict(x)
        key = (upd.count, upd.sub_state)
        if key != self._last_published or changed_struct:
            self._last_published = key
            self.b.publish("exercise_update", {"session": d, "phase": upd.phase, "events": upd.events}, d["subject_id"], audience="all", durable=bool(changed_struct))

    def mark_fall_suspended(self, session_id: str) -> None:
        now = self.clock.utc_now()
        with self.db.tx() as s:
            x = s.get(ExerciseSession, session_id)
            if x is None or x.status != "RUNNING":
                return
            x.status = "PAUSED"
            x.sub_state = "fall_suspected"
            s.add(ExerciseInterruption(session_id=x.id, kind="fall_suspected", reason="fall_event", start_at=now))
            x.version += 1
            d = session_to_dict(x)
        self.b.publish("exercise_update", {"session": d, "event": "fall_suspected"}, d["subject_id"], audience="all", durable=True)

    def recover(self) -> int:
        """Sessions that were running when the node stopped become INTERRUPTED;
        confirmed counts are kept; counting never resumes from stale pose state."""
        now = self.clock.utc_now()
        n = 0
        with self.db.tx() as s:
            for x in s.execute(select(ExerciseSession).where(ExerciseSession.status.in_(["RUNNING", "PAUSED", "CREATED"]))).scalars():
                x.status = "INTERRUPTED"
                x.ended_at = now
                x.sub_state = None
                x.target_reached = x.count_valid >= x.target_reps
                s.add(ExerciseInterruption(session_id=x.id, kind="restart", reason="node_restart", start_at=now, end_at=now))
                for it in s.execute(select(ExerciseInterruption).where(ExerciseInterruption.session_id == x.id, ExerciseInterruption.end_at.is_(None))).scalars():
                    it.end_at = now
                n += 1
        return n

    def detail(self, session_id: str) -> dict[str, Any]:
        with self.db.session() as s:
            x = s.get(ExerciseSession, session_id)
            if x is None:
                raise ExerciseError("not_found", "session not found", 404)
            d = session_to_dict(x)
            d["cycles"] = [
                {"idx": c.idx, "start_at": iso(c.cycle_start_at), "stand_at": iso(c.stand_reached_at), "end_at": iso(c.cycle_end_at), "valid": c.valid, "reject_reason": c.reject_reason, "duration_s": c.duration_s, "definition": c.definition}
                for c in s.execute(select(ExerciseCycle).where(ExerciseCycle.session_id == x.id).order_by(ExerciseCycle.idx)).scalars()
            ]
            d["interruptions"] = [
                {"kind": i.kind, "reason": i.reason, "start_at": iso(i.start_at), "end_at": iso(i.end_at), "duration_s": (i.end_at - i.start_at).total_seconds() if i.end_at else None}
                for i in s.execute(select(ExerciseInterruption).where(ExerciseInterruption.session_id == x.id).order_by(ExerciseInterruption.start_at)).scalars()
            ]
            return d
