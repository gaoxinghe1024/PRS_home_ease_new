"""Event domain service: grading, merging, caregiver actions, escalation timers.

Three separate state dimensions (never one overloaded ``status``):
  decision_level   : PENDING | EMERGENCY        (response grade; never auto-downgraded)
  lifecycle_status : OPEN | ACKNOWLEDGED | IN_PROGRESS | RESOLVED
  coverage         : handled by CoverageService (a camera dropping out never closes an event)

All mutations happen inside ``db.tx()`` (process lock + transaction) and are
written to ``event_transitions`` with optional idempotency keys.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from homeease.backend.services.messages import summarize_decision
from homeease.backend.services.notifications import NotificationService
from homeease.backend.services.realtime import Broadcaster
from homeease.backend.storage.db import Database
from homeease.backend.storage.models import (
    CandidateRecord,
    Contact,
    Event,
    EventTransition,
    EvidenceSnapshot,
    Notification,
    ScheduledTask,
)
from homeease.core.clock import Clock, ensure_utc, iso
from homeease.core.config import AppConfig
from homeease.core.util import get_logger, new_id
from homeease.schemas.evidence import FusionDecision

log = get_logger(__name__)

LEVEL_RANK = {"NORMAL": 0, "PENDING": 1, "EMERGENCY": 2}
OUTCOMES = {"confirmed_fall", "false_alarm", "safe_after_check", "unresolved", "test_event"}
ACTIONS = {"acknowledge", "contact_attempted", "start_handling", "escalate", "resolve", "view", "note"}


class EventError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def event_to_dict(e: Event, detail: bool = False) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": e.id,
        "subject_id": e.subject_id,
        "room_id": e.room_id,
        "source_mode": e.source_mode,
        "origin": e.origin,
        "decision_level": e.decision_level,
        "max_level": e.max_level,
        "lifecycle_status": e.lifecycle_status,
        "candidate_at": iso(e.candidate_at),
        "occurrence_est_at": iso(e.occurrence_est_at),
        "occurrence_source": e.occurrence_source,
        "classified_at": iso(e.classified_at),
        "created_at": iso(e.created_at),
        "updated_at": iso(e.updated_at),
        "resolved_at": iso(e.resolved_at),
        "outcome": e.outcome,
        "outcome_note": e.outcome_note,
        "escalation_step": e.escalation_step,
        "version": e.version,
        "is_test": e.is_test,
        "summary": e.summary,
        "fusion_backend": e.fusion_backend,
    }
    if detail:
        d.update(model_versions=e.model_versions, thresholds_version=e.thresholds_version, prediction=e.prediction, voice_inquiry=e.voice_inquiry)
    return d


class EventService:
    def __init__(self, db: Database, cfg: AppConfig, clock: Clock, notifier: NotificationService, broadcaster: Broadcaster) -> None:
        self.db = db
        self.cfg = cfg
        self.clock = clock
        self.notifier = notifier
        self.broadcaster = broadcaster

    # ------------------------------------------------------------ helpers
    def _contacts(self, s: Any, subject_id: str) -> list[Contact]:
        return list(
            s.execute(select(Contact).where(Contact.subject_id == subject_id, Contact.enabled.is_(True)).order_by(Contact.priority, Contact.id)).scalars()
        )

    def _transition(self, s: Any, e: Event, action: str, actor_type: str, actor_id: str | None, *, from_level: str | None = None, to_level: str | None = None, from_status: str | None = None, to_status: str | None = None, reason: str | None = None, payload: dict[str, Any] | None = None, idem: str | None = None) -> EventTransition:
        seq = (s.execute(select(func.max(EventTransition.seq)).where(EventTransition.event_id == e.id)).scalar() or 0) + 1
        t = EventTransition(
            event_id=e.id, seq=seq, action=action, from_level=from_level, to_level=to_level, from_status=from_status,
            to_status=to_status, actor_type=actor_type, actor_id=actor_id, reason=reason, payload=payload or {},
            idempotency_key=idem, created_at=self.clock.utc_now(),
        )
        s.add(t)
        return t

    def _schedule(self, s: Any, kind: str, e: Event, step: int, due: datetime, payload: dict[str, Any] | None = None) -> None:
        existing = s.execute(select(ScheduledTask).where(ScheduledTask.kind == kind, ScheduledTask.event_id == e.id, ScheduledTask.step == step)).scalar_one_or_none()
        if existing is not None:
            existing.due_at, existing.status, existing.done_at = due, "scheduled", None
            return
        s.add(ScheduledTask(kind=kind, event_id=e.id, step=step, due_at=due, status="scheduled", payload=payload or {}, created_at=self.clock.utc_now()))

    def _cancel_tasks(self, s: Any, e: Event, kinds: list[str] | None = None) -> None:
        q = select(ScheduledTask).where(ScheduledTask.event_id == e.id, ScheduledTask.status == "scheduled")
        if kinds:
            q = q.where(ScheduledTask.kind.in_(kinds))
        for t in s.execute(q).scalars():
            t.status, t.done_at = "cancelled", self.clock.utc_now()

    def _ack_timeout(self, level: str) -> float:
        e = self.cfg.escalation
        return e.emergency_ack_timeout_s if level == "EMERGENCY" else e.pending_ack_timeout_s

    def _notify_initial(self, s: Any, e: Event, now: datetime) -> None:
        contacts = self._contacts(s, e.subject_id)
        if not contacts:
            log.warning("event %s has no enabled contacts", e.id)
            self._schedule(s, "ack_timeout", e, 1, now + timedelta(seconds=self._ack_timeout(e.decision_level)))
            return
        ntype = "emergency" if e.decision_level == "EMERGENCY" else "verification"
        self.notifier.enqueue(s, e, ntype, 0, contacts[:1], now)
        self._schedule(s, "ack_timeout", e, 1, now + timedelta(seconds=self._ack_timeout(e.decision_level)))

    def _notified_contacts(self, s: Any, e: Event) -> list[Contact]:
        contacts = self._contacts(s, e.subject_id)
        return contacts[: max(1, e.escalation_step + 1)]

    def _publish(self, e: Event, change: str) -> None:
        self.broadcaster.publish("event_update", {"change": change, "event": event_to_dict(e)}, e.subject_id, audience="caregiver", durable=True)
        self.broadcaster.publish("elder_status", {"active_event": {"id": e.id, "level": e.decision_level, "status": e.lifecycle_status}}, e.subject_id, audience="elder", durable=False)

    def _snapshot_evidence(self, s: Any, e: Event | None, decision: FusionDecision, now: datetime) -> None:
        for ev in decision.evidences:
            s.add(
                EvidenceSnapshot(
                    event_id=e.id if e else None, decision_id=decision.decision_id, channel=ev.channel, available=ev.available,
                    candidate=ev.candidate, contradicts=ev.contradicts, score=ev.score, score_semantics=ev.score_semantics.value,
                    quality_score=ev.quality_score, time_start_at=self.clock.mono_to_utc(ev.time_start_ms),
                    time_end_at=self.clock.mono_to_utc(ev.time_end_ms),
                    peak_at=self.clock.mono_to_utc(ev.peak_time_ms) if ev.peak_time_ms is not None else None,
                    features=ev.features_summary, reason_codes=ev.reason_codes, quality_flags=ev.quality_flags,
                    model_id=ev.model_id, model_version=ev.model_version, calibration_id=ev.calibration_id,
                    source_mode=ev.source_mode.value if hasattr(ev.source_mode, "value") else str(ev.source_mode), created_at=now,
                )
            )

    def _find_open(self, s: Any, subject_id: str, anchor: datetime) -> Event | None:
        gap = timedelta(seconds=self.cfg.fall.merge_gap_s)
        rows = s.execute(select(Event).where(Event.subject_id == subject_id, Event.lifecycle_status != "RESOLVED").order_by(Event.classified_at.desc())).scalars().all()
        for e in rows:
            ref = ensure_utc(e.candidate_at) or ensure_utc(e.classified_at)
            if ref is not None and abs((anchor - ref).total_seconds()) <= gap.total_seconds():
                return e
        return None

    def _find_recently_resolved(self, s: Any, subject_id: str, anchor: datetime, now: datetime) -> Event | None:
        cool = timedelta(seconds=self.cfg.fall.cooldown_s)
        rows = s.execute(select(Event).where(Event.subject_id == subject_id, Event.lifecycle_status == "RESOLVED", Event.resolved_at >= now - cool)).scalars().all()
        for e in rows:
            ref = ensure_utc(e.candidate_at) or ensure_utc(e.classified_at)
            if ref is not None and abs((anchor - ref).total_seconds()) <= self.cfg.fall.merge_gap_s:
                return e
        return None

    # --------------------------------------------------- machine decisions
    def ingest_decision(self, decision: FusionDecision) -> Event | None:
        now = self.clock.utc_now()
        level = decision.level.value
        case_id = decision.decision_id.split(":")[0]
        change = None
        with self.db.tx() as s:
            if s.execute(select(CandidateRecord.id).where(CandidateRecord.decision_id == decision.decision_id)).first():
                return None
            rec = CandidateRecord(
                decision_id=decision.decision_id, case_id=case_id, subject_id=decision.subject_id, level=level,
                rejected=decision.rejected, decision=decision.model_dump(mode="json"), source_mode=decision.source_mode.value, created_at=now,
            )
            s.add(rec)
            if level == "NORMAL" or decision.rejected:
                self._snapshot_evidence(s, None, decision, now)
                return None
            anchor = self.clock.mono_to_utc(decision.anchor_time_ms)
            linked = s.execute(select(CandidateRecord).where(CandidateRecord.case_id == case_id, CandidateRecord.event_id.is_not(None))).scalars().first()
            e = s.get(Event, linked.event_id) if linked else None
            if e is not None and e.lifecycle_status == "RESOLVED":
                if LEVEL_RANK[level] <= LEVEL_RANK[e.max_level]:
                    rec.event_id = e.id
                    self._snapshot_evidence(s, e, decision, now)
                    self._transition(s, e, "post_resolution_evidence", "system", "fusion", reason=level, payload={"decision_id": decision.decision_id})
                    return e
                e = None  # the closed episode is now graded more severe: a new, notified event
            if e is None:
                e = self._find_open(s, decision.subject_id, anchor)
            if e is None:
                recent = self._find_recently_resolved(s, decision.subject_id, anchor, now)
                # The cooldown absorbs lingering evidence after a caregiver closed an event, but it
                # never swallows a clear new event (prompt 7.6): an EMERGENCY or a more severe grade.
                if recent is not None and level != "EMERGENCY" and LEVEL_RANK[level] <= LEVEL_RANK[recent.max_level]:
                    rec.event_id = recent.id
                    self._snapshot_evidence(s, recent, decision, now)
                    self._transition(s, recent, "post_resolution_evidence", "system", "fusion", reason=level, payload={"decision_id": decision.decision_id})
                    return recent
                d = decision.model_dump(mode="json")
                onset_src = None
                onset = None
                if decision.onset_estimate_ms is not None:
                    onset = self.clock.mono_to_utc(decision.onset_estimate_ms)
                    v = next((x for x in decision.evidences if x.channel == "vision" and x.candidate), None)
                    onset_src = "vision_descent_start_estimate" if v else "accel_peak_minus_1s_estimate"
                e = Event(
                    id=new_id("ev_"), subject_id=decision.subject_id, room_id=decision.room_id, source_mode=decision.source_mode.value,
                    origin="machine", decision_level=level, max_level=level, lifecycle_status="OPEN", candidate_at=anchor,
                    occurrence_est_at=onset, occurrence_source=onset_src, classified_at=now, anchor_mono_ms=decision.anchor_time_ms,
                    created_at=now, updated_at=now, version=1, is_test=decision.source_mode.value != "live",
                    fusion_backend=decision.fusion_backend,
                    model_versions={x.channel: f"{x.model_id}@{x.model_version}" for x in decision.evidences},
                    thresholds_version=self.cfg.hash(), summary=summarize_decision(d), prediction=d,
                )
                s.add(e)
                s.flush()
                rec.event_id = e.id
                self._snapshot_evidence(s, e, decision, now)
                self._transition(s, e, "created", "system", "fusion", to_level=level, to_status="OPEN", reason=",".join(decision.reason_codes), payload={"decision_id": decision.decision_id, "fused_score": decision.fused_score})
                self._notify_initial(s, e, now)
                change = "created"
            else:
                rec.event_id = e.id
                self._snapshot_evidence(s, e, decision, now)
                if LEVEL_RANK[level] > LEVEL_RANK[e.decision_level]:
                    old = e.decision_level
                    e.decision_level = e.max_level = level
                    e.summary = summarize_decision(decision.model_dump(mode="json"))
                    e.version += 1
                    e.updated_at = now
                    self._transition(s, e, "machine_upgrade", "system", "fusion", from_level=old, to_level=level, reason=",".join(decision.reason_codes), payload={"decision_id": decision.decision_id})
                    if e.lifecycle_status != "RESOLVED":
                        self.notifier.enqueue(s, e, "emergency", e.escalation_step, self._notified_contacts(s, e), now)
                        self._cancel_tasks(s, e, ["ack_timeout"])
                        if e.lifecycle_status == "OPEN":
                            self._schedule(s, "ack_timeout", e, e.escalation_step + 1, now + timedelta(seconds=self._ack_timeout(level)))
                    change = "upgraded"
                else:
                    self._transition(s, e, "evidence_added", "system", "fusion", reason=level, payload={"decision_id": decision.decision_id})
                    change = "evidence_added"
        if change:
            self._publish(e, change)
        return e

    # ---------------------------------------------------------- help button
    def help_request(self, subject_id: str, actor_type: str, actor_id: str | None, idem: str | None, source_mode: str = "live", room_id: str | None = None) -> Event:
        now = self.clock.utc_now()
        with self.db.tx() as s:
            if idem:
                t = s.execute(select(EventTransition).where(EventTransition.idempotency_key == idem, EventTransition.action.in_(["help_request", "created"]))).scalars().first()
                if t is not None:
                    return s.get(Event, t.event_id)
            e = self._find_open(s, subject_id, now)
            if e is not None:
                old = e.decision_level
                e.decision_level = e.max_level = "EMERGENCY"
                e.version += 1
                e.updated_at = now
                self._transition(s, e, "help_request", actor_type, actor_id, from_level=old, to_level="EMERGENCY", reason="explicit help request", idem=idem)
                self.notifier.enqueue(s, e, "help_request", e.escalation_step, self._notified_contacts(s, e), now)
                self._cancel_tasks(s, e, ["ack_timeout"])
                if e.lifecycle_status == "OPEN":
                    self._schedule(s, "ack_timeout", e, e.escalation_step + 1, now + timedelta(seconds=self._ack_timeout("EMERGENCY")))
                change = "upgraded"
            else:
                e = Event(
                    id=new_id("ev_"), subject_id=subject_id, room_id=room_id, source_mode=source_mode, origin="help_request",
                    decision_level="EMERGENCY", max_level="EMERGENCY", lifecycle_status="OPEN", candidate_at=now,
                    occurrence_est_at=now, occurrence_source="help_button_press", classified_at=now, created_at=now, updated_at=now,
                    version=1, is_test=source_mode != "live", summary={"phrases": [], "level": "EMERGENCY", "origin": "help_request"},
                    prediction={}, model_versions={}, thresholds_version=self.cfg.hash(),
                )
                s.add(e)
                s.flush()
                self._transition(s, e, "created", actor_type, actor_id, to_level="EMERGENCY", to_status="OPEN", reason="explicit help request (no dual-channel gate)", idem=idem)
                self._notify_initial(s, e, now)
                change = "created"
        self._publish(e, change)
        return e

    # ----------------------------------------------------- elder response
    def elder_response(self, event_id: str, response: str, actor_type: str, actor_id: str | None, idem: str | None, source: str = "button") -> Event:
        """The older adult's reply. 'need_help' escalates like the help button;
        'im_ok' is recorded as supporting evidence only and never closes or
        downgrades the event (MVP rule)."""
        if response == "need_help":
            with self.db.session() as s:
                e = s.get(Event, event_id)
                if e is None:
                    raise EventError("not_found", "event not found", 404)
                subject, room, mode = e.subject_id, e.room_id, e.source_mode
            return self.help_request(subject, actor_type, actor_id, idem, mode, room)
        if response != "im_ok":
            raise EventError("invalid_payload", "response must be need_help or im_ok")
        with self.db.tx() as s:
            e = s.get(Event, event_id)
            if e is None:
                raise EventError("not_found", "event not found", 404)
            if idem and s.execute(select(EventTransition.id).where(EventTransition.event_id == e.id, EventTransition.idempotency_key == idem)).first():
                return e
            self._transition(s, e, "elder_reassurance", actor_type, actor_id, reason="supporting evidence only; does not cancel or downgrade the event", payload={"source": source}, idem=idem)
            e.updated_at = self.clock.utc_now()
        self._publish(e, "elder_reassurance")
        return e

    # ------------------------------------------------------ voice inquiry
    def _storable_voice_record(self, rec: dict[str, Any]) -> dict[str, Any]:
        """What may be persisted of a voice reply: labels and numbers. The text of the reply is personal content (a visitor, the TV,
        a private remark) and is dropped unless ``privacy.persist_voice_transcript`` is on; the audit transitions never get it."""
        if "transcript" not in rec:
            return rec
        out = {k: v for k, v in rec.items() if k != "transcript"}
        out["transcript_chars"] = len(rec.get("transcript") or "")
        if self.cfg.privacy.persist_voice_transcript:
            out["transcript"] = str(rec.get("transcript") or "")[:100]
        return out

    def voice_response(self, event_id: str, rec: dict[str, Any]) -> Event:
        """Attach a voice-inquiry result as supporting evidence. HELP raises PENDING to
        EMERGENCY; REASSURANCE / NO_RESPONSE / UNCERTAIN never change level or status. The voiceprint evidence in the record
        (`cancel_eligible`, `speaker_score`) is stored with it and nothing else: it is not read here, so it cannot close or downgrade an event."""
        now = self.clock.utc_now()
        upgraded = False
        rec = self._storable_voice_record(rec)
        with self.db.tx() as s:
            e = s.get(Event, event_id)
            if e is None:
                raise EventError("not_found", "event not found", 404)
            hist = list((e.voice_inquiry or {}).get("history", []))
            hist.append(rec)
            e.voice_inquiry = {"latest": rec, "history": hist[-5:], "auto_cancel_emergency": False}
            self._transition(s, e, "voice_response", "system", "audio", reason=f"{rec.get('status')}:{rec.get('intent')} (supporting evidence only)", payload={k: rec.get(k) for k in ("prompt_id", "status", "intent", "lang", "cancel_eligible", "speaker_verified")})  # cancel_eligible is logged for audit, never read below
            if rec.get("intent") == "HELP" and e.lifecycle_status != "RESOLVED" and e.decision_level != "EMERGENCY":
                e.decision_level = e.max_level = "EMERGENCY"
                self._transition(s, e, "voice_help_escalation", "system", "audio", from_level="PENDING", to_level="EMERGENCY", reason="spoken request for help (speaker not verified)")
                self.notifier.enqueue(s, e, "emergency", e.escalation_step, self._notified_contacts(s, e), now)
                upgraded = True
            e.version += 1
            e.updated_at = now
        self._publish(e, "voice_response" + ("+upgraded" if upgraded else ""))
        return e

    # ------------------------------------------------------ caregiver actions
    def apply_action(self, event_id: str, action: str, user_id: int, payload: dict[str, Any] | None = None, idem: str | None = None, expected_version: int | None = None) -> Event:
        if action not in ACTIONS:
            raise EventError("invalid_action", f"unknown action {action}")
        payload = payload or {}
        now = self.clock.utc_now()
        actor = str(user_id)
        with self.db.tx() as s:
            e = s.get(Event, event_id)
            if e is None:
                raise EventError("not_found", "event not found", 404)
            if action == "view":
                self.notifier.mark_viewed_for_event(s, e.id, user_id, now)
                key = f"view:{user_id}"
                if not s.execute(select(EventTransition.id).where(EventTransition.event_id == e.id, EventTransition.idempotency_key == key)).first():
                    self._transition(s, e, "view", "user", actor, idem=key)
                return e
            if idem:
                prior = s.execute(select(EventTransition).where(EventTransition.event_id == e.id, EventTransition.idempotency_key == idem)).scalars().first()
                if prior is not None:
                    return e  # idempotent replay
            if expected_version is not None and expected_version != e.version:
                raise EventError("version_conflict", f"event changed (version {e.version}); reload and retry", 409)
            if e.lifecycle_status == "RESOLVED":
                raise EventError("invalid_transition", "event is already resolved", 409)
            st = e.lifecycle_status
            if action == "acknowledge":
                if st == "OPEN":
                    e.lifecycle_status = "ACKNOWLEDGED"
                self.notifier.mark_acknowledged(s, e.id, user_id, now)
                self._cancel_tasks(s, e, ["ack_timeout", "repeat_reminder"])
                if e.lifecycle_status == "ACKNOWLEDGED":
                    self._schedule(s, "handling_reminder", e, 0, now + timedelta(seconds=self.cfg.escalation.handling_reminder_s), {"user_id": user_id})
                self._transition(s, e, "acknowledge", "user", actor, from_status=st, to_status=e.lifecycle_status, idem=idem)
            elif action == "start_handling":
                e.lifecycle_status = "IN_PROGRESS"
                self.notifier.mark_acknowledged(s, e.id, user_id, now)
                self._cancel_tasks(s, e, ["ack_timeout", "handling_reminder", "repeat_reminder"])
                self._schedule(s, "handling_followup", e, 0, now + timedelta(seconds=self.cfg.escalation.handling_reminder_s), {"user_id": user_id})
                self._transition(s, e, "start_handling", "user", actor, from_status=st, to_status="IN_PROGRESS", payload={"note": payload.get("note")}, idem=idem)
            elif action == "contact_attempted":
                method = payload.get("method", "phone")
                result = payload.get("result", "unknown")
                if method not in ("phone", "message", "in_person", "video", "other") or result not in ("reached", "no_answer", "busy", "unknown", "other"):
                    raise EventError("invalid_payload", "invalid contact method/result")
                self._transition(s, e, "contact_attempted", "user", actor, payload={"method": method, "result": result, "note": payload.get("note")}, idem=idem)
            elif action == "note":
                note = str(payload.get("note", ""))[:2000]
                if not note:
                    raise EventError("invalid_payload", "note is required")
                self._transition(s, e, "note", "user", actor, payload={"note": note}, idem=idem)
            elif action == "escalate":
                old = e.decision_level
                if payload.get("raise_level") and old != "EMERGENCY":
                    e.decision_level = e.max_level = "EMERGENCY"
                contacts = self._contacts(s, e.subject_id)
                nxt = e.escalation_step + 1
                if nxt < len(contacts):
                    e.escalation_step = nxt
                    self.notifier.enqueue(s, e, "escalation", nxt, [contacts[nxt]], now)
                else:
                    self.notifier.enqueue(s, e, "escalation", nxt, contacts, now)
                self._transition(s, e, "escalate", "user", actor, from_level=old, to_level=e.decision_level, payload={"note": payload.get("note"), "to_step": nxt}, idem=idem)
            elif action == "resolve":
                outcome = payload.get("outcome")
                if outcome not in OUTCOMES:
                    raise EventError("invalid_outcome", f"outcome must be one of {sorted(OUTCOMES)}")
                e.lifecycle_status = "RESOLVED"
                e.outcome = outcome
                e.outcome_note = (payload.get("note") or "")[:2000] or None
                e.resolved_at = now
                e.resolved_by = user_id
                self._cancel_tasks(s, e)
                cancelled = self.notifier.cancel_pending(s, e.id, now, "event_resolved")
                self.notifier.mark_acknowledged(s, e.id, user_id, now)
                notified = self._notified_contacts(s, e)
                others = [c for c in notified if c.user_id != user_id]
                if others:
                    self.notifier.enqueue(s, e, "resolved", 0, others, now)
                self._transition(s, e, "resolve", "user", actor, from_status=st, to_status="RESOLVED", payload={"outcome": outcome, "note": e.outcome_note, "cancelled_notifications": cancelled}, idem=idem)
            e.version += 1
            e.updated_at = now
        self._publish(e, action)
        return e

    # ------------------------------------------------------------ scheduler
    def run_due_tasks(self) -> int:
        now = self.clock.utc_now()
        with self.db.tx() as s:
            ids = [t.id for t in s.execute(select(ScheduledTask).where(ScheduledTask.status == "scheduled", ScheduledTask.due_at <= now).order_by(ScheduledTask.due_at).limit(50)).scalars()]
        n = 0
        for tid in ids:
            if self.run_task(tid):
                n += 1
        return n

    def run_task(self, task_id: int) -> bool:
        now = self.clock.utc_now()
        published: Event | None = None
        with self.db.tx() as s:
            t = s.get(ScheduledTask, task_id)
            if t is None or t.status != "scheduled":
                return False
            e = s.get(Event, t.event_id)
            t.status, t.done_at = "done", now
            if e is None or e.lifecycle_status == "RESOLVED":
                t.status = "cancelled"
                return False
            if t.kind == "ack_timeout":
                if e.lifecycle_status != "OPEN":
                    return False
                contacts = self._contacts(s, e.subject_id)
                step = t.step
                ntype = "escalation" if e.decision_level == "EMERGENCY" else "verification"
                if step < len(contacts):
                    e.escalation_step = step
                    self.notifier.enqueue(s, e, ntype, step, [contacts[step]], now)
                    self._transition(s, e, "auto_escalation", "system", "scheduler", reason=f"not acknowledged within {self._ack_timeout(e.decision_level):.0f}s", payload={"to_step": step, "contact_id": contacts[step].id})
                    self._schedule(s, "ack_timeout", e, step + 1, now + timedelta(seconds=self._ack_timeout(e.decision_level)))
                else:
                    self._schedule(s, "repeat_reminder", e, 1, now)
                e.version += 1
                e.updated_at = now
                published = e
            elif t.kind == "repeat_reminder":
                if e.lifecycle_status != "OPEN":
                    return False
                contacts = self._contacts(s, e.subject_id)
                self.notifier.enqueue(s, e, "reminder", 100 + t.step, contacts, now)
                self._transition(s, e, "reminder_all_contacts", "system", "scheduler", payload={"round": t.step})
                if t.step < self.cfg.escalation.max_reminders:
                    self._schedule(s, "repeat_reminder", e, t.step + 1, now + timedelta(seconds=self.cfg.escalation.reminder_interval_s))
                published = e
            elif t.kind in ("handling_reminder", "handling_followup"):
                want = "ACKNOWLEDGED" if t.kind == "handling_reminder" else "IN_PROGRESS"
                if e.lifecycle_status != want:
                    return False
                uid = (t.payload or {}).get("user_id")
                contacts = [c for c in self._contacts(s, e.subject_id) if c.user_id == uid] or self._notified_contacts(s, e)
                self.notifier.enqueue(s, e, "reminder", 200 + t.step, contacts, now)
                self._transition(s, e, t.kind, "system", "scheduler", payload={"round": t.step})
                if t.step + 1 < self.cfg.escalation.max_reminders:
                    self._schedule(s, t.kind, e, t.step + 1, now + timedelta(seconds=self.cfg.escalation.reminder_interval_s), t.payload)
                published = e
        if published is not None:
            self._publish(published, "timer")
        return True

    # ---------------------------------------------------------------- reads
    def detail(self, event_id: str) -> dict[str, Any]:
        with self.db.session() as s:
            e = s.get(Event, event_id)
            if e is None:
                raise EventError("not_found", "event not found", 404)
            d = event_to_dict(e, detail=True)
            d["evidence"] = [
                {
                    "decision_id": x.decision_id, "channel": x.channel, "available": x.available, "candidate": x.candidate,
                    "contradicts": x.contradicts, "score": x.score, "score_semantics": x.score_semantics, "quality_score": x.quality_score,
                    "time_start_at": iso(x.time_start_at), "time_end_at": iso(x.time_end_at), "peak_at": iso(x.peak_at),
                    "features": x.features, "reason_codes": x.reason_codes, "quality_flags": x.quality_flags,
                    "model": f"{x.model_id}@{x.model_version}", "calibration_id": x.calibration_id, "created_at": iso(x.created_at),
                }
                for x in s.execute(select(EvidenceSnapshot).where(EvidenceSnapshot.event_id == e.id).order_by(EvidenceSnapshot.id)).scalars()
            ]
            d["timeline"] = [
                {
                    "seq": t.seq, "action": t.action, "from_level": t.from_level, "to_level": t.to_level, "from_status": t.from_status,
                    "to_status": t.to_status, "actor_type": t.actor_type, "actor_id": t.actor_id, "reason": t.reason, "payload": t.payload,
                    "at": iso(t.created_at),
                }
                for t in s.execute(select(EventTransition).where(EventTransition.event_id == e.id).order_by(EventTransition.seq)).scalars()
            ]
            contacts = {c.id: c.name for c in s.execute(select(Contact).where(Contact.subject_id == e.subject_id)).scalars()}
            d["notifications"] = [notification_to_dict(n, contacts) for n in s.execute(select(Notification).where(Notification.event_id == e.id).order_by(Notification.id)).scalars()]
            d["tasks"] = [
                {"kind": t.kind, "step": t.step, "due_at": iso(t.due_at), "status": t.status}
                for t in s.execute(select(ScheduledTask).where(ScheduledTask.event_id == e.id).order_by(ScheduledTask.id)).scalars()
            ]
            return d


def notification_to_dict(n: Notification, contacts: dict[int, str] | None = None) -> dict[str, Any]:
    return {
        "id": n.id, "event_id": n.event_id, "type": n.notification_type, "escalation_step": n.escalation_step,
        "recipient_contact_id": n.recipient_contact_id, "recipient": (contacts or {}).get(n.recipient_contact_id or -1),
        "recipient_user_id": n.recipient_user_id, "channel": n.channel, "status": n.status, "queued_at": iso(n.queued_at),
        "sending_at": iso(n.sending_at), "provider_accepted_at": iso(n.provider_accepted_at), "delivered_at": iso(n.delivered_at),
        "delivered_source": n.delivered_source, "viewed_at": iso(n.viewed_at), "acknowledged_at": iso(n.acknowledged_at),
        "failed_at": iso(n.failed_at), "expired_at": iso(n.expired_at), "attempts": n.attempts, "last_error": n.last_error,
        "delayed_delivery": n.delayed_delivery, "delay_seconds": n.delay_seconds, "is_test": n.is_test,
        "original_occurrence_at": iso(n.original_occurrence_at),
    }
