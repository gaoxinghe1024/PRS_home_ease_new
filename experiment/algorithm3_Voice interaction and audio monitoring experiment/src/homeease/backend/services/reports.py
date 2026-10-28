"""User-facing reports: per-event, per-exercise-session and period summaries.

Rules: an unlabelled emergency is not a confirmed fall; periods without
monitoring are reported separately and never counted as "safe time"; missing
days are shown as missing, never filled.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select

from homeease.backend.services.events import EventService
from homeease.backend.services.exercise import ExerciseService
from homeease.backend.storage.db import Database
from homeease.backend.storage.models import Event, ExerciseSession, HealthInterval
from homeease.core.clock import Clock, ensure_utc, iso


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    return ensure_utc(datetime.fromisoformat(ts.replace("Z", "+00:00")))


def _secs(a: str | None, b: str | None) -> float | None:
    da, db_ = _parse(a), _parse(b)
    if da is None or db_ is None:
        return None
    return round((db_ - da).total_seconds(), 2)


class ReportService:
    def __init__(self, db: Database, clock: Clock, events: EventService, exercise: ExerciseService, tz: str = "Asia/Singapore") -> None:
        self.db = db
        self.clock = clock
        self.events = events
        self.exercise = exercise
        self.tz = tz

    # ---------------------------------------------------------------- event
    def event_report(self, event_id: str) -> dict[str, Any]:
        d = self.events.detail(event_id)
        tl = d["timeline"]
        first = lambda action: next((t["at"] for t in tl if t["action"] == action), None)  # noqa: E731
        notifs = d["notifications"]
        first_sent = min((n["provider_accepted_at"] for n in notifs if n["provider_accepted_at"]), default=None)
        first_delivered = min((n["delivered_at"] for n in notifs if n["delivered_at"]), default=None)
        first_viewed = min((n["viewed_at"] for n in notifs if n["viewed_at"]), default=None)
        ack = first("acknowledge") or first("start_handling")
        ref = d["occurrence_est_at"] or d["candidate_at"]
        channels = {}
        for ev in d["evidence"]:
            channels.setdefault(ev["channel"], ev)
        return {
            "report_type": "fall_event",
            "generated_at": iso(self.clock.utc_now()),
            "event": {k: d[k] for k in ("id", "subject_id", "room_id", "source_mode", "origin", "decision_level", "max_level", "lifecycle_status", "outcome", "outcome_note", "is_test")},
            "times": {
                "occurrence_estimate": d["occurrence_est_at"],
                "occurrence_source": d["occurrence_source"],
                "occurrence_note": "automatic estimate from sensor evidence, not ground truth" if d["origin"] == "machine" else "time of help request",
                "candidate": d["candidate_at"],
                "classified": d["classified_at"],
                "first_provider_accepted": first_sent,
                "first_endpoint_delivery": first_delivered,
                "first_viewed": first_viewed,
                "acknowledged": ack,
                "resolved": d["resolved_at"],
            },
            "durations_s": {
                "estimate_to_classification": _secs(ref, d["classified_at"]),
                "classification_to_first_delivery": _secs(d["classified_at"], first_delivered),
                "estimate_to_first_delivery": _secs(ref, first_delivered),
                "classification_to_acknowledge": _secs(d["classified_at"], ack),
                "classification_to_resolution": _secs(d["classified_at"], d["resolved_at"]),
            },
            "channels": {
                k: {"available": v["available"], "candidate": v["candidate"], "contradicts": v["contradicts"], "score": round(v["score"], 3), "score_semantics": v["score_semantics"], "quality": round(v["quality_score"], 3), "reasons": v["reason_codes"], "quality_flags": v["quality_flags"], "model": v["model"]}
                for k, v in channels.items()
            },
            "evidence_summary": d["summary"],
            "models": {"versions": d["model_versions"], "fusion_backend": d["fusion_backend"], "thresholds_version": d["thresholds_version"]},
            "voice_inquiry": d.get("voice_inquiry"),
            "timeline": tl,
            "notifications": notifs,
            "uncertainty": [
                "Occurrence time is an automatic estimate." if d["origin"] == "machine" else "Help request time recorded by the device.",
                "DELIVERED means an end-point receipt was received; PROVIDER_ACCEPTED alone does not prove receipt.",
                "A caregiver outcome is feedback, not a scientific ground truth label.",
            ],
        }

    def event_report_csv(self, event_id: str) -> str:
        r = self.event_report(event_id)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["section", "key", "value"])
        for k, v in r["event"].items():
            w.writerow(["event", k, v])
        for k, v in r["times"].items():
            w.writerow(["times", k, v])
        for k, v in r["durations_s"].items():
            w.writerow(["durations_s", k, v])
        for t in r["timeline"]:
            w.writerow(["timeline", t["at"], f"{t['action']} {t.get('from_status') or ''}->{t.get('to_status') or ''} {t.get('to_level') or ''} by {t['actor_type']}"])
        for n in r["notifications"]:
            w.writerow(["notification", n["id"], f"{n['type']} via {n['channel']} to {n['recipient']}: {n['status']} delivered_at={n['delivered_at']}"])
        return buf.getvalue()

    # ------------------------------------------------------------- exercise
    def exercise_report(self, session_id: str) -> dict[str, Any]:
        d = self.exercise.detail(session_id)
        return {
            "report_type": "exercise_session",
            "generated_at": iso(self.clock.utc_now()),
            "session": d,
            "counting_definition": d["repetition_definition"],
            "summary": {
                "valid_repetitions": d["count_valid"], "target": d["target_reps"], "target_reached": d["target_reached"],
                "total_s": d["total_s"], "effective_s": d["effective_s"], "paused_manual_s": d["paused_manual_s"], "paused_auto_s": d["paused_auto_s"],
                "rejected_cycles": sum(1 for c in d["cycles"] if not c["valid"]),
                "interruptions": len(d["interruptions"]),
            },
            "notes": [
                "Counts only complete cycles under the stated definition.",
                "Periods when counting was unavailable are listed as interruptions; they are not filled in.",
                "This is an activity record, not a medical or fall-risk assessment.",
            ],
        }

    # -------------------------------------------------------------- summary
    def summary(self, subject_id: str, start: datetime, end: datetime, include_test: bool = False) -> dict[str, Any]:
        start, end = ensure_utc(start), ensure_utc(end)
        assert start is not None and end is not None
        with self.db.session() as s:
            q = select(Event).where(Event.subject_id == subject_id, Event.classified_at >= start, Event.classified_at < end)
            if not include_test:
                q = q.where(Event.is_test.is_(False))
            evs = list(s.execute(q).scalars())
            outcomes: dict[str, int] = defaultdict(int)
            for e in evs:
                outcomes[e.outcome or ("open" if e.lifecycle_status != "RESOLVED" else "unlabelled")] += 1
            safety = {
                "events_total": len(evs),
                "emergency_events": sum(1 for e in evs if e.max_level == "EMERGENCY"),
                "verification_events": sum(1 for e in evs if e.max_level == "PENDING"),
                "help_requests": sum(1 for e in evs if e.origin == "help_request"),
                "confirmed_falls": outcomes.get("confirmed_fall", 0),
                "false_alarms": outcomes.get("false_alarm", 0),
                "safe_after_check": outcomes.get("safe_after_check", 0),
                "unresolved_outcome": outcomes.get("unresolved", 0),
                "still_open": outcomes.get("open", 0),
                "emergencies_without_confirmed_outcome": sum(1 for e in evs if e.max_level == "EMERGENCY" and e.outcome != "confirmed_fall"),
                "note": "An emergency without a caregiver-confirmed outcome is not counted as a confirmed fall.",
            }
            cov = defaultdict(float)
            rows = s.execute(select(HealthInterval).where(HealthInterval.subject_id == subject_id, HealthInterval.channel == "coverage", HealthInterval.start_at < end)).scalars()
            covered = 0.0
            for r in rows:
                a = max(ensure_utc(r.start_at), start)
                b = min(ensure_utc(r.end_at) or self.clock.utc_now(), end)
                if b > a:
                    sec = (b - a).total_seconds()
                    cov[r.status] += sec
                    covered += sec
            period = (end - start).total_seconds()
            coverage = {
                "active_s": round(cov.get("ACTIVE", 0.0), 1),
                "degraded_s": round(cov.get("DEGRADED", 0.0), 1),
                "unavailable_s": round(cov.get("UNAVAILABLE", 0.0), 1),
                "no_record_s": round(max(0.0, period - covered), 1),
                "note": "Unavailable or unrecorded time is not 'safe time': no falls can be detected then.",
            }
            ex = list(s.execute(select(ExerciseSession).where(ExerciseSession.subject_id == subject_id, ExerciseSession.started_at >= start, ExerciseSession.started_at < end).order_by(ExerciseSession.started_at)).scalars())
            tz = ZoneInfo(self.tz)
            per_day: dict[str, dict[str, Any]] = {}
            for x in ex:
                day = ensure_utc(x.started_at).astimezone(tz).date().isoformat()
                dd = per_day.setdefault(day, {"sessions": 0, "repetitions": 0, "effective_s": 0.0})
                dd["sessions"] += 1
                dd["repetitions"] += x.count_valid
                dd["effective_s"] += x.effective_s
            exercise = {
                "sessions": len(ex),
                "participation_days": len(per_day),
                "repetitions_total": sum(x.count_valid for x in ex),
                "by_activity": {"sit_to_stand": sum(x.count_valid for x in ex if x.activity == "sit_to_stand")},
                "per_day": [{"date": k, **v} for k, v in sorted(per_day.items())],
                "interrupted_sessions": sum(1 for x in ex if x.status == "INTERRUPTED"),
                "note": "Only days with recorded sessions are listed; missing days are not filled.",
            }
        return {
            "report_type": "summary",
            "subject_id": subject_id,
            "from": iso(start),
            "to": iso(end),
            "include_test_events": include_test,
            "generated_at": iso(self.clock.utc_now()),
            "safety": safety,
            "coverage": coverage,
            "exercise": exercise,
        }

    def summary_csv(self, rep: dict[str, Any]) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["section", "key", "value"])
        for sec in ("safety", "coverage"):
            for k, v in rep[sec].items():
                w.writerow([sec, k, v])
        for k, v in rep["exercise"].items():
            if k != "per_day":
                w.writerow(["exercise", k, v])
        for d in rep["exercise"]["per_day"]:
            w.writerow(["exercise_day", d["date"], f"sessions={d['sessions']} reps={d['repetitions']} effective_s={round(d['effective_s'], 1)}"])
        return buf.getvalue()


def default_period(clock: Clock, days: int = 7) -> tuple[datetime, datetime]:
    end = clock.utc_now()
    return end - timedelta(days=days), end
