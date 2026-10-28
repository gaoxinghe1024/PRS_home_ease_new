"""Persists coverage changes as health intervals and publishes them."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from homeease.backend.services.realtime import Broadcaster
from homeease.backend.storage.db import Database
from homeease.backend.storage.models import HealthInterval
from homeease.core.clock import Clock, iso
from homeease.edge.quality.coverage import CoverageSnapshot


class CoverageService:
    def __init__(self, db: Database, clock: Clock, broadcaster: Broadcaster, subject_id: str, source_mode: str) -> None:
        self.db = db
        self.clock = clock
        self.b = broadcaster
        self.subject_id = subject_id
        self.source_mode = source_mode
        self.current: dict[str, Any] = {"status": "UNAVAILABLE", "reasons": ["starting"], "channels": {}, "since": iso(clock.utc_now())}
        self.audio_current: dict[str, Any] = {"status": "off", "reasons": ["disabled"], "since": iso(clock.utc_now())}

    def _rotate(self, s: Any, channel: str, status: str, reasons: list[str]) -> None:
        now = self.clock.utc_now()
        open_rows = s.execute(select(HealthInterval).where(HealthInterval.subject_id == self.subject_id, HealthInterval.channel == channel, HealthInterval.end_at.is_(None))).scalars().all()
        for r in open_rows:
            if r.status == status and list(r.reasons or []) == list(reasons):
                return
            r.end_at = now
        s.add(HealthInterval(subject_id=self.subject_id, channel=channel, status=status, reasons=reasons, start_at=now, end_at=None, source_mode=self.source_mode))

    def record(self, snap: CoverageSnapshot) -> None:
        channels = {k: {"usable": v.usable, "reasons": v.reasons, "last_valid_at": iso(self.clock.mono_to_utc(v.last_valid_ms)) if v.last_valid_ms else None, "detail": v.detail} for k, v in snap.channels.items()}
        prev_status = self.current.get("status")
        self.current = {"status": snap.status, "reasons": snap.reasons, "channels": channels, "since": iso(self.clock.mono_to_utc(snap.since_ms)) if snap.since_ms else self.current.get("since")}
        if not (snap.changed or snap.channel_changes):
            return
        with self.db.tx() as s:
            if snap.changed:
                self._rotate(s, "coverage", snap.status, snap.reasons)
            for name, usable, reasons in snap.channel_changes:
                self._rotate(s, name, "usable" if usable else "unusable", list(reasons))
        durable = snap.status != prev_status
        self.b.publish("coverage_update", self.current, self.subject_id, audience="all", durable=durable)

    def record_audio(self, state: str, reasons: list[str]) -> None:
        """Health of the optional audio layer ('off' | 'ok' | 'degraded' + reasons) as its own interval series.

        Deliberately separate from the fall coverage status: a missing microphone or sound model is shown
        as degraded audio, it never makes the accelerometer/camera coverage look better or worse."""
        self.audio_current = {"status": state, "reasons": list(reasons), "since": iso(self.clock.utc_now())}
        with self.db.tx() as s:
            self._rotate(s, "audio", state, list(reasons))

    def close_all(self, reason: str = "node_stopped") -> None:
        now = self.clock.utc_now()
        with self.db.tx() as s:
            for r in s.execute(select(HealthInterval).where(HealthInterval.end_at.is_(None))).scalars():
                r.end_at = now
                r.reasons = list(r.reasons or []) + [reason]

    def recover(self) -> None:
        """At startup, intervals left open by a crash are closed (their true end is
        unknown; the restart time is used and marked)."""
        self.close_all("closed_at_restart")
