"""Realtime fan-out to authenticated WebSocket clients.

Important messages (events, notifications, coverage changes) are written to
``realtime_messages`` first so a reconnecting client can catch up from its
last cursor; ephemeral updates (live counts, signal previews) may be dropped.
Thread-safe: publish() can be called from runtime/dispatcher threads.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select

from homeease.backend.storage.db import Database
from homeease.backend.storage.models import RealtimeMessage
from homeease.core.clock import Clock, iso
from homeease.core.util import get_logger, new_id

log = get_logger(__name__)


@dataclass
class Client:
    cid: int
    ws: Any
    user_id: int
    role: str
    subject_id: str | None
    queue: "asyncio.Queue[dict[str, Any]]" = field(default_factory=lambda: asyncio.Queue(maxsize=500))


def visible_to(audience: str, role: str) -> bool:
    if audience == "all":
        return True
    if audience == "caregiver":
        return role in ("caregiver", "admin")
    if audience == "elder":
        return role in ("elder", "admin", "caregiver")
    if audience == "admin":
        return role == "admin"
    return False


class Broadcaster:
    def __init__(self, db: Database | None, clock: Clock) -> None:
        self.db = db
        self.clock = clock
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: dict[int, Client] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.sink: list[dict[str, Any]] | None = None  # headless/test capture

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def envelope(self, type_: str, payload: dict[str, Any], subject_id: str | None, source_ts: datetime | None = None, cursor: int | None = None) -> dict[str, Any]:
        now = self.clock.utc_now()
        return {
            "message_id": new_id("m_"),
            "type": type_,
            "schema_version": "1.0",
            "source_timestamp": iso(source_ts or now),
            "server_timestamp": iso(now),
            "subject_id": subject_id,
            "cursor": cursor,
            "payload": payload,
        }

    def publish(self, type_: str, payload: dict[str, Any], subject_id: str | None = None, audience: str = "caregiver", durable: bool = False, source_ts: datetime | None = None) -> dict[str, Any]:
        cursor = None
        if durable and self.db is not None:
            with self.db.session() as s:
                row = RealtimeMessage(type=type_, subject_id=subject_id, audience=audience, payload=payload, created_at=self.clock.utc_now())
                s.add(row)
                s.flush()
                cursor = row.id
        msg = self.envelope(type_, payload, subject_id, source_ts, cursor)
        msg["_audience"] = audience
        if self.sink is not None:
            self.sink.append(msg)
        self._fanout(msg)
        return msg

    def _fanout(self, msg: dict[str, Any], only_user: int | None = None) -> int:
        n = 0
        with self._lock:
            clients = list(self.clients.values())
        for c in clients:
            if only_user is not None and c.user_id != only_user:
                continue
            if not visible_to(msg.get("_audience", "caregiver"), c.role):
                continue
            out = {k: v for k, v in msg.items() if not k.startswith("_")}
            if self.loop is None:
                continue
            try:
                self.loop.call_soon_threadsafe(self._put, c, out)
                n += 1
            except RuntimeError:
                pass
        return n

    @staticmethod
    def _put(c: Client, msg: dict[str, Any]) -> None:
        try:
            c.queue.put_nowait(msg)
        except asyncio.QueueFull:
            # Drop the oldest ephemeral message to keep the newest state.
            try:
                c.queue.get_nowait()
                c.queue.put_nowait(msg)
            except Exception:  # pragma: no cover
                pass

    def send_to_user(self, user_id: int, type_: str, payload: dict[str, Any], subject_id: str | None = None) -> int:
        msg = self.envelope(type_, payload, subject_id)
        msg["_audience"] = "all"
        if self.sink is not None:
            self.sink.append({**msg, "_user": user_id})
        return self._fanout(msg, only_user=user_id)

    def connected_users(self) -> set[int]:
        with self._lock:
            return {c.user_id for c in self.clients.values()}

    def register(self, ws: Any, user_id: int, role: str, subject_id: str | None) -> Client:
        c = Client(next(self._ids), ws, user_id, role, subject_id)
        with self._lock:
            self.clients[c.cid] = c
        return c

    def unregister(self, c: Client) -> None:
        with self._lock:
            self.clients.pop(c.cid, None)

    def catch_up(self, after: int, role: str, limit: int = 200) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        with self.db.session() as s:
            rows = s.execute(select(RealtimeMessage).where(RealtimeMessage.id > after).order_by(RealtimeMessage.id).limit(limit)).scalars().all()
            out = []
            for r in rows:
                if visible_to(r.audience, role):
                    m = self.envelope(r.type, r.payload, r.subject_id, r.created_at, r.id)
                    m["replayed"] = True
                    out.append(m)
            return out
