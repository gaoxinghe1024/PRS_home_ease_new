"""Notification outbox, channel adapters and delivery-state semantics.

Status ladder (forward only):
  QUEUED -> SENDING -> PROVIDER_ACCEPTED -> DELIVERED -> VIEWED -> ACKNOWLEDGED
Terminal: FAILED, EXPIRED.

* PROVIDER_ACCEPTED: the provider/socket accepted the message (HTTP 201 from a
  push service, or the server wrote to an open WebSocket). NOT proof of receipt.
* DELIVERED: an end-point receipt arrived (WebSocket client ack, Service Worker
  receipt, or the test sink). ``delivered_source`` records which.
* Channels without receipts (generic webhook) keep delivered_at = NULL (unknown).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

import httpx
from sqlalchemy import and_, or_, select

from homeease.backend.services.messages import compose
from homeease.backend.services.realtime import Broadcaster
from homeease.backend.services.security import new_token, token_hash
from homeease.backend.storage.db import Database
from homeease.backend.storage.models import Contact, Event, Notification, PushSubscription
from homeease.core.clock import Clock, ensure_utc, iso
from homeease.core.config import AppConfig
from homeease.core.paths import secrets_dir
from homeease.core.util import get_logger

log = get_logger(__name__)

ORDER = ["QUEUED", "SENDING", "PROVIDER_ACCEPTED", "DELIVERED", "VIEWED", "ACKNOWLEDGED"]
TERMINAL = {"FAILED", "EXPIRED"}


def _rank(s: str) -> int:
    return ORDER.index(s) if s in ORDER else -1


@dataclass
class SendResult:
    state: str  # provider_accepted | delivered | no_recipient | failed | disabled
    retryable: bool = True
    error: str | None = None
    delivered_source: str | None = None


class ChannelAdapter(Protocol):
    name: str

    def enabled(self) -> bool: ...

    def send(self, n: dict[str, Any], payload: dict[str, Any]) -> SendResult: ...


class InAppChannel:
    name = "inapp"

    def __init__(self, cfg: AppConfig, broadcaster: Broadcaster) -> None:
        self.cfg = cfg
        self.b = broadcaster

    def enabled(self) -> bool:
        return self.cfg.notifications.inapp.enabled

    def send(self, n: dict[str, Any], payload: dict[str, Any]) -> SendResult:
        uid = n.get("recipient_user_id")
        if uid is None:
            return SendResult("failed", retryable=False, error="contact has no user account for in-app delivery")
        sent = self.b.send_to_user(uid, "notification", payload, payload.get("subject_id"))
        if sent == 0:
            return SendResult("no_recipient", error="no open caregiver page for this user")
        return SendResult("provider_accepted")


class WebPushChannel:
    name = "webpush"

    def __init__(self, cfg: AppConfig, db: Database) -> None:
        self.cfg = cfg
        self.db = db

    def enabled(self) -> bool:
        return self.cfg.notifications.webpush.enabled and (secrets_dir() / "vapid_private.pem").is_file()

    def send(self, n: dict[str, Any], payload: dict[str, Any]) -> SendResult:
        from pywebpush import WebPushException, webpush

        uid = n.get("recipient_user_id")
        with self.db.session() as s:
            subs = s.execute(select(PushSubscription).where(PushSubscription.user_id == uid, PushSubscription.revoked_at.is_(None))).scalars().all()
            subs_data = [(x.id, x.endpoint, x.keys) for x in subs]
        if not subs_data:
            return SendResult("failed", retryable=False, error="no web push subscription")
        accepted, errors = 0, []
        for sid, endpoint, keys in subs_data:
            try:
                resp = webpush(
                    subscription_info={"endpoint": endpoint, "keys": keys},
                    data=json.dumps(payload),
                    vapid_private_key=str(secrets_dir() / "vapid_private.pem"),
                    vapid_claims={"sub": self.cfg.notifications.webpush.vapid_subject},
                    ttl=int(self.cfg.escalation.notification_ttl_s),
                    timeout=10,
                )
                if getattr(resp, "status_code", 201) in (200, 201, 202):
                    accepted += 1
            except WebPushException as e:  # pragma: no cover - needs a push service
                code = getattr(getattr(e, "response", None), "status_code", None)
                errors.append(f"{code}:{e}")
                if code in (404, 410):
                    with self.db.session() as s2:
                        row = s2.get(PushSubscription, sid)
                        if row is not None:
                            row.revoked_at = datetime.now().astimezone()
            except Exception as e:  # pragma: no cover
                errors.append(str(e))
        if accepted:
            return SendResult("provider_accepted")
        return SendResult("failed", retryable=True, error="; ".join(errors)[:500])


class WebhookChannel:
    name = "webhook"

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    def enabled(self) -> bool:
        return self.cfg.notifications.webhook.enabled

    def send(self, n: dict[str, Any], payload: dict[str, Any]) -> SendResult:
        url = n.get("webhook_url") or self.cfg.notifications.webhook.url
        if not url:
            return SendResult("failed", retryable=False, error="no webhook url configured")
        body = {k: v for k, v in payload.items() if k != "receipt_token"}
        try:
            r = httpx.post(url, json=body, timeout=self.cfg.notifications.webhook.timeout_s)
            if 200 <= r.status_code < 300:
                return SendResult("provider_accepted")
            return SendResult("failed", retryable=r.status_code >= 500 or r.status_code == 429, error=f"HTTP {r.status_code}")
        except httpx.HTTPError as e:
            return SendResult("failed", retryable=True, error=str(e)[:300])


class TestSinkChannel:
    """Explicit test receiver for development, replay and automated tests."""

    name = "test_sink"

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.received: list[dict[str, Any]] = []
        self.fail_next = 0
        self._lock = threading.Lock()

    def enabled(self) -> bool:
        return self.cfg.notifications.test_sink.enabled

    def send(self, n: dict[str, Any], payload: dict[str, Any]) -> SendResult:
        with self._lock:
            if self.fail_next > 0:
                self.fail_next -= 1
                return SendResult("failed", retryable=True, error="simulated provider failure")
            self.received.append({"notification": n, "payload": payload})
        return SendResult("delivered", delivered_source="test_sink")


class NotificationService:
    def __init__(self, db: Database, cfg: AppConfig, clock: Clock, broadcaster: Broadcaster) -> None:
        self.db = db
        self.cfg = cfg
        self.clock = clock
        self.broadcaster = broadcaster
        self.test_sink = TestSinkChannel(cfg)
        self.channels: dict[str, ChannelAdapter] = {
            "inapp": InAppChannel(cfg, broadcaster),
            "webpush": WebPushChannel(cfg, db),
            "webhook": WebhookChannel(cfg),
            "test_sink": self.test_sink,
        }
        self._kick = threading.Event()

    # ----------------------------------------------------------- enqueue
    def channels_for(self, contact: Contact, is_test_event: bool) -> list[str]:
        wanted = list(contact.channels or ["inapp"])
        if self.cfg.notifications.test_sink.enabled and "test_sink" not in wanted:
            wanted.append("test_sink")
        out = []
        for ch in wanted:
            ad = self.channels.get(ch)
            if ad is None or not ad.enabled():
                continue
            if is_test_event and ch in ("webpush", "webhook") and not self.cfg.notifications.test_events_external:
                # Simulated/replayed events never reach real external contacts by default.
                continue
            out.append(ch)
        return out

    def enqueue(
        self,
        s: Any,
        event: Event,
        ntype: str,
        step: int,
        contacts: list[Contact],
        now: datetime,
        is_test: bool | None = None,
    ) -> list[Notification]:
        if is_test is None:
            is_test = event.is_test or event.source_mode != "live"
        created = []
        ttl = timedelta(seconds=self.cfg.escalation.notification_ttl_s)
        for c in contacts:
            for ch in self.channels_for(c, bool(is_test)):
                key = f"{event.id}|{ntype}|{step}|{c.id}|{ch}"
                if s.execute(select(Notification.id).where(Notification.idem_key == key)).first():
                    continue
                n = Notification(
                    event_id=event.id, notification_type=ntype, escalation_step=step, recipient_contact_id=c.id,
                    recipient_user_id=c.user_id, channel=ch, status="QUEUED", payload={"webhook_url": c.webhook_url},
                    queued_at=now, next_attempt_at=now, expires_at=now + ttl,
                    original_occurrence_at=event.occurrence_est_at or event.classified_at, is_test=bool(is_test), idem_key=key,
                )
                s.add(n)
                created.append(n)
        s.flush()
        if created:
            self._kick.set()
        return created

    def enqueue_test(self, s: Any, contact: Contact, now: datetime) -> list[Notification]:
        out = []
        for ch in self.channels_for(contact, False):
            key = f"test|{contact.id}|{ch}|{now.isoformat()}"
            n = Notification(
                event_id=None, notification_type="test", escalation_step=0, recipient_contact_id=contact.id,
                recipient_user_id=contact.user_id, channel=ch, status="QUEUED", payload={"webhook_url": contact.webhook_url},
                queued_at=now, next_attempt_at=now, expires_at=now + timedelta(minutes=10), is_test=True, idem_key=key,
            )
            s.add(n)
            out.append(n)
        s.flush()
        self._kick.set()
        return out

    def cancel_pending(self, s: Any, event_id: str, now: datetime, reason: str) -> int:
        rows = s.execute(
            select(Notification).where(Notification.event_id == event_id, Notification.status.in_(["QUEUED"]), Notification.notification_type != "resolved")
        ).scalars().all()
        for n in rows:
            n.status = "EXPIRED"
            n.expired_at = now
            n.last_error = reason
        return len(rows)

    # -------------------------------------------------------------- send
    def _payload(self, n: Notification, event: Event | None, receipt_token: str, now: datetime) -> dict[str, Any]:
        tz = "Asia/Singapore"
        delayed_s = None
        if n.original_occurrence_at is not None:
            age = (now - ensure_utc(n.original_occurrence_at)).total_seconds()
            if age > self.cfg.escalation.delayed_delivery_threshold_s:
                delayed_s = age
        if event is None:
            text_en = {"title": "HomeEase test notification", "body": "This is a test. No event occurred."}
            text_zh = {"title": "HomeEase \u6d4b\u8bd5\u901a\u77e5", "body": "\u8fd9\u662f\u4e00\u6761\u6d4b\u8bd5\u901a\u77e5\uff0c\u6ca1\u6709\u53d1\u751f\u4e8b\u4ef6\u3002"}
            ev = {}
        else:
            ev = {
                "decision_level": event.decision_level, "origin": event.origin, "room_id": event.room_id,
                "occurred_at": ensure_utc(event.occurrence_est_at) or ensure_utc(event.classified_at),
                "classified_at": ensure_utc(event.classified_at), "summary": event.summary, "is_test": event.is_test,
                "source_mode": event.source_mode, "outcome": event.outcome,
            }
            text_en = compose(ev, n.notification_type, "en", tz, delayed_s)
            text_zh = compose(ev, n.notification_type, "zh", tz, delayed_s)
        return {
            "notification_id": n.id,
            "event_id": n.event_id,
            "subject_id": event.subject_id if event else None,
            "type": n.notification_type,
            "channel": n.channel,
            "level": event.decision_level if event else None,
            "lifecycle_status": event.lifecycle_status if event else None,
            "escalation_step": n.escalation_step,
            "title": text_en["title"],
            "body": text_en["body"],
            "title_zh": text_zh["title"],
            "body_zh": text_zh["body"],
            "occurred_at": iso(ev.get("occurred_at")) if event else None,
            "delayed_delivery": delayed_s is not None,
            "delay_seconds": round(delayed_s, 1) if delayed_s else None,
            "is_test": n.is_test,
            "url": f"/caregiver/event.html?id={n.event_id}" if n.event_id else "/caregiver/",
            "receipt_token": receipt_token,
            "sent_at": iso(now),
        }

    def kick(self) -> None:
        self._kick.set()

    def wait_kick(self, timeout: float) -> None:
        self._kick.wait(timeout)
        self._kick.clear()

    def dispatch_due(self, only_user: int | None = None) -> int:
        now = self.clock.utc_now()
        work: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []
        with self.db.tx() as s:
            stale = now - timedelta(seconds=30)
            q = select(Notification).where(
                or_(
                    and_(Notification.status == "QUEUED", or_(Notification.next_attempt_at.is_(None), Notification.next_attempt_at <= now)),
                    and_(Notification.status == "SENDING", Notification.sending_at <= stale),
                )
            )
            if only_user is not None:
                q = q.where(Notification.recipient_user_id == only_user)
            rows = s.execute(q.order_by(Notification.id).limit(100)).scalars().all()
            for n in rows:
                if n.expires_at is not None and ensure_utc(n.expires_at) <= now:
                    n.status, n.expired_at, n.last_error = "EXPIRED", now, "ttl_elapsed"
                    continue
                event = s.get(Event, n.event_id) if n.event_id else None
                if event is not None and event.lifecycle_status == "RESOLVED" and n.notification_type in ("escalation", "reminder", "verification", "emergency"):
                    n.status, n.expired_at, n.last_error = "EXPIRED", now, "event_resolved_before_send"
                    continue
                token = new_token(16)
                n.receipt_token_hash = token_hash(token)
                n.status, n.sending_at = "SENDING", now
                n.attempts += 1
                if n.original_occurrence_at is not None:
                    age = (now - ensure_utc(n.original_occurrence_at)).total_seconds()
                    if age > self.cfg.escalation.delayed_delivery_threshold_s:
                        n.delayed_delivery, n.delay_seconds = True, round(age, 1)
                payload = self._payload(n, event, token, now)
                meta = {"recipient_user_id": n.recipient_user_id, "webhook_url": (n.payload or {}).get("webhook_url"), "notification_id": n.id}
                work.append((n.id, n.channel, meta, payload))
        sent = 0
        for nid, ch, meta, payload in work:
            ad = self.channels.get(ch)
            if ad is None or not ad.enabled():
                res = SendResult("failed", retryable=False, error=f"channel {ch} disabled")
            else:
                try:
                    res = ad.send(meta, payload)
                except Exception as e:  # never let one channel break the loop
                    log.exception("channel send failed")
                    res = SendResult("failed", retryable=True, error=str(e)[:300])
            self._apply_result(nid, res, payload)
            sent += 1
        return sent

    def _apply_result(self, nid: int, res: SendResult, payload: dict[str, Any]) -> None:
        now = self.clock.utc_now()
        e = self.cfg.escalation
        with self.db.tx() as s:
            n = s.get(Notification, nid)
            if n is None or n.status in TERMINAL or _rank(n.status) > _rank("SENDING"):
                return
            if res.state == "provider_accepted":
                n.status, n.provider_accepted_at = "PROVIDER_ACCEPTED", now
            elif res.state == "delivered":
                n.status, n.provider_accepted_at, n.delivered_at, n.delivered_source = "DELIVERED", now, now, res.delivered_source
            elif res.state == "no_recipient":
                n.status = "QUEUED"
                n.attempts = max(0, n.attempts - 1)
                n.next_attempt_at = now + timedelta(seconds=5)
                n.last_error = res.error
            else:
                n.last_error = res.error
                if res.retryable and n.attempts < e.retry_max_attempts:
                    delay = min(e.retry_base_s * (2 ** (n.attempts - 1)), e.retry_max_s)
                    n.status, n.next_attempt_at = "QUEUED", now + timedelta(seconds=delay)
                else:
                    n.status, n.failed_at = "FAILED", now
            status = n.status
            event_id = n.event_id
        self.broadcaster.publish("notification_status", {"notification_id": nid, "event_id": event_id, "status": status, "channel": payload.get("channel")}, payload.get("subject_id"), durable=False)

    # ----------------------------------------------------------- receipts
    def receipt(self, nid: int, kind: str, *, user_id: int | None = None, receipt_token: str | None = None, client_ts: datetime | None = None, source: str = "client") -> Notification:
        if kind not in ("delivered", "viewed"):
            raise ValueError("kind must be delivered or viewed")
        now = self.clock.utc_now()
        with self.db.tx() as s:
            n = s.get(Notification, nid)
            if n is None:
                raise KeyError(nid)
            ok = (receipt_token is not None and n.receipt_token_hash == token_hash(receipt_token)) or (user_id is not None and n.recipient_user_id == user_id)
            if not ok:
                raise PermissionError("receipt not authorised")
            if n.status in TERMINAL:
                return n
            if kind == "delivered" and _rank(n.status) < _rank("DELIVERED"):
                n.status, n.delivered_at, n.delivered_source = "DELIVERED", now, source
                n.client_reported_at = client_ts
            elif kind == "viewed":
                if n.delivered_at is None:
                    n.delivered_at, n.delivered_source = now, source
                if _rank(n.status) < _rank("VIEWED"):
                    n.status, n.viewed_at = "VIEWED", now
            return n

    def mark_viewed_for_event(self, s: Any, event_id: str, user_id: int, now: datetime) -> None:
        rows = s.execute(select(Notification).where(Notification.event_id == event_id, Notification.recipient_user_id == user_id)).scalars().all()
        for n in rows:
            if n.status in TERMINAL or n.status == "QUEUED":
                continue
            if _rank(n.status) < _rank("VIEWED"):
                if n.delivered_at is None:
                    n.delivered_at, n.delivered_source = now, "event_page_opened"
                n.status, n.viewed_at = "VIEWED", now

    def mark_acknowledged(self, s: Any, event_id: str, user_id: int, now: datetime) -> None:
        rows = s.execute(select(Notification).where(Notification.event_id == event_id, Notification.recipient_user_id == user_id)).scalars().all()
        for n in rows:
            if n.status in TERMINAL or n.status == "QUEUED":
                continue
            if n.delivered_at is None:
                n.delivered_at, n.delivered_source = now, "user_action"
            if n.viewed_at is None:
                n.viewed_at = now
            n.status, n.acknowledged_at = "ACKNOWLEDGED", now

    def recover(self) -> int:
        """After a restart, SENDING rows are retried (not silently dropped)."""
        now = self.clock.utc_now()
        with self.db.tx() as s:
            rows = s.execute(select(Notification).where(Notification.status == "SENDING")).scalars().all()
            for n in rows:
                n.status, n.next_attempt_at, n.last_error = "QUEUED", now, "recovered_after_restart"
            return len(rows)
