"""Device pairing (time-limited one-time codes) and device credentials."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import select

from homeease.backend.services.security import new_pairing_code, new_token, token_hash
from homeease.backend.storage.db import Database
from homeease.backend.storage.models import Device, PairingCode
from homeease.core.clock import Clock, ensure_utc, iso
from homeease.core.util import new_id

PAIRING_TTL_S = 600
MAX_CODE_ATTEMPTS = 5


class PairingError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def device_to_dict(d: Device) -> dict[str, Any]:
    return {
        "id": d.id, "subject_id": d.subject_id, "kind": d.kind, "name": d.name, "platform": d.platform,
        "collector_type": d.collector_type, "placement": d.placement, "paired_at": iso(d.paired_at), "revoked": d.revoked_at is not None,
        "last_seen_at": iso(d.last_seen_at), "last_data_at": iso(d.last_data_at), "meta": d.meta,
    }


class DeviceService:
    def __init__(self, db: Database, clock: Clock) -> None:
        self.db = db
        self.clock = clock
        self._touch_cache: dict[str, float] = {}

    def create_pairing(self, subject_id: str, kind: str, user_id: int | None) -> dict[str, Any]:
        if kind not in ("phone",):
            raise PairingError("invalid_kind", "only phone collectors pair with a code")
        now = self.clock.utc_now()
        code = new_pairing_code()
        with self.db.tx() as s:
            s.add(PairingCode(code_hash=token_hash(code), subject_id=subject_id, kind=kind, created_by=user_id, created_at=now, expires_at=now + timedelta(seconds=PAIRING_TTL_S)))
        return {"code": code, "expires_at": iso(now + timedelta(seconds=PAIRING_TTL_S)), "ttl_s": PAIRING_TTL_S}

    def pair(self, code: str, name: str, platform: str | None, collector_type: str, placement: str) -> dict[str, Any]:
        now = self.clock.utc_now()
        code = (code or "").strip().upper().replace("-", "").replace(" ", "")
        with self.db.tx() as s:
            pc = s.execute(select(PairingCode).where(PairingCode.code_hash == token_hash(code))).scalar_one_or_none()
            if pc is None:
                raise PairingError("invalid_code", "pairing code not recognised", 400)
            if pc.used_at is not None:
                raise PairingError("code_used", "pairing code already used", 409)
            if ensure_utc(pc.expires_at) < now:
                raise PairingError("code_expired", "pairing code expired", 410)
            tok = new_token(32)
            dev = Device(
                id=new_id("phone-"), subject_id=pc.subject_id, kind=pc.kind, name=(name or "Phone")[:128], platform=(platform or "")[:128],
                collector_type=collector_type, placement=placement, token_hash=token_hash(tok), paired_at=now, meta={},
            )
            s.add(dev)
            pc.used_at = now
            pc.device_id = dev.id
            s.flush()
            return {"device_id": dev.id, "subject_id": dev.subject_id, "device_token": tok}

    def authenticate(self, tok: str | None) -> Device | None:
        if not tok:
            return None
        with self.db.session() as s:
            d = s.execute(select(Device).where(Device.token_hash == token_hash(tok), Device.revoked_at.is_(None))).scalar_one_or_none()
            if d is not None:
                s.expunge(d)
            return d

    def touch(self, device_id: str, data: bool = False, meta: dict[str, Any] | None = None) -> None:
        now = self.clock.utc_now()
        mono = self.clock.mono_ms()
        if not meta and mono - self._touch_cache.get(device_id, -1e9) < 2000:
            return  # throttle DB writes for high-rate ingestion
        self._touch_cache[device_id] = mono
        with self.db.session() as s:
            d = s.get(Device, device_id)
            if d is None:
                return
            d.last_seen_at = now
            if data:
                d.last_data_at = now
            if meta:
                d.meta = {**(d.meta or {}), **meta}
                if "placement" in meta:
                    d.placement = str(meta["placement"])

    def ensure_camera(self, subject_id: str, device_id: str, name: str, source: str) -> None:
        now = self.clock.utc_now()
        with self.db.session() as s:
            d = s.get(Device, device_id)
            if d is None:
                s.add(Device(id=device_id, subject_id=subject_id, kind="camera", name=name, platform="local", collector_type="opencv", placement="fixed", paired_at=now, meta={"source": source}))
            else:
                d.meta = {**(d.meta or {}), "source": source}

    def list(self, subject_id: str) -> list[dict[str, Any]]:
        with self.db.session() as s:
            return [device_to_dict(d) for d in s.execute(select(Device).where(Device.subject_id == subject_id).order_by(Device.kind, Device.paired_at)).scalars()]

    def revoke(self, device_id: str) -> None:
        with self.db.tx() as s:
            d = s.get(Device, device_id)
            if d is None:
                raise PairingError("not_found", "device not found", 404)
            d.revoked_at = self.clock.utc_now()
            d.token_hash = None
