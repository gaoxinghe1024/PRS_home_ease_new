"""Monitoring coverage (ACTIVE / DEGRADED / UNAVAILABLE), independent of events.

A device being online does not mean the person is visible; a still phone does
not mean it is not carried; ~1 g does not mean it is worn. Each channel is
usable only when its data are fresh, trustworthy and attributable to the
subject. Recovery requires continuous validity for ``recovery_stable_ms``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from homeease.core.config import AppConfig

ACTIVE, DEGRADED, UNAVAILABLE = "ACTIVE", "DEGRADED", "UNAVAILABLE"


@dataclass
class ChannelHealth:
    usable: bool = False
    reasons: list[str] = field(default_factory=list)
    last_valid_ms: float | None = None
    since_ms: float | None = None
    raw_ok_since: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"usable": self.usable, "reasons": self.reasons, "last_valid_ms": self.last_valid_ms, "since_ms": self.since_ms, "detail": self.detail}


@dataclass
class CoverageSnapshot:
    status: str
    reasons: list[str]
    channels: dict[str, ChannelHealth]
    since_ms: float
    changed: bool = False
    channel_changes: list[tuple[str, bool, list[str]]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": self.reasons,
            "since_ms": self.since_ms,
            "channels": {k: v.to_dict() for k, v in self.channels.items()},
        }


class CoverageEvaluator:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.channels = {"accel": ChannelHealth(), "vision": ChannelHealth()}
        self.status = UNAVAILABLE
        self.since_ms = 0.0
        self.reasons: list[str] = ["starting"]

    def _apply(self, name: str, raw_ok: bool, reasons: list[str], now: float, last_valid: float | None, detail: dict[str, Any]) -> bool:
        ch = self.channels[name]
        ch.detail = detail
        ch.last_valid_ms = last_valid
        changed = False
        stable = self.cfg.quality.recovery_stable_ms
        if raw_ok:
            if ch.raw_ok_since is None:
                ch.raw_ok_since = now
            if not ch.usable and now - ch.raw_ok_since >= stable:
                ch.usable, ch.since_ms, ch.reasons = True, now, []
                changed = True
            elif not ch.usable:
                ch.reasons = ["recovering"]
        else:
            ch.raw_ok_since = None
            if ch.usable or ch.reasons != reasons:
                changed = ch.usable or changed
                if ch.usable:
                    ch.since_ms = now
                ch.usable = False
                ch.reasons = reasons
        return changed

    def evaluate(self, now: float, accel: dict[str, Any], vision: dict[str, Any]) -> CoverageSnapshot:
        q = self.cfg.quality
        # ---------------------------------------------------------- accel
        ar: list[str] = []
        if not accel.get("has_device"):
            ar.append("phone_not_paired")
        else:
            last_recv = accel.get("last_recv_ms")
            last_sample = accel.get("last_sample_ms")
            if last_recv is None or now - last_recv > q.accel_stale_ms:
                ar.append("phone_no_recent_data")
            elif last_sample is None or now - last_sample > q.accel_stale_ms + 1000.0:
                ar.append("stale")
            rate = accel.get("rate_hz")
            if rate is not None and rate < q.accel_min_rate_hz:
                ar.append("sampling_rate_low")
            unc = accel.get("clock_uncertainty_ms")
            if unc is None or unc > q.max_clock_uncertainty_ms or accel.get("timestamp_untrusted"):
                ar.append("timestamp_untrusted")
            carriage = accel.get("carriage", "unknown")
            if carriage == "not_carried":
                ar.append("phone_not_carried")
            elif carriage == "unknown":
                ar.append("phone_carriage_unknown")
            if accel.get("page_hidden"):
                ar.append("collector_backgrounded")
            if accel.get("model_status", "ok") != "ok":
                ar.append("model_unavailable")
        a_changed = self._apply("accel", not ar, ar, now, accel.get("last_sample_ms"), {k: accel.get(k) for k in ("rate_hz", "clock_uncertainty_ms", "carriage", "placement")})
        # --------------------------------------------------------- vision
        vr: list[str] = []
        if not vision.get("camera_enabled", True):
            vr.append("camera_disabled")
        else:
            lf = vision.get("last_frame_ms")
            if vision.get("camera_error"):
                vr.append("camera_offline")
            elif lf is None or now - lf > q.camera_stale_ms:
                vr.append("camera_no_recent_frames")
            if vision.get("model_status", "ok") != "ok":
                vr.append("model_unavailable")
            lv = vision.get("last_target_ms")
            if not vr:
                if vision.get("target_ambiguous"):
                    vr.append("target_ambiguous")
                elif lv is None or now - lv > q.camera_stale_ms:
                    vr.append("person_not_visible")
                elif vision.get("low_pose_confidence"):
                    vr.append("low_pose_confidence")
            if vision.get("backlog"):
                vr.append("processing_backlog")
        v_changed = self._apply("vision", not vr, vr, now, vision.get("last_target_ms"), {k: vision.get(k) for k in ("fps", "backend", "person_count")})
        usable = [n for n, c in self.channels.items() if c.usable]
        status = ACTIVE if len(usable) == 2 else DEGRADED if usable else UNAVAILABLE
        reasons = sorted({f"{n}:{r}" for n, c in self.channels.items() for r in c.reasons if not c.usable})
        changed = status != self.status or reasons != self.reasons
        snap_changes: list[tuple[str, bool, list[str]]] = []
        if a_changed:
            snap_changes.append(("accel", self.channels["accel"].usable, self.channels["accel"].reasons))
        if v_changed:
            snap_changes.append(("vision", self.channels["vision"].usable, self.channels["vision"].reasons))
        if status != self.status:
            self.since_ms = now
        self.status, self.reasons = status, reasons
        return CoverageSnapshot(status, reasons, self.channels, self.since_ms, changed, snap_changes)
