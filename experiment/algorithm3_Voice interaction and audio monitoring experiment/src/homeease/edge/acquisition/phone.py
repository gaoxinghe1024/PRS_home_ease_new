"""Acceleration batch ingestion: validation, de-duplication, late/out-of-order
handling and device-to-node time alignment.

Batches from a reconnecting phone that are older than ``late_batch_ms`` are
NOT fused with current camera evidence; they are kept as history and flagged.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from homeease.core.config import AppConfig
from homeease.edge.quality.clock_sync import SyncResult, estimate_sync, fixed_sync
from homeease.ml.common.signal import to_g
from homeease.schemas.observations import AccelBatch, TimeSyncReport


@dataclass
class DeviceStream:
    device_id: str
    session_id: str
    placement: str = "unknown"
    collector_type: str = "web_device_motion"
    includes_gravity: bool = True
    sync: SyncResult | None = None
    last_seq: int = -1
    seen: "OrderedDict[int, None]" = field(default_factory=OrderedDict)
    batches: int = 0
    samples: int = 0
    duplicates: int = 0
    late: int = 0
    out_of_order: int = 0
    untrusted: int = 0
    last_recv_node_ms: float | None = None
    last_aligned_ms: float | None = None
    latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    client_quality: dict[str, Any] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)
    history_batches: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=50))
    # recent syncs of this session by id: batches already in flight when the phone re-syncs still
    # carry the previous id and are mapped with the sync they were timestamped under
    recent_syncs: "OrderedDict[str, SyncResult]" = field(default_factory=OrderedDict)


@dataclass
class IngestResult:
    accepted: bool
    ack_seq: int
    duplicate: bool = False
    late: bool = False
    out_of_order: bool = False
    flags: list[str] = field(default_factory=list)
    t_node: np.ndarray | None = None
    axes_g: np.ndarray | None = None
    includes_gravity: bool = True
    clock_uncertainty_ms: float | None = None
    message: str = ""


class AccelIngestor:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.streams: dict[tuple[str, str], DeviceStream] = {}
        self._lock = threading.Lock()

    def stream(self, device_id: str, session_id: str) -> DeviceStream:
        key = (device_id, session_id)
        with self._lock:
            st = self.streams.get(key)
            if st is None:
                st = DeviceStream(device_id, session_id)
                self.streams[key] = st
                # keep only the latest few sessions per device
                same = [k for k in self.streams if k[0] == device_id]
                for k in same[:-3]:
                    self.streams.pop(k, None)
            return st

    def register_sync(self, device_id: str, report: TimeSyncReport, node_now_ms: float) -> SyncResult:
        st = self.stream(device_id, report.session_id)
        res = estimate_sync(report.sync_id, report.samples, node_now_ms, st.sync)
        st.sync = res
        st.recent_syncs[res.sync_id] = res
        while len(st.recent_syncs) > 5:
            st.recent_syncs.popitem(last=False)
        st.flags.discard("timestamp_untrusted")
        return res

    def set_fixed_sync(self, device_id: str, session_id: str, sync_id: str, offset_ms: float, node_now_ms: float, unc_ms: float = 0.0) -> SyncResult:
        st = self.stream(device_id, session_id)
        st.sync = fixed_sync(sync_id, offset_ms, node_now_ms, unc_ms)
        return st.sync

    def latest_stream(self, device_id: str | None = None) -> DeviceStream | None:
        cands = [s for s in self.streams.values() if device_id is None or s.device_id == device_id]
        cands = [s for s in cands if s.last_recv_node_ms is not None]
        if not cands:
            return None
        return max(cands, key=lambda s: s.last_recv_node_ms or 0.0)

    def ingest(self, batch: AccelBatch, node_now_ms: float) -> IngestResult:
        st = self.stream(batch.device_id, batch.session_id)
        st.placement = batch.placement
        st.collector_type = batch.collector_type
        st.includes_gravity = batch.includes_gravity
        if batch.client_quality:
            st.client_quality = dict(batch.client_quality)
        seq = batch.sequence_no
        if seq in st.seen:
            st.duplicates += 1
            return IngestResult(True, ack_seq=seq, duplicate=True, message="duplicate batch ignored")
        st.seen[seq] = None
        while len(st.seen) > 2000:
            st.seen.popitem(last=False)
        st.batches += 1
        st.samples += len(batch.samples)
        st.last_recv_node_ms = node_now_ms
        flags: list[str] = []

        raw_t = np.array([s.t_ms for s in batch.samples], dtype=np.float64)
        axes = to_g(np.array([[s.ax, s.ay, s.az] for s in batch.samples], dtype=np.float64), batch.unit)

        # ---- time alignment
        unc: float | None = None
        if batch.timebase == "node_monotonic_ms":
            t_node = raw_t
            unc = 0.0
        else:
            sync = st.sync
            if sync is None:
                # Without a clock sync the device time axis cannot be mapped to node time.
                flags.append("timestamp_untrusted")
                st.untrusted += 1
                st.flags.add("timestamp_untrusted")
                st.history_batches.append({"seq": seq, "reason": "no_clock_sync", "n": len(raw_t)})
                return IngestResult(True, ack_seq=seq, flags=flags, message="no clock sync: kept as history only")
            mismatch = False
            if batch.clock_sync_id and batch.clock_sync_id != sync.sync_id:
                earlier = st.recent_syncs.get(batch.clock_sync_id)
                if earlier is not None:
                    sync = earlier  # timestamped under an earlier sync of this session (re-sync race)
                else:
                    mismatch = True  # unknown sync (other node run / forged): never trusted
            t_node = np.asarray(sync.to_node(raw_t, node_now_ms), dtype=np.float64)
            unc = sync.uncertainty_at(node_now_ms)
            max_unc = self.cfg.quality.max_clock_uncertainty_ms
            if mismatch:
                flags.append("clock_sync_id_mismatch")
            if mismatch or unc > max_unc or sync.age_ms(node_now_ms) > self.cfg.quality.clock_sync_max_age_s * 1000.0:
                flags.append("timestamp_untrusted")
                st.flags.add("timestamp_untrusted")
                unc = max(unc, max_unc + 1.0)
            else:
                st.flags.discard("timestamp_untrusted")

        latency = node_now_ms - float(t_node[-1])
        st.latency_ms.append(latency)
        tol = (unc or 0.0) + 50.0
        if latency < -tol:
            # Samples claim to come from the future: timestamps cannot be trusted.
            flags.append("timestamp_untrusted")
            st.untrusted += 1
            st.flags.add("timestamp_untrusted")
            st.history_batches.append({"seq": seq, "reason": "future_timestamps", "latency_ms": latency})
            return IngestResult(True, ack_seq=seq, flags=flags, message="future timestamps: kept as history only")
        if latency > self.cfg.acquisition.late_batch_ms:
            st.late += 1
            flags.append("stale")
            st.history_batches.append({"seq": seq, "reason": "late_batch", "latency_ms": latency, "n": len(raw_t)})
            return IngestResult(True, ack_seq=seq, late=True, flags=flags, message="late batch: history only, not fused")
        if st.last_aligned_ms is not None and float(t_node[-1]) <= st.last_aligned_ms:
            st.out_of_order += 1
            flags.append("out_of_order")
            st.history_batches.append({"seq": seq, "reason": "out_of_order"})
            return IngestResult(True, ack_seq=seq, out_of_order=True, flags=flags, message="out-of-order batch: history only")
        if st.last_aligned_ms is not None:
            keep = t_node > st.last_aligned_ms
            t_node, axes = t_node[keep], axes[keep]
        if seq < st.last_seq:
            flags.append("out_of_order")
        st.last_seq = max(st.last_seq, seq)
        if len(t_node):
            st.last_aligned_ms = float(t_node[-1])
        return IngestResult(
            True, ack_seq=seq, flags=flags, t_node=t_node, axes_g=axes, includes_gravity=batch.includes_gravity,
            clock_uncertainty_ms=unc,
        )

    def stats(self) -> list[dict[str, Any]]:
        out = []
        for (dev, sess), st in self.streams.items():
            out.append(
                {
                    "device_id": dev, "session_id": sess, "placement": st.placement, "collector_type": st.collector_type,
                    "batches": st.batches, "samples": st.samples, "duplicates": st.duplicates, "late": st.late,
                    "out_of_order": st.out_of_order, "untrusted": st.untrusted, "last_seq": st.last_seq,
                    "sync": None if st.sync is None else {
                        "sync_id": st.sync.sync_id, "offset_ms": round(st.sync.offset_ms, 2), "rtt_ms": round(st.sync.rtt_ms, 2),
                        "base_uncertainty_ms": round(st.sync.base_uncertainty_ms, 2), "drift_ppm": round(st.sync.drift_ppm, 1),
                    },
                    "median_latency_ms": float(np.median(st.latency_ms)) if st.latency_ms else None,
                    "client_quality": st.client_quality, "flags": sorted(st.flags),
                    "history_batches": list(st.history_batches)[-5:],
                }
            )
        return out
