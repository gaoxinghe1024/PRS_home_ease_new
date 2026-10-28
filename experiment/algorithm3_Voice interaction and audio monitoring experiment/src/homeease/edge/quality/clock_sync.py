"""Device-to-node clock alignment from repeated ping/pong exchanges (NTP-style).

For each exchange the device records t0 (send) and t3 (receive) on its
monotonic clock; the node records t1 (receive) and t2 (reply) on its monotonic
clock. offset = ((t1-t0) + (t2-t3)) / 2 maps device time to node time
(node = device + offset); rtt = (t3-t0) - (t2-t1). The lowest-RTT exchanges are
kept; the uncertainty is half the best RTT plus their spread, and it grows with
time since the last sync using a drift bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from homeease.schemas.observations import TimeSyncSample


@dataclass
class SyncResult:
    sync_id: str
    offset_ms: float
    rtt_ms: float
    base_uncertainty_ms: float
    n_samples: int
    created_node_ms: float
    drift_ppm: float = 0.0
    history: list[tuple[float, float]] = field(default_factory=list)  # (node_ms, offset)

    def uncertainty_at(self, node_ms: float, drift_bound_ppm: float = 100.0) -> float:
        age_s = max(0.0, (node_ms - self.created_node_ms) / 1000.0)
        drift = max(abs(self.drift_ppm), drift_bound_ppm)
        return self.base_uncertainty_ms + drift * 1e-6 * age_s * 1000.0

    def age_ms(self, node_ms: float) -> float:
        return node_ms - self.created_node_ms

    def to_node(self, device_ms: np.ndarray | float, at_node_ms: float | None = None) -> np.ndarray | float:
        off = self.offset_ms
        if at_node_ms is not None and self.drift_ppm:
            off += self.drift_ppm * 1e-6 * (at_node_ms - self.created_node_ms)
        return device_ms + off


class ClockSyncError(ValueError):
    pass


def estimate_sync(
    sync_id: str, samples: list[TimeSyncSample], node_now_ms: float, previous: SyncResult | None = None, keep_best: int = 3
) -> SyncResult:
    rows = []
    for s in samples:
        rtt = (s.t3_device_ms - s.t0_device_ms) - (s.t2_server_mono_ms - s.t1_server_mono_ms)
        if rtt < 0 or rtt > 5000 or s.t3_device_ms < s.t0_device_ms:
            continue
        off = ((s.t1_server_mono_ms - s.t0_device_ms) + (s.t2_server_mono_ms - s.t3_device_ms)) / 2.0
        rows.append((rtt, off, s.t2_server_mono_ms))
    if not rows:
        raise ClockSyncError("no usable time-sync samples")
    if any(r[2] > node_now_ms + 1.0 for r in rows):
        raise ClockSyncError("server timestamps in the future: samples were not issued by this node")
    rows.sort(key=lambda r: r[0])
    best = rows[: max(1, keep_best)]
    offs = np.array([r[1] for r in best])
    offset = float(np.median(offs))
    spread = float((offs.max() - offs.min()) / 2.0) if len(offs) > 1 else 0.0
    unc = best[0][0] / 2.0 + spread + 2.0
    drift = 0.0
    history: list[tuple[float, float]] = []
    if previous is not None:
        history = list(previous.history[-9:])
        dt = node_now_ms - previous.created_node_ms
        if dt > 10_000:
            drift = (offset - previous.offset_ms) / dt * 1e6
    history.append((node_now_ms, offset))
    return SyncResult(sync_id, offset, best[0][0], unc, len(rows), node_now_ms, drift, history)


def fixed_sync(sync_id: str, offset_ms: float, node_now_ms: float, uncertainty_ms: float = 0.0) -> SyncResult:
    """Known mapping (replay/synthetic recordings on the node time axis)."""
    return SyncResult(sync_id, offset_ms, 0.0, uncertainty_ms, 0, node_now_ms)
