"""Training/evaluation windows built through the SAME streaming code as live inference.

Accelerometer samples go through AccelChannel (resampler + optional causal filter);
poses go through the TargetTracker and VisionChannel. Labels come only from the
recording's ``labels.json`` sidecar and are never visible to feature extraction.

Window label rule (configurable, documented in docs/evaluation_protocol.md):
  positive  : an annotated impact lies in [t_end - W + 0.5 s, t_end - post_min_s]
  negative  : the window does not overlap any fall interval [onset - 1 s, end + 3 s]
  ambiguous : otherwise (excluded from training and window metrics)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from homeease.core.config import AppConfig
from homeease.datasets.recording import Recording
from homeease.edge.tracking import TargetTracker
from homeease.edge.acquisition.replay import frame_to_obs, score_scale_for
from homeease.ml.common.signal import to_g
from homeease.ml.fall.accel import ALL_FEATURES, BASE_FEATURES, AccelWindow, feature_vector, window_features
from homeease.ml.fall.channels import AccelChannel, VisionChannel
from homeease.ml.fall.vision import C as VC
from homeease.ml.fall.vision import VisionWindow, _tl_ref, vision_window_features

ACCEL_SEQ_CHANNELS = ["sv", "dsv", "tilt", "mask"]
VISION_SEQ_CHANNELS = ["y_rel", "torso", "aspect", "vel", "conf", "mask"]
VISION_BASIC = ["y_rel_last", "y_rel_max", "torso_last", "torso_max", "aspect_last", "visible_frac"]
VISION_EXTENDED = VISION_BASIC + ["drop_tl", "speed_tl_s", "low_frac", "recovered", "lost_after_drop", "post_obs_s", "conf_mean"]


def offline_cfg(cfg: AppConfig) -> AppConfig:
    c = cfg.model_copy(deep=True)
    c.acquisition.accel_buffer_seconds = 7200.0
    return c


# ----------------------------------------------------------------- streams
def accel_time_map(rec: Recording) -> tuple[float, float]:
    """(offset, uncertainty) mapping accelerometer sample times onto the recording time axis,
    using the same clock-sync information as replay (never assuming device time == recording time)."""
    b0 = rec.accel[0]
    if b0.timebase == "device_monotonic_ms":
        from homeease.edge.quality.clock_sync import estimate_sync

        syncs = rec.time_sync_reports()
        if not syncs:
            raise ValueError(f"{rec.recording_id}: device timebase without clock-sync records")
        _, report, received = syncs[0]
        est = estimate_sync("offline", report.samples, received + 1e9)
        return est.offset_ms, est.base_uncertainty_ms
    return float(rec.manifest.get("accel_fixed_offset_ms", 0.0) or 0.0), float(rec.manifest.get("accel_fixed_uncertainty_ms", 0.0) or 0.0)


def accel_batch_offsets(rec: Recording) -> tuple[list[float], float]:
    """Per-batch offset onto the recording axis, plus the worst uncertainty used.

    Each batch is mapped with the clock sync it was timestamped under (its ``clock_sync_id``),
    or else the latest sync received before the batch arrived, as in live ingestion and replay.
    Long own recordings contain periodic re-syncs, so one offset for the whole file would let
    clock drift accumulate."""
    b0 = rec.accel[0]
    if b0.timebase != "device_monotonic_ms":
        off, unc = accel_time_map(rec)
        return [off] * len(rec.accel), unc
    from homeease.edge.quality.clock_sync import estimate_sync

    syncs = sorted(rec.time_sync_reports(), key=lambda s: s[2])
    if not syncs:
        raise ValueError(f"{rec.recording_id}: device timebase without clock-sync records")
    est = [(recv, rep.sync_id, estimate_sync(rep.sync_id, rep.samples, recv + 1e9)) for _, rep, recv in syncs]
    by_id = {sid: e for _, sid, e in est}
    arrivals = rec.manifest.get("accel_arrivals_ms", {}) or {}
    offsets, worst = [], 0.0
    for b in rec.accel:
        e = by_id.get(b.clock_sync_id or "")
        if e is None:
            arr = arrivals.get(f"{b.session_id}:{b.sequence_no}", arrivals.get(str(b.sequence_no)))
            prior = [x for recv, _, x in est if arr is not None and recv <= float(arr)]
            e = prior[-1] if prior else est[0][2]
        offsets.append(e.offset_ms)
        worst = max(worst, e.base_uncertainty_ms)
    return offsets, worst


def accel_channel_for(rec: Recording, cfg: AppConfig, scorer: Any = None) -> AccelChannel | None:
    if not rec.accel:
        return None
    ch = AccelChannel(offline_cfg(cfg), "offline", "replay", scorer)
    offsets, unc = accel_batch_offsets(rec)
    t_all, x_all = [], []
    for b, offset in zip(rec.accel, offsets):
        t_all.append(np.array([s.t_ms for s in b.samples]) + offset)
        x_all.append(to_g(np.array([[s.ax, s.ay, s.az] for s in b.samples]), b.unit))
    t = np.concatenate(t_all)
    x = np.concatenate(x_all)
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    b0 = rec.accel[0]
    ch.push(t[keep], x[keep], b0.includes_gravity, device_id=b0.device_id, session_id=b0.session_id, carriage="likely", clock_uncertainty_ms=unc, recv_now=float(t[-1]))
    return ch


def vision_channel_for(rec: Recording, cfg: AppConfig, scorer: Any = None) -> VisionChannel | None:
    if not rec.detections:
        return None
    ch = VisionChannel(cfg, "offline", "replay", scorer)
    ch.rows = deque(maxlen=None)
    tracker = TargetTracker(cfg.acquisition.zone, min_score=cfg.acquisition.pose.min_keypoint_score)
    for f in rec.detections:
        obs = frame_to_obs(f, tracker, f.t_ms, subject_id="offline", device_id="cam", session_id="cam", source_mode="replay", score_scale=score_scale_for(cfg, f.backend))
        ch.push(obs)
    return ch


# -------------------------------------------------------------- labelling
def window_label(t_end: float, W: float, falls: list[dict[str, Any]], post_min_s: float, key: str = "impact_ms") -> int | None:
    lo, hi = t_end - W + 500.0, t_end - post_min_s * 1000.0
    for f in falls:
        if lo <= float(f[key]) <= hi:
            return 1
    for f in falls:
        a, b = float(f["onset_ms"]) - 1000.0, float(f.get("end_ms", f[key])) + 3000.0
        if not (t_end < a or t_end - W > b):
            return None
    return 0


# ------------------------------------------------------------ sequences
def accel_seq(w: AccelWindow, T: int = 200) -> np.ndarray:
    """Orientation-invariant channels: |a|, scaled jerk, tilt change vs window start, mask."""
    m = w.mask.astype(bool)
    ax = np.where(m[:, None], w.axes, np.nan)
    sv = np.sqrt(np.nansum(ax**2, axis=1))
    sv[~m] = np.nan
    dsv = np.concatenate([[0.0], np.diff(sv)]) * 5.0
    first = np.where(m)[0][:25]
    tilt = np.zeros(len(sv))
    if len(first):
        g0 = np.nanmean(ax[first], axis=0)
        n0 = np.linalg.norm(g0)
        if n0 > 1e-6:
            norms = np.linalg.norm(np.nan_to_num(ax), axis=1)
            cos = np.clip(np.nan_to_num(ax) @ g0 / np.maximum(norms * n0, 1e-6), -1, 1)
            tilt = np.degrees(np.arccos(cos)) / 90.0
    X = np.stack([np.nan_to_num(sv - 1.0), np.nan_to_num(dsv), np.nan_to_num(tilt), m.astype(float)], axis=1)
    X[~m, :3] = 0.0
    X = np.clip(X, -8.0, 8.0)
    if len(X) >= T:
        return X[-T:].astype(np.float32)
    return np.vstack([np.zeros((T - len(X), X.shape[1])), X]).astype(np.float32)


def vision_seq(w: VisionWindow, T: int = 60) -> np.ndarray:
    m = w.mask.astype(bool)
    ref = _tl_ref(w)
    out = np.zeros((len(w.t), len(VISION_SEQ_CHANNELS)))
    if ref and m.any():
        yn = w.col("hip_y") / ref
        first = np.where(m)[0][:15]
        y0 = float(np.nanmedian(yn[first]))
        y_rel = np.clip(yn - y0, -3, 3)
        vel = np.concatenate([[0.0], np.diff(y_rel)]) * w.fps / 5.0
        out[:, 0] = np.nan_to_num(y_rel)
        out[:, 1] = np.nan_to_num(w.col("torso_deg") / 90.0)
        out[:, 2] = np.clip(np.nan_to_num(w.col("aspect")), 0, 4) / 2.0
        out[:, 3] = np.clip(np.nan_to_num(vel), -1, 1)
        out[:, 4] = np.nan_to_num(w.col("conf"))
    out[:, 5] = m.astype(float)
    out[~m, :5] = 0.0
    if len(out) >= T:
        return out[-T:].astype(np.float32)
    return np.vstack([np.zeros((T - len(out), out.shape[1])), out]).astype(np.float32)


def vision_features(w: VisionWindow, cfg: AppConfig, names: list[str]) -> np.ndarray:
    m = w.mask.astype(bool)
    f = vision_window_features(w, cfg.fall.vision)
    ref = _tl_ref(w)
    d: dict[str, float] = {"visible_frac": float(m.mean()) if len(m) else 0.0, "conf_mean": float(np.nanmean(w.col("conf")[m])) if m.any() else 0.0}
    if ref and m.any():
        yn = w.col("hip_y") / ref
        y0 = float(np.nanmedian(yn[np.where(m)[0][:15]]))
        last = m & (w.t >= w.t[-1] - 1000.0)
        d["y_rel_last"] = float(np.nanmean(yn[last] - y0)) if last.any() else 0.0
        d["y_rel_max"] = float(np.nanmax(yn[m] - y0))
        tor = w.col("torso_deg")
        d["torso_last"] = float(np.nanmean(tor[last])) / 90.0 if last.any() else 0.0
        d["torso_max"] = float(np.nanmax(tor[m])) / 90.0
        asp = w.col("aspect")
        d["aspect_last"] = float(np.nanmean(asp[last])) if last.any() else 0.0
    d["drop_tl"] = float(f.get("drop_tl", f.get("max_drop_tl", 0.0)) or 0.0)
    d["speed_tl_s"] = float(f.get("speed_tl_s", 0.0) or 0.0)
    d["low_frac"] = float(f.get("low_frac", 0.0) or 0.0)
    d["recovered"] = float(bool(f.get("recovered", False)))
    d["lost_after_drop"] = float(bool(f.get("lost_after_drop", False)))
    d["post_obs_s"] = float(f.get("post_obs_s", 0.0) or 0.0)
    return np.array([float(np.nan_to_num(d.get(k, 0.0))) for k in names], dtype=np.float64)


# ---------------------------------------------------------------- windows
@dataclass
class Windows:
    X: np.ndarray
    y: np.ndarray
    groups: list[str] = field(default_factory=list)
    rec_ids: list[str] = field(default_factory=list)
    t_end: list[float] = field(default_factory=list)
    info: list[dict[str, Any]] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    def subset(self, idx: np.ndarray) -> "Windows":
        idx = np.asarray(idx, dtype=int)
        return Windows(self.X[idx], self.y[idx], [self.groups[i] for i in idx], [self.rec_ids[i] for i in idx], [self.t_end[i] for i in idx], [self.info[i] for i in idx], self.names)


MIN_HISTORY_MS = 1500.0  # vision / fused training windows: minimum observed history


def _grid(t0: float, t1: float, W: float, step: float, min_history_ms: float | None = None) -> Iterator[float]:
    """Window end times. By default the first window has a full ``W`` of history; with
    ``min_history_ms`` it starts earlier and the missing history stays masked, exactly as the
    streaming channel sees the start of a stream (many UR Fall clips are shorter than W)."""
    t = t0 + (W if min_history_ms is None else min(W, min_history_ms))
    while t <= t1 + 1e-6:
        yield t
        t += step


def build_accel_windows(recs: list[Recording], cfg: AppConfig, kind: str = "features", feature_names: list[str] | None = None, step_ms: float = 250.0, post_min_s: float = 0.5) -> Windows:
    W = cfg.fall.accel.window_s * 1000.0
    names = feature_names or (ALL_FEATURES if kind == "features" else ACCEL_SEQ_CHANNELS)
    X, y, g, r, te, info = [], [], [], [], [], []
    c = cfg.fall.accel
    for rec in recs:
        ch = accel_channel_for(rec, cfg)
        if ch is None or ch.latest_t is None:
            continue
        data = ch.buf.view()
        falls = rec.labels().get("falls", [])
        for t_end in _grid(float(data[0, 0]), float(data[-1, 0]), W, step_ms):
            lab = window_label(t_end, W, falls, post_min_s)
            if lab is None:
                continue
            win = ch.window(t_end)
            if len(win.t) < 0.8 * W / 20.0 or win.mask.mean() < 0.7:
                continue
            if kind == "features":
                f = window_features(win, c.post_start_s, post_min_s, c.post_window_s, saturation_g=cfg.quality.saturation_g)
                if not f.get("valid"):
                    continue
                X.append(feature_vector(f, names))
            else:
                X.append(accel_seq(win))
            y.append(lab)
            g.append(str(rec.manifest.get("group_id", rec.recording_id)))
            r.append(rec.recording_id)
            te.append(t_end)
            info.append({"dataset": rec.manifest.get("dataset"), "kind": rec.manifest.get("kind"), "activity": rec.manifest.get("activity_code"), "age_group": rec.manifest.get("age_group")})
    if kind == "features":
        Xa = np.nan_to_num(np.vstack(X), nan=0.0) if X else np.zeros((0, len(names)))
    else:
        Xa = np.stack(X) if X else np.zeros((0, 200, len(names)), dtype=np.float32)  # (N, T, C)
    return Windows(Xa, np.asarray(y, dtype=int), g, r, te, info, list(names))


def build_vision_windows(recs: list[Recording], cfg: AppConfig, kind: str = "seq", feature_names: list[str] | None = None, step_ms: float = 200.0, post_min_s: float = 0.3, window_s: float = 4.0, min_history_ms: float = MIN_HISTORY_MS) -> Windows:
    W = window_s * 1000.0
    names = feature_names or (VISION_SEQ_CHANNELS if kind == "seq" else VISION_EXTENDED)
    X, y, g, r, te, info = [], [], [], [], [], []
    for rec in recs:
        ch = vision_channel_for(rec, cfg)
        if ch is None or not ch.rows:
            continue
        rows = np.vstack(ch.rows)
        falls = rec.labels().get("falls", [])
        key = "lying_start_ms" if falls and "lying_start_ms" in falls[0] else "impact_ms"
        for t_end in _grid(float(rows[0, 0]), float(rows[-1, 0]), W, step_ms, min_history_ms):
            lab = window_label(t_end, W, falls, post_min_s, key)
            if lab is None:
                continue
            win = ch.window(t_end, W)
            if kind == "seq":
                X.append(vision_seq(win, int(round(W / 1000.0 * win.fps))))
            else:
                X.append(vision_features(win, cfg, names))
            y.append(lab)
            g.append(str(rec.manifest.get("group_id", rec.recording_id)))
            r.append(rec.recording_id)
            te.append(t_end)
            info.append({"dataset": rec.manifest.get("dataset"), "kind": rec.manifest.get("kind")})
    Xa = np.stack(X) if X else np.zeros((0, len(names)))
    return Windows(Xa, np.asarray(y, dtype=int), g, r, te, info, list(names))


def fused_seq(aw: AccelWindow | None, vw: VisionWindow | None, T: int = 60, fps: float = 15.0) -> np.ndarray:
    """Aligned multimodal sequence on the vision grid: per-step accel summaries + vision channels.
    Missing modalities are zero-filled WITH their mask at 0 (never presented as normal values)."""
    acc = np.zeros((T, 5))
    if aw is not None and len(aw.t):
        seq = accel_seq(aw, T=len(aw.t))
        grid = (aw.t[-1] - 1000.0 / fps * np.arange(T - 1, -1, -1)) if len(aw.t) else None
        if grid is not None:
            half = 1000.0 / fps / 2.0
            for k, gt in enumerate(grid):
                sel = (aw.t > gt - half * 2) & (aw.t <= gt)
                if sel.any() and seq[sel, 3].mean() > 0.5:
                    s = seq[sel]
                    acc[k] = [s[:, 0].max(), s[:, 0].min(), np.abs(s[:, 1]).max(), s[:, 2].mean(), 1.0]
    vis = vision_seq(vw, T) if vw is not None else np.zeros((T, len(VISION_SEQ_CHANNELS)), dtype=np.float32)
    return np.concatenate([acc, vis], axis=1).astype(np.float32)


FUSED_CHANNELS = ["a_max", "a_min", "a_jerk", "a_tilt", "a_mask"] + [f"v_{c}" for c in VISION_SEQ_CHANNELS]


def build_fused_windows(recs: list[Recording], cfg: AppConfig, step_ms: float = 200.0, post_min_s: float = 0.3, window_s: float = 4.0, min_history_ms: float = MIN_HISTORY_MS) -> Windows:
    W = window_s * 1000.0
    X, y, g, r, te, info = [], [], [], [], [], []
    for rec in recs:
        ach = accel_channel_for(rec, cfg)
        vch = vision_channel_for(rec, cfg)
        if ach is None or vch is None or not vch.rows or ach.latest_t is None:
            continue
        rows = np.vstack(vch.rows)
        data = ach.buf.view()
        t0 = max(float(rows[0, 0]), float(data[0, 0]))
        t1 = min(float(rows[-1, 0]), float(data[-1, 0]))
        falls = rec.labels().get("falls", [])
        for t_end in _grid(t0, t1, W, step_ms, min_history_ms):
            lab = window_label(t_end, W, falls, post_min_s)
            if lab is None:
                continue
            X.append(fused_seq(ach.window(t_end, W), vch.window(t_end, W), int(round(window_s * 15))))
            y.append(lab)
            g.append(str(rec.manifest.get("group_id", rec.recording_id)))
            r.append(rec.recording_id)
            te.append(t_end)
            info.append({"dataset": rec.manifest.get("dataset"), "kind": rec.manifest.get("kind")})
    Xa = np.stack(X) if X else np.zeros((0, int(window_s * 15), len(FUSED_CHANNELS)))
    return Windows(Xa, np.asarray(y, dtype=int), g, r, te, info, FUSED_CHANNELS)


def modality_dropout(x: np.ndarray, rng: np.random.Generator, p: float = 0.25) -> np.ndarray:
    """Training augmentation: randomly remove one modality (values and mask set to 0)."""
    x = x.copy()
    for i in range(len(x)):
        u = rng.random()
        if u < p / 2:
            x[i, :, :5] = 0.0
        elif u < p:
            x[i, :, 5:] = 0.0
    return x


__all__ = [
    "ACCEL_SEQ_CHANNELS", "VISION_SEQ_CHANNELS", "VISION_BASIC", "VISION_EXTENDED", "BASE_FEATURES", "ALL_FEATURES",
    "build_accel_windows", "build_vision_windows", "build_fused_windows", "accel_seq", "vision_seq", "vision_features",
    "fused_seq", "FUSED_CHANNELS", "modality_dropout", "Windows", "window_label", "VC",
]
