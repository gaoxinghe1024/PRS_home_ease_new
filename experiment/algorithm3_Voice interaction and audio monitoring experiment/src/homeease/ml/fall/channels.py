"""Streaming channel wrappers shared by live runtime, replay and offline evaluation.

Each channel buffers aligned observations, evaluates its scorer on causal
windows on the *data* time axis (deterministic under replay), groups repeated
triggers of the same physical event into one candidate, and can be asked to
assess a reference time for late fusion.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol

import numpy as np

from homeease.core.config import AppConfig
from homeease.ml.common.signal import RingBuffer, StreamingLowpass, StreamingResampler
from homeease.ml.fall.accel import FS, AccelRuleScorer, AccelWindow, ScoreResult
from homeease.ml.fall.vision import C as VC
from homeease.ml.fall.vision import VisionRuleScorer, VisionWindow, build_window, frame_features
from homeease.ml.pose.layout import PoseObs
from homeease.schemas.evidence import Evidence, ScoreSemantics

_COMPACT_KEYS_ACCEL = (
    "peak_sv", "peak_dyn", "jerk_max", "pre_min_sv", "post_std", "still_ratio", "post_obs_s",
    "orient_change_deg", "valid_fraction", "saturated", "p_model",
)
_COMPACT_KEYS_VISION = (
    "drop_tl", "speed_tl_s", "low_frac", "torso_post_mean", "aspect_post_mean", "recovered", "lost_after_drop",
    "post_obs_s", "visible_frac", "upright_torso_frac", "descent_start_ms", "descent_end_ms", "p_model",
)


def _compact(features: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        if k in features:
            v = features[k]
            if isinstance(v, (float, np.floating)):
                v = None if not np.isfinite(v) else round(float(v), 4)
            elif isinstance(v, np.bool_):
                v = bool(v)
            out[k] = v
    return out


class Scorer(Protocol):
    channel: str
    model_id: str
    model_version: str
    score_semantics: ScoreSemantics
    calibration_id: str | None
    strong_threshold: float
    candidate_threshold: float
    window_ms: float
    step_ms: float


class _Episode:
    def __init__(self, first_eval: float, res: ScoreResult, win_start: float, win_end: float, opened_now: float = 0.0):
        self.first_eval = first_eval
        self.best = res
        self.win_start = win_start
        self.win_end = win_end
        self.opened_now = opened_now


class AccelChannel:
    channel = "accel"

    def __init__(self, cfg: AppConfig, subject_id: str, source_mode: str = "live", scorer: Any = None):
        self.cfg = cfg
        self.subject_id = subject_id
        self.source_mode = source_mode
        q = cfg.quality
        self.scorer = scorer or AccelRuleScorer(cfg.fall.accel, cfg.acquisition.accel_target_hz, q.max_clock_uncertainty_ms, q.saturation_g)
        self.fs = FS
        cap = int(cfg.acquisition.accel_buffer_seconds * FS)
        self.buf = RingBuffer(cap, 5)  # t, ax, ay, az, mask
        self._resampler = StreamingResampler(FS, 3, q.accel_gap_interp_max_ms)
        self._lowpass = (
            StreamingLowpass(FS, cfg.fall.accel.lowpass_hz, 3) if cfg.fall.accel.filter == "lowpass" else None
        )
        self.includes_gravity: bool | None = None
        self.device_id: str | None = None
        self.session_id: str | None = None
        self.carriage = "unknown"
        self.clock_uncertainty_ms: float | None = None
        self._raw_times: deque[float] = deque(maxlen=400)
        self.last_raw_t: float | None = None
        self.last_recv_now: float | None = None
        self.next_eval: float | None = None
        self._episode: _Episode | None = None
        self._emitted_peaks: deque[float] = deque(maxlen=20)
        self.settle_ms = 500.0
        self.same_event_ms = 1500.0
        self.last_motion_t: float | None = None
        self._stall_flushed_for: float | None = None
        self.flags: set[str] = set()
        self.last_eval: dict[str, Any] = {}

    # ---------------------------------------------------------------- ingest
    def reset_stream(self) -> None:
        self._resampler.reset()
        if self._lowpass is not None:
            self._lowpass.reset()

    def clear(self) -> None:
        self.reset_stream()
        self.buf.clear()
        self.next_eval = None
        self._episode = None
        self._raw_times.clear()
        self.last_raw_t = None

    def push(
        self,
        t_node: np.ndarray,
        axes_g: np.ndarray,
        includes_gravity: bool,
        *,
        device_id: str,
        session_id: str,
        carriage: str,
        clock_uncertainty_ms: float | None,
        recv_now: float,
    ) -> None:
        if self.session_id is not None and (session_id != self.session_id or device_id != self.device_id):
            # New device session: never fuse samples across sessions.
            self.clear()
        if self.includes_gravity is not None and includes_gravity != self.includes_gravity:
            self.clear()
        self.includes_gravity = includes_gravity
        self.device_id, self.session_id = device_id, session_id
        self.carriage = carriage
        self.clock_uncertainty_ms = clock_uncertainty_ms
        self.last_recv_now = recv_now
        if len(t_node) == 0:
            return
        if self.last_raw_t is not None:
            keep = t_node > self.last_raw_t
            t_node, axes_g = t_node[keep], axes_g[keep]
            if len(t_node) == 0:
                return
        self._raw_times.extend(t_node.tolist())
        self.last_raw_t = float(t_node[-1])
        chunk = self._resampler.push(t_node, axes_g)
        if len(chunk.t) == 0:
            return
        x = chunk.x
        if self._lowpass is not None:
            x = self._lowpass.push(chunk.x, chunk.mask)
        rows = np.column_stack([chunk.t, x, chunk.mask.astype(np.float64)])
        self.buf.append(rows)
        if self.next_eval is None:
            self.next_eval = float(chunk.t[0] + self.scorer.window_ms)
        # device motion cue for carriage estimation (std of magnitude over the chunk)
        sv = np.sqrt(np.nansum(chunk.x**2, axis=1))
        valid = chunk.mask.astype(bool)
        if valid.sum() > 5 and np.nanstd(sv[valid]) > 0.03:
            self.last_motion_t = float(chunk.t[-1])

    def rate_hz(self) -> float | None:
        if len(self._raw_times) < 5:
            return None
        ts = np.asarray(self._raw_times)
        recent = ts[ts >= ts[-1] - 5000.0]
        if len(recent) < 5 or recent[-1] <= recent[0]:
            return None
        return float((len(recent) - 1) / ((recent[-1] - recent[0]) / 1000.0))

    @property
    def latest_t(self) -> float | None:
        last = self.buf.last()
        return None if last is None else float(last[0])

    def window(self, t_end: float, length_ms: float | None = None) -> AccelWindow:
        length_ms = length_ms or self.scorer.window_ms
        data = self.buf.view()
        sel = (data[:, 0] > t_end - length_ms - 1e-6) & (data[:, 0] <= t_end + 1e-6)
        d = data[sel]
        return AccelWindow(
            t=d[:, 0],
            axes=d[:, 1:4],
            mask=d[:, 4].astype(np.int8),
            includes_gravity=bool(self.includes_gravity) if self.includes_gravity is not None else True,
            clock_uncertainty_ms=self.clock_uncertainty_ms,
            carriage=self.carriage,
            rate_hz=self.rate_hz(),
        )

    # --------------------------------------------------------------- outputs
    def _evidence(self, res: ScoreResult, win_start: float, win_end: float, now: float) -> Evidence:
        sc = self.scorer
        feats = _compact(res.features, _COMPACT_KEYS_ACCEL)
        return Evidence(
            channel="accel",
            candidate=bool(res.candidate),
            score=float(np.clip(res.score, 0, 1)),
            score_semantics=sc.score_semantics,
            quality_score=float(np.clip(res.quality, 0, 1)),
            available=bool(res.valid),
            time_start_ms=float(win_start),
            time_end_ms=float(win_end),
            peak_time_ms=res.peak_time_ms,
            emitted_at_ms=float(now),
            features_summary=feats,
            reason_codes=list(res.reasons),
            quality_flags=list(res.flags),
            contradicts=False,
            model_id=sc.model_id,
            model_version=sc.model_version,
            calibration_id=sc.calibration_id,
            source_mode=self.source_mode,
            subject_id=self.subject_id,
            session_id=self.session_id,
            device_id=self.device_id,
            clock_uncertainty_ms=self.clock_uncertainty_ms,
            strong=bool(res.candidate and res.score >= sc.strong_threshold),
        )

    def _is_same_event(self, peak: float | None) -> bool:
        if peak is None:
            return False
        return any(abs(peak - p) < self.same_event_ms for p in self._emitted_peaks)

    def update(self, now: float, truncated_flush: bool = False) -> list[Evidence]:
        out: list[Evidence] = []
        latest = self.latest_t
        if latest is None or self.next_eval is None:
            return out
        sc = self.scorer
        while self.next_eval <= latest + 1e-6:
            te = self.next_eval
            self.next_eval += sc.step_ms
            win = self.window(te)
            res = sc.score(win)
            self.last_eval = {"t": te, "score": res.score, "candidate": res.candidate, "reasons": res.reasons}
            out.extend(self._episode_step(res, te, win, now))
        stalled = latest is not None and now - latest > self.cfg.quality.accel_stale_ms and self._stall_flushed_for != latest
        if (truncated_flush or stalled) and latest is not None:
            # End of the stream (recording end, or the phone stopped sending right after an
            # impact): evaluate the last window once with the truncated post-window.
            self._stall_flushed_for = latest
            win = self.window(latest)
            res = sc.score(win, truncated=True)
            out.extend(self._episode_step(res, latest, win, now))
            if self._episode is not None:
                ep = self._episode
                self._episode = None
                if ep.best.peak_time_ms is not None:
                    self._emitted_peaks.append(ep.best.peak_time_ms)
                out.append(self._evidence(ep.best, ep.win_start, ep.win_end, now))
        if self._episode is not None and now - self._episode.opened_now >= self.settle_ms + 250.0:
            # settle on the node clock even if no newer data arrive
            ep = self._episode
            self._episode = None
            if ep.best.peak_time_ms is not None:
                self._emitted_peaks.append(ep.best.peak_time_ms)
            out.append(self._evidence(ep.best, ep.win_start, ep.win_end, now))
        return out

    def _episode_step(self, res: ScoreResult, te: float, win: AccelWindow, now: float) -> list[Evidence]:
        out: list[Evidence] = []
        sc = self.scorer
        trig = res.valid and res.candidate and res.score >= sc.candidate_threshold
        if trig and not self._is_same_event(res.peak_time_ms):
            if self._episode is None:
                self._episode = _Episode(te, res, te - sc.window_ms, te, now)
            elif res.peak_time_ms is not None and self._episode.best.peak_time_ms is not None and abs(
                res.peak_time_ms - self._episode.best.peak_time_ms
            ) > self.same_event_ms:
                ep = self._episode
                self._emitted_peaks.append(ep.best.peak_time_ms)
                out.append(self._evidence(ep.best, ep.win_start, ep.win_end, now))
                self._episode = _Episode(te, res, te - sc.window_ms, te, now)
            elif res.score > self._episode.best.score:
                self._episode.best, self._episode.win_start, self._episode.win_end = res, te - sc.window_ms, te
        if self._episode is not None and te - self._episode.first_eval >= self.settle_ms:
            ep = self._episode
            self._episode = None
            if ep.best.peak_time_ms is not None:
                self._emitted_peaks.append(ep.best.peak_time_ms)
            out.append(self._evidence(ep.best, ep.win_start, ep.win_end, now))
        return out

    def assess(self, t_ref: float, now: float, tol_ms: float = 1000.0) -> Evidence:
        """What does the accelerometer show around ``t_ref``? Never 'contradicts'
        a vision candidate: the phone may simply not be on the body."""
        sc = self.scorer
        latest = self.latest_t
        if latest is None:
            return self._unavailable(t_ref, now, ["accel_no_data"])
        t_end = min(latest, t_ref + (self.cfg.fall.accel.post_start_s + self.cfg.fall.accel.post_window_s) * 1000.0)
        win = self.window(t_end)
        if len(win.t) == 0 or win.mask.mean() < 0.5 or t_end < t_ref:
            return self._unavailable(t_ref, now, ["accel_no_data_around_event"])
        res = sc.score(win, peak_search=(t_ref - tol_ms, t_ref + tol_ms), truncated=(t_end < t_ref + 1500.0))
        return self._evidence(res, t_end - sc.window_ms, t_end, now)

    def _unavailable(self, t_ref: float, now: float, reasons: list[str]) -> Evidence:
        sc = self.scorer
        return Evidence(
            channel="accel", candidate=False, score=0.0, score_semantics=sc.score_semantics, quality_score=0.0,
            available=False, time_start_ms=t_ref - 1000.0, time_end_ms=t_ref + 1000.0, peak_time_ms=None,
            emitted_at_ms=now, reason_codes=reasons, quality_flags=["missing"], model_id=sc.model_id,
            model_version=sc.model_version, calibration_id=sc.calibration_id, source_mode=self.source_mode,
            subject_id=self.subject_id, session_id=self.session_id, device_id=self.device_id,
            clock_uncertainty_ms=self.clock_uncertainty_ms,
        )

    def recent_series(self, seconds: float = 20.0, step: int = 2) -> dict[str, list[float]]:
        data = self.buf.view()
        if len(data) == 0:
            return {"t": [], "sv": []}
        sel = data[:, 0] >= data[-1, 0] - seconds * 1000.0
        d = data[sel][::step]
        sv = np.sqrt(np.sum(d[:, 1:4] ** 2, axis=1))
        return {"t": [round(float(x), 1) for x in d[:, 0]], "sv": [None if not np.isfinite(v) else round(float(v), 3) for v in sv]}


class VisionChannel:
    channel = "vision"

    def __init__(self, cfg: AppConfig, subject_id: str, source_mode: str = "live", scorer: Any = None):
        self.cfg = cfg
        self.subject_id = subject_id
        self.source_mode = source_mode
        self.scorer = scorer or VisionRuleScorer(cfg.fall.vision)
        self.min_score = cfg.acquisition.pose.min_keypoint_score
        self.rows: deque[np.ndarray] = deque(maxlen=int(60 * 30))  # up to ~60 s at 30 fps
        self.next_eval: float | None = None
        self._episode: _Episode | None = None
        self._emitted_peaks: deque[float] = deque(maxlen=20)
        self.settle_ms = 400.0
        self.same_event_ms = 2500.0
        self.device_id: str | None = None
        self.session_id: str | None = None
        self.last_obs: PoseObs | None = None
        self.last_valid_t: float | None = None
        self.last_frame_t: float | None = None
        self.frame_count = 0
        self._stall_flushed_for: float | None = None
        self.last_eval: dict[str, Any] = {}

    def clear(self) -> None:
        self.rows.clear()
        self.next_eval = None
        self._episode = None

    def push(self, obs: PoseObs) -> np.ndarray | None:
        if self.session_id is not None and obs.session_id != self.session_id:
            self.clear()
        if self.last_frame_t is not None and obs.t_ms <= self.last_frame_t:
            return None
        self.session_id, self.device_id = obs.session_id, obs.device_id
        row = frame_features(obs, self.min_score)
        self.rows.append(row)
        self.last_obs = obs
        self.last_frame_t = obs.t_ms
        self.frame_count += 1
        if row[VC["valid"]] == 1:
            self.last_valid_t = obs.t_ms
        if self.next_eval is None:
            self.next_eval = obs.t_ms + self.scorer.window_ms * 0.5
        return row

    @property
    def latest_t(self) -> float | None:
        return self.last_frame_t

    def _rows(self) -> np.ndarray:
        return np.vstack(self.rows) if self.rows else np.zeros((0, len(VC)))

    def window(self, t_end: float, length_ms: float | None = None) -> VisionWindow:
        length_ms = length_ms or self.scorer.window_ms
        rows = self._rows()
        if len(rows):
            rows = rows[(rows[:, 0] > t_end - length_ms - 500.0) & (rows[:, 0] <= t_end + 1e-6)]
        return build_window(rows, t_end, length_ms, self.scorer.fps)

    def _target_conf(self, win: VisionWindow) -> float | None:
        m = win.mask.astype(bool)
        if not m.any():
            return None
        return float(np.nanmean(win.col("target_conf")[m]))

    def _evidence(self, res: ScoreResult, win: VisionWindow, win_start: float, win_end: float, now: float) -> Evidence:
        sc = self.scorer
        last = self.last_obs
        return Evidence(
            channel="vision",
            candidate=bool(res.candidate),
            score=float(np.clip(res.score, 0, 1)),
            score_semantics=sc.score_semantics,
            quality_score=float(np.clip(res.quality, 0, 1)),
            available=bool(res.valid),
            time_start_ms=float(win_start),
            time_end_ms=float(win_end),
            peak_time_ms=res.peak_time_ms,
            emitted_at_ms=float(now),
            features_summary=_compact(res.features, _COMPACT_KEYS_VISION),
            reason_codes=list(res.reasons),
            quality_flags=list(res.flags),
            contradicts=bool(res.contradicts),
            model_id=sc.model_id,
            model_version=sc.model_version,
            calibration_id=sc.calibration_id,
            source_mode=self.source_mode,
            subject_id=self.subject_id,
            session_id=self.session_id,
            device_id=self.device_id,
            track_id=last.track_id if last else None,
            target_confidence=self._target_conf(win),
            clock_uncertainty_ms=0.0,
            strong=bool(res.candidate and res.score >= sc.strong_threshold),
        )

    def _is_same_event(self, peak: float | None) -> bool:
        if peak is None:
            return False
        return any(abs(peak - p) < self.same_event_ms for p in self._emitted_peaks)

    def update(self, now: float, truncated_flush: bool = False) -> list[Evidence]:
        out: list[Evidence] = []
        latest = self.latest_t
        if latest is None or self.next_eval is None:
            return out
        sc = self.scorer
        while self.next_eval <= latest + 1e-6:
            te = self.next_eval
            self.next_eval += sc.step_ms
            win = self.window(te)
            res = sc.score(win)
            self.last_eval = {"t": te, "score": res.score, "candidate": res.candidate, "reasons": res.reasons}
            out.extend(self._episode_step(res, te, win, now))
        stalled = now - latest > self.cfg.quality.camera_stale_ms and self._stall_flushed_for != latest
        if truncated_flush or stalled:
            # End of the stream (recording end, camera failure right after a descent).
            self._stall_flushed_for = latest
            win = self.window(latest)
            res = sc.score(win, truncated=True)
            out.extend(self._episode_step(res, latest, win, now))
            if self._episode is not None:
                ep = self._episode
                self._episode = None
                if ep.best.peak_time_ms is not None:
                    self._emitted_peaks.append(ep.best.peak_time_ms)
                out.append(self._evidence(ep.best, win, ep.win_start, ep.win_end, now))
        if self._episode is not None and now - self._episode.opened_now >= self.settle_ms + 250.0:
            ep = self._episode
            self._episode = None
            if ep.best.peak_time_ms is not None:
                self._emitted_peaks.append(ep.best.peak_time_ms)
            out.append(self._evidence(ep.best, self.window(ep.win_end), ep.win_start, ep.win_end, now))
        return out

    def _episode_step(self, res: ScoreResult, te: float, win: VisionWindow, now: float) -> list[Evidence]:
        out: list[Evidence] = []
        sc = self.scorer
        trig = res.valid and res.candidate and res.score >= sc.candidate_threshold
        if trig and not self._is_same_event(res.peak_time_ms):
            if self._episode is None:
                self._episode = _Episode(te, res, te - sc.window_ms, te, now)
            elif res.score > self._episode.best.score:
                self._episode.best, self._episode.win_start, self._episode.win_end = res, te - sc.window_ms, te
        if self._episode is not None and te - self._episode.first_eval >= self.settle_ms:
            ep = self._episode
            self._episode = None
            if ep.best.peak_time_ms is not None:
                self._emitted_peaks.append(ep.best.peak_time_ms)
            out.append(self._evidence(ep.best, win, ep.win_start, ep.win_end, now))
        return out

    def assess(self, t_ref: float, now: float) -> Evidence:
        latest = self.latest_t
        sc = self.scorer
        if latest is None:
            return self._unavailable(t_ref, now, ["vision_no_frames"])
        t_end = min(latest, t_ref + 3000.0)
        if t_end < t_ref - 500.0:
            return self._unavailable(t_ref, now, ["vision_no_frames_around_event"])
        win = self.window(t_end)
        if hasattr(sc, "assess_contradiction"):
            res = sc.assess_contradiction(win, t_ref)
        else:
            res = sc.score(win, descent_search=(t_ref - 2000.0, t_ref + 1500.0))
        return self._evidence(res, win, t_end - sc.window_ms, t_end, now)

    def _unavailable(self, t_ref: float, now: float, reasons: list[str]) -> Evidence:
        sc = self.scorer
        return Evidence(
            channel="vision", candidate=False, score=0.0, score_semantics=sc.score_semantics, quality_score=0.0,
            available=False, time_start_ms=t_ref - 1000.0, time_end_ms=t_ref + 1000.0, emitted_at_ms=now,
            reason_codes=reasons, quality_flags=["missing"], model_id=sc.model_id, model_version=sc.model_version,
            calibration_id=sc.calibration_id, source_mode=self.source_mode, subject_id=self.subject_id,
            session_id=self.session_id, device_id=self.device_id,
        )

    def recent_series(self, seconds: float = 20.0) -> dict[str, list[Any]]:
        rows = self._rows()
        if not len(rows):
            return {"t": [], "hip_y_tl": [], "torso_deg": []}
        rows = rows[rows[:, 0] >= rows[-1, 0] - seconds * 1000.0]
        tl = np.nanmedian(rows[:, VC["tl"]]) if np.isfinite(rows[:, VC["tl"]]).any() else np.nan

        def r(v: float) -> float | None:
            return None if not np.isfinite(v) else round(float(v), 3)

        return {
            "t": [round(float(x), 1) for x in rows[:, 0]],
            "hip_y_tl": [r(v / tl) if np.isfinite(tl) and tl > 0 else None for v in rows[:, VC["hip_y"]]],
            "torso_deg": [r(v) for v in rows[:, VC["torso_deg"]]],
        }
