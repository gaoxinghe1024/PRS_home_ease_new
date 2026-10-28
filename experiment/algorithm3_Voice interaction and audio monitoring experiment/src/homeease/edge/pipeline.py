"""MonitoringPipeline: the single inference chain used by live runtime, replay
and offline evaluation.

It is pure with respect to time: callers pass ``now`` (node monotonic ms for
live, the recording time axis for replay/evaluation). Inputs are aligned
acceleration samples and target pose observations; outputs are evidences,
fusion decisions, coverage snapshots and exercise updates.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from homeease.core.config import AppConfig
from homeease.edge.quality.coverage import CoverageEvaluator, CoverageSnapshot
from homeease.ml.exercise.phase import PhaseClassifier
from homeease.ml.exercise.tracker import ExerciseTracker, ExerciseUpdate
from homeease.ml.fall.channels import AccelChannel, VisionChannel
from homeease.ml.fall.fusion import LateFusionEngine, TrainedFusionModel
from homeease.ml.fall.vision import C as VC
from homeease.ml.pose.layout import PoseObs, kp_to_list
from homeease.schemas.evidence import DecisionLevel, Evidence, FusionDecision


@dataclass
class PipelineOutput:
    now: float
    evidences: list[Evidence] = field(default_factory=list)
    decisions: list[FusionDecision] = field(default_factory=list)
    coverage: CoverageSnapshot | None = None
    exercise: list[ExerciseUpdate] = field(default_factory=list)


class MonitoringPipeline:
    def __init__(
        self,
        cfg: AppConfig,
        subject_id: str,
        room_id: str | None = None,
        source_mode: str = "live",
        accel_scorer: Any = None,
        vision_scorer: Any = None,
        fusion_backend: str | None = None,
        trained_fusion: TrainedFusionModel | None = None,
        fusion_engine: Any = None,
        fusion_engine_factory: Any = None,
        fusion_package: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.subject_id = subject_id
        self.room_id = room_id
        self.source_mode = source_mode
        self.lock = threading.RLock()
        self.accel = AccelChannel(cfg, subject_id, source_mode, accel_scorer)
        self.vision = VisionChannel(cfg, subject_id, source_mode, vision_scorer)
        backend = fusion_backend or cfg.fall.fusion.backend
        if fusion_engine is not None:
            self.fusion = fusion_engine
        elif fusion_engine_factory is not None:
            self.fusion = fusion_engine_factory(cfg, self.accel, self.vision, subject_id, room_id, source_mode, fusion_package)
        else:
            self.fusion = LateFusionEngine(cfg, self.accel, self.vision, subject_id, room_id, source_mode, backend, trained_fusion)
        self.coverage = CoverageEvaluator(cfg)
        self.exercise: ExerciseTracker | None = None
        self._pending_ex: list[ExerciseUpdate] = []
        self.evidence_log: deque[dict[str, Any]] = deque(maxlen=60)
        self.decision_log: deque[dict[str, Any]] = deque(maxlen=60)
        # status inputs from the runtime
        self.camera: dict[str, Any] = {"camera_enabled": cfg.acquisition.camera.enabled, "camera_error": None, "model_status": "ok", "backend": None, "fps": None}
        self.phone: dict[str, Any] = {"has_device": False, "carriage_setting": "unknown", "placement": "unknown", "page_hidden": False, "timestamp_untrusted": False}
        self.last_person_count = 0
        self.last_target_ambiguous = False
        self._boost_until = -1e18
        self.recording_fps: float | None = None  # fixed camera rate during an explicit research recording
        self.last_tick: float | None = None
        self.stats = {"accel_samples": 0, "pose_frames": 0, "decisions": 0}

    # ---------------------------------------------------------------- inputs
    def ingest_accel(
        self,
        t_node: np.ndarray,
        axes_g: np.ndarray,
        includes_gravity: bool,
        *,
        device_id: str,
        session_id: str,
        clock_uncertainty_ms: float | None,
        now: float,
        placement: str = "unknown",
        timestamp_untrusted: bool = False,
    ) -> None:
        with self.lock:
            self.phone.update(has_device=True, placement=placement, timestamp_untrusted=timestamp_untrusted)
            self.accel.push(
                t_node, axes_g, includes_gravity, device_id=device_id, session_id=session_id,
                carriage=self.carriage(now), clock_uncertainty_ms=clock_uncertainty_ms, recv_now=now,
            )
            self.stats["accel_samples"] += len(t_node)

    def ingest_pose(self, obs: PoseObs) -> None:
        with self.lock:
            row = self.vision.push(obs)
            self.stats["pose_frames"] += 1
            self.last_person_count = obs.person_count
            self.last_target_ambiguous = "target_ambiguous" in obs.flags
            if row is not None and row[VC["valid"]] == 1 and len(self.vision.rows) >= 3:
                prev = self.vision.rows[-3]
                dt = (row[0] - prev[0]) / 1000.0
                tl = row[VC["tl"]]
                if dt > 0 and prev[VC["valid"]] == 1 and tl > 5 and (row[VC["hip_y"]] - prev[VC["hip_y"]]) / tl / dt > 1.0:
                    self._boost_until = obs.t_ms + self.cfg.acquisition.camera.event_boost_s * 1000.0
            if self.exercise is not None:
                self._pending_ex.append(self.exercise.push(obs))

    def set_camera_status(self, **kw: Any) -> None:
        with self.lock:
            self.camera.update(kw)

    def set_phone_status(self, **kw: Any) -> None:
        with self.lock:
            self.phone.update(kw)

    # ------------------------------------------------------------- carriage
    def carriage(self, now: float) -> str:
        """Uncertain carriage estimate (confirmed/likely/unknown/not_carried).

        Explicit settings and placement come first; motion cues only raise
        ``unknown`` to ``likely``. A still phone near 1 g is NOT proof of wearing.
        """
        setting = self.phone.get("carriage_setting", "unknown")
        placement = self.phone.get("placement", "unknown")
        if setting == "not_carried" or placement == "table":
            return "not_carried"
        lm = self.accel.last_motion_t
        recent_motion = lm is not None and now - lm <= self.cfg.quality.carriage_still_minutes * 60_000.0
        if setting == "confirmed" and placement in ("wrist", "pocket", "waist", "pelvis", "hand") and recent_motion:
            return "confirmed"
        if recent_motion:
            return "likely"
        return "unknown"

    # ------------------------------------------------------------------ fps
    def desired_fps(self, now: float) -> float:
        cam = self.cfg.acquisition.camera
        if self.recording_fps:
            return self.recording_fps
        if self.exercise is not None:
            return cam.exercise_fps
        if now < self._boost_until or getattr(self.fusion, "cases", None):
            return cam.event_fps
        return cam.base_fps

    def boost(self, now: float) -> None:
        self._boost_until = now + self.cfg.acquisition.camera.event_boost_s * 1000.0

    # ------------------------------------------------------------- exercise
    def start_exercise(self, session_id: str, target: int, definition: str | None = None, classifier: PhaseClassifier | None = None) -> None:
        with self.lock:
            self.exercise = ExerciseTracker(self.cfg.exercise, session_id, target, definition, classifier, self.cfg.acquisition.pose.min_keypoint_score)

    def stop_exercise(self, now: float) -> dict[str, Any] | None:
        with self.lock:
            if self.exercise is None:
                return None
            self.exercise.close(now)
            s = self.exercise.summary()
            self.exercise = None
            return s

    # ----------------------------------------------------------------- tick
    def tick(self, now: float, final: bool = False) -> PipelineOutput:
        with self.lock:
            out = PipelineOutput(now)
            evs = self.accel.update(now, truncated_flush=final) + self.vision.update(now, truncated_flush=final)
            for ev in evs:
                if ev.candidate:
                    self.fusion.on_evidence(ev, now)
                    self.boost(now)
                self.evidence_log.append({"t": now, "channel": ev.channel, "candidate": ev.candidate, "score": round(ev.score, 3), "q": round(ev.quality_score, 3), "reasons": ev.reason_codes, "peak": ev.peak_time_ms})
            out.evidences = evs
            decisions = self.fusion.tick(now, final=final)
            for d in decisions:
                self.decision_log.append({"t": now, "id": d.decision_id, "level": d.level.value, "rejected": d.rejected, "reasons": d.reason_codes, "fused": d.fused_score, "gate": d.gate})
                if d.level in (DecisionLevel.PENDING, DecisionLevel.EMERGENCY) and self.exercise is not None and not self.exercise.manual_paused:
                    self.exercise.suspend_for_fall(now)
            out.decisions = decisions
            self.stats["decisions"] += len(decisions)
            out.coverage = self.coverage.evaluate(now, self._accel_info(now), self._vision_info(now))
            out.exercise, self._pending_ex = self._pending_ex, []
            self.last_tick = now
            return out

    def _accel_info(self, now: float) -> dict[str, Any]:
        a = self.accel
        return {
            "has_device": self.phone.get("has_device", False),
            "last_recv_ms": a.last_recv_now,
            "last_sample_ms": a.last_raw_t,
            "rate_hz": a.rate_hz(),
            "clock_uncertainty_ms": a.clock_uncertainty_ms,
            "timestamp_untrusted": self.phone.get("timestamp_untrusted", False),
            "carriage": self.carriage(now),
            "placement": self.phone.get("placement"),
            "page_hidden": self.phone.get("page_hidden", False),
            "model_status": "ok",
        }

    def _vision_info(self, now: float) -> dict[str, Any]:
        v = self.vision
        conf_low = False
        if v.rows:
            last = v.rows[-1]
            c = last[VC["conf"]]
            conf_low = bool(np.isfinite(c) and c < 0.4)
        return {
            "camera_enabled": self.camera.get("camera_enabled", True),
            "camera_error": self.camera.get("camera_error"),
            "last_frame_ms": v.last_frame_t,
            "model_status": self.camera.get("model_status", "ok"),
            "last_target_ms": v.last_valid_t,
            "target_ambiguous": self.last_target_ambiguous,
            "low_pose_confidence": conf_low,
            "backlog": self.camera.get("backlog", False),
            "fps": self.camera.get("fps"),
            "backend": self.camera.get("backend"),
            "person_count": self.last_person_count,
        }

    # ---------------------------------------------------------------- debug
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            last = self.vision.last_obs
            return {
                "subject_id": self.subject_id,
                "source_mode": self.source_mode,
                "last_tick": self.last_tick,
                "accel": {
                    "series": self.accel.recent_series(),
                    "rate_hz": self.accel.rate_hz(),
                    "clock_uncertainty_ms": self.accel.clock_uncertainty_ms,
                    "carriage": self.carriage(self.last_tick or 0.0),
                    "scorer": f"{self.accel.scorer.model_id}@{self.accel.scorer.model_version}",
                    "last_eval": self.accel.last_eval,
                },
                "vision": {
                    "series": self.vision.recent_series(),
                    "scorer": f"{self.vision.scorer.model_id}@{self.vision.scorer.model_version}",
                    "last_eval": self.vision.last_eval,
                    "skeleton": None if last is None else {
                        "t": last.t_ms, "kp": kp_to_list(last.kp), "w": last.width, "h": last.height,
                        "track_id": last.track_id, "flags": last.flags, "persons": last.person_count, "backend": last.backend,
                    },
                    "camera": dict(self.camera),
                },
                "fusion": {"backend": getattr(self.fusion, "backend", "?"), "cases": self.fusion.snapshot() if hasattr(self.fusion, "snapshot") else []},
                "evidences": list(self.evidence_log)[-20:],
                "decisions": list(self.decision_log)[-20:],
                "coverage": {"status": self.coverage.status, "reasons": self.coverage.reasons, "channels": {k: v.to_dict() for k, v in self.coverage.channels.items()}},
                "exercise": None if self.exercise is None else {
                    "session_id": self.exercise.session_id, "count": self.exercise.counter.count, "phase": self.exercise.phase,
                    "sub_state": self.exercise.sub_state, "counter": self.exercise.counter.snapshot(),
                },
                "stats": dict(self.stats),
            }
