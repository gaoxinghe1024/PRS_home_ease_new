"""The single runtime instance: owns the pipeline, inputs and background loops.

Only one Runtime may own the camera and the escalation timers (run the server
with a single worker). CPU-bound inference runs in its own threads; the
FastAPI event loop only handles I/O.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil

from homeease.backend.services.coverage import CoverageService
from homeease.backend.services.devices import DeviceService
from homeease.backend.services.events import EventService
from homeease.backend.services.exercise import ExerciseService
from homeease.backend.services.notifications import NotificationService
from homeease.backend.services.realtime import Broadcaster
from homeease.backend.services.reports import ReportService
from homeease.backend.storage.db import Database
from homeease.backend.storage.models import Device, Subject
from homeease.core.clock import UTC, Clock, SystemClock
from homeease.core.config import AppConfig
from homeease.core.util import get_logger
from homeease.datasets.recording import Recording, load_recording
from homeease.edge.acquisition.phone import AccelIngestor, IngestResult
from homeease.edge.acquisition.replay import ReplayFeeder, ReplayItem
from homeease.edge.pipeline import MonitoringPipeline, PipelineOutput
from homeease.schemas.evidence import DecisionLevel
from homeease.schemas.observations import AccelBatch, TimeSyncReport

log = get_logger(__name__)


class ReplayClock:
    """Pipeline time follows the recording (scaled by speed); UTC follows the wall clock."""

    kind = "replay"

    def __init__(self, sys_clock: SystemClock, speed: float) -> None:
        self.sys = sys_clock
        self.speed = speed
        self.base_node = 1_000_000.0
        self.wall_start_mono = sys_clock.mono_ms()
        self.wall_start_utc = sys_clock.utc_now()
        self.paused_at: float | None = None
        self.paused_total = 0.0

    def restart(self) -> None:
        self.wall_start_mono = self.sys.mono_ms()
        self.wall_start_utc = self.sys.utc_now()
        self.paused_total = 0.0
        self.paused_at = None

    def pause(self) -> None:
        if self.paused_at is None:
            self.paused_at = self.sys.mono_ms()

    def resume(self) -> None:
        if self.paused_at is not None:
            self.paused_total += self.sys.mono_ms() - self.paused_at
            self.paused_at = None

    def rec_time(self) -> float:
        ref = self.paused_at if self.paused_at is not None else self.sys.mono_ms()
        return (ref - self.wall_start_mono - self.paused_total) * self.speed

    def mono_ms(self) -> float:
        return self.base_node + self.rec_time()

    def utc_now(self) -> datetime:
        return self.sys.utc_now()

    def mono_to_utc(self, mono_ms: float) -> datetime:
        return self.wall_start_utc + timedelta(milliseconds=(mono_ms - self.base_node) / self.speed + self.paused_total)

    def utc_to_mono(self, utc: datetime) -> float:
        return self.base_node + ((utc - self.wall_start_utc).total_seconds() * 1000.0 - self.paused_total) * self.speed


class ReplayDriver:
    """Feeds a recording into the live ingestion path in (scaled) real time."""

    def __init__(self, runtime: "Runtime", rec: Recording, clock: ReplayClock, loop: bool = False) -> None:
        self.rt = runtime
        self.rec = rec
        self.clock = clock
        self.loop = loop
        self.feeder = ReplayFeeder(rec, runtime.cfg, runtime.pipeline, runtime.ingestor, node_base_ms=clock.base_node)
        self.feeder._resolve_unknown_arrivals()
        self.queue: list[ReplayItem] = sorted(self.feeder.items)
        self.idx = 0
        self.finished = False
        self.faults: dict[str, bool] = {"drop_accel": False, "drop_vision": False}
        self.last_frame: Any = None  # last DetectionFrame delivered (technical-page preview)
        self.last_obs: Any = None  # tracked target for that frame
        self._renderer: Any = None

    def truth(self) -> dict[str, Any]:
        """Annotation of the recording, for the technical page's comparison only.
        Inference never reads labels.json."""
        lab = self.rec.labels()
        sec = lambda v: None if v is None else round(float(v) / 1000.0, 2)  # noqa: E731
        return {
            "source": lab.get("label_source"),
            "falls": [{"onset_s": sec(f.get("onset_ms")), "impact_s": sec(f.get("impact_ms")), "lying_s": sec(f.get("lying_start_ms")), "end_s": sec(f.get("end_ms"))} for f in lab.get("falls", [])],
            "activities": [{"activity": a.get("activity"), "start_s": sec(a.get("start_ms")), "end_s": sec(a.get("end_ms"))} for a in lab.get("adl", []) if isinstance(a, dict)],
            "kind": self.rec.manifest.get("kind") or self.rec.manifest.get("scenario"),
        }

    def has_video(self) -> bool:
        from homeease.edge.acquisition.preview import resolve_video

        return resolve_video(self.rec) is not None

    def render_preview(self) -> bytes | None:
        from homeease.edge.acquisition.preview import FrameRenderer, posture_of, resolve_video

        det, obs = self.last_frame, self.last_obs
        if det is None:
            return None
        if self._renderer is None:
            self._renderer = FrameRenderer(resolve_video(self.rec))
        kp = obs.kp if obs is not None and obs.target_visible and obs.frame_index == det.frame_index else None
        posture, angle = posture_of(kp, det.width, det.height)
        caption = f"{det.t_ms / 1000.0:5.1f} s | {posture}{f' {angle:.0f} deg' if angle is not None else ''} | persons {len(det.persons)}"
        if self.faults["drop_vision"]:
            caption += " | CAMERA CUT (fault)"
        return self._renderer.render(det, kp, posture, caption, self.rt.cfg.acquisition.pose.min_keypoint_score)

    def progress(self) -> dict[str, Any]:
        dur = max(self.rec.duration_ms(), 1.0)
        return {
            "recording_id": self.rec.recording_id, "source_mode": self.rec.source_mode, "rec_time_ms": round(self.clock.rec_time(), 1),
            "duration_ms": dur, "speed": self.clock.speed, "paused": self.clock.paused_at is not None, "finished": self.finished,
            "faults": self.faults, "delivered": self.idx, "total": len(self.queue),
        }

    def pump(self) -> None:
        t = self.clock.rec_time()
        while self.idx < len(self.queue) and self.queue[self.idx].arrival_ms <= t:
            it = self.queue[self.idx]
            self.idx += 1
            if it.kind == "accel" and self.faults["drop_accel"]:
                continue
            if it.kind == "frame" and self.faults["drop_vision"]:
                continue
            if it.kind == "exercise_start":
                continue  # sessions are started by the user through the API
            self.feeder._deliver(it)
            if it.kind == "frame":
                self.last_frame, self.last_obs = it.payload, self.rt.pipeline.vision.last_obs
        if self.idx >= len(self.queue) and not self.finished:
            self.finished = True
            if self.loop:
                self.restart()

    def restart(self) -> None:
        self.rt.pipeline_reset()
        self.feeder = ReplayFeeder(self.rec, self.rt.cfg, self.rt.pipeline, self.rt.ingestor, node_base_ms=self.clock.base_node)
        self.feeder._resolve_unknown_arrivals()
        self.queue = sorted(self.feeder.items)
        self.idx = 0
        self.finished = False
        self.last_frame = self.last_obs = None
        self.clock.restart()


class Runtime:
    def __init__(self, cfg: AppConfig, db: Database, clock: Clock | None = None, headless: bool = False) -> None:
        self.cfg = cfg
        self.db = db
        self.sys_clock: Any = clock or SystemClock()
        self.mode = cfg.runtime.mode
        self.source_mode = self.mode
        self.subject_id = cfg.runtime.subject_id
        self.room_id = cfg.runtime.room_id
        self.replay: ReplayDriver | None = None
        self.headless = headless
        if self.mode in ("replay", "synthetic") and not headless:
            self.clock: Any = ReplayClock(self.sys_clock, cfg.runtime.replay_speed)
        else:
            self.clock = self.sys_clock
        self.broadcaster = Broadcaster(db, self.clock)
        self.notifier = NotificationService(db, cfg, self.clock, self.broadcaster)
        self.events = EventService(db, cfg, self.clock, self.notifier, self.broadcaster)
        self.coverage = CoverageService(db, self.clock, self.broadcaster, self.subject_id, self.source_mode)
        self.exercise = ExerciseService(db, cfg, self.clock, self.broadcaster, self, self.source_mode)
        self.devices = DeviceService(db, self.clock)
        self.reports = ReportService(db, self.clock, self.events, self.exercise, cfg.runtime.timezone)
        self.ingestor = AccelIngestor(cfg)
        self.model_info: dict[str, Any] = {}
        self.pipeline = self._build_pipeline()
        self.camera: Any = None
        self.audio: Any = None
        self.sound: Any = None  # LiveSoundMonitor (daily sound monitoring); off unless audio.sound_monitor.enabled
        self.sound_start_error: str | None = None  # why the sound monitor could not be built: shown as degraded:start_failed
        self.audio_error: str | None = None  # why the voice inquiry service could not be built: shown as degraded:inquiry_unavailable
        self.stop_evt = threading.Event()
        self._threads: list[threading.Thread] = []
        self.errors: deque[dict[str, Any]] = deque(maxlen=50)
        self.health: dict[str, Any] = {"started_at": self.sys_clock.utc_now().isoformat(), "ticks": 0, "tick_lag_ms": 0.0, "last_tick_wall": None, "dispatch_runs": 0}
        self._last_live_pub = 0.0
        self._proc = psutil.Process()
        self.recovery: dict[str, Any] = {}
        self.recorder: Any = None  # ResearchRecorder during an explicit research recording (live only)
        self._rec_lock = threading.Lock()
        self._last_sync_reports: dict[tuple[str, str], tuple[TimeSyncReport, float]] = {}

    # ------------------------------------------------------------ pipeline
    def _build_pipeline(self) -> MonitoringPipeline:
        from homeease.ml.registry import build_fall_components

        accel_scorer, vision_scorer, fusion_kwargs, info = build_fall_components(self.cfg)
        self.model_info = info
        p = MonitoringPipeline(self.cfg, self.subject_id, self.room_id, self.source_mode, accel_scorer, vision_scorer, **fusion_kwargs)
        if self.mode != "live":
            # replay/synthetic: the recording's frames are the camera input (the live-camera switch does not apply)
            p.set_camera_status(camera_enabled=True)
        with self.db.session() as s:
            subj = s.get(Subject, self.subject_id)
            if subj is not None:
                p.set_phone_status(carriage_setting=subj.carriage_setting)
        return p

    def pipeline_reset(self) -> None:
        with self.pipeline.lock:
            self.pipeline = self._build_pipeline()
            self.ingestor = AccelIngestor(self.cfg)
            if self.camera is not None:
                self.camera.pipeline = self.pipeline

    def pipeline_now(self) -> float:
        return float(self.clock.mono_ms())

    # ----------------------------------------------------- TrackerControl
    def start_tracker(self, session_id: str, target: int, definition: str) -> None:
        from homeease.ml.registry import build_phase_classifier

        clf = build_phase_classifier(self.cfg)
        self.pipeline.start_exercise(session_id, target, definition, clf)

    def pause_tracker(self, session_id: str) -> None:
        ex = self.pipeline.exercise
        if ex is not None and ex.session_id == session_id:
            with self.pipeline.lock:
                ex.pause(self.pipeline_now())

    def resume_tracker(self, session_id: str) -> None:
        ex = self.pipeline.exercise
        if ex is not None and ex.session_id == session_id:
            with self.pipeline.lock:
                ex.resume(self.pipeline_now())

    def stop_tracker(self, session_id: str) -> dict[str, Any] | None:
        ex = self.pipeline.exercise
        if ex is None or ex.session_id != session_id:
            return None
        return self.pipeline.stop_exercise(self.pipeline_now())

    def tracker_mono_ms(self) -> float:
        return self.pipeline_now()

    # ------------------------------------------------------------- inputs
    def ingest_accel_batch(self, batch: AccelBatch, device: Device) -> IngestResult:
        if batch.device_id != device.id or batch.subject_id != device.subject_id:
            raise PermissionError("batch device/subject does not match the authenticated device")
        if self.mode != "live" and batch.source_mode.value == "live":
            # In replay/synthetic mode a real phone stream would mix real and simulated data.
            return IngestResult(False, batch.sequence_no, flags=["mode_mismatch"], message="node is not in live mode")
        now = self.pipeline_now()
        res = self.ingestor.ingest(batch, now)
        rec = self.recorder
        if rec is not None and not res.duplicate:
            rec.on_accel(batch, now)
        if res.t_node is not None and len(res.t_node):
            self.pipeline.ingest_accel(
                res.t_node, res.axes_g, res.includes_gravity, device_id=batch.device_id, session_id=batch.session_id,
                clock_uncertainty_ms=res.clock_uncertainty_ms, now=now, placement=batch.placement,
                timestamp_untrusted="timestamp_untrusted" in res.flags,
            )
        else:
            self.pipeline.set_phone_status(has_device=True, placement=batch.placement, timestamp_untrusted="timestamp_untrusted" in res.flags)
        cq = batch.client_quality or {}
        if "page_hidden" in cq:
            self.pipeline.set_phone_status(page_hidden=bool(cq.get("page_hidden")))
        self.devices.touch(device.id, data=True)
        return res

    def register_time_sync(self, device: Device, report: TimeSyncReport) -> dict[str, Any]:
        now = self.pipeline_now()
        res = self.ingestor.register_sync(device.id, report, now)
        self._last_sync_reports[(device.id, report.session_id)] = (report, now)
        rec = self.recorder
        if rec is not None:
            rec.on_sync(device.id, report, now)
        self.devices.touch(device.id, meta={"last_sync": {"sync_id": res.sync_id, "rtt_ms": round(res.rtt_ms, 1), "uncertainty_ms": round(res.base_uncertainty_ms, 1), "at": self.sys_clock.utc_now().isoformat()}})
        return {"sync_id": res.sync_id, "offset_ms": res.offset_ms, "rtt_ms": res.rtt_ms, "uncertainty_ms": res.base_uncertainty_ms, "accepted": res.base_uncertainty_ms <= self.cfg.quality.max_clock_uncertainty_ms}

    # ------------------------------------------------- research recording
    def start_research_recording(self, label: str, subject: str, notes: str = "", save_video: bool = False, consent_by: str | None = None, max_duration_s: float = 600.0) -> dict[str, Any]:
        """Explicit, bounded recording of the live inputs into data/captures/ (admin action)."""
        from homeease.core.paths import resolve
        from homeease.edge.acquisition.recorder import ResearchRecorder, new_recording_id

        if self.mode != "live":
            raise PermissionError("research recordings capture live inputs only (node is not in live mode)")
        with self._rec_lock:
            if self.recorder is not None:
                raise RuntimeError(f"already recording {self.recorder.recording_id}")
            now = self.pipeline_now()
            max_age = self.cfg.quality.clock_sync_max_age_s * 1000.0
            prior = [(dev, rep, t) for (dev, _), (rep, t) in self._last_sync_reports.items() if now - t <= max_age]
            rid = new_recording_id(label)
            rec = ResearchRecorder(
                resolve(self.cfg.privacy.research_recording_dir) / rid, recording_id=rid, label=label, subject=subject, notes=notes, start_node_ms=now,
                start_utc=self.sys_clock.utc_now().isoformat(), room_id=self.room_id, camera_device_id="camera-01",
                pose_backend=(self.camera.backend.name if self.camera is not None and self.camera.backend is not None else None),
                save_video=save_video, consent_confirmed_by=consent_by, max_duration_s=max_duration_s, config_hash=self.cfg.hash(),
            )
            rec.start(prior)
            self.recorder = rec
            self.pipeline.recording_fps = self.cfg.acquisition.camera.event_fps
            if self.camera is not None:
                self.camera.recorder = rec
        log.info("research recording started", extra={"ctx": {"recording_id": rid, "label": label, "video": save_video}})
        return rec.status(now)

    def stop_research_recording(self, reason: str = "manual") -> dict[str, Any] | None:
        with self._rec_lock:
            rec = self.recorder
            if rec is None:
                return None
            if self.camera is not None:
                self.camera.recorder = None
            self.pipeline.recording_fps = None
            self.recorder = None
        return rec.stop(self.pipeline_now(), reason)

    def set_carriage(self, setting: str) -> None:
        self.pipeline.set_phone_status(carriage_setting=setting)

    # ----------------------------------------------------------- lifecycle
    def recover(self) -> dict[str, Any]:
        r = {
            "notifications_requeued": self.notifier.recover(),
            "exercise_interrupted": self.exercise.recover(),
        }
        self.coverage.recover()
        self.recovery = r
        self.broadcaster.publish("node_started", {"mode": self.mode, "recovery": r}, self.subject_id, audience="all", durable=True)
        return r

    def start(self) -> None:
        self.recover()
        self._prune_sound_events()
        cam_cfg = self.cfg.acquisition.camera
        if self.mode == "live":
            if cam_cfg.enabled:
                from homeease.edge.acquisition.camera import CameraWorker

                self.devices.ensure_camera(self.subject_id, "camera-01", "Fixed camera", str(cam_cfg.source))
                hook = self._preview_hook if self.cfg.debug.enabled and self.cfg.debug.skeleton_preview == "local_only" else None
                self.camera = CameraWorker(self.cfg, self.clock, self.pipeline, self.subject_id, "camera-01", on_frame=hook)
                self.camera.start()
            else:
                self.pipeline.set_camera_status(camera_enabled=False)
        else:
            rec_path = self.cfg.runtime.replay_recording
            if self.mode == "synthetic" and not rec_path:
                from homeease.datasets.synthetic import generate_scenario
                from homeease.core.paths import data_dir

                rec_path = str(generate_scenario("fall_both", 42, data_dir() / "synthetic" / "runtime"))
            if rec_path:
                self.start_replay(rec_path)
        self._spawn(self._tick_loop, "runtime-tick")
        self._spawn(self._dispatch_loop, "runtime-dispatch")
        if self.cfg.audio.enabled:
            try:
                from homeease.ml.audio.inquiry import AudioInquiryService

                self.audio = AudioInquiryService(self)
            except Exception as e:  # optional research layer: never blocks the node, but never silent either
                log.warning("audio layer unavailable: %s", e)
                self.audio = None
                self.audio_error = f"{type(e).__name__}: {str(e)[:150]}"
                self.errors.append({"at": self.sys_clock.utc_now().isoformat(), "where": "inquiry_service", "error": self.audio_error})
            if self.cfg.audio.sound_monitor.enabled:
                self._start_sound_monitor()

    # ----------------------------------------------------- sound monitoring
    def _prune_sound_events(self) -> None:
        """Retention (privacy.retention_days_events) also applies while the monitor is switched off."""
        try:
            from homeease.ml.audio.monitor import SoundEventStore

            n = SoundEventStore(self.db, self.subject_id, clock=self.sys_clock).prune(self.cfg.privacy.retention_days_events)
            if n:
                log.info("sound events: %d older than %d days deleted", n, self.cfg.privacy.retention_days_events)
        except Exception as e:  # pragma: no cover - retention must never block start-up
            log.warning("sound event retention failed: %s", type(e).__name__)

    def audio_consent(self) -> bool:
        """Latest recorded audio consent of the subject. Re-read by the sound monitor (every 10 s) and by the voice
        inquiry (before it opens the microphone), so a withdrawal takes effect without a restart."""
        return bool(self.audio_consent_info()["granted"])

    def audio_consent_info(self) -> dict[str, Any]:
        """The latest audio consent row: granted or not, who recorded it (role only) and when. The consent can be
        recorded by the person or by a caregiver (the caregiver settings page has the switch), so it is shown."""
        from sqlalchemy import select

        from homeease.backend.storage.models import Consent, User

        with self.db.session() as s:
            c = s.execute(select(Consent).where(Consent.subject_id == self.subject_id, Consent.kind == "audio").order_by(Consent.created_at.desc(), Consent.id.desc())).scalars().first()
            if c is None:
                return {"granted": False, "recorded_by_role": None, "at": None}
            u = s.get(User, c.granted_by) if c.granted_by is not None else None
            return {"granted": bool(c.granted), "recorded_by_role": u.role if u is not None else None, "at": c.created_at.isoformat() if c.created_at else None}

    def _start_sound_monitor(self) -> None:
        """Daily sound monitoring (branch B). Live mode only; any failure is a degraded status, never a crash."""
        if self.mode != "live":
            self.coverage.record_audio("off", ["not_live_mode"])
            return
        try:
            from homeease.ml.audio.live import LiveSoundMonitor

            self.sound = LiveSoundMonitor(self.cfg, self.db, self.clock, self.subject_id, self.room_id, consent=self.audio_consent, on_state=self._on_sound_state)
            self.sound.start()
        except Exception as e:
            log.exception("sound monitor could not start")
            self.sound = None
            self.sound_start_error = f"{type(e).__name__}: {str(e)[:150]}"
            self._on_sound_state("degraded", ["start_failed"])
            self.errors.append({"at": self.sys_clock.utc_now().isoformat(), "where": "sound_monitor", "error": repr(e)})

    def _on_sound_state(self, state: str, reasons: list[str]) -> None:
        self.coverage.record_audio(state, reasons)
        if state == "degraded":
            self.errors.append({"at": self.sys_clock.utc_now().isoformat(), "where": "sound_monitor", "error": f"degraded:{','.join(reasons)}"})

    def sound_status(self) -> dict[str, Any]:
        if self.sound is not None:
            return self.sound.status()
        sm = self.cfg.audio.sound_monitor
        enabled = bool(sm.enabled and self.cfg.audio.enabled)
        if enabled and self.sound_start_error:  # a monitor that was asked for but could not be built is a failure, not "off"
            return {"enabled": True, "status": "degraded:start_failed", "state": "degraded", "reasons": ["start_failed"], "error": self.sound_start_error, "raw_audio_stored": False}
        reason = "disabled" if not enabled else "not_live_mode" if self.mode != "live" else "not_started"
        return {"enabled": enabled, "status": "off", "state": "off", "reasons": [reason], "raw_audio_stored": False}

    def audio_status(self) -> dict[str, Any]:
        """The existing audio status (inquiry layer) extended with the sound monitor."""
        if self.audio is not None:
            base = self.audio.status()
        elif self.cfg.audio.enabled and self.audio_error:  # enabled but the service is down: say so
            base = {"enabled": True, "status": "degraded:inquiry_unavailable", "error": self.audio_error}
        else:
            base = {"enabled": bool(self.cfg.audio.enabled)}
        return {**base, "sound_monitor": self.sound_status(), "consent_record": self.audio_consent_info()}

    def _preview_hook(self, frame: Any, obs: Any, det: Any = None) -> None:
        """Keep one small JPEG with the skeleton overlay in memory (never on disk) for the
        local technical page; refreshed at most 5 times per second."""
        now = time.monotonic()
        if now - getattr(self, "_last_preview", 0.0) < 0.2 or self.camera is None:
            return
        self._last_preview = now
        import cv2

        from homeease.edge.acquisition.preview import OTHER_BGR, POSTURE_BGR, draw_skeleton, posture_of

        img = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])))
        h, w = img.shape[:2]
        for p in (det.persons if det is not None else []):
            draw_skeleton(img, p, OTHER_BGR, 1, 0.2)
        kp = obs.kp if obs.target_visible else None
        posture, angle = posture_of(kp, frame.shape[1], frame.shape[0])
        if kp is not None:
            draw_skeleton(img, kp, POSTURE_BGR[posture], 3, self.cfg.acquisition.pose.min_keypoint_score)
        cv2.rectangle(img, (0, h - 24), (w, h), (0, 0, 0), -1)
        caption = f"LIVE | {posture}{f' {angle:.0f} deg' if angle is not None else ''} | persons {obs.person_count}"
        cv2.putText(img, caption, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.camera.preview_jpeg = buf.tobytes()

    def preview_jpeg(self) -> bytes | None:
        """Frame with the skeleton overlay for the technical page: live camera or replayed video."""
        if self.mode == "live":
            return self.camera.preview_jpeg if self.camera is not None else None
        return self.replay.render_preview() if self.replay is not None else None

    def viewer_state(self) -> dict[str, Any]:
        """What the technical page's detection viewer needs besides the pipeline snapshot."""
        from homeease.edge.acquisition.preview import posture_of

        now = self.pipeline_now()
        obs = self.pipeline.vision.last_obs
        kp = obs.kp if obs is not None and obs.target_visible else None
        posture, angle = posture_of(kp, obs.width if obs is not None else 1, obs.height if obs is not None else 1)
        out: dict[str, Any] = {
            "now_ms": now, "posture": posture, "torso_deg": angle, "person_visible": kp is not None,
            "persons": obs.person_count if obs is not None else 0,
        }
        if self.replay is not None:
            out.update(source="replay", t0_ms=self.clock.base_node, duration_ms=self.replay.rec.duration_ms(), truth=self.replay.truth(),
                       video=self.replay.has_video(), recording_id=self.replay.rec.recording_id, finished=self.replay.finished)
        else:
            out.update(source=self.mode, t0_ms=now - 60_000.0, duration_ms=60_000.0, truth=None, video=self.camera is not None, recording_id=None, finished=False)
        return out

    def start_replay(self, path: str, speed: float | None = None, loop: bool | None = None) -> dict[str, Any]:
        if self.mode == "live":
            raise PermissionError("replay is not allowed while the node is in live mode")
        from homeease.core.paths import resolve

        p = resolve(path)
        if p is None or not (p / "manifest.json").is_file():
            raise FileNotFoundError(f"recording not found: {path}")
        rec = load_recording(p)
        if speed:
            self.clock.speed = float(speed)
        self.pipeline_reset()
        self.clock.restart()
        self.replay = ReplayDriver(self, rec, self.clock, self.cfg.runtime.replay_loop if loop is None else loop)
        self.source_mode = "synthetic" if rec.source_mode == "synthetic" else "replay"
        return self.replay.progress()

    def _spawn(self, fn: Any, name: str) -> None:
        t = threading.Thread(target=self._guard(fn), name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _guard(self, fn: Any) -> Any:
        def run() -> None:
            while not self.stop_evt.is_set():
                try:
                    fn()
                    return
                except Exception as e:  # keep loops alive; record the fault
                    log.exception("runtime loop %s crashed; restarting", fn.__name__)
                    self.errors.append({"at": self.sys_clock.utc_now().isoformat(), "loop": fn.__name__, "error": repr(e)})
                    self.stop_evt.wait(1.0)

        return run

    def stop(self) -> None:
        self.stop_evt.set()
        if self.recorder is not None:
            self.stop_research_recording("node_stopped")
        if self.camera is not None:
            self.camera.stop()
        if self.audio is not None:
            try:
                self.audio.stop()
            except Exception:  # pragma: no cover
                pass
        if self.sound is not None:
            try:
                self.sound.stop()
            except Exception:  # pragma: no cover
                log.exception("sound monitor stop failed")
        for t in self._threads:
            t.join(timeout=3)
        self.coverage.close_all("node_stopped")

    # --------------------------------------------------------------- loops
    def _tick_loop(self) -> None:
        tick = self.cfg.runtime.tick_ms / 1000.0
        next_t = time.monotonic()
        while not self.stop_evt.is_set():
            if self.replay is not None:
                self.replay.pump()
            now = self.pipeline_now()
            t0 = time.perf_counter()
            out = self.pipeline.tick(now)
            self.handle(out)
            rec = self.recorder
            if rec is not None and rec.expired(now):
                self.stop_research_recording("max_duration")
            self.health["ticks"] += 1
            self.health["tick_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            self.health["last_tick_wall"] = self.sys_clock.utc_now().isoformat()
            next_t += tick
            lag = time.monotonic() - next_t
            self.health["tick_lag_ms"] = round(max(0.0, lag) * 1000.0, 1)
            if lag > 1.0:
                next_t = time.monotonic()
            self.stop_evt.wait(max(0.0, next_t - time.monotonic()))

    def handle(self, out: PipelineOutput) -> None:
        for d in out.decisions:
            try:
                ev = self.events.ingest_decision(d)
            except Exception as e:
                log.exception("ingest_decision failed")
                self.errors.append({"at": self.sys_clock.utc_now().isoformat(), "where": "ingest_decision", "error": repr(e)})
                continue
            if d.level in (DecisionLevel.PENDING, DecisionLevel.EMERGENCY):
                active = self.exercise.active_id
                if active:
                    self.exercise.mark_fall_suspended(active)
                if ev is not None and self.audio is not None:
                    self.audio.on_event(ev.id, d.level.value)
        if out.coverage is not None:
            self.coverage.record(out.coverage)
        for u in out.exercise:
            self.exercise.apply_update(u)
        mono = time.monotonic()
        if mono - self._last_live_pub >= 1.0:
            self._last_live_pub = mono
            self.broadcaster.publish("live_status", self.live_status(), self.subject_id, audience="all", durable=False)

    def _dispatch_loop(self) -> None:
        while not self.stop_evt.is_set():
            self.notifier.dispatch_due()
            self.events.run_due_tasks()
            self.health["dispatch_runs"] += 1
            self.notifier.wait_kick(0.5)

    # --------------------------------------------------------------- status
    def live_status(self) -> dict[str, Any]:
        ex = self.pipeline.exercise
        return {
            "coverage": self.coverage.current,
            "mode": self.mode,
            "source_mode": self.source_mode,
            "exercise": None if ex is None else {"session_id": ex.session_id, "count": ex.counter.count, "target": ex.target, "phase": ex.phase, "sub_state": ex.sub_state},
            "replay": self.replay.progress() if self.replay else None,
        }

    def resources(self) -> dict[str, Any]:
        with self._proc.oneshot():
            mem = self._proc.memory_info().rss / 1e6
            cpu = self._proc.cpu_percent(interval=None)
            threads = self._proc.num_threads()
        return {"rss_mb": round(mem, 1), "cpu_percent": cpu, "threads": threads}

    def readiness(self) -> dict[str, Any]:
        alive = all(t.is_alive() for t in self._threads) if self._threads else False
        last = self.health.get("last_tick_wall")
        fresh = False
        if last:
            fresh = (self.sys_clock.utc_now() - datetime.fromisoformat(last)).total_seconds() < 3.0
        cam = None
        if self.camera is not None:
            cam = dict(self.camera.status)
        return {
            "ready": alive and fresh,
            "threads_alive": alive,
            "tick_fresh": fresh,
            "mode": self.mode,
            "camera": cam,
            "pose_model": self.pipeline.camera.get("model_status"),
            "models": self.model_info,
            "recent_errors": list(self.errors)[-5:],
        }
