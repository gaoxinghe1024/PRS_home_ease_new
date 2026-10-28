"""Explicit research recording of the live inputs (team data collection, prompt §15.2 / §15.5).

While the node runs in live mode, an admin can record what the pipeline actually receives:

* every phone batch as received (device timebase), with its node arrival time;
* every clock-sync report, including the latest one before the start, so the complete offset
  history is kept;
* every processed camera frame, with all detected persons stored before target association;
* optionally the raw video, only with explicit consent.

The output is the canonical recording format under ``data/captures/<recording_id>/``. Replaying
it runs the same pipeline on the same inputs, and `homeease capture label` adds the manual
labels. A provisional manifest is written at the start, so a crash leaves a usable folder.
Recordings are bounded in duration.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from homeease.core.util import code_version, dumps, get_logger
from homeease.datasets.recording import DetectionFrame
from homeease.schemas.observations import AccelBatch, TimeSyncReport

log = get_logger(__name__)

CALIBRATION_INSTRUCTION = (
    "At the start, in view of the camera: hold the phone in the hand that will carry it and tap it "
    "firmly against the other palm three times, then put it in its placement. Label the first tap "
    "with `homeease capture label --calibration <seconds>` to check the clock alignment offline."
)


class ResearchRecorder:
    def __init__(
        self,
        out_dir: Path,
        *,
        recording_id: str,
        label: str,
        subject: str,
        notes: str,
        start_node_ms: float,
        start_utc: str,
        room_id: str | None,
        camera_device_id: str,
        pose_backend: str | None,
        save_video: bool,
        consent_confirmed_by: str | None,
        max_duration_s: float,
        config_hash: str,
    ) -> None:
        self.dir = Path(out_dir)
        self.recording_id = recording_id
        self.label = label
        self.subject = subject
        self.notes = notes
        self.t0 = float(start_node_ms)
        self.start_utc = start_utc
        self.room_id = room_id
        self.camera_device_id = camera_device_id
        self.pose_backend = pose_backend
        self.save_video = save_video
        self.consent_by = consent_confirmed_by
        self.max_ms = float(max_duration_s) * 1000.0
        self.config_hash = config_hash
        self.lock = threading.Lock()
        self.syncs: list[dict[str, Any]] = []
        self.arrivals: dict[str, float] = {}
        self.n_batches = 0
        self.n_samples = 0
        self.n_frames = 0
        self.frames_with_person = 0
        self.placement: str | None = None
        self.unit: str | None = None
        self.includes_gravity: bool | None = None
        self.sample_ts: list[float] = []
        self.frame_ts: list[float] = []
        self.last_rel_ms = 0.0
        self.closed = False
        self.stop_reason: str | None = None
        self._accel_fh: Any = None
        self._det_fh: Any = None
        self._video: Any = None
        self._seen: set[tuple[str, int]] = set()

    # ------------------------------------------------------------------ lifecycle
    def start(self, prior_syncs: list[tuple[str, TimeSyncReport, float]]) -> None:
        self.dir.mkdir(parents=True, exist_ok=False)
        self._accel_fh = open(self.dir / "accel.jsonl", "w", encoding="utf-8")
        self._det_fh = open(self.dir / "detections.jsonl", "w", encoding="utf-8")
        for dev, report, node_ms in prior_syncs:
            self._add_sync(dev, report, node_ms)
        (self.dir / "labels.json").write_text(dumps({
            "label_source": "manual (to be completed with `homeease capture label`)", "falls": [], "adl": [], "reps": [],
            "rep_count": None, "calibration_marker_ms": None,
            "instructions": "Fill reps with end times of each complete sit->rise->stand->descend->sit cycle (seconds from start), falls with onset times, and the first calibration tap.",
        }, indent=2), encoding="utf-8")
        self._write_manifest("recording")

    def rel(self, node_ms: float) -> float:
        return float(node_ms) - self.t0

    def expired(self, node_ms: float) -> bool:
        return not self.closed and self.rel(node_ms) >= self.max_ms

    def stop(self, node_ms: float, reason: str = "manual") -> dict[str, Any]:
        with self.lock:
            if self.closed:
                return self.status(node_ms)
            self.closed = True
            self.stop_reason = reason
            self.last_rel_ms = max(self.last_rel_ms, self.rel(node_ms))
            for fh in (self._accel_fh, self._det_fh):
                if fh is not None:
                    fh.close()
            if self._video is not None:
                self._video.release()
                self._video = None
            self._write_manifest("complete")
        log.info("research recording stopped", extra={"ctx": {"recording_id": self.recording_id, "reason": reason, "batches": self.n_batches, "frames": self.n_frames}})
        return self.status(node_ms)

    # ------------------------------------------------------------------ inputs
    def _add_sync(self, device_id: str, report: TimeSyncReport, node_ms: float) -> None:
        shifted = report.model_copy(deep=True)
        for s in shifted.samples:
            s.t1_server_mono_ms -= self.t0
            s.t2_server_mono_ms -= self.t0
        self.syncs.append({"device_id": device_id, "report": json.loads(shifted.model_dump_json()), "received_at_ms": round(self.rel(node_ms), 3)})

    def on_sync(self, device_id: str, report: TimeSyncReport, node_ms: float) -> None:
        with self.lock:
            if not self.closed:
                self._add_sync(device_id, report, node_ms)

    def on_accel(self, batch: AccelBatch, node_ms: float) -> None:
        with self.lock:
            key = (batch.session_id, batch.sequence_no)
            if self.closed or key in self._seen:
                return
            self._seen.add(key)
            self._accel_fh.write(batch.model_dump_json() + "\n")
            self.arrivals[f"{batch.session_id}:{batch.sequence_no}"] = round(self.rel(node_ms), 3)
            self.n_batches += 1
            self.n_samples += len(batch.samples)
            self.placement = self.placement or batch.placement
            self.unit = self.unit or batch.unit
            if self.includes_gravity is None:
                self.includes_gravity = batch.includes_gravity
            if len(self.sample_ts) < 20000:
                self.sample_ts.extend(s.t_ms for s in batch.samples)
            self.last_rel_ms = max(self.last_rel_ms, self.rel(node_ms))

    def on_frame(self, df: DetectionFrame, frame: np.ndarray | None) -> None:
        with self.lock:
            if self.closed:
                return
            rel = self.rel(df.t_ms)
            if rel < 0:
                return
            out = DetectionFrame(rel, self.n_frames, df.width, df.height, df.persons, df.backend, df.inference_ms, df.camera_ok)
            self._det_fh.write(json.dumps(out.to_json()) + "\n")
            if self.save_video and frame is not None:
                if self._video is None:
                    import cv2

                    self._video = cv2.VideoWriter(str(self.dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (frame.shape[1], frame.shape[0]))
                self._video.write(frame)
            self.n_frames += 1
            self.frames_with_person += 1 if df.persons else 0
            self.frame_ts.append(rel)
            self.last_rel_ms = max(self.last_rel_ms, rel)

    # ------------------------------------------------------------------ output
    def _rate(self, ts: list[float]) -> float | None:
        if len(ts) < 3:
            return None
        d = np.diff(np.sort(np.asarray(ts)))
        d = d[d > 0]
        return round(1000.0 / float(np.median(d)), 2) if len(d) else None

    def _write_manifest(self, status: str) -> None:
        mods = (["accel"] if self.n_batches else []) + (["pose"] if self.n_frames else []) + (["video"] if self.save_video and self.n_frames else [])
        manifest = {
            "recording_id": self.recording_id, "dataset": "own_capture", "source_mode": "replay", "captured_from": "live",
            "status": status, "stop_reason": self.stop_reason, "label": self.label, "kind": self.label,
            "subject_id": self.subject, "group_id": self.subject, "room_id": self.room_id, "start_utc": self.start_utc,
            "duration_ms": round(self.last_rel_ms, 1), "placement": self.placement or "unknown", "units": self.unit or "m_s2",
            "accel_includes_gravity": self.includes_gravity if self.includes_gravity is not None else True,
            "sample_rate_hz": self._rate(self.sample_ts), "fps": self._rate(self.frame_ts), "joint_layout": "coco17-v1",
            "timebase": {"camera": "recording_ms (node monotonic - start)", "accel": "device_monotonic_ms"},
            "sync_method": "NTP-style ping/pong between phone and node; every report recorded (clock_sync)",
            "clock_sync": self.syncs, "accel_arrivals_ms": self.arrivals,
            "detections_file": "detections.jsonl", "pose_backend": self.pose_backend, "camera_device_id": self.camera_device_id,
            "available_modalities": mods, "counts": {"batches": self.n_batches, "samples": self.n_samples, "frames": self.n_frames, "frames_with_person": self.frames_with_person},
            "raw_video_saved": bool(self.save_video and self.n_frames), "consent": {"video": self.save_video, "confirmed_by": self.consent_by},
            "calibration": {"instruction": CALIBRATION_INSTRUCTION},
            "conditions": {"notes": self.notes},
            "exercise_session": {"start_ms": 0.0, "target": 10, "definition": "full_cycle"} if self.label == "sit_to_stand" else None,
            "license": "own recording (team); participant consent required", "code_version": code_version(), "config_hash": self.config_hash,
        }
        tmp = self.dir / "manifest.json.part"
        tmp.write_text(dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(self.dir / "manifest.json")

    def status(self, node_ms: float | None = None) -> dict[str, Any]:
        elapsed = self.last_rel_ms if (self.closed or node_ms is None) else self.rel(node_ms)
        return {
            "recording_id": self.recording_id, "path": str(self.dir), "label": self.label, "subject": self.subject,
            "active": not self.closed, "stop_reason": self.stop_reason, "elapsed_s": round(elapsed / 1000.0, 1),
            "max_duration_s": round(self.max_ms / 1000.0), "batches": self.n_batches, "samples": self.n_samples,
            "frames": self.n_frames, "frames_with_person": self.frames_with_person, "clock_syncs": len(self.syncs),
            "video": self.save_video,
        }


def new_recording_id(label: str) -> str:
    return f"cap-{time.strftime('%Y%m%d-%H%M%S')}-{label}"
