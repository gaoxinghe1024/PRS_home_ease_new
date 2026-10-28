"""Replay of canonical recordings through the live ingestion path.

Every item is delivered at its *arrival* time on the recording time axis
(accel batches: last sample + transport latency, or the recorded arrival;
camera frames: capture time + inference time). The pipeline is ticked on the
same axis, so detection latency is measured in recording time regardless of
replay speed.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np

from homeease.core.config import AppConfig
from homeease.datasets.recording import DetectionFrame, Recording
from homeease.edge.acquisition.phone import AccelIngestor
from homeease.edge.pipeline import MonitoringPipeline, PipelineOutput
from homeease.edge.tracking import TargetTracker
from homeease.ml.pose.layout import PoseObs


def score_scale_for(cfg: AppConfig, backend: str) -> float:
    fam = backend.split("_")[0] if backend else ""
    return float(cfg.acquisition.pose.score_scale.get(fam, 1.0))


def frame_to_obs(
    frame: DetectionFrame,
    tracker: TargetTracker,
    t_node: float,
    *,
    subject_id: str,
    device_id: str,
    session_id: str,
    source_mode: str,
    score_scale: float = 1.0,
) -> PoseObs:
    persons = frame.persons
    if score_scale != 1.0:
        persons = []
        for p in frame.persons:
            q = p.copy()
            q[:, 2] = np.clip(q[:, 2] * score_scale, 0.0, 1.0)
            persons.append(q)
    dets = tracker.detections_from_kps(persons)
    res = tracker.update(dets, t_node)
    if res.detection is None:
        return PoseObs(
            t_ms=t_node, kp=np.full((17, 3), np.nan), width=frame.width, height=frame.height, frame_index=frame.frame_index,
            track_id=None, target_visible=False, target_conf=0.0, person_count=res.person_count, flags=res.flags,
            backend=frame.backend, device_id=device_id, session_id=session_id, source_mode=source_mode,
            inference_ms=frame.inference_ms,
        )
    return PoseObs(
        t_ms=t_node, kp=res.detection.kp, width=frame.width, height=frame.height, frame_index=frame.frame_index,
        track_id=res.track_id, target_visible=True, target_conf=res.confidence, person_count=res.person_count,
        flags=res.flags, backend=frame.backend, device_id=device_id, session_id=session_id, source_mode=source_mode,
        inference_ms=frame.inference_ms, bbox=res.detection.bbox,
    )


@dataclass
class ReplayItem:
    arrival_ms: float
    order: int
    kind: str
    payload: Any

    def __lt__(self, other: "ReplayItem") -> bool:
        return (self.arrival_ms, self.order) < (other.arrival_ms, other.order)


class ReplayFeeder:
    def __init__(
        self,
        rec: Recording,
        cfg: AppConfig,
        pipeline: MonitoringPipeline,
        ingestor: AccelIngestor | None = None,
        node_base_ms: float = 1_000_000.0,
        tick_ms: float | None = None,
        accel_latency_ms: float = 100.0,
        drop_accel: bool = False,
        drop_vision: bool = False,
        on_exercise_start: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.rec = rec
        self.cfg = cfg
        self.pipeline = pipeline
        self.ingestor = ingestor or AccelIngestor(cfg)
        self.base = node_base_ms
        self.tick_ms = tick_ms or float(cfg.runtime.tick_ms)
        self.tracker = TargetTracker(cfg.acquisition.zone, min_score=cfg.acquisition.pose.min_keypoint_score)
        self.subject_id = pipeline.subject_id
        self.source_mode = rec.source_mode if rec.source_mode in ("live", "replay", "synthetic") else "replay"
        self.camera_device = rec.manifest.get("camera_device_id", "camera-01")
        self.camera_session = f"{rec.recording_id}-cam"
        self.drop_accel = drop_accel
        self.drop_vision = drop_vision
        self.on_exercise_start = on_exercise_start
        self.items: list[ReplayItem] = []
        self._build(accel_latency_ms)
        self.ingest_log: list[dict[str, Any]] = []
        self.exercise_started = False

    def _build(self, latency: float) -> None:
        order = 0
        arrivals = self.rec.manifest.get("accel_arrivals_ms", {})
        for dev, report, recv in self.rec.time_sync_reports():
            self.items.append(ReplayItem(recv, order, "sync", (dev, report)))
            order += 1
        fixed = self.rec.manifest.get("accel_fixed_offset_ms")
        if fixed is not None:
            self.items.append(ReplayItem(-1.0, order, "fixed_sync", float(fixed)))
            order += 1
        if not self.drop_accel:
            for b in self.rec.accel:
                arr = arrivals.get(f"{b.session_id}:{b.sequence_no}", arrivals.get(str(b.sequence_no)))
                if arr is None:
                    last = b.samples[-1].t_ms
                    if b.timebase == "device_monotonic_ms" and fixed is None:
                        # Without a recorded arrival time, estimate from the sync offset later.
                        arr = None
                    else:
                        arr = last + (float(fixed) if fixed is not None else 0.0) + latency
                self.items.append(ReplayItem(arr if arr is not None else -2.0, order, "accel", b))
                order += 1
        if not self.drop_vision:
            for f in self.rec.detections:
                self.items.append(ReplayItem(f.t_ms + (f.inference_ms or 0.0), order, "frame", f))
                order += 1
        ex = self.rec.manifest.get("exercise_session")
        if ex:
            self.items.append(ReplayItem(float(ex.get("start_ms", 0.0)), order, "exercise_start", ex))
            order += 1
        self.items.sort()

    def _deliver(self, item: ReplayItem) -> None:
        p = self.pipeline
        now = item.arrival_ms + self.base
        if item.kind == "sync":
            dev, report = item.payload
            shifted = report.model_copy(deep=True)
            for s in shifted.samples:
                s.t1_server_mono_ms += self.base
                s.t2_server_mono_ms += self.base
            try:
                self.ingestor.register_sync(dev, shifted, now)
            except ValueError as e:
                self.ingest_log.append({"t": now, "kind": "sync_error", "error": str(e)})
        elif item.kind == "fixed_sync":
            unc = float(self.rec.manifest.get("accel_fixed_uncertainty_ms", 0.0) or 0.0)
            for b in self.rec.accel[:1]:
                self.ingestor.set_fixed_sync(b.device_id, b.session_id, "fixed", float(item.payload) + self.base, now, unc)
        elif item.kind == "accel":
            b = item.payload
            res = self.ingestor.ingest(b, now)
            self.ingest_log.append({"t": now, "seq": b.sequence_no, "flags": res.flags, "late": res.late})
            if res.t_node is not None and len(res.t_node):
                p.ingest_accel(
                    res.t_node, res.axes_g, res.includes_gravity, device_id=b.device_id, session_id=b.session_id,
                    clock_uncertainty_ms=res.clock_uncertainty_ms, now=now, placement=b.placement,
                    timestamp_untrusted="timestamp_untrusted" in res.flags,
                )
            elif res.flags:
                p.set_phone_status(has_device=True, placement=b.placement, timestamp_untrusted="timestamp_untrusted" in res.flags)
        elif item.kind == "frame":
            f: DetectionFrame = item.payload
            obs = frame_to_obs(
                f, self.tracker, f.t_ms + self.base, subject_id=self.subject_id, device_id=self.camera_device,
                session_id=self.camera_session, source_mode=self.source_mode, score_scale=score_scale_for(self.cfg, f.backend),
            )
            p.set_camera_status(camera_error=None if f.camera_ok else "camera_read_failed", backend=f.backend)
            p.ingest_pose(obs)
        elif item.kind == "exercise_start":
            ex = item.payload
            if self.on_exercise_start is not None:
                self.on_exercise_start(ex)
            else:
                p.start_exercise(f"{self.rec.recording_id}-ex", int(ex.get("target", 10)), ex.get("definition"))
            self.exercise_started = True

    def _resolve_unknown_arrivals(self) -> None:
        """Accel batches with device timestamps and no recorded arrival: estimate
        arrival from the first sync (device->recording) plus latency."""
        unknown = [it for it in self.items if it.kind == "accel" and it.arrival_ms == -2.0]
        if not unknown:
            return
        syncs = self.rec.time_sync_reports()
        if syncs:
            from homeease.edge.quality.clock_sync import estimate_sync

            est = estimate_sync("est", syncs[0][1].samples, syncs[0][2] + 1e9)
            off = est.offset_ms
        else:
            off = 0.0
        for it in unknown:
            it.arrival_ms = it.payload.samples[-1].t_ms + off + 100.0
        self.items.sort()

    def run(self, extra_ms: float = 6000.0, on_output: Callable[[PipelineOutput], None] | None = None, final_flush: bool = True) -> list[PipelineOutput]:
        self._resolve_unknown_arrivals()
        outs: list[PipelineOutput] = []
        for out in self.iter_ticks(extra_ms, final_flush):
            outs.append(out)
            if on_output:
                on_output(out)
        return outs

    def iter_ticks(self, extra_ms: float = 6000.0, final_flush: bool = True) -> Iterator[PipelineOutput]:
        self._resolve_unknown_arrivals()
        heap = list(self.items)
        heapq.heapify(heap)
        end = self.rec.duration_ms()
        if self.items:
            end = max(end, max(it.arrival_ms for it in self.items))
        t = 0.0
        while t <= end + extra_ms:
            while heap and heap[0].arrival_ms <= t:
                self._deliver(heapq.heappop(heap))
            last = t + self.tick_ms > end + extra_ms
            yield self.pipeline.tick(t + self.base, final=(last and final_flush))
            t += self.tick_ms
