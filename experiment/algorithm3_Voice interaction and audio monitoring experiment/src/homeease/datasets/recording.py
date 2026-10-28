"""Canonical recording format used by replay, synthetic data, own captures and
converted public datasets.

<recording_dir>/
  manifest.json      metadata (source_mode, units, timebases, sync, devices, provenance)
  accel.jsonl        one AccelBatch per line (optional)
  detections.jsonl   one camera frame per line with all detected persons (optional)
  labels.json        ground truth sidecar - NEVER read by inference code
  video.mp4          optional raw video (own captures / converted datasets)

Detections are stored before target association so that replay exercises the
same tracker as live operation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from homeease.core.util import dumps
from homeease.ml.pose.layout import kp_from_list
from homeease.schemas.observations import AccelBatch, TimeSyncReport


@dataclass
class DetectionFrame:
    t_ms: float
    frame_index: int
    width: int
    height: int
    persons: list[np.ndarray]
    backend: str = "unknown"
    inference_ms: float | None = None
    camera_ok: bool = True

    @staticmethod
    def from_json(d: dict[str, Any]) -> "DetectionFrame":
        return DetectionFrame(
            t_ms=float(d["t_ms"]),
            frame_index=int(d.get("frame_index", 0)),
            width=int(d["width"]),
            height=int(d["height"]),
            persons=[kp_from_list(p) for p in d.get("persons", [])],
            backend=d.get("backend", "unknown"),
            inference_ms=d.get("inference_ms"),
            camera_ok=bool(d.get("camera_ok", True)),
        )

    def to_json(self) -> dict[str, Any]:
        from homeease.ml.pose.layout import kp_to_list

        return {
            "t_ms": round(self.t_ms, 2),
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "persons": [kp_to_list(p) for p in self.persons],
            "backend": self.backend,
            "inference_ms": self.inference_ms,
            "camera_ok": self.camera_ok,
        }


@dataclass
class Recording:
    path: Path
    manifest: dict[str, Any]
    accel: list[AccelBatch] = field(default_factory=list)
    detections: list[DetectionFrame] = field(default_factory=list)

    @property
    def recording_id(self) -> str:
        return str(self.manifest.get("recording_id", self.path.name))

    @property
    def source_mode(self) -> str:
        return str(self.manifest.get("source_mode", "replay"))

    def time_sync_reports(self) -> list[tuple[str, TimeSyncReport, float]]:
        """(device_id, report, node_time_ms_when_received) from the manifest."""
        out = []
        for s in self.manifest.get("clock_sync", []):
            out.append((s["device_id"], TimeSyncReport.model_validate(s["report"]), float(s["received_at_ms"])))
        return out

    def duration_ms(self) -> float:
        ts: list[float] = []
        if self.detections:
            ts += [self.detections[0].t_ms, self.detections[-1].t_ms]
        if "duration_ms" in self.manifest:
            ts.append(float(self.manifest["duration_ms"]))
        return max(ts) if ts else 0.0

    def labels(self) -> dict[str, Any]:
        """Ground truth for evaluation only."""
        p = self.path / "labels.json"
        if not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))


def load_recording(path: str | Path, detections_file: str | None = None) -> Recording:
    p = Path(path)
    manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    rec = Recording(p, manifest)
    af = p / "accel.jsonl"
    if af.is_file():
        with open(af, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec.accel.append(AccelBatch.model_validate_json(line))
    df = p / (detections_file or manifest.get("detections_file", "detections.jsonl"))
    if df.is_file():
        with open(df, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec.detections.append(DetectionFrame.from_json(json.loads(line)))
    return rec


def write_recording(
    path: str | Path,
    manifest: dict[str, Any],
    accel: list[AccelBatch] | None,
    detections: list[DetectionFrame] | None,
    labels: dict[str, Any] | None,
    detections_file: str = "detections.jsonl",
) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    (p / "manifest.json").write_text(dumps(manifest, indent=2), encoding="utf-8")
    if accel is not None:
        with open(p / "accel.jsonl", "w", encoding="utf-8") as fh:
            for b in accel:
                fh.write(b.model_dump_json() + "\n")
    if detections is not None:
        with open(p / detections_file, "w", encoding="utf-8") as fh:
            for d in detections:
                fh.write(dumps(d.to_json()) + "\n")
    if labels is not None:
        (p / "labels.json").write_text(dumps(labels, indent=2), encoding="utf-8")
    return p


def iter_recordings(root: str | Path) -> Iterator[Path]:
    root = Path(root)
    if (root / "manifest.json").is_file():
        yield root
        return
    for m in sorted(root.rglob("manifest.json")):
        yield m.parent
