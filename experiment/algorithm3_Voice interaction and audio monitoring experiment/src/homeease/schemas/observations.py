"""Unified observation contracts (wire + file format).

The same schema is used by the phone web page (browser collector), replay
files, synthetic generators and dataset adapters, so that every source
goes through the same ingestion, quality and inference code.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"
G = 9.80665  # m/s^2 per g


class SourceMode(str, Enum):
    live = "live"
    replay = "replay"
    synthetic = "synthetic"
    policy_test = "policy_test"


def _finite(v: float, name: str) -> float:
    if v is None or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number")
    return v


class AccelSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t_ms: float
    ax: float
    ay: float
    az: float

    @field_validator("t_ms", "ax", "ay", "az")
    @classmethod
    def _fin(cls, v: float, info) -> float:  # type: ignore[no-untyped-def]
        return _finite(v, info.field_name)


Placement = Literal["wrist", "pocket", "waist", "pelvis", "hand", "bag", "table", "unknown"]
CollectorType = Literal["web_device_motion", "replay", "synthetic", "dataset"]
Timebase = Literal["device_monotonic_ms", "recording_ms", "node_monotonic_ms"]


class AccelBatch(BaseModel):
    """A batch of tri-axial acceleration samples from one device session.

    ``includes_gravity`` distinguishes ``accelerationIncludingGravity`` (static
    magnitude ~1 g) from linear acceleration (static magnitude ~0). The two are
    never mixed inside one batch.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    observation_id: str | None = None
    device_id: str = Field(min_length=1, max_length=64)
    subject_id: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    room_id: str | None = Field(default=None, max_length=64)
    source_mode: SourceMode = SourceMode.live
    collector_type: CollectorType = "web_device_motion"
    sequence_no: int = Field(ge=0)
    placement: Placement = "unknown"
    unit: Literal["m_s2", "g"] = "m_s2"
    includes_gravity: bool = True
    timebase: Timebase = "device_monotonic_ms"
    clock_sync_id: str | None = Field(default=None, max_length=64)
    reported_rate_hz: float | None = None
    client_quality: dict[str, float | int | str | bool | None] | None = None
    samples: list[AccelSample] = Field(min_length=1, max_length=2000)
    sent_at_device_ms: float | None = None

    @model_validator(mode="after")
    def _ranges(self) -> "AccelBatch":
        limit = 40.0 * (G if self.unit == "m_s2" else 1.0)  # far above any phone sensor range
        prev = -math.inf
        for s in self.samples:
            if abs(s.ax) > limit or abs(s.ay) > limit or abs(s.az) > limit:
                raise ValueError("acceleration outside plausible sensor range")
            if s.t_ms < prev - 1e-6:
                # Out-of-order inside a batch is a collector bug; reject so it is visible.
                raise ValueError("samples inside a batch must be time-ordered")
            prev = s.t_ms
        return self


COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
LAYOUT_VERSION = "coco17-v1"


class PoseFrame(BaseModel):
    """2D pose of the tracked target for one camera frame.

    Keypoints are COCO-17 in normalised image coordinates ``[x, y, score]``
    (x right, y down, both in [0,1] of image width/height) or ``null`` when not
    detected. Extra backend-specific points are kept under ``extras``.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    observation_id: str | None = None
    subject_id: str
    device_id: str
    room_id: str | None = None
    session_id: str
    source_mode: SourceMode = SourceMode.live
    frame_index: int = Field(ge=0)
    t_source_ms: float
    t_aligned_ms: float | None = None
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    track_id: int | None = None
    layout: Literal["coco17-v1"] = LAYOUT_VERSION
    coord_format: Literal["normalized_xy"] = "normalized_xy"
    keypoints: list[list[float] | None] = Field(min_length=17, max_length=17)
    extras: dict[str, list[float]] | None = None
    bbox: list[float] | None = None
    target_visible: bool = False
    person_count: int = Field(0, ge=0)
    target_confidence: float = Field(0.0, ge=0, le=1)
    target_flags: list[str] = Field(default_factory=list)
    backend: str = "unknown"
    inference_ms: float | None = None

    @field_validator("keypoints")
    @classmethod
    def _kp(cls, v: list[list[float] | None]) -> list[list[float] | None]:
        for kp in v:
            if kp is None:
                continue
            if len(kp) != 3 or not all(math.isfinite(x) for x in kp):
                raise ValueError("keypoint must be [x, y, score] finite")
            if not (-0.5 <= kp[0] <= 1.5 and -0.5 <= kp[1] <= 1.5 and 0 <= kp[2] <= 1.0001):
                raise ValueError("keypoint outside normalised range")
        return v


class AudioWindowMeta(BaseModel):
    """Metadata of an audio window. PCM stays in memory; it is not part of the wire format."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    observation_id: str | None = None
    subject_id: str
    device_id: str
    session_id: str
    source_mode: SourceMode = SourceMode.live
    sample_rate: int = Field(gt=0)
    channels: int = Field(1, ge=1, le=8)
    t_start_ms: float
    t_end_ms: float
    source: Literal["microphone", "file", "synthetic_tts", "dataset"] = "microphone"
    consent_version: str | None = None


class TimeSyncSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t0_device_ms: float
    t1_server_mono_ms: float
    t2_server_mono_ms: float
    t3_device_ms: float


class TimeSyncReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1, max_length=64)
    sync_id: str = Field(min_length=1, max_length=64)
    samples: list[TimeSyncSample] = Field(min_length=1, max_length=64)
