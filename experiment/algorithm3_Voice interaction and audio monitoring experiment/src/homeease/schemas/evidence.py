"""Per-channel evidence and fusion decision contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from homeease.schemas.observations import SourceMode


class ScoreSemantics(str, Enum):
    heuristic_score = "heuristic_score"  # rule output in [0,1], NOT a probability
    uncalibrated_probability = "uncalibrated_probability"  # model output, not calibrated
    calibrated_probability = "calibrated_probability"  # calibrated on a held-out calibration split


class DecisionLevel(str, Enum):
    NORMAL = "NORMAL"
    PENDING = "PENDING"
    EMERGENCY = "EMERGENCY"


LEVEL_RANK = {DecisionLevel.NORMAL: 0, DecisionLevel.PENDING: 1, DecisionLevel.EMERGENCY: 2}


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channel: Literal["accel", "vision", "audio", "manual", "fused"]
    candidate: bool
    score: float = Field(ge=0.0, le=1.0)
    score_semantics: ScoreSemantics
    quality_score: float = Field(ge=0.0, le=1.0)
    available: bool
    time_start_ms: float
    time_end_ms: float
    peak_time_ms: float | None = None
    emitted_at_ms: float
    features_summary: dict[str, Any] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    contradicts: bool = False
    model_id: str
    model_version: str
    calibration_id: str | None = None
    source_mode: SourceMode = SourceMode.live
    subject_id: str
    session_id: str | None = None
    device_id: str | None = None
    track_id: int | None = None
    target_confidence: float | None = None
    clock_uncertainty_ms: float | None = None
    strong: bool = False


class FusionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: str
    subject_id: str
    level: DecisionLevel
    rejected: bool = False
    decided_at_ms: float
    anchor_time_ms: float
    onset_estimate_ms: float | None = None
    fusion_backend: str
    fused_score: float | None = None
    gate: dict[str, Any] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    evidences: list[Evidence] = Field(default_factory=list)
    source_mode: SourceMode = SourceMode.live
    room_id: str | None = None
    summary: str = ""
