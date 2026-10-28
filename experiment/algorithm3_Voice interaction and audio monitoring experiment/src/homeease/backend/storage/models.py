"""SQLite schema (SQLAlchemy 2.x). High-rate sensor data is NOT stored here:
only summaries, state, timelines that need auditing, and configuration."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from homeease.core.clock import UTC


class UTCDateTime(TypeDecorator[datetime]):
    """Store naive UTC in SQLite, always return timezone-aware UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime passed to UTCDateTime")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    type_annotation_map = {datetime: UTCDateTime, dict[str, Any]: JSON, list[Any]: JSON}


# ------------------------------------------------------------------ identity
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    display_name: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(16))  # elder | caregiver | admin
    password_hash: Mapped[str] = mapped_column(String(256))
    locale: Mapped[str] = mapped_column(String(8), default="en")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime]


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    csrf_token: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime]
    expires_at: Mapped[datetime]
    revoked_at: Mapped[datetime | None]
    user_agent: Mapped[str | None] = mapped_column(String(256))
    user: Mapped[User] = relationship()


class Subject(Base):
    __tablename__ = "subjects"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128))
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Singapore")
    elder_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    carriage_setting: Mapped[str] = mapped_column(String(16), default="unknown")
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime]


class Room(Base):
    __tablename__ = "rooms"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(128))
    camera_device_id: Mapped[str | None] = mapped_column(String(64))
    zone: Mapped[list[Any]] = mapped_column(JSON, default=lambda: [0.0, 0.0, 1.0, 1.0])
    zone_version: Mapped[int] = mapped_column(Integer, default=1)


class Device(Base):
    __tablename__ = "devices"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(16))  # phone | camera | microphone
    name: Mapped[str] = mapped_column(String(128))
    platform: Mapped[str | None] = mapped_column(String(128))
    collector_type: Mapped[str | None] = mapped_column(String(32))
    placement: Mapped[str] = mapped_column(String(16), default="unknown")
    token_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    paired_at: Mapped[datetime | None]
    revoked_at: Mapped[datetime | None]
    last_seen_at: Mapped[datetime | None]
    last_data_at: Mapped[datetime | None]
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class PairingCode(Base):
    __tablename__ = "pairing_codes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code_hash: Mapped[str] = mapped_column(String(64), unique=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(16), default="phone")
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime]
    expires_at: Mapped[datetime]
    used_at: Mapped[datetime | None]
    device_id: Mapped[str | None] = mapped_column(String(64))
    attempts: Mapped[int] = mapped_column(Integer, default=0)


class DeviceSession(Base):
    __tablename__ = "device_sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"))
    started_at: Mapped[datetime]
    ended_at: Mapped[datetime | None]
    clock_sync: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Contact(Base):
    __tablename__ = "contacts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"))
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    name: Mapped[str] = mapped_column(String(128))
    relation: Mapped[str | None] = mapped_column(String(64))
    priority: Mapped[int] = mapped_column(Integer)
    channels: Mapped[list[Any]] = mapped_column(JSON, default=lambda: ["inapp"])
    webhook_url: Mapped[str | None] = mapped_column(String(512))
    phone_text: Mapped[str | None] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime]
    user: Mapped[User | None] = relationship()


class PushSubscription(Base):
    __tablename__ = "push_subscriptions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    endpoint: Mapped[str] = mapped_column(String(1024), unique=True)
    keys: Mapped[dict[str, Any]] = mapped_column(JSON)
    user_agent: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime]
    revoked_at: Mapped[datetime | None]


# -------------------------------------------------------------- monitoring
class HealthInterval(Base):
    __tablename__ = "health_intervals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(64), index=True)
    channel: Mapped[str] = mapped_column(String(16))  # coverage | accel | vision | audio
    status: Mapped[str] = mapped_column(String(16))
    reasons: Mapped[list[Any]] = mapped_column(JSON, default=list)
    start_at: Mapped[datetime]
    end_at: Mapped[datetime | None]
    source_mode: Mapped[str] = mapped_column(String(16), default="live")
    __table_args__ = (Index("ix_health_channel_start", "channel", "start_at"),)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    room_id: Mapped[str | None] = mapped_column(String(64))
    source_mode: Mapped[str] = mapped_column(String(16))
    origin: Mapped[str] = mapped_column(String(24))  # machine | help_request | caregiver | test
    decision_level: Mapped[str] = mapped_column(String(16))  # PENDING | EMERGENCY
    max_level: Mapped[str] = mapped_column(String(16))
    lifecycle_status: Mapped[str] = mapped_column(String(16))  # OPEN | ACKNOWLEDGED | IN_PROGRESS | RESOLVED
    candidate_at: Mapped[datetime | None]
    occurrence_est_at: Mapped[datetime | None]
    occurrence_source: Mapped[str | None] = mapped_column(String(64))
    classified_at: Mapped[datetime]
    anchor_mono_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    resolved_at: Mapped[datetime | None]
    outcome: Mapped[str | None] = mapped_column(String(32))
    outcome_note: Mapped[str | None] = mapped_column(Text)
    resolved_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    escalation_step: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_test: Mapped[bool] = mapped_column(Boolean, default=False)
    fusion_backend: Mapped[str | None] = mapped_column(String(32))
    model_versions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    thresholds_version: Mapped[str | None] = mapped_column(String(32))
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    prediction: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    voice_inquiry: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    __table_args__ = (Index("ix_events_status", "lifecycle_status", "classified_at"),)


class EvidenceSnapshot(Base):
    __tablename__ = "evidence_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str | None] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), index=True)
    decision_id: Mapped[str] = mapped_column(String(80), index=True)
    channel: Mapped[str] = mapped_column(String(16))
    available: Mapped[bool] = mapped_column(Boolean)
    candidate: Mapped[bool] = mapped_column(Boolean)
    contradicts: Mapped[bool] = mapped_column(Boolean, default=False)
    score: Mapped[float] = mapped_column(Float)
    score_semantics: Mapped[str] = mapped_column(String(32))
    quality_score: Mapped[float] = mapped_column(Float)
    time_start_at: Mapped[datetime | None]
    time_end_at: Mapped[datetime | None]
    peak_at: Mapped[datetime | None]
    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    reason_codes: Mapped[list[Any]] = mapped_column(JSON, default=list)
    quality_flags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    model_id: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(32))
    calibration_id: Mapped[str | None] = mapped_column(String(64))
    source_mode: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime]


class CandidateRecord(Base):
    """Every fusion decision, including rejected candidates (technical record)."""

    __tablename__ = "candidate_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decision_id: Mapped[str] = mapped_column(String(80), unique=True)
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    subject_id: Mapped[str] = mapped_column(String(64))
    level: Mapped[str] = mapped_column(String(16))
    rejected: Mapped[bool] = mapped_column(Boolean)
    event_id: Mapped[str | None] = mapped_column(ForeignKey("events.id", ondelete="SET NULL"))
    decision: Mapped[dict[str, Any]] = mapped_column(JSON)
    source_mode: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime]


class EventTransition(Base):
    __tablename__ = "event_transitions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(32))
    from_level: Mapped[str | None] = mapped_column(String(16))
    to_level: Mapped[str | None] = mapped_column(String(16))
    from_status: Mapped[str | None] = mapped_column(String(16))
    to_status: Mapped[str | None] = mapped_column(String(16))
    actor_type: Mapped[str] = mapped_column(String(16))  # system | user | device
    actor_id: Mapped[str | None] = mapped_column(String(64))
    reason: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    idempotency_key: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime]
    __table_args__ = (
        UniqueConstraint("event_id", "seq", name="uq_transition_seq"),
        UniqueConstraint("event_id", "idempotency_key", name="uq_transition_idem"),
    )


class Notification(Base):
    __tablename__ = "notifications"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str | None] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), index=True)
    notification_type: Mapped[str] = mapped_column(String(24))  # verification | emergency | escalation | reminder | resolved | test | coverage
    escalation_step: Mapped[int] = mapped_column(Integer, default=0)
    recipient_contact_id: Mapped[int | None] = mapped_column(ForeignKey("contacts.id", ondelete="SET NULL"))
    recipient_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    channel: Mapped[str] = mapped_column(String(16))  # inapp | webpush | webhook | test_sink
    status: Mapped[str] = mapped_column(String(20), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    receipt_token_hash: Mapped[str | None] = mapped_column(String(64))
    queued_at: Mapped[datetime]
    sending_at: Mapped[datetime | None]
    provider_accepted_at: Mapped[datetime | None]
    delivered_at: Mapped[datetime | None]
    delivered_source: Mapped[str | None] = mapped_column(String(32))
    client_reported_at: Mapped[datetime | None]
    viewed_at: Mapped[datetime | None]
    acknowledged_at: Mapped[datetime | None]
    failed_at: Mapped[datetime | None]
    expired_at: Mapped[datetime | None]
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime | None]
    last_error: Mapped[str | None] = mapped_column(Text)
    expires_at: Mapped[datetime | None]
    original_occurrence_at: Mapped[datetime | None]
    delayed_delivery: Mapped[bool] = mapped_column(Boolean, default=False)
    delay_seconds: Mapped[float | None] = mapped_column(Float)
    is_test: Mapped[bool] = mapped_column(Boolean, default=False)
    idem_key: Mapped[str] = mapped_column(String(160), unique=True)
    __table_args__ = (Index("ix_notifications_due", "status", "next_attempt_at"),)


class ScheduledTask(Base):
    __tablename__ = "scheduled_tasks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))  # ack_timeout | handling_reminder | repeat_reminder
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), index=True)
    step: Mapped[int] = mapped_column(Integer, default=0)
    due_at: Mapped[datetime] = mapped_column(index=True)
    status: Mapped[str] = mapped_column(String(16), default="scheduled")  # scheduled | done | cancelled
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime]
    done_at: Mapped[datetime | None]
    __table_args__ = (UniqueConstraint("kind", "event_id", "step", name="uq_task"),)


class ExerciseSession(Base):
    __tablename__ = "exercise_sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    activity: Mapped[str] = mapped_column(String(32), default="sit_to_stand")
    target_reps: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))  # CREATED RUNNING PAUSED COMPLETED ABORTED INTERRUPTED
    sub_state: Mapped[str | None] = mapped_column(String(24))
    repetition_definition: Mapped[str] = mapped_column(String(24))
    count_valid: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None]
    ended_at: Mapped[datetime | None]
    total_s: Mapped[float] = mapped_column(Float, default=0.0)
    effective_s: Mapped[float] = mapped_column(Float, default=0.0)
    paused_manual_s: Mapped[float] = mapped_column(Float, default=0.0)
    paused_auto_s: Mapped[float] = mapped_column(Float, default=0.0)
    phase_model: Mapped[str | None] = mapped_column(String(64))
    counter_version: Mapped[str | None] = mapped_column(String(32))
    quality: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    target_reached: Mapped[bool] = mapped_column(Boolean, default=False)
    source_mode: Mapped[str] = mapped_column(String(16), default="live")
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class ExerciseCycle(Base):
    __tablename__ = "exercise_cycles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("exercise_sessions.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    cycle_start_at: Mapped[datetime | None]
    stand_reached_at: Mapped[datetime | None]
    cycle_end_at: Mapped[datetime | None]
    valid: Mapped[bool] = mapped_column(Boolean)
    reject_reason: Mapped[str | None] = mapped_column(String(32))
    duration_s: Mapped[float | None] = mapped_column(Float)
    definition: Mapped[str] = mapped_column(String(24))


class ExerciseInterruption(Base):
    __tablename__ = "exercise_interruptions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("exercise_sessions.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str | None] = mapped_column(String(128))
    start_at: Mapped[datetime]
    end_at: Mapped[datetime | None]


class ModelRegistryEntry(Base):
    __tablename__ = "model_registry"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(32))
    task: Mapped[str] = mapped_column(String(32))
    path: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str | None] = mapped_column(String(64))
    layout: Mapped[str | None] = mapped_column(String(32))
    training_run_id: Mapped[str | None] = mapped_column(String(64))
    calibration_id: Mapped[str | None] = mapped_column(String(64))
    deploy_status: Mapped[str] = mapped_column(String(16), default="available")
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime]
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_model_version"),)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    suite: Mapped[str | None] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(24))
    config_hash: Mapped[str | None] = mapped_column(String(32))
    code_version: Mapped[str | None] = mapped_column(String(32))
    split_hash: Mapped[str | None] = mapped_column(String(32))
    seed: Mapped[int | None] = mapped_column(Integer)
    artifact_dir: Mapped[str] = mapped_column(String(512))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime]
    finished_at: Mapped[datetime | None]


class Consent(Base):
    __tablename__ = "consents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(32))  # audio | research_recording
    granted: Mapped[bool] = mapped_column(Boolean)
    version: Mapped[str] = mapped_column(String(16))
    granted_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime]


class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON)
    updated_at: Mapped[datetime]
    updated_by: Mapped[int | None] = mapped_column(Integer)


class RealtimeMessage(Base):
    """Durable log of important realtime messages (cursor for WS catch-up)."""

    __tablename__ = "realtime_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(32))
    subject_id: Mapped[str | None] = mapped_column(String(64))
    audience: Mapped[str] = mapped_column(String(16), default="caregiver")  # caregiver | elder | all
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(index=True)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    target: Mapped[str | None] = mapped_column(String(128))
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime]


# ------------------------------------------------------------------- audio
class SoundEventRecord(Base):
    """Label-only record of a detected household sound event (daily sound monitoring).

    Whitelist of scalars: category, confidence, time span and summary statistics. There is
    deliberately no audio, embedding, spectrogram, transcript or free-text column: raw audio
    never reaches the database (proposal 1.5.2). A test introspects this table."""

    __tablename__ = "sound_events"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(64))
    room_id: Mapped[str | None] = mapped_column(String(64))
    source_mode: Mapped[str] = mapped_column(String(16), default="live")
    category: Mapped[str] = mapped_column(String(16))  # HELP | IMPACT | ABNORMAL | DAILY | UNKNOWN
    confidence: Mapped[float] = mapped_column(Float)
    t_start: Mapped[datetime]
    t_end: Mapped[datetime]
    t_start_mono_ms: Mapped[float] = mapped_column(Float)
    t_end_mono_ms: Mapped[float] = mapped_column(Float)
    duration_s: Mapped[float] = mapped_column(Float)
    rms_db: Mapped[float] = mapped_column(Float)
    peak_db: Mapped[float] = mapped_column(Float)
    n_windows: Mapped[int] = mapped_column(Integer, default=1)
    model_id: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    __table_args__ = (Index("ix_sound_events_subject_start", "subject_id", "t_start"), Index("ix_sound_events_t_end", "t_end"))
