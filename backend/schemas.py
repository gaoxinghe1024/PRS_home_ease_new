from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PlanDefer(Model):
    """Older adult's page: "not today". ``days`` is how many days to skip (1: today, 2: today and tomorrow)."""
    days: int = Field(default=1, ge=1, le=7)


class Resident(Model):
    name: str = Field(max_length=60)
    room: str = Field(min_length=1, max_length=80)
    notes: str = Field(default="", max_length=1000)


class Contact(Model):
    name: str = Field(min_length=1, max_length=60)
    relation: str = Field(default="Family", max_length=30)
    phone: str = Field(min_length=3, max_length=40, pattern=r"^[+0-9 ()-]+$")


class Settings(Model):
    resident: Resident
    contacts: list[Contact] = Field(max_length=10)
    audio_consent: bool = False
    exercise_target: int = Field(default=10, ge=1, le=10000)
    escalation_seconds: int = Field(default=60, ge=15, le=3600)
    webhook_url: str = Field(default="", max_length=2000)
    notification_test_at: str | None = None

    @model_validator(mode="after")
    def valid_webhook(self):
        if self.webhook_url:
            from pydantic import TypeAdapter
            url = TypeAdapter(HttpUrl).validate_python(self.webhook_url)
            if url.username or url.password:
                raise ValueError("The webhook URL must not contain a username or password")
        return self


class VideoStart(Model):
    source: Literal["camera", "sample", "upload"]
    camera_index: int = Field(default=0, ge=0, le=10)
    upload_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}\.[a-z0-9]+$")


class HomeMonitoring(Model):
    enabled: bool
    camera_index: int = Field(default=0, ge=0, le=10)


class SessionStart(Model):
    exercise: Literal["chair_sit_to_stand", "squat", "standing_leg_raise", "hurdle_step"]
    target_reps: int = Field(default=10, ge=1, le=10000)
    source: Literal["camera", "replay", "upload"] = "camera"
    camera_index: int = Field(default=0, ge=0, le=10)
    capture: Literal["shared", "client"] = "shared"
    upload_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}\.[a-z0-9]+$")


class SessionControl(Model):
    action: Literal["pause", "resume", "end"]


class AlgorithmSwitches(Model):
    """Master switches of the background algorithms. Keys follow the original algorithm numbers; each of the four exercises has its own.

    ``resident`` is background monitoring: when on, video fall, acceleration fall and exercise detection are run by the background
    scheduler, whether or not any page is open. It only decides whether the original algorithms start, never their decisions.
    """
    resident: bool = False
    vision: bool = True
    accel: bool = True
    chair_sit_to_stand: bool = True
    squat: bool = True
    standing_leg_raise: bool = True
    hurdle_step: bool = True


class AlgorithmUpdate(Model):
    """Partial update: send only the switches that changed; the others stay as they are."""
    resident: bool | None = None
    vision: bool | None = None
    accel: bool | None = None
    chair_sit_to_stand: bool | None = None
    squat: bool | None = None
    standing_leg_raise: bool | None = None
    hurdle_step: bool | None = None


class PlanItem(Model):
    exercise: Literal["chair_sit_to_stand", "squat", "standing_leg_raise", "hurdle_step"]
    target_reps: int = Field(default=10, ge=1, le=10000)
    enabled: bool = True


class ExercisePlan(Model):
    """The caregiver's daily exercise plan, which the older adult works through item by item."""
    enabled: bool = False
    title: str = Field(default="Today's movement plan", max_length=60)
    note: str = Field(default="", max_length=300)
    target_reps: int = Field(default=10, ge=1, le=10000)
    items: list[PlanItem] = Field(default_factory=list, max_length=4)
    # 长者选择「今天先不做」时记下的日期（ISO 日期串，只保留当天及以后）。
    skipped_dates: list[str] = Field(default_factory=list, max_length=64)
    updated_at: str | None = None
    updated_by: Literal["caregiver", "elder"] = "caregiver"

    @model_validator(mode="after")
    def no_duplicate_items(self):
        seen = [item.exercise for item in self.items]
        if len(seen) != len(set(seen)):
            raise ValueError("A movement can appear only once in the plan")
        return self


class EventAction(Model):
    action: Literal["acknowledge", "contact", "escalate", "resolve", "safe", "withdraw"]
    outcome: Literal["confirmed_fall", "false_alarm", "safe", "unresolved"] | None = None
    note: str = Field(default="", max_length=2000)
    revision: int | None = Field(default=None, ge=1)


class InquiryStart(Model):
    prompt_played: bool
    reason: str | None = Field(default=None, max_length=200)


class SampleAudio(Model):
    sample: str = Field(pattern=r"^[a-z0-9_]+$")
    task: Literal["monitor", "sounds", "help"] = "monitor"


class RecordDelete(Model):
    """Delete the selected records; events and training sessions are listed separately and do not affect each other."""
    event_ids: list[str] = Field(default_factory=list, max_length=5000)
    session_ids: list[str] = Field(default_factory=list, max_length=5000)


class RecordClear(Model):
    """Clear the current report scope; the scope parameters are exactly those of /api/reports.

    ``confirm`` must be the literal ``DELETE``: clearing is an irreversible bulk delete of the whole scope, and the default parameters
    (all live events and training of the last 7 days) sent by mistake would erase real records, so the caller has to write the
    confirmation word instead of relying on a default.
    """
    mode: Literal["live", "replay"] = "live"
    days: int = Field(default=7, ge=1, le=90)
    events: bool = True
    sessions: bool = True
    confirm: Literal["DELETE"]
