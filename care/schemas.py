from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PlanDefer(Model):
    """长者端「今天先不做」。``days`` 是往后顺延的天数（今天 1 天、今天加明天 2 天）。"""
    days: int = Field(default=1, ge=1, le=7)


class Resident(Model):
    name: str = Field(max_length=60)
    room: str = Field(min_length=1, max_length=80)
    notes: str = Field(default="", max_length=1000)


class Contact(Model):
    name: str = Field(min_length=1, max_length=60)
    relation: str = Field(default="家属", max_length=30)
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
                raise ValueError("Webhook URL 不得含账户密码")
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
    """后台算法总开关。键名与原算法编号一致，四种运动分别开关。

    ``resident`` 为后台常驻：开启后，视频跌倒、加速度跌倒、运动检测都由后台
    调度自动运行，不依赖任何页面是否打开。只决定是否启动原算法，不改变其判定。
    """
    resident: bool = False
    vision: bool = True
    accel: bool = True
    chair_sit_to_stand: bool = True
    squat: bool = True
    standing_leg_raise: bool = True
    hurdle_step: bool = True


class AlgorithmUpdate(Model):
    """部分更新：只提交被改动的那几个开关，其余保持不变。"""
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
    """照护者制定的每日运动计划，长者端按此逐项完成。"""
    enabled: bool = False
    title: str = Field(default="今天的运动计划", max_length=60)
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
            raise ValueError("同一动作在计划中只能出现一次")
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
    """删除勾选的记录；事件与训练各传各的编号，互不影响。"""
    event_ids: list[str] = Field(default_factory=list, max_length=5000)
    session_ids: list[str] = Field(default_factory=list, max_length=5000)


class RecordClear(Model):
    """清空当前报告范围；范围参数与 /api/reports 完全一致。

    ``confirm`` 必须是字面量 ``DELETE``：清空是整段范围内不可撤销的批量删除，
    默认参数（近 7 天 live 全部事件与训练）一旦被误发就会把真实记录一起抹掉，
    所以要求调用方明确写出确认词，而不是靠默认值。
    """
    mode: Literal["live", "replay"] = "live"
    days: int = Field(default=7, ge=1, le=90)
    events: bool = True
    sessions: bool = True
    confirm: Literal["DELETE"]
