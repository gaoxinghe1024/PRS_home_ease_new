"""Plain-language (English / Chinese) explanations of decisions and events.

Wording follows proposal 9.2: describe what each channel observed, never claim
a confirmed injury.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

ROOM_NAMES = {"living-room": ("living room", "\u5ba2\u5385"), "bedroom": ("bedroom", "\u5367\u5ba4"), "kitchen": ("kitchen", "\u53a8\u623f")}

CHANNEL_TEXT = {
    "accel_candidate": ("the phone detected an impact followed by reduced movement", "\u624b\u673a\u68c0\u6d4b\u5230\u51b2\u51fb\u540e\u6d3b\u52a8\u51cf\u5c11"),
    "accel_no_impact": ("the phone did not show an impact (it may not be carried)", "\u624b\u673a\u672a\u663e\u793a\u51b2\u51fb\uff08\u53ef\u80fd\u672a\u968f\u8eab\u643a\u5e26\uff09"),
    "accel_missing": ("phone motion data were unavailable", "\u624b\u673a\u8fd0\u52a8\u6570\u636e\u4e0d\u53ef\u7528"),
    "accel_untrusted_or_low_quality": ("phone data were unreliable (timing or carriage uncertain)", "\u624b\u673a\u6570\u636e\u4e0d\u53ef\u9760\uff08\u65f6\u95f4\u6216\u643a\u5e26\u72b6\u6001\u4e0d\u786e\u5b9a\uff09"),
    "accel_weak": ("the phone showed only a weak impact pattern", "\u624b\u673a\u4ec5\u663e\u793a\u8f83\u5f31\u7684\u51b2\u51fb\u7279\u5f81"),
    "vision_candidate": ("the camera detected a rapid decrease in body height and a persistent low posture", "\u6444\u50cf\u5934\u68c0\u6d4b\u5230\u8eab\u4f53\u9ad8\u5ea6\u5feb\u901f\u4e0b\u964d\u5e76\u6301\u7eed\u4f4e\u4f4d"),
    "vision_missing": ("the camera could not see the person", "\u6444\u50cf\u5934\u672a\u80fd\u770b\u5230\u8001\u4eba"),
    "vision_low_quality": ("the camera view was unclear", "\u6444\u50cf\u5934\u753b\u9762\u8d28\u91cf\u4e0d\u8db3"),
    "vision_no_fall_posture": ("the camera did not see a fall posture", "\u6444\u50cf\u5934\u672a\u89c2\u5bdf\u5230\u6454\u5012\u59ff\u6001"),
    "vision_weak": ("the camera saw only a weak fall pattern", "\u6444\u50cf\u5934\u4ec5\u89c2\u5bdf\u5230\u8f83\u5f31\u7684\u6454\u5012\u7279\u5f81"),
    "target_not_confirmed": ("the camera could not confirm it was the monitored person", "\u6444\u50cf\u5934\u65e0\u6cd5\u786e\u8ba4\u662f\u88ab\u7167\u62a4\u8005\u672c\u4eba"),
    "time_inconsistent": ("phone and camera timings did not match", "\u624b\u673a\u4e0e\u6444\u50cf\u5934\u7684\u65f6\u95f4\u4e0d\u4e00\u81f4"),
    "target_lost_after_drop": ("the person dropped quickly and then left the camera view", "\u4eba\u4f53\u5feb\u901f\u4e0b\u964d\u540e\u79bb\u5f00\u4e86\u6444\u50cf\u5934\u753b\u9762"),
}


def room_name(room_id: str | None, lang: str) -> str:
    if not room_id:
        return "home" if lang == "en" else "\u5bb6\u4e2d"
    en, zh = ROOM_NAMES.get(room_id, (room_id.replace("-", " "), room_id))
    return en if lang == "en" else zh


def local_time(dt: datetime | None, tz: str = "Asia/Singapore", fmt: str = "%H:%M") -> str:
    if dt is None:
        return "--:--"
    return dt.astimezone(ZoneInfo(tz)).strftime(fmt)


def evidence_phrases(summary: dict[str, Any], lang: str) -> list[str]:
    i = 0 if lang == "en" else 1
    out = []
    for key in summary.get("phrases", []):
        if key in CHANNEL_TEXT:
            out.append(CHANNEL_TEXT[key][i])
    return out


def summarize_decision(decision: dict[str, Any]) -> dict[str, Any]:
    """Derive phrase keys from a fusion decision (stored on the event)."""
    phrases: list[str] = []
    ev = {e["channel"]: e for e in decision.get("evidences", [])}
    a, v = ev.get("accel"), ev.get("vision")
    reasons = set(decision.get("reason_codes", []))
    if a and a.get("candidate"):
        phrases.append("accel_candidate")
    if v and v.get("candidate"):
        phrases.append("target_lost_after_drop" if "target_lost_after_drop" in v.get("reason_codes", []) else "vision_candidate")
    for r in ("accel_missing", "accel_untrusted_or_low_quality", "accel_no_impact", "accel_weak", "vision_missing", "vision_low_quality", "target_not_confirmed", "vision_weak", "vision_no_fall_posture", "time_inconsistent"):
        if r in reasons and r not in phrases:
            phrases.append(r)
    return {"phrases": phrases, "level": decision.get("level"), "fusion_backend": decision.get("fusion_backend")}


def compose(event: dict[str, Any], ntype: str, lang: str = "en", tz: str = "Asia/Singapore", delayed_s: float | None = None) -> dict[str, str]:
    level = event.get("decision_level")
    origin = event.get("origin")
    room = room_name(event.get("room_id"), lang)
    when = local_time(event.get("occurred_at") or event.get("classified_at"), tz)
    phrases = evidence_phrases(event.get("summary") or {}, lang)
    test = event.get("is_test") or event.get("source_mode") not in (None, "live")
    if lang == "en":
        prefix = "[TEST] " if test else ""
        if origin == "help_request":
            title = f"{prefix}Help requested - {room}, {when}"
            body = "The older adult pressed the help button. Please contact them now."
        elif origin == "caregiver":
            title = f"{prefix}Escalated by caregiver - {room}, {when}"
            body = "A caregiver raised this event to emergency. Please contact the older adult."
        elif level == "EMERGENCY":
            title = f"{prefix}Suspected fall - {room}, {when}"
            body = ("; ".join(phrases) + ". " if phrases else "") + "Please contact the older adult to check whether help is needed."
        else:
            title = f"{prefix}Please verify - possible fall, {room}, {when}"
            body = ("; ".join(phrases) + ". " if phrases else "") + "Only partial evidence is available. Please check on the older adult."
        if ntype == "escalation":
            title = "Escalation: " + title
        elif ntype == "reminder":
            title = "Reminder: " + title
        elif ntype == "resolved":
            title = f"{prefix}Event closed - {room}, {when}"
            body = f"Outcome: {event.get('outcome') or 'recorded'}."
        if delayed_s:
            body += f" (Delivered late: occurred {int(delayed_s)} s before this message.)"
    else:
        prefix = "\u3010\u6d4b\u8bd5\u3011" if test else ""
        if origin == "help_request":
            title = f"{prefix}\u8001\u4eba\u8bf7\u6c42\u5e2e\u52a9 - {room} {when}"
            body = "\u8001\u4eba\u6309\u4e0b\u4e86\u6c42\u52a9\u6309\u94ae\uff0c\u8bf7\u7acb\u5373\u8054\u7cfb\u3002"
        elif origin == "caregiver":
            title = f"{prefix}\u7167\u62a4\u8005\u5df2\u5347\u7ea7\u4e3a\u7d27\u6025 - {room} {when}"
            body = "\u7167\u62a4\u8005\u5c06\u6b64\u4e8b\u4ef6\u63d0\u5347\u4e3a\u7d27\u6025\uff0c\u8bf7\u8054\u7cfb\u8001\u4eba\u3002"
        elif level == "EMERGENCY":
            title = f"{prefix}\u7591\u4f3c\u6454\u5012 - {room} {when}"
            body = ("\uff1b".join(phrases) + "\u3002" if phrases else "") + "\u8bf7\u8054\u7cfb\u8001\u4eba\u786e\u8ba4\u662f\u5426\u9700\u8981\u5e2e\u52a9\u3002"
        else:
            title = f"{prefix}\u8bf7\u6838\u5b9e - \u53ef\u80fd\u6454\u5012\uff0c{room} {when}"
            body = ("\uff1b".join(phrases) + "\u3002" if phrases else "") + "\u76ee\u524d\u53ea\u6709\u90e8\u5206\u8bc1\u636e\uff0c\u8bf7\u786e\u8ba4\u8001\u4eba\u60c5\u51b5\u3002"
        if ntype == "escalation":
            title = "\u5347\u7ea7\u901a\u77e5\uff1a" + title
        elif ntype == "reminder":
            title = "\u63d0\u9192\uff1a" + title
        elif ntype == "resolved":
            title = f"{prefix}\u4e8b\u4ef6\u5df2\u5173\u95ed - {room} {when}"
            body = f"\u7ed3\u5c40\uff1a{event.get('outcome') or '\u5df2\u8bb0\u5f55'}\u3002"
        if delayed_s:
            body += f"\uff08\u5ef6\u8fdf\u9001\u8fbe\uff1a\u4e8b\u4ef6\u53d1\u751f\u4e8e {int(delayed_s)} \u79d2\u524d\u3002\uff09"
    return {"title": title, "body": body}
