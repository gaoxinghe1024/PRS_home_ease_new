import sys
import time
from datetime import datetime, timezone

from care.config import AUDIO
from care.storage import now

# Only the original pure policy functions run in the API process.
sys.path.append(str(AUDIO / "src"))
from algorithm3.policy import alarm_tag, inquiry_outcome

# An alarm that the older adult raised (help button, spoken call for help) can be withdrawn by the same person for a short time, for
# example after a button press by mistake or a television that said "help". A withdrawal never closes or downgrades the event: it
# stays open at its level, the contact escalation stops, the caregivers are told, and only a caregiver closes it. A fall alarm of a
# sensor can be withdrawn too once the voice inquiry has raised it to an emergency (no answer, or a call for help), for the same
# time counted from that moment: before that the older adult can answer the inquiry, and the caregiver decides on the rest.
WITHDRAW_WINDOW_S = 30
SELF_TRIGGERED = ("manual", "audio")

# One fall seen by several channels is one event: algorithm3's alarm tag counts the channels of one event (single / dual / triple).
# An alarm of another channel close to an open live alarm therefore joins that event instead of opening a second one, so a camera
# fall followed by a spoken call for help is one emergency that is asked about once. A spoken call always raises the event to
# EMERGENCY, as it would on its own. Sound labels (impact, abnormal, help-like) heard around an alarm are only noted on it: they
# never raise, close or downgrade an event (algorithm3's sound monitor rule).
LINK_WINDOW_S = 30
LINKED_CHANNELS = ("accel", "vision", "audio")
CHANNEL_NAMES = {"accel": "加速度跌倒检测", "vision": "视频跌倒检测", "audio": "语音求助"}
SOUND_HINTS = {"HELP": "求助相关声音", "IMPACT": "撞击声", "ABNORMAL": "异常声音"}


def _time(stamp):
    try:
        moment = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _gap(event, when):
    """Seconds between ``when`` and the nearest alarm marker of the event (None without a counted channel)."""
    gaps = [abs((when - t).total_seconds()) for m in event.get("markers", [])
            if m["channel"] in LINKED_CHANNELS and (t := _time(m["at"])) is not None]
    return min(gaps) if gaps else None


def is_withdrawn(event):
    w = event.get("withdrawal")
    return bool(w) and not w.get("overridden_at")


def withdraw_status(event, at=None):
    """{"withdrawable", "withdraw_left_s"}: can the older adult still withdraw this event, and for how long (server clock)."""
    no = {"withdrawable": False, "withdraw_left_s": 0}
    if event.get("state") != "open" or event.get("withdrawal"):
        return no
    if event.get("channel") in SELF_TRIGGERED:
        stamp = event.get("detected_at")
    elif event.get("level") == "EMERGENCY" and event.get("escalated_at"):
        stamp = event["escalated_at"]  # raised by the inquiry: the window starts when the alarm was raised
    else:
        return no
    try:
        started = datetime.fromisoformat(stamp).timestamp()
    except (KeyError, TypeError, ValueError):
        return no
    left = WITHDRAW_WINDOW_S - ((time.time() if at is None else at) - started)
    return {"withdrawable": left > 0, "withdraw_left_s": max(0, int(left + .999))}


class Events:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def timeline(event, action, note, actor="系统", at=None):
        event["timeline"].append({"at": at or now(), "action": action, "note": note, "actor": actor})

    def notify(self, event, reason="新事件", contact_index=0):
        contacts = self.store.settings()["contacts"]
        contact = contacts[min(contact_index, len(contacts) - 1)] if contacts else None
        payload = {"title": "紧急求助" if event["level"] == "EMERGENCY" else "疑似跌倒待核实",
                   "message": f"{event['room']} · {reason}", "contact": contact, "mode": event["mode"], "level": event["level"]}
        self.store.notify(event["id"], payload)
        if self.store.settings().get("webhook_url") and event["mode"] == "live":
            self.store.notify(event["id"], payload, kind="webhook")

    def create(self, channel, evidence, mode="live", source="", dedup_key=None, at=None):
        settings = self.store.settings()
        timestamp = at or now()
        if mode == "live" and channel in LINKED_CHANNELS:
            if dedup_key and (known := self.store.event_by_key(dedup_key)):
                return known
            incident, joined = self._incident(channel, timestamp, dedup_key)
            if joined:
                return incident
            if incident:
                return self._link(incident["id"], channel, evidence, source, timestamp, dedup_key)
        emergency = channel in {"manual", "audio"}
        markers = [{"channel": channel, "at": timestamp, "kind": "spoken_help" if channel == "audio" else channel}]
        tag = alarm_tag(markers)
        event = {
            "occurred_at": timestamp, "detected_at": now(), "mode": mode,
            "level": "EMERGENCY" if emergency else "PENDING", "state": "open", "channel": channel,
            "resident": settings["resident"]["name"] or "未命名长者", "room": settings["resident"]["room"],
            "source": source, "evidence": evidence, "markers": tag["markers"], "tag": tag["tag"],
            "channel_quality": [{k: row[k] for k in ("channel", "state", "reason", "started_at")}
                                for row in self.store.coverage() if row["ended_at"] is None],
            "channels": tag["channels"], "timeline": [], "outcome": None, "resolved_at": None,
            "acknowledged_at": None, "contact_index": 0, "next_escalation_at": time.time() + settings["escalation_seconds"],
            "inquiry": {"status": "not_started"}, "notes": "", "response_s": None,
        }
        self.timeline(event, "detected", "主动求助" if emergency else "原算法检测到疑似跌倒，等待照护者核实", at=timestamp)
        self.timeline(event, "notification_queued", "站内通知已排队")
        event, created = self.store.add_event(event, dedup_key)
        if created:
            self.notify(event)
            if mode == "live" and channel in LINKED_CHANNELS:
                event = self._recent_sounds(event)
        return event

    def _incident(self, channel, timestamp, dedup_key=None):
        """(event, already_joined): the open live alarm of another channel that this alarm belongs to, or (None, False)."""
        when = _time(timestamp)
        best = None
        for event in self.store.events(mode="live", state="open", limit=200):
            if dedup_key and any(link.get("dedup_key") == dedup_key for link in event.get("linked", [])):
                return event, True
            # A withdrawn alarm stays apart; a second alarm of the same channel is a new alarm, as before.
            if when is None or is_withdrawn(event) or channel in {m["channel"] for m in event.get("markers", [])}:
                continue
            gap = _gap(event, when)
            if gap is not None and gap <= LINK_WINDOW_S and (best is None or gap < best[0]):
                best = (gap, event)
        return (best[1] if best else None), False

    def _link(self, event_id, channel, evidence, source, timestamp, dedup_key):
        raised = []

        def mutate(event):
            kind = "spoken_help" if channel == "audio" else channel
            tag = alarm_tag([*event["markers"], {"channel": channel, "at": timestamp, "kind": kind}])
            event.update(markers=tag["markers"], channels=tag["channels"], tag=tag["tag"])
            event.setdefault("linked", []).append({"channel": channel, "at": timestamp, "detected_at": now(), "source": source,
                                                   "evidence": evidence, "dedup_key": dedup_key})
            if channel == "audio" and event["level"] != "EMERGENCY":
                event.update(level="EMERGENCY", escalated_at=now(),
                             next_escalation_at=time.time() + self.store.settings()["escalation_seconds"])
                raised.append(True)
            self.timeline(event, "linked", f"{CHANNEL_NAMES[channel]}在同一时段报警，已并入本事件" + ("并升级为紧急求助" if raised else ""))

        event = self.store.update_event(event_id, mutate)
        if raised:
            self.notify(event, "跌倒后语音求助", event["contact_index"])
        return event

    def attach_sound(self, sound, mode="live", candidates=None):
        """Note a sound label heard around an open live alarm on that event. Assistive only: level, state and inquiry stay as they are."""
        when = _time(sound.get("t_start_utc"))
        if mode != "live" or sound.get("category") not in SOUND_HINTS or when is None:
            return None
        target = None
        for event in candidates or self.store.events(mode="live", state="open", limit=200):
            gap = _gap(event, when)
            if gap is not None and gap <= LINK_WINDOW_S and (target is None or gap < target[0]):
                target = (gap, event)
        if not target or any(h["sound_id"] == sound.get("event_id") for h in target[1].get("sound_hints", [])):
            return None

        def mutate(event):
            if any(h["sound_id"] == sound.get("event_id") for h in event.get("sound_hints", [])):
                return
            event.setdefault("sound_hints", []).append({"sound_id": sound.get("event_id"), "category": sound["category"],
                                                        "confidence": sound.get("confidence"), "at": sound.get("t_start_utc")})
            confidence = sound.get("confidence")
            score = f"（置信度 {confidence:.0%}）" if isinstance(confidence, (int, float)) else ""
            self.timeline(event, "sound_hint", f"同一时段听到{SOUND_HINTS[sound['category']]}{score}，仅作辅助参考，不改变事件级别")
        return self.store.update_event(target[1]["id"], mutate)

    def _recent_sounds(self, event):
        """Sounds labelled shortly before a new alarm (a fall is heard before the camera or sensor decides) are noted on it too."""
        for row in self.store.sounds(limit=50):
            if row.get("mode") != "live":
                continue
            for sound in (row.get("result") or {}).get("events", []):
                event = self.attach_sound(sound, candidates=[event]) or event
        return event

    def action(self, event_id, action, outcome=None, note="", actor="照护者", revision=None):
        notification = []
        withdrawn = []

        def mutate(event):
            if event["state"] == "resolved":
                raise ValueError("事件已经关闭")
            if action == "acknowledge":
                event["acknowledged_at"] = event["acknowledged_at"] or now()
                self.timeline(event, action, note or "已接手处理", actor)
            elif action == "contact":
                self.timeline(event, action, note or "已尝试联系长者", actor)
            elif action == "escalate":
                event["level"] = "EMERGENCY"
                event["contact_index"] += 1
                event["next_escalation_at"] = time.time() + self.store.settings()["escalation_seconds"]
                if event.get("withdrawal"):
                    event["withdrawal"]["overridden_at"] = now()  # the caregiver decided that it is real
                self.timeline(event, action, note or "已升级并通知下一位联系人", actor)
                notification.append(True)
            elif action == "withdraw":
                if event.get("withdrawal"):
                    return  # already withdrawn: nothing to do
                if not withdraw_status(event)["withdrawable"] and event["channel"] not in SELF_TRIGGERED and not event.get("escalated_at"):
                    raise ValueError("\u53ea\u6709\u6c42\u52a9\u6309\u94ae\u3001\u8bed\u97f3\u6c42\u52a9\u6216\u5df2\u5347\u7ea7\u7684\u8dcc\u5012\u8b66\u62a5\u624d\u53ef\u4ee5\u7531\u957f\u8005\u64a4\u56de")
                if not withdraw_status(event)["withdrawable"]:
                    raise ValueError(f"\u64a4\u56de\u65f6\u9650\uff08{WITHDRAW_WINDOW_S} \u79d2\uff09\u5df2\u8fc7\uff0c\u8bf7\u8054\u7cfb\u7167\u62a4\u8005")
                event["withdrawal"] = {"at": now(), "by": "elder"}
                self.timeline(event, action, "\u957f\u8005\u64a4\u56de\u4e86\u6c42\u52a9\uff0c\u7b49\u5f85\u7167\u62a4\u8005\u786e\u8ba4\uff1b\u4e8b\u4ef6\u7ea7\u522b\u548c\u8bb0\u5f55\u4fdd\u7559\uff0c\u4e0d\u518d\u9010\u4f4d\u901a\u77e5\u8054\u7cfb\u4eba", "\u957f\u8005")
                withdrawn.append(True)
            elif action == "resolve":
                if outcome not in {"confirmed_fall", "false_alarm", "safe", "unresolved"}:
                    raise ValueError("请选择处置结果")
                event.update(state="resolved", outcome=outcome, resolved_at=now(), notes=note)
                event["response_s"] = max(0, (datetime.fromisoformat(event["resolved_at"]) - datetime.fromisoformat(event["detected_at"])).total_seconds())
                self.timeline(event, action, note or outcome, actor)
            elif action == "safe":
                result = inquiry_outcome({"status": "RESPONDED", "intent": "OTHER", "prompt_played": True},
                                         decision_level=event["level"], button_ok=True)
                ongoing = event["inquiry"].get("status") in {"listening", "processing"}
                event["inquiry"].update(button_ok=True, result=result)
                if not ongoing:
                    event["inquiry"]["status"] = "button_safe"
                self.timeline(event, action, "长者表示安全，保留事件等待照护者核实", "长者")
            else:
                raise ValueError("未知操作")

        event = self.store.update_event(event_id, mutate, revision)
        if notification:
            self.notify(event, "事件升级", event["contact_index"])
        if withdrawn:
            payload = {"title": "\u957f\u8005\u5df2\u64a4\u56de\u6c42\u52a9", "message": f"{event['room']} \u00b7 \u8bf7\u786e\u8ba4\u540e\u5173\u95ed\u4e8b\u4ef6", "contact": None,
                       "mode": event["mode"], "level": event["level"]}
            self.store.notify(event["id"], payload)
            if self.store.settings().get("webhook_url") and event["mode"] == "live":
                self.store.notify(event["id"], payload, kind="webhook")
        return event

    def inquiry_start(self, event_id, prompt_played, reason=None):
        def mutate(event):
            if event["state"] == "resolved":
                raise ValueError("事件已经关闭")
            if event["inquiry"].get("status") in {"listening", "processing"}:
                raise ValueError("询问正在进行")
            event["inquiry"] = {"status": "listening" if prompt_played else "audio_unavailable", "prompt_played": prompt_played,
                                "asked_at": now(), "deadline": time.time() + 10, "reason": reason}
            self.timeline(event, "inquiry_started", "语音询问已播放，等待回答" if prompt_played else "语音询问不可用", "长者端")
        return self.store.update_event(event_id, mutate)

    def inquiry_processing(self, event_id):
        def mutate(event):
            if event["state"] == "resolved" or event["inquiry"].get("status") != "listening":
                raise ValueError("当前没有正在等待回复的询问")
            event["inquiry"].update(status="processing", processing_deadline=time.time() + 125)
        return self.store.update_event(event_id, mutate)

    def inquiry_reply(self, event_id, record):
        notification = []

        existing = self.store.event(event_id)
        if existing["state"] == "resolved":
            decision = inquiry_outcome({**record, "prompt_played": existing["inquiry"].get("prompt_played")},
                                       decision_level=existing["level"], resolved=True)
            if decision["call"]:
                evidence = {k: v for k, v in record.items() if k not in {"transcript", "error"}}
                return self.create("audio", evidence, existing["mode"], "已处置事件后的语音求助")
            raise ValueError("事件已经关闭，回复未改变事件状态")

        def mutate(event):
            if event["state"] == "resolved":
                raise ValueError("事件已关闭，求助请建立新事件")
            if event["inquiry"].get("status") not in {"listening", "processing"}:
                raise ValueError("当前没有正在等待回复的询问")
            reply = {k: v for k, v in record.items() if k not in {"transcript", "error"}}
            reply["prompt_played"] = event["inquiry"].get("prompt_played", False)
            decision = inquiry_outcome(reply, decision_level=event["level"], button_ok=event["inquiry"].get("button_ok", False))
            if decision["marker"]:
                tag = alarm_tag([*event["markers"], {"channel": "audio", "kind": decision["marker"], "at": event["inquiry"]["asked_at"]}])
                event.update(markers=tag["markers"], channels=tag["channels"], tag=tag["tag"])
            if decision["escalate"]:
                if event["level"] != "EMERGENCY":
                    event["escalated_at"] = now()
                event["level"] = "EMERGENCY"
            if decision["marker"] == "spoken_help" and event.get("withdrawal") and not event["withdrawal"].get("overridden_at"):
                event["withdrawal"]["overridden_at"] = now()
                event["next_escalation_at"] = time.time() + self.store.settings()["escalation_seconds"]
                self.timeline(event, "withdrawal_overridden", "\u957f\u8005\u518d\u6b21\u6c42\u52a9\uff0c\u64a4\u56de\u5df2\u5931\u6548", "\u957f\u8005")
            if decision["escalate"] or decision["notify"]:
                notification.append(True)
            event["inquiry"].update(status="completed", record=reply, result=decision, completed_at=now())
            self.timeline(event, "inquiry_completed", f"回复结果：{reply.get('intent') or reply.get('status')}；" + ("已升级" if decision["escalate"] else "保留待核实事件"))
        event = self.store.update_event(event_id, mutate)
        if notification:
            self.notify(event, "语音询问结果需要协助", event["contact_index"])
        return event

    def tick(self):
        settings = self.store.settings()
        for event in self.store.events(state="open", limit=10000):
            inquiry = event["inquiry"]
            # Reserve a short transport margin; the recording still covers exactly
            # the original 10 s response window. ASR has a separate bounded wait.
            if inquiry.get("status") == "listening" and time.time() >= inquiry["deadline"] + 2:
                self.inquiry_reply(event["id"], {"status": "NO_RESPONSE"})
            elif inquiry.get("status") == "processing" and time.time() >= inquiry["processing_deadline"]:
                self.inquiry_reply(event["id"], {"status": "audio_unavailable"})
            if (event["level"] == "EMERGENCY" and not event.get("acknowledged_at") and event["mode"] == "live"
                    and not is_withdrawn(event)
                    and time.time() >= event["next_escalation_at"] and event["contact_index"] + 1 < len(settings["contacts"])):
                self.action(event["id"], "escalate", actor="通知调度")
