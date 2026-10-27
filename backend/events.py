import sys
import time
from datetime import datetime, timezone

from backend.config import AUDIO
from backend.storage import now

# Only the original pure policy functions run in the API process.
sys.path.append(str(AUDIO / "src"))
from algorithm3.policy import alarm_tag, inquiry_outcome

# The older adult can withdraw an alarm for a short time: one from the help button or a spoken call, or a sensor fall alarm once the
# voice inquiry has raised it to an emergency (the time then counts from that moment). A withdrawal never closes or lowers the event;
# it stops the contact escalation, tells the caregivers, and a caregiver still closes the event.
WITHDRAW_WINDOW_S = 30
SELF_TRIGGERED = ("manual", "audio")


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
    def timeline(event, action, note, actor="System", at=None):
        event["timeline"].append({"at": at or now(), "action": action, "note": note, "actor": actor})

    def notify(self, event, reason="New event", contact_index=0):
        contacts = self.store.settings()["contacts"]
        contact = contacts[min(contact_index, len(contacts) - 1)] if contacts else None
        payload = {"title": "Emergency call for help" if event["level"] == "EMERGENCY" else "Suspected fall, pending verification",
                   "message": f"{event['room']} · {reason}", "contact": contact, "mode": event["mode"], "level": event["level"]}
        self.store.notify(event["id"], payload)
        if self.store.settings().get("webhook_url") and event["mode"] == "live":
            self.store.notify(event["id"], payload, kind="webhook")

    def create(self, channel, evidence, mode="live", source="", dedup_key=None, at=None):
        settings = self.store.settings()
        timestamp = at or now()
        emergency = channel in {"manual", "audio"}
        markers = [{"channel": channel, "at": timestamp, "kind": "spoken_help" if channel == "audio" else channel}]
        tag = alarm_tag(markers)
        event = {
            "occurred_at": timestamp, "detected_at": now(), "mode": mode,
            "level": "EMERGENCY" if emergency else "PENDING", "state": "open", "channel": channel,
            "resident": settings["resident"]["name"] or "Unnamed older adult", "room": settings["resident"]["room"],
            "source": source, "evidence": evidence, "markers": tag["markers"], "tag": tag["tag"],
            "channel_quality": [{k: row[k] for k in ("channel", "state", "reason", "started_at")}
                                for row in self.store.coverage() if row["ended_at"] is None],
            "channels": tag["channels"], "timeline": [], "outcome": None, "resolved_at": None,
            "acknowledged_at": None, "contact_index": 0, "next_escalation_at": time.time() + settings["escalation_seconds"],
            "inquiry": {"status": "not_started"}, "notes": "", "response_s": None,
        }
        self.timeline(event, "detected", "Asked for help" if emergency else "The original algorithm detected a suspected fall; waiting for a caregiver to verify", at=timestamp)
        self.timeline(event, "notification_queued", "In-app notification queued")
        event, created = self.store.add_event(event, dedup_key)
        if created:
            self.notify(event)
        return event

    def action(self, event_id, action, outcome=None, note="", actor="Caregiver", revision=None):
        notification = []
        withdrawn = []

        def mutate(event):
            if event["state"] == "resolved":
                raise ValueError("The event is already closed")
            if action == "acknowledge":
                event["acknowledged_at"] = event["acknowledged_at"] or now()
                self.timeline(event, action, note or "Taken over", actor)
            elif action == "contact":
                self.timeline(event, action, note or "An attempt was made to contact the older adult", actor)
            elif action == "escalate":
                event["level"] = "EMERGENCY"
                event["contact_index"] += 1
                event["next_escalation_at"] = time.time() + self.store.settings()["escalation_seconds"]
                if event.get("withdrawal"):
                    event["withdrawal"]["overridden_at"] = now()  # the caregiver decided that it is real
                self.timeline(event, action, note or "Escalated and the next contact was notified", actor)
                notification.append(True)
            elif action == "withdraw":
                if event.get("withdrawal"):
                    return  # already withdrawn: nothing to do
                if not withdraw_status(event)["withdrawable"] and event["channel"] not in SELF_TRIGGERED and not event.get("escalated_at"):
                    raise ValueError("Only an alarm raised by the help button, a spoken call for help, or a fall alarm that has already been escalated can be withdrawn by the older adult")
                if not withdraw_status(event)["withdrawable"]:
                    raise ValueError(f"The withdrawal window ({WITHDRAW_WINDOW_S} seconds) has passed. Please contact the caregiver")
                event["withdrawal"] = {"at": now(), "by": "elder"}
                self.timeline(event, action, "The older adult withdrew the request for help and is waiting for the caregiver to confirm. The event level and records are kept and contacts are no longer notified one by one.", "Older adult")
                withdrawn.append(True)
            elif action == "resolve":
                if outcome not in {"confirmed_fall", "false_alarm", "safe", "unresolved"}:
                    raise ValueError("Please choose a resolution")
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
                self.timeline(event, action, "The older adult says they are safe. The event is kept for the caregiver to verify", "Older adult")
            else:
                raise ValueError("Unknown action")

        event = self.store.update_event(event_id, mutate, revision)
        if notification:
            self.notify(event, "Event escalation", event["contact_index"])
        if withdrawn:
            payload = {"title": "The older adult withdrew the call for help", "message": f"{event['room']} · Please confirm and close the event", "contact": None,
                       "mode": event["mode"], "level": event["level"]}
            self.store.notify(event["id"], payload)
            if self.store.settings().get("webhook_url") and event["mode"] == "live":
                self.store.notify(event["id"], payload, kind="webhook")
        return event

    def inquiry_start(self, event_id, prompt_played, reason=None):
        def mutate(event):
            if event["state"] == "resolved":
                raise ValueError("The event is already closed")
            if event["inquiry"].get("status") in {"listening", "processing"}:
                raise ValueError("An inquiry is in progress")
            event["inquiry"] = {"status": "listening" if prompt_played else "audio_unavailable", "prompt_played": prompt_played,
                                "asked_at": now(), "deadline": time.time() + 10, "reason": reason}
            self.timeline(event, "inquiry_started", "The voice question was played; waiting for the answer" if prompt_played else "The voice question is unavailable", "Older adult page")
        return self.store.update_event(event_id, mutate)

    def inquiry_processing(self, event_id):
        def mutate(event):
            if event["state"] == "resolved" or event["inquiry"].get("status") != "listening":
                raise ValueError("There is no inquiry waiting for a reply")
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
                return self.create("audio", evidence, existing["mode"], "Spoken call for help after the event was closed")
            raise ValueError("The event is already closed; the reply did not change its state")

        def mutate(event):
            if event["state"] == "resolved":
                raise ValueError("The event is closed; a new call for help creates a new event")
            if event["inquiry"].get("status") not in {"listening", "processing"}:
                raise ValueError("There is no inquiry waiting for a reply")
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
                self.timeline(event, "withdrawal_overridden", "The older adult called for help again; the withdrawal no longer applies", "Older adult")
            if decision["escalate"] or decision["notify"]:
                notification.append(True)
            event["inquiry"].update(status="completed", record=reply, result=decision, completed_at=now())
            self.timeline(event, "inquiry_completed", f"Reply result: {reply.get('intent') or reply.get('status')}; " + ("Escalated" if decision["escalate"] else "The event is kept pending verification"))
        event = self.store.update_event(event_id, mutate)
        if notification:
            self.notify(event, "The voice inquiry result needs assistance", event["contact_index"])
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
                self.action(event["id"], "escalate", actor="Notification scheduler")
