"""Voice-inquiry conclusion: classify the ``inquiry`` stored in an event into the three conclusions a report needs.

After a fall event is created, ``backend/events.py`` writes the voice-inquiry result into the event's ``inquiry``
field following the original algorithm policy. This module only **classifies it, read-only**, for reports, exports and the workbench:

* ``answered``: an answer that could be judged was received. Three kinds:
  ``safe`` the older adult said they are safe (trusted voice confirmation or the "I'm OK" button), ``help`` the answer clearly called for help,
  ``unclear`` there was an answer, but not a valid confirmation of safety.
* ``high_risk``: the inquiry was started but got no valid answer (silence, or an answer that could not be recognized).
* ``not_asked``: the inquiry never happened (not started, still waiting or recognizing, or audio / speech recognition unavailable).

This module changes no event state and no decision of the original algorithm: a voice reply never closes or downgrades an event,
and the event level and resolution are still decided by ``inquiry_outcome`` and the caregiver. Raw audio and transcripts are not used.
"""

ANSWERED = "answered"
HIGH_RISK = "high_risk"
NOT_ASKED = "not_asked"

STATES = (ANSWERED, HIGH_RISK, NOT_ASKED)

LABELS = {ANSWERED: "Answered", HIGH_RISK: "High risk", NOT_ASKED: "Not asked"}

# The three kinds of an answered inquiry; high risk and not asked have no sub-kinds.
ANSWER_KINDS = {"safe": "Said they are safe", "help": "Called for help", "unclear": "Not confirmed"}

CSS_CLASSES = {ANSWERED: "ok", HIGH_RISK: "risk", NOT_ASKED: "muted"}


def _conclusion(state, kind, reason, at=None):
    return {"state": state, "label": LABELS[state], "kind": kind,
            "detail": ANSWER_KINDS.get(kind) if kind else None,
            "reason": reason, "at": at,
            "answered": state == ANSWERED, "high_risk": state == HIGH_RISK,
            "css": CSS_CLASSES[state]}


def voice_inquiry(event):
    """Event -> voice-inquiry conclusion. Any missing field degrades to "not asked"; it never raises."""
    inquiry = event.get("inquiry") or {}
    record = inquiry.get("record") or {}
    result = inquiry.get("result") or {}
    status = inquiry.get("status") or "not_started"

    if status == "completed":
        reply = record.get("status")
        if reply == "RESPONDED":
            if record.get("intent") == "HELP" or result.get("marker") == "spoken_help":
                return _conclusion(ANSWERED, "help", "The answer clearly called for help", inquiry.get("completed_at"))
            if result.get("cancelled"):
                by = "button" if result.get("by") == "button" else "voice"
                return _conclusion(ANSWERED, "safe", f"The older adult confirmed they are safe by {by}", inquiry.get("completed_at"))
            return _conclusion(ANSWERED, "unclear", "There was an answer, but it was not a valid confirmation of safety", inquiry.get("completed_at"))
        if reply == "NO_RESPONSE":
            return _conclusion(HIGH_RISK, None, "The question was played; there was no answer within the window", inquiry.get("completed_at"))
        if reply == "UNCERTAIN":
            return _conclusion(HIGH_RISK, None, "There was sound, but it could not be recognized as a valid answer", inquiry.get("completed_at"))
        return _conclusion(NOT_ASKED, None, "Audio or speech recognition is unavailable; the inquiry got no answer", inquiry.get("completed_at"))

    # The older adult pressed "I'm OK": an active confirmation that counts as answered even before a recorded inquiry.
    if inquiry.get("button_ok") or status == "button_safe":
        return _conclusion(ANSWERED, "safe", "The older adult pressed the \"I'm OK\" button")

    if status == "listening":
        return _conclusion(NOT_ASKED, None, "The question was played; waiting for the answer")
    if status == "processing":
        return _conclusion(NOT_ASKED, None, "Recognizing the older adult's answer")
    if status == "audio_unavailable":
        return _conclusion(NOT_ASKED, None, inquiry.get("reason") or "The question audio could not be played")
    return _conclusion(NOT_ASKED, None, inquiry.get("reason") or "No voice inquiry has been started")


def voice_counts(events):
    """Counts of the three states for reports."""
    states = [voice_inquiry(event)["state"] for event in events]
    return {state: states.count(state) for state in STATES}
