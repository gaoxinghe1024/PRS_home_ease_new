"""语音问询结论：把事件里已经落库的 ``inquiry`` 归类成报告需要的三种结论。

跌倒事件建立后，语音问询的结果由 ``care/events.py`` 按原算法策略写入事件的
``inquiry`` 字段；本模块只做**只读归类**，供报告、导出和工作台展示：

* ``answered`` 已应答：确实拿到了可判定的回答。细分三档——
  ``safe`` 长者表示安全（可信语音确认或按下「我没事」按钮）、``help`` 回答中明确呼救、
  ``unclear`` 有回答但不构成有效的安全确认。
* ``high_risk`` 高风险：问询已经发起，但没有得到有效回答（静音无回答、回答无法识别）。
* ``not_asked`` 未问询：问询根本没有发生（尚未发起、正在等待或识别中、音频与识别不可用）。

本模块不改写任何事件状态，也不改变原算法的判定：语音回复依旧不会自行关闭或降级事件，
事件级别与处置结论仍由 ``inquiry_outcome`` 与照护者决定。原始录音与识别文字不参与归类。
"""

ANSWERED = "answered"
HIGH_RISK = "high_risk"
NOT_ASKED = "not_asked"

STATES = (ANSWERED, HIGH_RISK, NOT_ASKED)

LABELS = {ANSWERED: "Answered", HIGH_RISK: "High risk", NOT_ASKED: "Not asked"}

# 已应答的三个细分档；未应答与未问询没有细分。
ANSWER_KINDS = {"safe": "Said they are safe", "help": "Called for help", "unclear": "Not confirmed"}

CSS_CLASSES = {ANSWERED: "ok", HIGH_RISK: "risk", NOT_ASKED: "muted"}


def _conclusion(state, kind, reason, at=None):
    return {"state": state, "label": LABELS[state], "kind": kind,
            "detail": ANSWER_KINDS.get(kind) if kind else None,
            "reason": reason, "at": at,
            "answered": state == ANSWERED, "high_risk": state == HIGH_RISK,
            "css": CSS_CLASSES[state]}


def voice_inquiry(event):
    """事件 → 语音问询结论。任何字段缺失都降级为「未问询」，绝不抛异常。"""
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

    # 长者按下「我没事」按钮：长者的主动确认，尚未进入录音问询也算已应答。
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
    """报告用的三态计数。"""
    states = [voice_inquiry(event)["state"] for event in events]
    return {state: states.count(state) for state in STATES}
