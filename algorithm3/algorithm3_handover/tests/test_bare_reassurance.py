"""A bare, short Mandarin "I am fine" ("mei shi") is recognised with a low confidence; the cancel test gives only that form some slack."""

from __future__ import annotations

import pytest

from algorithm3 import policy
from homeease.ml.audio.intent_v2 import RuleIntentV2

R2 = RuleIntentV2()

BARE = ["\u6ca1\u4e8b", "\u6ca1\u4e8b\u513f", "\u6ca1\u4e8b\u4e86", "\u6ca1\u4e8b\u5566", "\u6ca1\u95ee\u9898", "\u6ca1\u5173\u7cfb", "\u4e0d\u8981\u7d27", "\u6211\u6ca1\u4e8b", "\u6211\u6ca1\u4e8b\u513f", "\u6ca1\u4e8b\u6ca1\u4e8b", "\u6c92\u4e8b", "\u6211\u6c92\u4e8b", "\u90fd\u6ca1\u4e8b\u4e86", "\u6ca1\u4e8b,\u6ca1\u4e8b!", "\u6ca1\u4e8b \u554a"]
NOT_BARE = ["\u6ca1\u4e8b\u5427", "\u6ca1\u4e8b\u5417", "\u6211\u6ca1\u4e8b\u4f46\u662f\u8d77\u4e0d\u6765\u4e86", "\u6ca1\u4e8b,\u6551\u547d", "\u4ed6\u6ca1\u4e8b", "\u6709\u4e8b", "\u51fa\u4e8b\u4e86", "\u6ca1\u6709", "\u4e0d\u662f\u6ca1\u4e8b", "\u6211\u6ca1\u4e8b,\u4e0d\u7528\u62c5\u5fc3", "I'm fine", "Fine", "\u597d\u7684", "\u6ca1\u94b1", ""]


@pytest.mark.parametrize("text", BARE)
def test_a_bare_mandarin_im_fine_is_recognised(text: str) -> None:
    assert R2.is_bare_reassurance(text), text
    assert R2.predict(text)["intent"] == "REASSURANCE"


@pytest.mark.parametrize("text", NOT_BARE)
def test_nothing_else_is_a_bare_reassurance(text: str) -> None:
    assert not R2.is_bare_reassurance(text), text


def rec(conf: float, **extra) -> dict:
    return {"status": "RESPONDED", "intent": "REASSURANCE", "asr_avg_logprob": conf, "prompt_played": True, **extra}


def test_the_bare_form_may_be_a_little_below_the_gate() -> None:
    assert policy.voice_cancel_verdict(rec(-1.04)) is False  # the gate as it was
    assert policy.voice_cancel_verdict(rec(-1.04, bare_reassurance=True)) is True
    assert policy.voice_cancel_verdict(rec(-1.22, bare_reassurance=True)) is True
    assert policy.voice_cancel_verdict(rec(-1.29, bare_reassurance=True)) is True
    assert policy.voice_cancel_verdict(rec(-1.31, bare_reassurance=True)) is False  # but not further
    assert policy.voice_cancel_verdict(rec(float("nan"), bare_reassurance=True)) is False


def test_the_slack_is_relative_to_the_configured_gate() -> None:
    assert policy.voice_cancel_verdict(rec(-0.7, bare_reassurance=True), min_asr_logprob=-0.5) is True
    assert policy.voice_cancel_verdict(rec(-0.9, bare_reassurance=True), min_asr_logprob=-0.5) is False  # a strict gate is not loosened beyond the slack
    assert policy.voice_cancel_verdict(rec(-0.7), min_asr_logprob=-0.5) is False


def test_the_slack_changes_nothing_else_about_a_cancel() -> None:
    assert policy.voice_cancel_verdict(rec(-1.1, bare_reassurance=True, speaker_verified=False)) is False  # another voice
    assert policy.voice_cancel_verdict(rec(-1.1, bare_reassurance=True, voiceprint_enrolled=True)) is False  # enrolled, voices not compared
    assert policy.voice_cancel_verdict(rec(-1.1, bare_reassurance=True, speaker_verified=True)) is True
    assert policy.voice_cancel_verdict({**rec(-1.1, bare_reassurance=True), "intent": "OTHER"}) is False
    assert policy.voice_cancel_verdict({**rec(-1.1, bare_reassurance=True), "intent": "HELP"}) is False


def test_the_outcome_of_the_inquiry_follows() -> None:
    cancelled = policy.inquiry_outcome(rec(-1.1, bare_reassurance=True), decision_level="PENDING")
    assert cancelled["state"] == "cancelled" and cancelled["escalate"] is False
    kept = policy.inquiry_outcome(rec(-1.1), decision_level="PENDING")
    assert kept["state"] == "no_cancel" and kept["escalate"] is True
    far = policy.inquiry_outcome(rec(-1.4, bare_reassurance=True), decision_level="PENDING")
    assert far["state"] == "no_cancel" and far["escalate"] is True


def test_classify_reply_marks_the_bare_form_and_only_that() -> None:
    from algorithm3 import Algorithm3

    a3 = Algorithm3()
    assert a3.classify_reply(text="\u6ca1\u4e8b")["bare_reassurance"] is True
    assert a3.classify_reply(text="\u6211\u6ca1\u4e8b")["bare_reassurance"] is True
    for text in ("\u6ca1\u4e8b\u5427", "\u6211\u6ca1\u4e8b,\u4e0d\u7528\u62c5\u5fc3", "I'm fine", "\u6551\u547d", "\u6211\u6ca1\u4e8b\u4f46\u662f\u8d77\u4e0d\u6765\u4e86"):
        assert a3.classify_reply(text=text)["bare_reassurance"] is False, text
