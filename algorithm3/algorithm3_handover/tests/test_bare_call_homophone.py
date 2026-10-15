"""A bare, short Mandarin call for help that the recogniser writes with a homophone is still a call (text only, no models)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from homeease.core.clock import VirtualClock
from homeease.core.config import AppConfig, load_config
from homeease.ml.audio import intent_v2 as I
from homeease.ml.audio.help_listener import HelpListener
from homeease.ml.audio.pipeline import SpeechSegment

R2 = I.RuleIntentV2()

# What Whisper tiny actually wrote for a half-second "jiu ming" in the evaluation probes, and the same homophones repeated or with a particle.
HOMOPHONE_CALLS = [
    "\u5c31\u547d", "\u5c31\u547d\u554a", "\u5c31\u547d\u5c31\u547d", "\u5c31\u547d\uff0c\u5c31\u547d\uff01", "\u5c31\u6709\u547d\u4e86", "\u65e7\u547d", "\u8205\u547d", "\u4e5d\u547d", "\u6551\u547d", "\u6551\u547d\u554a", "\u6551\u547d\u6551\u547d", "\u6551\u547d\uff01\u6551\u547d\uff01", "\u5c31\u547d\u5440 \u5c31\u547d\u5566",
]
# The same characters inside an ordinary sentence, or as a title: nothing calls for help.
NOT_CALLS = [
    "\u6211\u5c31\u547d\u4ee4\u4f60\u4eec\u51fa\u53bb", "\u4e5d\u547d\u732b\u5996\u662f\u4e00\u90e8\u7535\u5f71", "\u65e7\u547d\u8fd0", "\u8fd9\u4e2a\u5c31\u547d\u540d\u4e3a\u5c0f\u660e", "\u4eca\u5929\u5c31\u547d\u540d\u5b8c\u4e86", "\u5c31\u547d\u8fd0\u6765\u8bf4",
]


@pytest.mark.parametrize("text", HOMOPHONE_CALLS)
def test_a_bare_homophone_call_is_help(text: str) -> None:
    helps, oks = R2.cues(text)
    assert "zh_help" in helps and not oks, (helps, oks)
    assert R2.predict(text)["intent"] == "HELP"


@pytest.mark.parametrize("text", NOT_CALLS)
def test_the_same_characters_in_a_sentence_are_not_a_call(text: str) -> None:
    assert "zh_help" not in R2.cues(text)[0], text
    assert R2.predict(text)["intent"] != "HELP", text


def test_a_refusal_still_wins_over_the_homophone() -> None:
    assert R2.predict("\u4e0d\u7528\u6551\u547d")["intent"] != "HELP"
    assert R2.predict("\u6211\u4e0d\u9700\u8981\u5e2e\u52a9")["intent"] == "REASSURANCE"


class _Asr:
    def __init__(self, text: str, logprob: float) -> None:
        self._r = {"text": text, "avg_logprob": logprob, "model": "stub"}

    def transcribe(self, samples: Any, language: Any = None) -> dict[str, Any]:
        return dict(self._r)


def _judge(text: str, logprob: float = -0.8, seconds: float = 0.5) -> str:
    cfg: AppConfig = load_config(overrides={"audio": {"enabled": True, "help_listener": {"enabled": True}}})
    clock = VirtualClock(1_000_000.0)
    calls: list[dict[str, Any]] = []
    listener = HelpListener(cfg, clock, on_help=calls.append, consent=lambda: True, asr=_Asr(text, logprob))
    seg = SpeechSegment(t_start_mono_ms=1_000_000.0, t_end_mono_ms=1_000_000.0 + seconds * 1000, t_start_utc="2026-01-01T00:00:00+00:00",
                        t_end_utc="2026-01-01T00:00:01+00:00", duration_s=seconds, sr=16000, truncated=False,
                        samples=np.zeros(int(16000 * seconds), dtype=np.float32))
    return listener.process(seg)


def test_the_listener_raises_the_alarm_for_the_homophone() -> None:
    assert _judge("\u5c31\u547d") == "help"
    assert _judge("\u5c31\u6709\u547d\u4e86") == "help"


def test_the_listener_still_ignores_what_it_ignored_before() -> None:
    assert _judge("\u4eca\u5929\u5929\u6c14\u4e0d\u9519") == "not_help"
    assert _judge("\u6211\u5c31\u547d\u4ee4\u4f60\u4eec\u51fa\u53bb") == "not_help"
    assert _judge("\u5c31\u547d", logprob=-1.4) == "low_confidence"  # the confidence gate is unchanged
