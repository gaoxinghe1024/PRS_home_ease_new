"""The recogniser-confidence gate of the spoken-call listener is -1.2; the voiceprint gate that decides a cancel stays at -1.0."""

from __future__ import annotations

from typing import Any

import numpy as np

from homeease.core.clock import VirtualClock
from homeease.core.config import AppConfig, load_config
from homeease.ml.audio.help_listener import HelpListener
from homeease.ml.audio.pipeline import SpeechSegment


class _Asr:
    def __init__(self, text: str, logprob: float) -> None:
        self._r = {"text": text, "avg_logprob": logprob, "model": "stub"}

    def transcribe(self, samples: Any, language: Any = None) -> dict[str, Any]:
        return dict(self._r)


def _judge(text: str, logprob: float, seconds: float = 0.6) -> str:
    cfg: AppConfig = load_config(overrides={"audio": {"enabled": True, "help_listener": {"enabled": True}}})
    clock = VirtualClock(1_000_000.0)
    listener = HelpListener(cfg, clock, on_help=lambda call: None, consent=lambda: True, asr=_Asr(text, logprob))
    seg = SpeechSegment(t_start_mono_ms=1_000_000.0, t_end_mono_ms=1_000_000.0 + seconds * 1000, t_start_utc="2026-01-01T00:00:00+00:00",
                        t_end_utc="2026-01-01T00:00:01+00:00", duration_s=seconds, sr=16000, truncated=False,
                        samples=np.zeros(int(16000 * seconds), dtype=np.float32))
    return listener.process(seg)


def test_the_defaults_in_the_config_file_and_in_the_schema_agree() -> None:
    from homeease.core.config import HelpListenerCfg

    cfg = load_config()
    assert cfg.audio.help_listener.min_asr_logprob == -1.2
    assert HelpListenerCfg().min_asr_logprob == -1.2
    assert cfg.audio.voiceprint.min_asr_logprob == -1.0  # a cancel stays as strict as before


def test_a_short_call_just_below_the_old_gate_is_heard_now() -> None:
    assert _judge("Help", -1.0) == "help"
    assert _judge("Help", -1.1) == "help"  # was low_confidence before
    assert _judge("Help", -1.19) == "help"
    assert _judge("\u6551\u547d", -1.15) == "help"


def test_below_the_gate_it_is_still_ignored() -> None:
    assert _judge("Help", -1.21) == "low_confidence"
    assert _judge("\u6551\u547d", -1.5) == "low_confidence"
    assert _judge("Help", float("nan")) == "low_confidence"


def test_the_gate_lets_no_non_call_through() -> None:
    # a lower confidence never turns something that is not a call into one
    for text in ("The weather is nice today", "Thanks for your help", "Help is on the way", "What time is it"):
        assert _judge(text, -1.15) in ("not_help", "not_a_call"), text
    assert _judge("\u6211\u6ca1\u4e8b", -1.15) == "not_help"
    assert _judge("\u8c22\u8c22\u4f60\u7684\u5e2e\u52a9", -1.15) in ("not_help", "not_a_call")
