"""Audio research components: VAD alignment, intent rules, speaker metrics."""

from __future__ import annotations

import numpy as np
import pytest

from homeease.ml.audio.intent import RuleIntent
from homeease.ml.audio.speaker import eer, tar_at_far
from homeease.ml.audio.vad import EnergyVAD


def test_energy_vad_detects_speechlike_bursts_and_continuous_sound() -> None:
    sr = 16000
    t = np.arange(10 * sr) / sr
    x = (0.2 * np.sin(2 * np.pi * 200 * t) * ((t > 2) & (t < 4))).astype(np.float32)
    f = EnergyVAD().frames(x)
    assert f[250:350].mean() > 0.9 and f[:150].mean() < 0.05 and f[500:].mean() < 0.05
    cont = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    assert EnergyVAD().frames(cont).mean() > 0.9


def test_silero_frames_align_with_time_axis() -> None:
    pytest.importorskip("silero_vad")
    from homeease.ml.audio.vad import SileroVAD

    v = SileroVAD()
    x = np.zeros(16000 * 12, dtype=np.float32)
    p = v.probs(x)
    assert len(p) == len(x) // 160


@pytest.mark.parametrize("text,intent", [
    ("I'm fine, don't worry", "REASSURANCE"), ("\u6211\u6ca1\u4e8b", "REASSURANCE"), ("help me please", "HELP"), ("\u6551\u547d", "HELP"),
    ("I'm fine but I can't get up", "HELP"), ("I'm not okay", "HELP"), ("\u6211\u4e0d\u597d", "HELP"), ("what time is it", "OTHER"), ("", "UNCERTAIN"),
])
def test_rule_intent_is_conservative(text: str, intent: str) -> None:
    assert RuleIntent().predict(text)["intent"] == intent


def test_speaker_metrics_report_trial_counts_and_reliability() -> None:
    rng = np.random.default_rng(0)
    gen = rng.normal(0.7, 0.1, 200)
    imp = rng.normal(0.1, 0.1, 500)
    s = np.concatenate([gen, imp])
    y = np.concatenate([np.ones(200), np.zeros(500)]).astype(int)
    assert eer(s, y)["eer"] < 0.05
    t = tar_at_far(s, y, 0.001)
    assert t["impostor_trials"] == 500 and t["reliable"] is False and "cannot be estimated reliably" in t["note"]
