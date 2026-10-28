"""Offline multi-speaker neural TTS for evaluation: many different voices saying the SAME words (experiment A6).

Speaker verification needs many speakers under controlled content, which the three Windows SAPI voices cannot give.
Piper ``en_US-libritts_r-medium`` (VITS, 904 speakers, 22.05 kHz, CC BY 4.0 training data, 78.6 MB) runs on the CPU in tens of
milliseconds per phrase. The model is fetched with ``homeease models fetch --audio-tts`` (pinned revision and sha256 in
``cli_research.AUDIO_MODELS``); the ``piper-tts`` package comes from the optional ``audio-tts`` extra (GPL-3.0, so it is an
evaluation tool and not a runtime dependency).

Synthesised audio exists in memory only. Piper's default sampling noise (``noise_scale`` and ``noise_w_scale``, both 0.333 for this voice model) is not seedable, so two
calls with the same text and speaker differ slightly: experiments that use it save their per-trial tables (scores, transcripts, no audio) and can
be re-analysed from those. ``deterministic=True`` (or ``HOMEEASE_PIPER_DETERMINISTIC=1``) sets both noise scales to 0, which makes the synthesis
bit-identical from call to call (checked on this machine); the voices then sound flatter and the numbers of experiment A6 are not comparable with
runs made with the default sampling. The published A6 runs used the default sampling.
All voices are SYNTHETIC imitations of read audiobook speakers: speaker scores on them say nothing precise about real households.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from homeease.core.paths import models_dir
from homeease.core.util import get_logger
from homeease.ml.audio.features import SR, resample

log = get_logger(__name__)

MODEL_KEY = "piper-libritts-r-medium"  # key in models/audio/MODELS.json (written by `homeease models fetch --audio-tts`)
MODEL_REL = Path("piper") / "en" / "en_US" / "libritts_r" / "medium" / "en_US-libritts_r-medium.onnx"
TARGET_PEAK = 0.5  # every utterance is peak-normalised to this level (the engine already normalises to 1.0)


class MultiSpeakerUnavailable(RuntimeError):
    """The piper-tts package or the voice model is not installed; the message says what to run."""


def model_path() -> Path:
    return models_dir() / "audio" / MODEL_REL


def availability() -> tuple[bool, str | None]:
    """(usable, reason when not). Does not load the model."""
    try:
        import piper  # noqa: F401
    except ImportError:
        return False, "the piper-tts package is not installed (uv pip install 'piper-tts>=1.3', optional extra audio-tts)"
    if not model_path().is_file():
        return False, f"Piper voice model missing: {model_path()} (homeease models fetch --audio-tts)"
    return True, None


def manifest_entry() -> dict[str, Any]:
    """Pinned revision / sha256 / size of the voice model as recorded by `models fetch`; {} when unknown."""
    p = models_dir() / "audio" / "MODELS.json"
    try:
        return dict(json.loads(p.read_text(encoding="utf-8")).get(MODEL_KEY, {}))
    except (OSError, ValueError):
        return {}


class PiperSpeakers:
    """``synthesize(text, speaker_id)`` -> 16 kHz mono float32, peak-normalised. Speaker ids are 0 .. n_speakers-1."""

    def __init__(self, path: str | Path | None = None, deterministic: bool | None = None) -> None:
        self.deterministic = bool(os.environ.get("HOMEEASE_PIPER_DETERMINISTIC", "").strip() not in ("", "0")) if deterministic is None else bool(deterministic)
        ok, why = availability() if path is None else (True, None)
        if not ok:
            raise MultiSpeakerUnavailable(why)
        from piper import PiperVoice

        p = Path(path) if path else model_path()
        if not p.is_file():
            raise MultiSpeakerUnavailable(f"Piper voice model missing: {p}")
        self.voice = PiperVoice.load(str(p))
        self.n_speakers = int(self.voice.config.num_speakers)
        self.sample_rate = int(self.voice.config.sample_rate)
        self.name = "piper:en_US-libritts_r-medium"

    def synthesize(self, text: str, speaker_id: int) -> np.ndarray:
        from piper import SynthesisConfig

        if not 0 <= int(speaker_id) < self.n_speakers:
            raise ValueError(f"speaker id {speaker_id} outside 0..{self.n_speakers - 1}")
        kw = {"noise_scale": 0.0, "noise_w_scale": 0.0} if self.deterministic else {}  # no sampling noise: the same call gives the same samples
        chunks = [c.audio_float_array for c in self.voice.synthesize(text, syn_config=SynthesisConfig(speaker_id=int(speaker_id), **kw))]
        if not chunks:
            raise RuntimeError("Piper returned no audio")
        x = resample(np.concatenate(chunks).astype(np.float32), self.sample_rate, SR)
        peak = float(np.max(np.abs(x))) or 1.0
        return (x / peak * TARGET_PEAK).astype(np.float32)

    def describe(self) -> dict[str, Any]:
        try:
            from importlib import metadata

            ver = metadata.version("piper-tts")
        except Exception:  # pragma: no cover
            ver = None
        m = manifest_entry()
        return {"engine": self.name, "deterministic": self.deterministic, "piper_tts": ver, "n_speakers": self.n_speakers, "sample_rate": self.sample_rate, "model_revision": m.get("revision"),
                "model_sha256": m.get("sha256"), "model_bytes": m.get("bytes"), "license_note": "voice model trained on LibriTTS-R (CC BY 4.0); piper-tts is GPL-3.0"}
