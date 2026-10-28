"""Synthetic audio scenes and stubs for the offline audio tests (no network, no microphone, no model files).

Everything is generated from a seeded RNG, so a scene is bit-identical on every run. The stub
classifier and the stub VAD are deliberately crude spectral rules: the tests exercise the gate, the
monitor logic and the privacy guarantees, not recognition accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

SR = 16000
CLASSES = ["HELP", "IMPACT", "ABNORMAL", "DAILY"]


def amp(dbfs: float) -> float:
    """Peak amplitude of a sine at the given RMS level in dBFS (rms = peak / sqrt 2)."""
    return float(10 ** (dbfs / 20.0) * np.sqrt(2.0))


def noise_bed(seconds: float, dbfs: float = -55.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * 10 ** (dbfs / 20.0)).astype(np.float32)


def speechlike(seconds: float, dbfs: float = -22.0, f0: float = 220.0) -> np.ndarray:
    """Harmonic complex with a 4 Hz syllable envelope and gentle vibrato: voiced-speech-like, narrowband around f0."""
    t = np.arange(int(seconds * SR)) / SR
    ph = 2 * np.pi * f0 * t + 2.0 * np.sin(2 * np.pi * 5.0 * t)
    x = sum(np.sin(k * ph) / k for k in range(1, 8))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t) ** 2
    x = x * env
    return (x / np.sqrt(np.mean(x**2)) * 10 ** (dbfs / 20.0)).astype(np.float32)


def tone(seconds: float, freq: float, dbfs: float = -25.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * freq * t) * amp(dbfs)).astype(np.float32)


def impact(seconds: float = 0.25, peak: float = 0.5, seed: int = 1) -> np.ndarray:
    """Broadband noise burst with a fast exponential decay (a thud / crash proxy)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    env = np.exp(-np.arange(n) / (0.04 * SR))
    x = rng.standard_normal(n) * env
    return (x / np.abs(x).max() * peak).astype(np.float32)


@dataclass(frozen=True)
class Placed:
    kind: str
    t0: float
    seconds: float


def scene(seconds: float, items: Sequence[tuple[float, np.ndarray, str]], bed_dbfs: float = -55.0, seed: int = 0) -> tuple[np.ndarray, list[Placed]]:
    """Noise bed plus (start_s, signal, kind) items; returns the audio and the ground-truth placements."""
    x = noise_bed(seconds, bed_dbfs, seed)
    truth: list[Placed] = []
    for t0, sig, kind in items:
        i = int(t0 * SR)
        x[i:i + len(sig)] += sig[: max(len(x) - i, 0)]
        truth.append(Placed(kind, t0, len(sig) / SR))
    return np.clip(x, -1.0, 1.0).astype(np.float32), truth


class StubSoundClassifier:
    """Spectral rule standing in for a trained model: flat spectrum -> IMPACT, low harmonic -> DAILY,
    mid tone -> ABNORMAL, high tone -> HELP; a near-silent window gets a flat posterior (-> UNKNOWN)."""

    model_id = "stub_rule"
    model_version = "0"
    classes = CLASSES

    def __init__(self, window_s: float = 2.0, min_conf: float | None = None) -> None:
        self.window_s = window_s
        self.min_conf = min_conf
        self.calls = 0
        self.window_lengths: list[int] = []

    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.window_lengths.append(len(window))
        x = np.asarray(window, dtype=np.float64)
        rms_db = 10 * np.log10(np.mean(x**2) + 1e-12)
        if rms_db < -52.0:
            return np.array([0.25, 0.25, 0.25, 0.25])
        p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 + 1e-18
        f = np.fft.rfftfreq(len(x), 1 / SR)
        flat = float(np.exp(np.mean(np.log(p))) / np.mean(p))
        fpk = float(f[int(np.argmax(p))])
        k = 1 if flat > 0.02 else 3 if fpk < 400 else 2 if fpk < 900 else 0
        out = np.full(4, 0.1 / 3)
        out[k] = 0.9
        return out


class HarmonicStubVAD:
    """Speech iff most of a 10 ms frame's energy lies in 150-1500 Hz (voiced-speech-like) and it is not quiet."""

    name = "stub_harmonic"

    def frames(self, x: np.ndarray, sr: int = SR) -> np.ndarray:
        n = len(x) // 160
        if n == 0:
            return np.zeros(0, dtype=np.int8)
        fr = x[: n * 160].reshape(n, 160).astype(np.float64)
        spec = np.abs(np.fft.rfft(fr * np.hanning(160), n=512, axis=1)) ** 2
        f = np.fft.rfftfreq(512, 1 / sr)
        band = spec[:, (f >= 150) & (f <= 1500)].sum(axis=1)
        ratio = band / (spec.sum(axis=1) + 1e-18)
        level = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-12)
        return ((ratio > 0.6) & (level > -40.0)).astype(np.int8)


# ---------------------------------------------------------------------------------------------------- end-to-end stubs (scenes, voice branch)
class ScriptedASR:
    """Stands in for faster-whisper: returns the scripted transcript, or raises (an ASR that is down must never read as 'safe')."""

    def __init__(self, text: str | None, avg_logprob: float = -0.3, error: Exception | None = None) -> None:
        self.text, self.avg_logprob, self.error = text, avg_logprob, error
        self.calls = 0
        self.closed = False

    def transcribe(self, audio: np.ndarray, language: str | None = None, **_: object) -> dict:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {"text": self.text, "avg_logprob": self.avg_logprob, "model": "scripted"}

    def close(self) -> None:
        self.closed = True


class PitchEmbedder:
    """Speaker model stub: a bump at the pitch of the voice (spectral peak 90-400 Hz) on a log-frequency grid, so the same voice matches and another pitch does not."""

    model_id = "stub_pitch"

    def __init__(self) -> None:
        self.calls = 0
        self.centres = np.geomspace(90.0, 400.0, 64)

    def embed(self, audio16k: np.ndarray) -> np.ndarray:
        self.calls += 1
        x = np.asarray(audio16k, dtype=np.float64)
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=1 << 16)) ** 2
        f = np.fft.rfftfreq(1 << 16, 1.0 / SR)
        band = (f >= 90.0) & (f <= 400.0)
        f_peak = float(f[band][int(np.argmax(spec[band]))])
        v = np.exp(-0.5 * (np.log2(self.centres / f_peak) / 0.06) ** 2)
        return (v / np.linalg.norm(v)).astype(np.float32)


VOICE_F0 = {"Zira": 220.0, "David": 170.0}


def fake_tts(text: str, voice: str, lang: str) -> np.ndarray:
    """Harmonic 'speech' whose pitch identifies the voice; length grows a little with the text. No engine, deterministic."""
    seconds = 2.0 + 0.04 * len(text)
    return speechlike(seconds, dbfs=-22.0, f0=VOICE_F0.get(voice, 200.0))


class SyntheticCatalog:
    """Scene source clips generated from code (a hum, an impact, a tone cry, harmonic speech): same interface as ``DiskCatalog``, no data on disk."""

    def __init__(self, missing: Sequence[str] = ()) -> None:
        self.missing = set(missing)

    def candidates(self, role: str) -> list:
        from homeease.ml.audio.scenes import Clip

        if role in self.missing:
            return []
        return [Clip("synthetic", f"synthetic/{role}/{i}.wav", role, 5, f"synthetic:{role}{i}", "generated by the test") for i in range(3)]

    def load(self, clip) -> np.ndarray:  # type: ignore[no-untyped-def]
        role, i = clip.subtype, int(clip.path.split("/")[-1].split(".")[0])
        if role == "background":
            return (tone(5.0, 100.0, -40.0) + tone(5.0, 200.0, -48.0)).astype(np.float32)  # low harmonic hum: the stub classifier calls it DAILY
        if role == "daily":
            return tone(1.0, 300.0 + 20 * i, -30.0)
        if role in ("glass", "thud", "knock"):
            return impact(0.3, 0.5, seed=10 + i)
        if role == "scream":
            return tone(0.9, 1500.0 + 100 * i, -25.0)
        if role == "tv_speech":
            return speechlike(8.0, -30.0, f0=200.0 + 10 * i)
        raise KeyError(role)
