"""Offline multi-speaker TTS wrapper (ml/audio/multi_tts.py): availability reporting, speaker-id bounds, level and sample-rate handling with a fake `piper`
module (no model, no package needed), plus one real-engine check that is skipped when Piper or its voice model is not installed."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

from homeease.ml.audio import multi_tts as MT


class FakeChunk:
    def __init__(self, x: np.ndarray) -> None:
        self.audio_float_array = x


class FakeVoice:
    def __init__(self, n: int = 5, sr: int = 22050) -> None:
        self.config = types.SimpleNamespace(num_speakers=n, sample_rate=sr)
        self.calls: list[tuple[str, int]] = []

    def synthesize(self, text: str, syn_config: Any = None) -> Iterator[FakeChunk]:
        sid = syn_config.speaker_id
        self.calls.append((text, sid))
        t = np.arange(int(0.5 * self.config.sample_rate)) / self.config.sample_rate
        for k in range(2):  # two sentences -> two chunks
            yield FakeChunk((np.sin(2 * np.pi * (200 + 40 * sid + 30 * k) * t) * (1.0 if k == 0 else 0.3)).astype(np.float32))


@pytest.fixture()
def fake_piper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeVoice:
    voice = FakeVoice()
    mod = types.ModuleType("piper")
    mod.PiperVoice = types.SimpleNamespace(load=lambda path: voice)  # type: ignore[attr-defined]
    mod.SynthesisConfig = lambda speaker_id=None, **kw: types.SimpleNamespace(speaker_id=speaker_id)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", mod)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    monkeypatch.setattr(MT, "model_path", lambda: model)
    return voice


def test_availability_names_what_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "piper", None)  # import piper -> ImportError
    ok, why = MT.availability()
    assert not ok and "piper-tts" in why and "audio-tts" in why
    mod = types.ModuleType("piper")
    monkeypatch.setitem(sys.modules, "piper", mod)
    monkeypatch.setattr(MT, "model_path", lambda: tmp_path / "missing.onnx")
    ok, why = MT.availability()
    assert not ok and "voice model missing" in why and "models fetch --audio-tts" in why
    (tmp_path / "missing.onnx").write_bytes(b"x")
    assert MT.availability() == (True, None)
    monkeypatch.setitem(sys.modules, "piper", None)
    with pytest.raises(MT.MultiSpeakerUnavailable, match="piper-tts"):
        MT.PiperSpeakers()


def test_synthesize_returns_16k_peak_normalised_audio_and_checks_the_speaker_id(fake_piper: FakeVoice) -> None:
    eng = MT.PiperSpeakers()
    assert eng.n_speakers == 5 and eng.sample_rate == 22050
    x = eng.synthesize("I am fine", 2)
    assert x.dtype == np.float32 and x.ndim == 1 and abs(len(x) - 16000) < 400  # 2 x 0.5 s at 22.05 kHz -> about 1 s at 16 kHz
    assert float(np.max(np.abs(x))) == pytest.approx(MT.TARGET_PEAK, abs=1e-4)
    y = eng.synthesize("I am fine", 3)
    assert not np.allclose(x[:8000], y[:8000], atol=0.05)  # different speakers, different audio
    assert fake_piper.calls == [("I am fine", 2), ("I am fine", 3)]
    for bad in (-1, 5, 904):
        with pytest.raises(ValueError, match="outside"):
            eng.synthesize("x", bad)


def test_an_engine_without_output_is_an_error_not_silence(fake_piper: FakeVoice, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = MT.PiperSpeakers()
    monkeypatch.setattr(fake_piper, "synthesize", lambda text, syn_config=None: iter(()))
    with pytest.raises(RuntimeError, match="no audio"):
        eng.synthesize("x", 0)


def test_describe_reads_the_pinned_model_from_the_manifest(fake_piper: FakeVoice, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    models = tmp_path / "models"
    (models / "audio").mkdir(parents=True)
    (models / "audio" / "MODELS.json").write_text(json.dumps({MT.MODEL_KEY: {"revision": "c10ece1a", "sha256": "10bb85e0", "bytes": 78580914}}), encoding="utf-8")
    monkeypatch.setattr(MT, "models_dir", lambda: models)
    d = MT.PiperSpeakers().describe()
    assert d["model_revision"] == "c10ece1a" and d["model_sha256"] == "10bb85e0" and d["n_speakers"] == 5 and "GPL" in d["license_note"]
    monkeypatch.setattr(MT, "models_dir", lambda: tmp_path / "nowhere")
    assert MT.manifest_entry() == {}


def test_the_model_download_entry_is_pinned_and_checksummed() -> None:
    from homeease.cli import AUDIO_MODELS

    m = AUDIO_MODELS["piper-libritts-r-medium"]
    assert len(m["revision"]) == 40 and m["expected_sha256"] == "10bb85e071d616fcf4071f369f1799d0491492ab3c5d552ec19fb548fac13195" and m["repo"] == "rhasspy/piper-voices"
    assert m["check"] == str(MT.MODEL_REL.relative_to("piper")).replace("\\", "/") and m["dest"] == "audio/piper"
    assert AUDIO_MODELS["piper-libritts-r-medium-config"]["check"].endswith(".onnx.json")
    assert m["group"] == "tts" and AUDIO_MODELS["piper-libritts-r-medium-config"]["group"] == "tts"
    assert not [k for k, v in AUDIO_MODELS.items() if v.get("group") != "tts" and "piper" in k]  # nothing of Piper in the default set


def test_the_78mb_voice_model_is_only_fetched_on_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import huggingface_hub

    from homeease import cli

    fetched: list[str] = []
    monkeypatch.setattr(cli, "models_dir", lambda: tmp_path / "models")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, file, revision=None, local_dir=None: fetched.append(file) or (Path(local_dir) / file).parent.mkdir(parents=True, exist_ok=True) or (Path(local_dir) / file).write_bytes(b"x"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, revision=None, local_dir=None: fetched.append(repo) or (Path(local_dir) / "model.bin").parent.mkdir(parents=True, exist_ok=True) or (Path(local_dir) / "model.bin").write_bytes(b"x"))
    entry = {"repo": "r/piper", "revision": "a" * 40, "file": "v.onnx", "dest": "audio/piper", "check": "v.onnx", "license": "x"}
    monkeypatch.setattr(cli, "AUDIO_MODELS", {"plain": {**entry, "repo": "r/plain", "file": "p.onnx", "dest": "audio/plain", "check": "p.onnx"}, "voice": {**entry, "group": "tts"}})
    assert cli.fetch_audio_models() == 0
    assert fetched == ["p.onnx"]  # the default set does not touch the opt-in group
    assert cli.fetch_audio_models(False, ("tts",)) == 0
    assert fetched == ["p.onnx", "v.onnx"]
    manifest = json.loads((tmp_path / "models" / "audio" / "MODELS.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"plain", "voice"}


@pytest.mark.slow
@pytest.mark.realdata
def test_real_piper_gives_different_voices_for_the_same_words() -> None:
    ok, why = MT.availability()
    if not ok:
        pytest.skip(why)
    eng = MT.PiperSpeakers()
    a, b = eng.synthesize("I'm fine, don't worry.", 10), eng.synthesize("I'm fine, don't worry.", 500)
    assert eng.n_speakers == 904 and 0.5 < len(a) / 16000 < 4.0 and 0.5 < len(b) / 16000 < 4.0
    assert float(np.max(np.abs(a))) == pytest.approx(MT.TARGET_PEAK, abs=1e-4)
    from homeease.ml.audio.speaker import WeSpeakerEmbedder

    try:
        emb = WeSpeakerEmbedder()
    except FileNotFoundError:
        return
    a2 = eng.synthesize("Please put the blue cup on the kitchen table.", 10)
    ea, ea2, eb = emb.embed(a), emb.embed(a2), emb.embed(b)
    assert float(np.dot(ea, ea2)) > float(np.dot(ea, eb))  # same speaker, other words, is closer than another speaker, same words


def test_models_fetch_refuses_a_file_whose_sha256_differs_from_the_pinned_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import hashlib

    import huggingface_hub

    from homeease import cli

    payload = b"voice-model-bytes"
    good = hashlib.sha256(payload).hexdigest()
    entry = {"repo": "r/x", "revision": "a" * 40, "file": "sub/m.onnx", "dest": "audio/piper", "check": "sub/m.onnx", "license": "x"}
    monkeypatch.setattr(cli, "models_dir", lambda: tmp_path / "models")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, file, revision=None, local_dir=None: (Path(local_dir) / file).parent.mkdir(parents=True, exist_ok=True) or (Path(local_dir) / file).write_bytes(payload))
    monkeypatch.setattr(cli, "AUDIO_MODELS", {"m": {**entry, "expected_sha256": "0" * 64}})
    assert cli.fetch_audio_models() == 3
    assert "FAILED m: sha256" in capsys.readouterr().err
    manifest = json.loads((tmp_path / "models" / "audio" / "MODELS.json").read_text(encoding="utf-8"))
    assert "m" not in manifest  # a mismatching file is never recorded as good
    monkeypatch.setattr(cli, "AUDIO_MODELS", {"m": {**entry, "expected_sha256": good}})
    assert cli.fetch_audio_models() == 0
    assert json.loads((tmp_path / "models" / "audio" / "MODELS.json").read_text(encoding="utf-8"))["m"]["sha256"] == good


# ------------------------------------------------------------------------------------------------ review (L2-05): deterministic synthesis is switchable
def test_deterministic_mode_turns_the_sampling_noise_off(fake_piper: FakeVoice, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    piper_mod = sys.modules["piper"]
    piper_mod.SynthesisConfig = lambda speaker_id=None, **kw: (seen.append(kw), types.SimpleNamespace(speaker_id=speaker_id))[1]  # type: ignore[attr-defined]
    monkeypatch.delenv("HOMEEASE_PIPER_DETERMINISTIC", raising=False)
    default = MT.PiperSpeakers()
    default.synthesize("hello", 1)
    assert seen[-1] == {} and default.deterministic is False and default.describe()["deterministic"] is False  # the published A6 runs: Piper's own sampling
    det = MT.PiperSpeakers(deterministic=True)
    det.synthesize("hello", 1)
    assert seen[-1] == {"noise_scale": 0.0, "noise_w_scale": 0.0} and det.describe()["deterministic"] is True
    monkeypatch.setenv("HOMEEASE_PIPER_DETERMINISTIC", "1")
    assert MT.PiperSpeakers().deterministic is True and MT.PiperSpeakers(deterministic=False).deterministic is False
    monkeypatch.setenv("HOMEEASE_PIPER_DETERMINISTIC", "0")
    assert MT.PiperSpeakers().deterministic is False
