"""Scripted scenes (`homeease audio scenes`) and the offline replay command (`homeease audio replay`): determinism, graceful degradation when data or a
TTS voice is missing, size budget, held-out sources, exit codes, and a full CLI run with stub models. No microphone, no network, no real model or engine."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from homeease import cli
from homeease.ml.audio import replay as rp
from homeease.ml.audio import scenes as sc
from tests.audio_fixtures import HarmonicStubVAD, PitchEmbedder, ScriptedASR, StubSoundClassifier, SyntheticCatalog, fake_tts, speechlike

pytest.importorskip("soundfile", reason="needs the optional audio extra (uv sync --extra audio)")

SR = 16000


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ scene generation
def test_scenes_are_bit_identical_for_a_seed_and_differ_between_seeds(tmp_path: Path) -> None:
    cat = SyntheticCatalog()
    a, sk_a = sc.generate_scenes(None, 3, tmp_path / "a", cat, fake_tts)
    b, sk_b = sc.generate_scenes(None, 3, tmp_path / "b", cat, fake_tts)
    c, _ = sc.generate_scenes(None, 4, tmp_path / "c", cat, fake_tts)
    assert not sk_a and not sk_b and len(a) == len(b) == len(sc.SCENE_NAMES)
    assert [sha(p) for p in a] == [sha(p) for p in b]
    assert [json.loads(p.with_suffix(".json").read_text(encoding="utf-8")) for p in a] == [json.loads(p.with_suffix(".json").read_text(encoding="utf-8")) for p in b]
    assert any(sha(x) != sha(y) for x, y in zip(a, c))  # the seed changes jitter and clip choice


def test_scene_structure_levels_and_truth(tmp_path: Path) -> None:
    for name in sc.SCENE_NAMES:
        s = sc.build_scene(name, SyntheticCatalog(), seed=11, tts=fake_tts)
        spec = sc.SCENES[name]
        assert abs(s.duration_s - spec.duration_s) < 1e-3 and s.audio.dtype == np.float32 and np.all(np.isfinite(s.audio))
        assert float(np.max(np.abs(s.audio))) <= 0.98 + 1e-6 and float(np.min(np.abs(s.audio))) >= 0.0
        for t in s.truth:
            assert 0.0 <= t["t0_s"] < t["t1_s"] <= s.duration_s + 1e-6
        # the background sits at about -48 dBFS (the quietest second of the scene) and nothing is digitally silent
        quiet = min(10 * np.log10(np.mean(s.audio[i:i + SR] ** 2) + 1e-12) for i in range(0, len(s.audio) - SR, SR))
        assert -52.0 < quiet < -44.0, (name, quiet)
        if spec.fall_at_s is not None:
            assert s.inquiry is not None and s.inquiry["inquiry_start_s"] == pytest.approx(spec.fall_at_s + sc.PROMPT_S)
            reply = next(t for t in s.truth if t["kind"] == "reply")
            assert s.inquiry["inquiry_start_s"] < reply["t0_s"] < s.inquiry["inquiry_start_s"] + s.inquiry["window_s"] and reply["t1_s"] <= s.inquiry["inquiry_start_s"] + s.inquiry["window_s"]
            assert s.inquiry["reply_text"] == spec.reply[1] and "never creates" in s.inquiry["note"]  # type: ignore[index]
        else:
            assert s.inquiry is None
        meta = s.meta()
        json.dumps(meta)
        assert not any(isinstance(v, np.ndarray) for v in meta.values()) and meta["schema"] == sc.SCENE_VERSION


def test_files_are_small_16_bit_and_round_trip(tmp_path: Path) -> None:
    paths, _ = sc.generate_scenes(None, 5, tmp_path, SyntheticCatalog(), fake_tts)
    total = sum(p.stat().st_size for p in paths)
    assert all(p.stat().st_size < 900_000 for p in paths) and total < 5_000_000  # "a few MB in total"
    sf = pytest.importorskip("soundfile", reason="needs the optional audio extra (uv sync --extra audio)")

    for p in paths:
        info = sf.info(str(p))
        assert (info.samplerate, info.channels, info.subtype) == (SR, 1, "PCM_16")
        x, _ = rp.load_wav(p)
        ref = sc.build_scene(p.stem.rsplit("-s", 1)[0], SyntheticCatalog(), seed=5, tts=fake_tts).audio
        assert np.max(np.abs(x - ref)) <= 1.0 / 32767 + 1e-6  # only the 16-bit quantisation differs
        side = sc.read_sidecar(p)
        assert side is not None and side["wav_sha256"] == sha(p) and side["wav_bytes"] == p.stat().st_size and side["sample_rate"] == SR


def test_missing_data_skips_the_scene_with_a_message_and_builds_the_rest(tmp_path: Path) -> None:
    cat = SyntheticCatalog(missing=["scream"])
    written, skipped = sc.generate_scenes(None, 1, tmp_path, cat, fake_tts)
    assert set(skipped) == {"fall_help_call"} and "scream" in skipped["fall_help_call"] and "homeease data download --dataset vivae" in skipped["fall_help_call"]
    assert len(written) == len(sc.SCENE_NAMES) - 1
    written, skipped = sc.generate_scenes(None, 1, tmp_path / "none", SyntheticCatalog(missing=["background"]), fake_tts)
    assert written == [] and set(skipped) == set(sc.SCENE_NAMES) and all("esc50" in m and "background" in m for m in skipped.values())


def test_missing_tts_skips_only_the_scenes_with_a_spoken_reply(tmp_path: Path) -> None:
    def no_engine(text: str, voice: str, lang: str) -> np.ndarray:
        raise sc.SceneUnavailable(f"needs a text-to-speech voice ({voice}): engine not installed")

    written, skipped = sc.generate_scenes(None, 1, tmp_path, SyntheticCatalog(), no_engine)
    assert set(skipped) == {n for n, s in sc.SCENES.items() if s.reply} and all("text-to-speech" in m for m in skipped.values())
    assert {p.stem.rsplit("-s", 1)[0] for p in written} == {n for n, s in sc.SCENES.items() if not s.reply}
    with pytest.raises(KeyError):
        sc.build_scene("nope", SyntheticCatalog())


def test_default_tts_without_an_engine_is_a_scene_unavailable_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    from homeease.ml.audio import tts as tts_mod

    def boom(*a: Any, **k: Any) -> np.ndarray:
        raise tts_mod.TTSUnavailable("no engine here")

    monkeypatch.setattr(tts_mod, "synthesize", boom)
    with pytest.raises(sc.SceneUnavailable, match="text-to-speech"):
        sc.default_tts("hello", "Zira", "en")


def test_disk_catalog_prefers_the_held_out_fold_and_flags_a_fallback() -> None:
    rows = [{"source": "esc50", "path": f"/x/{i}.wav", "subtype": "washing_machine", "fold": f, "group": f"g{i}"} for i, f in enumerate([1, 2, 5, 5, 3])]
    cands = sc.DiskCatalog(rows=rows).candidates("background")
    assert [c.fold for c in cands] == [5, 5]  # only fold 5: the fold the shipped model was neither trained nor tuned on
    cands = sc.DiskCatalog(rows=[r for r in rows if r["fold"] != 5]).candidates("background")
    assert cands and all(c.fold != 5 for c in cands)  # fallback exists, and the sidecar marks held_out=false (checked next)
    assert sc.DiskCatalog(rows=[]).candidates("glass") == []
    # priority: FSD50K thump first, ESC-50 knock only when there is no thump
    rows2 = [{"source": "esc50", "path": "/x/k.wav", "subtype": "door_wood_knock", "fold": 5}, {"source": "fsd50k", "path": "/x/t.wav", "subtype": "Thump_and_thud", "fold": 5}]
    assert [c.source for c in sc.DiskCatalog(rows=rows2).candidates("thud")] == ["fsd50k"]
    assert [c.source for c in sc.DiskCatalog(rows=rows2[:1]).candidates("thud")] == ["esc50"]


def test_sidecar_records_sources_licences_and_whether_they_were_held_out() -> None:
    s = sc.build_scene("fall_help_call", SyntheticCatalog(), seed=2, tts=fake_tts)
    roles = {x["role"] for x in s.sources}
    assert roles == {"background", "thud", "scream", "reply"}
    assert all(x["held_out"] is True for x in s.sources if x["role"] != "reply") and next(x for x in s.sources if x["role"] == "reply")["voice"] == "Zira"
    assert all(x.get("license") for x in s.sources)


# ------------------------------------------------------------------------------------------------ CLI
@pytest.fixture()
def stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenes from generated clips and a fake TTS voice; sound classifier, ASR and speaker model replaced by stubs."""
    monkeypatch.setattr(sc, "DiskCatalog", lambda *a, **k: SyntheticCatalog())
    monkeypatch.setattr(sc, "default_tts", fake_tts)
    from homeease import cli_audio

    monkeypatch.setattr(cli_audio, "_load_classifier", lambda cfg, ref: (StubSoundClassifier(), None))
    import homeease.ml.audio.asr as asr_mod
    import homeease.ml.audio.speaker as spk_mod

    class FakeASRProcess(ScriptedASR):
        def __init__(self, model: str, temperature: float | None = None) -> None:
            super().__init__("I'm fine, don't worry.")

    monkeypatch.setattr(asr_mod, "ASRProcess", FakeASRProcess)
    monkeypatch.setattr(spk_mod, "WeSpeakerEmbedder", lambda *a, **k: PitchEmbedder())


def test_cli_scenes_list_and_errors(capsys: pytest.CaptureFixture[str], tmp_path: Path, stubbed: None) -> None:
    assert cli.main(["audio", "scenes", "--list"]) == 0
    listing = capsys.readouterr().out
    assert all(n in listing for n in sc.SCENE_NAMES)
    assert cli.main(["audio", "scenes", "--scene", "nope", "--out", str(tmp_path)]) == 3
    assert "unknown scene" in capsys.readouterr().err
    assert cli.main(["audio", "scenes", "--scene", "glass_break", "--seed", "9", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "glass_break-s9.wav" in out and (tmp_path / "glass_break-s9.wav").is_file() and (tmp_path / "glass_break-s9.json").is_file()


def test_cli_scenes_degrade_with_a_clear_message_when_data_is_missing(capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sc, "DiskCatalog", lambda *a, **k: SyntheticCatalog(missing=["background"]))
    assert cli.main(["audio", "scenes", "--out", str(tmp_path)]) == 3  # nothing could be built
    err = capsys.readouterr().err
    assert "skipped" in err and "homeease data download --dataset esc50" in err
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(sc, "DiskCatalog", lambda *a, **k: SyntheticCatalog(missing=["scream"]))
    monkeypatch.setattr(sc, "default_tts", fake_tts)
    assert cli.main(["audio", "scenes", "--out", str(tmp_path)]) == 0  # the others were built
    assert "skipped fall_help_call" in capsys.readouterr().err


def test_cli_replay_input_errors(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert cli.main(["audio", "replay"]) == 3
    assert "exactly one of" in capsys.readouterr().err
    assert cli.main(["audio", "replay", "--wav", str(tmp_path / "missing.wav")]) == 3
    assert "not found" in capsys.readouterr().err
    assert cli.main(["audio", "replay", "--scene", "glass_break", "--scenes-dir", str(tmp_path)]) == 3
    assert "homeease audio scenes --scene glass_break" in capsys.readouterr().err
    assert cli.main(["audio", "replay", "--scene", "nope"]) == 3
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    assert cli.main(["audio", "replay", "--wav", str(bad)]) == 3
    assert cli.main(["audio", "replay", "--wav", str(bad), "--reply-text", "hi"]) == 3


def test_cli_replay_without_a_sound_model_stops_with_the_fix(capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from homeease import cli_audio

    s = sc.build_scene("glass_break", SyntheticCatalog(), seed=1, tts=fake_tts)
    wav = sc.write_scene(s, tmp_path)
    monkeypatch.setattr(cli_audio, "_load_classifier", lambda cfg, ref: (None, "package not found"))
    assert cli.main(["audio", "replay", "--wav", str(wav), "--vad", "energy"]) == 3
    err = capsys.readouterr().err
    assert "sound model unavailable" in err and "train-sound" in err and "--no-sound" in err
    assert cli.main(["audio", "replay", "--wav", str(wav), "--vad", "energy", "--no-sound"]) == 0  # voice branch only, explicitly requested
    out = capsys.readouterr().out
    assert "degraded:model_missing" in out and "sound events : none" in out


def test_cli_replay_scene_prints_labels_and_writes_labels_only(capsys: pytest.CaptureFixture[str], tmp_path: Path, stubbed: None) -> None:
    assert cli.main(["audio", "scenes", "--scene", "tv_and_knock", "--seed", "3", "--out", str(tmp_path)]) == 0
    capsys.readouterr()
    before = sorted(p.name for p in tmp_path.iterdir())
    jo = tmp_path / "labels.json"
    assert cli.main(["audio", "replay", "--scene", "tv_and_knock", "--seed", "3", "--scenes-dir", str(tmp_path), "--vad", "energy", "--json-out", str(jo)]) == 0
    out = capsys.readouterr().out
    assert "sound events (category, confidence and statistics only; raw audio stored: False)" in out and "IMPACT" in out and "ONE scene, not an accuracy result" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(before + ["labels.json"])  # the only file written is the labels file
    d = json.loads(jo.read_text(encoding="utf-8"))
    assert d["raw_audio_stored"] is False and d["scene"] == "tv_and_knock" and d["sound_events"] and d["inquiry"] is None
    assert all(set(e) >= {"category", "confidence", "t_start_utc", "rms_db", "peak_db"} for e in d["sound_events"])


def test_cli_replay_with_a_scripted_inquiry_keeps_the_event_and_prints_the_evidence(capsys: pytest.CaptureFixture[str], tmp_path: Path, stubbed: None) -> None:
    assert cli.main(["audio", "scenes", "--scene", "fall_reassured", "--seed", "3", "--out", str(tmp_path)]) == 0
    capsys.readouterr()
    jo = tmp_path / "labels.json"
    assert cli.main(["audio", "replay", "--scene", "fall_reassured", "--seed", "3", "--scenes-dir", str(tmp_path), "--vad", "energy", "--inquiry", "--enroll-voice", "Zira", "--threshold", "0.6",
                     "--after-seconds", "5", "--json-out", str(jo)]) == 0
    out = capsys.readouterr().out
    assert "scripted suspected fall" in out and "EVIDENCE     : cancel_eligible=True" in out and "evidence only" in out
    iq = json.loads(jo.read_text(encoding="utf-8"))["inquiry"]
    assert iq["record"]["cancel_eligible"] is True and iq["record"]["cancel_policy"] == "evidence_only" and iq["voiceprint_enrolled"] is True and iq["threshold"] == 0.6
    assert "transcript" not in iq["record"] and iq["record"]["transcript_chars"] > 0  # --json-out keeps labels and numbers, not the text of the reply
    assert "transcript=" in out  # the console still shows it
    for k in ("event_before", "after_reply", "after_timers"):
        assert (iq[k]["level"], iq[k]["status"]) == ("EMERGENCY", "OPEN")
    # no enrollment: evidence False with a reason, event unchanged
    assert cli.main(["audio", "replay", "--scene", "fall_reassured", "--seed", "3", "--scenes-dir", str(tmp_path), "--vad", "energy", "--inquiry", "--after-seconds", "5", "--json-out", str(jo)]) == 0
    capsys.readouterr()
    iq = json.loads(jo.read_text(encoding="utf-8"))["inquiry"]
    assert iq["record"]["cancel_eligible"] is False and "no_enrollment" in iq["record"]["cancel_reasons"] and iq["after_timers"]["status"] == "OPEN"


def test_cli_replay_of_a_plain_wav_with_a_scripted_reply(capsys: pytest.CaptureFixture[str], tmp_path: Path, stubbed: None) -> None:
    sf = pytest.importorskip("soundfile", reason="needs the optional audio extra (uv sync --extra audio)")

    from tests.audio_fixtures import scene as synth_scene
    x, _ = synth_scene(20.0, [(5.0, speechlike(0.5, -30.0), "noise")], seed=1)
    wav = tmp_path / "plain.wav"
    sf.write(str(wav), x, SR)
    assert cli.main(["audio", "replay", "--wav", str(wav), "--vad", "energy", "--inquiry"]) == 3  # a plain WAV needs --fall-at
    assert "--fall-at" in capsys.readouterr().err
    assert cli.main(["audio", "replay", "--wav", str(wav), "--vad", "energy", "--reply-text", "I'm fine"]) == 3  # a reply without --inquiry is meaningless
    capsys.readouterr()
    assert cli.main(["audio", "replay", "--wav", str(wav), "--vad", "energy", "--inquiry", "--fall-at", "3", "--reply-text", "I'm fine, don't worry.", "--reply-voice", "Zira",
                     "--enroll-voice", "Zira", "--threshold", "0.6", "--after-seconds", "3"]) == 0
    out = capsys.readouterr().out
    assert "intent=REASSURANCE" in out and "cancel_eligible=True" in out and "EMERGENCY/OPEN" in out


def test_replay_internal_temp_database_is_removed(tmp_path: Path, cfg: Any) -> None:
    before = {p.name for p in Path(tempfile.gettempdir()).glob("homeease_headless_*")}
    s = sc.build_scene("quiet_home", SyntheticCatalog(), seed=1, tts=fake_tts)
    out = rp.run_replay(s.audio, SR, cfg, StubSoundClassifier(), vad=HarmonicStubVAD(), truth=s.truth)
    assert out.sound_status["status"] == "ok"
    assert {p.name for p in Path(tempfile.gettempdir()).glob("homeease_headless_*")} == before


def test_check_truth_reports_found_other_and_missed() -> None:
    ev = [{"category": "IMPACT", "t_start_mono_ms": 1_004_000.0, "t_end_mono_ms": 1_004_500.0, "confidence": 0.9},
          {"category": "DAILY", "t_start_mono_ms": 1_010_000.0, "t_end_mono_ms": 1_012_000.0, "confidence": 0.9},
          {"category": "HELP", "t_start_mono_ms": 1_020_000.0, "t_end_mono_ms": 1_020_500.0, "confidence": 0.8}]
    truth = [{"kind": "thud", "t0_s": 4.0, "t1_s": 4.4, "expect": "IMPACT"}, {"kind": "glass", "t0_s": 10.0, "t1_s": 10.5, "expect": "ABNORMAL"}, {"kind": "scream", "t0_s": 14.0, "t1_s": 14.9, "expect": "HELP"},
             {"kind": "reply", "t0_s": 9.0, "t1_s": 12.0, "expect": None}]
    rows, unexplained = rp.check_truth(truth, ev, 1_000_000.0)
    assert [r["result"] for r in rows] == ["found", "other:DAILY", "no event"]
    assert [u["category"] for u in unexplained] == ["HELP"]  # a HELP label 20 s into the scene that no scripted sound explains


def test_cli_replay_argument_combinations(capsys: pytest.CaptureFixture[str], tmp_path: Path, stubbed: None) -> None:
    assert cli.main(["audio", "scenes", "--scene", "fall_reassured", "--seed", "3", "--out", str(tmp_path)]) == 0
    capsys.readouterr()
    base = ["audio", "replay", "--scene", "fall_reassured", "--seed", "3", "--scenes-dir", str(tmp_path), "--vad", "energy"]
    assert cli.main(base + ["--enroll-voice", "Zira"]) == 3  # an enrollment without an inquiry is meaningless
    assert "only make sense together with --inquiry" in capsys.readouterr().err
    assert cli.main(base + ["--inquiry", "--inquiry-at", "2"]) == 3  # the listening window cannot open before the fall (scene fall at 4 s)
    assert "invalid inquiry script" in capsys.readouterr().err
    assert cli.main(base + ["--inquiry", "--reply-text", "hello"]) == 3  # the scene already has a reply
    assert "already contains a spoken reply" in capsys.readouterr().err
    jo = tmp_path / "l.json"
    assert cli.main(base + ["--inquiry", "--fall-at", "5", "--after-seconds", "3", "--json-out", str(jo), "--event-level", "pending"]) == 0
    capsys.readouterr()
    iq = json.loads(jo.read_text(encoding="utf-8"))["inquiry"]
    assert iq["event_before"]["level"] == "PENDING" and iq["record"]["status"] in ("RESPONDED", "NO_RESPONSE")  # the script moved with --fall-at (window opens at 9 s)


def test_a_transcript_the_console_cannot_encode_does_not_crash_the_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review (L3 F6): under a Windows code page (GBK / cp1252) a redirected stdout raised UnicodeEncodeError on a music note after the whole replay."""
    import io

    from homeease import cli_audio

    raw = io.BytesIO()
    fake = io.TextIOWrapper(raw, encoding="gbk", errors="strict")
    monkeypatch.setattr(sys, "stdout", fake)
    with pytest.raises(UnicodeEncodeError):
        print("♪", file=fake, flush=True)  # the default behaviour that crashed
    cli_audio._safe_console()
    print("transcript='♪ \u6ca1\u4e8b'", flush=True)
    fake.flush()
    text = raw.getvalue().decode("gbk")
    assert "\\u266a" in text and "\u6ca1\u4e8b" in text  # the note is escaped, the Chinese reply stays readable in the console's own encoding


def test_json_out_drops_the_reply_text_and_keeps_its_length() -> None:
    from homeease.cli_audio import _labels_only

    p = {"a": 1, "inquiry": {"record": {"intent": "HELP", "transcript": "help me please"}, "policy": "x"}}
    q = _labels_only(p)
    assert q["inquiry"]["record"] == {"intent": "HELP", "transcript_chars": 14} and p["inquiry"]["record"]["transcript"] == "help me please"  # the input is not modified
    assert _labels_only({"inquiry": None}) == {"inquiry": None}
