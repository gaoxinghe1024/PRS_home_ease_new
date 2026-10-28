"""TTS abstraction: backend selection, injection safety, voice resolution, playback, inquiry wiring (no real speech except the slow test)."""

from __future__ import annotations

import base64
import json
import os
import platform
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from homeease.core.config import AudioCfg
from homeease.ml.audio import tts

pytest.importorskip("soundfile", reason="needs the optional audio extra (uv sync --extra audio)")

HOSTILE = [
    "plain text",
    'He said "help"; rm -rf / ; echo "pwned"',
    "it's a `backtick` and $(Get-Process) and ${env:USERNAME} and $env:PATH",
    "'; Remove-Item -Recurse C:\\ ; '",
    '"; calc.exe; "',
    "line one\nline two\r\nline three",
    "<speak><break time='5s'/>ssml?</speak>",
    "-v Alex -o /tmp/x.aiff",
    "\u6551\u547d\u554a\uff1b$(whoami)`id`",
    "%PATH% & dir | more > nul",
]


def _wav_bytes(seconds: float = 0.6, sr: int = 16000) -> bytes:
    import io
    import wave

    t = np.arange(int(seconds * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())
    return buf.getvalue()


class FakePowerShell:
    """Stands in for subprocess.run when the target is powershell.exe; records everything it was given."""

    def __init__(self, voices: list[dict[str, str]] | None = None, fail: bool = False) -> None:
        self.voices = voices if voices is not None else [
            {"name": "Microsoft Zira Desktop", "culture": "en-US", "gender": "Female"},
            {"name": "Microsoft David Desktop", "culture": "en-US", "gender": "Male"},
            {"name": "Microsoft Huihui Desktop", "culture": "zh-CN", "gender": "Female"},
        ]
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def __call__(self, args: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        assert isinstance(args, list) and kw.get("shell", False) is False
        script = base64.b64decode(args[args.index("-EncodedCommand") + 1]).decode("utf-16-le")
        rec = {"args": list(args), "env": dict(kw.get("env") or {}), "script": script, "jobs": None}
        self.calls.append(rec)
        if self.fail:
            return subprocess.CompletedProcess(args, 1, b"", b"boom: something failed\nsecond line")
        if script is tts._SAPI_LIST or script == tts._SAPI_LIST:
            return subprocess.CompletedProcess(args, 0, json.dumps(self.voices).encode("utf-8"), b"")
        jobs = json.loads(Path(rec["env"]["HOMEEASE_TTS_JOBS"]).read_text(encoding="utf-8"))
        rec["jobs"] = jobs
        for j in jobs:
            Path(j["out"]).write_bytes(_wav_bytes())
        return subprocess.CompletedProcess(args, 0, b"", b"")


@pytest.fixture(autouse=True)
def _fresh() -> Any:
    tts.clear_caches()
    yield
    tts.clear_caches()


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> FakePowerShell:
    fake = FakePowerShell()
    monkeypatch.setattr(tts.platform, "system", lambda: "Windows")
    monkeypatch.setattr(tts, "_powershell", lambda: r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
    monkeypatch.setattr(tts.subprocess, "run", fake)
    return fake


# ---------------------------------------------------------------- selection
def test_backend_selection_by_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts, "_powershell", lambda: "powershell.exe")
    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/say" if name == "say" else None)
    for system, expected in (("Windows", "sapi"), ("Darwin", "say"), ("Linux", "none")):
        monkeypatch.setattr(tts.platform, "system", lambda s=system: s)
        for alias in ("auto", "system", "system_say"):
            assert tts.get_backend(alias).name == expected, (system, alias)
    monkeypatch.setattr(tts.platform, "system", lambda: "Windows")
    monkeypatch.setattr(tts, "_powershell", lambda: None)
    assert tts.get_backend("system").name == "none"  # no PowerShell -> degraded, never a crash
    monkeypatch.setattr(tts.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(tts.shutil, "which", lambda name: None)
    assert tts.get_backend("system").name == "none"  # no say binary


def test_explicit_backends_and_unknown_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts.platform, "system", lambda: "Linux")
    assert tts.get_backend("sapi").name == "sapi" and not tts.get_backend("sapi").available()
    assert tts.get_backend("say").name == "say" and not tts.get_backend("say").available()
    assert tts.get_backend("none").name == "none" and tts.available_voices("none") == []
    with pytest.raises(ValueError):
        tts.get_backend("espeak; rm -rf /")
    with pytest.raises(tts.TTSUnavailable):
        tts.synthesize("hello", "Zira", "en", backend="none")
    assert tts.play("hello", "en", backend="none") is False


def test_norm_lang() -> None:
    assert tts.norm_lang("en-US") == "en" and tts.norm_lang("zh_CN") == "zh" and tts.norm_lang("ZH") == "zh" and tts.norm_lang(None) == "en"


# ---------------------------------------------------------------- SAPI backend with a fake PowerShell
def test_sapi_voice_listing_and_resolution(windows: FakePowerShell) -> None:
    voices = tts.available_voices()
    assert [(v.short, v.lang) for v in voices] == [("Zira", "en"), ("David", "en"), ("Huihui", "zh")]
    assert tts.available_voices() == voices and len(windows.calls) == 1  # cached
    assert tts.resolve_voice("Zira", "en").name == "Microsoft Zira Desktop"
    assert tts.resolve_voice("microsoft david desktop", "en").short == "David"
    assert tts.resolve_voice(None, "zh").name == "Microsoft Huihui Desktop"
    assert tts.resolve_voice(None, "en-GB").short == "Zira"  # preferred voice first
    with pytest.raises(tts.TTSError):
        tts.resolve_voice(None, "fr")
    with pytest.raises(tts.TTSError):
        tts.resolve_voice("Mark", "en")


def test_sapi_single_voice_json_object_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePowerShell(voices={"name": "Microsoft Zira Desktop", "culture": "en-US", "gender": "Female"})  # type: ignore[arg-type]
    monkeypatch.setattr(tts.platform, "system", lambda: "Windows")
    monkeypatch.setattr(tts, "_powershell", lambda: "powershell.exe")
    monkeypatch.setattr(tts.subprocess, "run", fake)
    assert [v.short for v in tts.available_voices("sapi")] == ["Zira"]  # PowerShell prints one voice as an object, not a list


def test_sapi_synthesize_returns_16k_float32(windows: FakePowerShell) -> None:
    x = tts.synthesize("Help, I fell", "Zira", "en")
    assert x.dtype == np.float32 and x.ndim == 1
    assert abs(len(x) / 16000 - 0.6) < 0.02 and float(np.sqrt(np.mean(x**2))) > 0.05


@pytest.mark.parametrize("text", HOSTILE)
def test_hostile_text_never_reaches_a_command_string(windows: FakePowerShell, text: str) -> None:
    tts.synthesize(text, "Zira", "en")
    synth = [c for c in windows.calls if c["jobs"] is not None]
    assert len(synth) == 1
    call = synth[0]
    expected = tts.clean_text(text)
    assert [j["text"] for j in call["jobs"]] == [expected]  # data arrives intact through the JSON file only
    for a in call["args"]:
        assert expected not in a and text not in a  # not in argv
    assert call["script"] == tts._SAPI_SYNTH  # the script is the constant, nothing was interpolated into it
    added = {k: v for k, v in call["env"].items() if os.environ.get(k) != v}  # what the module put into the child environment
    assert set(added) == {"HOMEEASE_TTS_JOBS"} and not any(expected in str(v) for v in added.values())  # a file path only, never the text
    assert call["args"][1:4] == ["-NoProfile", "-NonInteractive", "-EncodedCommand"]


@pytest.mark.parametrize("voice", ["Zira'; calc; '", "$(Get-Process)", 'Zira" ; Start-Process x', "../../etc/passwd", "`whoami`", "Zira\nDavid"])
def test_hostile_voice_name_is_rejected_before_any_synthesis_call(windows: FakePowerShell, voice: str) -> None:
    with pytest.raises(tts.TTSError):
        tts.synthesize("hello", voice, "en")
    assert all(c["jobs"] is None for c in windows.calls)  # only the (constant) listing script may have run
    assert all(voice not in " ".join(c["args"]) for c in windows.calls)
    assert tts.play("hello", "en", voice=voice) is False


def test_hostile_text_in_batch_keeps_order_and_single_engine_call(windows: FakePowerShell) -> None:
    items = [(t, "Zira" if i % 2 == 0 else "David", "en") for i, t in enumerate(HOSTILE)]
    out = tts.synthesize_many(items)
    assert len(out) == len(HOSTILE)
    synth = [c for c in windows.calls if c["jobs"] is not None]
    assert len(synth) == 1 and [j["text"] for j in synth[0]["jobs"]] == [tts.clean_text(t) for t in HOSTILE]
    assert [j["voice"] for j in synth[0]["jobs"]] == ["Microsoft Zira Desktop" if i % 2 == 0 else "Microsoft David Desktop" for i in range(len(HOSTILE))]


def test_text_validation() -> None:
    assert tts.clean_text("  a \n\t b\x00c  ") == "a b c"
    for bad in ("", "   ", "\x00\x00", "x" * (tts.MAX_TEXT_CHARS + 1)):
        with pytest.raises(tts.TTSError):
            tts.clean_text(bad)
    with pytest.raises(tts.TTSError):
        tts.clean_text(123)  # type: ignore[arg-type]


def test_sapi_failures_are_errors_not_silence(windows: FakePowerShell) -> None:
    tts.available_voices()  # populate the cache with the working fake
    windows.fail = True
    with pytest.raises(tts.TTSError, match="boom"):
        tts.synthesize("hello", "Zira", "en")
    assert tts.play("hello", "en") is False  # inquiry: a failed prompt is reported, never raised


def test_sapi_missing_output_is_an_error(windows: FakePowerShell, monkeypatch: pytest.MonkeyPatch) -> None:
    tts.available_voices()

    def no_files(args: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(tts.subprocess, "run", no_files)
    with pytest.raises(tts.TTSError, match="no audio"):
        tts.synthesize("hello", "Zira", "en")


def test_temp_dir_is_removed_after_synthesis(windows: FakePowerShell) -> None:
    tts.synthesize("hello", "Zira", "en")
    call = [c for c in windows.calls if c["jobs"] is not None][0]
    assert not Path(call["jobs"][0]["out"]).exists() and not Path(call["env"]["HOMEEASE_TTS_JOBS"]).exists()  # prompt audio does not linger


def test_sapi_play_uses_winsound_and_blocks(windows: FakePowerShell, monkeypatch: pytest.MonkeyPatch) -> None:
    played: list[Any] = []
    fake = types.SimpleNamespace(SND_MEMORY=4, PlaySound=lambda data, flags: played.append((len(data), flags)))
    monkeypatch.setitem(sys.modules, "winsound", fake)
    assert tts.play("Do you need help?", "en") is True
    assert len(played) == 1 and played[0][0] > 100 and played[0][1] == 4  # synchronous (no SND_ASYNC): echo avoidance


# ---------------------------------------------------------------- say backend with a fake subprocess
SAY_LIST = (b"Alex                en_US    # Most people recognize me by my voice.\n"
            b"Samantha            en_US    # Hello, my name is Samantha.\n"
            b"Tingting            zh_CN    # \xe4\xbd\xa0\xe5\xa5\xbd\n"
            b"Grandpa (\xe4\xb8\xad\xe6\x96\x87\xef\xbc\x88\xe4\xb8\xad\xe5\x9b\xbd\xe5\xa4\xa7\xe9\x99\x86\xef\xbc\x89)  zh_CN    # \xe4\xbd\xa0\xe5\xa5\xbd\n")


def _fake_say(calls: list[dict[str, Any]]) -> Any:
    def run(args: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        assert kw.get("shell", False) is False
        calls.append({"args": list(args), "input": kw.get("input")})
        if args[:3] == ["say", "-v", "?"]:
            return subprocess.CompletedProcess(args, 0, SAY_LIST, b"")
        if "-o" in args:
            Path(args[args.index("-o") + 1]).write_bytes(_wav_bytes())  # RIFF/WAV stands in for the AIFF (soundfile reads both)
        return subprocess.CompletedProcess(args, 0, b"", b"")

    return run


@pytest.fixture
def mac(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(tts.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(tts.shutil, "which", lambda n: "/usr/bin/say" if n == "say" else None)
    monkeypatch.setattr(tts.subprocess, "run", _fake_say(calls))
    return calls


def test_say_voices_and_text_goes_through_stdin(mac: list[dict[str, Any]]) -> None:
    voices = tts.available_voices()
    assert {(v.name, v.lang) for v in voices} >= {("Samantha", "en"), ("Tingting", "zh")}
    assert any(v.name.startswith("Grandpa") and v.lang == "zh" for v in voices)
    hostile = "-v Alex -o /tmp/pwn.aiff; $(whoami)"
    x = tts.synthesize(hostile, "Samantha", "en")
    assert x.dtype == np.float32 and len(x) > 1000
    call = [c for c in mac if "-o" in c["args"]][0]
    assert call["args"][:3] == ["say", "-v", "Samantha"] and call["args"][-2:] == ["-f", "-"]
    assert hostile.encode() == call["input"] and hostile not in " ".join(call["args"])


def test_say_play_and_unknown_voice(mac: list[dict[str, Any]]) -> None:
    assert tts.play("\u60a8\u9700\u8981\u5e2e\u52a9\u5417\uff1f\u8bf7\u56de\u7b54\u3002", "zh") is True
    spoke = [c for c in mac if c["input"] and c["args"][:2] == ["say", "-v"] and "-o" not in c["args"]][-1]
    assert spoke["args"][2] == "Tingting" and spoke["input"].decode("utf-8") == "\u60a8\u9700\u8981\u5e2e\u52a9\u5417\uff1f\u8bf7\u56de\u7b54\u3002"
    assert tts.play("hello", "en", voice="Zira") is False  # not a macOS voice


# ---------------------------------------------------------------- emulated second speaker
def test_emulate_voice_changes_pitch_and_speed_together() -> None:
    sr = 16000
    x = (0.3 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
    lower = tts.emulate_voice(x, 0.8)
    assert abs(len(lower) / len(x) - 1 / 0.8) < 0.01  # slower
    f = np.fft.rfftfreq(len(lower), 1 / sr)[np.argmax(np.abs(np.fft.rfft(lower)))]
    assert abs(f - 160) < 5  # lower pitch
    assert np.array_equal(tts.emulate_voice(x, 1.0), x) and tts.emulate_voice(x, 1.0).dtype == np.float32
    for bad in (0.2, 3.0):
        with pytest.raises(ValueError):
            tts.emulate_voice(x, bad)


# ---------------------------------------------------------------- config + inquiry wiring
def test_audio_cfg_accepts_system_and_the_historic_alias() -> None:
    assert AudioCfg(tts="system").tts == "system" and AudioCfg(tts="system_say").tts == "system_say"
    assert AudioCfg().tts == "browser"
    with pytest.raises(ValueError):
        AudioCfg(tts="espeak")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AudioCfg(auto_cancel_emergency=True)  # still impossible


def _service(mode: str, tts_cb: Any = None) -> Any:
    from homeease.ml.audio.inquiry import AudioInquiryService

    svc = AudioInquiryService.__new__(AudioInquiryService)
    published: list[Any] = []
    svc.cfg = types.SimpleNamespace(audio=AudioCfg(tts=mode))  # type: ignore[arg-type]
    svc.tts = tts_cb
    svc.rt = types.SimpleNamespace(subject_id="s1", broadcaster=types.SimpleNamespace(publish=lambda *a, **k: published.append((a, k))))
    svc._published = published
    return svc


@pytest.mark.parametrize("mode", ["system", "system_say"])
def test_inquiry_speak_uses_the_tts_abstraction(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(tts, "play", lambda text, lang="en", voice=None, backend="auto": calls.append((text, lang)) or True)
    svc = _service(mode)
    assert svc._speak("Do you need help? Please say yes or no.", "en") is True
    assert calls == [("Do you need help? Please say yes or no.", "en")]
    monkeypatch.setattr(tts, "play", lambda *a, **k: False)
    assert svc._speak("x", "zh") is False  # failure is reported (prompt_played false), not raised


def test_inquiry_speak_browser_none_and_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("homeease.ml.audio.inquiry.time.sleep", lambda s: None)
    monkeypatch.setattr(tts, "play", lambda *a, **k: pytest.fail("browser mode must not use the local speaker"))
    b = _service("browser")
    assert b._speak("hi", "en") is True and b._published and b._published[0][0][0] == "voice_prompt"
    assert _service("none")._speak("hi", "en") is False
    assert _service("system", tts_cb=lambda t, l: True)._speak("hi", "en") is True  # an injected callable always wins


# ---------------------------------------------------------------- real engine (Windows only)
@pytest.mark.slow
@pytest.mark.skipif(platform.system() != "Windows", reason="Windows SAPI voices only")
def test_real_sapi_voices_are_non_silent_and_text_is_not_executed(tmp_path: Path) -> None:
    be = tts.get_backend("sapi")
    if not be.available() or not tts.available_voices("sapi"):
        pytest.skip("no SAPI voices installed")
    en = tts.synthesize("Do you need help?", "Zira", "en", backend="sapi")
    assert en.dtype == np.float32 and 0.4 < len(en) / 16000 < 6.0
    assert float(np.sqrt(np.mean(en**2))) > 0.005 and float(np.abs(en).max()) > 0.05  # speech, not silence
    victim = tmp_path / "pwned.txt"
    nasty = [f"hello'; New-Item -ItemType File -Path '{victim}' ; '", f'"; New-Item -ItemType File -Path "{victim}"; "', f"$(New-Item -ItemType File -Path '{victim}')", f"`New-Item -Path {victim}`"]
    outs = tts.synthesize_many([(t, "Zira", "en") for t in nasty], backend="sapi")
    assert all(len(o) > 1600 for o in outs)  # read aloud as plain text
    assert not victim.exists()  # and never executed
    voices = {v.short for v in tts.available_voices("sapi")}
    if "Huihui" in voices:
        zh = tts.synthesize("\u6211\u6ca1\u4e8b", "Huihui", "zh", backend="sapi")
        assert float(np.sqrt(np.mean(zh**2))) > 0.005
