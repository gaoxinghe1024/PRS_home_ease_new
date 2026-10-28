"""A3b protocol (phrase bank + TTS + ASR grid): voice plan, selection rule, scoring, order of operations, report and
status paths - with stub TTS / ASR / pools, so no speech engine, model, network or dataset is touched."""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from homeease.experiments import audio_intent as AI
from homeease.ml.audio import tts as T
from homeease.ml.audio.phrases import load_phrase_bank

SR = 16000


@pytest.fixture(scope="module")
def bank():
    return load_phrase_bank()


def _voices(*pairs: tuple[str, str], backend: str = "sapi") -> list[T.Voice]:
    return [T.Voice(n, lg, backend) for n, lg in pairs]


# ---------------------------------------------------------------- voice plan
def test_plan_voices_windows_with_three_voices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "get_backend", lambda name="auto": T.SapiBackend())
    plan = AI.plan_voices("sapi", _voices(("Microsoft Zira Desktop", "en"), ("Microsoft David Desktop", "en"), ("Microsoft Huihui Desktop", "zh")))
    assert plan[("self", "en")].label == "Zira" and plan[("self", "zh")].label == "Huihui"
    assert plan[("visitor", "en")].label == "David" and not plan[("visitor", "en")].emulated
    assert plan[("tv", "en")].label == "David"
    # Mandarin has one installed voice: visitor and tv are EMULATED and say so in the label
    for role in ("visitor", "tv"):
        vc = plan[(role, "zh")]
        assert vc.emulated and vc.ratio == AI.EMULATION_RATIO[role] and "emulated" in vc.label and vc.name == plan[("self", "zh")].name
    assert plan[("visitor", "zh")].ratio != plan[("tv", "zh")].ratio  # two different shifts, not the same fake voice twice


def test_plan_voices_without_a_second_english_voice_emulates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "get_backend", lambda name="auto": T.SapiBackend())
    plan = AI.plan_voices("sapi", _voices(("Microsoft Zira Desktop", "en"), ("Microsoft Huihui Desktop", "zh")))
    assert plan[("visitor", "en")].emulated and "Zira" in plan[("visitor", "en")].label


def test_plan_voices_macos_uses_real_voices_for_every_role(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "get_backend", lambda name="auto": T.SayBackend())
    v = _voices(("Samantha", "en"), ("Daniel", "en"), ("Fred", "en"), ("Tingting", "zh"), ("Grandpa (\u4e2d\u6587\uff08\u4e2d\u56fd\u5927\u9646\uff09)", "zh"), backend="say")
    plan = AI.plan_voices("say", v)
    assert not any(vc.emulated for vc in plan.values())
    assert plan[("self", "en")].name == "Samantha" and plan[("visitor", "en")].name == "Daniel" and plan[("tv", "en")].name == "Fred"
    assert plan[("self", "zh")].name == "Tingting" and plan[("visitor", "zh")].name.startswith("Grandpa")


def test_plan_voices_needs_a_voice_for_each_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "get_backend", lambda name="auto": T.SapiBackend())
    with pytest.raises(T.TTSError):
        AI.plan_voices("sapi", _voices(("Microsoft Zira Desktop", "en")))


# ---------------------------------------------------------------- decoding options
def test_decode_args_per_language_mode_and_bias() -> None:
    lang, kw = AI.decode_args(AI.ASRCfg("faster-whisper-tiny", "none", "forced"), "zh")
    assert lang == "zh" and kw == {}
    lang, kw = AI.decode_args(AI.ASRCfg("faster-whisper-tiny", "none", "auto"), "zh")
    assert lang is None and kw == {}
    lang, kw = AI.decode_args(AI.ASRCfg("faster-whisper-tiny", "hotwords", "auto_en_zh"), "en")
    assert lang is None and kw["restrict_languages"] == ("en", "zh") and "\u6551\u547d" in kw["hotwords"] and "help" in kw["hotwords"] and "initial_prompt" not in kw
    lang, kw = AI.decode_args(AI.ASRCfg("faster-whisper-base", "prompt", "forced"), "zh")
    assert lang == "zh" and kw["initial_prompt"] == AI.PROMPTS["zh"] and "hotwords" not in kw
    _, kw = AI.decode_args(AI.ASRCfg("faster-whisper-base", "prompt", "auto"), "en")
    assert kw["initial_prompt"] == AI.PROMPTS["auto"]  # unknown language -> bilingual vocabulary


def test_full_grid_is_the_requested_comparison() -> None:
    g = AI.full_grid()
    assert len(g) == len({c.key for c in g}) == 2 * 3 * 3
    assert {c.model for c in g} == {"faster-whisper-tiny", "faster-whisper-base"}  # no model above base (disk budget)
    assert {c.bias for c in g} == {"none", "hotwords", "prompt"} and {c.lang_mode for c in g} == {"forced", "auto", "auto_en_zh"}


# ---------------------------------------------------------------- selection (declared before the test split exists)
def _cell(f1: float, missed: int, lat: float | None, key: str) -> dict[str, Any]:
    return {"cfg": AI.cfg_from_key(key), "scores": {"rule_v2": {"macro_f1": f1, "safety": {"self_help_missed": missed}, "latency_s": {"median": lat}}}}


def test_selection_ignores_oracle_language_and_slow_cells() -> None:
    dev = {
        "tiny|none|forced": _cell(0.99, 0, 0.4, "tiny|none|forced"),  # best score but tells the ASR the language
        "base|prompt|auto": _cell(0.97, 0, 4.0, "base|prompt|auto"),  # best deployable score but over the latency budget
        "tiny|hotwords|auto_en_zh": _cell(0.90, 5, 0.5, "tiny|hotwords|auto_en_zh"),
        "tiny|none|auto": _cell(0.80, 9, 0.5, "tiny|none|auto"),
    }
    pick, rows = AI.select_config(dev)
    assert pick == ("tiny|hotwords|auto_en_zh", "rule_v2")
    assert {r["cfg"]: r["eligible"] for r in rows} == {"tiny|none|forced": False, "base|prompt|auto": False, "tiny|hotwords|auto_en_zh": True, "tiny|none|auto": True}
    assert [r for r in rows if r["cfg"] == "tiny|none|forced"][0]["oracle_language"]


def test_selection_ties_go_to_fewer_missed_calls_for_help_then_latency() -> None:
    dev = {
        "tiny|none|auto": _cell(0.900, 6, 0.3, "tiny|none|auto"),
        "base|hotwords|auto": _cell(0.897, 2, 1.1, "base|hotwords|auto"),  # within 0.005 of the best, fewer missed
        "base|prompt|auto": _cell(0.896, 2, 0.9, "base|prompt|auto"),  # same misses, lower latency -> wins
        "tiny|prompt|auto": _cell(0.850, 0, 0.3, "tiny|prompt|auto"),  # clearly worse macro-F1: not a tie
    }
    assert AI.select_config(dev)[0] == ("base|prompt|auto", "rule_v2")


def test_selection_without_an_eligible_cell_selects_nothing() -> None:
    assert AI.select_config({"tiny|none|forced": _cell(0.9, 0, 0.2, "tiny|none|forced"), "base|none|auto": _cell(0.9, 0, None, "base|none|auto")})[0] is None


def test_neighbours_change_exactly_one_factor() -> None:
    base = AI.cfg_from_key("tiny|hotwords|auto_en_zh")
    ns = AI.neighbours(base.key)
    assert len(ns) == 1 + 2 + 2 and len({n.key for n in ns}) == len(ns)
    for n in ns:
        diff = sum(a != b for a, b in zip((n.model, n.bias, n.lang_mode), (base.model, base.bias, base.lang_mode)))
        assert diff == 1


# ---------------------------------------------------------------- scoring
def _clip(bucket: str, role: str, expected: str, group: str, probe: bool = False, lang: str = "en") -> dict[str, Any]:
    return {"bucket": bucket, "role": role, "expected": expected, "group": group, "probe": probe, "lang": lang, "family": group, "split": "dev", "text": "x"}


def test_score_safety_counts_confusion_and_buckets() -> None:
    clips = [
        _clip("self_reply", "self", "HELP", "a"), _clip("self_reply", "self", "HELP", "b"), _clip("self_reply", "self", "HELP", "c"),
        _clip("self_reply", "self", "REASSURANCE", "d"), _clip("self_reply", "self", "OTHER", "e"),
        _clip("visitor_answer", "visitor", "REASSURANCE", "f"), _clip("visitor_answer", "visitor", "HELP", "g"),
        _clip("tv_speech", "tv", "OTHER", "h"), _clip("tv_speech", "tv", "REASSURANCE", "i"), _clip("tv_says_help", "tv", "HELP", "j"),
        _clip("silence", "none", "NO_RESPONSE", "k"), _clip("noise_only", "none", "NO_RESPONSE", "l", probe=True),
    ]
    preds = ["HELP", "OTHER", "REASSURANCE", "REASSURANCE", "HELP", "REASSURANCE", "HELP", "REASSURANCE", "REASSURANCE", "HELP", "NO_RESPONSE", "HELP"]
    s = AI.score(clips, preds)
    sf = s["safety"]
    assert s["n"] == 11  # the probe clip is not scored
    assert (sf["self_help_n"], sf["self_help_missed"], sf["self_help_as_reassurance"]) == (3, 2, 1)
    assert sf["nonself_n"] == 5 and sf["nonself_reassurance_accepted"] == 3  # visitor 'fine', tv 'other' read as fine, tv 'fine'
    assert sf["nonself_reassurance_but_words_not_reassuring"] == 1 and sf["nonself_words_reassuring_n"] == 2
    assert sf["self_false_reassurance"] == 1 and sf["self_not_reassurance_n"] == 4
    assert sf["false_help_n"] == 5 and sf["false_help"] == 1  # 3 REASSURANCE + 2 OTHER clips; only the self OTHER one is read as HELP; HELP-worded clips and silence are not counted
    assert sf["noresp_missed_as_speech"] == 0
    assert sf["self_help_missed_ci"][0] < 2 / 3 < sf["self_help_missed_ci"][1]
    cm = np.array(s["report"]["confusion"])
    labels = s["report"]["labels"]
    assert cm[labels.index("HELP"), labels.index("HELP")] == 3 and cm.sum() == 11
    assert s["buckets"]["self_reply"] == {"n": 5, "correct": 2, "accuracy": 0.4, "ci": s["buckets"]["self_reply"]["ci"]}
    assert s["buckets"]["silence"]["accuracy"] == 1.0 and "noise_only" not in s["buckets"]
    assert 0.0 <= s["macro_f1"] <= 1.0 and "macro_f1_ci" not in s


def test_score_silence_read_as_speech_is_counted() -> None:
    clips = [_clip("silence", "none", "NO_RESPONSE", f"s{i}") for i in range(3)] + [_clip("self_reply", "self", "OTHER", "o")]
    s = AI.score(clips, ["OTHER", "NO_RESPONSE", "NO_RESPONSE", "OTHER"])
    assert s["safety"]["noresp_missed_as_speech"] == 1


def test_macro_f1_ignores_classes_without_support_but_punishes_missing_predictions() -> None:
    y = np.array(["HELP", "HELP", "OTHER", "OTHER"])
    assert AI._macro3(y, np.array(["HELP", "HELP", "OTHER", "OTHER"])) == 1.0
    assert AI._macro3(y, np.array(["OTHER"] * 4)) == pytest.approx((0.0 + 2 * 2 / (2 * 2 + 2)) / 2)


def test_uncertain_and_no_response_are_never_safe_in_the_scoring() -> None:
    clips = [_clip("self_reply", "self", "HELP", f"h{i}") for i in range(4)]
    s = AI.score(clips, ["UNCERTAIN", "NO_RESPONSE", "OTHER", "HELP"])
    assert s["safety"]["self_help_missed"] == 3 and s["safety"]["self_help_as_reassurance"] == 0


# ---------------------------------------------------------------- clip building (stub synthesis, in memory only)
class StubSynth:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str, str]]] = []

    def __call__(self, jobs: list[tuple[str, str, str]]) -> list[np.ndarray]:
        self.calls.append(list(jobs))
        t = np.arange(SR) / SR
        return [(0.3 * np.sin(2 * np.pi * (200 + 7 * (len(text) % 11)) * t)).astype(np.float32) for text, _v, _l in jobs]


def _plan(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], AI.VoiceChoice]:
    monkeypatch.setattr(T, "get_backend", lambda name="auto": T.SapiBackend())
    return AI.plan_voices("sapi", _voices(("Microsoft Zira Desktop", "en"), ("Microsoft David Desktop", "en"), ("Microsoft Huihui Desktop", "zh")))


def _load(path: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(path.encode()))
    return (0.05 * rng.standard_normal(5 * SR)).astype(np.float32)


def test_build_clips_covers_every_phrase_bucket_and_labels_voices(bank, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(monkeypatch)
    synth = StubSynth()
    libri = [{"path": f"l{i}.flac", "speaker": f"{100 + i}", "ref": "some text"} for i in range(10)]
    clips = AI.build_clips(bank, "dev", synth, plan, [f"n{i}.wav" for i in range(30)], libri, 42, _load)
    assert sum(len(j) for j in synth.calls) == len(bank.select(split="dev"))
    assert max(len(j) for j in synth.calls) <= 100  # batches of at most 100 phrases per engine call
    assert {c["split"] for c in clips} == {"dev"}
    by = {b: [c for c in clips if c["bucket"] == b] for b in {c["bucket"] for c in clips}}
    assert {"self_reply", "self_noise_5dB", "visitor_answer", "tv_speech", "tv_says_help", "silence", "noise_only"} <= set(by)
    assert len(by["tv_speech"]) == len([e for e in bank.select(split="dev", role="tv") if e.intent != "HELP"]) + 8  # + LibriSpeech
    assert all(c["expected"] == "NO_RESPONSE" for c in by["silence"] + by["noise_only"]) and all(c["probe"] for c in by["noise_only"])
    assert not any(c["probe"] for c in clips if c["bucket"] != "noise_only")
    emu = [c for c in clips if c["emulated"]]
    assert emu and all("emulated" in c["voice"] for c in emu) and {c["role"] for c in emu} == {"visitor", "tv"} and {c["lang"] for c in emu} == {"zh"}
    assert all(c["voice"] in ("Zira", "Huihui") for c in by["self_reply"])
    # noise variants only for the monitored person, ~ every third phrase
    n_self = len(bank.select(split="dev", role="self"))
    assert abs(len(by["self_noise_5dB"]) - n_self / 3) <= 1
    assert all(c["audio"].dtype == np.float32 and c["audio"].ndim == 1 and np.isfinite(c["audio"]).all() for c in clips)
    assert all(isinstance(c["speech"], bool) for c in clips)
    assert sum(c["speech"] for c in by["silence"]) == 0  # the energy VAD calls the silence clips silent
    assert sum(c["speech"] for c in by["self_reply"]) >= 0.9 * len(by["self_reply"])


def test_build_clips_never_mixes_splits_and_persists_no_audio(bank, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = _plan(monkeypatch)
    monkeypatch.chdir(tmp_path)
    synth = StubSynth()
    clips = AI.build_clips(bank, "test", synth, plan, [f"n{i}.wav" for i in range(30)], [], 42, _load)
    sent = {t for j in synth.calls for t, _v, _l in j}
    assert sent == {e.text for e in bank.select(split="test")} and not sent & {e.text for e in bank.select(split="dev")}
    assert {c["split"] for c in clips} == {"test"}
    assert list(tmp_path.rglob("*")) == []  # nothing written anywhere
    # the clip table that goes to disk (predictions.jsonl) is built from these keys: no audio field
    assert "audio" in clips[0]  # in memory only; exp_intent_bank writes explicit label columns, see test_written_run_directory_holds_labels_only


def test_build_clips_is_deterministic_per_seed(bank, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(monkeypatch)
    a = AI.build_clips(bank, "dev", StubSynth(), plan, [f"n{i}.wav" for i in range(30)], [], 7, _load)
    b = AI.build_clips(bank, "dev", StubSynth(), plan, [f"n{i}.wav" for i in range(30)], [], 7, _load)
    assert [c["cid"] for c in a] == [c["cid"] for c in b]
    assert all(np.array_equal(x["audio"], y["audio"]) for x, y in zip(a, b))


# ---------------------------------------------------------------- the protocol: order of operations
class FakeASR:
    """Returns the scripted text of a clip, found by the identity of the audio array it is given."""

    registry: dict[int, str] = {}

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: np.ndarray, language: str | None = None, **kw: Any) -> dict[str, Any]:
        self.calls.append({"language": language, **kw})
        txt = self.registry.get(id(audio), "")
        # base is 'better' than tiny on one scripted case: a call for help the tiny model garbles
        if txt == "help tiny-garbles" and self.model.endswith("tiny"):
            txt = "hello"
        elif txt == "help tiny-garbles":
            txt = "help"
        return {"text": txt, "avg_logprob": -0.3, "language": "en", "language_prob": 0.9, "no_speech_prob": 0.1}


def _scripted_clips(split: str) -> list[dict[str, Any]]:
    rows = [("self_reply", "self", "HELP", "help tiny-garbles"), ("self_reply", "self", "HELP", "I fell down"), ("self_reply", "self", "REASSURANCE", "I'm fine"),
            ("self_reply", "self", "OTHER", "what time is it"), ("visitor_answer", "visitor", "REASSURANCE", "she is fine"), ("tv_says_help", "tv", "HELP", "help us"),
            ("tv_speech", "tv", "OTHER", "tomorrow will be cloudy"), ("silence", "none", "NO_RESPONSE", None), ("noise_only", "none", "NO_RESPONSE", "")]
    out = []
    for i, (b, role, exp, txt) in enumerate(rows):
        a = np.full(SR, 0.01 * (i + 1), dtype=np.float32)
        FakeASR.registry[id(a)] = txt or ""
        out.append({"cid": f"{split}-{i}", "split": split, "bucket": b, "role": role, "lang": "en", "expected": exp, "text": txt, "group": f"{split}-g{i}", "family": f"{split}-f{i}",
                    "voice": "stub", "emulated": False, "probe": b == "noise_only", "audio": a, "speech": b not in ("silence",) and not (b == "noise_only")})
    return out


def test_protocol_order_dev_choice_is_written_before_the_test_split_is_built(bank, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    built: list[str] = []

    def fake_build(bank_, split, *a, **k):
        events.append(f"build:{split}")
        built.append(split)
        return _scripted_clips(split)

    monkeypatch.setattr(AI, "build_clips", fake_build)
    plan = _plan(monkeypatch)
    grid = [AI.ASRCfg("faster-whisper-tiny", "none", "auto"), AI.ASRCfg("faster-whisper-base", "none", "auto"), AI.ASRCfg("faster-whisper-tiny", "hotwords", "auto_en_zh")]
    res = AI.run_protocol(bank, lambda jobs: [], plan, FakeASR, {"dev": [], "test": []}, {"dev": [], "test": []}, 42, grid=grid, on_dev_done=lambda sel: events.append("dev_done:" + str(sel["selected"])))
    assert events[0] == "build:dev" and events[1].startswith("dev_done:") and events[2] == "build:test"
    assert built == ["dev", "test"]
    assert res["selected"] is not None and res["test"] is not None
    # test cells = the production baseline, the selected cell and its one-factor neighbours - nothing else
    sel = res["selected"][0]
    expected = {AI.ASRCfg(*AI.BASELINE[:3]).key, sel} | {n.key for n in AI.neighbours(sel)}
    assert set(res["test"]) == expected
    assert "test_selected" in res and "test_baseline" in res


def test_dev_only_run_never_touches_the_test_split(bank, monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[str] = []

    def fake_build(bank_, split, *a, **k):
        built.append(split)
        return _scripted_clips(split)

    monkeypatch.setattr(AI, "build_clips", fake_build)
    plan = _plan(monkeypatch)
    res = AI.run_protocol(bank, lambda jobs: [], plan, FakeASR, {"dev": [], "test": []}, {"dev": [], "test": []}, 42, dev_only=True, grid=[AI.ASRCfg("faster-whisper-tiny", "none", "auto")])
    assert built == ["dev"] and res["test"] is None and "test_clip_rows" not in res


def test_silence_is_no_response_and_the_ladder_of_methods_is_scored(bank, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AI, "build_clips", lambda bank_, split, *a, **k: _scripted_clips(split))
    plan = _plan(monkeypatch)
    res = AI.run_protocol(bank, lambda jobs: [], plan, FakeASR, {"dev": [], "test": []}, {"dev": [], "test": []}, 42, dev_only=True,
                          grid=[AI.ASRCfg("faster-whisper-tiny", "none", "auto"), AI.ASRCfg("faster-whisper-base", "none", "auto")])
    tiny, base = res["dev"]["tiny|none|auto"], res["dev"]["base|none|auto"]
    assert set(tiny["scores"]) == set(AI.METHODS)
    assert tiny["preds"]["rule_v2"][7] == "NO_RESPONSE"  # the silence clip never reaches the ASR
    assert tiny["trs"][7] is None
    assert tiny["scores"]["rule_v2"]["safety"]["self_help_missed"] == 1 and base["scores"]["rule_v2"]["safety"]["self_help_missed"] == 0  # base 'hears' the garbled call
    elig = [r for r in res["dev_ranked"] if r["eligible"]]
    best = max(r["macro_f1"] for r in elig)
    pick = next(r for r in elig if (r["cfg"], r["method"]) == tuple(res["selected"]))
    near = [r for r in elig if r["macro_f1"] >= best - AI.TIE]
    assert pick["macro_f1"] >= best - AI.TIE and pick["self_help_missed"] == min(r["self_help_missed"] for r in near)  # the declared rule, whatever the stub's ties
    assert all(isinstance(r["latency_median_s"], float) for r in res["dev_ranked"])


# ---------------------------------------------------------------- report
def _full_result(bank, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[str, Any]]:
    monkeypatch.setattr(AI, "build_clips", lambda bank_, split, *a, **k: _scripted_clips(split))
    plan = _plan(monkeypatch)
    grid = [AI.ASRCfg("faster-whisper-tiny", "none", "auto"), AI.ASRCfg("faster-whisper-base", "none", "auto")]
    res = AI.run_protocol(bank, lambda jobs: [], plan, FakeASR, {"dev": [], "test": []}, {"dev": [], "test": []}, 42, grid=grid)
    profile = {"platform": "Windows", "backend": "sapi", "voices": {k: {"label": v.label, "emulated": v.emulated} for k, v in plan.items()}}
    return res, profile


def test_report_names_platform_voices_and_emulation_and_reports_test_once(bank, monkeypatch: pytest.MonkeyPatch) -> None:
    res, profile = _full_result(bank, monkeypatch)
    head, md, metrics = AI.render_report(res, profile, bank)
    assert "**Windows**" in md and "sapi" in md and "Zira" in md and "David" in md and "YES (resampling shift)" in md
    for needle in ("## TEST (reported once)", "Selection on DEV only", "Confusion matrix", "Accuracy per interference bucket", "Safety counts", "Hygiene audit", "oracle",
                   "self call for help NOT recognised as HELP", "TV / visitor speech recognised as REASSURANCE", "Hallucination probe"):
        assert needle in md, needle
    assert md.count("## TEST (reported once)") == 1
    assert "selected on dev" in head and "baseline" in head
    json.dumps(metrics, default=str)  # JSON-able
    assert metrics["profile"] == {"platform": "Windows", "backend": "sapi"} and "test" in metrics and metrics["lexicon_audit"]["n_cues_fired"] > 0


def test_dev_only_report_has_no_test_section(bank, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AI, "build_clips", lambda bank_, split, *a, **k: _scripted_clips(split))
    plan = _plan(monkeypatch)
    res = AI.run_protocol(bank, lambda jobs: [], plan, FakeASR, {"dev": [], "test": []}, {"dev": [], "test": []}, 42, dev_only=True, grid=[AI.ASRCfg("faster-whisper-tiny", "none", "auto")])
    head, md, metrics = AI.render_report(res, {"platform": "Windows", "backend": "sapi", "voices": {}}, bank)
    assert "dev-only run" in head and "TEST (reported once)" not in md and "test" not in metrics


def test_lexicon_audit_flags_cues_that_only_test_wording_triggers(bank) -> None:
    a = AI.lexicon_audit(bank)
    assert a["n_cues_fired"] > 20 and 0 <= a["share_only_test"] <= 1
    assert set(a["only_test"]).isdisjoint(a["only_dev"])
    assert all(v["dev"] + v["test"] > 0 for v in a["by_cue"].values())


# ---------------------------------------------------------------- suite entry: statuses, naming, nothing on disk but labels
@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("homeease.experiments.runner.artifacts_dir", lambda: tmp_path)
    return tmp_path


def test_no_tts_engine_is_not_evaluated(artifacts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "native_backend_name", lambda: "none")
    r = AI.exp_intent_bank()
    assert r["status"] == "not_evaluated" and r["missing"] and r["name"] == "A3b_response_intent_bank_none"
    assert not r["metrics"]  # no invented numbers


def test_missing_models_are_data_pending_and_runs_are_named_per_backend(artifacts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "native_backend_name", lambda: "sapi")
    monkeypatch.setattr("homeease.core.paths.models_dir", lambda: artifacts / "no_models")
    r = AI.exp_intent_bank()
    assert r["status"] == "data_pending" and any("faster-whisper" in m for m in r["missing"])
    assert r["name"] == "A3b_response_intent_bank_sapi"  # a macOS run would be ..._say: rows are never mixed
    assert "A3b_response_intent_bank_sapi" in r["provenance"]["run_id"]


def test_invalid_phrase_bank_fails_loudly(artifacts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "native_backend_name", lambda: "sapi")

    def boom(path=None):
        raise ValueError("bank broken")

    monkeypatch.setattr(AI, "load_phrase_bank", boom)
    r = AI.exp_intent_bank()
    assert r["status"] == "failed" and "bank broken" in r["headline"]


def test_legacy_a3_helpers_use_the_tts_abstraction() -> None:
    from homeease.experiments import audio_suite as S

    assert S._a3_voice_plan("say")["tv"] == "Fred" and S._a3_voice_plan("say")["proxy_zh_ratio"] is None  # the macOS plan is untouched
    p = S._a3_voice_plan("sapi")
    assert p["self"] == {"en": "Zira", "zh": "Huihui"} and p["proxy"]["en"] == "David" and p["proxy_zh_ratio"] and p["tv"] == "David"


def test_written_run_directory_holds_labels_only(artifacts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whole suite entry with stubs: statuses, files and columns that reach the disk (raw audio is never persisted)."""
    models = artifacts / "models"
    for m in AI.MODELS:
        (models / "audio" / m).mkdir(parents=True)
        (models / "audio" / m / "model.bin").write_bytes(b"x")
    monkeypatch.setattr("homeease.core.paths.models_dir", lambda: models)
    monkeypatch.setattr("homeease.ml.audio.asr.WhisperASR", FakeASR)
    monkeypatch.setattr(T, "native_backend_name", lambda: "sapi")
    monkeypatch.setattr(AI, "_esc_pools", lambda: {"dev": ["a.wav"], "test": ["b.wav"]})
    monkeypatch.setattr(AI, "_libri_pool", lambda seed: {"dev": [{"path": "x"}], "test": [{"path": "y"}]})
    monkeypatch.setattr(AI, "build_clips", lambda bank_, split, *a, **k: _scripted_clips(split))
    plan = _plan(monkeypatch)
    monkeypatch.setattr(AI, "plan_voices", lambda backend: plan)
    monkeypatch.setattr(AI, "full_grid", lambda: [AI.ASRCfg("faster-whisper-tiny", "none", "auto"), AI.ASRCfg("faster-whisper-base", "hotwords", "auto_en_zh")])
    r = AI.exp_intent_bank()
    assert r["status"] == "completed", r
    run_dir = next(artifacts.glob("*A3b_response_intent_bank_sapi*"))
    files = sorted(p.name for p in run_dir.iterdir())
    assert files == ["config.yaml", "dev_selection.json", "metrics.json", "predictions.jsonl", "report.md"]
    sel = json.loads((run_dir / "dev_selection.json").read_text(encoding="utf-8"))
    assert sel["selected"] and sel["dev_ranked"] and sel["grid"]
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and all("audio" not in r_ and "speech" not in r_ for r_ in rows)
    assert {"split", "cid", "bucket", "role", "lang", "voice", "expected", "said", "asr", "heard", "latency_s"} <= set(rows[0]) and {"dev", "test"} <= {r_["split"] for r_ in rows}
    assert max(p.stat().st_size for p in run_dir.iterdir()) < 2_000_000  # no sample dumps
    assert not list(artifacts.rglob("*.wav")) and not list(artifacts.rglob("*.flac")) and not list(artifacts.rglob("*.npy")) and not list(artifacts.rglob("*.npz"))
    assert "Windows" in (run_dir / "report.md").read_text(encoding="utf-8") or "sapi" in (run_dir / "report.md").read_text(encoding="utf-8")
