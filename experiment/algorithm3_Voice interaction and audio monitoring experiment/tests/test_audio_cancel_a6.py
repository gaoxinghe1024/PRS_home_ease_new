"""Experiment A6 (cancel-eligibility, experiments/audio_cancel.py): plan and split hygiene, statistics, the vectorised analysis versus the gate's own rule,
the full protocol with stub voices / embedder / ASR (no Piper, no model, no network), the partial fallback, status paths and the contents of a run folder.
Real Piper / WeSpeaker / Whisper are only used by the real run (`homeease compare --suite audio --only A6`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from homeease.experiments import audio_cancel as AC
from homeease.ml.audio import cancel_gate as cg
from homeease.ml.audio.phrases import load_phrase_bank

SR = 16000
PAD = 2 * int(AC.PAD_S * SR)


@pytest.fixture(scope="module")
def bank():  # type: ignore[no-untyped-def]
    return load_phrase_bank()


# ---------------------------------------------------------------------------------------------------- stub engine
class StubWorld:
    """Voices, embedder and ASR that agree with each other: a speaker = a fundamental frequency, a text = a clip length."""

    def __init__(self) -> None:
        self.registry: list[str] = []
        self.synth_calls = 0

    def synth(self, text: str, spk: int) -> np.ndarray:
        self.synth_calls += 1
        if text not in self.registry:
            self.registry.append(text)
        k = self.registry.index(text)
        n = int((1.6 + 0.02 * k) * SR)
        t = np.arange(n) / SR
        f0 = 100.0 + 12.0 * spk
        x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6)) * np.hanning(n) ** 0.25
        return (0.4 * x / np.abs(x).max()).astype(np.float32)

    # embedder: log band energies between 60 and 1800 Hz (a harmonic comb moves with f0), unit norm
    model_id = "stub_model"

    def embed(self, audio16k: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(audio16k[: 2 * SR], n=2 * SR)) ** 2
        freqs = np.fft.rfftfreq(2 * SR, 1 / SR)
        edges = np.geomspace(60, 1800, 25)
        v = np.array([np.log(spec[(freqs >= a) & (freqs < b)].sum() + 1e-6) for a, b in zip(edges[:-1], edges[1:])])
        v = v - v.mean()
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)

    def transcribe(self, audio: np.ndarray, lang: Any, **kw: Any) -> dict[str, Any]:
        k = round(((len(audio) - PAD) / SR - 1.6) / 0.02)
        text = self.registry[k] if 0 <= k < len(self.registry) and abs(((len(audio) - PAD) / SR) - (1.6 + 0.02 * k)) < 0.011 else ""
        return {"text": text, "avg_logprob": -0.3 if text else -2.0, "model": "stub"}


def stub_ctx(world: StubWorld, min_speech_s: float = 1.0) -> AC.Ctx:
    from homeease.ml.audio.intent_v2 import RuleIntentV2

    rng = np.random.default_rng(0)
    noise = [rng.normal(0, 0.05, 5 * SR).astype(np.float32) for _ in range(12)]
    return AC.Ctx(world.synth, world, world, RuleIntentV2(), noise, min_speech_s, cg.DEFAULT_MIN_ASR_LOGPROB, None, AC.load_texts(load_phrase_bank()))


# ---------------------------------------------------------------------------------------------------- plan
def test_plan_speakers_are_disjoint_and_phrases_keep_their_split(bank) -> None:  # type: ignore[no-untyped-def]
    plan = AC.make_plan(bank, 904, 150, 150, seed=42)
    assert len(plan.dev) == 150 and len(plan.eval) == 150 and not set(plan.dev) & set(plan.eval)
    assert all(0 <= s < 904 for s in plan.dev + plan.eval)
    dev_ids, eval_ids = plan.phrase_ids()
    by_id = {e.id: e for e in bank.entries}
    assert dev_ids and eval_ids and not dev_ids & eval_ids
    assert {by_id[i].split for i in dev_ids} == {"dev"} and {by_id[i].split for i in eval_ids} == {"test"}
    assert not {by_id[i].family for i in dev_ids} & {by_id[i].family for i in eval_ids}  # paraphrase families never cross
    assert {by_id[i].lang for i in dev_ids | eval_ids} == {"en"}
    for src, role, intent in (("self_reas", "self", "REASSURANCE"), ("help", "self", "HELP"), ("visitor", "visitor", "REASSURANCE"), ("tv", "tv", "REASSURANCE")):
        used = {p for v in plan.eval_phrases[src].values() for p in v}
        assert used and all(by_id[p].speaker_role == role and by_id[p].intent == intent for p in used), src
    assert {by_id[p].intent for v in plan.dev_probes.values() for p in v} == {"REASSURANCE"}
    s = plan.eval[0]
    assert plan.eval_phrases["self_reas_n5"][s] == plan.eval_phrases["self_reas"][s][:2] and plan.eval_phrases["replay"][s] == plan.eval_phrases["self_reas"][s][:2]
    assert len(plan.eval_phrases["self_reas"][s]) == 4 and len(plan.eval_phrases["help"][s]) == 3 and len(plan.dev_probes[plan.dev[0]]) == 4


def test_plan_is_deterministic_and_seed_dependent_and_validates_counts(bank) -> None:  # type: ignore[no-untyped-def]
    a, b, c = (AC.make_plan(bank, 100, 10, 10, seed=s) for s in (1, 1, 2))
    assert a == b and a != c
    with pytest.raises(ValueError, match="need 30 speakers"):
        AC.make_plan(bank, 20, 10, 20, seed=1)
    p = AC.make_plan(bank, 3, 0, 3, seed=1)  # the SAPI fallback shape: no dev speakers
    assert p.dev == () and len(p.eval) == 3


# ---------------------------------------------------------------------------------------------------- statistics
def test_identity_bootstrap_brackets_the_rate_and_is_wide_when_one_identity_carries_it() -> None:
    R = C = 40
    rng = np.random.default_rng(0)
    N = np.ones((R, C)) * 5
    np.fill_diagonal(N, 0)
    K = rng.binomial(5, 0.2, (R, C)).astype(float)
    np.fill_diagonal(K, 0)
    lo, hi = AC.identity_bootstrap(K, N, paired=False, seed=1)  # type: ignore[misc]
    rate = K.sum() / N.sum()
    assert lo < rate < hi and hi - lo < 0.1
    K2 = np.zeros((R, C))
    K2[3, :] = 5.0  # one clip speaker produces ALL the false accepts
    np.fill_diagonal(K2, 0)
    lo2, hi2 = AC.identity_bootstrap(K2, N, paired=False, seed=1)  # type: ignore[misc]
    assert (hi2 - lo2) > (hi - lo) * 0.5 and lo2 < 0.005  # trials are not independent: the interval says so (Wilson on 7800 trials would be ~0.0005 wide)
    kd, nd = np.diag(rng.binomial(2, 0.5, R).astype(float)), np.diag(np.full(R, 2.0))
    lo3, hi3 = AC.identity_bootstrap(kd, nd, paired=True, seed=1)  # type: ignore[misc]
    assert lo3 < kd.sum() / nd.sum() < hi3
    assert AC.identity_bootstrap(np.ones((2, 2)), np.ones((2, 2)), paired=False) is None  # too few identities


def test_rate_ci_and_formatting() -> None:
    r = AC.rate_ci(0, 2208)
    assert r["rate"] == 0.0 and r["wilson95"][0] == 0.0 and 0.0016 < r["wilson95"][1] < 0.0018
    assert AC.rate_ci(0, 0)["rate"] is None and AC.pct(AC.rate_ci(0, 0)) == "n/a"
    assert AC.pct(AC.rate_ci(14, 450)).startswith("3.11% (14/450) [")
    assert "degenerate" in AC.boot({"k": 0, "identity_bootstrap95": (0.0, 0.0)})


def test_trial_mask_kinds() -> None:
    voices, users = np.array([5, 6, -1]), np.array([5, 6, 7])
    assert AC.trial_mask("own", voices, users).tolist() == [[True, False, False], [False, True, False], [False, False, False]]
    assert AC.trial_mask("others", voices, users).tolist() == [[False, True, True], [True, False, True], [False, False, False]]
    assert AC.trial_mask("all", voices, users).all()


def test_duration_buckets_report_eer_only_with_enough_trials() -> None:
    r = np.random.default_rng(0)
    n = 12000
    labels = (r.random(n) < 0.1).astype(int)
    scores = np.where(labels == 1, r.normal(0.7, 0.1, n), r.normal(0.2, 0.1, n))
    sp = r.uniform(0.3, 3.0, n)
    b = AC.duration_buckets(scores, labels, sp)
    assert [x["speech_s"] for x in b] == ["0-0.75", "0.75-1", "1-1.5", "1.5-2", ">=2"]
    assert all(x["eer"] is not None and x["eer"] < 0.05 for x in b) and sum(x["genuine"] + x["impostor"] for x in b) == n
    tiny = AC.duration_buckets(scores[:50], labels[:50], sp[:50])
    assert all(x["eer"] is None for x in tiny)


# ---------------------------------------------------------------------------------------------------- vectorised analysis == gate rule
def table(n_users: int = 6, seed: int = 0) -> AC.EvalTable:
    """Hand-made eval table: every source with random flags and scores."""
    r = np.random.default_rng(seed)
    users = np.arange(10, 10 + n_users)
    rows: list[dict[str, Any]] = []
    for src in AC.SOURCES:
        for k in range(30):
            voice = -1 if src in ("noise", "silence") else int(users[k % n_users])
            status = "NO_RESPONSE" if src == "silence" else "RESPONDED"
            intent = None if status == "NO_RESPONSE" else str(r.choice(["REASSURANCE", "HELP", "OTHER", "UNCERTAIN"], p=[0.6, 0.1, 0.2, 0.1]))
            rows.append({"cid": f"{src}-{len(rows)}", "source": src, "voice": voice, "phrase_id": None, "said": "x", "group": f"g{k}", "status": status, "intent": intent, "intent_reason": "r",
                         "asr_conf": None if status == "NO_RESPONSE" else float(r.uniform(-1.6, -0.1)), "transcript": "t", "speech_s": float(r.uniform(0.3, 3.0)) if src != "silence" else 0.0, "embedded": True})
    S = r.uniform(0.0, 1.0, (len(rows), n_users)).astype(np.float32)
    S[r.random(S.shape) < 0.05] = np.nan
    return AC.EvalTable(users, rows, S)


def test_vectorised_eligibility_equals_the_gate_decision_for_every_trial() -> None:
    ev = table()
    thr = {0.001: 0.8, 0.01: 0.55}
    ms, mc = 1.0, -1.0
    stats = AC.scenario_stats(ev, AC.base_flags(ev.rows, ms, mc), thr, seed=0)
    voices = np.array([r["voice"] for r in ev.rows])
    for name, source, kind in AC.SCENARIOS:
        sel = [i for i, r in enumerate(ev.rows) if r["source"] == source]
        mask = AC.trial_mask(kind, voices[sel], ev.users)
        for far, t in thr.items():
            want = 0
            for a, i in enumerate(sel):
                for j in range(len(ev.users)):
                    if not mask[a, j]:
                        continue
                    r = ev.rows[i]
                    sc = ev.S[i, j]
                    d = cg.decide(enrolled=True, threshold=t, speech_s=r["speech_s"], min_speech_s=ms, intent=r["intent"] if r["status"] == "RESPONDED" else None, asr_conf=r["asr_conf"],
                                  min_asr_logprob=mc, score=float(sc) if np.isfinite(sc) else None)
                    want += int(d.cancel_eligible)
            got = stats[name]["operating_points"][AC.OP_NAMES[far]]
            assert got["k"] == want and got["n"] == int(mask.sum()), (name, far)


def test_scenario_rates_are_ordered_by_construction() -> None:
    ev = table(8, seed=3)
    s = AC.scenario_stats(ev, AC.base_flags(ev.rows, 1.0, -1.0), {0.001: 0.9, 0.01: 0.5}, seed=0)
    for name, d in s.items():
        strict, loose = (d["operating_points"][k] for k in ("far_0.1pct", "far_1pct"))
        assert strict["k"] <= loose["k"] <= d["all_but_voiceprint"]["k"] <= d["intent_only"]["k"] or d["all_but_voiceprint"]["k"] <= d["intent_only"]["k"], name
        assert strict["voiceprint_only"]["k"] >= strict["k"]


def test_a_missing_threshold_means_no_eligibility_figures() -> None:
    ev = table()
    s = AC.scenario_stats(ev, AC.base_flags(ev.rows, 1.0, -1.0), {0.001: None, 0.01: None}, seed=0)
    assert all(d["operating_points"]["far_0.1pct"] == {"threshold": None, "status": "no_threshold"} for d in s.values())
    assert all(d["intent_only"]["n"] > 0 for d in s.values())  # the intent-only rate does not need a voiceprint


def test_reason_breakdown_uses_the_gate_rule() -> None:
    ev = table()
    rb = AC.reason_breakdown(ev, {0.001: 0.8}, 1.0, -1.0, per_scenario=50, seed=0)
    assert set(rb) == {n for n, _, _ in AC.SCENARIOS}
    assert all(set(d["reason_counts"]) <= set(cg.REASON_CODES) and d["sampled_trials"] <= 50 for d in rb.values())
    nothr = AC.reason_breakdown(ev, {0.001: None}, 1.0, -1.0, per_scenario=50, seed=0)
    assert all(d["eligible"] == 0 and d["reason_counts"].get("no_threshold", 0) == d["sampled_trials"] for d in nothr.values())


# ---------------------------------------------------------------------------------------------------- calibration on dev
def test_dev_trials_only_use_clips_with_enough_speech_and_label_the_enrolled_speaker() -> None:
    dev = AC.DevTable(np.array([1, 2, 3]), np.array([1, 2, 3, 1]), np.array([2.0, 0.5, 2.0, 2.0]), np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7], [0.8, np.nan, 0.1]], dtype=np.float32))
    s, y = AC.dev_trials(dev, 1.0)
    assert len(s) == 8  # 3 probes with enough speech x 3 enrollments, minus one NaN
    assert y.sum() == 3 and (y == 0).sum() == len(y) - 3 and np.all(np.isfinite(s))  # the 0.5 s probe and the NaN are gone
    cal = AC.calibrate_dev(dev, 1.0)
    assert set(cal["calibrations"]) == {"far_0.1pct", "far_1pct"} and cal["n_speakers"] == 3 and cal["probes_with_enough_speech"] == 3


# ---------------------------------------------------------------------------------------------------- full protocol with stubs
@pytest.fixture(scope="module")
def stub_run(bank):  # type: ignore[no-untyped-def]
    world = StubWorld()
    ctx = stub_ctx(world)
    plan = AC.make_plan(bank, 80, 24, 24, seed=7, n_noise_only=10, n_silence=4)
    dev = AC.collect_dev(plan, ctx)
    cal = AC.calibrate_dev(dev, ctx.min_speech_s)
    ev = AC.collect_eval(plan, ctx, AC.thresholds_from(cal), audit_share=0.25)
    m = AC.analyse(dev, ev, ctx.min_speech_s, ctx.min_asr_logprob, 7)
    return world, plan, dev, ev, cal, m


def test_stub_protocol_counts_and_hygiene(stub_run) -> None:  # type: ignore[no-untyped-def]
    world, plan, dev, ev, cal, m = stub_run
    n = len(plan.eval)
    by_src = {s: sum(1 for r in ev.rows if r["source"] == s) for s in AC.SOURCES}
    assert by_src == {"self_reas": 4 * n, "self_reas_n5": 2 * n, "help": 3 * n, "visitor": 2 * n, "tv": 2 * n, "replay": 2 * n, "noise": 10, "silence": 4}
    assert ev.S.shape == (len(ev.rows), n) and list(ev.users) == sorted(plan.eval) and dev.S.shape == (4 * len(plan.dev), len(plan.dev))
    assert not set(int(s) for s in dev.speakers) & set(int(s) for s in ev.users)
    # dev speakers never spoke eval words and vice versa
    dev_texts = {r for r in world.registry}
    assert dev_texts  # the stub registers every synthesised text
    assert all(r["voice"] in plan.eval or r["voice"] == -1 for r in ev.rows)
    # no audio, no embeddings in the rows
    allowed = {"cid", "source", "voice", "phrase_id", "said", "group", "status", "intent", "intent_reason", "asr_conf", "transcript", "speech_s", "embedded"}
    assert all(set(r) == allowed for r in ev.rows)
    assert all(isinstance(v, (str, int, float, bool, type(None))) for r in ev.rows for v in r.values())


def test_stub_protocol_results_make_sense(stub_run) -> None:  # type: ignore[no-untyped-def]
    world, plan, dev, ev, cal, m = stub_run
    sc = m["scenarios"]
    thr = m["thresholds"]["far_1pct"]
    assert thr is not None and cal["calibrations"]["far_0.1pct"]["threshold"] is not None and m["thresholds"]["far_0.1pct"] >= thr
    tc = sc["true_cancel_clean"]["operating_points"]["far_1pct"]
    assert tc["rate"] > 0.5, tc  # the enrolled voice, reassuring words, clean: mostly eligible
    assert sc["other_person_same_words"]["intent_only"]["rate"] > 0.6  # the words alone would have cancelled
    assert sc["other_person_same_words"]["operating_points"]["far_1pct"]["rate"] < 0.1  # the voiceprint stops most of it
    assert sc["enrolled_says_help"]["operating_points"]["far_1pct"]["k"] == 0  # a call for help is never eligible
    assert sc["silence"]["operating_points"]["far_1pct"]["k"] == 0 and sc["household_noise_only"]["operating_points"]["far_1pct"]["k"] == 0
    assert m["gate_audit"]["n"] >= 20 and m["gate_audit"]["mismatch"] == 0  # CancelGate.evaluate on the audio == the vectorised rule
    assert sc["true_cancel_clean"]["should_cancel"] and not sc["tv_radio_speech"]["should_cancel"]
    assert m["partial"] is False and m["eval_users"] == len(plan.eval) and m["speaker_layer"]["with_enough_speech"]["eer"] is not None
    json.dumps(m, default=float)  # the metrics are JSON-serialisable


def test_stub_protocol_report_states_the_caveats(stub_run) -> None:  # type: ignore[no-untyped-def]
    *_, m = stub_run
    head, md = AC.render(m, {"engine": {"engine": "stub", "n_speakers": 80}, "speaker_model": "stub", "min_speech_s": 1.0, "min_asr_logprob": -1.0})
    assert "FAR 0.1%" in head and "synthetic" in head
    for must in ("10,000 impostor trials", "synthetic and clean", "never acted on", "Wilson", "speaker-bootstrap", "intent only", "replay_of_enrolled_voice", "English only", "Consistency audit", "enrolled_says_help"):
        assert must in md, must
    assert "PARTIAL" not in md


def test_rescore_from_saved_tables_reproduces_the_analysis_and_stores_no_audio(stub_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    world, plan, dev, ev, cal, m = stub_run
    monkeypatch.setattr("homeease.experiments.runner.artifacts_dir", lambda: tmp_path)
    run = AC.Run("audio", "A6_test", {"seed": 7, "min_speech_s": 1.0, "min_asr_logprob": -1.0, "speaker_model": "stub", "partial": False, "engine": {"engine": "stub"}}, "stub", 7)
    AC._save_tables(run, dev, ev)
    names = sorted(p.name for p in run.dir.iterdir())
    assert names == ["config.yaml", "predictions.jsonl", "scores.npz"]
    z = np.load(run.dir / "scores.npz")
    assert set(z.files) == {"dev_speakers", "dev_probe_spk", "dev_probe_speech_s", "dev_S", "eval_users", "eval_S", "dev_refused", "eval_refused", "audit_json"}
    assert all(z[k].dtype.kind in "iufU" for k in z.files) and sum(z[k].nbytes for k in z.files) < 3_000_000  # scores only: tiny, no waveform-sized array
    assert max(z[k].size for k in z.files) <= len(ev.rows) * len(ev.users)
    m2, text = AC.rescore_run(run.dir)
    assert m2["gate_audit"] == m["gate_audit"] and m2["gate_audit"]["n"] > 0  # the consistency audit survives the round trip
    for name in ("true_cancel_clean", "other_person_same_words", "replay_of_enrolled_voice"):
        for op in ("far_0.1pct", "far_1pct"):
            a, b = m["scenarios"][name]["operating_points"][op], m2["scenarios"][name]["operating_points"][op]
            assert (a["k"], a["n"]) == (b["k"], b["n"])
    assert m2["thresholds"] == m["thresholds"] and "Scenarios" in text


# ---------------------------------------------------------------------------------------------------- partial fallback and statuses
def test_partial_mode_reports_no_eligibility_rate(bank) -> None:  # type: ignore[no-untyped-def]
    world = StubWorld()
    ctx = stub_ctx(world)
    plan = AC.make_plan(bank, 3, 0, 3, seed=1, n_noise_only=3, n_silence=2)
    empty = AC.DevTable(np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0), np.zeros((0, 0), dtype=np.float32))
    ev = AC.collect_eval(plan, ctx, {0.001: None, 0.01: None})
    m = AC.analyse(empty, ev, 1.0, -1.0, 1, partial=True)
    assert m["partial"] and m["thresholds"] == {"far_0.1pct": None, "far_1pct": None}
    head, md = AC.render(m, {"engine": {"engine": "system TTS voices", "n_speakers": 3}, "speaker_model": "x", "min_speech_s": 1.0, "min_asr_logprob": -1.0})
    assert head.startswith("PARTIAL") and "PARTIAL RUN" in md and "not evaluated (no calibrated threshold)" in md
    assert "eligible 0.00%" not in head  # no invented rate
    assert m["reliability"]["far_0.1pct"]["reliable"] is False and m["reliability"]["dev_impostor_trials"] == 0
    assert m["gate_audit"]["n"] == 0  # no threshold: nothing to audit


def run_with_stubs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, piper: bool, sapi: bool = True, models: bool = True, esc: bool = True) -> dict[str, Any]:
    from homeease.ml.audio import asr as asr_mod
    from homeease.ml.audio import multi_tts
    from homeease.ml.audio import speaker as spk_mod

    world = StubWorld()
    monkeypatch.setattr("homeease.experiments.runner.artifacts_dir", lambda: tmp_path / "artifacts")
    monkeypatch.setattr(AC, "models_dir", lambda: tmp_path / "models")
    if models:
        for rel in ("audio/wespeaker/voxceleb_resnet34_LM.onnx", "audio/faster-whisper-tiny/model.bin"):
            (tmp_path / "models" / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / "models" / rel).write_bytes(b"x")
    monkeypatch.setattr("homeease.experiments.audio_intent._esc_pools", lambda: {"dev": ["a.wav"], "test": ["b.wav"] * 20} if esc else {"dev": [], "test": []})
    monkeypatch.setattr(AC, "load_audio", lambda p: np.random.default_rng(0).normal(0, 0.05, 5 * SR).astype(np.float32))
    monkeypatch.setattr(asr_mod, "WhisperASR", lambda name: world)
    monkeypatch.setattr(spk_mod, "WeSpeakerEmbedder", lambda name: world)

    class StubPiper:
        n_speakers = 80

        def __init__(self) -> None:
            self.synthesize = world.synth

        def describe(self) -> dict[str, Any]:
            return {"engine": "stub-piper", "n_speakers": 80, "model_sha256": "abc"}

    monkeypatch.setattr(multi_tts, "availability", lambda: (True, None) if piper else (False, "piper-tts is not installed"))
    monkeypatch.setattr(multi_tts, "PiperSpeakers", StubPiper)
    monkeypatch.setattr(AC, "sapi_speakers", (lambda: (world.synth, ["Zira", "David", "Huihui"])) if sapi else (lambda: None))
    return AC.exp_cancel_eligibility(seed=5, n_dev=24, n_eval=24, n_noise_only=6, n_silence=3, with_real_check=False)


def test_exp_with_a_multispeaker_engine_completes_and_writes_a_clean_run_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    res = run_with_stubs(monkeypatch, tmp_path, piper=True)
    assert res["status"] == "completed" and not res["headline"].startswith("PARTIAL")
    d = Path(next((tmp_path / "artifacts").iterdir()))
    assert sorted(p.name for p in d.iterdir()) == ["config.yaml", "metrics.json", "predictions.jsonl", "report.md", "scores.npz"]
    assert not [p for p in d.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg", ".npy")]
    cfg = (d / "config.yaml").read_text(encoding="utf-8")
    assert "stub-piper" in cfg and "min_speech_s" in cfg
    assert res["metrics"]["scenarios"]["true_cancel_clean"]["operating_points"]["far_1pct"]["rate"] is not None


def test_exp_falls_back_to_a_labelled_partial_run_without_piper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    res = run_with_stubs(monkeypatch, tmp_path, piper=False, sapi=True)
    assert res["status"] == "synthetic_only" and res["headline"].startswith("PARTIAL RUN") and res["metrics"]["partial"] is True
    assert res["metrics"]["thresholds"] == {"far_0.1pct": None, "far_1pct": None}
    assert "PARTIAL RUN" in (Path(next((tmp_path / "artifacts").iterdir())) / "report.md").read_text(encoding="utf-8")
    assert res["metrics"]["eval_users"] == 3


def test_exp_statuses_when_inputs_are_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    r = run_with_stubs(monkeypatch, tmp_path / "a", piper=False, sapi=False)
    assert r["status"] == "data_pending" and any("piper-tts" in m for m in r["missing"]) and r["metrics"] == {}  # no engine at all: no numbers invented
    r = run_with_stubs(monkeypatch, tmp_path / "b", piper=True, models=False)
    assert r["status"] == "data_pending" and any("WeSpeaker" in m for m in r["missing"]) and any("faster-whisper" in m for m in r["missing"])
    r = run_with_stubs(monkeypatch, tmp_path / "c", piper=True, esc=False)
    assert r["status"] == "data_pending" and any("ESC-50" in m for m in r["missing"])


# ---------------------------------------------------------------------------------------------------- real-speech transfer check, suite wiring, TTS wrapper
def test_real_speech_check_with_stubs_and_status_without_data() -> None:
    world = StubWorld()
    speakers = {f"s{i}": [f"s{i}-{k}" for k in range(8)] for i in range(6)}

    def loader(p: str) -> np.ndarray:
        i = int(p.split("-")[0][1:])
        return world.synth("a long enough sentence for the check", i + 2)

    out = AC.real_speech_check(world, {0.001: 0.9, 0.01: None}, 1, 1.0, loader=loader, speakers=speakers, probes_per_speaker=3)
    assert out["status"] == "completed" and out["speakers"] == 6 and out["genuine_trials"] == 18 and out["impostor_trials"] == 18 * 5
    assert set(out["at_tts_dev_thresholds"]) == {"far_0.1pct"}  # no threshold, no figure
    assert AC.real_speech_check(world, {}, 1, 1.0, speakers={"a": ["x"]})["status"] == "data_pending"


def test_suite_runs_a6_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    from homeease.experiments import audio_suite as S

    called: list[str] = []

    def fake(name: str):  # type: ignore[no-untyped-def]
        def f(seed: int = 42) -> dict[str, Any]:
            called.append(name)
            return {"name": name, "status": "completed", "data_source": "x", "headline": "h", "provenance": {"run_id": name}}

        return f

    import homeease.experiments.audio_intent as AI

    for n, attr in (("A1", "exp_vad"), ("A2", "exp_sound"), ("A3", "exp_intent"), ("A4", "exp_speaker"), ("A5", "exp_alarm_behaviour"), ("A6", "exp_cancel_eligibility")):
        monkeypatch.setattr(S, attr, fake(n))
    monkeypatch.setattr(AI, "exp_intent_bank", fake("A3b"))
    monkeypatch.setattr(S, "write_suite_index", lambda *a, **k: None)
    assert [r["id"] for r in S.run_suite(["A6"])] == ["A6"] and called == ["A6"]
    called.clear()
    assert [r["id"] for r in S.run_suite(["A5"])] == ["A5"] and called == ["A5"]
    called.clear()
    assert [r["id"] for r in S.run_suite(["A3"])] == ["A3", "A3b"]  # the existing 'A3 runs A3b too' behaviour is unchanged
    called.clear()
    assert [r["id"] for r in S.run_suite(None)] == ["A1", "A2", "A3", "A3b", "A4", "A5", "A6"]


def test_audio_suite_wrapper_delegates_with_keyword_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    from homeease.experiments import audio_suite as S

    seen = {}
    monkeypatch.setattr(AC, "exp_cancel_eligibility", lambda seed=42, **kw: seen.update(seed=seed, **kw) or {"status": "x"})
    assert S.exp_cancel_eligibility(3, n_eval=10)["status"] == "x" and seen == {"seed": 3, "n_eval": 10}
