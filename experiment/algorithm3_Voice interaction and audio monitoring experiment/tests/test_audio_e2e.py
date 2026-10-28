"""Headless end-to-end check of Function 3 (proposal Figure 3-3), offline and synthetic only.

    scripted scene (seeded) -> Function3Pipeline (front end, sound branch, speech VAD) -> stored sound events (labels only)
    scripted suspected fall -> voice inquiry on the scene's microphone window (stub ASR, stub speaker model)
                            -> the event stays EMERGENCY / OPEN and `cancel_eligible` is recorded as EVIDENCE

No microphone, no model download, no TTS engine, no network: stub classifier / VAD / ASR / speaker model and generated sounds.
The stubs are crude on purpose; the tests exercise the plumbing, the policy and the privacy guarantees, not recognition accuracy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.audio_fixtures import HarmonicStubVAD, PitchEmbedder, ScriptedASR, StubSoundClassifier, SyntheticCatalog, fake_tts

from homeease.core.config import AppConfig, AudioCfg, PrivacyCfg
from homeease.experiments.headless import HeadlessSystem
from homeease.ml.audio import replay as rp
from homeease.ml.audio import scenes as sc
from homeease.ml.audio.cancel_gate import VoiceprintEnrollment
from homeease.ml.audio.monitor import CATEGORIES, SoundEvent

SR = 16000
THRESHOLD = 0.6  # cosine threshold of the stub speaker model (an enrolled pitch scores ~1.0, another pitch ~0.0)
SEED = 7


@pytest.fixture()
def system(cfg: AppConfig) -> Any:
    sysm = HeadlessSystem(cfg)
    yield sysm
    sysm.close()


def scene(name: str, **kw: Any) -> sc.Scene:
    return sc.build_scene(name, SyntheticCatalog(**kw), seed=SEED, tts=fake_tts)


def plan_of(s: sc.Scene, **kw: Any) -> rp.InquiryPlan:
    assert s.inquiry is not None
    return rp.InquiryPlan(s.inquiry["fall_at_s"], s.inquiry["inquiry_start_s"], s.inquiry["window_s"], **kw)


def enrolment(embedder: PitchEmbedder, voice: str = "Zira") -> VoiceprintEnrollment:
    return VoiceprintEnrollment.from_audio(embedder, [fake_tts(t, voice, "en") for t in sc.ENROLL_TEXTS])


def run(cfg: AppConfig, s: sc.Scene, text: str | None, system: Any | None = None, *, enrolled: bool = True, classifier: Any = "stub", vad: Any = None,
        plan: rp.InquiryPlan | None = None, asr: ScriptedASR | None = None) -> rp.ReplayOutcome:
    emb = PitchEmbedder()
    voice = rp.VoiceStack(asr=asr or ScriptedASR(text), embedder=emb, enrollment=enrolment(emb) if enrolled else None, threshold=THRESHOLD)
    return rp.run_replay(s.audio, SR, cfg, StubSoundClassifier() if classifier == "stub" else classifier, vad=vad or HarmonicStubVAD(),
                         plan=plan if plan is not None else (plan_of(s) if s.inquiry else None), voice=voice, truth=s.truth, system=system)


def state(d: dict[str, Any]) -> tuple[Any, ...]:
    """What an inquiry must not change: level, status, outcome, escalation step, notifications."""
    return (d["level"], d["max_level"], d["status"], d["outcome"], d["escalation_step"], d["notifications"])


# ------------------------------------------------------------------------------------------------ the whole chain
def test_scripted_scene_to_stored_labels_to_event_with_cancel_eligible_evidence(cfg: AppConfig, system: Any) -> None:
    s = scene("fall_reassured")
    out = run(cfg, s, "I'm fine, don't worry.", system)
    # front end and both branches ran, nothing degraded
    assert out.sound_status["status"] == "ok" and out.pipeline_status["status"] == "ok" and out.pipeline_status["voice"]["status"] == "ok"
    assert out.pipeline_status["raw_audio_stored"] is False and out.sound_status["raw_audio_stored"] is False
    # voice branch: the speech VAD found the reply (and only speech), sound branch: the thud was labelled
    reply = next(t for t in s.truth if t["kind"] == "reply")
    assert len(out.speech) == 1 and abs((out.speech[0]["t_start_mono_ms"] - out.t0_mono_ms) / 1000.0 - reply["t0_s"]) < 0.6
    assert [r["result"] for r in out.truth_check] == ["found"] and out.truth_check[0]["expect"] == "IMPACT"
    # the events are in the database as labels and scalars, and read back identical
    from sqlalchemy import select

    from homeease.backend.storage.models import SoundEventRecord

    with system.db.session() as ses:
        db_rows = ses.execute(select(SoundEventRecord).order_by(SoundEventRecord.t_start)).scalars().all()
        stored = {r.id: (r.category, r.confidence, r.n_windows, r.source_mode) for r in db_rows}
    assert stored and {e["event_id"] for e in out.events} == set(stored)
    assert all(v[0] in CATEGORIES and v[3] == "replay" for v in stored.values()) and {"IMPACT", "DAILY"} <= {v[0] for v in stored.values()}
    # the voice inquiry: evidence recorded, event untouched
    iq = out.inquiry
    assert iq is not None
    rec = iq["record"]
    assert rec["status"] == "RESPONDED" and rec["intent"] == "REASSURANCE" and rec["asr_model"] == "scripted"
    assert rec["cancel_eligible"] is True and rec["cancel_reasons"] == [] and rec["cancel_policy"] == "evidence_only" and rec["speaker_verified"] is True
    assert rec["speaker_score"] >= THRESHOLD and rec["voiceprint_threshold"] == THRESHOLD
    assert iq["event_before"]["notifications"] > 0  # caregivers were notified before the inquiry ran
    for st in (iq["event_before"], iq["after_reply"], iq["after_timers"]):
        assert st["level"] == "EMERGENCY" and st["status"] == "OPEN" and st["outcome"] is None
    assert iq["after_timers"]["escalation_step"] >= iq["event_before"]["escalation_step"] and iq["after_timers"]["notifications"] >= iq["event_before"]["notifications"]
    detail = system.rt.events.detail(iq["event_before"]["event_id"])
    vr = [t for t in detail["timeline"] if t["action"] == "voice_response"]
    assert len(vr) == 1 and vr[0]["payload"]["cancel_eligible"] is True and vr[0]["to_level"] is None and vr[0]["to_status"] is None  # logged for audit, no transition
    assert detail["resolved_at"] is None and detail["lifecycle_status"] == "OPEN"


@pytest.mark.parametrize("name,text,intent,eligible,reasons", [
    ("fall_reassured", "I'm fine, don't worry.", "REASSURANCE", True, []),
    ("fall_visitor_reply", "Don't worry, she is fine.", "REASSURANCE", False, ["speaker_mismatch"]),  # right words, another voice
    ("fall_help_call", "Help me, I fell and I cannot get up.", "HELP", False, ["intent_not_reassurance"]),  # the enrolled voice calling for help is never 'eligible'
])
def test_whatever_the_reply_and_the_voiceprint_say_the_emergency_stays_an_open_emergency(cfg: AppConfig, name: str, text: str, intent: str, eligible: bool, reasons: list[str]) -> None:
    s = scene(name)
    out = run(cfg, s, text)
    rec = out.inquiry["record"]  # type: ignore[index]
    assert rec["intent"] == intent and rec["cancel_eligible"] is eligible and rec["cancel_reasons"] == reasons
    iq = out.inquiry
    assert iq is not None
    for st in (iq["event_before"], iq["after_reply"], iq["after_timers"]):
        assert (st["level"], st["status"], st["outcome"]) == ("EMERGENCY", "OPEN", None)


def test_eligible_evidence_changes_nothing_compared_with_no_reply_at_all(cfg: AppConfig) -> None:
    """Differential: the same scripted fall; (a) the person answers 'I'm fine' in the enrolled voice, (b) nobody answers. Level, status, escalation and
    notifications must be identical now and after the escalation timers have run."""
    s = scene("fall_reassured")
    answered = run(cfg, s, "I'm fine, don't worry.")
    silent_plan = rp.InquiryPlan(s.inquiry["fall_at_s"], 15.0, 8.0)  # type: ignore[index]  # the listening window starts after the reply is over: only the hum
    unanswered = run(cfg, s, "I'm fine, don't worry.", plan=silent_plan)
    a, u = answered.inquiry, unanswered.inquiry
    assert a is not None and u is not None
    assert a["record"]["cancel_eligible"] is True
    assert u["record"]["status"] == "NO_RESPONSE" and u["record"]["cancel_eligible"] is False and u["record"]["cancel_reasons"] == ["no_speech"]  # silence is never 'safe'
    assert state(a["after_timers"]) == state(u["after_timers"]) and state(a["after_reply"]) == state(u["after_reply"])
    assert a["after_timers"]["level"] == "EMERGENCY" and a["after_timers"]["status"] == "OPEN"


def test_pending_event_is_not_lowered_by_a_verified_reassurance_but_a_help_reply_raises_it(cfg: AppConfig) -> None:
    s = scene("fall_reassured")
    s_help = scene("fall_help_call")
    reassured = run(cfg, s, "I'm fine, don't worry.", plan=plan_of(s, level="pending"))
    silent = run(cfg, s, None, plan=rp.InquiryPlan(4.0, 15.0, 8.0, level="pending"))
    helped = run(cfg, s_help, "Help me, I fell and I cannot get up.", plan=plan_of(s_help, level="pending"))
    r, q, h = reassured.inquiry, silent.inquiry, helped.inquiry
    assert r is not None and q is not None and h is not None
    assert r["record"]["cancel_eligible"] is True and r["event_before"]["level"] == "PENDING"
    assert r["after_reply"]["level"] == "PENDING" and r["after_reply"]["status"] == "OPEN"  # not cancelled, not downgraded
    assert state(r["after_timers"]) == state(q["after_timers"])  # the verification escalation runs exactly as without any reply
    assert h["record"]["intent"] == "HELP" and h["after_reply"]["level"] == "EMERGENCY"  # existing behaviour stays: an explicit HELP raises PENDING


# ------------------------------------------------------------------------------------------------ privacy: no raw audio anywhere
def signatures(x: np.ndarray, at_s: float, n: int = 160) -> list[bytes]:
    i = int(at_s * SR)
    seg = np.asarray(x[i:i + n], dtype=np.float32)
    return [seg.tobytes(), seg.astype(np.float64).tobytes(), np.round(seg * 32767).astype("<i2").tobytes()]


def test_no_raw_audio_is_stored_anywhere(cfg: AppConfig, system: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.chdir(tmp_path)
    caplog.set_level("DEBUG")
    s = scene("fall_help_call")
    asr = ScriptedASR("Help me, I fell and I cannot get up.")
    seen: list[np.ndarray] = []
    orig = asr.transcribe

    def spy(audio: np.ndarray, language: str | None = None, **kw: Any) -> dict:
        seen.append(audio)
        assert float(np.max(np.abs(audio))) > 0.01  # the service really got the microphone window
        return orig(audio, language, **kw)

    asr.transcribe = spy  # type: ignore[method-assign]
    out = run(cfg, s, None, system, asr=asr)
    assert out.events and out.inquiry is not None
    # 1. the sound_events table has no column that could hold audio (or anything derived from it)
    from sqlalchemy import inspect

    cols = inspect(system.db.engine).get_columns("sound_events")
    assert {c["name"] for c in cols} >= {"category", "confidence", "t_start", "t_end", "duration_s", "rms_db", "peak_db", "n_windows"}
    for c in cols:
        assert type(c["type"]).__name__.upper() not in ("BLOB", "LARGEBINARY", "BINARY", "VARBINARY", "JSON", "TEXT", "PICKLETYPE"), c
        assert not any(w in c["name"].lower() for w in ("audio", "wave", "sample", "spectro", "embedding", "transcript", "pcm"))
    # 2. every file the run left (the private database) is free of the scene's samples, scene-wide, in three encodings; the search itself works
    sigs = [sg for t in (4.2, 5.7, 9.6, 10.5) for sg in signatures(s.audio, t)]
    probe = tmp_path / "probe.bin"
    probe.write_bytes(b"xx" + sigs[0] + b"yy")
    assert sigs[0] in probe.read_bytes()
    files = [Path(system._tmp.name) / f for f in __import__("os").listdir(system._tmp.name)]
    assert files and all(f.suffix in (".db", ".db-wal", ".db-shm", "") or f.name.endswith(("-wal", "-shm")) for f in files), files
    for f in files:
        blob = f.read_bytes()
        assert not any(sg in blob for sg in sigs), f.name
    leftovers = sorted(p.name for p in tmp_path.iterdir())
    assert leftovers == ["probe.bin"], leftovers  # nothing was written to the working directory
    # 3. payloads: labels and statistics only
    payload = json.dumps(out.to_dict())
    d = json.loads(payload)
    assert d["raw_audio_stored"] is False and d["pipeline_status"]["raw_audio_stored"] is False
    allowed_event_keys = {f for f in SoundEvent.__dataclass_fields__} | {"t_start_s", "t_end_s"}
    assert all(set(e) == allowed_event_keys for e in d["sound_events"])
    assert all(set(g) == {"t_start_utc", "t_end_utc", "t_start_mono_ms", "t_end_mono_ms", "duration_s", "truncated"} for g in d["speech_segments"])
    assert not any(isinstance(v, (list, tuple)) and len(v) > 32 for v in d["inquiry"]["record"].values())
    keys: set[str] = set()
    big: list[int] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            keys.update(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            big.append(len(o))
            for v in o:
                walk(v)

    walk(d)
    assert not keys & {"audio", "samples", "waveform", "pcm", "embedding", "spectrogram", "transcript_audio"} and max(big) <= 64  # no key or array that could carry a signal
    detail = json.dumps(system.rt.events.detail(out.inquiry["event_before"]["event_id"]), default=str)
    assert "embedding" not in detail
    # 4. memory: the microphone window handed to the ASR is zeroed again, the speech segments never carried samples in their metadata
    assert len(seen) == 1 and not np.any(seen[0])
    # 5. logs never carry samples
    for r in caplog.records:
        assert not any(isinstance(a, np.ndarray) for a in (r.args if isinstance(r.args, tuple) else ()))
        assert len(r.getMessage()) < 1000


def test_the_ring_buffer_and_speech_segments_hold_audio_only_in_memory_and_are_wiped(cfg: AppConfig) -> None:
    from homeease.core.clock import VirtualClock
    from homeease.ml.audio.monitor import SoundMonitor
    from homeease.ml.audio.pipeline import Function3Pipeline

    s = scene("fall_help_call")
    clock = VirtualClock(1_000_000.0)
    got: list[Any] = []
    pipe = Function3Pipeline(cfg.audio, clock, SoundMonitor(cfg.audio.sound_monitor, StubSoundClassifier(), clock), voice_sink=got.append, vad=HarmonicStubVAD())
    rep = pipe.process_array(s.audio, SR, t0_mono_ms=1_000_000.0, flush=False)
    assert got and all("samples" not in repr(g) and "array(" not in repr(g) for g in got) and all(set(m) == set(got[0].meta()) for m in rep.speech)
    assert pipe.ring.buf.any()  # audio is held for the shortest necessary time ...
    pipe.close()
    assert not pipe.ring.buf.any() and pipe.status()["raw_audio_stored"] is False  # ... and wiped on stop


# ------------------------------------------------------------------------------------------------ failures are degraded, never 'safe'
def test_no_sound_model_is_a_degraded_status_and_no_invented_labels(cfg: AppConfig) -> None:
    s = scene("fall_reassured")
    out = run(cfg, s, "I'm fine, don't worry.", classifier=None)
    assert out.sound_status["status"] == "degraded:model_missing" and out.pipeline_status["status"] == "degraded:model_missing"
    assert out.events == [] and out.daily_summary["total"] == 0  # no model: no event, in particular no DAILY / 'all quiet'
    assert out.inquiry is not None and out.inquiry["after_timers"]["level"] == "EMERGENCY"  # the voice branch and the event are unaffected


def test_a_failing_classifier_or_vad_or_asr_surfaces_as_degraded_never_as_safe(cfg: AppConfig) -> None:
    class Boom:
        model_id, classes, window_s = "boom", ["HELP", "IMPACT", "ABNORMAL", "DAILY"], 2.0

        def predict_proba(self, w: np.ndarray) -> np.ndarray:
            raise RuntimeError("model crashed")

    class BrokenVAD:
        name = "broken"

        def frames(self, x: np.ndarray, sr: int = SR) -> np.ndarray:
            raise RuntimeError("vad crashed")

    s = scene("fall_reassured")
    out = run(cfg, s, "I'm fine, don't worry.", classifier=Boom())
    assert out.sound_status["status"] == "degraded:classifier_error" and out.events == [] and out.sound_status["classifier_errors"] > 0
    out = run(cfg, s, "I'm fine, don't worry.", vad=BrokenVAD())
    assert out.pipeline_status["voice"]["status"] == "degraded:vad_error" and out.pipeline_status["status"] == "degraded:vad_error" and out.speech == []
    assert any(e["category"] == "IMPACT" for e in out.events)  # a broken speech VAD does not take the sound branch down
    out = run(cfg, s, None, asr=ScriptedASR(None, error=RuntimeError("asr down")))
    rec = out.inquiry["record"]  # type: ignore[index]
    assert rec["status"] == "UNCERTAIN" and rec["intent"] == "UNCERTAIN" and rec["cancel_eligible"] is False and rec["cancel_reasons"] == ["asr_failed"]
    assert out.inquiry["after_timers"]["level"] == "EMERGENCY" and out.inquiry["after_timers"]["status"] == "OPEN"  # type: ignore[index]


def test_without_an_enrollment_the_evidence_is_false_with_a_reason(cfg: AppConfig) -> None:
    out = run(cfg, scene("fall_reassured"), "I'm fine, don't worry.", enrolled=False)
    rec = out.inquiry["record"]  # type: ignore[index]
    assert rec["cancel_eligible"] is False and "no_enrollment" in rec["cancel_reasons"] and rec["speaker_verified"] is None


def test_replay_is_deterministic(cfg: AppConfig) -> None:
    s = scene("fall_help_call")
    runs = [run(cfg, s, "Help me, I fell and I cannot get up.") for _ in range(2)]
    key = lambda o: [(e["category"], e["t_start_s"], e["t_end_s"], e["confidence"], e["n_windows"], e["rms_db"]) for e in o.events]  # noqa: E731
    assert key(runs[0]) == key(runs[1]) and key(runs[0])
    assert {k: v for k, v in runs[0].inquiry["record"].items() if k != "speaker_score"} == {k: v for k, v in runs[1].inquiry["record"].items() if k != "speaker_score"}  # type: ignore[index]


# ------------------------------------------------------------------------------------------------ the policy switches the figure's auto-cancel box would need stay impossible
def test_the_auto_cancel_and_raw_audio_switches_cannot_be_turned_on() -> None:
    with pytest.raises(ValueError, match="auto_cancel_emergency"):
        AudioCfg(auto_cancel_emergency=True)
    with pytest.raises(ValueError, match="persist_raw_audio"):
        PrivacyCfg(persist_raw_audio=True)


def test_the_differential_check_really_detects_an_auto_cancel(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation check kept as a test: if the event service DID resolve an event on `cancel_eligible`, the differential end-to-end test above must fail."""
    from homeease.backend.services.events import EventService
    from homeease.backend.storage.models import Event

    orig = EventService.voice_response

    def mutated(self: EventService, event_id: str, rec: dict[str, Any]) -> Any:
        e = orig(self, event_id, rec)
        if rec.get("cancel_eligible"):
            with self.db.tx() as ses:
                ses.get(Event, event_id).lifecycle_status = "RESOLVED"  # type: ignore[union-attr]
        return e

    monkeypatch.setattr(EventService, "voice_response", mutated)
    sysm = HeadlessSystem(cfg)
    try:
        with pytest.raises(AssertionError):
            test_scripted_scene_to_stored_labels_to_event_with_cancel_eligible_evidence(cfg, sysm)
    finally:
        sysm.close()
