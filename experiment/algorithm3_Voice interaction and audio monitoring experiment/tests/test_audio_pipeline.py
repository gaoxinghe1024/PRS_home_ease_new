"""Function 3 online pipeline: ring buffer, activity gate routing, timeline, status, and the privacy guarantees.

All offline and deterministic: synthetic scenes (tests/audio_fixtures.py), a stub classifier and a stub
VAD, a virtual clock. No microphone, no model download, no real TTS."""

from __future__ import annotations

import builtins
import json
import logging
import re
import sqlite3
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from homeease.backend.storage.db import Database
from homeease.core.clock import VirtualClock
from homeease.core.config import AppConfig, AudioCfg, PrivacyCfg, load_config
from homeease.core.paths import project_root
from homeease.ml.audio.monitor import SoundEventStore, SoundMonitor
from homeease.ml.audio.pipeline import AudioRing, Function3Pipeline, SpeechSegment
from tests.audio_fixtures import HarmonicStubVAD, StubSoundClassifier, impact, noise_bed, scene, speechlike, tone

SR = 16000
T0_UTC = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_cfg(**gate_or_monitor: Any) -> AudioCfg:
    return AudioCfg.model_validate({"vad": "energy", **gate_or_monitor})


def make_pipe(cfg: AudioCfg | None = None, clf: Any = "stub", sink: Any = None, vad: Any = None, store: Any = None, base_mono: float = 1000.0):
    cfg = cfg or make_cfg()
    clock = VirtualClock(base_mono, T0_UTC)
    clf = StubSoundClassifier() if clf == "stub" else clf
    mon = SoundMonitor(cfg.sound_monitor, clf, clock, store)
    pipe = Function3Pipeline(cfg, clock, mon, voice_sink=sink, vad=vad)
    return pipe, mon, clf, clock


def utc_offset_s(iso_ts: str) -> float:
    return (datetime.fromisoformat(iso_ts.replace("Z", "+00:00")) - T0_UTC).total_seconds()


# ------------------------------------------------------------------ ring buffer
def test_ring_buffer_wraps_and_clamps_to_what_is_still_buffered() -> None:
    ring = AudioRing(1.0, 100)  # 100 samples
    ring.append(np.arange(60, dtype=np.float32))
    ring.append(np.arange(60, 130, dtype=np.float32))  # wraps; samples 0..29 are gone
    assert ring.total == 130 and ring.oldest == 30
    assert np.array_equal(ring.get(100, 130), np.arange(100, 130))
    assert np.array_equal(ring.get(0, 130), np.arange(30, 130))  # clamped, never padded with fiction
    ring.append(np.arange(130, 500, dtype=np.float32))  # one chunk larger than the buffer
    assert np.array_equal(ring.get(400, 500), np.arange(400, 500)) and ring.oldest == 400
    ring.clear()
    assert ring.total == 0 and not ring.buf.any() and len(ring.get(0, 10)) == 0


# ------------------------------------------------------------------ gate routing
def test_gate_routes_speech_to_voice_and_everything_to_the_sound_branch() -> None:
    segs: list[SpeechSegment] = []
    pipe, mon, clf, _ = make_pipe(sink=segs.append, vad=HarmonicStubVAD())
    x, _ = scene(40, [(3.0, speechlike(3.0), "speech"), (15.0, impact(), "impact")])
    rep = pipe.process_array(x, SR, t0_mono_ms=1000.0)
    # VOICE branch: exactly the speech passage, nothing for the impact
    assert len(segs) == 1
    assert 2.7 <= (segs[0].t_start_mono_ms - 1000.0) / 1000.0 <= 3.4 and 5.8 <= (segs[0].t_end_mono_ms - 1000.0) / 1000.0 <= 6.8
    assert len(segs[0].samples) >= 2.9 * SR
    # SOUND branch: not gated by the speech VAD, so the impact is classified too
    cats = [(e.category, round(utc_offset_s(e.t_start_utc))) for e in rep.events]
    assert cats == [("DAILY", 3), ("IMPACT", 15)]
    st = rep.status
    assert st["status"] == "ok" and st["sound"]["windows_scored"] == clf.calls
    assert st["gate"]["idle_windows"] > 40 and st["gate"]["periodic_windows"] == 0  # silence is not scored


def test_sound_branch_does_not_depend_on_the_speech_vad() -> None:
    class NeverSpeech:
        name = "never"

        def frames(self, x: np.ndarray, sr: int = SR) -> np.ndarray:
            return np.zeros(len(x) // 160, dtype=np.int8)

    segs: list[SpeechSegment] = []
    pipe, _, _, _ = make_pipe(sink=segs.append, vad=NeverSpeech())
    x, _ = scene(20, [(8.0, impact(), "impact")])
    rep = pipe.process_array(x, SR, t0_mono_ms=1000.0)
    assert segs == [] and [e.category for e in rep.events] == ["IMPACT"]


def test_voice_branch_is_off_without_a_consumer_and_degraded_without_a_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, _, _, _ = make_pipe()
    assert pipe.gate.speech is None and pipe.status()["voice"]["status"] == "off"
    from homeease.ml.audio import vad as vadmod

    def boom(name: str) -> Any:
        raise ImportError("silero_vad is not installed")

    monkeypatch.setattr(vadmod, "create", boom)
    segs: list[SpeechSegment] = []
    pipe2, _, _, _ = make_pipe(sink=segs.append)
    x, _ = scene(20, [(3.0, speechlike(3.0), "speech"), (12.0, impact(), "impact")])
    rep = pipe2.process_array(x, SR, t0_mono_ms=1000.0)
    assert rep.status["voice"]["status"] == "degraded:vad_unavailable" and rep.status["status"] == "degraded:vad_unavailable"
    assert segs == []  # never pretends to have heard speech
    assert [e.category for e in rep.events] == ["DAILY", "IMPACT"]  # the sound branch keeps working


def test_energy_vad_produces_speech_segments_end_to_end() -> None:
    segs: list[SpeechSegment] = []
    pipe, _, _, _ = make_pipe(make_cfg(), sink=segs.append)  # audio.vad = "energy" through vad.create
    x, _ = scene(30, [(4.0, speechlike(2.5), "speech"), (14.0, speechlike(14.0), "long speech")])
    pipe.process_array(x, SR, t0_mono_ms=1000.0)
    assert pipe.status()["voice"]["vad"] == "energy"
    spans = [(round((s.t_start_mono_ms - 1000) / 1000, 1), round((s.t_end_mono_ms - 1000) / 1000, 1)) for s in segs]
    assert abs(spans[0][0] - 4.0) < 0.4 and abs(spans[0][1] - 6.5) < 0.8
    # long speech is cut into bounded segments (speech_max_s = 12 s) instead of one unbounded buffer
    assert all(s.duration_s <= 12.01 for s in segs) and len(segs) >= 3


def test_speech_segment_repr_and_meta_carry_no_samples() -> None:
    seg = SpeechSegment(1.0, 2.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", 1.0, SR, False, np.random.default_rng(0).standard_normal(SR).astype(np.float32))
    assert "samples" not in repr(seg) and "array" not in repr(seg)
    assert all(not isinstance(v, (np.ndarray, bytes)) for v in seg.meta().values())


def test_failing_voice_consumer_does_not_stop_the_pipeline(caplog: pytest.LogCaptureFixture) -> None:
    def bad_sink(seg: SpeechSegment) -> None:
        raise RuntimeError("consumer exploded")

    pipe, _, _, _ = make_pipe(sink=bad_sink, vad=HarmonicStubVAD())
    x, _ = scene(20, [(3.0, speechlike(2.0), "speech"), (12.0, impact(), "impact")])
    with caplog.at_level(logging.DEBUG):
        rep = pipe.process_array(x, SR, t0_mono_ms=1000.0)
    assert rep.status["voice"]["sink_errors"] == 1 and [e.category for e in rep.events] == ["DAILY", "IMPACT"]


# ------------------------------------------------------- onset / periodic gate
def test_single_click_is_not_an_activity_span_but_a_real_impact_is() -> None:
    pipe, mon, clf, _ = make_pipe()
    click = np.zeros(SR // 100, dtype=np.float32)
    click[:] = 0.5  # exactly one 10 ms frame
    x, _ = scene(20, [(5.0, click, "click")])
    rep = pipe.process_array(x, SR, t0_mono_ms=1000.0)
    assert clf.calls == 0 and rep.events == []


def test_periodic_window_samples_a_sound_the_onset_trigger_cannot_see() -> None:
    """A sound that is already on when the stream starts never rises above its own floor; the slow
    periodic window still scores it (a burst of confirm_windows consecutive windows)."""
    cfg = make_cfg(gate={"periodic_s": 5.0})
    pipe, mon, _, _ = make_pipe(cfg)
    x = (noise_bed(30, -55.0) + tone(30, 300, -30)).astype(np.float32)  # constant loud tone from t = 0
    rep = pipe.process_array(x, SR, t0_mono_ms=1000.0)
    st = rep.status
    assert st["gate"]["onset_windows"] == 0 and st["gate"]["periodic_windows"] >= 2
    assert st["sound"]["windows_by_trigger"] == {"periodic": st["gate"]["periodic_windows"]}
    assert rep.events and all(e.category == "DAILY" for e in rep.events)


def test_quiet_room_is_never_sampled_by_the_periodic_window() -> None:
    cfg = make_cfg(gate={"periodic_s": 5.0})
    pipe, _, clf, _ = make_pipe(cfg)
    rep = pipe.process_array(noise_bed(60, -75.0), SR, t0_mono_ms=1000.0)  # below audio.gate.min_level_dbfs
    assert clf.calls == 0 and rep.events == [] and rep.status["gate"]["quiet_periodic_skips"] >= 2


def test_gate_hysteresis_cooldown_and_merge_through_the_pipeline() -> None:
    pipe, mon, _, _ = make_pipe()  # cooldown_s = 20
    x, _ = scene(80, [(10.0, impact(seed=1), "i1"), (16.0, impact(seed=2), "i2"), (60.0, impact(seed=3), "i3")])
    rep = pipe.process_array(x, SR, t0_mono_ms=1000.0)
    imp = [e for e in rep.events if e.category == "IMPACT"]
    assert len(imp) == 2  # the 6 s repeat is merged into the first event, the one 44 s later is new
    assert imp[0].n_windows >= 6 and 15.5 < utc_offset_s(imp[0].t_end_utc) < 16.5
    assert abs(utc_offset_s(imp[1].t_start_utc) - 60.0) < 0.5


# --------------------------------------------------------------- timeline
def test_timestamps_follow_the_virtual_clock_not_the_wall_clock() -> None:
    pipe, _, _, clock = make_pipe(base_mono=5_000_000.0)
    x, truth = scene(30, [(12.0, impact(), "impact"), (20.0, tone(3.0, 600, -25), "alarm")])
    rep = pipe.process_array(x, SR, t0_mono_ms=5_000_000.0)
    by_cat = {e.category: e for e in rep.events}
    assert abs(utc_offset_s(by_cat["IMPACT"].t_start_utc) - 12.0) < 0.2
    assert abs(utc_offset_s(by_cat["ABNORMAL"].t_start_utc) - 20.0) < 0.2 and abs(utc_offset_s(by_cat["ABNORMAL"].t_end_utc) - 23.0) < 0.2
    for e in rep.events:
        assert abs((e.t_start_mono_ms - 5_000_000.0) / 1000.0 - utc_offset_s(e.t_start_utc)) < 1e-6
    assert rep.status["stream"]["last_chunk_utc"] == "2026-03-01T12:00:30Z"
    assert clock.mono_ms() == 5_000_000.0  # the pipeline never moves the clock it was given


def test_chunk_size_does_not_change_the_result() -> None:
    x, _ = scene(30, [(7.0, impact(), "impact"), (15.0, tone(3.0, 1100, -22), "help")])
    ref = None
    for chunk_s in (0.02, 0.1, 0.37):
        pipe, _, _, _ = make_pipe()
        rep = pipe.process_array(x, SR, t0_mono_ms=1000.0, chunk_s=chunk_s)
        got = [(e.category, round(e.t_start_mono_ms), round(e.t_end_mono_ms)) for e in rep.events]
        ref = ref or got
        assert got == ref and len(got) == 2
    pipe, _, _, _ = make_pipe()  # one chunk of 30 s is processed in bounded pieces
    pipe.process_chunk(x, 1000.0)
    pipe.flush()
    assert [(e.category, round(e.t_start_mono_ms), round(e.t_end_mono_ms)) for e in pipe.monitor.recent] == ref  # type: ignore[union-attr]


def test_timestamp_jitter_is_not_a_discontinuity_but_a_gap_is() -> None:
    x, _ = scene(40, [(10.0, impact(), "impact"), (30.0, impact(seed=5), "impact2")])
    rng = np.random.default_rng(3)
    pipe, mon, _, _ = make_pipe()
    n = SR // 10
    for k, i in enumerate(range(0, len(x), n)):
        t = 1000.0 + i * 1000.0 / SR + float(rng.uniform(-8, 8))  # +-8 ms capture-time jitter
        if i >= 20 * SR:
            t += 6000.0  # the driver dropped 6 s: audio before and after is not contiguous
        pipe.process_chunk(x[i:i + n], t)
    pipe.flush()
    st = pipe.status()
    assert st["stream"]["discontinuities"] == 1
    ev = {round(e.t_start_mono_ms / 1000.0): e for e in mon.recent}
    assert 11 in ev or 10 in ev  # before the gap: placed by the original timeline (1000 ms + 10 s)
    after = [e for e in mon.recent if e.t_start_mono_ms > 20_000]
    assert after and abs((after[0].t_start_mono_ms - 7000.0) / 1000.0 - 30.0) < 0.3  # after the gap: re-anchored on the new timestamps


def test_non_finite_and_integer_input_is_handled() -> None:
    pipe, _, _, _ = make_pipe()
    x, _ = scene(20, [(8.0, impact(), "impact")])
    x16 = (x * 32767).astype(np.int16)
    x16f = x16.astype(np.float32) / 32768.0
    bad = x16f.copy()
    bad[1000:1100] = np.nan
    bad[2000:2010] = np.inf
    rep = pipe.process_array(bad, SR, t0_mono_ms=1000.0)
    assert rep.status["stream"]["bad_samples"] == 110 and [e.category for e in rep.events] == ["IMPACT"]
    pipe2, _, _, _ = make_pipe()
    rep2 = pipe2.process_array(x16, SR, t0_mono_ms=1000.0)  # int16 PCM is scaled, not mistaken for huge floats
    assert [e.category for e in rep2.events] == ["IMPACT"]


def test_other_sample_rates_and_stereo_are_resampled_for_replay() -> None:
    from homeease.ml.audio.features import resample

    pipe, _, _, _ = make_pipe()
    x, _ = scene(20, [(8.0, tone(3.0, 600, -25), "alarm")])
    x44 = resample(x, SR, 44100)
    stereo = np.stack([x44, x44], axis=1)
    rep = pipe.process_array(stereo, 44100, t0_mono_ms=1000.0)
    assert [e.category for e in rep.events] == ["ABNORMAL"] and abs(utc_offset_s(rep.events[0].t_start_utc) - 8.0) < 0.3
    with pytest.raises(ValueError):
        pipe.process_chunk(np.zeros((100, 2), dtype=np.float32))  # live chunks are mono 16 kHz by contract


# ------------------------------------------------------------------ status
def test_model_missing_is_degraded_and_emits_nothing() -> None:
    pipe, mon, _, _ = make_pipe(clf=None)
    x, _ = scene(20, [(8.0, impact(), "impact")])
    rep = pipe.process_array(x, SR, t0_mono_ms=1000.0)
    assert rep.events == []
    assert rep.status["status"] == "degraded:model_missing" and rep.status["sound"]["status"] == "degraded:model_missing"
    assert rep.status["sound"]["windows_skipped_no_model"] > 0 and rep.status["sound"]["events_emitted"] == 0


def test_failing_classifier_degrades_status_and_recovers() -> None:
    class Flaky(StubSoundClassifier):
        fail = True

        def predict_proba(self, window: np.ndarray) -> np.ndarray:
            if self.fail:
                raise RuntimeError("onnx session crashed")
            return super().predict_proba(window)

    clf = Flaky()
    pipe, mon, _, _ = make_pipe(clf=clf)
    x, _ = scene(40, [(8.0, impact(), "impact"), (28.0, impact(seed=4), "impact")])
    n = SR // 10
    for i in range(0, len(x), n):
        if i == 20 * SR:
            assert pipe.status()["status"] == "degraded:classifier_error"
            clf.fail = False
        pipe.process_chunk(x[i:i + n], 1000.0 + i * 1000.0 / SR)
    pipe.flush()
    st = pipe.status()
    assert st["status"] == "ok" and st["sound"]["classifier_errors"] > 0
    assert [e.category for e in mon.recent] == ["IMPACT"] and abs(utc_offset_s(mon.recent[0].t_start_utc) - 28.0) < 0.3  # nothing invented for the failed stretch


# --------------------------------------------------------------- policy
def test_labels_and_speech_cannot_touch_events_by_construction() -> None:
    """Invariant 1: nothing in the audio front end or the monitor can close/downgrade/cancel an event."""
    for mod in ("monitor", "pipeline", "live"):
        src = (project_root() / "src" / "homeease" / "ml" / "audio" / f"{mod}.py").read_text(encoding="utf-8")
        assert "services.events" not in src and "EventService" not in src and "backend.services.notifications" not in src, mod
    with pytest.raises(ValidationError):
        AudioCfg(auto_cancel_emergency=True)


# ------------------------------------------------------------------ PRIVACY
WHITELIST = {"id", "subject_id", "room_id", "source_mode", "category", "confidence", "t_start", "t_end", "t_start_mono_ms", "t_end_mono_ms", "duration_s", "rms_db", "peak_db", "n_windows", "model_id", "created_at", "updated_at"}


def _files(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file()} if root.is_dir() else {}


def test_privacy_no_raw_audio_is_persisted_logged_or_kept(tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    soundfile = pytest.importorskip("soundfile", reason="needs the optional audio extra (uv sync --extra audio)")

    db_dir = tmp_path / "db"
    db = Database(f"sqlite:///{db_dir}/privacy.db")
    db.migrate()
    db.dispose()
    base_size = sum(_files(db_dir).values())
    db = Database(f"sqlite:///{db_dir}/privacy.db")
    watched = [project_root() / "var", project_root() / "data" / "captures", project_root() / "data" / "synthetic"]
    before = {str(w): set(_files(w)) for w in watched}

    # tripwires: any attempt to write audio through the usual routes fails the test
    def forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError("audio must never be written to disk")

    real_open = builtins.open

    def guarded_open(file: Any, mode: str = "r", *a: Any, **k: Any) -> Any:
        if any(c in str(mode) for c in "wax+"):
            forbidden()
        return real_open(file, mode, *a, **k)

    # the stores under test
    store = SoundEventStore(db, "elder-01", "living-room", clock=VirtualClock(1000.0, T0_UTC))
    segs: list[SpeechSegment] = []
    scn, _ = scene(30, [(3.0, speechlike(3.0), "speech"), (12.0, impact(), "impact"), (18.0, tone(3.0, 600, -25), "alarm"), (25.0, tone(2.0, 1100, -20), "help")])
    cfg = make_cfg()
    pipe, mon, clf, clock = make_pipe(cfg, sink=segs.append, vad=HarmonicStubVAD(), store=store)
    caplog.set_level(logging.DEBUG)
    with monkeypatch.context() as m:
        m.setattr(soundfile, "write", forbidden)
        m.setattr(wave, "open", forbidden)
        m.setattr(np, "save", forbidden)
        m.setattr(np, "savez", forbidden)
        m.setattr(np, "savez_compressed", forbidden)
        m.setattr(builtins, "open", guarded_open)
        rep = pipe.process_array(scn, SR, t0_mono_ms=1000.0)
        # one failing component so that warning logs are produced too
        pipe.voice_sink = lambda seg: (_ for _ in ()).throw(RuntimeError(f"PAYLOAD-{seg.samples[:3]}"))
        pipe.process_array(scn[: 10 * SR], SR, t0_mono_ms=40_000.0)
    pipe.close()
    assert [e.category for e in rep.events] == ["DAILY", "IMPACT", "ABNORMAL", "HELP"] and segs
    db.dispose()

    # (a) rows: only whitelisted scalar columns, nothing audio-sized anywhere in the database files
    con = sqlite3.connect(db_dir / "privacy.db")
    cols = [r[1] for r in con.execute("PRAGMA table_info(sound_events)")]
    assert set(cols) == WHITELIST
    rows = con.execute("SELECT * FROM sound_events").fetchall()
    con.close()
    assert len(rows) == 5  # 4 events of the 30 s scene + the speech of the 10 s replay that provoked the consumer failure
    assert all(v is None or isinstance(v, (int, float, str)) for r in rows for v in r)
    assert sum(_files(db_dir).values()) - base_size < 64_000  # 30 s of float32 audio would be 1.9 MB

    # (b) no file other than the database was created anywhere the test controls (or in the project's data/var dirs)
    assert {Path(k).name for k in _files(tmp_path)} <= {"privacy.db", "privacy.db-wal", "privacy.db-shm"}
    for w in watched:
        assert set(_files(w)) == before[str(w)], w

    # (c) no ndarray/bytes field in SoundEvent or the row model
    import dataclasses
    import typing

    from sqlalchemy import Float, Integer, String

    from homeease.backend.storage.models import SoundEventRecord, UTCDateTime
    from homeease.ml.audio.monitor import SoundEvent

    hints = typing.get_type_hints(SoundEvent)
    assert {f.name for f in dataclasses.fields(SoundEvent)} == set(hints) and set(hints.values()) <= {str, float, int}
    assert all(isinstance(c.type, (String, Float, Integer, UTCDateTime)) for c in SoundEventRecord.__table__.columns)
    assert {c.name for c in SoundEventRecord.__table__.columns} == WHITELIST
    ev = rep.events[0].to_dict()
    assert all(isinstance(v, (str, float, int)) for v in ev.values()) and json.dumps(ev)

    # (d) captured logs contain no sample arrays (neither in messages nor in arguments)
    assert caplog.records, "the failing consumer must have produced a warning"
    num_run = re.compile(r"(-?\d+\.\d+(e-?\d+)?[\s,]+){5,}")
    for rec in caplog.records:
        msg = rec.getMessage()
        assert "array(" not in msg and not num_run.search(msg), msg
        assert not any(isinstance(a, (np.ndarray, bytes, bytearray)) for a in (rec.args if isinstance(rec.args, tuple) else (rec.args,)))
    assert "PAYLOAD" not in caplog.text  # the exception *type* is logged, never its message (which could carry samples)

    # audio held in memory is zeroed on close
    assert not pipe.ring.buf.any() and len(pipe._carry) == 0

    # (e) the configuration cannot turn raw-audio persistence on
    with pytest.raises(ValidationError):
        PrivacyCfg(persist_raw_audio=True)
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"privacy": {"persist_raw_audio": True}})
    monkeypatch.setenv("HOMEEASE__PRIVACY__PERSIST_RAW_AUDIO", "true")
    with pytest.raises(ValidationError):
        load_config(None)
    assert PrivacyCfg().persist_raw_audio is False and rep.status["raw_audio_stored"] is False
