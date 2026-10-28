"""Offline replay of audio through Function 3 (proposal Figure 3-3): no microphone, no human, no network.

    audio (WAV or generated scene) --> Function3Pipeline (ring buffer, front end, sound branch, speech segments)
                                          |--> SoundMonitor --> sound_events table (labels only)
    scripted suspected fall at t_fall ---> HeadlessSystem (virtual clock, private SQLite database)
    scripted inquiry at t_inquiry -------> AudioInquiryService.run_inquiry on the next ``window_s`` seconds of the SAME audio
                                           (ASR -> intent -> voiceprint -> ``cancel_eligible`` EVIDENCE) -> event record

The replay reuses the product code paths (``Function3Pipeline``, ``SoundMonitor``, ``SoundEventStore``,
``AudioInquiryService``, ``CancelGate``, ``EventService``); only the microphone is replaced by an array and the person by a
scene or a WAV file. What it shows is plumbing and policy: labels are stored without audio, and the scripted event keeps its
level and status whatever the reply and the voiceprint say. It is NOT an accuracy measurement.

The fall event is scripted on purpose: in this system a fall is detected by the camera and phone channels, never by audio.
Audio only adds supporting evidence and assistive labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from homeease.core.config import AppConfig
from homeease.core.util import get_logger
from homeease.ml.audio.cancel_gate import CancelGate, VoiceprintEnrollment
from homeease.ml.audio.features import SR, resample

log = get_logger(__name__)

DEMO_THRESHOLD = 0.686  # cosine; the FAR 0.1% point calibrated on 150 SYNTHETIC dev speakers in experiment A6 (docs/results.md 3.3). A demo value, not a household setting
VOICE_KEYS = ("status", "intent", "intent_reason", "intent_method", "transcript", "asr_model", "asr_avg_logprob", "lang", "prompt_played", "cancel_eligible", "cancel_reasons",
              "cancel_policy", "speaker_verified", "speaker_score", "voiceprint_threshold", "speech_s", "error", "source_note")
CHUNK_S = 0.1


@dataclass(frozen=True)
class InquiryPlan:
    """When the scripted suspected fall happens and when the microphone window of the voice inquiry opens (seconds into the audio)."""

    fall_at_s: float
    inquiry_start_s: float
    window_s: float = 10.0
    level: str = "emergency"  # emergency (explicit help request) | pending (fusion decision awaiting verification)
    after_s: float = 30.0  # virtual seconds the system keeps running after the inquiry (escalation timers)

    def __post_init__(self) -> None:
        if self.level not in ("emergency", "pending"):
            raise ValueError("level must be 'emergency' or 'pending'")
        if self.inquiry_start_s < self.fall_at_s:
            raise ValueError("the inquiry cannot start before the fall")


@dataclass
class VoiceStack:
    """Injectable parts of the voice branch. ``None`` means the product default (faster-whisper tiny, WeSpeaker)."""

    asr: Any | None = None  # object with ``transcribe(audio, language) -> {"text", "avg_logprob", "model"}``
    embedder: Any | None = None  # speaker model with ``embed(audio16k)`` and ``model_id``
    enrollment: VoiceprintEnrollment | None = None
    threshold: float | None = None  # None: audio.voiceprint.threshold (null by default: nothing is eligible)


@dataclass
class ReplayOutcome:
    duration_s: float
    t0_mono_ms: float  # monotonic time of the first replayed sample (event and speech times are on this axis)
    classes: list[str]
    sound_status: dict[str, Any]
    pipeline_status: dict[str, Any]
    events: list[dict[str, Any]]  # sound events as stored in (and read back from) the database: scalars only
    speech: list[dict[str, Any]]  # speech segments of the voice branch: metadata only
    daily_summary: dict[str, Any]
    truth_check: list[dict[str, Any]] = field(default_factory=list)
    unexplained: list[dict[str, Any]] = field(default_factory=list)
    inquiry: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe, labels and statistics only (``raw_audio_stored`` is stated, and the tests check it)."""
        return {"duration_s": round(self.duration_s, 3), "t0_mono_ms": self.t0_mono_ms, "classes": self.classes, "sound_status": self.sound_status, "pipeline_status": self.pipeline_status, "sound_events": self.events,
                "speech_segments": self.speech, "daily_summary": self.daily_summary, "truth_check": self.truth_check, "unexplained_events": self.unexplained, "inquiry": self.inquiry,
                "raw_audio_stored": False}


# ----------------------------------------------------------------------------- input
def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono float32 samples and the file's sample rate (any WAV/FLAC the soundfile library reads). Memory only."""
    import soundfile as sf

    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(axis=1).astype(np.float32), int(fs)


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return resample(x, sr, SR) if sr != SR else x.astype(np.float32)


# ----------------------------------------------------------------------------- truth check
def check_truth(truth: list[dict[str, Any]], events: list[dict[str, Any]], base_mono_ms: float, tol_s: float = 1.5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """For every scripted sound with an expected category: did the sound branch report that category near it?

    Returns (one row per expectation, events that explain no scripted sound). A demonstration of what the branch saw on THIS scene, not a metric."""
    rel = [(e, (e["t_start_mono_ms"] - base_mono_ms) / 1000.0, (e["t_end_mono_ms"] - base_mono_ms) / 1000.0) for e in events]
    rows: list[dict[str, Any]] = []
    for it in truth:
        if not it.get("expect"):
            continue
        lo, hi = float(it["t0_s"]) - tol_s, float(it["t1_s"]) + tol_s
        near = [e for e, s, t in rel if s <= hi and t >= lo]
        cats = sorted({e["category"] for e in near})
        res = "found" if it["expect"] in cats else ("other:" + "/".join(cats) if cats else "no event")
        rows.append({"kind": it["kind"], "t0_s": it["t0_s"], "t1_s": it["t1_s"], "expect": it["expect"], "result": res, "categories_near": cats})
    spans = [(float(it["t0_s"]) - tol_s, float(it["t1_s"]) + tol_s) for it in truth]
    unexplained = [{"category": e["category"], "t0_s": round(s, 2), "t1_s": round(t, 2), "confidence": e["confidence"]} for e, s, t in rel
                   if e["category"] in ("HELP", "IMPACT", "ABNORMAL") and not any(s <= hi and t >= lo for lo, hi in spans)]
    return rows, unexplained


# ----------------------------------------------------------------------------- the replay
def _event_state(sysm: Any, event_id: str) -> dict[str, Any]:
    d = sysm.rt.events.detail(event_id)
    return {"event_id": event_id, "level": d["decision_level"], "max_level": d["max_level"], "status": d["lifecycle_status"], "outcome": d["outcome"], "escalation_step": d["escalation_step"],
            "notifications": len(d["notifications"])}


def _create_event(sysm: Any, level: str, key: str) -> str:
    if level == "emergency":
        e = sysm.rt.events.help_request(sysm.rt.subject_id, "user", "1", key, "replay")
    else:
        from homeease.schemas.evidence import DecisionLevel, FusionDecision

        now = sysm.clock.mono_ms()
        d = FusionDecision(decision_id=key, subject_id=sysm.rt.subject_id, level=DecisionLevel.PENDING, decided_at_ms=now, anchor_time_ms=now - 3000, fusion_backend="late_quality_gated",
                           reason_codes=["vision_missing"], source_mode="replay")
        e = sysm.rt.events.ingest_decision(d)
        if e is None:
            raise RuntimeError("the scripted PENDING decision created no event")
    sysm.step(sysm.clock.mono_ms())
    return str(e.id)


def run_replay(
    audio: np.ndarray,
    sr: int,
    cfg: AppConfig,
    classifier: Any | None,
    *,
    vad: Any | None = None,
    plan: InquiryPlan | None = None,
    voice: VoiceStack | None = None,
    truth: list[dict[str, Any]] | None = None,
    on_progress: Callable[[str], None] | None = None,
    system: Any | None = None,
) -> ReplayOutcome:
    """Push ``audio`` through the pipeline in 0.1 s chunks on a virtual clock; optionally script a suspected fall and a voice inquiry.

    ``classifier`` None means no sound model: the sound branch reports ``degraded:model_missing`` and stores nothing (never an invented label).
    The audio is read, never written; the ring buffer is wiped at the end. The database is private and temporary, unless the caller passes its own
    ``HeadlessSystem`` as ``system`` (tests do, to inspect the database afterwards; the caller then closes it)."""
    from homeease.experiments.headless import HeadlessSystem
    from homeease.ml.audio.monitor import SoundEventStore, SoundMonitor
    from homeease.ml.audio.pipeline import Function3Pipeline

    say = on_progress or (lambda m: None)
    x = to_16k(audio, sr)
    voice = voice or VoiceStack()
    sysm = system if system is not None else HeadlessSystem(cfg)
    cfg = sysm.cfg  # the copy the runtime was built with
    try:
        clock = sysm.clock
        base = clock.mono_ms()
        store = SoundEventStore(sysm.db, sysm.rt.subject_id, sysm.rt.room_id, "replay", clock)
        monitor = SoundMonitor(cfg.audio.sound_monitor, classifier, clock, store)
        pipe = Function3Pipeline(cfg.audio, clock, monitor, voice_sink=lambda seg: None, vad=vad)  # the sink only keeps the voice branch (speech VAD) switched on; segments are logged as metadata

        n = len(x)
        grid = int(CHUNK_S * SR)
        k_fall = k_inq = None
        if plan is not None:
            k_fall = min(int(round(plan.fall_at_s * SR / grid)) * grid, n)
            k_inq = min(max(int(round(plan.inquiry_start_s * SR / grid)) * grid, k_fall), n)
        cuts = [0] + [k for k in (k_fall, k_inq) if k is not None] + [n]
        inquiry: dict[str, Any] | None = None
        event_id: str | None = None
        before: dict[str, Any] | None = None
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b > a:
                pipe.process_array(x[a:b], SR, t0_mono_ms=base + a * 1000.0 / SR, chunk_s=CHUNK_S, flush=False)
            if plan is not None and b == k_fall and event_id is None:
                clock.set(max(clock.mono_ms(), base + plan.fall_at_s * 1000.0))
                event_id = _create_event(sysm, plan.level, "replay-fall-1")
                before = _event_state(sysm, event_id)
                say(f"t={plan.fall_at_s:.1f}s  scripted suspected fall -> event {event_id} {before['level']}/{before['status']}")
            if plan is not None and b == k_inq and event_id is not None and inquiry is None:
                clock.set(max(clock.mono_ms(), base + plan.inquiry_start_s * 1000.0))
                inquiry = _run_inquiry(sysm, cfg, x, k_inq, plan, voice, event_id, before or {})
                say(f"t={plan.inquiry_start_s:.1f}s  voice inquiry -> {inquiry['record'].get('status')} intent={inquiry['record'].get('intent')} cancel_eligible={inquiry['record'].get('cancel_eligible')}")
        pipe.flush()  # end of stream: settle the speech VAD, write the open sound events
        report = pipe.process_array(np.zeros(0, dtype=np.float32), SR, t0_mono_ms=base, flush=False)  # collects events and speech metadata of the whole run
        if inquiry is not None and plan is not None and event_id is not None:
            clock.set(max(clock.mono_ms(), base + (plan.inquiry_start_s + plan.window_s) * 1000.0))
            sysm.advance(2.0)
            inquiry["after_reply"] = _event_state(sysm, event_id)
            sysm.advance(plan.after_s)
            inquiry["after_timers"] = _event_state(sysm, event_id)
            inquiry["after_timers"]["virtual_s_waited"] = plan.after_s + 2.0
        stored = [{**e.to_dict(), "t_start_s": round((e.t_start_mono_ms - base) / 1000.0, 3), "t_end_s": round((e.t_end_mono_ms - base) / 1000.0, 3)}  # seconds into the replayed audio
                  for e in sorted(store.recent(1000), key=lambda e: e.t_start_mono_ms)]
        rows, unexplained = check_truth(truth or [], stored, base) if truth else ([], [])
        outcome = ReplayOutcome(duration_s=n / SR, t0_mono_ms=base, classes=list(monitor.classes), sound_status=monitor.status(), pipeline_status=pipe.status(), events=stored, speech=report.speech,
                                daily_summary=store.daily_summary(now=clock.utc_now()), truth_check=rows, unexplained=unexplained, inquiry=inquiry)
        pipe.close()  # wipes the ring buffer
        return outcome
    finally:
        if system is None:
            sysm.close()


def _run_inquiry(sysm: Any, cfg: AppConfig, x: np.ndarray, k0: int, plan: InquiryPlan, voice: VoiceStack, event_id: str, before: dict[str, Any]) -> dict[str, Any]:
    """The product's inquiry service with the microphone replaced by the next ``window_s`` seconds of the replayed audio."""
    from homeease.ml.audio.inquiry import AudioInquiryService

    window = x[k0:k0 + int(plan.window_s * SR)].copy()

    def recorder(seconds: float, sr: int = SR) -> np.ndarray:
        return window[: int(seconds * sr)].copy()  # the service zeroes what it is given, on every path

    gate = CancelGate.from_config(cfg.audio, voice.embedder)
    if voice.threshold is not None:
        gate.threshold = voice.threshold  # explicit demo / calibration value; the configured default is null (nothing eligible)
    svc = AudioInquiryService(sysm.rt, recorder=recorder, asr=voice.asr, tts=lambda text, lang: True, cancel_gate=gate, enrollment=voice.enrollment)
    try:
        rec = svc.run_inquiry(event_id)
    finally:
        window[:] = 0.0
        svc.stop()
    return {"event_before": before, "record": {k: rec.get(k) for k in VOICE_KEYS if k in rec}, "prompt": "simulated: nothing is played in a replay", "window_s": plan.window_s,
            "voiceprint_enrolled": voice.enrollment is not None, "threshold": gate.threshold, "policy": "evidence only: the reply and the voiceprint never close, downgrade or cancel an event"}


__all__ = ["DEMO_THRESHOLD", "InquiryPlan", "VoiceStack", "ReplayOutcome", "run_replay", "check_truth", "load_wav", "to_16k"]
