"""Optional voice inquiry after a suspected fall (feature-flagged, consent-gated).

Product rules (proposal 6.3, revised slide 1, prompt 9.1):
* Notifications are sent immediately; the inquiry never delays them.
* A reply is supporting evidence only. REASSURANCE never closes or downgrades an event.
  An explicit HELP reply raises PENDING to EMERGENCY (escalation is the safe direction).
* Unclear speech, ASR failure or silence is recorded as UNCERTAIN / NO_RESPONSE, never "safe".
* Listening starts only after the prompt finished playing (echo avoidance); prompt_id is kept.
* Raw audio is held only in memory for the inquiry window and then discarded.
* Consent is read again at the moment of use (when the event arrives and just before the microphone opens), not once at
  start-up: a withdrawal takes effect at once and a grant needs no restart. A second suspected fall while an inquiry is
  running gets a ``not_run: inquiry_busy`` marker instead of silently nothing.
* The reply transcript is not stored (``privacy.persist_voice_transcript``, default off): only intent, reason, cues.
* With a voiceprint enrollment the record also carries cancel-eligibility EVIDENCE (`cancel_eligible`, `speaker_score`,
  reason codes; ml/audio/cancel_gate.py). It is never acted on: `EventService.voice_response` does not read it, so an
  event keeps its level and status whatever it says (the proposal text, not Figure 3-3's auto-cancel box, is the policy).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from homeease.core.util import get_logger, new_id
from homeease.ml.audio import cancel_gate as cg

log = get_logger(__name__)

PROMPTS = {"en": "Do you need help? Please say yes or no.", "zh": "\u60a8\u9700\u8981\u5e2e\u52a9\u5417\uff1f\u8bf7\u56de\u7b54\u3002"}


class AudioInquiryService:
    cancel_gate: Any = None  # class defaults, so a partially constructed service (tests build one with __new__) simply has no voiceprint
    enrollment: Any = None

    def __init__(self, runtime: Any, recorder: Any = None, asr: Any = None, tts: Any = None, cancel_gate: Any = None, enrollment: Any = None) -> None:
        self.rt = runtime
        self.cfg = runtime.cfg
        self._state: dict[str, Any] = {"enabled": bool(self.cfg.audio.enabled), "consent": False, "microphone": None, "last": None, "errors": []}
        self.recorder = recorder or self._default_recorder
        self.asr = asr
        self.tts = tts
        self.cancel_gate = cancel_gate  # ml.audio.cancel_gate.CancelGate; built from audio.voiceprint on first use
        self.enrollment = enrollment  # ml.audio.cancel_gate.VoiceprintEnrollment; in memory only (persisting a voiceprint is a privacy decision, not made here)
        self._busy = threading.Lock()
        self._check()

    # -------------------------------------------------------------- checks
    def _read_consent(self) -> bool:
        """The latest recorded audio consent, read now. Unknown (database error) is no consent."""
        try:
            fn = getattr(self.rt, "audio_consent", None)
            if callable(fn):
                ok = bool(fn())
            else:  # a minimal runtime stand-in without the helper
                from sqlalchemy import select

                from homeease.backend.storage.models import Consent

                with self.rt.db.session() as s:
                    c = s.execute(select(Consent).where(Consent.subject_id == self.rt.subject_id, Consent.kind == "audio").order_by(Consent.created_at.desc(), Consent.id.desc())).scalars().first()
                    ok = bool(c and c.granted)
        except Exception as e:
            log.warning("audio consent could not be read: %s", type(e).__name__)
            ok = False
        self._state["consent"] = ok
        return ok

    def _check(self) -> None:
        self._read_consent()
        try:
            import sounddevice as sd

            self._state["microphone"] = any(d.get("max_input_channels", 0) > 0 for d in sd.query_devices())
        except Exception as e:  # pragma: no cover - hardware dependent
            self._state["microphone"] = False
            self._state["errors"].append(f"microphone check failed: {e}")

    def status(self) -> dict[str, Any]:
        self._read_consent()  # never report a consent that was withdrawn (or granted) after start-up
        return {**self._state, "auto_cancel_emergency": self.cfg.audio.auto_cancel_emergency, "voiceprint_enrolled": self.enrollment is not None}

    def set_enrollment(self, enrollment: Any) -> None:
        """Install (or with None remove) the enrolled voiceprint. Memory only."""
        self.enrollment = enrollment

    def stop(self) -> None:
        if self.asr is not None and hasattr(self.asr, "close"):
            self.asr.close()

    # ------------------------------------------------------------- default IO
    def _default_recorder(self, seconds: float, sr: int = 16000) -> np.ndarray:
        import sounddevice as sd

        x = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        return x[:, 0]

    def _speak(self, text: str, lang: str) -> bool:
        mode = self.cfg.audio.tts
        if self.tts is not None:
            return bool(self.tts(text, lang))
        if mode in ("system", "system_say"):
            # macOS say / Windows SAPI through one abstraction; text and voice never reach a command string
            from homeease.ml.audio import tts as tts_mod

            return tts_mod.play(text, lang)
        if mode == "browser":
            self.rt.broadcaster.publish("voice_prompt", {"text": text, "lang": lang}, self.rt.subject_id, audience="elder")
            time.sleep(4.0)  # approximate playback time on the elder device
            return True
        return False

    # ------------------------------------------------------------- inquiry
    def on_event(self, event_id: str, level: str) -> None:
        if not (self._state["enabled"] and self._read_consent()):
            self._attach(event_id, {"status": "not_run", "reason": "audio disabled or no consent"})
            return
        threading.Thread(target=self._run, args=(event_id,), name="voice-inquiry", daemon=True).start()

    def _run(self, event_id: str) -> None:
        if not self._busy.acquire(blocking=False):
            # one microphone, one inquiry at a time: the later event is marked instead of silently having no voice record
            self._attach(event_id, {"status": "not_run", "reason": "inquiry_busy"})
            return
        try:
            self.run_inquiry(event_id, require_consent=True)
        finally:
            self._busy.release()

    def _intent_model(self) -> Any:
        """Reply-intent method from the config. rule_v2 is the default (docs/results.md, A3b); "rule" keeps the original v1 rule."""
        if self.cfg.audio.intent_method == "rule":
            from homeease.ml.audio.intent import RuleIntent

            return RuleIntent()
        from homeease.ml.audio.intent_v2 import RuleIntentV2

        return RuleIntentV2()

    def run_inquiry(self, event_id: str, lang: str | None = None, require_consent: bool = False) -> dict[str, Any]:
        """Prompt, listen, recognise, attach. With ``require_consent`` (the production path) the consent is checked again
        before the prompt and again just before the microphone opens (the prompt takes seconds to play)."""
        prompt_id = new_id("pr_")
        lang = lang or (self.cfg.audio.languages[0] if self.cfg.audio.languages else "en")
        rec: dict[str, Any] = {"prompt_id": prompt_id, "lang": lang, "started_at": self.rt.clock.utc_now().isoformat()}
        if require_consent and not self._read_consent():
            rec.update(status="not_run", reason="audio consent withdrawn")
            self._attach(event_id, rec)
            return rec
        spoken = self._speak(PROMPTS.get(lang, PROMPTS["en"]), lang)
        rec["prompt_played"] = spoken
        if not spoken:
            rec["prompt_error"] = "TTS playback failed; screen prompt and alerts are unaffected"
        if require_consent and not self._read_consent():
            rec.update(status="not_run", reason="audio consent withdrawn before listening")
            self._attach(event_id, rec)
            return rec
        rec["listen_started_at"] = self.rt.clock.utc_now().isoformat()
        try:
            audio = self.recorder(self.cfg.audio.inquiry_response_window_s)
        except Exception as e:
            rec.update(status="audio_unavailable", error=str(e)[:200], **self._no_evidence(cg.AUDIO_UNAVAILABLE))
            self._attach(event_id, rec)
            return rec
        try:
            return self._handle_reply(event_id, rec, audio)
        finally:
            try:
                audio[:] = 0.0  # discard the raw samples on EVERY path (silence, ASR failure, reply), not only after a reply
            except Exception:  # pragma: no cover - a recorder that hands out a read-only buffer
                pass

    def _handle_reply(self, event_id: str, rec: dict[str, Any], audio: np.ndarray) -> dict[str, Any]:
        from homeease.ml.audio.intent_v2 import read_as_answer_to_help_prompt
        from homeease.ml.audio.vad import EnergyVAD, segments

        speech = segments(EnergyVAD().frames(audio))
        if not speech:
            rec.update(status="NO_RESPONSE", intent=None, **self._no_evidence(cg.NO_SPEECH))
            self._attach(event_id, rec)
            return rec
        try:
            if self.asr is None:
                from homeease.ml.audio.asr import ASRProcess

                self.asr = ASRProcess(f"faster-whisper-{self.cfg.audio.asr_model}", temperature=self.cfg.audio.asr_temperature)
            res = self.asr.transcribe(audio, None)
        except Exception as e:
            rec.update(status="UNCERTAIN", intent="UNCERTAIN", error=f"asr failed: {str(e)[:150]}", **self._no_evidence(cg.ASR_FAILED))
            self._attach(event_id, rec)
            return rec
        it = read_as_answer_to_help_prompt(res["text"], self._intent_model().predict(res["text"], res.get("avg_logprob")))
        rec.update(status="RESPONDED", intent=it["intent"], intent_reason=it["reason"], intent_method=self.cfg.audio.intent_method, transcript=(res["text"] or "")[:100], asr_model=res.get("model"),
                   asr_avg_logprob=res.get("avg_logprob"), source_note="speaker identity not verified (TV/visitor speech possible)")
        rec.update(self._cancel_evidence(audio, it, res.get("avg_logprob")))
        if rec.get("speaker_verified") is not None:
            rec["source_note"] = "voiceprint compared with the enrollment; evidence only, never cancels or downgrades the event (TV, replay and visitors can still pass or fail it)"
        self._attach(event_id, rec)
        return rec

    # ------------------------------------------------------------- cancel-eligibility evidence
    @staticmethod
    def _no_evidence(reason: str) -> dict[str, Any]:
        """Fields of a reply that never reached the voiceprint gate: explicitly NOT eligible, never missing."""
        return {"cancel_eligible": False, "cancel_reasons": [reason], "cancel_policy": cg.POLICY, "speaker_verified": None, "speaker_score": None}

    def _cancel_evidence(self, audio: np.ndarray, intent: dict[str, Any], asr_conf: float | None) -> dict[str, Any]:
        """Voiceprint match + reply intent -> cancel_eligible EVIDENCE. Without an enrollment it is False (no_enrollment) and the speaker model is not even loaded.
        Any failure is False with a reason code; nothing in the result is ever used to change an event."""
        try:
            if self.cancel_gate is None:
                self.cancel_gate = cg.CancelGate.from_config(self.cfg.audio)
            return self.cancel_gate.evaluate(audio, intent, self.enrollment, asr_conf).to_record()
        except Exception as e:  # pragma: no cover - the gate does not raise for model/audio problems; this is the last line of defence
            log.exception("cancel-eligibility evaluation failed")
            self._state["errors"].append(f"voiceprint gate: {str(e)[:150]}")
            return self._no_evidence(cg.GATE_ERROR)

    def _attach(self, event_id: str, rec: dict[str, Any]) -> None:
        self._state["last"] = {"event_id": event_id, **{k: rec.get(k) for k in ("status", "intent", "prompt_id")}}
        try:
            self.rt.events.voice_response(event_id, rec)
        except Exception as e:  # pragma: no cover
            log.exception("attaching voice response failed")
            self._state["errors"].append(str(e)[:200])
