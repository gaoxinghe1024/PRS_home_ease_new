"""`homeease audio ...`: the Function 3 (voice interaction and audio monitoring) demo without a microphone or a human.

* ``audio scenes``  generates scripted, seeded scenes (ESC-50 background, sound clips, TTS replies) into ``data/synthetic/audio_scenes/``.
* ``audio replay``  pushes a WAV file or a named scene through ``Function3Pipeline`` offline, prints the labelled sound events (labels
  only) and, with ``--inquiry``, runs a scripted suspected fall and voice inquiry and shows that the event keeps its level whatever the
  reply and the voiceprint say. A plumbing and policy demonstration, not an accuracy measurement.

Nothing here opens a microphone, writes recorded audio or needs the network. Missing data or models give a message that names what to run.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

from homeease.core.config import load_config
from homeease.core.paths import data_dir, resolve
from homeease.core.util import dumps, setup_logging

EXIT_MISSING_INPUT = 3
DEFAULT_SCENE_DIR = "data/synthetic/audio_scenes"


def _safe_console() -> None:
    """A transcript can hold characters the Windows code page of a redirected stdout cannot encode (a music note, Hangul): print them
    escaped instead of crashing after the whole replay. A real console is unaffected; the encoding is not changed."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # not a text stream that can be reconfigured
            pass


def _labels_only(payload: dict[str, Any]) -> dict[str, Any]:
    """What --json-out may contain: labels, reasons, numbers. The text of the reply is personal content: only its length is kept."""
    iq = payload.get("inquiry")
    if isinstance(iq, dict) and isinstance(iq.get("record"), dict):
        rec = dict(iq["record"])
        text = rec.pop("transcript", None)
        rec["transcript_chars"] = len(text or "")
        payload = {**payload, "inquiry": {**iq, "record": rec}}
    return payload


def _scene_dir(arg: str | None) -> Path:
    d = resolve(arg or DEFAULT_SCENE_DIR)
    assert d is not None
    return d


# ------------------------------------------------------------------------------- scenes
def cmd_audio_scenes(args: argparse.Namespace) -> int:
    from homeease.ml.audio import scenes as sc

    _safe_console()

    if args.list:
        for name, spec in sc.SCENES.items():
            print(f"{name:<20} {spec.duration_s:>4.0f} s  {spec.description}")
        return 0
    unknown = [n for n in (args.scene or []) if n not in sc.SCENES]
    if unknown:
        print(f"unknown scene(s): {', '.join(unknown)}; available: {', '.join(sc.SCENE_NAMES)}", file=sys.stderr)
        return EXIT_MISSING_INPUT
    try:
        catalog = sc.DiskCatalog(data_dir())
    except Exception as e:  # missing corpora are handled per scene; this is a broken manifest or an import problem
        print(f"cannot read the audio corpora under {data_dir() / 'raw'}: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_MISSING_INPUT
    out = _scene_dir(args.out)
    written, skipped = sc.generate_scenes(args.scene, args.seed, out, catalog)
    total = 0
    for p in written:
        total += p.stat().st_size
        print(f"wrote {p}  ({p.stat().st_size / 1e6:.2f} MB)  + {p.with_suffix('.json').name}")
    for name, why in skipped.items():
        print(f"skipped {name}: {why}", file=sys.stderr)
    if written:
        print(f"{len(written)} scene(s), {total / 1e6:.1f} MB in {out} (gitignored; seed {args.seed}; sources from the held-out fold of the sound model; synthetic data, a plumbing demo)")
    return 0 if written else EXIT_MISSING_INPUT


# ------------------------------------------------------------------------------- replay
def _load_classifier(cfg: Any, ref: str | None) -> tuple[Any | None, str | None]:
    from homeease.ml.audio.sound_package import load_sound_classifier
    from homeease.ml.registry import ModelLoadError

    try:
        return load_sound_classifier(ref or cfg.audio.sound_monitor.model_id), None
    except ModelLoadError as e:
        return None, str(e)[:300]


def _utterances(args: argparse.Namespace) -> list[Any] | None:
    """Enrollment audio: three TTS sentences in one voice, or three WAV files. Memory only; ``None`` when no enrollment was requested."""
    from homeease.ml.audio import replay as rp
    from homeease.ml.audio import scenes as sc

    if args.enroll_wav:
        if len(args.enroll_wav) < 3:
            raise ValueError("--enroll-wav needs at least three files (one voice, each with at least 1.5 s of speech)")
        return [rp.to_16k(*rp.load_wav(p)) for p in args.enroll_wav]
    if args.enroll_voice:
        return [sc.default_tts(t, args.enroll_voice, "en") for t in sc.ENROLL_TEXTS]
    return None


def _reply_audio(args: argparse.Namespace) -> Any | None:
    from homeease.ml.audio import replay as rp
    from homeease.ml.audio import scenes as sc

    if args.reply_wav:
        return rp.to_16k(*rp.load_wav(args.reply_wav))
    if args.reply_text:
        return sc.default_tts(args.reply_text, args.reply_voice, args.reply_lang)
    return None


def cmd_audio_replay(args: argparse.Namespace) -> int:
    import numpy as np

    from homeease.ml.audio import replay as rp
    from homeease.ml.audio import scenes as sc
    from homeease.ml.audio.cancel_gate import EnrollmentError, VoiceprintEnrollment

    setup_logging("WARNING", to_file=False)  # warnings (degraded model, VAD) stay visible; INFO is noise for a demo
    _safe_console()
    warnings.filterwarnings("ignore", message=".*torch.jit.load.*", category=FutureWarning)  # silero-vad loads its TorchScript model; the deprecation notice is not actionable here
    cfg = load_config(getattr(args, "config", None))
    if bool(args.wav) == bool(args.scene):
        print("give exactly one of --wav FILE or --scene NAME (see `homeease audio scenes --list`)", file=sys.stderr)
        return EXIT_MISSING_INPUT
    # ---- input audio and ground truth
    side: dict[str, Any] | None = None
    if args.scene:
        if args.scene not in sc.SCENES:
            print(f"unknown scene {args.scene!r}; available: {', '.join(sc.SCENE_NAMES)}", file=sys.stderr)
            return EXIT_MISSING_INPUT
        wav = _scene_dir(args.scenes_dir) / f"{args.scene}-s{args.seed}.wav"
        if not wav.is_file():
            print(f"scene file not found: {wav}\nrun: homeease audio scenes --scene {args.scene} --seed {args.seed}", file=sys.stderr)
            return EXIT_MISSING_INPUT
    else:
        wav = Path(args.wav)
        if not wav.is_file():
            print(f"audio file not found: {wav}", file=sys.stderr)
            return EXIT_MISSING_INPUT
    try:
        audio, sr = rp.load_wav(wav)
    except Exception as e:
        print(f"cannot read {wav}: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
        return EXIT_MISSING_INPUT
    audio = rp.to_16k(audio, sr)
    side = sc.read_sidecar(wav)
    truth = (side or {}).get("truth") or None
    # ---- scripted reply for a plain WAV (mixed into the audio so that both branches hear it)
    try:
        reply = _reply_audio(args)
    except (sc.SceneUnavailable, OSError, ValueError) as e:
        print(f"cannot prepare the scripted reply: {e}", file=sys.stderr)
        return EXIT_MISSING_INPUT
    plan = None
    inq = (side or {}).get("inquiry")
    if args.inquiry:
        if args.fall_at is not None:
            fall = float(args.fall_at)
            start = float(args.inquiry_at) if args.inquiry_at is not None else fall + sc.PROMPT_S
        elif inq:
            fall = float(inq["fall_at_s"])
            start = float(args.inquiry_at) if args.inquiry_at is not None else float(inq["inquiry_start_s"])
        else:
            print("--inquiry on a plain WAV needs --fall-at SECONDS (when the scripted suspected fall happens)", file=sys.stderr)
            return EXIT_MISSING_INPUT
        window = float((inq or {}).get("window_s", cfg.audio.inquiry_response_window_s))
        try:
            plan = rp.InquiryPlan(fall_at_s=fall, inquiry_start_s=start, window_s=window, level=args.event_level, after_s=args.after_seconds)
        except ValueError as e:
            print(f"invalid inquiry script: {e}", file=sys.stderr)
            return EXIT_MISSING_INPUT
        if reply is not None:
            if inq and inq.get("reply_text"):
                print("this scene already contains a spoken reply; --reply-text/--reply-wav are for plain WAV files", file=sys.stderr)
                return EXIT_MISSING_INPUT
            i0 = int((start + 1.0) * rp.SR)
            audio = audio.copy()
            audio[i0:i0 + len(reply)] += reply[: max(len(audio) - i0, 0)]
            audio = np.clip(audio, -1.0, 1.0)
    elif reply is not None or args.enroll_voice or args.enroll_wav:
        print("--reply-text/--reply-wav/--enroll-voice/--enroll-wav only make sense together with --inquiry", file=sys.stderr)
        return EXIT_MISSING_INPUT
    # ---- sound model, voice stack
    clf, why = (None, "disabled by --no-sound") if args.no_sound else _load_classifier(cfg, args.model)
    if clf is None and not args.no_sound:
        print(f"sound model unavailable: {why}\nrun `homeease models fetch --audio` and `homeease train-sound --model-id {args.model or cfg.audio.sound_monitor.model_id}`, or pass --no-sound for the voice branch only", file=sys.stderr)
        return EXIT_MISSING_INPUT
    if args.vad:
        cfg.audio.vad = args.vad  # the pipeline builds the VAD from the config; one that cannot be built is reported as a degraded status
    voice = rp.VoiceStack()
    if plan is not None and (args.enroll_voice or args.enroll_wav):
        try:
            utts = _utterances(args)
            from homeease.ml.audio.speaker import WeSpeakerEmbedder

            emb = WeSpeakerEmbedder(cfg.audio.voiceprint.speaker_model)
            voice.embedder = emb
            voice.enrollment = VoiceprintEnrollment.from_audio(emb, utts or [])
            voice.threshold = args.threshold if args.threshold is not None else rp.DEMO_THRESHOLD
        except (sc.SceneUnavailable, EnrollmentError, OSError, ValueError, ImportError, RuntimeError) as e:
            print(f"cannot enrol the voiceprint: {e}", file=sys.stderr)
            return EXIT_MISSING_INPUT
    elif args.threshold is not None:
        voice.threshold = args.threshold  # without an enrollment the threshold changes nothing (no_enrollment), but it is recorded
    # ---- go
    print(f"== replay {wav.name}: {len(audio) / rp.SR:.1f} s, mono 16 kHz (read into memory; nothing is written except --json-out labels)")
    out = rp.run_replay(audio, rp.SR, cfg, clf, plan=plan, voice=voice, truth=truth, on_progress=lambda m: print("  " + m))
    if args.json_out:  # written first: a console problem while printing must not lose the result
        payload = {"input": wav.name, "scene": (side or {}).get("name"), "seed": (side or {}).get("seed"), **out.to_dict()}
        Path(args.json_out).write_text(dumps(_labels_only(payload), indent=1), encoding="utf-8")
    _print_outcome(out, args.threshold is None and voice.enrollment is not None)
    if args.json_out:
        print(f"labels and statistics written to {args.json_out} (no audio, no reply text)")
    return 0


def _print_outcome(out: Any, demo_thr: bool) -> None:
    ss, ps = out.sound_status, out.pipeline_status
    print(f"sound branch : {ss['status']}  model={ss.get('model_id') or '-'}  windows scored {ss['windows_scored']} (onset {ss['windows_by_trigger'].get('onset', 0)}, periodic {ss['windows_by_trigger'].get('periodic', 0)}), "
          f"events {ss['events_emitted']}")
    vs = ps["voice"]
    print(f"voice branch : {vs['status']}  vad={vs.get('vad') or '-'}  speech segments {vs.get('segments', 0)} ({vs.get('speech_s', 0)} s)")
    print(f"gate         : noise floor {ps['gate']['noise_floor_db']} dBFS, onset +{ps['gate']['onset_db']} dB, min {ps['gate']['min_level_dbfs']} dBFS")
    if out.events:
        print("sound events (category, confidence and statistics only; raw audio stored: False)")
        print(f"  {'category':<9} {'conf':>5}  {'start':>6}-{'end':<6}  {'dur':>5}  {'rms dB':>7} {'peak dB':>8}  windows")
        for e in out.events:
            print(f"  {e['category']:<9} {e['confidence']:>5.2f}  {e['t_start_s']:>6.1f}-{e['t_end_s']:<6.1f}  {e['duration_s']:>4.1f}s  {e['rms_db']:>7.1f} {e['peak_db']:>8.1f}  {e['n_windows']}")
    else:
        print("sound events : none")
    if out.speech:
        print("speech segments (voice branch; metadata only): " + ", ".join(f"{(s['t_start_mono_ms'] - out.t0_mono_ms) / 1000:.1f}-{(s['t_end_mono_ms'] - out.t0_mono_ms) / 1000:.1f}s" for s in out.speech))
    if out.truth_check:
        print("scripted sounds of this scene (what the sound branch reported near them; ONE scene, not an accuracy result):")
        for r in out.truth_check:
            print(f"  {r['kind']:<8} {r['t0_s']:>5.1f}-{r['t1_s']:<5.1f}s expected {r['expect']:<8} -> {r['result']}")
        if out.unexplained:
            print("  events away from every scripted sound: " + ", ".join(f"{u['category']}@{u['t0_s']}s" for u in out.unexplained))
    if out.inquiry:
        iq, r = out.inquiry, out.inquiry["record"]
        b, a1, a2 = iq["event_before"], iq["after_reply"], iq["after_timers"]
        print("voice inquiry (scripted; the prompt is simulated, the reply is read from the replayed audio):")
        print(f"  reply        : status={r.get('status')} intent={r.get('intent')} ({r.get('intent_reason')}) transcript={r.get('transcript')!r} asr={r.get('asr_model')} logprob={r.get('asr_avg_logprob')}")
        thr = iq.get("threshold")
        print(f"  voiceprint   : enrolled={iq['voiceprint_enrolled']} threshold={thr}{' (demo value calibrated on synthetic speakers, docs/results.md 3.3)' if demo_thr else ''} "
              f"score={r.get('speaker_score')} verified={r.get('speaker_verified')} speech={r.get('speech_s')} s")
        print(f"  EVIDENCE     : cancel_eligible={r.get('cancel_eligible')} reasons={r.get('cancel_reasons')} policy={r.get('cancel_policy')}")
        print(f"  event before : {b['level']}/{b['status']} step {b['escalation_step']}, {b['notifications']} notification(s)")
        print(f"  event after  : {a1['level']}/{a1['status']} step {a1['escalation_step']}, {a1['notifications']} notification(s)   after {a2['virtual_s_waited']:.0f} virtual s: {a2['level']}/{a2['status']} step {a2['escalation_step']}, {a2['notifications']} notification(s)")
        print("  " + iq["policy"])


# ------------------------------------------------------------------------------- parser
def register(sub: Any) -> None:
    p = sub.add_parser("audio", help="Function 3 demo without a microphone: generate scripted scenes, replay audio through the voice / sound pipeline")
    asub = p.add_subparsers(dest="audio_cmd", required=True)

    s = asub.add_parser("scenes", help="generate scripted scenes (ESC-50 background + clips + TTS reply) from a seed into data/synthetic/audio_scenes/ (about 0.8 MB each)")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--scene", action="append", default=None, help="scene name (repeatable); default: all that can be built here")
    s.add_argument("--out", default=None, help=f"output folder (default {DEFAULT_SCENE_DIR}, gitignored)")
    s.add_argument("--list", action="store_true", help="list the scene names and exit")
    s.set_defaults(fn=cmd_audio_scenes)

    r = asub.add_parser("replay", help="push a WAV file or a generated scene through Function3Pipeline offline; optional scripted fall + voice inquiry")
    r.add_argument("--wav", default=None, help="audio file (any sample rate, mono or stereo)")
    r.add_argument("--scene", default=None, help="scene name from `homeease audio scenes --list`")
    r.add_argument("--seed", type=int, default=42, help="seed of the generated scene file (--scene)")
    r.add_argument("--scenes-dir", default=None, help=f"where the scenes were generated (default {DEFAULT_SCENE_DIR})")
    r.add_argument("--model", default=None, help="sound model package (default: audio.sound_monitor.model_id)")
    r.add_argument("--no-sound", action="store_true", help="skip the sound branch (voice branch and inquiry only)")
    r.add_argument("--vad", choices=["silero", "energy", "webrtc"], default=None, help="speech VAD of the voice branch (default: audio.vad)")
    r.add_argument("--inquiry", action="store_true", help="script a suspected fall and a voice inquiry on the replayed audio")
    r.add_argument("--fall-at", type=float, default=None, help="seconds into the audio of the scripted suspected fall (scenes carry their own)")
    r.add_argument("--inquiry-at", type=float, default=None, help="seconds when the listening window opens (default: fall + 4 s prompt)")
    r.add_argument("--event-level", choices=["emergency", "pending"], default="emergency", help="level of the scripted event")
    r.add_argument("--after-seconds", type=float, default=30.0, help="virtual seconds to keep the system running after the inquiry (escalation timers)")
    r.add_argument("--reply-text", default=None, help="scripted reply for a plain WAV: spoken with a TTS voice and mixed in at inquiry start + 1 s")
    r.add_argument("--reply-voice", default="Zira")
    r.add_argument("--reply-lang", default="en")
    r.add_argument("--reply-wav", default=None, help="scripted reply as an audio file, mixed in at inquiry start + 1 s")
    r.add_argument("--enroll-voice", default=None, help="enrol the voiceprint from three TTS sentences in this voice (e.g. Zira); needs the WeSpeaker model")
    r.add_argument("--enroll-wav", action="append", default=None, help="enrol the voiceprint from audio files (repeat at least 3 times, each >= 1.5 s of speech)")
    r.add_argument("--threshold", type=float, default=None, help="cosine threshold of the voiceprint (default with an enrollment: 0.686, the FAR 0.1%% point of experiment A6 on synthetic speakers; production default: null = nothing eligible)")
    r.add_argument("--json-out", default=None, help="write the labels and statistics (no audio) as JSON")
    r.set_defaults(fn=cmd_audio_replay)


__all__ = ["register", "cmd_audio_scenes", "cmd_audio_replay"]
