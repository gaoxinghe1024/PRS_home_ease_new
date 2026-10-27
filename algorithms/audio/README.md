# Algorithm 3: voice interaction and audio monitoring

Algorithm 3 does three things. Everything runs on the local machine, nothing goes to a cloud service, and audio is only processed in memory.

| Function | Input | Output | Role in the system |
|---|---|---|---|
| Sound-event labelling | continuous audio | HELP / IMPACT / ABNORMAL / DAILY / UNKNOWN labels with time, confidence and level | a hint only; it never raises or cancels an alarm |
| Spoken call for help | speech segments cut from the audio | an alarm, or the reason the speech was ignored | a clear call in Chinese or English ("help me", "jiu ming") raises an emergency |
| Voice inquiry | the reply to "What's wrong? Are you OK?" after a fall candidate | HELP / REASSURANCE / OTHER / UNCERTAIN, and whether the reply is a valid cancel | without a valid cancel a PENDING event is raised to EMERGENCY; a valid cancel only stops that escalation |

A fixed 10 second window adds one rule: when 1, 2 or 3 of the channels accelerometer, vision and audio give an alarm marker inside the window, the event tag is `single_channel`, `dual_channel` or `triple_channel`.

## How HomeEase uses it

The integrated system ([`backend/workers.py`](../../backend/workers.py), `AudioWorker`) loads the `Algorithm3` facade below in its own process with strict Silero VAD. Offline analysis passes one decoding to `label_sounds` and `detect_spoken_help`; the continuous microphone stream from the browser goes through the original `Function3Pipeline`, `SoundMonitor` and `HelpListener`, and inquiry results are judged with the original policy functions.

Using it from code:

```python
import sys; sys.path.insert(0, "src")
from algorithm3 import Algorithm3, inquiry_outcome, alarm_tag
from homeease.ml.audio.features import load_audio    # any audio file -> 16 kHz mono

a3 = Algorithm3()                                     # models load on first use
x = load_audio("samples/scene_fall_help_call.wav")

a3.label_sounds(x, 16000)         # {"events": [...], "speech": [...], ...}
a3.detect_spoken_help(x, 16000)   # {"help": True, "segments": [{"verdict": "help", ...}]}
rec = a3.classify_reply(audio=x[9*16000:14*16000], sr=16000)    # or text="I'm fine"
inquiry_outcome({**rec, "prompt_played": True}, decision_level="PENDING")
alarm_tag([{"channel": "vision", "at_s": 0}, {"channel": "audio", "at_s": 3}])    # dual_channel
```

## Layout

```
src/homeease/              trimmed copy of the HomeEase audio code (package paths unchanged)
  ml/audio/                pipeline, VAD, sound monitor, ASR, reply rules, help listener, voiceprint gate
  backend/services/alarm_tags.py    the 10 s tag rule
src/algorithm3/            api.py (the Algorithm3 class), policy.py (inquiry outcome and tags as pure functions),
                           decode.py (any audio file -> 16 kHz mono)
config/default.yaml        default configuration; Algorithm 3 reads the audio and privacy sections
models/                    weights (see Models)
samples/                   6 scenes and 10 short replies, described in samples/index.json
docs/DATA_CARD.md          data sources and splits
THIRD_PARTY_NOTICES.md     licences of the pretrained models and data
```

## Data flow

```
microphone, 16 kHz mono, 15 s ring buffer in memory
  -> front end (ml/audio/pipeline.py)
       speech gate (Silero VAD)            -> speech segments -> voice branch
       onset trigger + sound gate          -> 2 s windows     -> sound branch

sound branch: window -> RMS normalisation -> YAMNet embedding -> logistic regression -> per-class threshold -> label or UNKNOWN
              -> SoundMonitor (two consecutive windows with the same label make an event; events merge within 20 s)
voice branch: segment -> Whisper tiny -> reply rules v2 -> HELP / REASSURANCE / OTHER / UNCERTAIN
   a) help listener: 0.3 to 6 s, log-probability >= -1.2, a clear call for help, 30 s cooldown
   b) inquiry: asks the question, 10 s window for the answer
   c) voiceprint gate: speaker_verified only when a voiceprint is enrolled; evidence only

decision: inquiry result -> inquiry_outcome (policy.py) -> audio marker and escalation
          sensor markers + audio marker -> alarm_tag -> number of channels inside 10 s -> tag
```

## Interface (`algorithm3`)

`Algorithm3(vad="auto", overrides=None)`: `vad` is `auto` (Silero when torch is installed, otherwise energy), `silero` or `energy`. `overrides` is a nested dictionary of settings, for example `{"audio": {"help_listener": {"cooldown_s": 60}}}`. One call is processed at a time.

| Method | Returns |
|---|---|
| `label_sounds(samples, sr=16000)` | `events` (category, confidence, `start_s`, `end_s`, levels, windows), `speech` (time of each speech segment), `vad`, `status`, `raw_audio_stored: False` |
| `classify_reply(audio=None, sr=16000, text=None, enrollment=None)` | a reply record: `status` (RESPONDED / NO_RESPONSE / audio_unavailable), `intent`, `intent_reason`, `cues`, `transcript`, `asr_avg_logprob`, `bare_reassurance`, `cancel_eligible`, `cancel_reasons`, `trusted_reassurance` |
| `detect_spoken_help(samples, sr=16000)` | `segments` with a `verdict` each, and `help` (whether anything raised an alarm) |
| `inquiry_outcome(rec, ...)` | see below |
| `alarm_tag(markers)` | `tag`, `channels` and the `markers` with a `counted` flag |
| `warm_up()`, `info()` | loads all models; the VAD, models and main settings in use |

Verdicts of `detect_spoken_help`: `help` (alarm); ignored with a reason: `too_short`, `too_long`, `cooldown`, `low_confidence`, `no_text`, `not_help`, `not_a_call`, `too_many_words`; status codes: `disabled`, `no_consent`, `inquiry_listening` (the speech belongs to a running inquiry), `asr_unavailable`, `asr_error`, `stopped`, `callback_error`.

`inquiry_outcome(rec, *, decision_level="PENDING", resolved=False, button_ok=False, escalate_on_no_cancel=True, min_asr_logprob=-1.0)` is a pure function without database or clock:

| Field | Meaning |
|---|---|
| `state` | `cancelled`, `no_cancel`, `not_run` or `audio_unavailable` |
| `cancelled`, `by` | a valid cancel, by `voice` (trusted "I'm fine") or `button` |
| `marker` | the audio marker to record: `inquiry_no_cancel` or `spoken_help`, otherwise `None` |
| `escalate` | raise a PENDING event to EMERGENCY |
| `notify` | a call for help on an event that is already EMERGENCY: tell the caregivers again |
| `call` | a call for help after the event was closed: an alarm of its own |

## Rules

One rule above all: a reply never closes or lowers an event.

| Result of the inquiry | Action |
|---|---|
| trusted "I'm fine" (intent REASSURANCE, enough recogniser confidence, matching voice when a voiceprint is enrolled) or the older adult's "I'm OK" button | only stops the escalation; the event stays PENDING until a caregiver closes it; no audio marker |
| silence, unclear or unrelated speech, a call for help, an untrusted "I'm fine" | audio marker; PENDING is raised to EMERGENCY (audio is the second channel) |
| audio off, no consent, microphone or recogniser failure, question not delivered | no marker, no escalation, shown as degraded, never treated as safe |
| a call for help on an event that is already EMERGENCY | no further escalation, the caregivers are told |

A reply that is only a short, standard Mandarin "I am fine" ("mei shi", "mei wen ti", "mei guan xi") is recognised with a low confidence because the score is averaged over a few tokens, so the cancel test gives that whole-utterance form 0.3 of slack (`bare_reassurance`). Other replies keep the gate.

Tag rules (`alarm_tags.py`, the 10 s window is fixed): the channels accel, vision and audio count; `manual` (the help button) is recorded but not counted; the window starts at the earliest counted marker; markers outside it are kept with `counted: false`; the audio marker is stamped when the question was asked; sound-event labels never become markers.

## Settings

Defaults are in `config/default.yaml` (`audio:` section, models in `homeease/core/config.py`).

| Setting | Default | Meaning |
|---|---|---|
| `audio.vad` | silero | speech gate; energy and webrtc are available |
| `audio.buffer_seconds` | 15 | ring buffer in memory |
| `audio.gate.speech_min_s / speech_end_silence_s / speech_max_s` | 0.25 / 0.6 / 12 | cutting speech segments |
| `audio.sound_monitor.window_s / hop_s / confirm_windows / cooldown_s` | 2.0 / 0.5 / 2 / 20 | sound events |
| per-class thresholds (in the model's `meta.json`) | HELP 0.42, IMPACT 0.99, ABNORMAL 0.91, DAILY 0.52 | below the threshold the label is UNKNOWN |
| `audio.help_listener.min_utterance_s / max_utterance_s` | 0.3 / 6.0 | length gate of the help listener |
| `audio.help_listener.cooldown_s / min_asr_logprob` | 30 / -1.2 | cooldown; recogniser confidence |
| `audio.inquiry_response_window_s` | 10 | time to answer |
| `audio.escalate_on_no_cancel` | true | escalate when there is no valid cancel |
| `audio.voiceprint.threshold` | null | no default threshold, so no reply is `cancel_eligible` until it is calibrated with real voices |
| `audio.voiceprint.min_speech_s / min_asr_logprob` | 1.0 / -1.0 | voiceprint gate |
| `audio.auto_cancel_emergency` | false | the configuration validator refuses `true`; so does `privacy.persist_raw_audio` |

None of these values was tuned on home recordings, and the false-alarm rate on continuous audio was not measured.

## Models

| File | Size | Source | Licence | Use |
|---|---|---|---|---|
| `audio/yamnet/yamnet.onnx` | 16.1 MB | audiomagic/yamnet-onnx @ f25b741c (Google YAMNet) | Apache-2.0 | sound embedding |
| `audio_sound/sound_events/v2/model.joblib` + `meta.json` | 83 KB | trained for this project (folds 1 to 4) | n/a | logistic regression and thresholds on the embedding |
| `audio/faster-whisper-tiny/` | 75.5 MB | Systran/faster-whisper-tiny @ d90ca5fe | MIT | speech recognition |
| `audio/wespeaker/voxceleb_resnet34_LM.onnx` | 26.5 MB | Wespeaker/wespeaker-voxceleb-resnet34-LM @ f0c48c29 | CC BY 4.0 | voiceprint embedding |
| Silero VAD | in the `silero-vad` pip package | snakers4/silero-vad | MIT | speech gate |

Checksums are in `models/MODELS.json`. The sound model package is checked by sha256 and against the YAMNet file it was trained with.

## Evaluation summary

All numbers come from clean, read, synthetic or acted audio, mostly English. There is no data from older adults, real television or real homes. The full experiments are in section 8.4 of the report.

| Experiment | Result |
|---|---|
| A1 speech gate (45 constructed mixtures, 113,863 frames) | Silero precision 0.995 / recall 0.813 / F1 0.895; energy VAD 0.871 / 0.946 / 0.907; WebRTC 0.869 / 0.931 / 0.899. With an impact in a pause between words the energy and WebRTC precision drops to 0.81, Silero stays at 0.988 |
| A2 sound events (four classes, 2,382 windows, five grouped folds) | YAMNet embedding + logistic regression macro-F1 0.677 (interval 0.633 to 0.740); HELP recall 0.688, precision 0.494 |
| A2b shipped model (fold 5, 466 windows) | macro-F1 0.658; HELP recall 0.610, precision 0.575 (with 17.6% HELP; about 6% at 1%); ABNORMAL recall 0.408 |
| A3b reply recognition (196 test items, 38 own calls for help) | rules v1 macro-F1 0.480, 23 calls missed; rules v2 0.894 [0.82, 0.94], 5 missed. An upper bound: the lexicon was written after seeing both splits, and 20 unseen calls written afterwards were recognised 6 times |
| A4 speaker verification (LibriSpeech) | VoxCeleb model EER 0.007 clean, 0.022 at 5 dB noise, 0.022 with replay simulation |
| A6 false cancels (150 + 150 synthetic speakers, 2,370 items) | at the strict threshold another person saying the same words is accepted 0.05%, a visitor 0.09%, television 0.00%; but only 41.7% of real users are accepted (21.3% at 5 dB) and a replayed recording is accepted 5.3% |

What A6 means: judged by text alone, someone else or the television saying "I'm fine" is taken as a cancel 78% to 89% of the time. A voiceprint brings that under 0.1% but rejects more than half of the real users and does not stop replays. So the voiceprint is evidence only; there is no enrolment flow and no threshold in the shipped configuration.

## Privacy

- Audio exists only in the 15 s ring buffer, in the speech segment handed to the next component and in the reply buffer of one inquiry. Each path zeroes it at the end. Nothing is written to a file or a log.
- Only labels, times and statistics are stored. The recognised text is not stored (`privacy.persist_voice_transcript` is false); an event keeps the intent, the cue classes, the confidence and the duration.
- A voiceprint is biometric data and is not saved; an enrolment lives in memory only.
- Using the microphone needs consent; withdrawing consent stops listening at once and clears the buffers.
- Unavailable is not safe: a missing model, a broken microphone or recogniser, or a question that was not delivered shows as degraded.

## Known problems

- Not tested with a real microphone, loudspeaker and television. Recogniser latency (Whisper tiny on the test laptop: median about 0.6 s, p95 about 1.2 s) and the false-alarm rate with television were not measured.
- The HELP sound class is weak and only a hint; calls for help rely on the recognised words.
- The accuracy of the reply rules is an upper bound; with new wording it is probably lower. Mandarin misses more than English (4 of 19 against 1).
- The Mandarin visitor and television voices in the evaluation are simulated by resampling, not a real second speaker.
- A voiceprint has no enrolment flow in the product and the threshold is empty.
- The 10 s window is fixed. An inquiry that starts more than 10 s after the first marker is recorded but not counted, so an event can be EMERGENCY with the tag `single_channel`.
- Recogniser transcripts of very short or noisy speech depend on the numeric environment (the BLAS thread count); the same clip can be read "help is on the way" or "help us on the way" at 10 dB.
- `homeease.ml.audio.tts` uses SAPI (PowerShell) on Windows and `say` on macOS; only Windows was tested.
