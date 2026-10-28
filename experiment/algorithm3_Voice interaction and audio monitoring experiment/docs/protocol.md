# Evaluation protocol of the audio experiments (Algorithm 3)

Section numbers follow the project evaluation protocol (sections 1 to 7 cover the fall and exercise experiments).

## 8. Audio experiments (A1 to A6, end-to-end demo)

| ID | Question | Data | Metric |
|---|---|---|---|
| A1 | VAD for the speech branch | LibriSpeech speech mixed with ESC-50 non-speech at controlled SNR (45 constructed mixtures) | frame precision/recall/F1 per condition and overall |
| A2 | sound events, 3-class (unchanged) | ESC-50 mapped subset (`esc50-homeease-v1`), official folds | macro-F1 over IMPACT/ABNORMAL/DAILY, three-class aggregate. HELP is **not evaluated** here (ESC-50 has no "help" class) |
| A2 | sound events, 4-class (HELP/IMPACT/ABNORMAL/DAILY) | mapping `esc50-vivae-homeease-v2`: ESC-50 + VIVAE + FSD50K subset (Screaming and negatives) + LibriSpeech speech; 2 s windows, RMS-normalised; five group folds | pooled and per-fold macro-F1 with a group-bootstrap interval, class-wise precision/recall with Wilson intervals, confusion matrix, per-corpus recall and false-HELP rate, leave-one-corpus-out HELP recall (train without VIVAE / without FSD50K) |
| A3 | reply intent, legacy protocol (42 hand-picked replies) | TTS replies through `ml/audio/tts.py`: macOS `say` (self voices Samantha/Tingting; proxy voices Daniel and a Chinese "Grandpa" voice; TV voice Fred) **or** Windows SAPI (Zira/Huihui, David, emulated Mandarin proxy); LibriSpeech as TV speech, ESC-50 noise; faster-whisper tiny and base; rule and supervised intent | macro-F1 over HELP/REASSURANCE/OTHER/UNCERTAIN. The run name carries the backend (`A3_response_intent` = the published macOS rows, `A3_response_intent_sapi` = Windows), so rows of the two platforms are never mixed. TTS is synthetic speech, **not** older-adult speech |
| A3b | reply intent, phrase-bank protocol (312 phrases, dev/test split) | `config/audio_phrases.yaml` rendered with the platform voices (section 8.2); LibriSpeech and TTS lines as TV speech; ESC-50 noise at 5 dB; silence | per-class precision/recall/F1 with Wilson intervals, confusion matrix, accuracy per interference bucket, ASR latency, safety counts; settings chosen on the dev split only, test reported once |
| A4 | speaker verification | LibriSpeech, speaker-disjoint halves; clean, 5 dB noise and simulated loudspeaker replay | EER, TAR@FAR with a reliability note on the number of trials |
| A5 | alarm behaviour | headless system on a virtual clock with scripted inquiry outcomes (including one where the voiceprint gate reports `cancel_eligible = True`) | policy pass/fail: a voice reply, with or without cancel-eligible evidence, never closes or downgrades an EMERGENCY; HELP can raise PENDING |
| A6 | cancel-eligibility evidence (voiceprint + reassurance): true-cancel and FALSE-cancel rates | Piper `libritts_r` voices (up to 904 synthetic speakers) saying the phrase bank (section 8.3), ESC-50 household noise at 5 dB and alone, silence, loudspeaker simulation (TV, replay); LibriSpeech for a real-speech transfer check of the threshold | per scenario: trials, eligible count and rate with Wilson and speaker-bootstrap 95% intervals, the intent-only rate for comparison, gate reason codes; speaker-layer EER (by length of speech) and TAR / FAR at the dev thresholds; thresholds for FAR 0.1% and 1% chosen on dev speakers only, reliability note (FAR 0.1% needs at least 10,000 impostor trials) |
| demo | end-to-end replay of generated scenes (`homeease audio scenes` / `audio replay`, section 8.4) | six seeded scenes from fold-5 clips and TTS replies | **none: descriptive only** (what the sound branch reported near each scripted sound, the stored labels, the evidence record, the event state before and after). No rate is computed |

### 8.1 Sound-event model: splits, thresholds, what is reported

* **Split unit** (fixed before any window is cut): ESC-50 source recording (official folds; a source that
  straddles two folds is moved completely into one), VIVAE speaker, FSD50K uploader, LibriSpeech speaker.
  VIVAE speakers, FSD50K uploaders and LibriSpeech speakers are spread over five folds by a seeded greedy
  balance. Extra training windows of a clip never leave the clip's fold. `tests/test_audio_sound_data.py`
  and `tests/test_sound_train.py` fail if a group is in two folds or if the test fold reaches training,
  validation, the candidate choice or the threshold choice.
* **Experiment A2** uses test fold k, validation fold k+1 (CNN early stopping) and the rest for training.
  The headline candidate is the best by CV macro-F1 on those same test folds, which is slightly optimistic
  and is labelled so.
* **Deployable model** (`homeease train-sound`): fold 5 is the test fold. The candidate, the CNN
  temperature and the per-class UNKNOWN cut-offs come from **out-of-fold predictions on folds 1-4 only**;
  the cut-off of a class is the lowest value at which the accepted windows of that class reach the
  precision target (0.90) on dev data. The model is then refitted on folds 1-4 and scored on fold 5. **Fold 5 was scored twice for the shipped model:** the fallback for a target that cannot be reached was changed from "highest dev
  precision" to "best dev F1" after the first attempt's test result had been seen (results.md section 3.1); both runs are kept. A target that cannot be reached is reported (`target_met: false`), never hidden. UNKNOWN counts as a
  miss for recall and as no prediction for precision.
* **Sample sizes** are printed next to every rate. Rates carry Wilson intervals; macro-F1 carries an
  interval from a bootstrap over whole groups.
* **Corpus confound**: results are split by corpus (recall per corpus, false-HELP rate on non-HELP windows
  of each corpus, HELP precision inside VIVAE where both HELP and non-HELP voice exist), and HELP recall
  is measured when a whole corpus is left out of training.
* **Known uncontrolled cues** (documented, not removed): (a) the **padding of the 2-s window** - a clip shorter than 2 s sits in a window that is mostly -50 dB noise, the loudest 2 s of a longer clip does not, and the quiet share of the
  window alone separates HELP from the long-clip classes (AUC 0.785; a reversed cue inside FSD50K); `scripts/f3_review_analyses.py padding` measures it and probes the shipped model with padded short clips; (b) the **same Freesound recording in two corpora**
  (ESC-50 source file and FSD50K clip) can land in different folds because the group check works inside one corpus (`audio_v2.freesound_overlap`: 14 recordings, 11 in different folds, none HELP); (c) the **prevalence** of HELP in the
  test fold (17.6%) fixes the precision that is reported; at a lower prevalence the same recall and false-HELP rate give far lower precision (`scripts/f3_review_analyses.py help_prevalence`). A rate for ESC-50 non-HELP windows is not a
  specificity for short events, and no false-alarms-per-hour measurement exists.
* No number from these clean, short, acted or Freesound clips is a household or older-adult claim.

### 8.2 Spoken-reply recognition (A3b): phrase bank, splits, selection, what is reported

* **Phrase bank** (`config/audio_phrases.yaml`, schema 1, loader and validator `ml/audio/phrases.py`): entries
  carry `intent` (what the *words* mean: HELP / REASSURANCE / OTHER), `speaker_role` (self = the monitored person,
  visitor, tv), `family` (paraphrase family), `split` (dev / test). The validator (run by the loader and by
  `tests/test_audio_phrases.py`) enforces: a family lives in exactly one split / role / intent / language; ids and
  normalised texts are unique (no phrase can sit in both splits); every (role, intent, language) stratum has
  at least two families in dev and two in test; self at least 24 phrases and 8 families per intent and language,
  visitor and tv at least 8 phrases. `near_duplicates()` lints cross-split paraphrases (no pair reaches Jaccard 0.6; 9 pairs still reach 0.5). `config/audio_unseen_probe.yaml` holds 20 help and 15 reassurance phrases written afterwards by one reviewer as a hold-out text probe of the rule (not a random sample; **never tune the lexicon on it**).
  Mixed sentences ("I'm fine but I can't get up") are HELP; negated wellbeing ("I'm not okay") is HELP; negated
  pain ("I'm not hurt") is REASSURANCE; bare yes / no, hesitation, questions and off-topic speech are OTHER (A3b scores words, not answers to the prompt: the product reads a bare "yes" to "Do you need help?" as HELP in a separate step that A3b and A6 do not run).
* **Voices**: self = Zira (en) and Huihui (zh); visitor = David (en) and, because Windows ships one Mandarin voice,
  an **emulated** visitor (Huihui resampled 0.84x: lower pitch and slower, not a second real speaker); tv = David
  (en) / emulated Huihui 1.14x (zh), both through the loudspeaker simulation. LibriSpeech utterances without
  help or reassurance words (reference transcript) are the real-speech TV class. Every table prints platform,
  backend and voice names, and marks emulated voices.
* **Order of operations** (enforced by `experiments/audio_intent.run_protocol` and tested): (1) the dev grid is scored:
  ASR model (tiny, base) x decoding bias (none, hotwords, initial prompt) x language (forced, automatic,
  automatic restricted to en/zh) x intent method (rule v1, rule v1 on Simplified script, rule v2, supervised trained on the
  legacy corpus, supervised trained on dev phrases, conservative combination); (2) the selection rule picks one cell;
  (3) the choice is written to `dev_selection.json` **before the test split is rendered**; (4) the test split is scored
  once for the pre-change production setting, the selected cell and its one-factor neighbours.
* **Selection rule** (declared before any test clip existed): highest dev macro-F1 over HELP / REASSURANCE / OTHER among
  *deployable* cells with median ASR latency <= 3 s; ties within 0.005 go to fewer missed self calls for help, then
  lower latency. "Forced language" cells are oracle cells (the ASR is told the clip's language, which the product
  does not know and which fails for people who answer in the other language): scored, reported, never selected.
  The supervised-on-dev method is scored on dev with leave-family-group-out folds, never on phrases it was trained on.
* **Reported with every table**: P/R/F1 and a confusion matrix over HELP / REASSURANCE / OTHER / UNCERTAIN /
  NO_RESPONSE; accuracy per bucket (self reply, self at 5 dB noise, TV speech, TV says help, visitor answer, silence);
  ASR latency (median, p95); safety counts: *self call for help not recognised as HELP*, *TV or visitor speech
  recognised as REASSURANCE* (false evidence if no voiceprint were checked), *words that are not a call for help read as
  HELP* (false escalation). A silent clip is decided by the energy VAD exactly like the inquiry service; NO_RESPONSE and
  UNCERTAIN are never counted as safe.
* **Hygiene limits stated in the report**: (a) the v2 rule lexicon was written by the author of the phrase bank with both
  splits in sight, so its test numbers are not a blind estimate (the report lists the cues that fire only on test wording);
  (b) all speech is synthetic (or read), short and clean; (c) one TTS engine family, a handful of voices;
  (d) no older-adult, dialect, reverberation or microphone-distance effects. No number is a household claim.

### 8.3 Cancel-eligibility (A6): design, hygiene, what is reported

The proposal figure draws "person responds and voiceprint matches -> auto-cancel false alarm"; the approved policy keeps the proposal text (a reply is
supporting evidence and never changes an event). `ml/audio/cancel_gate.py` therefore produces `cancel_eligible` EVIDENCE only, and A6 measures how often
that evidence would be wrong, i.e. how often an automatic cancel would have fired when it should not have.

* **Gate** (all must hold, each failure has a reason code): an enrollment made with the same embedding model; the speaker model loads; a calibrated threshold is configured;
  at least `audio.voiceprint.min_speech_s` (1.0 s) of VAD speech; reply intent REASSURANCE; ASR `avg_logprob` >= `audio.voiceprint.min_asr_logprob` (-1.0); speaker score
  (cosine to the enrolled voiceprint) >= the threshold. No enrollment, no threshold, no model, too little speech, an embedding failure, silence: all False.
* **Speakers**: Piper `en_US-libritts_r-medium` renders the SAME words in different voices. Speakers are split into disjoint DEV and EVAL sets before any clip exists
  (150 + 150 by default, seeded). Words are split by paraphrase family (the phrase bank's dev / test split): dev speakers only say dev phrases, eval speakers only test phrases.
  Every speaker enrols with the same three neutral sentences (text-independent verification: the reply phrases are other words).
* **Order**: (1) dev: enrol every dev speaker, score dev reassurance phrases against every dev enrollment, `calibrate_threshold` at FAR 0.1% (proposal 10.1(3)) and at 1%
  (a looser point), clips with enough speech only; (2) the thresholds are fixed; (3) eval: the product chain (energy VAD -> faster-whisper tiny -> intent rule v2 -> gate)
  runs on every eval clip; (4) analysis; (5) a real-speech check applies the thresholds to LibriSpeech speakers. Nothing about eval influences a choice.
* **Scenarios** (a trial is one clip scored against one enrolled user, so a clip from another person gives many impostor trials): the enrolled speaker says a reassurance, clean and at
  5 dB household noise (true cancel, should be eligible); another person says the same words, clean and at 5 dB; a visitor says a visitor-role reassurance ("she is fine");
  TV / radio lines through the loudspeaker simulation; the enrolled user's own reassurance replayed through the loudspeaker simulation; ESC-50 household sounds alone; silence;
  the enrolled speaker says a call for help (must never be eligible).
* **Reported**: counts and rates with Wilson intervals (which treat trials as independent and are too narrow) and identity-level bootstrap intervals (resampling speakers);
  the intent-only rate of every scenario (what a bare spoken reply without voiceprint would have cancelled); which reason code blocks the trials; the speaker-layer EER by
  length of speech (the evidence for the minimum speech length); a consistency audit that runs the product `CancelGate.evaluate` on the audio of random clips and compares it with
  the vectorised analysis; the real-speech transfer check.
* **Reliability**: a false-accept rate of 0.1% needs at least 10,000 impostor trials to mean anything (`tar_at_far` flags it). The trials come from a limited set of speakers and are not
  independent; `reliable` in the report only says that enough trials exist. The independent units are the probes (215 behind the dev threshold) and the clips (100 noise clips and 20 silences behind the zero-event rows, each tested against 150 users):
  rows without events are bounded per clip, and the FAR carries a two-way speaker-bootstrap interval next to the trial-level Wilson one (`scripts/f3_review_analyses.py a6_resampling`).
* **Limits**: all voices are synthetic imitations of read audiobook speakers, clean and from one engine; English only (no multi-speaker Mandarin voice within the download budget); no older
  adults, no real TV, no room acoustics beyond a crude loudspeaker / reverb simulation. Piper's default sampling noise is not seedable, so a rerun differs slightly (`HOMEEASE_PIPER_DETERMINISTIC=1` sets `noise_scale` and `noise_w_scale` to 0 and makes the synthesis bit-identical; the published runs did not use it); the per-clip table (scores,
  transcripts; no audio) is saved and `python -m homeease.experiments.audio_cancel rescore <run dir>` re-analyses it. Without Piper the run falls back to the installed system voices,
  is labelled PARTIAL (`synthetic_only`) and reports no eligibility rate.

### 8.4 End-to-end replay and demo scenes (descriptive, not an evaluation)

`homeease audio scenes` builds six scripted scenes and `homeease audio replay` pushes a scene or any WAV through `Function3Pipeline` on a virtual clock (see the README of this folder). They exist to show that the
blocks of Figure 3-3 work together and that the policy and privacy rules hold end to end; `tests/test_audio_e2e.py` asserts those properties offline with stubs.

* **Source hygiene.** Every clip of a scene is drawn from fold 5 of the class-mapping-v2 split, the fold the shipped `sound_events` model never saw (training and thresholds use folds 1-4). The sidecar records file, fold,
  `held_out`, group, licence and sha256 of each clip. If a role has no fold-5 clip the scene falls back to another fold and the sidecar says `held_out: false`. Scene generation is deterministic in the seed (`rng([seed, crc32(name)])`), so a
  scene can be regenerated bit for bit on the same data and TTS engine (`test_audio_scenes.py`).
* **No scoring.** The replay prints, for each scripted sound with an expected category, whether the sound branch reported that category near it, another category, or nothing ("found" / "other:..." / "no event"), and the
  events away from every scripted sound. This is a description of ONE scene. It is never turned into precision or recall, never tuned on, and must not be quoted as performance: six scenes with one clip per role cannot
  support a rate, the clips are clean, and the noise is one ESC-50 class at a fixed level.
* **Scripted fall.** The suspected fall is an input of the demo, not a detection: in this system audio never creates, raises, closes or cancels a fall event. The reply is read from the same audio stream (the 10 s
  after the prompt) and goes through the product code (`AudioInquiryService`, `CancelGate`, `EventService.voice_response`). The replayed prompt is simulated and says so.
* **What the end-to-end test asserts** (`tests/test_audio_e2e.py`): the sound branch stores labels as scalars and nothing that could hold audio (table introspection plus a byte scan of every file the run leaves, in three encodings of
  the scene's samples); the memory holding the reply is zeroed on every path; failures (no model, a crashing classifier, VAD or ASR, no enrollment) surface as degraded statuses or `cancel_eligible: false` with a reason; and
  whatever the reply and the voiceprint say, the event keeps its level and status (differential test against a silent window, PENDING stays PENDING, an explicit HELP reply still raises PENDING to EMERGENCY, a mutation test
  proves the differential check would catch an auto-cancel).
* **Windows re-runs of A1 and A4.** The published A1 and A4 numbers come from a macOS run of 3-4 October that is not part of this package; they were re-run on Windows with the same code paths (run ids in docs/results.md section 3.4)
  and the generated table there is copied from those run folders.

### 8.5 Run statuses

| Status | Meaning |
|---|---|
| `completed` | real data, protocol followed |
| `synthetic_only` | ran on synthetic data; software and design check, no accuracy claim |
| `data_pending` | required data are not available; `missing` lists what is needed |
| `not_evaluated` | data exist but a class or condition cannot be evaluated (e.g. HELP in A2) |
| `failed` | exception; `traceback.txt` is kept |

### 8.6 Determinism of the audio experiments

A1, A2 (except the CNN), A3 (legacy), A5 and the A4 of the full-suite step reproduce to the printed precision. A3b's ASR decodes (faster-whisper retries bad decodes with random sampling; `HOMEEASE_ASR_TEMPERATURE=0` removes it) and A6's Piper noise (`HOMEEASE_PIPER_DETERMINISTIC=1` removes it) differ between runs; two A3b runs and three A6 runs are compared in `docs/results.md`, and the published runs used the default sampling. The small CNN of A2 is not bit-reproducible on CPU. A4 was not reproducible before the full-suite step because it seeded its noise from the process-salted `hash(path)`; it now seeds from the file name (`tests/test_audio_suite_determinism.py`).
