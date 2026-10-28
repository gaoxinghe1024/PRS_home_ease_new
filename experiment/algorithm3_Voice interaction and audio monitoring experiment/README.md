# Voice Interaction and Audio Monitoring (Algorithm 3): Experiments

This folder holds the experiments of Algorithm 3: the code that ran them, the data they used, the result of every run, and the tests. The algorithm that ships in the system is in `system code/algorithms/audio`. This folder is the record of how it was built, tested and compared.

Algorithm 3 has two branches:

- **Voice branch.** Speech is cut out of the microphone stream (VAD), turned into text (Whisper), and read by a rule set as HELP, REASSURANCE, OTHER or UNCERTAIN (NO_RESPONSE when nobody answers). A clear call for help raises an alarm. After a suspected fall the system asks the person whether they are all right and judges the reply. A voiceprint check adds evidence about who answered.
- **Sound branch.** Two-second windows of the audio are labelled HELP, IMPACT, ABNORMAL or DAILY (a YAMNet embedding with logistic regression and one cut-off per class). The labels are hints for the caregiver and do not raise an alarm by themselves.

The experiments compare the alternatives for each block under one fixed protocol, so a difference in the numbers can be traced to the block that changed. Every run keeps its configuration, metrics, report and per-clip table, without audio.

| ID | Block under test | Alternatives compared | Data | Main measure |
|----|------------------|-----------------------|------|--------------|
| A1 | Speech gate (VAD) | energy, WebRTC, Silero | LibriSpeech speech mixed with ESC-50 sounds at set SNR, 45 mixtures | frame precision / recall / F1 |
| A2 | Sound-event classifier | MFCC + log-mel statistics with logistic regression or random forest, small CNN, YAMNet embedding with logistic regression | ESC-50 (3 classes); ESC-50 + VIVAE + FSD50K + LibriSpeech (4 classes, 2,382 windows) | macro-F1, per-class precision / recall, leave-one-corpus-out HELP recall |
| A2b | Deployable sound model (class cut-offs) | candidate and cut-offs chosen on folds 1-4, fold 5 held out | same 4-class data | macro-F1, HELP precision / recall |
| A3 | Reply recognition, first protocol | Whisper tiny / base, rule / supervised intent | 42 synthetic replies | macro-F1 |
| A3b | Reply recognition, phrase bank | 108 settings: ASR model, decoding bias, language handling, 6 intent methods | 312 phrases, dev / test split by paraphrase family | macro-F1, self calls for help missed, TV / visitor read as reassurance |
| A4 | Speaker verification | WeSpeaker VoxCeleb vs CN-Celeb model; clean, 5 dB noise, simulated replay | LibriSpeech, speaker-disjoint | EER, TAR at FAR 0.1 % |
| A5 | Alarm behaviour (policy) | scripted inquiry outcomes on a headless system | none (virtual clock) | pass / fail: a reply never closes or lowers an emergency |
| A6 | Cancel evidence (voiceprint + reassurance) | thresholds for FAR 0.1 % and 1 % | 150 + 150 synthetic speakers (Piper), 2,370 evaluation clips | true-cancel and false-cancel rates |
| S | Integrated system | spoken calls, live stream, inquiry after a fall, cancel and withdraw | the running system | input and returned value, see `docs/system_test.md` |

All audio in these experiments is clean, read, synthetic or acted. None of the numbers is a statement about older adults or real homes.

---

## 1. Folder layout

```
algorithm3_Voice interaction and audio monitoring experiment/
├── README.md                    this file
├── data.txt                     where the data is
├── pyproject.toml               dependencies (extras: audio, audio-tts)
│
├── src/homeease/                code of the experiments
│   ├── cli.py, cli_audio.py       command line: python -m homeease ...
│   ├── experiments/               audio_suite.py (A1-A5), audio_intent.py (A3b), audio_cancel.py (A6),
│   │                              train_sound.py (A2b), headless.py (A5 system), runner.py (run folders, statuses)
│   ├── ml/audio/                  the algorithm under test: pipeline, VAD, sound model, ASR, reply rules,
│   │                              voiceprint gate, TTS, scenes, replay
│   ├── datasets/                  download and class mapping of ESC-50, LibriSpeech, VIVAE, FSD50K
│   └── core/, backend/, edge/, ml/{common,fall,exercise,pose}/, schemas/
│                                  shared project code that the runner, A5 and the tests import
│
├── config/
│   ├── default.yaml               default configuration
│   ├── audio_phrases.yaml         the 312-phrase reply bank (A3b, A6)
│   └── audio_unseen_probe.yaml    35 phrases written afterwards, a hold-out text probe of the reply rules
│
├── data/                        see section 3
│   ├── raw/vivae, raw/fsd50k_help   audio of the two corpora that are hard to rebuild
│   ├── raw/esc50, raw/librispeech_mini   download record (and licence, metadata) only
│   ├── manifests/                 class mapping of every VIVAE and FSD50K clip
│   └── synthetic/audio_scenes/    six scripted demo scenes with sidecar files
│
├── models/
│   ├── audio_sound/sound_events/v2/   the trained sound-event model (83 kB) and its meta.json
│   └── audio/MODELS.json              repository, revision and sha256 of the pretrained models (fetched by script)
│
├── artifacts/experiments/       one folder per run: config, metrics.json, report.md, per-clip tables
├── docs/
│   ├── results.md                 results of A1 to A6 with the run ids
│   ├── protocol.md                protocol of each experiment: splits, selection rules, what is reported
│   └── system_test.md             input and returned value on the integrated system
├── scripts/                     results table and the analyses behind the caveats
└── tests/                       19 test files for the audio code
```

The folders under `src/homeease` that are not audio (`backend`, `edge`, `ml/fall`, `ml/exercise`, `ml/pose`, `schemas`) are shared project code. A5 runs the alarm policy on the project's headless system and the runner imports the common helpers, so they are included unchanged.

---

## 2. Environment

Developed and run on Windows 11 with Python 3.12, CPU only.

```bash
pip install -e ".[audio]"          # add ",audio-tts" for experiment A6 (piper-tts is GPL-3.0, so it is optional)
```

Without installing, `PYTHONPATH=src python -m homeease ...` from this folder works too.

A3 and A3b render the replies with the Windows speech voices Microsoft Zira, David (en-US) and Huihui (zh-CN). Without them these two experiments end as `not_evaluated` and say so. Nothing in the experiments needs a microphone or a person.

| Package | Purpose |
|---|---|
| numpy, scipy, scikit-learn, joblib | features, classical models, metrics |
| torch | small CNN of A2, Silero VAD |
| faster-whisper | speech recognition (tiny, base) |
| onnxruntime | YAMNet and WeSpeaker |
| silero-vad, webrtcvad-wheels | the VAD candidates of A1 |
| piper-tts | multi-speaker synthetic voices for A6 |

---

## 3. Data

`data.txt` lists where each part is. The datasets, their use and their licences:

| Dataset | Used for | In this folder | Licence |
|---|---|---|---|
| ESC-50 (Piczak 2015) | A1 non-speech, A2 sound classes, background of the scenes | download record, metadata and licence; audio is fetched by script (646 MB archive) | CC BY-NC 3.0 |
| Mini LibriSpeech dev-clean-2 (OpenSLR 31) | A1 speech, A4 speaker verification, A2 speech windows, TV speech | download record; audio is fetched by script (126 MB) | CC BY 4.0 |
| VIVAE (Holz et al. 2022) | A2 HELP class (fear and pain vocalisations) and same-corpus negatives | all 1,085 clips | CC BY-NC 4.0 |
| FSD50K subset (Fonseca et al. 2022) | A2 HELP class (screams) and negatives | 265 clips (about 40 MB); `selection.json` lists the source clips and each clip's licence | per clip: CC0, CC BY, CC BY-NC |
| Reply phrase bank | A3b, A6 | `config/audio_phrases.yaml`, 312 original phrases (English and Mandarin) | project content |
| Unseen-wording probe | hold-out check of the reply rules | `config/audio_unseen_probe.yaml`, 20 help and 15 reassurance phrases | project content |
| Piper `en_US-libritts_r-medium` voices | A6 speakers | not stored; synthesised in memory | CC BY 4.0 (LibriTTS-R training data) |
| Windows speech voices | A3, A3b replies | not stored; synthesised in memory | operating-system component |

Class mapping `esc50-vivae-homeease-v2`: ESC-50 sound classes are mapped to IMPACT, ABNORMAL or DAILY, VIVAE fear and pain and FSD50K Screaming to HELP, and everything else from VIVAE and FSD50K to negatives. The per-clip mapping of VIVAE and FSD50K is in `data/manifests/`; the rules are in `src/homeease/datasets/adapters/audio_v2.py`.

Splits are made by group before any window is cut, so no source recording, speaker or uploader is in two folds: ESC-50 source recording, VIVAE speaker, FSD50K uploader, LibriSpeech speaker; the phrase bank by paraphrase family; A6 by speaker. Thresholds and candidates are chosen on dev data and the test fold is scored once. The one exception is the deployable sound model, whose fold 5 was scored twice; `docs/results.md` section 3.1 explains why.

Download the two large corpora once:

```bash
python -m homeease data download --dataset esc50
python -m homeease data download --dataset librispeech_mini
```

`vivae` and `fsd50k_help` can be downloaded the same way; the copies in `data/raw` are the ones the results were computed on.

---

## 4. Reproduction

```bash
python -m homeease models fetch --audio --audio-tts       # Whisper tiny / base, WeSpeaker x2, YAMNet, Piper voice (revisions pinned, sha256 recorded)
python -m homeease compare --suite audio                  # A1 to A6
python -m homeease compare --suite audio --only A1,A4     # quick subset
python -m homeease train-sound --model-id sound_events --version v3
python -m homeease audio scenes                           # the six scripted demo scenes
python -m homeease audio replay --scene fall_reassured --inquiry
python scripts/f3_results_table.py --check                # the results table against the run folders
python scripts/f3_review_analyses.py                      # padding probe, prevalence, speaker bootstrap, lexicon switch-off, ...
python -m pytest -q tests                                 # offline: no network, microphone or model download
```

Run times on the development machine (16 cores): A1 19 s, A4 about 9 min, A6 about 35 min, A3b about 2.6 h, `train-sound` about 8 min; the whole suite about 4.3 h. Each run writes a folder `artifacts/experiments/<time>_audio_<id>_<hash>/`. A rerun creates a new folder and does not overwrite the ones in this package.

Reruns differ slightly where the tools are random: Piper's sampling noise (A6, about 3 points on the true-cancel rate) and Whisper's retry sampling (A3b, about 0.01 macro-F1). They can be switched off with `HOMEEASE_PIPER_DETERMINISTIC=1` and `HOMEEASE_ASR_TEMPERATURE=0`; the published runs used the default sampling. The small CNN of A2 is not bit-reproducible on CPU.

Before publishing, A1 was rerun from a copy of this folder and gave the stored numbers (energy 0.907, WebRTC 0.899, Silero 0.895), and the audio tests were run there.

---

## 5. Results

The full tables, with run ids, sample sizes and confidence intervals, are in `docs/results.md`. The protocol is in `docs/protocol.md`. The main numbers:

| ID | Result |
|----|--------|
| A1 VAD | Frame F1 over all conditions: energy 0.907, WebRTC 0.899, Silero 0.895. Silero has the highest precision (0.995); energy and WebRTC have the highest recall. With impact sounds in the speech gaps the precision of energy and WebRTC falls to 0.81, Silero's stays at 0.99. |
| A2 sound events | 3 classes (ESC-50 folds): logistic regression 0.783 macro-F1, random forest 0.730, CNN 0.572. 4 classes (2,382 windows, 407 HELP): YAMNet embedding + logistic regression is best with 0.677 macro-F1 [0.633, 0.740]; HELP precision 0.494, recall 0.688. Inside VIVAE (same speakers for HELP and non-HELP voice) HELP precision is 0.416: fear and pain are hard to separate from other vocal emotions in 0.8 s clips. |
| A2b deployable model | Held-out fold 5 (466 windows): macro-F1 0.658; HELP precision 0.575 / recall 0.610; ABNORMAL 0.935 / 0.408; IMPACT 0.857 / 0.545 (11 windows only). Cut-offs HELP 0.42, IMPACT 0.99, ABNORMAL 0.91, DAILY 0.52. At a realistic HELP share the precision would be far lower (25 % at 5 % HELP, 6 % at 1 %). |
| A3 / A3b replies | Phrase bank, test split, 38 self calls for help: the earlier rules (v1) gave macro-F1 0.480 and missed 23 calls. Rules v2 on the same recogniser gave 0.894 and missed 5. The v2 lexicon was written with both splits in view, and on 20 help phrases written afterwards it read only 6 as HELP, so 0.894 is an upper bound for unseen wording. Decoding bias and Whisper base gave no reliable gain, so tiny with plain decoding stays. |
| A4 speakers | EER of the VoxCeleb model: 0.007 clean, 0.022 at 5 dB noise, 0.022 with simulated replay; CN-Celeb model 0.026 / 0.086 / 0.082. FAR 0.1 % cannot be estimated from 3,216 impostor trials, which is why A6 exists. |
| A5 policy | Passed: reassurance, a call for help, silence, audio failure and a reassurance in the enrolled voice with `cancel_eligible = True` all leave the emergency open at EMERGENCY. |
| A6 cancel evidence | At the threshold for FAR 0.1 % (chosen on dev speakers): the enrolled speaker's reassurance is eligible 41.7 % of the time (21.3 % at 5 dB noise); another person saying the same words 0.05 % (88.5 % on the words alone); TV or radio 2 of 44,700; the user's own recording replayed through a loudspeaker simulation 5.3 %; the enrolled speaker calling for help 0 of 450. The voiceprint removes the false cancels but also lowers the true ones from 67 % to 42 %, so it is treated as evidence and never cancels anything by itself. |

### Integrated system (S)

`docs/system_test.md` lists what was put in and what the running system returned, for `main` at commit `d303a70`:

| Test | Result |
|---|---|
| 19 spoken calls for help (Mandarin and English) | alarm for 19 of 19 clean, 19 of 19 with 20 dB noise, 19 of 19 with 10 dB noise, 18 of 19 very quiet |
| 17 sentences that are not calls (including "thanks for your help", "do you need any help") | no false alarm clean, at 20 dB or very quiet; one at 10 dB ("Help is on the way" heard as "help us on the way") |
| Live stream, 10 calls and 7 other sentences | 9 of 10 calls raise one event; none of the 7 others; the slowest 0.5 s block took 0.14 s |
| Inquiry after a fall (accelerometer fall, 18 answers) | a trusted "I'm fine" stops the escalation and the event stays PENDING; no answer, an unclear answer or a call for help raises it to EMERGENCY |
| Withdraw (13 cases) | allowed for the help button, a spoken call, and an escalated fall alarm, within 30 s; refused otherwise |
| Six sound samples | no alarm from sound labels alone; the call for help after a fall raises one |

The system test was made before later changes on `main`. For example, the reply rules there now contain an extra pattern for bare reassurance such as "mei shi", which this package's copy of the rules does not have. Two limits in the test are still open: "Thank you for your help" and a news sentence were read as HELP in the inquiry (the safe direction, since it escalates).

---

## 6. Limits

- All speech is synthetic or read, short and clean, mostly from one engine family. There are no older adults, accents, dialects, real distress, rooms or microphone distance.
- HELP is the weakest sound class (precision about 0.5 on the benchmark, much lower at a realistic share). Sirens, alarm beeps, glass breaking and shouts are found more reliably than a thud, which is usually labelled DAILY.
- The reply rules were written with the phrase bank in view; read their test numbers as an upper bound.
- A6 uses English synthetic speakers only; Mandarin speakers are not evaluated. Replays are not detected as replays; the score only drops because of the channel difference.
- No measurement of false alarms per hour exists; it needs continuous household recordings.
- The 2-second window of a short clip is mostly padding, and the model reacts to that (`docs/results.md` section 3.1). A model trained on padding-matched windows was not made.
- The results are from the code as of October 2026. The system code under `system code/` is what runs and may have changed since.
