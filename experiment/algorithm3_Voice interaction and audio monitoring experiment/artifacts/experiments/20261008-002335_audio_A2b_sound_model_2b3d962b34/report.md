# audio / A2b_sound_model

- run_id: `20261008-002335_audio_A2b_sound_model_2b3d962b34`
- status: **completed**
- data: ESC-50 + VIVAE + FSD50K subset + LibriSpeech (group folds, mapping v2)
- code: `f5f4a1b89aa1` · config `2b3d962b34` · split `None` · seed `42`

## Headline
yamnet_embedding+logreg: test-fold macro-F1 0.539, HELP recall 0.061 / precision 1.000 (n=82 HELP windows) at dev cut-offs; package models/audio_sound/sound_events/v1

Package: `models/audio_sound/sound_events/v1`

Candidate chosen on dev: **yamnet_embedding+logreg** (dev macro-F1 per candidate: mfcc_logmel_stats+rf 0.645, mfcc_logmel_stats+logreg 0.601, logmel_small_cnn 0.589, yamnet_embedding+logreg 0.675)

Temperature 1.00; UNKNOWN cut-offs (dev precision target 0.90, chosen on out-of-fold dev predictions): HELP 0.99 (target not met), IMPACT 0.99, ABNORMAL 0.91, DAILY 0.52; global 0.85

## Test fold 5 (466 windows, 212 groups; untouched by training and threshold choice)

| class | support | precision | precision_ci95 | recall | recall_ci95 | f1 |
|---|---|---|---|---|---|---|
| HELP | 82 | 1.000 | [0.566, 1.000] | 0.061 | [0.026, 0.135] | 0.115 |
| IMPACT | 11 | 0.857 | [0.487, 0.974] | 0.545 | [0.280, 0.787] | 0.667 |
| ABNORMAL | 71 | 0.935 | [0.793, 0.982] | 0.408 | [0.302, 0.525] | 0.569 |
| DAILY | 302 | 0.923 | [0.882, 0.951] | 0.715 | [0.662, 0.763] | 0.806 |

macro-F1 with cut-offs 0.539 (95% group-bootstrap CI (0.42363750075474177, 0.6741388727575595)); argmax without cut-offs 0.660; coverage (not UNKNOWN) 0.594; accuracy on accepted windows 0.924

Confusion (rows: truth; columns: HELP, IMPACT, ABNORMAL, DAILY, UNKNOWN):

| truth | HELP | IMPACT | ABNORMAL | DAILY | UNKNOWN |
|---|---|---|---|---|---|
| HELP | 5 | 0 | 0 | 12 | 65 |
| IMPACT | 0 | 6 | 0 | 2 | 3 |
| ABNORMAL | 0 | 0 | 29 | 4 | 38 |
| DAILY | 0 | 1 | 2 | 216 | 83 |

Reload check through the production loader: max |p_saved - p_evaluated| = 5.36e-07 over 13 windows.

Scene check (SoundMonitor on real test-fold clips, status ok): DAILY @ 7.02s; HELP @ 14.85s; UNKNOWN @ 17.5s; IMPACT @ 23.36s; ABNORMAL @ 35.0s
Truth: HELP (vivae) @ 7.0s; HELP (fsd50k) @ 14.74s; IMPACT (esc50) @ 23.36s; ABNORMAL (esc50) @ 34.36s; DAILY (esc50) @ 45.36s; DAILY (librispeech) @ 56.36s

Trained on clean public corpora; not validated on household or older-adult audio. The labels are assistive hints and never change an event state.
