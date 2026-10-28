# audio / A2_sound_events

- run_id: `20261008-145848_audio_A2_sound_events_083bdeb9b0`
- status: **completed**
- data: ESC-50 mapped subset (real isolated clips), official 5 folds; 4-class run adds VIVAE + FSD50K subset + LibriSpeech (group folds)
- code: `354f1478138c` · config `083bdeb9b0` · split `None` · seed `42`

## Headline
3-class: best macro-F1 mfcc_logmel_stats+logreg 0.783 (IMPACT/ABNORMAL/DAILY; HELP not evaluated). 4-class: best macro-F1 yamnet_embedding+logreg 0.677 (95% CI 0.633-0.740), HELP recall 0.688 / precision 0.494 on 407 HELP windows; leave-one-corpus-out HELP recall vivae 0.801 (n=277), fsd50k 0.385 (n=130)

# 3-class run (mapping v1, ESC-50 official folds, unchanged)

| config | macro_f1 | impact_recall | abnormal_recall | daily_precision | macro_f1_3class_agg |
|---|---|---|---|---|---|
| mfcc_logmel_stats+rf | 0.730 | 0.500 | 0.575 | 0.893 | 0.793 |
| mfcc_logmel_stats+logreg | 0.783 | 0.950 | 0.675 | 0.926 | 0.798 |
| logmel_small_cnn | 0.572 | 0.750 | 0.631 | 0.948 | 0.714 |

Class mapping esc50-homeease-v1 (see datasets/adapters/audio_sets.py). HELP is **not evaluated** in this 3-class run: ESC-50 has no call-for-help class (the 4-class run below uses VIVAE / FSD50K). Isolated clips overstate accuracy relative to continuous household audio. Sound labels are assistive hints only.

# 4-class run (mapping `esc50-vivae-homeease-v2`, 2 s windows, rms-normalised, grouped 5-fold CV)

Clips per corpus and class (each clip gives one evaluation window; groups = ESC-50 source recording, VIVAE speaker, FSD50K uploader, LibriSpeech speaker):

| corpus | groups | HELP | IMPACT | ABNORMAL | DAILY |
|---|---|---|---|---|---|
| esc50 | 800 | 0 | 40 | 160 | 760 |
| fsd50k | 171 | 130 | 45 | 25 | 65 |
| librispeech | 26 | 0 | 0 | 0 | 156 |
| vivae | 11 | 277 | 0 | 174 | 550 |

## Candidates (pooled over the 5 test folds; argmax decisions, UNKNOWN is not used here)

| candidate | macro_f1 | macro_f1_ci95 | fold_mean | fold_std | HELP_P | HELP_R | HELP_R_ci95 | IMPACT_R | ABNORMAL_R | DAILY_R | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mfcc_logmel_stats+rf | 0.667 | [0.624, 0.714] | 0.662 | 0.043 | 0.559 | 0.572 | [0.524, 0.620] | 0.671 | 0.412 | 0.890 | 2382 |
| mfcc_logmel_stats+logreg | 0.605 | [0.569, 0.646] | 0.604 | 0.021 | 0.454 | 0.582 | [0.534, 0.629] | 0.859 | 0.487 | 0.690 | 2382 |
| logmel_small_cnn | 0.625 | [0.580, 0.678] | 0.626 | 0.042 | 0.364 | 0.862 | [0.826, 0.893] | 0.824 | 0.373 | 0.652 | 2382 |
| yamnet_embedding+logreg | 0.677 | [0.633, 0.740] | 0.679 | 0.030 | 0.494 | 0.688 | [0.641, 0.731] | 0.835 | 0.663 | 0.740 | 2382 |

macro_f1_ci95 = percentile interval of a bootstrap over whole groups (not windows); fold_mean / fold_std = mean and standard deviation of the 5 per-fold macro-F1 values; HELP_R_ci95 = Wilson interval.

## Best candidate by CV macro-F1: `yamnet_embedding+logreg`

(selected on the same test folds, so its score is slightly optimistic; the deployable model's candidate is chosen on dev folds only, see `homeease train-sound`.)

Class-wise (Wilson 95% intervals):

| class | support | precision | precision_ci95 | recall | recall_ci95 | f1 |
|---|---|---|---|---|---|---|
| HELP | 407 | 0.494 | [0.453, 0.535] | 0.688 | [0.641, 0.731] | 0.575 |
| IMPACT | 85 | 0.645 | [0.553, 0.729] | 0.835 | [0.742, 0.899] | 0.728 |
| ABNORMAL | 359 | 0.540 | [0.493, 0.586] | 0.663 | [0.613, 0.710] | 0.595 |
| DAILY | 1531 | 0.896 | [0.878, 0.912] | 0.740 | [0.717, 0.761] | 0.811 |

Confusion matrix (rows: truth, columns: predicted):

| truth | HELP | IMPACT | ABNORMAL | DAILY |
|---|---|---|---|---|
| HELP | 280 | 0 | 57 | 70 |
| IMPACT | 1 | 71 | 2 | 11 |
| ABNORMAL | 65 | 6 | 238 | 50 |
| DAILY | 221 | 33 | 144 | 1133 |

Per fold: fold 1: macro-F1 0.666, HELP recall 0.72 (n=80); fold 2: macro-F1 0.659, HELP recall 0.69 (n=78); fold 3: macro-F1 0.692, HELP recall 0.71 (n=89); fold 4: macro-F1 0.731, HELP recall 0.71 (n=78); fold 5: macro-F1 0.647, HELP recall 0.61 (n=82)

### Per corpus (confound check: how much of the result is the corpus rather than the sound)

| corpus | windows | HELP_recall | HELP_n | HELP_precision_in_corpus | false_HELP_rate | false_HELP_ci95 | non_HELP_n |
|---|---|---|---|---|---|---|---|
| esc50 | 960 | n/a | n/a | n/a | 0.024 | [0.016, 0.036] | 960 |
| fsd50k | 265 | 0.754 | 130 | 0.925 | 0.059 | [0.030, 0.113] | 135 |
| librispeech | 156 | n/a | n/a | n/a | 0.000 | [0.000, 0.024] | 156 |
| vivae | 1001 | 0.657 | 277 | 0.416 | 0.354 | [0.320, 0.389] | 724 |

false_HELP_rate = share of that corpus' non-HELP windows predicted HELP. VIVAE holds HELP (fear/pain) AND non-HELP voice (anger, achievement, pleasure, surprise) from the same speakers and microphone, so HELP_precision_in_corpus / false_HELP_rate there cannot be explained by the corpus. The LibriSpeech row shows what read speech does to the HELP class.

## Leave-one-corpus-out (train without the corpus, score all of it; argmax decisions)

| candidate | held_out | trained_on | HELP_recall | HELP_recall_ci95 | HELP_n | false_HELP_rate | non_HELP_n | macro_f1 |
|---|---|---|---|---|---|---|---|---|
| mfcc_logmel_stats+rf | vivae | esc50,fsd50k,librispeech | 0.783 | [0.731, 0.828] | 277 | 0.541 | 724 | 0.352 |
| mfcc_logmel_stats+rf | fsd50k | esc50,librispeech,vivae | 0.108 | [0.065, 0.173] | 130 | 0.007 | 135 | 0.314 |
| mfcc_logmel_stats+logreg | vivae | esc50,fsd50k,librispeech | 0.744 | [0.689, 0.792] | 277 | 0.718 | 724 | 0.276 |
| mfcc_logmel_stats+logreg | fsd50k | esc50,librispeech,vivae | 0.177 | [0.121, 0.252] | 130 | 0.022 | 135 | 0.403 |
| logmel_small_cnn | vivae | esc50,fsd50k,librispeech | 0.671 | [0.614, 0.724] | 277 | 0.711 | 724 | 0.228 |
| logmel_small_cnn | fsd50k | esc50,librispeech,vivae | 0.354 | [0.277, 0.439] | 130 | 0.044 | 135 | 0.561 |
| yamnet_embedding+logreg | vivae | esc50,fsd50k,librispeech | 0.801 | [0.750, 0.844] | 277 | 0.662 | 724 | 0.305 |
| yamnet_embedding+logreg | fsd50k | esc50,librispeech,vivae | 0.385 | [0.305, 0.470] | 130 | 0.044 | 135 | 0.560 |

A HELP recall far below the in-corpus recall means the classifier leans on the corpus / channel. The FSD50K 'Screaming' label also covers film and game effects (for example monster screams), so it is a looser notion of a human call for help than VIVAE fear/pain.

All corpora are clean, short, public clips (acted vocalisations, Freesound uploads, read speech): these numbers say nothing about older adults, real distress or a home with TV, appliances and reverberation. HELP is an assistive hint and never changes an event state.
