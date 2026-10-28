# audio / A3b_response_intent_bank_sapi

- run_id: `20261008-082808_audio_A3b_response_intent_bank_sapi_cee2ad886d`
- status: **completed**
- data: phrase bank config/audio_phrases.yaml rendered with sapi voices (Windows) + LibriSpeech (TV speech) + ESC-50 noise
- code: `6699be206b2f` · config `cee2ad886d` · split `None` · seed `42`

## Headline
selected on dev: tiny|hotwords|auto + rule_v2; TEST macro-F1 0.878 [0.81, 0.94] vs baseline 0.480 [0.37, 0.58]; self HELP missed 7/38 (baseline 23/38); TV/visitor read as REASSURANCE 19/80

## Setup

- platform: **Windows**, TTS backend **sapi**
| role | lang | voice | emulated |
|---|---|---|---|
| self | en | Zira | no |
| self | zh | Huihui | no |
| tv | en | David | no |
| tv | zh | Huihui (emulated x1.14) | YES (resampling shift) |
| visitor | en | David | no |
| visitor | zh | Huihui (emulated x0.84) | YES (resampling shift) |

Voices are synthetic. Emulated voices are the self voice with a pitch/speed shift (not a second real speaker). TV speech goes through a loudspeaker simulation; LibriSpeech utterances are real read speech through the same simulation.

### Phrase bank

| split | phrases | families |
|---|---|---|
| dev | 156 | 78 |
| test | 156 | 78 |


### Clips

dev: 208 clips, test: 208 clips.

## Selection on DEV only

Rule: highest dev macro-F1 over HELP/REASSURANCE/OTHER among cells with median ASR latency <= 3.0 s; ties within 0.005 go to fewer missed self calls for help, then lower latency.

**Selected: ASR `tiny|hotwords|auto` + intent `rule_v2`**

Top 12 dev cells (cells marked oracle tell the ASR the clip's language; they are reference only and cannot be selected):

| asr (model|bias|language) | intent | macro_f1 | self_help_missed | asr_median_s |
|---|---|---|---|---|
| base|hotwords|forced  [oracle language] | rule_v2 | 0.972 | 1 | 0.500 |
| tiny|prompt|forced  [oracle language] | rule_v2 | 0.969 | 3 | 0.256 |
| base|prompt|forced  [oracle language] | rule_v2 | 0.960 | 2 | 0.491 |
| tiny|prompt|auto | rule_v2 | 0.956 | 3 | 0.485 |
| tiny|prompt|auto_en_zh | rule_v2 | 0.954 | 3 | 0.485 |
| tiny|hotwords|auto_en_zh | rule_v2 | 0.952 | 2 | 0.507 |
| tiny|hotwords|auto | rule_v2 | 0.952 | 2 | 0.505 |
| base|none|forced  [oracle language] | rule_v2 | 0.951 | 5 | 0.483 |
| base|none|auto | rule_v2 | 0.951 | 5 | 0.925 |
| base|none|auto_en_zh | rule_v2 | 0.951 | 5 | 0.935 |
| tiny|none|forced  [oracle language] | rule_v2 | 0.940 | 7 | 0.253 |
| tiny|none|auto_en_zh | rule_v2 | 0.938 | 7 | 0.502 |

Production baseline before this step on dev:

| asr (model|bias|language) | intent | macro_f1 | accuracy | n | self_help_missed | nonself_reassurance | asr_median_s | asr_p95_s |
|---|---|---|---|---|---|---|---|---|
| tiny|none|auto | rule_v1 | 0.605 | 0.628 | 196 | 19/38 | 5/80 | 0.501 | 0.912 |

Main effects on dev (mean dev macro-F1 over all cells of the other factors):

| factor | level | mean_dev_macro_f1 | cells |
|---|---|---|---|
| ASR model | tiny | 0.719 | 54 |
| ASR model | base | 0.706 | 54 |
| decoding bias | none | 0.688 | 36 |
| decoding bias | hotwords | 0.724 | 36 |
| decoding bias | prompt | 0.725 | 36 |
| language | forced | 0.721 | 36 |
| language | auto | 0.710 | 36 |
| language | auto_en_zh | 0.706 | 36 |
| intent method | rule_v1 | 0.679 | 18 |
| intent method | rule_v1_t2s | 0.720 | 18 |
| intent method | rule_v2 | 0.943 | 18 |
| intent method | sup_legacy | 0.602 | 18 |
| intent method | sup_dev | 0.612 | 18 |
| intent method | combined | 0.717 | 18 |

Reference-text upper bound on dev (no ASR; how much the intent method alone can do):

| intent | macro_f1 | n |
|---|---|---|
| rule_v1 | 0.759 | 196 |
| rule_v1_t2s | 0.759 | 196 |
| rule_v2 | 1.000 | 196 |
| sup_legacy | 0.658 | 196 |
| sup_dev | 0.617 | 196 |
| combined | 0.739 | 196 |

### Hygiene audit of the hand-written v2 rule

The v2 lexicon was written by the author of the phrase bank, with both splits in sight. So its TEST numbers (and the text upper bound) are NOT a blind estimate; the dev/test gap cannot reveal overfitting of the rule either. Cues that fire on test phrases and on no dev phrase: 13 of 51 cues that fire at all (harm:bad fall, harm:pain, harm:slipped, harm:\u4e0d\u8212\u670d, harm:\u6454\u4e86, harm:\u6ed1\u5012, harm:\u75bc, need_help, no_fall, not_\u6454\u4f24, not_\u6454\u5012, not_\u75bc, something_wrong). The v1 rule (written by the teammate before this bank existed) and the supervised models trained on the legacy corpus are blind to the test wording; sup_dev is trained on dev phrases only.


## TEST (reported once)

The test split was not used for any choice above.

### Baseline vs selected

| config | macro_f1 | macro_f1_CI | accuracy | n | self_help_missed | nonself_reassurance | asr_median_s | asr_p95_s |
|---|---|---|---|---|---|---|---|---|
| baseline: tiny|none|auto + rule_v1 | 0.480 | [0.37, 0.58] | 0.546 | 196 | 23/38 | 3/80 | 0.778 | 1.496 |
| selected: tiny|hotwords|auto + rule_v2 | 0.878 | [0.81, 0.94] | 0.872 | 196 | 7/38 | 19/80 | 0.756 | 12.055 |

CI = family/speaker-grouped bootstrap (1000 resamples). Rates below carry Wilson intervals.

### One factor at a time around the selected cell (test)

| change | asr (model|bias|language) | intent | macro_f1 | accuracy | n | self_help_missed | nonself_reassurance | asr_median_s | asr_p95_s |
|---|---|---|---|---|---|---|---|---|---|
| selected (dev) | tiny|hotwords|auto | rule_v2 | 0.878 | 0.872 | 196 | 7/38 | 19/80 | 0.756 | 12.055 |
| ASR bias | tiny|none|auto | rule_v2 | 0.891 | 0.888 | 196 | 5/38 | 17/80 | 0.778 | 1.496 |
| ASR model | base|hotwords|auto | rule_v2 | 0.912 | 0.903 | 196 | 6/38 | 22/80 | 1.413 | 23.714 |
| ASR bias | tiny|prompt|auto | rule_v2 | 0.881 | 0.878 | 196 | 6/38 | 17/80 | 0.654 | 5.495 |
| ASR language | tiny|hotwords|forced | rule_v2 | 0.902 | 0.903 | 196 | 5/38 | 20/80 | 0.407 | 6.613 |
| ASR language | tiny|hotwords|auto_en_zh | rule_v2 | 0.883 | 0.878 | 196 | 7/38 | 20/80 | 0.737 | 10.531 |
| intent method | tiny|hotwords|auto | rule_v1 | 0.573 | 0.602 | 196 | 22/38 | 5/80 | 0.756 | 12.055 |
| intent method | tiny|hotwords|auto | rule_v1_t2s | 0.573 | 0.602 | 196 | 22/38 | 5/80 | 0.756 | 12.055 |
| intent method | tiny|hotwords|auto | sup_legacy | 0.455 | 0.321 | 196 | 26/38 | 6/80 | 0.756 | 12.055 |
| intent method | tiny|hotwords|auto | sup_dev | 0.594 | 0.469 | 196 | 26/38 | 12/80 | 0.756 | 12.055 |
| intent method | tiny|hotwords|auto | combined | 0.720 | 0.602 | 196 | 7/38 | 10/80 | 0.756 | 12.055 |

### Selected cell on test: per-class precision / recall / F1

| class | support | precision | precision_CI | recall | recall_CI | f1 |
|---|---|---|---|---|---|---|
| HELP | 62 | 0.943 | [0.85, 0.98] | 0.806 | [0.69, 0.89] | 0.870 |
| REASSURANCE | 60 | 0.944 | [0.85, 0.98] | 0.850 | [0.74, 0.92] | 0.895 |
| OTHER | 70 | 0.805 | [0.71, 0.88] | 0.943 | [0.86, 0.98] | 0.868 |
| UNCERTAIN | 0 | 0.000 | [0.00, 0.56] | n/a | n/a | n/a |
| NO_RESPONSE | 4 | 1.000 | [0.51, 1.00] | 1.000 | [0.51, 1.00] | 1.000 |

### Confusion matrix (rows = truth)

| true \ pred | HELP | REASSURANCE | OTHER | UNCERTAIN | NO_RESPONSE | n |
|---|---|---|---|---|---|---|
| HELP | 50 | 3 | 9 | 0 | 0 | 62 |
| REASSURANCE | 2 | 51 | 7 | 0 | 0 | 60 |
| OTHER | 1 | 0 | 66 | 3 | 0 | 70 |
| NO_RESPONSE | 0 | 0 | 0 | 0 | 4 | 4 |

### Accuracy per interference bucket

| bucket | n | correct | accuracy | CI |
|---|---|---|---|---|
| self_noise_5dB | 28 | 22 | 0.786 | [0.60, 0.90] |
| self_reply | 84 | 76 | 0.905 | [0.82, 0.95] |
| silence | 4 | 4 | 1.000 | [0.51, 1.00] |
| tv_says_help | 12 | 8 | 0.667 | [0.39, 0.86] |
| tv_speech | 32 | 26 | 0.812 | [0.65, 0.91] |
| visitor_answer | 36 | 35 | 0.972 | [0.86, 1.00] |

### Safety counts

| safety count | value |
|---|---|
| self call for help NOT recognised as HELP | 7/38 (0.18, CI [0.09, 0.33]) |
|   of which recognised as REASSURANCE | 3/38 (0.08) |
| TV / visitor speech recognised as REASSURANCE (would be false evidence without a voiceprint) | 19/80 (0.24, CI [0.16, 0.34]) |
|   of which the words are NOT reassuring (recognition error) | 0/56 (0.00) |
| self reply that is NOT reassuring but recognised as REASSURANCE | 3/76 (0.04) |
| words that are not a call for help recognised as HELP (false escalation, any speaker) | 3/130 (0.02, CI [0.01, 0.07]) |

### Baseline on test: per-class and safety counts

| class | support | precision | precision_CI | recall | recall_CI | f1 |
|---|---|---|---|---|---|---|
| HELP | 62 | 0.958 | [0.80, 0.99] | 0.371 | [0.26, 0.50] | 0.535 |
| REASSURANCE | 60 | 1.000 | [0.72, 1.00] | 0.167 | [0.09, 0.28] | 0.286 |
| OTHER | 70 | 0.449 | [0.37, 0.53] | 1.000 | [0.95, 1.00] | 0.619 |
| UNCERTAIN | 0 | 0.000 | [0.00, 0.66] | n/a | n/a | n/a |
| NO_RESPONSE | 4 | 1.000 | [0.51, 1.00] | 1.000 | [0.51, 1.00] | 1.000 |

| safety count | value |
|---|---|
| self call for help NOT recognised as HELP | 23/38 (0.61, CI [0.45, 0.74]) |
|   of which recognised as REASSURANCE | 0/38 (0.00) |
| TV / visitor speech recognised as REASSURANCE (would be false evidence without a voiceprint) | 3/80 (0.04, CI [0.01, 0.10]) |
|   of which the words are NOT reassuring (recognition error) | 0/56 (0.00) |
| self reply that is NOT reassuring but recognised as REASSURANCE | 0/76 (0.00) |
| words that are not a call for help recognised as HELP (false escalation, any speaker) | 1/130 (0.01, CI [0.00, 0.04]) |

### Baseline confusion matrix

| true \ pred | HELP | REASSURANCE | OTHER | UNCERTAIN | NO_RESPONSE | n |
|---|---|---|---|---|---|---|
| HELP | 23 | 0 | 39 | 0 | 0 | 62 |
| REASSURANCE | 1 | 10 | 47 | 2 | 0 | 60 |
| OTHER | 0 | 0 | 70 | 0 | 0 | 70 |
| NO_RESPONSE | 0 | 0 | 0 | 0 | 4 | 4 |


### Reference-text upper bound on test (no ASR)

| intent | macro_f1 | n |
|---|---|---|
| rule_v1 | 0.593 | 196 |
| rule_v1_t2s | 0.593 | 196 |
| rule_v2 | 0.990 | 196 |
| sup_legacy | 0.479 | 196 |
| sup_dev | 0.659 | 196 |
| combined | 0.801 | 196 |

### Hallucination probe: ESC-50 appliance noise reaching the ASR (expected: no HELP / REASSURANCE)

| asr | noise clips | passed VAD | read as HELP | read as REASSURANCE | examples |
|---|---|---|---|---|---|
| tiny|none|auto | 12 | 12 | 0 | 0 | Thank you for watching! | Let's go! | Thank you. |
| tiny|hotwords|auto | 12 | 12 | 3 | 2 | \u540a, \u53f8\u554a, \u6709\u53e3, \u6709\u53e3, \u7528\u5c1d\u5c1d\u5b89\u6392\u5899\u961f apartment | \u6551\u547d, \u6551\u547d, \u6551\u547d, \u6551\u547d, \u6551\u547d, \u6551\u547d, \u6551\u547d, \u6551\u547d, \u6551\u547d, \u6551\u547d,  | lumière, \u610f\u601d, \u786c, \u6b7b, \u7537, \u5fc3, \u4e2a\u4ec1, \u79bbfulness |
| base|hotwords|auto | 12 | 12 | 1 | 0 | \u6551\u62a4\u8f66, \u6551\u62a4\u8f66, \u6551\u62a4\u8f66, \u6551\u62a4\u8f66, \u6551\u62a4\u8f66, \u6551\u62a4\u8f66, \u6551\u62a4\u8f66, \u6551\u62a4\u8f66,  |
| tiny|prompt|auto | 12 | 12 | 0 | 0 | \u7092\u9488! | ʊ´ʊʊʊ-ʊ! | \u8c22\u8c22\u518d\u89c1\u52aa\u529b\u8c22\u8c22\u62cd\u63e1\u7b49\u4e8e\u4e50\u95e8\u662f\u5f88\u591a\u5730\u677f\u3002 \u518d\u6b21\u8bf7\u4f60\u716e\u7099,\u518d\u6b21\u8bf7\u4f60\u716e\u7099\u628a\u95e8\u7f8e\u8336\u997d\u4e86 |
| tiny|hotwords|forced | 12 | 12 | 0 | 1 | Thank you for watching, please like and  | Thank you for watching! | Be sure to check the things done! |
| tiny|hotwords|auto_en_zh | 12 | 12 | 4 | 1 | This is all I've got to do here, for my  | Thanks for watching! | \u5938, \u5938, \u5938, \u5938, \u5938, \u5938, \u5938, \u5938 |

### Transcript examples (selected cell, test, first 12 errors)

| bucket | lang | said | heard | expected | got |
|---|---|---|---|---|---|
| self_reply | en | Aiyo, my back | AL, my back | HELP | OTHER |
| self_reply | en | Nah | NAT | OTHER | UNCERTAIN |
| self_reply | zh | \u6211\u5728\u6d74\u5ba4\u6ed1\u5012\u4e86 | \u6211\u5728\u7389\u77f3\u82b1\u5c9b\u4e86 | HELP | OTHER |
| self_noise_5dB | zh | \u6211\u4e0d\u5c0f\u5fc3\u6454\u4e86\u4e00\u8de4 | \u6211\u4e0d\u5c0f\u5fc3, \u5e05\u4e86\u4e00\u4ea4 | HELP | OTHER |
| self_reply | zh | \u554a\uff0c\u6211\u7684\u8170 | \u554a, \u6211\u7684\u8170 | HELP | OTHER |
| self_reply | zh | \u6211\u6ca1\u95ee\u9898\uff0c\u4e0d\u8fc7\u9700\u8981\u6709\u4eba\u6276\u6211\u8d77\u6765 | \u6211\u6ca1\u95ee\u9898, \u4e0d\u8fc7\u9700\u8981\u6709\u4eba\u670d\u6211\u8d77\u6765 | HELP | REASSURANCE |
| self_noise_5dB | zh | \u6211\u6ca1\u95ee\u9898\uff0c\u4e0d\u8fc7\u9700\u8981\u6709\u4eba\u6276\u6211\u8d77\u6765 | \u6211\u6ca1\u95ee\u9898, \u8ddf\u6211\u9700\u8981\u6709\u4eba\u8ddf\u6211\u53bb\u6765 | HELP | REASSURANCE |
| self_reply | zh | \u4e0d\u8981\u7d27\uff0c\u53ef\u662f\u6211\u8d70\u4e0d\u4e86\u8def | \u4e0d\u8981\u7d27, \u53ef\u662f\u6211\u8d70\u4e0d\u804a\u8def | HELP | REASSURANCE |
| self_noise_5dB | zh | \u7b49\u4e00\u4e0b\uff0c\u6211\u60f3\u4e00\u60f3 | \u53eb\u6211\u4eec\u4e00\u4e0b, \u6211\u60f3\u6b32\u60f3 | OTHER | UNCERTAIN |
| self_noise_5dB | zh | \u4e0d\u5bf9 | \u4e0d\u817f | OTHER | UNCERTAIN |
| self_reply | zh | \u8fd8\u597d\u5566 | \u8fd8\u597d\u5462? | REASSURANCE | OTHER |
| self_reply | zh | \u6211\u6ca1\u6454\u4f24 | \u6211\u6ca1\u6454\u4e0a | REASSURANCE | OTHER |
