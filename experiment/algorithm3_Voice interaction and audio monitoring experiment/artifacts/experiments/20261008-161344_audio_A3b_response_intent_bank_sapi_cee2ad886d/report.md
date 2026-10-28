# audio / A3b_response_intent_bank_sapi

- run_id: `20261008-161344_audio_A3b_response_intent_bank_sapi_cee2ad886d`
- status: **completed**
- data: phrase bank config/audio_phrases.yaml rendered with sapi voices (Windows) + LibriSpeech (TV speech) + ESC-50 noise
- code: `354f1478138c` · config `cee2ad886d` · split `None` · seed `42`

## Headline
selected on dev: tiny|hotwords|auto + rule_v2; TEST macro-F1 0.875 [0.81, 0.93] vs baseline 0.480 [0.37, 0.58]; self HELP missed 7/38 (baseline 23/38); TV/visitor read as REASSURANCE 20/80

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

Rule: highest dev macro-F1 over HELP/REASSURANCE/OTHER among DEPLOYABLE cells (language not forced) with median ASR latency <= 3.0 s; ties within 0.005 go to fewer missed self calls for help, then lower latency.

**Selected: ASR `tiny|hotwords|auto` + intent `rule_v2`**

Top 12 dev cells (cells marked oracle tell the ASR the clip's language; they are reference only and cannot be selected):

| asr (model|bias|language) | intent | macro_f1 | self_help_missed | asr_median_s |
|---|---|---|---|---|
| base|hotwords|forced  [oracle language] | rule_v2 | 0.972 | 1 | 0.640 |
| tiny|prompt|forced  [oracle language] | rule_v2 | 0.969 | 3 | 0.305 |
| tiny|prompt|auto | rule_v2 | 0.961 | 3 | 0.605 |
| tiny|hotwords|auto | rule_v2 | 0.960 | 2 | 0.618 |
| base|prompt|forced  [oracle language] | rule_v2 | 0.957 | 2 | 0.629 |
| tiny|hotwords|auto_en_zh | rule_v2 | 0.955 | 2 | 0.612 |
| base|none|forced  [oracle language] | rule_v2 | 0.954 | 5 | 0.625 |
| base|none|auto | rule_v2 | 0.951 | 5 | 1.205 |
| base|none|auto_en_zh | rule_v2 | 0.951 | 5 | 1.198 |
| tiny|prompt|auto_en_zh | rule_v2 | 0.946 | 3 | 0.629 |
| tiny|hotwords|forced  [oracle language] | rule_v2 | 0.945 | 2 | 0.337 |
| tiny|none|forced  [oracle language] | rule_v2 | 0.940 | 7 | 0.328 |

Production baseline before this step on dev:

| asr (model|bias|language) | intent | macro_f1 | accuracy | n | self_help_missed | nonself_reassurance | asr_median_s | asr_p95_s |
|---|---|---|---|---|---|---|---|---|
| tiny|none|auto | rule_v1 | 0.605 | 0.628 | 196 | 19/38 | 5/80 | 0.600 | 1.054 |

Main effects on dev (mean dev macro-F1 over all cells of the other factors):

| factor | level | mean_dev_macro_f1 | cells |
|---|---|---|---|
| ASR model | tiny | 0.720 | 54 |
| ASR model | base | 0.706 | 54 |
| decoding bias | none | 0.687 | 36 |
| decoding bias | hotwords | 0.730 | 36 |
| decoding bias | prompt | 0.723 | 36 |
| language | forced | 0.721 | 36 |
| language | auto | 0.711 | 36 |
| language | auto_en_zh | 0.708 | 36 |
| intent method | rule_v1 | 0.680 | 18 |
| intent method | rule_v1_t2s | 0.722 | 18 |
| intent method | rule_v2 | 0.945 | 18 |
| intent method | sup_legacy | 0.600 | 18 |
| intent method | sup_dev | 0.613 | 18 |
| intent method | combined | 0.719 | 18 |

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
| baseline: tiny|none|auto + rule_v1 | 0.480 | [0.37, 0.58] | 0.546 | 196 | 23/38 | 3/80 | 0.610 | 1.225 |
| selected: tiny|hotwords|auto + rule_v2 | 0.875 | [0.81, 0.93] | 0.867 | 196 | 7/38 | 20/80 | 0.624 | 9.011 |

CI = family/speaker-grouped bootstrap (1000 resamples). Rates below carry Wilson intervals.

### One factor at a time around the selected cell (test)

| change | asr (model|bias|language) | intent | macro_f1 | accuracy | n | self_help_missed | nonself_reassurance | asr_median_s | asr_p95_s |
|---|---|---|---|---|---|---|---|---|---|
| selected (dev) | tiny|hotwords|auto | rule_v2 | 0.875 | 0.867 | 196 | 7/38 | 20/80 | 0.624 | 9.011 |
| ASR bias | tiny|none|auto | rule_v2 | 0.894 | 0.893 | 196 | 5/38 | 17/80 | 0.610 | 1.225 |
| ASR model | base|hotwords|auto | rule_v2 | 0.911 | 0.893 | 196 | 7/38 | 21/80 | 1.228 | 19.753 |
| ASR bias | tiny|prompt|auto | rule_v2 | 0.876 | 0.872 | 196 | 6/38 | 16/80 | 0.607 | 4.523 |
| ASR language | tiny|hotwords|forced | rule_v2 | 0.900 | 0.898 | 196 | 5/38 | 20/80 | 0.319 | 4.962 |
| ASR language | tiny|hotwords|auto_en_zh | rule_v2 | 0.873 | 0.867 | 196 | 7/38 | 19/80 | 0.604 | 8.819 |
| intent method | tiny|hotwords|auto | rule_v1 | 0.578 | 0.602 | 196 | 22/38 | 6/80 | 0.624 | 9.011 |
| intent method | tiny|hotwords|auto | rule_v1_t2s | 0.578 | 0.602 | 196 | 22/38 | 6/80 | 0.624 | 9.011 |
| intent method | tiny|hotwords|auto | sup_legacy | 0.461 | 0.327 | 196 | 26/38 | 7/80 | 0.624 | 9.011 |
| intent method | tiny|hotwords|auto | sup_dev | 0.597 | 0.474 | 196 | 26/38 | 12/80 | 0.624 | 9.011 |
| intent method | tiny|hotwords|auto | combined | 0.715 | 0.597 | 196 | 7/38 | 10/80 | 0.624 | 9.011 |

### Selected cell on test: per-class precision / recall / F1

| class | support | precision | precision_CI | recall | recall_CI | f1 |
|---|---|---|---|---|---|---|
| HELP | 62 | 0.942 | [0.84, 0.98] | 0.790 | [0.67, 0.87] | 0.860 |
| REASSURANCE | 60 | 0.945 | [0.85, 0.98] | 0.867 | [0.76, 0.93] | 0.904 |
| OTHER | 70 | 0.802 | [0.70, 0.87] | 0.929 | [0.84, 0.97] | 0.861 |
| UNCERTAIN | 0 | 0.000 | [0.00, 0.49] | n/a | n/a | n/a |
| NO_RESPONSE | 4 | 1.000 | [0.51, 1.00] | 1.000 | [0.51, 1.00] | 1.000 |

### Confusion matrix (rows = truth)

| true \ pred | HELP | REASSURANCE | OTHER | UNCERTAIN | NO_RESPONSE | n |
|---|---|---|---|---|---|---|
| HELP | 49 | 3 | 10 | 0 | 0 | 62 |
| REASSURANCE | 2 | 52 | 6 | 0 | 0 | 60 |
| OTHER | 1 | 0 | 65 | 4 | 0 | 70 |
| NO_RESPONSE | 0 | 0 | 0 | 0 | 4 | 4 |

### Accuracy per interference bucket

| bucket | n | correct | accuracy | CI |
|---|---|---|---|---|
| self_noise_5dB | 28 | 23 | 0.821 | [0.64, 0.92] |
| self_reply | 84 | 75 | 0.893 | [0.81, 0.94] |
| silence | 4 | 4 | 1.000 | [0.51, 1.00] |
| tv_says_help | 12 | 7 | 0.583 | [0.32, 0.81] |
| tv_speech | 32 | 26 | 0.812 | [0.65, 0.91] |
| visitor_answer | 36 | 35 | 0.972 | [0.86, 1.00] |

### Safety counts

| safety count | value |
|---|---|
| self call for help NOT recognised as HELP | 7/38 (0.18, CI [0.09, 0.33]) |
|   of which recognised as REASSURANCE | 3/38 (0.08) |
| TV / visitor speech recognised as REASSURANCE (would be false evidence without a voiceprint) | 20/80 (0.25, CI [0.17, 0.35]) |
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
| tiny|none|auto | 12 | 12 | 0 | 0 | See you next time! | F**k! | Thank you. |
| tiny|hotwords|auto | 12 | 12 | 1 | 1 | \u8dd1\u8fc7, \u9012\u547d, \u9012\u547d, \u7ea6\u547d, \u9012\u547d, \u9012\u547d, \u8ddd\u79bb \u6211, \u8def, | \u62c5\u5fc3\u554a \u6211\u8981\u7b54\u5e94\u81ea\u5df1\u70e6\u547d! | \u4e0d\u7528\u62c5\u5fc3, \u300b |
| base|hotwords|auto | 12 | 12 | 4 | 0 | \u6551\u62a4\u8f66, \u6551\u62a4\u8f66, \u6551\u62a4\u8f66, \u6551\u62a4\u8f66, \u6551\u62a4\u8f66, \u6551\u62a4\u8f66, \u6551\u62a4\u8f66, \u6551\u62a4\u8f66,  | \u4e0d\u7528\u62c5\u5fc3, \u4e0d\u7528\u62c5\u5fc3, \u6551\u62a4\u8f66\u3002 | \u6551\u62a4\u8f66, \u6551\u62a4\u8f66 \u6551\u62a4\u8f66 \u6551\u62a4\u8f66, |
| tiny|prompt|auto | 12 | 12 | 1 | 0 | \u80fd\u4e0d\u56de\u6765\u7761\u89c9\u3002 \u5750 quiet. \u540e\u8fd9\u5929\u3002 \u4e09\u5929\u3002 | \u8981\u4e0d\u62c5\u5fc3\u5e2e\u6211\u9003 splend \u966a\u4f60\u9c7c\u513f\u5b50\u626d\u58f0 | thanks for watching! |
| tiny|hotwords|forced | 12 | 12 | 2 | 1 | if you don't have anything else to do, d | Who are you? | . |
| tiny|hotwords|auto_en_zh | 12 | 12 | 2 | 1 | But that's it. | \u51e0\u9879, \u8865\u767d\u8272, \u7231\u6211\u548c\u4f60, \u91cc\u9762\u7684\u5be7\u771f\u662f\u65e0\u4eba\u7684\u795e\u8bed\u8a00\u8bed\u8a00\u3002 | \u90fd\u8981\u6c42\u4f60, \u7231\u60f3\u4f60 tooth and soul can never kill  |

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
| self_reply | zh | \u597d\u7684 | \u597d\u7b49 | OTHER | UNCERTAIN |
| self_noise_5dB | zh | \u4e0d\u5bf9 | \u4e0d\u5bf9, \u4e0d\u5bf9\u3002 | OTHER | UNCERTAIN |
| self_reply | zh | \u8fd8\u597d\u5566 | \u8fd8\u597d\u5462? | REASSURANCE | OTHER |
| self_reply | zh | \u6211\u6ca1\u6454\u4f24 | \u6211\u6ca1\u6454\u4e0a | REASSURANCE | OTHER |

### Rule v2 without the cues that only one split's wording triggers (post-selection diagnostic)

| split | input | cues switched off | macro_f1 full | macro_f1 without | self HELP missed full | self HELP missed without |
|---|---|---|---|---|---|---|
| dev | reference text (no ASR) | 9 | 1.000 | 0.948 | 0/38 | 4/38 |
| dev | ASR transcript (tiny|hotwords|auto) | 9 | 0.969 | 0.918 | 2/38 | 6/38 |
| test | reference text (no ASR) | 13 | 0.990 | 0.914 | 1/38 | 9/38 |
| test | ASR transcript (tiny|hotwords|auto) | 13 | 0.885 | 0.827 | 7/38 | 14/38 |

Read as a range: the true generalisation of a lexicon written without sight of the data lies between 'without' (too pessimistic: ordinary words such as pain / slipped are removed) and 'full' (optimistic).

### Rule v2 without the cues that only one split's wording triggers (post-selection diagnostic)

| split | input | cues switched off | macro_f1 full | macro_f1 without | self HELP missed full | self HELP missed without |
|---|---|---|---|---|---|---|
| dev | reference text (no ASR) | 9 | 1.000 | 0.948 | 0/38 | 4/38 |
| dev | ASR transcript (tiny|none|auto) | 9 | 0.940 | 0.898 | 7/38 | 10/38 |
| test | reference text (no ASR) | 13 | 0.990 | 0.914 | 1/38 | 9/38 |
| test | ASR transcript (tiny|none|auto) | 13 | 0.892 | 0.834 | 5/38 | 13/38 |

Read as a range: the true generalisation of a lexicon written without sight of the data lies between 'without' (too pessimistic: ordinary words such as pain / slipped are removed) and 'full' (optimistic).
