# audio / A6_cancel_eligibility

- run_id: `20261008-185642_audio_A6_cancel_eligibility_cac405cb21`
- status: **completed**
- data: Piper libritts_r (904 synthetic voices) saying the phrase bank; ESC-50 household noise; LibriSpeech dev-clean-2 for a real-speech check
- code: `354f1478138c` · config `cac405cb21` · split `None` · seed `42`

## Headline
At speaker threshold FAR 0.1% (calibrated on 150 dev speakers): true-cancel 41.67% (250/600); other person saying the same words eligible 0.05% (48/89400) (intent-only 88.50% (79119/89400)); TV/radio 0.00% (2/44700); replay of the enrolled voice 5.33% (16/300); enrolled speaker saying HELP 0.00% (0/450). 150 eval speakers, synthetic clean speech.

Engine: piper:en_US-libritts_r-medium (904 speakers); dev speakers 150, eval speakers 150, eval clips 2370; ASR faster-whisper tiny, intent rule v2, speaker model voxceleb_resnet34_LM.onnx; min speech 1.0 s, min ASR avg_logprob -1.0.

All voices are synthetic and clean (read-style TTS); no older adults, no real household acoustics. Rates are rates of an EVIDENCE flag that is never acted on.

## Thresholds (calibrated on dev speakers only)

| operating point | threshold | dev FAR achieved | dev TAR | dev impostor trials | dev genuine trials | reliable |
|---|---|---|---|---|---|---|
| FAR 0.1% | 0.682 | 0.001 | 0.679 | 32035 | 215 | True |
| FAR 1% | 0.605 | 0.010 | 0.940 | 32035 | 215 | True |

Dev EER (speaker layer, clips with enough speech): 0.023255813953488393. A false-accept rate of 0.1% needs at least 10,000 impostor trials to mean anything. Trials here are pairs of clips and enrolled users built from a limited set of speakers, so they are not independent: the identity-bootstrap intervals (which resample speakers) are the more honest ones, and they are wider than the Wilson intervals.

Speaker-layer EER by length of speech, dev speakers (justifies the minimum speech length; EER shown only with >= 30 genuine and >= 300 impostor trials):

| speech_s | genuine | impostor | eer |
|---|---|---|---|
| 0-0.75 | 102 | 15198 | 0.079 |
| 0.75-1 | 283 | 42167 | 0.042 |
| 1-1.5 | 163 | 24287 | 0.018 |
| 1.5-2 | 45 | 6705 | 0.022 |
| >=2 | 7 | 1043 | n/a |

## Speaker layer on the eval speakers (disjoint from dev)

EER 0.025451618929911025 on 467 genuine / 69583 impostor trials (clips with at least the minimum speech); genuine scores mean 0.704, impostor scores mean 0.246, impostor 99.9th percentile 0.673.

| speech_s | genuine | impostor | eer |
|---|---|---|---|
| 0-0.75 | 6 | 894 | n/a |
| 0.75-1 | 127 | 18923 | 0.024 |
| 1-1.5 | 377 | 56173 | 0.021 |
| 1.5-2 | 81 | 12069 | 0.009 |
| >=2 | 9 | 1341 | n/a |

At the dev thresholds:

| operating point | threshold | TAR | FAR |
|---|---|---|---|
| far_0.1pct | 0.682 | 62.53% (292/467) [58.05, 66.80] | 0.08% (55/69583) [0.06, 0.10] |
| far_1pct | 0.605 | 89.94% (420/467) [86.87, 92.35] | 0.63% (439/69583) [0.57, 0.69] |

## Scenarios

Trial = one clip scored against one enrolled user. 'intent only' = what a bare spoken reply without voiceprint would have cancelled (intent == REASSURANCE). 'eligible' = all gate conditions (speech length, reassurance intent, ASR confidence, voiceprint). Intervals in brackets are Wilson 95% (trials treated as independent: too narrow); 'speaker-bootstrap' resamples speakers.


### Operating point FAR 0.1% (threshold 0.682)

| scenario | should cancel | clips | trials | intent only | eligible | speaker-bootstrap 95% | voiceprint alone passes |
|---|---|---|---|---|---|---|---|
| true_cancel_clean | yes | 600 | 600 | 88.50% (531/600) [85.70, 90.81] | 41.67% (250/600) [37.79, 45.65] | [37.67, 45.83] | 48.67% (292/600) |
| true_cancel_noise5dB | yes | 300 | 300 | 81.33% (244/300) [76.54, 85.34] | 21.33% (64/300) [17.08, 26.32] | [17.00, 26.33] | 25.00% (75/300) |
| other_person_same_words | no (false cancel) | 600 | 89400 | 88.50% (79119/89400) [88.29, 88.71] | 0.05% (48/89400) [0.04, 0.07] | [0.03, 0.09] | 0.06% (55/89400) |
| other_person_noise5dB | no (false cancel) | 300 | 44700 | 81.33% (36356/44700) [80.97, 81.69] | 0.00% (1/44700) [0.00, 0.01] | [0.00, 0.01] | 0.00% (1/44700) |
| visitor_on_behalf | no (false cancel) | 300 | 44700 | 87.33% (39038/44700) [87.02, 87.64] | 0.09% (41/44700) [0.07, 0.12] | [0.04, 0.16] | 0.12% (53/44700) |
| tv_radio_speech | no (false cancel) | 300 | 44700 | 78.00% (34866/44700) [77.61, 78.38] | 0.00% (2/44700) [0.00, 0.02] | [0.00, 0.02] | 0.00% (2/44700) |
| replay_of_enrolled_voice | no (false cancel) | 300 | 300 | 78.33% (235/300) [73.33, 82.62] | 5.33% (16/300) [3.31, 8.49] | [3.33, 8.00] | 5.33% (16/300) |
| household_noise_only | no (false cancel) | 100 | 15000 | 0.00% (0/15000) [0.00, 0.03] | 0.00% (0/15000) [0.00, 0.03] | 0 events (interval degenerate; use the Wilson upper bound) | 0.00% (0/15000) |
| silence | no (false cancel) | 20 | 3000 | 0.00% (0/3000) [0.00, 0.13] | 0.00% (0/3000) [0.00, 0.13] | 0 events (interval degenerate; use the Wilson upper bound) | 0.00% (0/3000) |
| enrolled_says_help | no (false cancel) | 450 | 450 | 0.00% (0/450) [0.00, 0.85] | 0.00% (0/450) [0.00, 0.85] | 0 events (interval degenerate; use the Wilson upper bound) | 60.89% (274/450) |

### Operating point FAR 1% (threshold 0.605)

| scenario | should cancel | clips | trials | intent only | eligible | speaker-bootstrap 95% | voiceprint alone passes |
|---|---|---|---|---|---|---|---|
| true_cancel_clean | yes | 600 | 600 | 88.50% (531/600) [85.70, 90.81] | 60.00% (360/600) [56.03, 63.84] | [55.33, 64.50] | 70.00% (420/600) |
| true_cancel_noise5dB | yes | 300 | 300 | 81.33% (244/300) [76.54, 85.34] | 38.67% (116/300) [33.33, 44.29] | [33.33, 43.67] | 44.67% (134/300) |
| other_person_same_words | no (false cancel) | 600 | 89400 | 88.50% (79119/89400) [88.29, 88.71] | 0.41% (367/89400) [0.37, 0.45] | [0.29, 0.55] | 0.49% (439/89400) |
| other_person_noise5dB | no (false cancel) | 300 | 44700 | 81.33% (36356/44700) [80.97, 81.69] | 0.11% (47/44700) [0.08, 0.14] | [0.05, 0.17] | 0.14% (61/44700) |
| visitor_on_behalf | no (false cancel) | 300 | 44700 | 87.33% (39038/44700) [87.02, 87.64] | 0.74% (329/44700) [0.66, 0.82] | [0.54, 0.98] | 0.92% (411/44700) |
| tv_radio_speech | no (false cancel) | 300 | 44700 | 78.00% (34866/44700) [77.61, 78.38] | 0.09% (41/44700) [0.07, 0.12] | [0.04, 0.17] | 0.11% (47/44700) |
| replay_of_enrolled_voice | no (false cancel) | 300 | 300 | 78.33% (235/300) [73.33, 82.62] | 17.00% (51/300) [13.17, 21.66] | [13.00, 21.67] | 20.33% (61/300) |
| household_noise_only | no (false cancel) | 100 | 15000 | 0.00% (0/15000) [0.00, 0.03] | 0.00% (0/15000) [0.00, 0.03] | 0 events (interval degenerate; use the Wilson upper bound) | 0.00% (0/15000) |
| silence | no (false cancel) | 20 | 3000 | 0.00% (0/3000) [0.00, 0.13] | 0.00% (0/3000) [0.00, 0.13] | 0 events (interval degenerate; use the Wilson upper bound) | 0.00% (0/3000) |
| enrolled_says_help | no (false cancel) | 450 | 450 | 0.00% (0/450) [0.00, 0.85] | 0.00% (0/450) [0.00, 0.85] | 0 events (interval degenerate; use the Wilson upper bound) | 83.56% (376/450) |

## Why trials failed (gate reason codes, strict operating point, up to 3000 sampled trials per scenario)

| scenario | sampled | eligible | too_short | intent_unavailable | intent_not_reassurance | low_asr_confidence | speaker_mismatch | no_threshold |
|---|---|---|---|---|---|---|---|---|
| true_cancel_clean | 600 | 250 | 133 | 0 | 69 | 3 | 275 | 0 |
| true_cancel_noise5dB | 300 | 64 | 6 | 0 | 56 | 2 | 225 | 0 |
| other_person_same_words | 3000 | 4 | 671 | 0 | 372 | 14 | 2996 | 0 |
| other_person_noise5dB | 3000 | 0 | 45 | 0 | 566 | 21 | 3000 | 0 |
| visitor_on_behalf | 3000 | 1 | 6 | 0 | 375 | 0 | 2998 | 0 |
| tv_radio_speech | 3000 | 0 | 0 | 0 | 686 | 0 | 3000 | 0 |
| replay_of_enrolled_voice | 300 | 16 | 91 | 0 | 65 | 2 | 284 | 0 |
| household_noise_only | 3000 | 0 | 0 | 0 | 3000 | 0 | 3000 | 0 |
| silence | 3000 | 0 | 3000 | 3000 | 0 | 0 | 0 | 0 |
| enrolled_says_help | 450 | 0 | 21 | 0 | 450 | 0 | 175 | 0 |

A trial can fail for several reasons at once; intent_unavailable = no speech was found (NO_RESPONSE) or the ASR failed.

Consistency audit: the product gate (`CancelGate.evaluate` on the audio) was run on 199 random trials and disagreed with the vectorised analysis in 0 of them.

## Real-speech transfer check (LibriSpeech, read English, speaker layer only)

26 real speakers, 208 probes of 2.5 s; 208 genuine / 5200 impostor trials; EER 0.0004. The thresholds below were chosen on TTS dev speakers and never tuned here.

| operating point | threshold | TAR | FAR |
|---|---|---|---|
| far_0.1pct | 0.682 | 83.17% (173/208) [77.50, 87.65] | 0.00% (0/5200) [0.00, 0.07] |
| far_1pct | 0.605 | 93.75% (195/208) [89.60, 96.31] | 0.00% (0/5200) [0.00, 0.07] |

Real impostor scores: mean 0.100, 99th percentile 0.392; genuine mean 0.744.

## Reading these numbers

* Cancel-eligible is EVIDENCE: no code path lets it change an event (A5 and tests/test_cancel_policy.py check that an EMERGENCY stays EMERGENCY/OPEN whatever it says).
* The voiceprint cannot tell a live reply from a replay of the same voice: replay_of_enrolled_voice measures how much the loudspeaker channel alone changes the score.
* Synthetic voices from one engine are more separable (or less) than real people in a real room; the real-speech check above is the only real-voice evidence and it is read, clean English.
* English only. Mandarin is not evaluated: no multi-speaker Mandarin voice is available within the download budget (one SAPI voice, Huihui).
