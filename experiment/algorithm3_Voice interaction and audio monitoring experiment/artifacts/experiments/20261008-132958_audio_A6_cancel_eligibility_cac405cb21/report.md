# audio / A6_cancel_eligibility

- run_id: `20261008-132958_audio_A6_cancel_eligibility_cac405cb21`
- status: **completed**
- data: Piper libritts_r (904 synthetic voices) saying the phrase bank; ESC-50 household noise; LibriSpeech dev-clean-2 for a real-speech check
- code: `5dde83b9a180` · config `cac405cb21` · split `None` · seed `42`

## Headline
At speaker threshold FAR 0.1% (calibrated on 150 dev speakers): true-cancel 39.00% (234/600); other person saying the same words eligible 0.04% (32/89400) (intent-only 88.50% (79119/89400)); TV/radio 0.01% (3/44700); replay of the enrolled voice 2.67% (8/300); enrolled speaker saying HELP 0.00% (0/450). 150 eval speakers, synthetic clean speech.

Engine: piper:en_US-libritts_r-medium (904 speakers); dev speakers 150, eval speakers 150, eval clips 2370; ASR faster-whisper tiny, intent rule v2, speaker model voxceleb_resnet34_LM.onnx; min speech 1.0 s, min ASR avg_logprob -1.0.

All voices are synthetic and clean (read-style TTS); no older adults, no real household acoustics. Rates are rates of an EVIDENCE flag that is never acted on.

## Thresholds (calibrated on dev speakers only)

| operating point | threshold | dev FAR achieved | dev TAR | dev impostor trials | dev genuine trials | reliable |
|---|---|---|---|---|---|---|
| FAR 0.1% | 0.686 | 0.001 | 0.653 | 31737 | 213 | True |
| FAR 1% | 0.604 | 0.010 | 0.925 | 31737 | 213 | True |

Dev EER (speaker layer, clips with enough speech): 0.028169014084507026. A false-accept rate of 0.1% needs at least 10,000 impostor trials to mean anything. Trials here are pairs of clips and enrolled users built from a limited set of speakers, so they are not independent: the identity-bootstrap intervals (which resample speakers) are the more honest ones, and they are wider than the Wilson intervals.

Speaker-layer EER by length of speech, dev speakers (justifies the minimum speech length; EER shown only with >= 30 genuine and >= 300 impostor trials):

| speech_s | genuine | impostor | eer |
|---|---|---|---|
| 0-0.75 | 98 | 14602 | 0.091 |
| 0.75-1 | 289 | 43061 | 0.038 |
| 1-1.5 | 162 | 24138 | 0.025 |
| 1.5-2 | 45 | 6705 | 0.022 |
| >=2 | 6 | 894 | n/a |

## Speaker layer on the eval speakers (disjoint from dev)

EER 0.02597402597402598 on 462 genuine / 68838 impostor trials (clips with at least the minimum speech); genuine scores mean 0.700, impostor scores mean 0.245, impostor 99.9th percentile 0.669.

| speech_s | genuine | impostor | eer |
|---|---|---|---|
| 0-0.75 | 6 | 894 | n/a |
| 0.75-1 | 132 | 19668 | 0.023 |
| 1-1.5 | 366 | 54534 | 0.025 |
| 1.5-2 | 88 | 13112 | 0.020 |
| >=2 | 8 | 1192 | n/a |

At the dev thresholds:

| operating point | threshold | TAR | FAR |
|---|---|---|---|
| far_0.1pct | 0.686 | 59.31% (274/462) [54.77, 63.69] | 0.06% (38/68838) [0.04, 0.08] |
| far_1pct | 0.604 | 88.96% (411/462) [85.78, 91.50] | 0.64% (442/68838) [0.59, 0.70] |

## Scenarios

Trial = one clip scored against one enrolled user. 'intent only' = what a bare spoken reply without voiceprint would have cancelled (intent == REASSURANCE). 'eligible' = all gate conditions (speech length, reassurance intent, ASR confidence, voiceprint). Intervals in brackets are Wilson 95% (trials treated as independent: too narrow); 'speaker-bootstrap' resamples speakers.


### Operating point FAR 0.1% (threshold 0.686)

| scenario | should cancel | clips | trials | intent only | eligible | speaker-bootstrap 95% | voiceprint alone passes |
|---|---|---|---|---|---|---|---|
| true_cancel_clean | yes | 600 | 600 | 88.50% (531/600) [85.70, 90.81] | 39.00% (234/600) [35.18, 42.96] | [35.00, 43.00] | 45.67% (274/600) |
| true_cancel_noise5dB | yes | 300 | 300 | 85.33% (256/300) [80.88, 88.89] | 18.00% (54/300) [14.07, 22.74] | [14.00, 22.67] | 20.67% (62/300) |
| other_person_same_words | no (false cancel) | 600 | 89400 | 88.50% (79119/89400) [88.29, 88.71] | 0.04% (32/89400) [0.03, 0.05] | [0.02, 0.06] | 0.04% (38/89400) |
| other_person_noise5dB | no (false cancel) | 300 | 44700 | 85.33% (38144/44700) [85.00, 85.66] | 0.01% (4/44700) [0.00, 0.02] | [0.00, 0.03] | 0.01% (4/44700) |
| visitor_on_behalf | no (false cancel) | 300 | 44700 | 87.00% (38889/44700) [86.69, 87.31] | 0.06% (29/44700) [0.05, 0.09] | [0.03, 0.12] | 0.09% (39/44700) |
| tv_radio_speech | no (false cancel) | 300 | 44700 | 80.33% (35909/44700) [79.96, 80.70] | 0.01% (3/44700) [0.00, 0.02] | [0.00, 0.02] | 0.01% (3/44700) |
| replay_of_enrolled_voice | no (false cancel) | 300 | 300 | 75.67% (227/300) [70.51, 80.18] | 2.67% (8/300) [1.36, 5.17] | [1.00, 4.67] | 3.67% (11/300) |
| household_noise_only | no (false cancel) | 100 | 15000 | 0.00% (0/15000) [0.00, 0.03] | 0.00% (0/15000) [0.00, 0.03] | 0 events (interval degenerate; use the Wilson upper bound) | 0.00% (0/15000) |
| silence | no (false cancel) | 20 | 3000 | 0.00% (0/3000) [0.00, 0.13] | 0.00% (0/3000) [0.00, 0.13] | 0 events (interval degenerate; use the Wilson upper bound) | 0.00% (0/3000) |
| enrolled_says_help | no (false cancel) | 450 | 450 | 0.00% (0/450) [0.00, 0.85] | 0.00% (0/450) [0.00, 0.85] | 0 events (interval degenerate; use the Wilson upper bound) | 59.11% (266/450) |

### Operating point FAR 1% (threshold 0.604)

| scenario | should cancel | clips | trials | intent only | eligible | speaker-bootstrap 95% | voiceprint alone passes |
|---|---|---|---|---|---|---|---|
| true_cancel_clean | yes | 600 | 600 | 88.50% (531/600) [85.70, 90.81] | 58.50% (351/600) [54.52, 62.38] | [54.00, 62.67] | 68.50% (411/600) |
| true_cancel_noise5dB | yes | 300 | 300 | 85.33% (256/300) [80.88, 88.89] | 41.33% (124/300) [35.90, 46.98] | [36.00, 46.67] | 47.00% (141/300) |
| other_person_same_words | no (false cancel) | 600 | 89400 | 88.50% (79119/89400) [88.29, 88.71] | 0.42% (378/89400) [0.38, 0.47] | [0.31, 0.57] | 0.49% (442/89400) |
| other_person_noise5dB | no (false cancel) | 300 | 44700 | 85.33% (38144/44700) [85.00, 85.66] | 0.14% (63/44700) [0.11, 0.18] | [0.07, 0.23] | 0.15% (69/44700) |
| visitor_on_behalf | no (false cancel) | 300 | 44700 | 87.00% (38889/44700) [86.69, 87.31] | 0.79% (355/44700) [0.72, 0.88] | [0.58, 1.08] | 1.01% (450/44700) |
| tv_radio_speech | no (false cancel) | 300 | 44700 | 80.33% (35909/44700) [79.96, 80.70] | 0.11% (49/44700) [0.08, 0.14] | [0.05, 0.19] | 0.13% (58/44700) |
| replay_of_enrolled_voice | no (false cancel) | 300 | 300 | 75.67% (227/300) [70.51, 80.18] | 15.00% (45/300) [11.40, 19.48] | [11.00, 19.00] | 18.33% (55/300) |
| household_noise_only | no (false cancel) | 100 | 15000 | 0.00% (0/15000) [0.00, 0.03] | 0.00% (0/15000) [0.00, 0.03] | 0 events (interval degenerate; use the Wilson upper bound) | 0.00% (0/15000) |
| silence | no (false cancel) | 20 | 3000 | 0.00% (0/3000) [0.00, 0.13] | 0.00% (0/3000) [0.00, 0.13] | 0 events (interval degenerate; use the Wilson upper bound) | 0.00% (0/3000) |
| enrolled_says_help | no (false cancel) | 450 | 450 | 0.00% (0/450) [0.00, 0.85] | 0.00% (0/450) [0.00, 0.85] | 0 events (interval degenerate; use the Wilson upper bound) | 83.33% (375/450) |

## Why trials failed (gate reason codes, strict operating point, up to 3000 sampled trials per scenario)

| scenario | sampled | eligible | too_short | intent_unavailable | intent_not_reassurance | low_asr_confidence | speaker_mismatch | no_threshold |
|---|---|---|---|---|---|---|---|---|
| true_cancel_clean | 600 | 234 | 138 | 0 | 69 | 1 | 290 | 0 |
| true_cancel_noise5dB | 300 | 54 | 6 | 0 | 44 | 2 | 238 | 0 |
| other_person_same_words | 3000 | 1 | 672 | 0 | 359 | 5 | 2999 | 0 |
| other_person_noise5dB | 3000 | 1 | 54 | 0 | 438 | 22 | 2999 | 0 |
| visitor_on_behalf | 3000 | 1 | 13 | 0 | 413 | 0 | 2999 | 0 |
| tv_radio_speech | 3000 | 0 | 0 | 0 | 599 | 0 | 3000 | 0 |
| replay_of_enrolled_voice | 300 | 8 | 94 | 0 | 73 | 1 | 289 | 0 |
| household_noise_only | 3000 | 0 | 0 | 0 | 3000 | 0 | 3000 | 0 |
| silence | 3000 | 0 | 3000 | 3000 | 0 | 0 | 0 | 0 |
| enrolled_says_help | 450 | 0 | 25 | 0 | 450 | 0 | 183 | 0 |

A trial can fail for several reasons at once; intent_unavailable = no speech was found (NO_RESPONSE) or the ASR failed.

Consistency audit: the product gate (`CancelGate.evaluate` on the audio) was run on 199 random trials and disagreed with the vectorised analysis in 0 of them.

## Real-speech transfer check (LibriSpeech, read English, speaker layer only)

26 real speakers, 208 probes of 2.5 s; 208 genuine / 5200 impostor trials; EER 0.0004. The thresholds below were chosen on TTS dev speakers and never tuned here.

| operating point | threshold | TAR | FAR |
|---|---|---|---|
| far_0.1pct | 0.686 | 82.21% (171/208) [76.44, 86.81] | 0.00% (0/5200) [0.00, 0.07] |
| far_1pct | 0.604 | 93.75% (195/208) [89.60, 96.31] | 0.00% (0/5200) [0.00, 0.07] |

Real impostor scores: mean 0.100, 99th percentile 0.392; genuine mean 0.744.

## Reading these numbers

* Cancel-eligible is EVIDENCE: no code path lets it change an event (A5 and tests/test_cancel_policy.py check that an EMERGENCY stays EMERGENCY/OPEN whatever it says).
* The voiceprint cannot tell a live reply from a replay of the same voice: replay_of_enrolled_voice measures how much the loudspeaker channel alone changes the score.
* Synthetic voices from one engine are more separable (or less) than real people in a real room; the real-speech check above is the only real-voice evidence and it is read, clean English.
* English only. Mandarin is not evaluated: no multi-speaker Mandarin voice is available within the download budget (one SAPI voice, Huihui).
