# audio / A6_cancel_eligibility

- run_id: `20261008-113720_audio_A6_cancel_eligibility_cac405cb21`
- status: **completed**
- data: Piper libritts_r (904 synthetic voices) saying the phrase bank; ESC-50 household noise; LibriSpeech dev-clean-2 for a real-speech check
- code: `4b77bcb56eef` · config `cac405cb21` · split `None` · seed `42`

## Headline
At speaker threshold FAR 0.1% (calibrated on 150 dev speakers): true-cancel 42.33% (254/600); other person saying the same words eligible 0.06% (50/89400) (intent-only 88.17% (78821/89400)); TV/radio 0.00% (1/44700); replay of the enrolled voice 4.33% (13/300); enrolled speaker saying HELP 0.00% (0/450). 150 eval speakers, synthetic clean speech.

Engine: piper:en_US-libritts_r-medium (904 speakers); dev speakers 150, eval speakers 150, eval clips 2370; ASR faster-whisper tiny, intent rule v2, speaker model voxceleb_resnet34_LM.onnx; min speech 1.0 s, min ASR avg_logprob -1.0.

All voices are synthetic and clean (read-style TTS); no older adults, no real household acoustics. Rates are rates of an EVIDENCE flag that is never acted on.

## Thresholds (calibrated on dev speakers only)

| operating point | threshold | dev FAR achieved | dev TAR | dev impostor trials | dev genuine trials | reliable |
|---|---|---|---|---|---|---|
| FAR 0.1% | 0.679 | 0.001 | 0.637 | 30396 | 204 | True |
| FAR 1% | 0.601 | 0.010 | 0.936 | 30396 | 204 | True |

Dev EER (speaker layer, clips with enough speech): 0.0191637057507567. A false-accept rate of 0.1% needs at least 10,000 impostor trials to mean anything. Trials here are pairs of clips and enrolled users built from a limited set of speakers, so they are not independent: the identity-bootstrap intervals (which resample speakers) are the more honest ones, and they are wider than the Wilson intervals.

Speaker-layer EER by length of speech, dev speakers (justifies the minimum speech length; EER shown only with >= 30 genuine and >= 300 impostor trials):

| speech_s | genuine | impostor | eer |
|---|---|---|---|
| 0-0.75 | 104 | 15496 | 0.087 |
| 0.75-1 | 292 | 43508 | 0.044 |
| 1-1.5 | 152 | 22648 | 0.013 |
| 1.5-2 | 42 | 6258 | 0.024 |
| >=2 | 10 | 1490 | n/a |

## Speaker layer on the eval speakers (disjoint from dev)

EER 0.023404255319148925 on 470 genuine / 70030 impostor trials (clips with at least the minimum speech); genuine scores mean 0.702, impostor scores mean 0.245, impostor 99.9th percentile 0.670.

| speech_s | genuine | impostor | eer |
|---|---|---|---|
| 0-0.75 | 7 | 1043 | n/a |
| 0.75-1 | 123 | 18327 | 0.033 |
| 1-1.5 | 373 | 55577 | 0.022 |
| 1.5-2 | 86 | 12814 | 0.012 |
| >=2 | 11 | 1639 | n/a |

At the dev thresholds:

| operating point | threshold | TAR | FAR |
|---|---|---|---|
| far_0.1pct | 0.679 | 62.13% (292/470) [57.66, 66.40] | 0.08% (56/70030) [0.06, 0.10] |
| far_1pct | 0.601 | 89.36% (420/470) [86.25, 91.84] | 0.76% (529/70030) [0.69, 0.82] |

## Scenarios

Trial = one clip scored against one enrolled user. 'intent only' = what a bare spoken reply without voiceprint would have cancelled (intent == REASSURANCE). 'eligible' = all gate conditions (speech length, reassurance intent, ASR confidence, voiceprint). Intervals in brackets are Wilson 95% (trials treated as independent: too narrow); 'speaker-bootstrap' resamples speakers.


### Operating point FAR 0.1% (threshold 0.679)

| scenario | should cancel | clips | trials | intent only | eligible | speaker-bootstrap 95% | voiceprint alone passes |
|---|---|---|---|---|---|---|---|
| true_cancel_clean | yes | 600 | 600 | 88.17% (529/600) [85.34, 90.51] | 42.33% (254/600) [38.44, 46.32] | [38.00, 46.17] | 48.67% (292/600) |
| true_cancel_noise5dB | yes | 300 | 300 | 80.67% (242/300) [75.82, 84.74] | 19.33% (58/300) [15.26, 24.18] | [15.00, 24.00] | 22.33% (67/300) |
| other_person_same_words | no (false cancel) | 600 | 89400 | 88.17% (78821/89400) [87.95, 88.38] | 0.06% (50/89400) [0.04, 0.07] | [0.03, 0.10] | 0.06% (56/89400) |
| other_person_noise5dB | no (false cancel) | 300 | 44700 | 80.67% (36058/44700) [80.30, 81.03] | 0.00% (1/44700) [0.00, 0.01] | [0.00, 0.01] | 0.00% (2/44700) |
| visitor_on_behalf | no (false cancel) | 300 | 44700 | 88.33% (39485/44700) [88.03, 88.63] | 0.08% (35/44700) [0.06, 0.11] | [0.04, 0.14] | 0.11% (51/44700) |
| tv_radio_speech | no (false cancel) | 300 | 44700 | 80.00% (35760/44700) [79.63, 80.37] | 0.00% (1/44700) [0.00, 0.01] | [0.00, 0.01] | 0.00% (1/44700) |
| replay_of_enrolled_voice | no (false cancel) | 300 | 300 | 80.00% (240/300) [75.11, 84.13] | 4.33% (13/300) [2.55, 7.27] | [2.33, 6.67] | 5.33% (16/300) |
| household_noise_only | no (false cancel) | 100 | 15000 | 0.00% (0/15000) [0.00, 0.03] | 0.00% (0/15000) [0.00, 0.03] | 0 events (interval degenerate; use the Wilson upper bound) | 0.00% (0/15000) |
| silence | no (false cancel) | 20 | 3000 | 0.00% (0/3000) [0.00, 0.13] | 0.00% (0/3000) [0.00, 0.13] | 0 events (interval degenerate; use the Wilson upper bound) | 0.00% (0/3000) |
| enrolled_says_help | no (false cancel) | 450 | 450 | 0.00% (0/450) [0.00, 0.85] | 0.00% (0/450) [0.00, 0.85] | 0 events (interval degenerate; use the Wilson upper bound) | 62.67% (282/450) |

### Operating point FAR 1% (threshold 0.601)

| scenario | should cancel | clips | trials | intent only | eligible | speaker-bootstrap 95% | voiceprint alone passes |
|---|---|---|---|---|---|---|---|
| true_cancel_clean | yes | 600 | 600 | 88.17% (529/600) [85.34, 90.51] | 60.00% (360/600) [56.03, 63.84] | [55.33, 64.67] | 70.00% (420/600) |
| true_cancel_noise5dB | yes | 300 | 300 | 80.67% (242/300) [75.82, 84.74] | 42.67% (128/300) [37.20, 48.32] | [37.00, 48.00] | 49.00% (147/300) |
| other_person_same_words | no (false cancel) | 600 | 89400 | 88.17% (78821/89400) [87.95, 88.38] | 0.52% (467/89400) [0.48, 0.57] | [0.38, 0.72] | 0.59% (529/89400) |
| other_person_noise5dB | no (false cancel) | 300 | 44700 | 80.67% (36058/44700) [80.30, 81.03] | 0.21% (96/44700) [0.18, 0.26] | [0.11, 0.36] | 0.24% (107/44700) |
| visitor_on_behalf | no (false cancel) | 300 | 44700 | 88.33% (39485/44700) [88.03, 88.63] | 0.90% (402/44700) [0.82, 0.99] | [0.67, 1.21] | 1.11% (494/44700) |
| tv_radio_speech | no (false cancel) | 300 | 44700 | 80.00% (35760/44700) [79.63, 80.37] | 0.11% (49/44700) [0.08, 0.14] | [0.05, 0.19] | 0.12% (52/44700) |
| replay_of_enrolled_voice | no (false cancel) | 300 | 300 | 80.00% (240/300) [75.11, 84.13] | 17.67% (53/300) [13.77, 22.38] | [13.33, 22.00] | 20.33% (61/300) |
| household_noise_only | no (false cancel) | 100 | 15000 | 0.00% (0/15000) [0.00, 0.03] | 0.00% (0/15000) [0.00, 0.03] | 0 events (interval degenerate; use the Wilson upper bound) | 0.00% (0/15000) |
| silence | no (false cancel) | 20 | 3000 | 0.00% (0/3000) [0.00, 0.13] | 0.00% (0/3000) [0.00, 0.13] | 0 events (interval degenerate; use the Wilson upper bound) | 0.00% (0/3000) |
| enrolled_says_help | no (false cancel) | 450 | 450 | 0.00% (0/450) [0.00, 0.85] | 0.00% (0/450) [0.00, 0.85] | 0 events (interval degenerate; use the Wilson upper bound) | 83.78% (377/450) |

## Why trials failed (gate reason codes, strict operating point, up to 3000 sampled trials per scenario)

| scenario | sampled | eligible | too_short | intent_unavailable | intent_not_reassurance | low_asr_confidence | speaker_mismatch | no_threshold |
|---|---|---|---|---|---|---|---|---|
| true_cancel_clean | 600 | 254 | 130 | 0 | 71 | 1 | 273 | 0 |
| true_cancel_noise5dB | 300 | 58 | 4 | 0 | 58 | 0 | 233 | 0 |
| other_person_same_words | 3000 | 1 | 649 | 0 | 377 | 9 | 2999 | 0 |
| other_person_noise5dB | 3000 | 1 | 33 | 0 | 588 | 0 | 2999 | 0 |
| visitor_on_behalf | 3000 | 0 | 25 | 0 | 364 | 0 | 2998 | 0 |
| tv_radio_speech | 3000 | 0 | 0 | 0 | 637 | 0 | 3000 | 0 |
| replay_of_enrolled_voice | 300 | 13 | 85 | 0 | 60 | 1 | 284 | 0 |
| household_noise_only | 3000 | 0 | 0 | 0 | 3000 | 0 | 3000 | 0 |
| silence | 3000 | 0 | 3000 | 3000 | 0 | 0 | 0 | 0 |
| enrolled_says_help | 450 | 0 | 27 | 0 | 450 | 0 | 166 | 0 |

A trial can fail for several reasons at once; intent_unavailable = no speech was found (NO_RESPONSE) or the ASR failed.

Consistency audit: the product gate (`CancelGate.evaluate` on the audio) was run on 199 random trials and disagreed with the vectorised analysis in 0 of them.

## Real-speech transfer check (LibriSpeech, read English, speaker layer only)

26 real speakers, 208 probes of 2.5 s; 208 genuine / 5200 impostor trials; EER 0.0004. The thresholds below were chosen on TTS dev speakers and never tuned here.

| operating point | threshold | TAR | FAR |
|---|---|---|---|
| far_0.1pct | 0.679 | 83.65% (174/208) [78.03, 88.06] | 0.00% (0/5200) [0.00, 0.07] |
| far_1pct | 0.601 | 93.75% (195/208) [89.60, 96.31] | 0.00% (0/5200) [0.00, 0.07] |

Real impostor scores: mean 0.100, 99th percentile 0.392; genuine mean 0.744.

## Reading these numbers

* Cancel-eligible is EVIDENCE: no code path lets it change an event (A5 and tests/test_cancel_policy.py check that an EMERGENCY stays EMERGENCY/OPEN whatever it says).
* The voiceprint cannot tell a live reply from a replay of the same voice: replay_of_enrolled_voice measures how much the loudspeaker channel alone changes the score.
* Synthetic voices from one engine are more separable (or less) than real people in a real room; the real-speech check above is the only real-voice evidence and it is read, clean English.
* English only. Mandarin is not evaluated: no multi-speaker Mandarin voice is available within the download budget (one SAPI voice, Huihui).
