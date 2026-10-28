# audio / A4_speaker_verification

- run_id: `20261008-185033_audio_A4_speaker_verification_646d18511c`
- status: **completed**
- data: LibriSpeech dev-clean-2 (real read speech, 26 speakers), speaker-disjoint dev/eval halves
- code: `354f1478138c` · config `646d18511c` · split `None` · seed `42`

## Headline
voxceleb_resnet34_LM/clean: EER 0.007; voxceleb_resnet34_LM/noise_5dB: EER 0.019; voxceleb_resnet34_LM/replay_sim: EER 0.018; cnceleb_resnet34_LM/clean: EER 0.026; cnceleb_resnet34_LM/noise_5dB: EER 0.078; cnceleb_resnet34_LM/replay_sim: EER 0.074

| config | condition | eer_eval | tar_at_dev_far0.1pct_threshold | far_eval_at_that_threshold | genuine_trials_eval | impostor_trials_eval | far_0.1pct_reliable |
|---|---|---|---|---|---|---|---|
| voxceleb_resnet34_LM | clean | 0.007 | 0.996 | 0.018 | 268 | 3216 | False |
| voxceleb_resnet34_LM | noise_5dB | 0.019 | 0.981 | 0.021 | 268 | 3216 | False |
| voxceleb_resnet34_LM | replay_sim | 0.018 | 0.989 | 0.034 | 268 | 3216 | False |
| cnceleb_resnet34_LM | clean | 0.026 | 0.944 | 0.003 | 268 | 3216 | False |
| cnceleb_resnet34_LM | noise_5dB | 0.078 | 0.720 | 0.003 | 268 | 3216 | False |
| cnceleb_resnet34_LM | replay_sim | 0.074 | 0.694 | 0.011 | 268 | 3216 | False |

Threshold for FAR=0.1% chosen on the dev speakers, applied to disjoint eval speakers. With a few thousand impostor trials FAR=0.1% is only coarsely estimable; zero false accepts here would NOT prove household safety (TV, visitors, recordings of the user). The cnceleb model is trained on Chinese speech and is evaluated on English speech here (domain shift). Verification never unlocks automatic cancellation in this MVP.
