# audio / A1_vad

- run_id: `20261008-145829_audio_A1_vad_e13bcbb547`
- status: **completed**
- data: LibriSpeech dev-clean-2 speech + ESC-50 non-speech (real clips, constructed mixtures)
- code: `354f1478138c` · config `e13bcbb547` · split `None` · seed `42`

## Headline
energy F1 0.907; webrtc F1 0.899; silero F1 0.895

| config | condition | precision | recall | f1 | frames |
|---|---|---|---|---|---|
| energy | clean | 0.992 | 0.988 | 0.990 | 36783 |
| energy | household_noise_10dB | 0.827 | 0.960 | 0.889 | 40814 |
| energy | impact_sounds_in_gaps | 0.812 | 0.887 | 0.848 | 36266 |
| silero | clean | 1.000 | 0.818 | 0.900 | 36783 |
| silero | household_noise_10dB | 0.997 | 0.808 | 0.893 | 40814 |
| silero | impact_sounds_in_gaps | 0.988 | 0.815 | 0.893 | 36266 |
| webrtc | clean | 0.981 | 0.946 | 0.963 | 36783 |
| webrtc | household_noise_10dB | 0.834 | 0.931 | 0.880 | 40814 |
| webrtc | impact_sounds_in_gaps | 0.812 | 0.916 | 0.861 | 36266 |
| energy | ALL | 0.871 | 0.946 | 0.907 | 113863 |
| webrtc | ALL | 0.869 | 0.931 | 0.899 | 113863 |
| silero | ALL | 0.995 | 0.813 | 0.895 | 113863 |

Frame truth = utterance span (LibriSpeech utterances contain short internal pauses, so recall is slightly understated for all VADs). VAD only gates the speech branch; impact sounds are classified by the separate sound branch.
