# audio / A3_response_intent_sapi

- run_id: `20261008-161233_audio_A3_response_intent_sapi_9fe9479c15`
- status: **completed**
- data: Windows SAPI TTS replies (synthetic voices; Mandarin proxy voice EMULATED) + LibriSpeech (TV speech proxy) + ESC-50 noise
- code: `354f1478138c` · config `9fe9479c15` · split `None` · seed `42`

## Headline
faster-whisper-tiny + rule macro-F1 0.709; faster-whisper-tiny + supervised macro-F1 0.699; faster-whisper-base + rule macro-F1 0.709; faster-whisper-base + supervised macro-F1 0.694

| config | macro_f1 | accuracy | n | asr_latency_s_median |
|---|---|---|---|---|
| faster-whisper-tiny + rule | 0.709 | 0.643 | 42 | 0.307 |
| faster-whisper-tiny + supervised | 0.699 | 0.548 | 42 | 0.307 |
| faster-whisper-base + rule | 0.709 | 0.643 | 42 | 0.621 |
| faster-whisper-base + supervised | 0.694 | 0.548 | 42 | 0.621 |

## Per interference bucket

| asr | bucket | n | rule_correct | supervised_correct |
|---|---|---|---|---|
| faster-whisper-tiny | ambient_noise_5dB | 16 | 10 | 10 |
| faster-whisper-tiny | no_response | 3 | 3 | 3 |
| faster-whisper-tiny | proxy_answer | 2 | 0 | 0 |
| faster-whisper-tiny | self_call_for_help | 6 | 2 | 3 |
| faster-whisper-tiny | self_other | 4 | 4 | 0 |
| faster-whisper-tiny | self_response | 6 | 3 | 5 |
| faster-whisper-tiny | tv_says_help | 1 | 1 | 1 |
| faster-whisper-tiny | tv_speech | 4 | 4 | 1 |
| faster-whisper-base | ambient_noise_5dB | 16 | 9 | 9 |
| faster-whisper-base | no_response | 3 | 3 | 3 |
| faster-whisper-base | proxy_answer | 2 | 0 | 0 |
| faster-whisper-base | self_call_for_help | 6 | 2 | 3 |
| faster-whisper-base | self_other | 4 | 4 | 0 |
| faster-whisper-base | self_response | 6 | 4 | 6 |
| faster-whisper-base | tv_says_help | 1 | 1 | 1 |
| faster-whisper-base | tv_speech | 4 | 4 | 1 |

Key safety observation: proxy answers and TV speech produce REASSURANCE/HELP intents that did not come from the monitored person; this is why a reply never cancels an MVP emergency (A5). Voices are synthetic TTS; real older-adult speech, dialects and microphones are not covered.
