# audio / A5_alarm_behaviour

- run_id: `20261008-185637_audio_A5_alarm_behaviour_1e5792df46`
- status: **completed**
- data: headless system (virtual clock) + scripted inquiry outcomes
- code: `354f1478138c` · config `1e5792df46` · split `None` · seed `42`

## Headline
policy PASSED: voice replies never closed or downgraded an emergency

| reply | inquiry_status | intent | cancel_eligible | level_after | status_after | notifications_before_inquiry | closed_by_voice |
|---|---|---|---|---|---|---|---|
| reassurance | RESPONDED | REASSURANCE | False | EMERGENCY | OPEN | 1 | False |
| help | RESPONDED | HELP | False | EMERGENCY | OPEN | 1 | False |
| no_response | NO_RESPONSE | n/a | False | EMERGENCY | OPEN | 1 | False |
| audio_failure | audio_unavailable | n/a | False | EMERGENCY | OPEN | 1 | False |
| reassurance_voiceprint_match | RESPONDED | REASSURANCE | True | EMERGENCY | OPEN | 1 | False |

Policy check: an emergency stays EMERGENCY/OPEN after a spoken reassurance, after silence and after an audio failure, and ALSO when the voiceprint gate reports cancel_eligible=True (enrolled voice, reassurance, confident ASR): that flag is evidence only and is never acted on. Caregiver notifications were already dispatched before the inquiry ran. (ASR and speaker model are scripted here to isolate the policy; A3 measures real ASR/intent, A6 the rates of the evidence.)
