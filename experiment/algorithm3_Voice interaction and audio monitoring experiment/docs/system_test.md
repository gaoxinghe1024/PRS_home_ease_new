# System test of Algorithm 3 on the integrated system

Tested version: `main` of `gaoxinghe1024/PRS_home_ease_new` at commit `d303a70` (pull requests 1 and 2 included), server instance on the local machine, weights from the repository. Input and returned values are as the server gave them.

How to read the tables

- The spoken input is Windows text-to-speech (Mandarin female, English female, English male), mixed with background noise or turned down for some conditions. It is not a human voice and not an older person's voice.
- Decisions, cues and confidences are what the server interface returned. The *heard* text is the recogniser output of the same code run in a local process with the thread setting of the server; the server interface does not return the transcript.
- Mandarin phrases and Mandarin transcripts are written in pinyin (Chinese characters are not used in this repository); the English meaning is given in brackets for what was said. Pinyin hides which character the recogniser chose, so a homophone error (for example a different character with the sound "jiu" in "jiu ming") is not visible in the transcript; the rules accept such homophones for a bare call.
- Noise is a household recording (washing machine plus everyday sounds) mixed in at the stated signal-to-noise ratio. *Very quiet* means the peak level was set to -34 dBFS.
- A cross marks a result that differs from the expected one.

## 1 One spoken sentence: is it taken as a call for help? (uploaded recording, `/api/audio/analyze`, task `monitor`)

Expected: a call for help raises an alarm, anything else does not. Each cell: what the system returned (decision, matched cue, recognition confidence) and the heard text.

### 1.1 Calls for help

| Said | Voice | clean | noise 20 dB | noise 10 dB | very quiet (-34 dBFS) |
|---|---|---|---|---|---|
| jiu ming (Mandarin: help) | Mandarin, female | **alarm**<br>cue zh_help, confidence -0.82<br>heard: "jiu ming" | **alarm**<br>cue zh_help, confidence -0.91<br>heard: "jiu ming" | **alarm**<br>cue zh_help, confidence -1.05<br>heard: "jiu ming" | **alarm**<br>cue zh_help, confidence -0.76<br>heard: "jiu ming" |
| jiu ming a (Mandarin: help!) | Mandarin, female | **alarm**<br>cue zh_help, confidence -0.64<br>heard: "jiu ming a" | **alarm**<br>cue zh_help, confidence -0.66<br>heard: "jiu ming a" | **alarm**<br>cue zh_help, confidence -0.74<br>heard: "jiu ming a" | **alarm**<br>cue zh_help, confidence -0.56<br>heard: "jiu ming a" |
| jiu ming jiu ming (Mandarin: help, help) | Mandarin, female | **alarm**<br>cue zh_help, confidence -0.52<br>heard: "jiu ming jiu ming" | **alarm**<br>cue zh_help, confidence -0.51<br>heard: "jiu ming jiu ming" | **alarm**<br>cue zh_help, confidence -0.69<br>heard: "jiu ming jiu ming" | **alarm**<br>cue zh_help, confidence -0.45<br>heard: "jiu ming jiu ming" |
| jiu ming!jiu ming! (Mandarin: help! help!) | Mandarin, female | **alarm**<br>cue zh_help, confidence -0.59<br>heard: "jiu ming jiu ming" | **alarm**<br>cue zh_help, confidence -0.61<br>heard: "jiu ming jiu ming" | **alarm**<br>cue zh_help, confidence -0.62<br>heard: "jiu ming jiu ming" | **alarm**<br>cue zh_help, confidence -0.59<br>heard: "jiu ming jiu ming" |
| bang bang wo (Mandarin: help me) | Mandarin, female | **alarm**<br>cue zh_help, confidence -0.46<br>heard: "bang bang wo" | **alarm**<br>cue zh_help, confidence -0.57<br>heard: "dang dang wo" | **alarm**<br>cue zh_help, confidence -0.67<br>heard: "bang bang wo" | **alarm**<br>cue zh_help, confidence -0.49<br>heard: "bang bang wo" |
| lai ren a (Mandarin: somebody come) | Mandarin, female | **alarm**<br>cue zh_help, confidence -0.80<br>heard: "lai ren ne" | **alarm**<br>cue zh_help, confidence -0.72<br>heard: "lai ren ne" | **alarm**<br>cue zh_help, confidence -0.86<br>heard: "lai ren ne" | **alarm**<br>cue zh_help, confidence -0.66<br>heard: "lai ren le" |
| wo shuai dao le,qi bu lai le (Mandarin: I fell and can't get up) | Mandarin, female | **alarm**<br>cue zh_cant, confidence -0.45<br>heard: "wo shuai dao le,qi bu lai le" | **alarm**<br>cue zh_cant, confidence -0.44<br>heard: "wo shuai dao le,qi bu lai le" | **alarm**<br>cue zh_cant, confidence -0.51<br>heard: "wo shuai dao le,qi bu lai le" | **alarm**<br>cue zh_cant, confidence -0.58<br>heard: "wo shuai dao le qi bu lai le" |
| wo dong bu liao le,bang bang wo (Mandarin: I can't move, help me) | Mandarin, female | **alarm**<br>cue zh_cant,zh_help, confidence -0.49<br>heard: "wo dong bu liao le,bang bang wo." | **alarm**<br>cue zh_cant,zh_help, confidence -0.66<br>heard: "wo dong bu liao le bang bang wo" | **alarm**<br>cue zh_cant,zh_help, confidence -0.61<br>heard: "huo dong bu liao le,bang bang wo" | **alarm**<br>cue zh_cant, confidence -0.55<br>heard: "wo dong bu liao le fang fang wo" |
| kuai jiao jiu hu che (Mandarin: call an ambulance quickly) | Mandarin, female | **alarm**<br>cue zh_ambulance, confidence -0.44<br>heard: "kuai jiao jiu hu che" | **alarm**<br>cue zh_ambulance, confidence -0.55<br>heard: "kuai jiao jiu hu che" | **alarm**<br>cue zh_ambulance, confidence -0.50<br>heard: "kuai jiao jiu hu che" | **alarm**<br>cue zh_ambulance, confidence -0.47<br>heard: "kuai jiao jiu hu che" |
| wo bu shu fu,bang wo jiao yi sheng (Mandarin: I feel unwell, call a do... | Mandarin, female | **alarm**<br>cue zh_ambulance,zh_help, confidence -0.56<br>heard: "wo bu shu fu,bang wo jiao yi sheng" | **alarm**<br>cue zh_ambulance,zh_help, confidence -0.51<br>heard: "wo bu shu fu,bang wo jiao yi sheng" | **alarm**<br>cue zh_ambulance,zh_help, confidence -0.57<br>heard: "wo bu shu fu bang wo jiao yi sheng" | **alarm**<br>cue zh_help, confidence -0.46<br>heard: "wo bu shu fu bang wo jiao yi sheng" |
| Help | English, female | **alarm**<br>cue help, confidence -0.69<br>heard: "Help." | **alarm**<br>cue help, confidence -0.95<br>heard: "help." | **alarm**<br>cue help, confidence -1.08<br>heard: "help." | **alarm**<br>cue help, confidence -0.87<br>heard: "Help." |
| Help me | English, female | **alarm**<br>cue help, confidence -0.55<br>heard: "Help me." | **alarm**<br>cue help, confidence -0.83<br>heard: "help me." | **alarm**<br>cue help, confidence -0.90<br>heard: "help me." | **alarm**<br>cue help, confidence -0.58<br>heard: "Help me." |
| Help! Help! | English, female | **alarm; no alarm: inside the cooldown**<br>cue help, confidence -0.69<br>heard: "Help." | **alarm; no alarm: inside the cooldown**<br>cue help, confidence -0.98<br>heard: "help." | **alarm; no alarm: inside the cooldown**<br>cue help, confidence -1.11<br>heard: "Help." | **alarm; no alarm: inside the cooldown**<br>cue help, confidence -0.87<br>heard: "Help." |
| I fell, please help | English, female | **alarm**<br>cue help, confidence -0.48<br>heard: "I fell. Please help." | **alarm**<br>cue help, confidence -0.50<br>heard: "I fell. Please help." | **alarm**<br>cue help, confidence -0.57<br>heard: "I fell. Please help." | **alarm**<br>cue help, confidence -0.47<br>heard: "I fell. Please help." |
| I can't get up | English, female | **alarm**<br>cue cant_do, confidence -0.30<br>heard: "I can't get up." | **alarm**<br>cue cant_do, confidence -0.34<br>heard: "I can't get up." | **alarm**<br>cue cant_do, confidence -0.36<br>heard: "I can't get up." | **alarm**<br>cue cant_do, confidence -0.30<br>heard: "I can't get up." |
| Somebody call an ambulance | English, female | **alarm**<br>cue ambulance, confidence -0.42<br>heard: "Somebody call an ambulance." | **alarm**<br>cue ambulance, confidence -0.50<br>heard: "Somebody call an ambulance." | **alarm**<br>cue ambulance, confidence -0.54<br>heard: "Somebody call an ambulance." | **alarm**<br>cue ambulance, confidence -0.52<br>heard: "Somebody call an ambulance." |
| Help | English, male | **alarm**<br>cue help, confidence -0.74<br>heard: "Help." | **alarm**<br>cue help, confidence -1.00<br>heard: "help." | **alarm**<br>cue help, confidence -1.16<br>heard: "Help." | (x) **no alarm: no speech segment found**<br>heard: (nothing) |
| Help me please | English, male | **alarm**<br>cue help, confidence -0.51<br>heard: "Help me please." | **alarm**<br>cue help, confidence -0.66<br>heard: "Help me please." | **alarm**<br>cue help, confidence -0.72<br>heard: "Help me please." | **alarm**<br>cue help, confidence -0.45<br>heard: "Help me please." |
| I fell and I can't get up | English, male | **alarm**<br>cue cant_do, confidence -0.40<br>heard: "I fell and I can't get up." | **alarm**<br>cue cant_do, confidence -0.42<br>heard: "I feel and I can't get up." | **alarm**<br>cue cant_do, confidence -0.39<br>heard: "I feel and I can't get up." | **alarm**<br>cue cant_do, confidence -0.38<br>heard: "I feel and I can't get up." |

### 1.2 Sentences that are not calls for help

| Said | Voice | clean | noise 20 dB | noise 10 dB | very quiet (-34 dBFS) |
|---|---|---|---|---|---|
| wo mei shi (Mandarin: I'm fine) | Mandarin, female | **no alarm: not a call for help**<br>heard: "wo mei shi" | **no alarm: not a call for help**<br>heard: "wo mei shi" | **no alarm: not a call for help**<br>heard: "wo mei shi" | **no alarm: not a call for help**<br>heard: "wo mei shi" |
| wo mei shi,bu yong dan xin (Mandarin: I'm fine, don't worry) | Mandarin, female | **no alarm: not a call for help**<br>heard: "wo mei shi bu yong dan xin" | **no alarm: not a call for help**<br>heard: "wo mei shi bu yong dan xin" | **no alarm: not a call for help**<br>heard: "wo mei shi bu yong dan xin" | **no alarm: not a call for help**<br>heard: "wo mei shi bu yong dan xin" |
| jin tian tian qi bu cuo (Mandarin: nice weather today) | Mandarin, female | **no alarm: not a call for help**<br>heard: "jin tian tian qi bu cuo" | **no alarm: not a call for help**<br>heard: "jin tian tian qi bu cuo" | **no alarm: not a call for help**<br>heard: "jin tian tian qi bu cuo" | **no alarm: not a call for help**<br>heard: "jin tian tian qi bu cuo" |
| xie xie ni de bang zhu (Mandarin: thanks for your help) | Mandarin, female | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "xie xie ni de bang zhu" | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "xie xie ni de bang zhu" | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "xie xie ni de bang zhu" | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "xie xie ni de bang zhu" |
| ni xu yao bang zhu ma (Mandarin: do you need help?) | Mandarin, female | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "ni xu yao bang zhu ma?" | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "ni xu yao bang zhu ma?" | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "ni xu yao bang zhu mang" | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "ni xu yao bang zhu ma" |
| wo lai bang ni (Mandarin: I'll help you) | Mandarin, female | **no alarm: not a call for help**<br>heard: "wo lai bang ni" | **no alarm: not a call for help**<br>heard: "wo lai bang ni" | **no alarm: not a call for help**<br>heard: "wo lai bang ni" | **no alarm: not a call for help**<br>heard: "wo lai bang ni" |
| jiu hu che lai le (Mandarin: the ambulance is coming) | Mandarin, female | **no alarm: not a call for help**<br>heard: "jiu bu che lai le" | **no alarm: not a call for help**<br>heard: "jiu bu che lai le" | **no alarm: not a call for help**<br>heard: "jiu bu che lai le" | **no alarm: not a call for help**<br>heard: "jiu bu che lai le" |
| qing wen xian zai ji dian le (Mandarin: what time is it?) | Mandarin, female | **no alarm: not a call for help**<br>heard: "qing wen xian zai ji dian le" | **no alarm: not a call for help**<br>heard: "qing wen xian zai ji dian le" | **no alarm: not a call for help**<br>heard: "qing wen xian zai ji dian le" | **no alarm: not a call for help**<br>heard: "qing wen xian zai ji dian le" |
| wo qu dao bei shui (Mandarin: I'm going to pour a glass of water) | Mandarin, female | **no alarm: not a call for help**<br>heard: "wo qu dao bing shui" | **no alarm: not a call for help**<br>heard: "wo qu dao fei shui" | **no alarm: not a call for help**<br>heard: "wo qu dao fei shui" | **no alarm: not a call for help**<br>heard: "wo qu dao bei shui" |
| I'm fine | English, female | **no alarm: not a call for help**<br>heard: "I'm fine." | **no alarm: not a call for help**<br>heard: "I'm fine." | **no alarm: not a call for help**<br>heard: "I'm fine." | **no alarm: not a call for help**<br>heard: "I'm fine." |
| Thanks for your help | English, female | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "Thanks for your help." | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "Thanks for your help." | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "Thanks for your help." | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "Thanks for your help." |
| Do you need any help | English, female | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "Do you need any help?" | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "Do you need any help?" | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "do you need any help?" | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "Do you need any help?" |
| The ambulance is coming | English, female | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "The ambulance is coming." | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "The ambulance is coming." | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "The ambulance is coming." | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "The ambulance is coming." |
| What time is it | English, female | **no alarm: not a call for help**<br>heard: "What time is it?" | **no alarm: not a call for help**<br>heard: "What time is it?" | **no alarm: not a call for help**<br>heard: "What time is it?" | **no alarm: not a call for help**<br>heard: "What time is it?" |
| I'm fine, thank you | English, male | **no alarm: not a call for help**<br>heard: "I'm fine. Thank you." | **no alarm: not a call for help**<br>heard: "I'm fine. Thank you." | **no alarm: not a call for help**<br>heard: "I'm fine. Thank you." | **no alarm: not a call for help**<br>heard: "I'm fine. Thank you." |
| Help is on the way | English, male | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "Help is on the way." | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "help is on the way." | (x) **alarm**<br>cue help, confidence -0.76<br>heard: "help us on the way." | **no alarm: a 'help'-type phrase that is not a call, filtered**<br>heard: "Help is on the way." |
| And in tonight's news the city council has asked residents who need he... | English, female | **no alarm: too many words**<br>heard: "And in tonight's news, the city council ..." | **no alarm: too many words**<br>heard: "And in tonight's news, the City Council ..." | **no alarm: too many words**<br>heard: "And in tonight's news, the City Council ..." | **no alarm: too many words**<br>heard: "And in tonight's news, the City Council ..." |

### 1.3 Summary

| Condition | Calls for help that raised an alarm | Other sentences that raised a false alarm |
|---|---|---|
| clean | 19 / 19 | 0 / 17 |
| noise 20 dB | 19 / 19 | 0 / 17 |
| noise 10 dB | 19 / 19 | 1 / 17 |
| very quiet (-34 dBFS) | 18 / 19 | 0 / 17 |

Missed calls for help (1):

- "Help" (English, male, very quiet (-34 dBFS)): no speech segment found; heard (nothing)

False alarms (1):

- "Help is on the way" (English, male, noise 10 dB): alarm, cue ['help']; heard "help us on the way."

## 2 Live microphone stream (`/ws/audio`, one 0.5 s block at real speed, washing-machine background)

Speech starts at second 4. An *event* is the call-for-help event the system creates by itself. Every scenario is a new listening session.

| Said | Voice | Events created | Level / channel / tag | Cue / confidence | Processing time per block (max / mean) |
|---|---|---|---|---|---|
| jiu ming (Mandarin: help) | Mandarin, female | 1 | EMERGENCY / audio / single_channel | zh_help / -1.00 | 0.11 s / 0.049 s |
| jiu ming a (Mandarin: help!) | Mandarin, female | 1 | EMERGENCY / audio / single_channel | zh_help / -0.71 | 0.11 s / 0.049 s |
| jiu ming jiu ming (Mandarin: help, help) | Mandarin, female | 1 | EMERGENCY / audio / single_channel | zh_help / -0.52 | 0.11 s / 0.057 s |
| bang bang wo (Mandarin: help me) | Mandarin, female | 1 | EMERGENCY / audio / single_channel | zh_help / -0.54 | 0.12 s / 0.053 s |
| lai ren a (Mandarin: somebody come) | Mandarin, female | 1 | EMERGENCY / audio / single_channel | zh_help / -0.78 | 0.11 s / 0.051 s |
| wo shuai dao le,qi bu lai le (Mandarin: I fell and can't get up) | Mandarin, female | 1 | EMERGENCY / audio / single_channel | zh_cant / -0.45 | 0.13 s / 0.058 s |
| Help | English, female | 1 | EMERGENCY / audio / single_channel | help / -0.95 | 0.11 s / 0.048 s |
| Help me | English, female | 1 | EMERGENCY / audio / single_channel | help / -0.65 | 0.09 s / 0.046 s |
| I fell, please help | English, female | 1 | EMERGENCY / audio / single_channel | help / -0.51 | 0.11 s / 0.057 s |
| Help | English, male | (x) none | - | - | 0.14 s / 0.054 s |
| wo mei shi (Mandarin: I'm fine) | Mandarin, female | none | - | - | 0.10 s / 0.051 s |
| jin tian tian qi bu cuo (Mandarin: nice weather today) | Mandarin, female | none | - | - | 0.10 s / 0.052 s |
| xie xie ni de bang zhu (Mandarin: thanks for your help) | Mandarin, female | none | - | - | 0.08 s / 0.048 s |
| I'm fine | English, female | none | - | - | 0.09 s / 0.047 s |
| Thanks for your help | English, female | none | - | - | 0.10 s / 0.047 s |
| The ambulance is coming | English, female | none | - | - | 0.10 s / 0.048 s |
| And in tonight's news the city council has asked residents who need he... | English, female | none | - | - | 0.10 s / 0.050 s |
| one listening session: Mandarin 'jiu ming a', then 'Help me' within 8 ... | - | 1 | EMERGENCY / audio / single_channel | zh_help / -0.70 | 0.12 s / 0.050 s |

## 3 Voice inquiry after a fall: what the older adult answers and what happens to the event

Each row: Algorithm 1 detects a fall from the real accelerometer recording (URFD fall-01), which creates a new **live fall event** (level PENDING, waiting for verification). The older adult's page then starts the inquiry, uploads the answer and reads the result.

| Answer | Condition | Heard | Reply class | Trusted 'I'm fine' | Confidence | Inquiry result | Event level (before to after) | Tag | Text shown on the older adult's page |
|---|---|---|---|---|---|---|---|---|---|
| wo mei shi,bu yong dan xin (Mandarin: I'm fine, don't worry) | clean | "wo mei shi bu yong dan xin" | REASSURANCE | yes | -0.56 | escalation cancelled | PENDING to PENDING | single_channel | Your reply was recorded. The alarm stays active until a caregiver verifies and closes it. |
| mei shi (Mandarin: it's nothing) | clean | "mei shi" | REASSURANCE | no | -1.04 | no cancel, escalated to EMERGENCY | PENDING to EMERGENCY | dual_channel | The call for help was escalated; waiting for a caregiver. |
| I'm fine, don't worry | clean | "I'm fine. Don't worry." | REASSURANCE | yes | -0.39 | escalation cancelled | PENDING to PENDING | single_channel | Your reply was recorded. The alarm stays active until a caregiver verifies and closes it. |
| Don't worry, she is fine | clean | "Don't worry, she is fine." | REASSURANCE | yes | -0.45 | escalation cancelled | PENDING to PENDING | single_channel | Your reply was recorded. The alarm stays active until a caregiver verifies and closes it. |
| jiu ming (Mandarin: help) | clean | "jiu ming" | HELP | no | -1.08 | no cancel, escalated to EMERGENCY | PENDING to EMERGENCY | dual_channel | The call for help was escalated; waiting for a caregiver. |
| jiu ming a (Mandarin: help!) | clean | "jiu ming a" | HELP | no | -0.89 | no cancel, escalated to EMERGENCY | PENDING to EMERGENCY | dual_channel | The call for help was escalated; waiting for a caregiver. |
| bang bang wo (Mandarin: help me) | clean | "bang bang wo" | HELP | no | -0.76 | no cancel, escalated to EMERGENCY | PENDING to EMERGENCY | dual_channel | The call for help was escalated; waiting for a caregiver. |
| Help me, I can't get up | clean | "Help me, I can't get up." | HELP | no | -0.44 | no cancel, escalated to EMERGENCY | PENDING to EMERGENCY | dual_channel | The call for help was escalated; waiting for a caregiver. |
| wo mei shi,dan shi wo qi bu lai le (Mandarin: I'm fine, but ... | clean | "wo mei shi,dan shi wo qi bu lai le" | HELP | no | -0.56 | no cancel, escalated to EMERGENCY | PENDING to EMERGENCY | dual_channel | The call for help was escalated; waiting for a caregiver. |
| I'm fine, but I can't get up | clean | "I'm fine, but I can't get up." | HELP | no | -0.39 | no cancel, escalated to EMERGENCY | PENDING to EMERGENCY | dual_channel | The call for help was escalated; waiting for a caregiver. |
| wo qu dao bei shui (Mandarin: I'm going to pour a glass of w... | clean | "wo qu dao bei shui" | OTHER | no | -0.69 | no cancel, escalated to EMERGENCY | PENDING to EMERGENCY | dual_channel | The call for help was escalated; waiting for a caregiver. |
| Okay, I will make some tea later | clean | "Okay, I will make some tea later." | OTHER | no | -0.44 | no cancel, escalated to EMERGENCY | PENDING to EMERGENCY | dual_channel | The call for help was escalated; waiting for a caregiver. |
| Thank you for your help | clean | "Thank you for your help." | HELP | no | -0.40 | no cancel, escalated to EMERGENCY | PENDING to EMERGENCY | dual_channel | The call for help was escalated; waiting for a caregiver. |
| And in tonight's news the city council has asked residents w... | clean | "An Internet's news the city council has ..." | HELP | no | -0.31 | no cancel, escalated to EMERGENCY | PENDING to EMERGENCY | dual_channel | The call for help was escalated; waiting for a caregiver. |
| wo mei shi (Mandarin: I'm fine) | very quiet (-34 dBFS) | "wo mei shi" | REASSURANCE | yes | -0.86 | escalation cancelled | PENDING to PENDING | single_channel | Your reply was recorded. The alarm stays active until a caregiver verifies and closes it. |
| wo mei shi (Mandarin: I'm fine) | noise 10 dB | "wo mei shi" | REASSURANCE | yes | -0.58 | escalation cancelled | PENDING to PENDING | single_channel | Your reply was recorded. The alarm stays active until a caregiver verifies and closes it. |
| jiu ming (Mandarin: help) | noise 10 dB | "jiu ming" | HELP | no | -0.84 | no cancel, escalated to EMERGENCY | PENDING to EMERGENCY | dual_channel | The call for help was escalated; waiting for a caregiver. |
| (no answer, 3 s of silence) | clean | (nothing) | NO_RESPONSE | no | - | no cancel, escalated to EMERGENCY | PENDING to EMERGENCY | dual_channel | The call for help was escalated; waiting for a caregiver. |

REASSURANCE = the reply says the person is fine, HELP = call for help, OTHER = unrelated, NO_RESPONSE = no answer. Tag `single_channel` = one detector (the sensor), `dual_channel` = sensor plus voice.
A cancel only stops the escalation: the event stays open at PENDING until a caregiver closes it.

## 4 Cancel and withdraw: action, interface response, final state of the event

| Case | Actions and interface responses | Final state of the event |
|---|---|---|
| C1 Older adult presses the help button, withdraws, withdraws again | tap 'Withdraw call for help': 200<br>tap it again: 200 | level EMERGENCY, open, withdrawn, contact index 0, inquiry: not_started |
| C2 Help-button event is 31 s old, withdraw | withdraw: 409 (the withdraw window (30 s) is over, contact the caregiver) | level EMERGENCY, open, not withdrawn, contact index 0, inquiry: not_started |
| C3 Fall just detected (pending, no inquiry yet), withdraw | withdraw: 409 (only the help button, a spoken call for help or an escalated fall alarm can be withdrawn by the older adult) | level PENDING, open, not withdrawn, contact index 0, inquiry: not_started |
| C4 Fall, no answer to the inquiry so it escalates to emergency, withdraw at once | state after escalation: EMERGENCY (withdrawable=True, 30 s left)<br>withdraw: 200 | level EMERGENCY, open, withdrawn, contact index 0, inquiry: completed |
| C5 Fall, escalated, withdraw after 31 s | withdraw: 409 (the withdraw window (30 s) is over, contact the caregiver) | level EMERGENCY, open, not withdrawn, contact index 0, inquiry: completed |
| C6 Fall, 'I'm fine' cancels the escalation (still pending), withdraw | withdraw: 409 (only the help button, a spoken call for help or an escalated fall alarm can be withdrawn by the older adult) | level PENDING, open, not withdrawn, contact index 0, inquiry: completed |
| C7 Fall, caregiver escalates by hand, older adult tries to withdraw | withdraw: 409 (only the help button, a spoken call for help or an escalated fall alarm can be withdrawn by the older adult) | level EMERGENCY, open, not withdrawn, contact index 1, inquiry: not_started |
| C8 Fall, escalated to emergency, older adult taps the 'I'm OK' button | 'I'm OK' button: 200 | level EMERGENCY, open, not withdrawn, contact index 0, inquiry: button_safe |
| C9 Fall just detected, older adult taps the 'I'm OK' button | 'I'm OK' button: 200 | level PENDING, open, not withdrawn, contact index 0, inquiry: button_safe |
| C10 Help button, withdraw, then a spoken 'help' afterwards | withdraw: 200<br>inquiry reply 'help': - | level EMERGENCY, open, not withdrawn, contact index 0, inquiry: completed |
| C11 Contact escalation timer reached: withdrawn vs not withdrawn event (2 contacts each) | withdrawn: contact index: 0<br>not withdrawn: contact index: 1 | level EMERGENCY, open, withdrawn, contact index 0, inquiry: not_started |
| C12 Caregiver closes the event, then an inquiry is started for it | close: 200<br>start inquiry: 409 (the event is already closed) | level PENDING, closed, not withdrawn, contact index 0, inquiry: not_started |
| C13 An answer is uploaded without an inquiry having been started | upload an answer: 409 (no inquiry is waiting for a reply) | level PENDING, open, not withdrawn, contact index 0, inquiry: not_started |

Interface response 200 = accepted, 409 = refused.

## 5 The six sound samples that come with the system (`/api/audio/sample`, task `monitor`)

| Sample | Sound-event labels (class, start to end, confidence) | Speech segments | Alarm |
|---|---|---|---|
| Glass breaking | UNKNOWN, 8.1 to 8.7 s, 0.70<br>DAILY, 8.1 to 8.7 s, 0.90 | no speech | no |
| TV and knocking | DAILY, 3.3 to 23.0 s, 0.91<br>UNKNOWN, 18.1 to 22.3 s, 0.79 | at 3.3 s: no alarm: not a call for help | no |
| Quiet home | DAILY, 9.4 to 13.6 s, 0.98 | no speech | no |
| Call for help after a fall | DAILY, 4.1 to 11.5 s, 0.93 | at 9.3 s: alarm ['cant_do', 'help'] | yes |
| "I'm fine" after a fall | DAILY, 0.8 to 11.0 s, 1.00 | at 9.4 s: no alarm: not a call for help | no |
| A visitor answers for the older adult | DAILY, 0.8 to 11.0 s, 0.98 | at 9.3 s: no alarm: not a call for help | no |

- **Glass breaking**: 20 s of a quiet home with the sound of breaking glass at 8 to 13 s (labelled ABNORMAL). The model's recall for ABNORMAL is low (about 0.41), so it is mostly labelled DAILY or UNKNOWN; this limit is stated in the evaluation.
- **TV and knocking**: 20 s: someone speaks on a TV at 3 to 6 s (not the older adult's answer), a knock at 18 to 23 s.
- **Quiet home**: 20 s: washing-machine background and one everyday sound (labelled DAILY). No HELP and no alarm expected.
- **Call for help after a fall**: 24 s: an impact, a scream, and from 9 to 12 s the person says "Help me, I fell and I cannot get up."
- **"I'm fine" after a fall**: 24 s: an impact, then the person says "I'm fine, don't worry." The reply is read as REASSURANCE; no call for help.
- **A visitor answers for the older adult**: 24 s: an impact, then another voice (a visitor) says "Don't worry, she is fine." The words alone cannot tell who is speaking.

Sound-event labels are shown for information. In this version they do not raise an alarm by themselves.

## 6 After this test

This test was run on commit `d303a70`. The reply rules on `main` have been changed since: they now contain an extra pattern for bare reassurance such as "mei shi" (in the table of section 3 that reply was heard correctly but not trusted, confidence -1.04), so the copy of the rules in this package differs from `main`. Two limits shown above are still open: "Thank you for your help" and a news sentence were read as HELP in the inquiry, which escalates the event and so errs on the safe side. Sound-event labels remain hints and do not raise an alarm.
