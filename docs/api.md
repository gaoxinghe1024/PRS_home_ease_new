# API and Data Formats

Once the server runs, the interactive request schemas are at `/docs`. HTTP and WebSocket share the same access control, and cross-origin write requests are rejected. Errors are returned as `error` or the FastAPI `detail`: 404 not found, 409 state conflict, 422 invalid input, 503 module unavailable.

| Area | Main endpoints |
| --- | --- |
| Status, settings | `GET /api/health`, `GET /api/status`, `GET/PUT /api/settings` |
| Module recovery | `POST /api/modules/{name}/initialize` |
| Home monitoring | `PUT /api/home/monitoring`, `POST /api/home/audio-consent` |
| Video | `POST /api/uploads/video`, `POST /api/video/start`, `POST /api/video/stop`, `GET /api/video/frame` |
| Continuous image | `GET /api/video/stream`, `GET /api/sessions/{sid}/camera/stream` (MJPEG) |
| Evidence clips | `GET /api/video/clip/{event_id}` |
| Accelerometer replay | `POST /api/signal/sample`, `POST /api/signal/analyze` (multipart `file`) |
| Live accelerometer | `WS /ws/sensor` |
| Exercise sessions | `POST /api/sessions`, `GET /api/sessions/{sid}`, `POST /api/sessions/{sid}/control` |
| Background algorithms | `GET/PUT /api/algorithms` |
| Movement plan | `GET/PUT /api/plan`, `POST /api/plan/defer` |
| Older adult page | `GET /api/elder/context`, `POST /api/elder/open` |
| Exercise input | `POST /api/sessions/{sid}/frames`, `POST /api/sessions/{sid}/image` |
| Audio | `POST /api/audio/sample`, `POST /api/audio/analyze`, `WS /ws/audio` |
| Events | `GET /api/events`, `GET /api/events/{id}`, `POST /api/events/{id}/action` |
| Voice inquiry | `POST /api/events/{id}/inquiry`, `/inquiry/processing`, `/inquiry/unavailable` |
| Help button | `POST /api/help` |
| Notifications | `GET /api/notifications`, `POST /api/notifications/{id}/seen`, `POST /api/notifications/test` |
| Reports | `GET /api/reports`, `/api/reports/export.csv`, `/reports/print`, `/reports/events/{id}` |
| Session export | `GET /api/sessions/{sid}/export.json` |

## Video

Start video detection with `{"source":"sample"}` or `{"source":"camera","camera_index":0}`; after an upload use `{"source":"upload","upload_id":"<value returned by the upload endpoint>"}`.

The home page `/home#safety` calls `PUT /api/home/monitoring` with `{"enabled":true,"camera_index":0}`. The setting is persisted; the background scheduler starts the real camera and retries with a rate limit after failures, whether or not any preview page is open. `home_monitoring` in `GET /api/status` contains `enabled`, `camera_index` and the latest start `error`. `enabled:false` pauses monitoring, and `POST /api/video/stop` also pauses automatic recovery. While monitoring is on, a video sample cannot replace live monitoring; pause it first.

`POST /api/home/audio-consent` only sets `audio_consent` to true in the local settings and keeps the profile, contacts and other settings. The browser still has to grant microphone permission separately, and audio is monitored through `/ws/audio`. Opening the home exercise page creates no session and does not interrupt monitoring.

## Exercise sessions

Start a session:

```json
{"exercise":"chair_sit_to_stand","target_reps":10,"source":"camera"}
```

The camera defaults to `capture:"shared"` and `camera_index:0`: the server captures frames itself and can run in parallel with video monitoring. External image / skeleton clients use `capture:"client"` and submit input in the original contract. Other exercises are `squat`, `standing_leg_raise` and `hurdle_step`; other sources are `replay` and `upload`. Control with `{"action":"pause"}`, `resume` or `end`.

System camera and upload sessions set Algorithm 2's `action_scope:"training"`, which computes action support only among the four training exercises; straight-leg raise and hurdle step additionally average the model probabilities of the original and the left-right mirrored input, marked `inference_views:"mirror_average"`. Squat, sit-to-stand and sample replays use `inference_views:"original"`, and replays keep the original default `open`. The returned state, session report and diagnostics contain both fields. The standalone Algorithm 2 API still defaults to `open`. Differences and validation limits: [exercise-integration.md](exercise-integration.md).

Skeleton batches follow the original Algorithm 2 contract:

```json
{"request_id":"batch-0","frames":[{"timestamp_ms":0,"person_id":"no_person","keypoints":[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]}]}
```

This example is one frame without a person and is not counted; a real person needs 17 `[x,y,confidence]` points of the original pose model. Image input is multipart `file` with a strictly increasing `timestamp_ms` and a unique `request_id` in the query. Different content under the same id returns a conflict.

## Audio

Audio analysis takes multipart `file`, `task` and `mode`. The page uses `task:"monitor"`, which returns both original results, `sounds` and `help_result`. `sounds`, `help` and `reply` can still be called separately; `reply` needs an associated `event_id` with an inquiry already started. Samples are analyzed jointly by default, e.g. JSON `{"sample":"reply_help_en"}`, always as replay. Continuous listening handles every sound class and spoken calls for help; there is no class parameter.

A continuous audio connection answers `{"state":"listening"}`. Send `{"pcm":"<base64 little-endian float32 mono PCM>","timestamp_ms":0}` at 16 kHz, at most 32,000 samples per block, with increasing timestamps (the browser sends 8,000 samples per block). Each block returns the original pipeline / listener state with the sound and help results. To stop, send `{"type":"stop"}` and wait for the stopped confirmation.

## Live accelerometer

A successful connection answers `{"state":"connected"}`. Send:

```json
{"samples":[[0.0,0.0,0.0,1.0],[0.02,0.0,0.0,1.0]]}
```

1 to 200 samples per batch, time in seconds and acceleration in g. The original window / alarm results are returned. To finish, send `{"type":"flush"}` and disconnect after `{"state":"stopped"}`; the final events are stored by then.

## Events

Event actions are `acknowledge`, `contact`, `escalate`, `safe` and `resolve`. Caregiver updates can include `revision` so concurrent handling is never overwritten. `resolve` needs an `outcome` (`confirmed_fall`, `false_alarm`, `safe`, `unresolved`) and accepts a `note`. Report parameters are `days=1..90` and `mode=live|replay`.

### Withdrawing a self-triggered alarm

`POST /api/events/{id}/action` accepts `{"action": "withdraw"}` from the older adult's page.

- An event raised by the help button (`channel: "manual"`) or a spoken call for help (`channel: "audio"`) can be withdrawn within `WITHDRAW_WINDOW_S` (30 s) of detection.
- A fall alarm of a sensor (`accel`, `vision`) can be withdrawn only after the voice inquiry has raised it to an emergency (no answer, or a call for help; the event then carries `escalated_at`), within the same 30 s counted from `escalated_at`. A fall candidate that is still PENDING, one that a valid answer cancelled, and one that a caregiver escalated cannot be withdrawn.
- Otherwise the request is refused with a 409 response.
- A withdrawal never closes or downgrades the event. It stays open at its level, the contact escalation timer stops, the caregivers get a notification, and the caregiver still confirms and resolves it.
- `GET /api/events` and `GET /api/events/{id}` add `withdrawn`, `withdrawable` and `withdraw_left_s`; the stored event carries `withdrawal: {"at", "by"}`.
- A later spoken call for help on the same event clears the withdrawal and restarts the escalation clock; a caregiver `escalate` overrides it.

## Background algorithms and the movement plan

`GET /api/algorithms` returns the switches and running state per channel; `PUT` only submits the keys to change (the others keep their values):

```json
{"switches":{"resident":true,"vision":true,"accel":true,"chair_sit_to_stand":true,"squat":true,"standing_leg_raise":true,"hurdle_step":false},
 "channels":{"resident":{"label":"Always on in background","enabled":true,"state":"running","detail":"..."}},
 "blockers":["..."],"resident_ready":true}
```

`resident` is the master switch: only when it is on are video, accelerometer and the four exercise detections started according to their own switches. Exercise rotation and video detection share one camera: a session started from the workbench has priority, and when nobody uses it the camera is handed over directly without closing the device or dropping frames. `state` in `channels` is `running`, `starting`, `idle`, `unavailable` and so on; `blockers` explains why a channel cannot run (for example, the camera is in use). A switched-off exercise cannot start a session: `POST /api/sessions` returns 409.

`GET/PUT /api/plan` reads and writes the caregiver's daily plan; the response includes `progress` (today's completion). `items` has at most 4 entries and each exercise may appear once; duplicates or unknown exercises return 422:

```json
{"enabled":true,"title":"Today's movement plan","note":"...","target_reps":10,
 "items":[{"exercise":"chair_sit_to_stand","target_reps":10,"enabled":true}],
 "skipped_dates":["2026-10-18"]}
```

`reps` in `progress.items[]` comes from today's finished sessions (`status:"ended"`, `mode:"live"`); an item is `done` once `reps` reaches `target_reps`. `disabled` means the detection switch of that exercise is off. `progress.skipped` is true when the older adult chose "not today".

`POST /api/plan/defer` records a skip from the older adult's page; the body is optional:

```json
{"days":1}
```

`days` is 1-7 (default 1) and is written to `skipped_dates` (today plus `days - 1` more days). The plan itself does not change: on skipped days the older adult's page hides the plan card, and it returns the next day; the caregiver still sees the plan and today's completion. `GET /api/elder/context` returns the name, room, plan and progress in one call; `POST /api/elder/open` records when the older adult's page was opened.

## Clearing a report range

`POST /api/records/clear` deletes all safety events, exercise sessions, notifications and video clips in the range and **cannot be undone**. Since every field has a default, an empty body would mean "clear all live records of the last 7 days", so `confirm` is a required literal confirmation word:

```json
{"mode":"live","days":7,"events":true,"sessions":true,"confirm":"DELETE"}
```

Without `confirm`, or with any other value, the request returns 422 and nothing is deleted.

## Interface language

The interface is English only, with no language switch. Pages, dialogs, browser notifications, the spoken voice inquiry, the CSV export and webhook notifications are all written in English in the source. The speech recognition of Algorithm 3 still understands both English and Mandarin replies and calls for help.
