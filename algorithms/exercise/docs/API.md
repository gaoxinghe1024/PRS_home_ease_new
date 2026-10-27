# HomeEase Algorithm 2: Interface and Integration Guide

## 1. Integration summary

The recommended setup converts camera images into COCO 17 keypoints locally and then calls this service's skeleton-stream interface, so Algorithm 1 and Algorithm 2 can share the same pose CNN. An image entry point (`homeease.pose.PoseEstimator`) and a local video program (`scripts/run_video.py`) are included; raw video never has to be sent to a server.

HTTP API v1, default address `http://127.0.0.1:8002`. The interface is exported from the current code to `docs/openapi.json`. The real weights are loaded at startup; missing weights make startup fail instead of returning fake predictions. `examples/*_replay.json` can be used for an end-to-end acceptance test of all four exercises.

Deployment extension: a session can be created with `action_scope:"open"|"training"`. The default `open` keeps the original report mode. `training` is only for the case where the user has already chosen one of the four training exercises; the original 0.15 class gate is then applied to the relative support among the four exercises. The counter and the input-interruption rules are unchanged. The SDK's `ExerciseSession(..., action_scope="training")` is equivalent, and state and exports keep the field. Action probabilities are still the model's five-class distribution, so `recognized_exercise` can still be `other`; it is not a renaming of the user's chosen target.

Only `training` sessions of `standing_leg_raise` and `hurdle_step` use `inference_views:"mirror_average"`: inference runs on the original input and on the left-right mirrored input used by the original training script, the 17-class joint probabilities are averaged, and then the original decoder and counter run. The mirror flips x and swaps all six left/right joint pairs without changing y or confidence; both histories reset together at every interruption of the original input. This mode costs extra inference, and both phase and action probabilities are averages of the two views, so they must not be read as single-pass probabilities. Squat, sit-to-stand and all `open` sessions use `inference_views:"original"`, the original input path. Weights, feature functions, configured thresholds and the counter are not modified. This extension does not inherit the accuracy of the original report; see the deployment record in the project docs.

The model is selected by `artifacts/selected.json`. The system uses HTTP or `Predictor` directly; window length, stable80 features, training mean/std and the hierarchical probability composition are loaded from the checkpoint inside the SDK, and the system keeps sending camera-plane COCO 17 points. Do not add skeleton normalization on the client or hard-code an older network structure. After a service restart, create new sessions and check `model_id` in `/health`.

### Current optimized default algorithm

**`opt_tcn_b4_w48`**: hierarchical residual causal TCN, 4 blocks, width 48, 62,453 parameters; stable80 input, 64-frame deployment window at 30 Hz. The action head outputs five action probabilities, and each target exercise has four conditional phase probabilities. Weights: `artifacts/opt_tcn_b4_w48.pt`, with mean/std saved alongside. The pose frontend is YOLO11n-pose@640.

The default decoder is `ordered_transition`, phase evidence threshold 0.25, debounce 0.033 s, and the target-action support gate for motion phases is still 0.15. Once the outward and peak phases have been seen, it can infer a short return phase from a stable posterior of being back at the start; the baseline must be confirmed again before a rep is counted, and cycles missing the peak or interrupted are never completed artificially.

| Metric | Original default GRU | Optimized default TCN |
|---|---:|---:|
| Validation count MAE | 1.00 reps | 0.10 reps |
| Test action Macro-F1 | 0.8213 | 0.9612 |
| Test joint phase Macro-F1 | 0.7007 | 0.8203 |
| Test count MAE | 1.80 reps | 0.10 reps |
| Test MAPE on non-zero sessions | 23.47% | 1.25% |
| Test cycle recall / precision | see original report | 0.9872 / 1.0000 |
| Same-run CPU temporal inference P50 | 8.926 ms | 5.619 ms |
| Parameters | 55,185 | 62,453 |

The test set has ten reconstructed sessions with 78 reference cycles; the TCN matches 77 cycles, misses one leg raise and has no spurious count. Phase labels are kinematic weak labels and counts come from reconstructed replays of the original segment repetitions of public mocap data, **not accuracy on real videos of older adults**. The original test set had been seen in an earlier round; this round locked the choice on the validation set and then reused the test set for comparison, so it is not a fresh blind test.

The optimized GRU reaches a count MAPE of 0% on this test replay, but its validation MAE is 0.80 reps and its compute cost is higher; seeing the test result is no reason to replace the locked TCN. The deployment choice balances validation performance and compute cost. Training process, CPU measurements, the three candidates and per-session results: see [OPTIMIZATION.md](OPTIMIZATION.md) and [RESULTS.md](RESULTS.md).

Timing is a single CPU temporal forward pass on the development machine, without the pose CNN or HTTP overhead. In projection stress tests the fixed model has a MAPE of 15% at both 15 and 75 degrees, and 24.58% at 45 degrees with coordinate noise of standard deviation 1 (original units); results under regular projections do not cover extreme viewpoints and real keypoint noise, so system acceptance has to test these input conditions too.

## 2. Startup and deployment limits

```bash
cd algorithms/exercise
python -m uvicorn homeease.api:app --host 127.0.0.1 --port 8002 --workers 1
```

- Use exactly one worker. Sessions and the idempotency cache live in the process; several workers cannot share them.
- Listens on localhost by default. For a home LAN the system is responsible for authentication, TLS, device and user permissions and request routing.
- A restart clears all sessions. The system stores final reports, starts a new session and marks the interruption in the UI; half a cycle cannot be resumed.
- At most 100 retained sessions; call DELETE after ending one. Reading state does not modify the model; every session has its own feature history and counter.
- Request bodies are limited to 2 MB and batches to 300 frames. Sending a batch every 5 to 15 frames keeps the UI responsive.
- The temporal model runs on the CPU per request. The video pose CNN can use the GPU separately; `Predictor(device="cuda")` uses CUDA in a Python integration.
- No raw image or audio is stored by this service; the HTTP interface only receives keypoints, and session reports contain no raw keypoints.

## 3. Exercises and complete cycles

| `exercise` value | Meaning | One complete cycle |
|---|---|---|
| `chair_sit_to_stand` | Chair sit-to-stand | sitting -> rising -> standing -> lowering -> sitting |
| `squat` | Squat | standing -> lowering -> squatting -> rising -> standing |
| `standing_leg_raise` | Standing straight-leg raise | standing -> lifting -> raised -> lowering -> standing |
| `hurdle_step` | Hurdle step | standing -> stepping -> raised -> returning -> standing |

The read-only `counter_stage` field reports the stage the original counter has confirmed: `-1` start posture not confirmed, `0` start posture, `1` outward, `2` peak, `3` return. It can differ from `phase`, the momentary most likely phase; after a complete return to the start posture `count` increases and the stage goes back to `0`. Pausing, an interruption or ending clears an unfinished stage to `-1` without clearing completed reps. The field is only for UI hints and does not change the model or the count.

The exercise is given by `exercise`, and the model also returns `recognized_exercise`. The still baseline can look the same for several exercises, so counting phases are conditioned on the chosen exercise; motion phases additionally require a probability of at least 0.15 for the chosen exercise, otherwise `other` is output and that cycle stops. A full raise of one side and the return to the baseline count as one cycle; left and right legs are not counted separately. Reaching the target does not end the session, and further complete cycles keep increasing the count.

The exercise names correspond to the actual movements of the public dataset. The dataset is not a training prescription for frail older adults; movement range, support and scheduling of home training are decided by the product or by professionals. The model cannot judge whether an exercise suits a particular person and outputs no therapeutic effect, balance score or fall risk.

## 4. Keypoint and time protocol

`schema_name="coco17"`, `coordinate_space="camera_xy"`. Each frame is 17x3, in this order:

| Index | Joint | Index | Joint |
|---|---|---|---|
| 0 | nose | 1 | left_eye |
| 2 | right_eye | 3 | left_ear |
| 4 | right_ear | 5 | left_shoulder |
| 6 | right_shoulder | 7 | left_elbow |
| 8 | right_elbow | 9 | left_wrist |
| 10 | right_wrist | 11 | left_hip |
| 12 | right_hip | 13 | left_knee |
| 14 | right_knee | 15 | left_ankle |
| 16 | right_ankle | | |

Each point is `[x, y, confidence]`: x to the right, y downward, confidence between 0 and 1. Pixel coordinates or any camera-plane unit with the same scale for x and y can be used. Do not divide x by the image width and y by the height separately; on non-square images that distorts angles. When mirroring, swap the left/right joint semantics, consistently in training and deployment. NaN, Inf, wrong lengths, out-of-range confidence and undeclared fields are rejected; |x| and |y| are limited to 10,000,000 so huge values cannot overflow float32.

`timestamp_ms` is the device's monotonically increasing observation time in milliseconds, between 0 and 1e15. It can start at 0 or come from a running monotonic clock. Timestamps must strictly increase within a batch and across batches. Do not use the HTTP arrival time instead of the frame capture time, and do not repeat the last frame to fill network gaps.

30 Hz input is recommended; other rates are accepted and causally resampled to 30 Hz by linear interpolation, processing interpolation times only after the current observation has arrived, without predicting the future. A gap above 250 ms resets the half cycle and the feature history. Low sampling rates cannot guarantee that short phases survive, and offline results at 30 Hz say nothing about 5 Hz.

Both shoulders, hips, knees and ankles need confidence >= 0.3. When the leg or torso length degenerates to zero the state becomes `degenerate_pose` and nothing is counted, even with high confidence. When the person is missing or heavily occluded, still send the current timestamp with 17 points of confidence 0 so the system enters `insufficient_pose` promptly. A stable `person_id` identifies the current person, and a change interrupts counting. The bundled video adapter refuses to count frames with zero or several people; a multi-person system should track and select the target person first.

## 5. HTTP routes

| Method and path | Purpose | Success status |
|---|---|---|
| GET `/health` | Confirm the service and model are loaded | 200 |
| POST `/v1/sessions` | Create and start an exercise session | 201 |
| POST `/v1/sessions/{id}/frames` | Send a batch of skeletons and get the current state | 200 |
| GET `/v1/sessions/{id}` | Read the current state or the final report | 200 |
| POST `/v1/sessions/{id}/control` | pause, resume, end | 200 |
| DELETE `/v1/sessions/{id}` | Release the session and its cache | 204 |

### Create a session

```json
{"exercise": "squat", "target_reps": 10}
```

`exercise` defaults to chair_sit_to_stand, `target_reps` to 10 (range 1 to 10000). The service returns a UUID `session_id` and the full state with `status="running"`, `quality="waiting_for_input"` and count=0.

### Send skeletons

The request looks like this; `keypoints` must be the complete 17 points. A real valid example is `examples/squat_frames.json`; do not send the descriptive string below.

```json
{
  "request_id": "device-a-batch-00001",
  "schema_name": "coco17",
  "coordinate_space": "camera_xy",
  "frames": [
    {"timestamp_ms": 0, "person_id": "person-1", "keypoints": "17 x [x, y, confidence]"}
  ]
}
```

`request_id` is required (1 to 128 characters). Each session keeps the results of its latest 128 requests: retrying an identical request returns the original result without running the model or adding reps, and the same request_id with different content returns 409. Retries beyond the cache window usually fail with 409 because the timestamps are stale. The system must send sequentially per session and wait for batch N to be acknowledged before sending batch N+1.

Every batch is validated completely (structure, finite values, time order) before processing, so an invalid second half never silently applies the first half. After a runtime error of the service, read the session's current timestamp to confirm what was accepted; if that cannot be confirmed, end the session and start a new one instead of assuming a count.

### Session control

```json
{"action": "pause"}
```

pause clears the unfinished cycle and keeps the completed count; resume also starts a fresh cycle. Frames sent while paused return 409. end freezes the final state; repeating end returns that state again, and no frames are accepted afterwards. After DELETE every route returns 404. The system manages the order of control commands; a session cannot be resumed after end.

## 6. Output fields

| Field | Type | Meaning |
|---|---|---|
| api_version | string | always v1 |
| session_id | string | session UUID |
| status | string | running, paused, ended |
| exercise | string | the training exercise chosen by the system |
| action_scope | string | open: original report mode; training: four-exercise training context |
| inference_views | string | original: single view; mirror_average: joint probabilities averaged over the original and mirrored input |
| recognized_exercise | string | the most likely exercise now, or other; can be uncertain during the still start |
| exercise_probabilities | object | class probabilities of the four exercises and other, summing to about 1 |
| phase | string | current phase of the exercise or other (see the exercise table) |
| phase_confidence | number | highest class probability of the joint action/phase classifier, not reliably calibrated |
| phase_probabilities | object | conditional phase probabilities within the chosen exercise; other=1 when the action support is too low in a motion phase |
| count | integer | completed valid cycles, not a rounded regression value |
| counter_stage | integer | stage confirmed by the original counter: -1 not ready, 0 start, 1 outward, 2 peak, 3 return |
| target_reps | integer | target set by the system |
| progress | number | min(count/target_reps, 1) |
| target_completed | boolean | count >= target_reps |
| elapsed_s | number | observation time from the first to the last frame, including pauses |
| active_duration_s | number | time actually covered by observations while running, excluding pause spans and gaps above 250 ms |
| valid_duration_s | number | time covered by consecutive valid visible observations without invalidating interruptions |
| quality | string | waiting_for_input, good, insufficient_pose, paused, ... |
| last_timestamp_ms | number/null | last accepted observation time |
| interruptions | array | reason and timestamp_ms; an interruption resets the half cycle |
| cycles | array | completed repetitions with rep_index, start_s, end_s, duration_s |
| model_id | string | id of the temporal model actually selected |
| model_validation | string | marker of the experimental validation scope of the model |
| request_id | string | frames route only: current request id |
| accepted_frames | integer | frames route only: frames accepted in this batch |
| new_cycles | array | frames route only: cycles added by this batch |

Cycle times are seconds relative to the first frame of the session. `start_s` is when the start baseline phase was confirmed after debouncing, `end_s` when the return to the baseline was confirmed; they are not exact hand-labelled medical start and end points. quality can show good right after visibility returns, but a complete new cycle is needed before count increases. `interruptions` records reasons such as timestamp_gap, person_changed, insufficient_pose and paused.

## 7. Errors and how the system handles them

Business errors: `{"error":{"code":"..."},"api_version":"v1"}`. Validation errors use the standard FastAPI `{"detail":[...]}` with loc, msg and type; clients should not depend on the error text.

| HTTP | code / source | Handling |
|---|---|---|
| 404 | SESSION_NOT_FOUND | create a new session, never reuse the old id |
| 409 | SESSION_NOT_RUNNING | stop sending, check for paused / ended |
| 409 | SESSION_ENDED | store the final state, then create a new session |
| 409 | TIMESTAMP_NOT_INCREASING | fix the queue order; never silently change the original frame timestamps |
| 409 | REQUEST_ID_CONFLICT | fix the idempotency id generation; different content cannot reuse an id |
| 413 | BODY_TOO_LARGE | send smaller batches |
| 422 | parameter / frame validation | fix the schema, enumeration, coordinates, confidence or count |
| 429 | SESSION_LIMIT_REACHED | release old sessions |

After a network timeout, resend the same request_id with exactly the same JSON field values. A degraded quality is a normal business state returned with 200; the UI shows that counting is temporarily unavailable and must never fill missing frames with the last trusted phase.

## 8. Runnable Python HTTP client

```python
import json
from pathlib import Path
import requests

base = "http://127.0.0.1:8002"
record = json.loads(Path("examples/squat_replay.json").read_text())
created = requests.post(base + "/v1/sessions", json={"exercise": record["exercise"], "target_reps": 10})
created.raise_for_status()
sid = created.json()["session_id"]
for start in range(0, len(record["frames"]), 10):
    body = {"request_id": f"batch-{start}", "frames": record["frames"][start:start+10]}
    response = requests.post(f"{base}/v1/sessions/{sid}/frames", json=body, timeout=30)
    response.raise_for_status()
    print(response.json()["count"])
ended = requests.post(f"{base}/v1/sessions/{sid}/control", json={"action": "end"})
ended.raise_for_status()
Path("session-report.json").write_text(json.dumps(ended.json(), indent=2))
requests.delete(f"{base}/v1/sessions/{sid}").raise_for_status()
```

## 9. SDK and pose CNN reuse

```python
from homeease.engine import Predictor, ExerciseSession
from homeease.pose import PoseEstimator

predictor = Predictor(device="cpu")  # weights shared per process, session histories are not
pose = PoseEstimator(device=0)  # or cpu; the CNN is chosen by selected_pose.json
session = ExerciseSession(predictor, exercise="standing_leg_raise", target_reps=10)
points, person_id = pose.estimate(frame_bgr)
state = session.update(capture_timestamp_ms, points, person_id)
session.control("pause")
session.control("resume")
final_report = session.control("end")
```

An existing pose CNN can bypass PoseEstimator and call update with the same COCO format. The SDK also accepts 12x3 in the order of COCO indices 5 to 16; to avoid ambiguity HTTP only accepts 17x3. SDK timestamps must also strictly increase, and the stable codes of the raised ValueError match the HTTP table. Multi-person camera tracking belongs to the system; the bundled adapter only handles single-person frames.

## 10. Integration checklist

1. GET /health returns ready, and model_id matches artifacts/selected.json.
2. All four examples replay completely, with counts matching reports/replay_sessions.json.
3. Resending a batch does not increase the count, and out-of-order frames return 409.
4. Pausing, occlusion, a change of person and gaps above 250 ms never join repetitions across the interruption.
5. Raw video stays local and the system only stores the reports it needs.
6. A service restart handles lost sessions correctly, and ended sessions are deleted promptly.
7. Continuous camera data has independent manual counts, and the report states model errors, occlusion and viewpoint conditions.
