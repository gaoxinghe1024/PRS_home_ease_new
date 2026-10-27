# Exercise Integration Record (Algorithm 2)

This record explains how Algorithm 2 was connected to live cameras in HomeEase, why a few **deployment-only** counting extensions were added on top of the original model, and how each was checked. The trained weights, feature functions, the original decoder thresholds and the experiment reports of Algorithm 2 are unchanged. The standalone SDK / API default (`action_scope="open"`, original decoding) still reproduces the published results; the extensions below are only enabled for the system's camera and upload sessions.

> The results of the extensions come from replays of recorded camera skeletons and from user feedback in field tests without per-repetition manual labels. They are **not** accuracy figures and do not inherit the 1.25% MAPE of the original report.

## Summary of deployment settings

| Exercise | `action_scope` | `inference_views` | Phase / counting policy | Max input gap |
|---|---|---|---|---|
| Squat | `training` | `original` | original decoder | 250 ms |
| Chair sit-to-stand | `training` | `original` | `counting_policy="pose_cycle_v2"` | 400 ms |
| Standing straight-leg raise | `training` | `mirror_average` | `counting_policy="pose_cycle_v1"` | 250 ms |
| Hurdle step | `training` | `mirror_average` | `phase_policy="action_gap_hold_v1"` | 250 ms |
| Sample replays (all) | `open` | `original` | original | 250 ms |

Every session state, report and diagnostic export records these fields, so a stored session can always be replayed with the policy it actually used.

## 1. Training action scope

**Problem.** On real camera skeletons the original decoder often decoded the chosen exercise as `other`. In motion phases it requires `P(chosen exercise) >= 0.15`, and with real input most probability went to `other` (around 1-2% support for the squat during the descent), even when the four training exercises clearly favoured the squat. Repetitions were cleared before completion, so live squats counted zero.

**Change.** With `action_scope="training"`, used only when the user has explicitly chosen one of the four exercises, support becomes `P(chosen) / sum(P(four training exercises))`; the original 0.15 gate still separates the four exercises. Phase posteriors, the 0.25 phase threshold, the 0.033 s debounce, the four-phase order and all interruption rules are unchanged, and the original five-class probabilities are still returned.

**Checks.** Removing the gate entirely also produced 3 counts but gave 414 false cycles in 68 negative checks, so it was rejected. The adopted normalization left all 16 public positive sessions unchanged and kept all 68 negative checks (shoulder negatives against four targets, cross-exercise input and still input) at zero. A recorded live session went from 0 to 3 through both the SDK and the full HTTP chain, and the user confirmed the live squat count as accurate.

## 2. Mirror averaging for the two leg exercises

**Problem.** After on-screen guidance, straight-leg raise and hurdle step still counted zero. With about 47 s of continuous valid input, the conditional peak probability of the straight-leg raise never exceeded about 0.04, far below 0.25, so occlusion alone did not explain it, and the SDK gave the same zero as the web page.

**Change.** For these two exercises in `training` sessions, inference runs on the original input and on its left-right mirror (x negated, the six left/right joint pairs swapped, exactly the augmentation of the original training script), with two independent causal histories and the same weights. The 17-class joint probabilities are averaged before the original decoder and counter. Any interruption of the original input clears both histories.

**Checks.** A truncated hurdle-step recording went from 0 to 3 in the SDK and the HTTP chain; the straight-leg recording stayed at 0. The 8 matching public sessions did not regress (one improved from 9 to 10), and 32 cross-exercise / shoulder negatives and 16 still checks stayed at 0. Time cost on the CPU: about 2.4 ms P50 per step for one view, about 5.4 ms for two.

Alternatives tried and **not** deployed: phase-probability calibration (counted a real hurdle-step input as 4 straight-leg raises), extra class-competition rules, two other trained candidates (false counts on public cross-exercise checks), lower thresholds, fixed arm coordinates, and fine-tuning only the phase heads.

## 3. Phase continuity for the hurdle step (`action_gap_hold_v1`)

**Problem.** In hurdle-step sessions the decoder briefly rejected the action just after the support gate was met, clearing a repetition that the counter had already advanced to outward / peak / return.

**Change.** Hurdle-step training sessions hold the confirmed phase for at most 0.15 s while waiting for the next valid decision. During the hold only the already confirmed phase is passed to the original counter; no new phase or repetition can be confirmed, and after the timeout the original interruption applies. Insufficient keypoints, a change of person, pausing and gaps above 250 ms still clear everything immediately.

**Checks.** A recorded hurdle-step clip went from 4 to 5; public matched sessions did not regress and 48 cross-exercise and still checks stayed at 0. Applying the same rule to the straight-leg raise or smoothing phase posteriors with an EMA degraded public sessions, so neither was used.

## 4. Batched mirror inference

Mirroring doubled the model calls, and each real observation can require several 30 Hz interpolation steps, which lowered the effective camera sampling rate (temporal processing P50 about 63 / 84 ms per frame for the two leg exercises on CUDA). `MirroredPhaseStream.update_many` now evaluates all steps available at the current observation, for both views, in one batch. Each step still builds its own causal window of the original length from past data only, and outputs are passed to the decoder and counter in the original time order.

Temporal processing dropped to about 13 / 15 ms P50. A frame-by-frame comparison of the old and new paths (public inputs fed at 7.5 Hz, cross-exercise and still checks, five field recordings) showed identical counts, phases, recognized actions, quality, interruptions, cycles and valid durations, with a maximum posterior difference of about 1e-6 on CUDA and 5e-7 on the CPU.

## 5. Pose-based cycle confirmation for sit-to-stand and straight-leg raise

**Problem (2026-10-23).** Continuous visible sit-to-stand input still lacked the model's rising phase, and the motion part of straight-leg raises was still rejected by the action support gate. The original SDK also counted zero, so the web page was not losing counts.

**Change.** Camera / upload training sessions of these two exercises use an explicit deployment counting policy that derives the phase given to the existing `CycleCounter` from the end points and direction of the visible skeleton, with 0.1 s phase debouncing:

- `pose_cycle_v1` (straight-leg raise): both feet down first, then the same fairly straight leg raised and put back. Half movements, bent-knee lifts and switching legs without putting the foot down cannot complete a cycle.
- `pose_cycle_v2` (sit-to-stand): a confirmed seated start, standing fully upright, then seated again. Knee angle and normalized heights use end-point conditions with a gap between them. `setup_pose` reports standing / seated / transition / waiting separately, so standing at the start shows "Standing recognized - sit steady to start" instead of an error. Gaps of 250-400 ms with continuously valid keypoints keep the cycle; above 400 ms, insufficient confidence, a change of person or pausing still clear it. v2 also batches the single-view temporal inference of all steps available at the current frame.

The model's action and phase probabilities are still computed and stored: `model_phase` is the model's phase and `phase` the one used for counting. This policy is **not** the original classifier and does not inherit its accuracy; a skeleton also cannot prove that the hips actually touched the chair. Squat and hurdle step keep their policies, and the SDK / API default stays `original`.

**Checks.** Recorded inputs went from 0/0 to 3/2 (sit-to-stand) and from 0/0 to 3/3 (straight-leg raise); the newest sit-to-stand recording went from 0 to 5. Regression checks cover complete cycles, still input, half movements, bent knees, switching legs, interruptions, the 400 ms gap rule and the causality of batching.

## Integration fixes

- **Uploaded video** used to go through BGR decode, JPEG encode and JPEG decode before pose estimation. The worker now decodes frames directly and submits raw BGR, like the original `scripts/run_video.py`, keeping the file's frame rate and timestamps (keypoint error 0 on spot checks).
- **Shared camera.** One process opens each camera and passes raw BGR frames to both the video and exercise algorithms through shared memory (keypoint error 0 against the original pose model). Frames older than the session start are skipped, which removed duplicate 0 ms timestamps; camera hand-over between video monitoring and exercise sessions rebinds the owner without reopening the device.
- **Diagnostics survive control actions.** Pause, resume, end and snapshots keep the latest input diagnostics (keypoint completeness, missing joints, sampling rate, largest gap, interruptions, continuous valid duration). Diagnostics never take part in inference or counting.
- **UI.** The page shows the confidence of the eight required joints and a five-step progress indicator driven by the counter's read-only `counter_stage`; the count shown is always the API's `count`. "Body visible, selected exercise not yet recognized" is shown separately from input problems.

## Diagnostics export

The download icon next to a session exports up to the last 180 s or 1,800 skeleton observations (no images or video) through `GET /api/sessions/{sid}/diagnostics.json`: `schema_name: coco17`, `coordinate_space: camera_xy`, the raw `frames` with `timestamp_ms`, `person_id` and 17 `[x,y,confidence]` points, the session `state` (phase probabilities, count, interruptions) and `truncated`. Ended sessions are kept in `runtime/exercise-traces/` and deleted together with the session record; they are never written to Git.

## Exercise guidance

The original training data are oblique projections at 25 / 45 / 65 degrees, so the pages recommend turning about 45 degrees to the camera with the head and both feet in frame and the camera level at hip height. The guides for the two leg exercises are rendered from real training skeletons (`m06_s01_e10`, `m02_s04_e08` at the original 45 degree projection) and include the arm posture of the training data: upper arms at shoulder height with bent elbows for the straight-leg raise, arms slightly out with hands at shoulder height for the hurdle step. These guides are visualizations of the training data, not counting rules or evidence of accuracy.

## Field-test history

| Date | Exercise | Observation |
|---|---|---|
| 2026-10-16 | squat | 0 before the training action scope, then 3; user confirmed the count |
| 2026-10-16 | sit-to-stand | 2 after guidance; user confirmed |
| 2026-10-16 | straight-leg raise / hurdle step | 0 / 0 after guidance; about 5 each performed |
| 2026-10-16 | both leg exercises, 5 performed each | 3 / 1 before batching (system and SDK identical) |
| 2026-10-16 | both leg exercises after batching | 6 / 6; user reported "no problems" (no per-repetition labels) |
| 2026-10-23 | sit-to-stand with `pose_cycle_v2` | user reported the result as good |

Open limits: no per-repetition manual timing exists for the field tests, the leg exercises still show some cross-exercise confusion on recorded inputs, and none of the extensions has been validated on a larger group of older adults. Each change is a deployment adaptation for the camera domain, not a retrained or re-evaluated model.
