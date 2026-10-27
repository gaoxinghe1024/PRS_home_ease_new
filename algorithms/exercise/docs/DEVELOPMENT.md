# HomeEase Algorithm 2: Development Record

## Task and scope

Based on sections 3.2, 9.3 and 10 of the project proposal and slides 15 to 17 of the original presentation, implement camera-skeleton-driven phase recognition for the chair sit-to-stand, complete-cycle counting, session logs and a system interface. Pose CNNs, skeleton kinematic features, temporal models and single-model / ensemble strategies are compared, and accuracy and compute cost are measured. The original slides and proposal are not modified.

A count is only added for a complete `sitting -> rising -> standing -> lowering -> sitting` cycle. Pausing, insufficient visibility and broken timestamps must never join a cycle across the interruption. Repetitions, target progress, timing and data quality are recorded; no unvalidated medical, balance or posture-quality score is provided.

## Initial environment and data survey

- Date: 2026-10-08, user time zone Asia/Shanghai.
- The working folder initially contained only the two original Office files: no code and no Git repository.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB.
- Existing environment: Python 3.11.15 with PyTorch 2.11.0+cu128, CUDA available.
- The project uses its own virtual environment and reuses the existing PyTorch without changing the global environment.
- The official UI-PRMD home page returned 403 at the time. The public mirror `jkaethee/UI-PRMD_Exercise_Model` contains the original Segmented Movements and Incorrect Segmented Movements text files. Downloads pin a commit and store the SHA-256, Git blob SHA and original URL of every file.
- UI-PRMD `m05` is sit-to-stand and `m01` deep squat, which serves as a challenging negative. The mirror's preprocessed `.npy` files are not used as raw input, to avoid inheriting an unclear filtering method.
- The official MM-Fit site was reachable (about 1.74 GB), but its exercise set has no chair sit-to-stand, so it cannot serve as a test set for this task.
- Original skeleton segments / action annotations are not manual per-frame phase annotations. The source of the weak labels and the scope of the count ground truth are recorded as they are; derived phase labels are never called a manual gold standard.

## Development progress

1. Done: read the requirements, checked hardware and the existing environment, confirmed the four groups of experiments and the integration documentation requirements.
2. Done: downloaded the real data and checked the skeleton format, sit-to-stand segment boundaries and annotation capability; all 2,000 files passed the SHA-256 and Git blob SHA check offline.
3. Done: features, models, causal inference, counter, HTTP / SDK, camera entry point and interface tests.
4. Done: training of 11 model configurations, common re-evaluation of single models / ensembles, 28 decoder candidates, 4 pose CNN measurements, viewpoint and noise stress tests and the result summary.
5. Done: real-weight streaming / batch inference, HTTP / SDK replay consistency for all four exercises, a video entry smoke test, 24 tests, service startup and the final handover material.
6. Done: the requested self-check optimization: three strong candidates trained, causal normalization and a hierarchical network, validation-set calibration, a window-shrinking equivalence check and a selection lock before testing. The detailed process and every epoch record: see [OPTIMIZATION.md](OPTIMIZATION.md).

## Final result

The optimized default model is **`opt_tcn_b4_w48`**: a hierarchical residual causal TCN with 4 blocks, width 48, 62,453 parameters, stable80 features and a 64-frame deployment window. The best training epoch is 10; the original 96-frame training window was shrunk to 64 after the validation equivalence check. The default decoder is ordered_transition with threshold 0.25 and debounce 0.033 s. Weights, statistics and the actual selected configuration are all saved. Current CPU measurements: see [RESULTS.md](RESULTS.md) and the optimization record. The default frontend is still YOLO11n-pose@640, with PCK 93.55% on the COCO subset and a full GPU call P50 of 15.33 ms / P95 98.22 ms.

Current test action Macro-F1 **0.9612**, joint phase Macro-F1 **0.8203**, count MAE over 10 reconstructed sessions **0.10 reps**, MAPE over the eight non-zero reference sessions **1.25%**. The corresponding values of the original default GRU are 0.8213, 0.7007, 1.80 reps and 23.47%. The current model matches 77/78 cycles, misses one straight-leg raise and has no spurious count; the counts of the other three exercises are all correct. It reaches <=5% on the limited public replay, but **accuracy acceptance on real home video is still open**.

Of the three optimization candidates, the GRU happens to have no count error on the test replay, but its validation MAE is 0.80 reps, higher than the selected TCN's 0.10 reps, and its compute cost is higher. The TCN was chosen on validation performance and CPU latency, and a hash lock of model and decoder was recorded before this round's test. The original test set had been seen in the first round and during diagnosis, so reusing it here is not a new independent blind test; the limits of weak labels, 2D projection and two test participants remain.

The handover includes the README, this development record, OPTIMIZATION.md, API.md, the machine-readable OpenAPI, DATA_CARD.md, RESULTS.md, the first-round and three optimized checkpoints, public pose weights, the selection lock, sources and hashes of the raw data and images, per-experiment / per-session results, and replays of the four exercises. The service starts without retraining.

## Requirement changes and implementation decisions

Two explicit requirements were added later: an interface document the system team can use directly, and more than one exercise (three or four instead of only sit-to-stand). The final implementation therefore covers four exercises: chair sit-to-stand `m05`, deep squat `m01`, standing straight-leg raise `m06` and hurdle step `m02`. Shoulder raise `m07` was downloaded as a real negative that must not be counted. The earlier sit-to-stand-only exploration was stopped, and the final report and weights all come from the four-exercise version.

The four exercises relate to lower-limb and balance training, but the squat, hurdle-step and leg-raise ranges of the public data are not a home training plan suited to frail older adults. The implementation provides recognition and statistics; it has not validated a clinical fall-prevention effect and prescribes no individual training.

## Complete data processing

### Download and provenance

`scripts/download_data.py` finds the original text files through the read-only GitHub API and pins mirror commit `475c66aba07731adbe1c90a15462fd30be564593`. Only Kinect Positions and Angles are used, including correct and incorrect executions. The five exercises total 2,000 text files and 98,438,042 bytes. Each skeleton segment is a pair of position / angle files, 1,000 segments in total: 10 people per exercise, each with 10 correct and 10 incorrect executions. The mirror and the data are third-party sources; the project does not read public downloadability of the mirror as an unconditional redistribution licence.

Original units are kept. The download manifest stores the URL, size, SHA-256 and Git blob SHA of every file. Training does not use the mirror's arrays that were already filtered, scaled or split into derived `.npy` files; the skeletons are rebuilt from scratch. Network errors are retried with exponential backoff, and no substitute samples are synthesized.

### Coordinate reconstruction

The raw `Tx66` data contains local coordinates of 22 joints and the corresponding Euler angles. The root is absolute; the other joints are rebuilt by forward kinematics along the tree:

```text
R_global[j] = R_global[parent] @ R_local[j]
p_global[j] = p_global[parent] + R_global[parent] @ p_local[j]
```

Euler angles use the standard xyz rotation matrices, accumulated along each parent for the root, torso, both arms and both legs. Compared with treating local offsets directly as camera positions, this keeps the real knee flexion, hip flexion and leg raising.

Twelve COCO-compatible joints are kept: both shoulders, elbows, wrists, hips, knees and ankles. The real 3D skeleton is projected onto an oblique camera plane with the vertical axis flipped so y points down. Training views are 25, 45 and 65 degrees; regular validation / test uses 45 degrees. These are **programmatic projections of real skeletons as augmentation**, not samples from an actual RGB camera, and results must keep that distinction.

### Participant split

The split happens before projection, normalization and windowing: s01-s06 training, s07-s08 validation, s09-s10 test, i.e. 600 training, 200 validation and 200 test segments. Correct and incorrect executions follow their original participant, so projection variants can never share a person across sets.

The fixed split makes the course project reproducible, but with only two test participants it cannot support a stable generalization claim for all older adults. The test set takes no part in normalization, checkpoints, ensemble members, debounce thresholds or deployment model selection.

### Labels and count ground truth

UI-PRMD's original action type, participant and segment repetition number are source annotations. The data does not provide the manual per-frame phase gold standard described in the slides. The project explicitly uses `weak_kinematic_offline_v2`: sit-to-stand and squat use the root height, hurdle step the height of the higher knee relative to the root, and straight-leg raise the height of the higher ankle relative to the root. After offline Gaussian smoothing, the motion range is normalized with the 5% / 95% quantiles of the segment, and baseline, outward, peak and return phases are derived from the low end, the high end and the direction of the derivative. The squat direction is the opposite of sit-to-stand.

This is a kinematic weak label. Label generation may use the whole segment because it is an offline training target, but online input only uses the current and past frames, never the future, whole-segment quantiles, true counts or segment boundaries. Phase F1 reflects the fit to these weak labels and must not be reported as "manual phase recognition accuracy".

Correct-execution segments are concatenated per participant from e01 to e10, shifting the horizontal and depth axes to the initial foot position to remove per-segment camera offsets while keeping the original timing and vertical motion. Each person and exercise forms one **reconstructed replay**. For squat, leg raise and hurdle step each segment completes one baseline-return cycle, so the source reference is count=10. Sit-to-stand segments are "stand - sit - stand", so between the first and last seated posture there are only 9 complete "sit - stand - sit" cycles, and the reference is count=9; the half cycles at the start and end are not completed, and the data is not reversed or rotated cyclically.

Each set has four exercises x two people = 8 positive replays, plus negative shoulder-exercise replays of the two people: 10 sessions and 78 positive cycles. The count ground truth is derived from the source segment numbers and complete-cycle boundaries; the reference boundaries for exact cycle timing come from the weak labels, so the two have different validation power. Cycles are matched with a tolerance of +/-0.5 s by greedy one-to-one matching to count TP, FP and FN.

The motion signals of all correct segments are plotted in 40 audit figures, `reports/figures/source_audit_mXX_sYY.png`. Some leg-raise / hurdle-step data has spikes and jitter and cannot be treated as high-quality continuous video labels. The count log keeps every predicted event so false and missed counts can be inspected; MAPE excludes sessions with reference count=0, and false counts on negative sessions are reported separately.

## Skeleton features

This section and the following sections on architecture ablation, training and the counter describe the first-round implementation. The current deployment adds stable80, hierarchical heads and validation-set optimization on top; the exact configuration and training parameters are in [OPTIMIZATION.md](OPTIMIZATION.md). Do not rebuild the current weights from the original hybrid / 64-frame training configuration in this section.

Each frame is translated to the hip centre and normalized by the average thigh plus shin length of both legs, so features do not depend directly on body height or image scale. Three feature groups share the participant split and training procedure:

| Feature group | Dimensions | Content |
|---|---:|---|
| coords | 24 | normalized 2D coordinates of 12 joints |
| kinematics | 16 | cosines of left/right knee and hip angles, hip height relative to the ankles, torso horizontal / vertical direction, ankle distance, plus 8 causal velocities |
| hybrid | 64 | 24 coordinates, 8 kinematic values, 24 coordinate velocities, 8 kinematic velocities |

The causal EMA coefficient is 0.65 and only mixes with the previous observation. Velocities are backward differences over the real dt, clipped to [-8, 8] so jitter cannot explode. Training and serving share one `FeatureExtractor` rather than two separate preprocessing implementations. Training statistics are per-feature mean/std with std at least 0.1, stored as they are in the checkpoint and used in deployment.

## Temporal models and architecture ablation

The models output 17 joint action/phase classes: four phases for each of the four exercises, plus other. Action probabilities are the sums of each exercise's four phase probabilities. Action recognition does not depend on the exercise chosen by the system. Counting phases are conditioned on the training exercise chosen by the user, which resolves the natural ambiguity of standing still belonging to several exercises; motion phases additionally require a probability of at least 0.15 for the chosen exercise, so a cycle is never created from a tiny conditional probability.

Everything is implemented in PyTorch with our own networks, training loop and state machine, without any external action-classification service. Established convolution, recurrence and attention operations are used; a simple MLP is not used as the final model.

| Experiment | Change |
|---|---|
| tcn_b2_w32 / b3_w32 / b4_w32 | 2, 3, 4 residual blocks |
| tcn_b3_w32 / b3_w64 | 32 and 64 hidden channels |
| ds_tcn_b3_w32 / ds_tcn_b4_w64 | depthwise separable convolutions instead of regular ones |
| gru_b2_w32 / gru_b2_w64 | two-layer unidirectional GRU, different widths |
| transformer_b2_w32 | two-layer, four-head causal Transformer |
| tcn_coords / tcn_kinematics / tcn_b3_w32 | coordinate, kinematic and hybrid features |
| soft_vote_top2 | equal-weight probability ensemble of the two models selected on validation |

The TCN uses kernel width 3 with two causal convolutions per block and dilations 1, 2, 4, 8. With 2, 3 and 4 blocks the receptive fields are 13, 29 and 61 frames, all within the 64-frame deployment window. The depthwise separable version splits depthwise and pointwise convolutions. The GRU is unidirectional, and the Transformer uses an upper-triangular attention mask with sinusoidal positional encoding. Tests verify that changing future frames leaves the current output unchanged.

## Training procedure and parameters

- Fixed random seed 42 for Python, NumPy and PyTorch.
- 64-frame input windows, about 2.13 s. Training windows are sampled per segment with stride 16 frames, plus the final-frame window.
- Before the start of a segment the input is padded by repeating the first frame; padded labels are ignore_index=-100 and take no part in the loss.
- Batch size 64, single-process DataLoader, avoiding the start-up cost and state differences of Windows worker processes.
- AdamW, learning rate 0.002, weight decay 0.001, cosine learning-rate schedule.
- Dropout 0.1 and feature noise with standard deviation 0.015 during training; no random noise in validation and test.
- Cross-entropy weighted by the inverse square root of the training label frequencies, gradient norm clipped to 1.
- 25 epochs per configuration; joint phase Macro-F1 on validation every five epochs, keeping the checkpoint with the best validation score.
- Local GPU: RTX 4060 Laptop 8 GB; CPU: Intel Core i7-12700H; CPU timings always use 4 PyTorch threads.

Early sit-to-stand training had essentially converged after 5 to 10 epochs; afterwards the loss kept falling while validation scores fluctuated, showing that small public skeleton datasets overfit easily. The four-exercise phase models therefore use a common budget of 25 epochs and are evaluated at the validation checkpoint instead of the last training epoch.

## Complete-cycle state machine and real-time behaviour

The generic phase order is `rest -> outbound -> hold -> return -> rest`, mapped to the display names of each exercise. Standing at the start does not add a seated phase for sit-to-stand, and a half cycle at the end is not counted. Baseline phases need debounced confirmation; a cycle lasts at least 0.6 s and at most 30 s. The state machine keeps the complete progress already observed, does not create duplicate cycles when jitter near a boundary steps back one phase, and never allows a missing outward phase, peak or final return to the baseline.

Pausing, low keypoint confidence, a change of person and gaps above 250 ms immediately drop the half cycle and clear the feature window. After recovery the completed count is kept, but a complete new cycle is required. The service causally resamples to 30 Hz, and input quality checks are separate from the phase model, so missing data is never read as "no motion".

A short peak may have a solid posterior probability without becoming the argmax, so two decoders are implemented: per-frame argmax, and a posterior state machine that advances in the complete order. The latter only advances when the next required phase has enough probability: the baseline needs >= 0.5, and motion-phase thresholds of 0.25, 0.35 and 0.5 are candidates; phases cannot be skipped. It uses the model's predicted probabilities and never reads true joint heights or reference boundaries; tests confirm that nothing is counted when the peak is missing.

The straight-leg-raise data of validation participant s07 has high-amplitude spikes, and several models predicted almost no return phase, missing the whole session. So `ordered_transition` is also compared: once the model has confirmed the outward phase and the peak, a stable start-posture posterior >= 0.6 also counts as evidence of the return, which then enters the return phase and waits for the final baseline confirmation. It allows inferring a very short return from "raised and back at the start", without inventing a peak or completing cycles across invalid data; the displayed phase stays the model's raw prediction. Tests verify that a missing peak or no return to the baseline is never counted.

Debounce candidates are 0, 0.033, 0.067 and 0.1 s, so the three decoders have 28 parameter sets. Selection uses only the validation replays: lowest MAE first, then cycle recall, longer debounce and higher probability threshold. The same candidate grid is used to re-evaluate all models, and the default model is then chosen by validation accuracy and CPU latency. The runtime uses the selected parameters. This is a structured decoding comparison for the measured problem of little training data and short transition phases. More candidates also raise the risk of overfitting a small validation set, so further threshold trials cannot replace evaluation on new participants.

## Pose CNN comparison design

Official pre-trained Ultralytics YOLO11n, s and m pose weights are compared as n@320, n@640, s@640 and m@640. They are public pre-trained perception frontends and are not presented as COCO networks trained from scratch; the temporal action model is what this project actually trained.

The official COCO 2017 keypoint annotations were downloaded and sorted by image_id, taking the first 100 validation images that contain a non-crowd person with >= 6 annotated keypoints and area > 4096: 198 annotated people and 2,482 visible keypoints. Image hashes and sources are stored in full, with no picking of samples the models do well on.

Predicted boxes are matched one-to-one to ground truth at IoU >= 0.5. PCK@0.2 uses 20% of the longest side of the person bbox as the distance threshold, and keypoints of missed people count as errors. The mean OKS with misses counted as zero is also reported. This PCK and mean OKS are not the standard full COCO AP; a lenient PCK is not reported as fine joint accuracy or chair counting accuracy. The subset only serves a relative comparison of lightweight frontends.

Timing includes real image preprocessing, the forward pass, post-processing and returning results, with batch=1 and 8 warm-up runs; the GPU is synchronized before and after, and P50 / P95 are reported. The first measurements ran in parallel with training and were contended, so independent timing was repeated after training; the final report counts. Among candidates within 3 percentage points of the best PCK, the one with the lowest P50 is selected.

## Metrics, selection rules and result files

Joint phase Macro-F1, action Macro-F1, the classification confusion matrix, count MAE, MAPE, relative count accuracy as `max(0, 1-MAPE)`, exact-session rate, complete-cycle precision/recall and missed/spurious cycles are all stored.

Single-model CPU timing uses a fixed 1x64xD input with the model in eval and inference_mode, 30 warm-up runs and 200 measurements, recording the parameter count, raw weight size and P50 / P95. The ensemble's temporal CPU latency in the main table is the sum of the two members measured separately and is marked as such in the JSON; in particular the summed P95 is not the P95 of the joint distribution. The final deployment path was verified separately through the SDK / HTTP and perception entry points.

The default model is selected on validation results only: keep the candidates whose count MAE is within 0.25 of the best, of those keep the ones whose joint phase F1 is within 0.02 of the best, and finally choose the lowest CPU P50. The test set is only used for the final report and is never a deployment selection criterion.

The final numbers are aggregated automatically from the experiment JSON into [RESULTS.md](RESULTS.md); the main comparison table is `reports/comparison.csv`, per-session and per-class details are in `reports/experiments.json`, and the pose frontends in `reports/pose_comparison.json`. Read the per-exercise results together with the report: an overall average can hide missed hurdle steps or leg raises.

## Interface and system integration

`homeease.engine.Predictor` holds the weights shared per process, and `ExerciseSession` holds the state of one person's session. HTTP routes, data schema, error codes, idempotency and lifecycle are defined in [API.md](API.md). `scripts/export_openapi.py` exports the machine-readable interface from the real FastAPI app so the handwritten document cannot drift from the code.

Frame batches are validated completely first; request_id idempotency over the latest 128 requests uses a SHA-256 content digest; all timestamps strictly increase, and each session is locked. The body limit is 2 MB, also for requests without Content-Length. Pause, resume, end and release are supported, and completed events, durations, the training target and interruptions are kept. Sessions are in memory with a single-worker deployment; the system stores the final report and handles service restarts.

The camera program supports local files and webcams and writes a JSON report without uploading video. The shared PoseEstimator lets Algorithms 1 and 2 use the same keypoints. The SDK / HTTP infer no training prescription, and their output includes the actual model_id and the current validation scope so the system can trace them.

## Verification and issues found

Automated tests (used during development) covered complete cycles of all four exercises, jitter at phase boundaries, future-frame isolation of every model, atomic validation of invalid batches, idempotency, out-of-order frames, the session lifecycle, participant isolation, pause durations, recovery after interruptions, refusing to count degenerate skeletons, ordered probability decoding, and batch / streaming inference of the real checkpoints with HTTP / SDK consistency for all four exercises.

During development, one preview of the common re-evaluation script did not load the checkpoint state_dict and showed an action F1 clearly inconsistent with the training report. Comparing inference of the saved weights on the same raw segment found and fixed it; the faulty preview never entered model selection or the delivered results. A consistency check of the SDK / batch inference of real checkpoints was added afterwards to make sure reports come from the actually trained weights.

After the long main training command printed the completion of all 11 models and the selection result, the tool returned exit code 1 without a Python traceback. All checkpoints were then reloaded, evaluated together and checked, all successfully; the final evaluation, replay, video entry and test commands returned 0. The state of the long-running process was never taken as proof that the model files are valid; actual reloading and inference were the basis.

The first round passed 24 tests, with gru_b2_w64 as the model id in health. The optimization round added tests for hierarchical causality / probability composition, stable80 invariance and on-demand window equivalence, and verified the real weights and interface again (28 tests passed).

## Reproduction commands and folder layout

A regular run in order: download_data, prepare_data, train_compare, evaluate_select, benchmark_pose, replay_demo, export_openapi, build_report. The common re-evaluation must run after all four-exercise training has finished, so no early single-exercise checkpoint is mixed into the final results.

```text
algorithms/exercise/
  homeease/               features, networks, counter, SDK, API, pose CNN
  scripts/                data download, preprocessing, training, evaluation, demos, reports
  artifacts/              all training checkpoints, pose weights, the default selection
  data/raw/uiprmd/        raw text files and the traceable manifest (downloaded, not in Git)
  data/processed/         rebuilt skeletons, weak labels, participant split
  data/coco/              separate pose-estimation comparison data (downloaded, not in Git)
  reports/                measured results, events, figures
  docs/                   development record, API, OpenAPI, results and data card
  examples/               four-exercise reconstructed replays of real test participants
```

## Limitations and next validation steps

1. Only two public-data participants are in the test set; a single-seed course experiment cannot represent generalization to older adults.
2. Projected 2D skeletons have a different error distribution from a real pose CNN; the four-exercise skeleton results do not validate end-to-end counting on real video.
3. Phase labels are weak labels; independent manual phase labels and a count gold standard on continuous natural video still need to be collected.
4. Reconstructed sessions only contain consecutive segments of the same exercise and one shoulder negative; real household disturbances such as walking, bending or changing chairs are not covered yet.
5. Frontal views, occlusion, loose clothing and legs leaving the frame can fail; viewpoint augmentation cannot replace real multi-view tests.
6. Incorrect-execution segments are used for action / phase training, but no "movement quality acceptable" classifier was trained; valid in count only means a completely visible cycle.
7. Interface and offline replay validation is not a test with a real device camera; seven-day stability, system notifications and real use by older adults were not tested.
8. Whether the proposal's <=5% target is met can only be decided by evaluation on independent continuous video. If it is not met, that will be reported as it is.

The priority is real continuous video of the four exercises, split by participant, with manually labelled repetitions and four phases, plus negatives from daily activities. The current PoseEstimator can extract keypoints in batch exactly as in deployment, and the current training and evaluation scripts can be reused for fine-tuning. Do not tune thresholds on failing test videos and then report them as the original test.

## Summary of the optimization round

The request was to self-check and optimize only the 1-3 best algorithms. So two TCNs (4 / 5 blocks, width 48) and a two-layer GRU (width 64) were optimized further, without retraining all candidates. Common improvements are stable80, hierarchical action / phase learning, supervision of the last eight frames of each window, cross-repetition context of training participants, speed / mirror augmentation, AdamW regularization and validation-set decoder calibration. The separate optimization record stores all candidates, metrics every five epochs, the best checkpoints, the selection lock and compute costs.

After finding that pre-expanding long windows failed to allocate a contiguous array of about 994 MiB, an on-demand slicing dataset was implemented and tested item by item for equivalence. All three optimized weights were reloaded in a later independent evaluation and their validation / test probability caches completed. The long training command again returned a non-zero status after printing the completion and selection results, without a Python exception; the subsequent independent calibration, test evaluation, deployment audit, real HTTP replay and tests all returned 0, so actual reloading and inference are the basis of the delivery.

The selected TCN was shrunk from 96 to 64 frames over all validation data with a maximum probability difference of 5.36e-7 and identical action and phase argmax everywhere. After calibration MAE=0.10, threshold 0.25, debounce 0.033. The checkpoint hash was recorded before the formal test, and the fixed model then ran the test and stress checks. In the same run the CPU single forward P50 dropped from 8.926 ms for the original GRU to 5.619 ms for the TCN, while parameters grew from 55,185 to 62,453.

Finally **28 automated tests passed**. The real localhost service was restarted with `opt_tcn_b4_w48`, and the SDK / HTTP counts of the s09 replays of the four exercises are 9 / 10 / 10 / 10 in both. Both people's shoulder negatives count zero for all four targets, a still posture counts zero for all four, and a still video made from real COCO images also passes the pose CNN to JSON report chain with count=0.

The regular 45 degree test projection reaches MAPE=1.25%, but the 15 and 75 degree stress tests give 15%, and 24.58% with coordinate noise of standard deviation 1 in original units. The optimization did not remove every viewpoint and noise problem, so the averages of the main table must not be reported as unconditional deployment accuracy. Raw data, labels and splits are unchanged; the first-round results are archived in reports/optimization/baseline. Detailed experiments, failure fixes and reproduction commands: see [OPTIMIZATION.md](OPTIMIZATION.md).

## References and third-party dependencies

- Project requirements: sections 3.2, 9.3 and 10 of the original proposal; slides 15-17 of the original presentation, later explicitly extended to four exercises and a system interface document.
- UI-PRMD official: <https://webpages.uidaho.edu/ui-prmd/>; pinned mirror used here: <https://github.com/jkaethee/UI-PRMD_Exercise_Model/tree/475c66aba07731adbe1c90a15462fd30be564593>.
- UI-PRMD coordinate references: <https://github.com/aakash2016/ui-prmd-python> and <https://github.com/tejas1904/UI-PRMD-Visualize-python-port>; the project implements its own forward kinematics and features.
- MM-Fit official: <https://mmfit.github.io/>, surveyed but not used for the final sit-to-stand / four-exercise training.
- COCO 2017 keypoints: <https://cocodataset.org/#keypoints-2017>; image licences follow each original image record and the COCO terms.
- YOLO11: <https://docs.ultralytics.com/models/yolo11/>; Ultralytics weights and library are subject to AGPL-3.0 / enterprise licence terms, to be confirmed by the system side before distribution.
- TCN: Bai, Kolter, Koltun, 2018, <https://arxiv.org/abs/1803.01271>.
- GRU: Cho et al., 2014, <https://arxiv.org/abs/1406.1078>.
- Transformer: Vaswani et al., 2017, <https://arxiv.org/abs/1706.03762>.
