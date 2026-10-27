# Algorithm 2: Exercise Recognition and Rep Counting

Skeleton-based temporal recognition of four home exercises with complete-cycle repetition counting. This folder contains the real-data download, training, architecture / feature ablations, a pose CNN comparison, the trained weights, a Python SDK, a local video / camera entry point, an HTTP API and the documentation.

Supported exercises: `chair_sit_to_stand`, `squat`, `standing_leg_raise`, `hurdle_step`. A repetition is only counted after the full four-phase sequence; after a loss of visibility, a pause or a change of person a complete new cycle is required.

**Deployed model:** the optimized **hierarchical causal TCN `opt_tcn_b4_w48`** (62,453 parameters, 64-frame window). On the fixed reconstructed test replays it reaches action Macro-F1 0.9612, joint phase Macro-F1 0.8203, count MAE 0.10 reps and MAPE 1.25% (the first-round default had a MAPE of 23.47%). Model and decoder were selected on the validation set and locked before the test. These results come from projected public mocap data and do not represent accuracy on real camera video of older adults. Full process: [OPTIMIZATION.md](docs/OPTIMIZATION.md).

## Pipeline

```text
camera frame -> YOLO11n-pose (COCO 17 keypoints) -> stable80 skeleton features
             -> causal TCN (action head + per-exercise phase heads)
             -> ordered-transition decoder -> complete-cycle counter
```

## How HomeEase uses it

The integrated system ([`backend/workers.py`](../../backend/workers.py)) runs this package in its own worker process through the original FastAPI contract (`homeease.api.create_app`) and feeds it camera frames through `homeease.pose.PoseEstimator`. The standalone SDK / API defaults to `action_scope="open"`, the original decoding of the report. The system's camera and upload sessions use the added `action_scope="training"`, which conditions action support on the four chosen training exercises; for straight-leg raise and hurdle step it also averages the predictions of the original and the left-right mirrored input. Weights, feature functions, phase thresholds and the counter are unchanged, and squat and sit-to-stand still infer the original input only. The state reports the inference mode in `inference_views`. Details: [exercise integration record](../../docs/exercise-integration.md).

## Run it on its own

```bash
cd algorithms/exercise
python -m uvicorn homeease.api:app --host 127.0.0.1 --port 8002 --workers 1
```

API explorer: <http://127.0.0.1:8002/docs>, health check: <http://127.0.0.1:8002/health>. The trained weights are included, so inference works without retraining.

```bash
# offline replay of real skeletons, also checking the HTTP integration
python scripts/replay_demo.py --url http://127.0.0.1:8002
# camera: q quits, p pauses / resumes
python scripts/run_video.py --source 0 --exercise chair_sit_to_stand
# local video without a window
python scripts/run_video.py --source path/to/exercise.mp4 --exercise squat --no-display
```

```python
from homeease.engine import Predictor, ExerciseSession

predictor = Predictor(device="cpu")
session = ExerciseSession(predictor, exercise="squat", target_reps=10)
state = session.update(timestamp_ms=0, keypoints=coco17_points, person_id="person-1")
report = session.control("end")
```

`coco17_points` are 17 `[x, y, confidence]` points with x to the right and y downward, both in the same unit. Live streams are causally resampled to 30 Hz.

## Reproduce the experiments

Create a Python 3.11 environment, install the CPU or CUDA build of PyTorch for your device from the official site, then install `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py --movements m01 m02 m05 m06 m07
python scripts/prepare_data.py
python scripts/train_compare.py --epochs 25 --seed 42
python scripts/evaluate_select.py
python scripts/benchmark_pose.py --images 100
python scripts/optimize_top.py --epochs 80 --seed 2026
python scripts/optimize_top.py --calibrate
python scripts/optimize_top.py --finalize
python scripts/audit_optimized.py
python scripts/robustness.py
python scripts/replay_demo.py
python scripts/build_optimization_report.py
python scripts/build_report.py
```

## Contents

- [DEVELOPMENT.md](docs/DEVELOPMENT.md): requirements, data, labels, models, experiments, selection rationale, limitations and reproduction.
- [OPTIMIZATION.md](docs/OPTIMIZATION.md): optimization of the three strongest candidates, comparison with the original model, the validation selection lock and training logs.
- [RESULTS.md](docs/RESULTS.md): default configuration, first-round vs optimized comparison, per-exercise errors, compute cost and stress tests.
- [API.md](docs/API.md): the complete HTTP / SDK contract, fields, errors and examples.
- [DATA_CARD.md](docs/DATA_CARD.md): data sources, splits, labels and evaluation limits.
- [openapi.json](docs/openapi.json): machine-readable OpenAPI.
- `artifacts/selected.json` and `selected_pose.json`: the default models selected on validation.
- `reports/comparison.csv`, `experiments.json`, `pose_comparison.json`: measured results.
- `examples/*_replay.json`: reconstructed skeleton replays of real test participants for all four exercises.

## Scope of the results

The four-exercise experiments use projections of real UI-PRMD 3D skeletons, with explicitly documented kinematic weak labels for the phases. Count evaluation uses replays built from the original segmented repetitions, not natural continuous home video. The COCO subset compares general pose estimation and does not represent counting accuracy on older adults. Meeting the proposal's <=5% relative error target requires manually labelled phases and counts on continuous camera video and a validation of the full camera chain.
