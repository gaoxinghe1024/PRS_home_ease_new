# Algorithm 1b: Accelerometer Fall Detection

Fall detection from a single three-axis accelerometer. This folder holds the best recognition pipeline **A2** selected by the staged `Fall_Acc_Single` experiments: hand-crafted statistical features classified by a probability average of a Random Forest and XGBoost.

**Reference results** (experiment 3, 3 seeds x stratified 5-fold CV on URFD): sequence-level AUROC **0.9853 +/- 0.0010**, event accuracy 0.826, event recall 0.967.

## Pipeline

```text
three-axis acceleration (raw ~62.5 Hz, irregular timestamps)
  -> linear interpolation to 60 Hz
  -> 4th-order Butterworth low-pass at 5 Hz (zero-phase filtfilt, gravity baseline kept)
  -> 2.0 s windows with a 0.25 s step
  -> 39 statistical features (motion 13 / impact 8 / orient 8 / freq 7 / calm 3)
  -> arithmetic mean of RF(400) and XGB(400) predict_proba = window fall probability p
  -> p >= threshold (default 0.5) marks a hit window
  -> event aggregation: hits at most 2 windows (0.5 s) apart form one event
  -> alarm time = end of the earliest hit window of the event
```

`core/config.py` is the single source of truth for every parameter:

| Stage | Parameters |
| --- | --- |
| Resampling | `RESAMPLE_FS = 60.0` Hz |
| Filter | `FILTER_METHOD = butter5` (4th-order low-pass at 5 Hz, zero phase, gravity kept) |
| Windows | `WIN_S = 2.0` s, `STEP_S = 0.25` s |
| Event merge | `MERGE_GAP_WIN = 2` (0.5 s) |
| Features | 5 groups, 39 dimensions: `motion / impact / orient / freq / calm` |
| RF | `n_estimators=400, max_depth=10, min_samples_leaf=2, class_weight=balanced` |
| XGB | `n_estimators=400, max_depth=5, lr=0.06, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0` |
| Class weight | `scale_pos_weight = 4.9794` (16.72% positive windows in training) |
| Alarm threshold | `ALARM_THRESHOLD = 0.5` (fixed, no calibration) |
| Training seed | `TRAIN_SEED = 2026` |

Trained model: `model/best_model.pkl` (joblib-serialized RF + XGB) and `model/meta.json` (parameters, feature names, reference metrics).

## Why A2 (experiment 3)

| Algorithm | Sequence AUROC | Event accuracy | Note |
| --- | --- | --- | --- |
| A1 threshold rules (no learning) | 0.6496 | 0.067 | weights coupled to thresholds and peaks flattened by filtering; fails badly (0.9825 on raw signals) |
| **A2 features + RF/XGB probability average** | **0.9853** | **0.826** | **deployed** |
| A3 temporal networks (LSTM/GRU/BiLSTM/TCN soft vote) | 0.9411 | 0.777 | 1,160 windows are too few for deep models |
| A4 stacking of 7 base learners (OOF) | 0.9622 | 0.823 | negative ensemble gain: below its strongest base learner |

A2 is also the cheapest to compute (about 23 ms per window), which makes it the one suited to a live stream.

## Two ways to call it

```python
from core import BestFallModel, detect_offline, StreamingDetector
from core.preprocess import read_acc_csv

model = BestFallModel.load("model/best_model.pkl")

# offline: a whole recording (sample data or an uploaded CSV)
t, a = read_acc_csv("fall-01-acc.csv")
events, info = detect_offline(t, a, model)

# streaming: a sensor sending samples continuously
det = StreamingDetector(model)
for t, ax, ay, az in sensor_stream:
    for event in det.push(t, ax, ay, az):
        save(event)
for event in det.flush():
    save(event)
```

Inside HomeEase, [`backend/workers.py`](../../backend/workers.py) (`SignalWorker`) loads exactly these entry points in a separate process; the phone page `/sensor` or any external program streams `[t, ax, ay, az]` batches (in g) to `/ws/sensor`.

## Engineering decisions

- **Zero-phase filtering in real time (fixed 0.25 s delay).** `filtfilt` is non-causal. Each streamed window takes 0.25 s of extra context on both sides and is evaluated 0.25 s after it ends, so streaming results match offline filtering of the whole recording; the cost is a fixed 0.25 s detection delay.
- **Event closing delay of 1.0 s.** An event closes 1 s after its last hit, so the 1 s of data after the alarm needed by the report has arrived. `flush` closes any open event.
- **Single-threaded inference (100 ms -> 23 ms per window).** Streaming evaluates one window at a time, where `n_jobs=-1` only adds thread-pool overhead; predictions are deterministic and bit-identical. Training still uses all cores.
- **Alarm wall time.** The streaming detector records the wall time of the first hit, so stored alarm times are not delayed by the closing delay.
- **Tail windows at the end of a stream.** `flush(complete_tail=True)` evaluates the last windows with whatever context is available, so a fall at the very end of a recording is not missed.

## Consistency with the experiments

| Check | Result |
| --- | --- |
| Training windows / positive ratio / `scale_pos_weight` | 1160 / 16.72% / 4.9794, identical to exp03 |
| Feature names and order | identical |
| Streaming vs offline, all 70 sequences | 0 / 70 sequences with a different event-level result |
| Per-window probability difference | 98.1% within 0.01, 99.7% within 0.05 (differences come from filter boundary reflection in the sliding buffer) |
| fall-01 / fall-10 / fall-20 | identical alarm times offline and streaming (2.98 s / 1.98 s / 1.98 s) |
| adl-01 | no alarm in either mode |

## Training

```bash
cd algorithms/fall_signal
FALL_SYSTEM_DATA_DIR=/path/to/urfd/csv python scripts/train_best_model.py
```

The script trains on all 70 URFD sequences (normal practice for deployment); the generalization figures above come from the cross-validation of the experiment project, not from this script.

## Limits

- Accelerometer only; it is combined with the video branch only at the event level in HomeEase, never fused into a new model.
- The threshold is fixed at 0.5; deployments may tune it to their preference between false alarms and misses.
- All metrics come from folds within URFD; a different device or wearing position needs new data and evaluation.
