# Algorithm 1a: Video Fall Detection

Fall detection from a single RGB camera with a three-stage pipeline: person detection, our own pose network and a temporal ensemble. The weights in this folder are used as trained; the integrated system does not retrain or modify them.

## Pipeline

```text
camera / video -> 1. YOLOv8n person detection -> 2. P1 pose CNN (17 keypoints) -> 3. A4 hybrid ensemble -> P(fall)
                                                                                              |
                                                      P(fall) above the threshold, confirmed repeatedly
                                                                                              v
                                          fall event: system time + 3 s clip (1 s before, 2 s after) + alarm flag
```

| Stage | Weights | Source |
|---|---|---|
| 1. Person detection | `standalone/yolov8n-pose.onnx` | pre-trained YOLOv8n, box head only |
| 2. Pose estimation | `standalone/pose_best.pt` | our small P1 U-Net, trained from scratch (PCK@0.2 = 43.1%) |
| 3. Temporal decision | `a4_weights.pkl` (falls back to `standalone/a4_weights.pkl`) | A4 hybrid ensemble: 7 base learners, OOF stacking + XGBoost meta-learner |

The A4 window is 30 frames (2 s at 15 fps). Default decision threshold 0.60, with consecutive confirmation and a cool-down so one fall is recorded once.

## Files

| File | Purpose |
|---|---|
| `engine.py` | `DetectionEngine`: background thread with per-frame inference, 3 s clips and record storage |
| `store.py` | SQLite persistence of fall records (`records/falls.db`) |
| `standalone/main.py` | self-contained end-to-end pipeline with visualization (`python main.py --demo urfd fall-01`) |
| `standalone/a4_core.py` | A4 core: sliding windows, kinematic features, temporal network, ensemble fit / predict |
| `standalone/train_a4.py` | trains A4 on the unified pool (URFD + MCF + Le2i, 223 sequences) |
| `train_weights.py` | re-exports the A4 weights in the current environment when sklearn / xgboost versions differ |

## How HomeEase uses it

[`backend/workers.py`](../../backend/workers.py) (`VideoWorker`) mirrors `engine.py`, `store.py` and the weights into the runtime folder byte for byte and loads the original `DetectionEngine` in its own process. Two adaptations sit outside the algorithm:

- **Shared camera.** The system opens each camera once and hands raw BGR frames to both this engine and the exercise algorithm through shared memory (`backend/camera.py`), replacing only the engine's camera `FrameSource`.
- **Optional acceleration.** With `HOMEEASE_VIDEO_DNN=auto` the same YOLO ONNX file runs on CUDA through onnxruntime when available; otherwise the original OpenCV DNN path is used. On machines without CUDA, the CUDA-exported A4 tensors are mapped to the CPU. Weights, thresholds and decisions are unchanged.

The engine's overlays (boxes and skeleton) are disabled; the workbench draws them itself from the engine's keypoints, so the model always sees the raw frame.

## Performance notes

YOLO runs on the CPU (~50-70 ms), pose on the GPU (~12 ms), and A4 originally needed ~90 ms. Three measures bring 640x480 input to about 20 fps, above the 15 fps sampling rate, without changing decision quality (peak probability 0.996-0.999 on the same sequence):

| Measure | Default | Effect |
|---|---|---|
| `detect_every` | 3 | YOLO every 3 frames, reusing the last box in between |
| `prob_every` | 2 | A4 every 2 frames, keeping the last probability in between |
| no joblib / OpenMP scheduling for A4 | built in | A4 ~90 ms -> ~30 ms |

Measured on an RTX 3050 Ti: per-frame full inference 9.3 fps -> default configuration 19.3 fps. Clips are collected by wall-clock time, so they always cover the real 3 seconds regardless of inference speed.
