# Video-Based Fall Detection — Minimal Experiment Package

## 1. Directory layout

```
Fall_Video_Single_Minimal/
├── main.py                      # end-to-end demo: detect → pose → A4 → visualization
├── run_main_pipeline.py         # full pipeline driver (arch selection → skeletons → A1-A4)
│
├── 00_common/                   # shared layer: paths, loaders, metrics, sliding windows
│   ├── paths.py                 # dataset path registry (the only place paths are defined)
│   ├── urfd_loader.py           # URFD reader (frames + frame-level labels)
│   ├── mc_loader.py             # Multiple Cameras Fall reader (single channel: cam1)
│   ├── le2i_loader.py           # Le2i / ImViA reader (4 labelled scenes)
│   ├── splits.py                # unified pool, grouped 5-fold CV, leave-one-dataset-out
│   ├── windowize.py             # 2 s window / 0.5 s step → window samples
│   ├── metrics.py               # window / sequence / event metrics, threshold calibration
│   ├── pose_extract.py          # stage 2: detection cache → skeleton cache
│   ├── models/yolov8n-pose.onnx # stage-1 detector weights
│   └── cache/                   # NOT SHIPPED — the whole directory is created at run time
│       ├── detections/          # 223 .npz, written by 01_detect_yolo (2.3)
│       ├── skeletons/           # 223 .npz, written by pose_extract.py (2.3)
│       └── kaggle/              # optional Le2i + MCF copy, absent by default (2.2)
│
├── 01_detect_yolo/
│   └── yolo_person.py           # stage 1: ONNX inference, writes cache/detections
│
├── 02_pose_cnn/                 # stage 2: pose network
│   ├── dataset_coco.py          # COCO keypoints dataset, affine crop, heatmap targets
│   ├── models.py                # P1 SmallUNet / P2 CPM3Stage / P3 SimpleBaseline
│   ├── train.py                 # single-architecture training
│   ├── eval_pose.py             # PCK evaluation
│   ├── export_infer.py          # export weights, render demo predictions
│   ├── exp_arch_compare/
│   │   ├── run_compare.py       # architecture comparison driver
│   │   └── comparison.json      # recorded comparison results
│   └── outputs/                 # trained weights: best.pt, p1/, p2/, p3/
│
├── 03_alg_rule/src/             # A1  geometric rule baseline (no training)
├── 04_alg_ml/src/               # A2  rule + kinematic features (gradient boosting)
├── 05_alg_temporal/src/         # A3  temporal CNN over skeleton windows
├── 06_alg_hybrid/src/           # A4  stacked ensemble (7 base learners + XGBoost meta)
│
├── experiments/
│   ├── exp01_alg_compare.py     # A1/A2/A3/A4 comparison under the shared protocol
│   └── exp01_results.json       # recorded metrics (5-fold CV + LODO)
│
├── standalone/                  # self-contained demo (no 00_common dependency)
│   ├── main.py                  # detect + pose + A4 in one file
│   ├── a4_core.py               # model/ensemble definitions (required by a4_weights.pkl)
│   ├── train_a4.py              # retrain A4 from the skeleton cache
│   └── *.pt / *.pkl / *.onnx    # runtime weights
│
└── data/
    ├── urfd/
    │   ├── frames/              # 70 sequences (30 fall + 40 ADL), PNG frames
    │   └── csv/                 # frame-level fall annotations
    └── coco/
        ├── val2017/             # 5,000 images
        └── annotations/
            └── person_keypoints_val2017.json
```

---

## 2. How to reproduce

All commands are run from the package root unless stated otherwise.

### 2.1 Environment

Tested on Python 3.12.

```bash
pip install numpy pandas opencv-python torch scikit-learn xgboost joblib
```

```bash
python -c "import numpy, cv2, torch, sklearn, xgboost, pandas; print('ok')"
```

**CUDA is optional.** Every model-loading site uses
`torch.device("cuda" if torch.cuda.is_available() else "cpu")` and weights are read with
`map_location="cpu"`. Without a GPU everything still runs, only slower.

No path is hard-coded to an absolute location, so the folder can be moved or renamed as
long as the internal layout is preserved.

### 2.2 Verify the datasets

```bash
cd 00_common
python -c "import urfd_loader, mc_loader, le2i_loader as l; \
print(len(urfd_loader.list_sequences()[0]), len(mc_loader.list_sequences()), len(l.list_videos()))"
```

Expected output: `30 23 130`.

URFD is read from `data/urfd/` inside this folder. Le2i and MCF are resolved by
`_find_kaggle_dir()`, which checks `~/.cache/kagglehub/datasets` first and the local
`00_common/cache/kaggle` second, so a locally bundled copy is optional when kagglehub
already holds both datasets. To make the folder self-contained, materialise the local copy:

```bash
python 00_common/download_kaggle.py
```

If either dataset is missing from both locations, the loaders raise
`FileNotFoundError: ... not downloaded: run download_kaggle.py`. That script needs the
`kagglehub` package and Kaggle credentials (`pip install kagglehub`, then place
`kaggle.json` under `~/.kaggle/`). It links the download into `00_common/cache/kaggle`,
creating the directory if needed.

### 2.3 Rebuild the feature caches 

The detection and skeleton caches are not shipped. Rebuild them once. Stage 2 consumes
the output of stage 1, so the order matters:

```bash
python 01_detect_yolo/yolo_person.py urfd
python 01_detect_yolo/yolo_person.py mc
python 01_detect_yolo/yolo_person.py le2i

python 00_common/pose_extract.py urfd
python 00_common/pose_extract.py mc
python 00_common/pose_extract.py le2i
```

Both scripts skip sequences that are already cached, so an interrupted run resumes by
repeating the same command. Expected result: 223 `.npz` under `cache/detections/` and 223
under `cache/skeletons/`. Stage 1 uses `00_common/models/yolov8n-pose.onnx`, stage 2 uses
`02_pose_cnn/outputs/best.pt`, so no training is needed beforehand.

Full rebuild takes about 3 hours on a machine with CUDA (0.27 s/frame; stage 1 runs on CPU
through OpenCV DNN). Use `--limit N` to process only the first N sequences.

Skipping this step makes `pose_extract.py` raise `FileNotFoundError` and ask for the
detection batch.

### 2.4 A1/A2/A3/A4 comparison

```bash
python experiments/exp01_alg_compare.py
```

Runs all four algorithms. To run a single one, pass `--alg`, for example `--alg a4`.

Reads the skeleton cache built in 2.3; stage 1–2 inference is not re-run. Writes
`experiments/exp01_results.json`.

### 2.5 Pose architecture comparison

```bash
cd 02_pose_cnn/exp_arch_compare
python run_compare.py --mode val_split
```

Trains P1/P2/P3 under identical data, hyperparameters and seed, then writes
`comparison.json` and `results.md`. Weights land in `02_pose_cnn/outputs/{p1,p2,p3}/`.
Roughly 20–30 minutes per architecture on a laptop GPU.

### 2.6 Full pipeline

```bash
python run_main_pipeline.py
```

Selects the best pose architecture from `comparison.json`, regenerates skeletons unless
`--skip-pose` is given, then runs the algorithm comparison.

```bash
python run_main_pipeline.py --skip-pose --arch p1
```

### 2.7 End-to-end demo

```bash
python main.py --demo urfd fall-01
python main.py --demo le2i Coffee_room_01_video1 --headless --save out.mp4
python main.py --list
```

Or use the self-contained version, which needs no `00_common` imports:

```bash
cd standalone
python main.py --demo urfd fall-01
```

### 2.8 Weights in use

| Stage | File | Role |
|---|---|---|
| 1 | `00_common/models/yolov8n-pose.onnx` | person detector (ONNX, box head only) |
| 2 | `02_pose_cnn/outputs/best.pt` | final pose model (P1 SmallUNet) |
| 3 | `06_alg_hybrid/`, `05_alg_temporal/` | trained inside 2.4, not shipped |

`02_pose_cnn/outputs/{p1,p2,p3}/best.pt` are the three architectures compared in 2.5.
`standalone/` carries its own copies (`pose_best.pt`, `yolov8n-pose.onnx`,
`a4_weights.pkl`) so it can run without `00_common`.

`a4_weights.pkl` is a pickle of custom classes defined in `standalone/a4_core.py`. Loading
it requires that file to be importable and is sensitive to the scikit-learn / xgboost
versions used at save time. If it fails to unpickle, rebuild it:

```bash
cd standalone
python train_a4.py
```

Reproducibility notes:

- Splits use `seed=2026`; training uses the same seed unless overridden on the command line.
- `_find_kaggle_dir()` searches `~/.cache/kagglehub/datasets` **before** the bundled
  `00_common/cache/kaggle`, so an older MCF/Le2i copy in the user-level cache takes
  precedence over the bundled one. Confirm which copy is active with

  ```bash
  cd 00_common
  python -c "import paths; print(paths.mc_root()); print(paths.le2i_root())"
  ```

  Either location is acceptable as long as 2.2 still prints `30 23 130`.
- Paths are derived from `__file__`, so the folder can be moved or renamed freely as long
  as the internal layout is preserved.
