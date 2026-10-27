# -*- coding: utf-8 -*-
"""The **single source of truth** for the hyper-parameters of the best combination.

This file fixes the **best recognition pipeline** selected by the five staged `Fall_Acc_Single` experiments (A2 =
hand-crafted features + RF/XGB probability averaging); the evidence is in section 7 of the research report:

  input = three-axis acceleration (~62.5 Hz, irregular)
       -> resample to 60 Hz
       -> 4th-order Butterworth low-pass at 5 Hz (zero phase, gravity kept)
       -> 2.0 s / 0.25 s sliding windows
       -> 39 statistical features (motion / impact / orient / freq / calm)
       -> RF(400) + XGB(400) probability average
       -> event aggregation (hits at most 2 windows apart are merged; alarm time = end of the earliest hit window)

Experiment 3 (3 seeds x stratified 5-fold): sequence-level AUROC 0.9853 +/- 0.0010,
event accuracy 0.826 - the best overall among the A1 rules / A3 temporal network / A4 stacking.
"""

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
RESAMPLE_FS = 60.0          # resampling target rate (Hz)
FILTER_METHOD = "butter5"   # 4th-order Butterworth low-pass at 5 Hz, zero phase (gravity kept)
BUTTER_ORDER = 4
BUTTER_CUTOFF = 5.0         # Hz
RAW_FS_NOMINAL = 62.5       # nominal URFD sampling rate (reference only)

# --------------------------------------------------------------------------
# Sliding windows and event aggregation
# --------------------------------------------------------------------------
WIN_S = 2.0                 # window length (s)
STEP_S = 0.25               # step (s)
MERGE_GAP_WIN = 2           # hits at most 2 windows (0.5 s) apart belong to the same fall event

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
FEATURE_GROUPS = ("motion", "impact", "orient", "freq", "calm")
N_FEATURES = 39             # 13 + 8 + 8 + 7 + 3

# --------------------------------------------------------------------------
# Classifier (identical to 04_alg_ml/src/ml_algorithm.py)
# --------------------------------------------------------------------------
RF_PARAMS = dict(n_estimators=400, max_depth=10, min_samples_leaf=2,
                 class_weight="balanced")
XGB_PARAMS = dict(n_estimators=400, max_depth=5, learning_rate=0.06,
                  subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0)
TRAIN_SEED = 2026           # seed used to reproduce the experiment report

# --------------------------------------------------------------------------
# Inference / alarm
# --------------------------------------------------------------------------
ALARM_THRESHOLD = 0.5       # window alarm threshold (main event-level setting = fixed 0.5, no calibration)

# Inference threads: streaming detection evaluates only 1 window at a time, so multithreading is pure scheduling overhead
# (measured: n_jobs=-1 about 100 ms per window -> 1 thread about 23 ms, with bit-identical predictions;
#   of which RF (400 trees) about 22 ms, XGB (400 trees) about 0.4 ms, feature extraction about 0.5 ms).
# Training still uses all cores (n_jobs=-1 of RF/XGB applies during fit).
INFER_JOBS = 1

# --------------------------------------------------------------------------
# Report data slice
# --------------------------------------------------------------------------
REPORT_PRE_S = 2.0          # keep 2.0 s before the alarm
REPORT_POST_S = 1.0         # keep 1.0 s after the alarm (to show the post-impact rest phase)
REPORT_SNIPPET_S = REPORT_PRE_S + REPORT_POST_S   # 3.0 s acceleration curve in total

# Length of the streaming sliding buffer (must exceed the window length to give the zero-phase filter context)
STREAM_BUFFER_S = 6.0

# Context / causal delay of the zero-phase filter (seconds).
# The best combination uses filtfilt (non-causal) so that offline analysis has no phase lag; when streaming, a window is
# evaluated only 0.25 s after its end and takes 0.25 s of context on both sides, so the window content is almost identical
# to filtering the whole recording. The system therefore pays a fixed 0.25 s detection delay (noted as the deployment cost in the report).
FILTER_PAD_S = 0.25

# Gravity baseline and free-fall threshold (physical constants shared by features and rules)
G = 1.0
FREEFALL_TH = 0.6
