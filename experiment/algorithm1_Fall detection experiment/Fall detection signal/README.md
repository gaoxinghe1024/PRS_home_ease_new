# Fall Detection from a Single Accelerometer Channel

Layered comparison of fall-detection algorithms on the **URFD** accelerometer channel.
This repository contains the **experiments only** - no service, no UI, no deployment code.
Everything here is a research script that reads the dataset, fits a model, and writes a JSON.

Four algorithm families are compared under **one fixed evaluation protocol**, so that
every difference in the results can be attributed to the component under test:

| ID | Algorithm | Representation | Decision |
|----|-----------|----------------|----------|
| A1 | Threshold rule | hand-set physical thresholds | learning-free |
| A2 | Features + classical ML | 39 statistical features, 5 groups | RF(400) + XGB(400), probability average |
| A3 | Temporal deep nets | raw `[120, 4]` waveform window | LSTM / GRU / BiLSTM / TCN, soft voting |
| A4 | Hybrid stacking | 3 feature paths + 4 temporal paths | 7 base learners + OOF meta-learner |

---

## 1. Repository layout

```
Fall_Acc_Single_1/
├── README.md                       <- this file (the only Markdown document)
│
├── data/urfd/csv/                  <- read-only dataset, 141 files, ~1.8 MB
│   ├── fall-01..30-acc.csv           raw tri-axial acceleration (t_ms, |a|, ax, ay, az)
│   ├── adl-01..40-acc.csv            activities of daily living (negative class)
│   ├── {seq}-data.csv                frame-alignment table (frame@30fps <-> t_ms)
│   └── urfall-cam0-falls.csv         frame-level fall annotations
│
├── 00_common/                      <- shared upstream: every experiment uses these
│   ├── paths.py                      dataset path registry (read-only contract)
│   ├── acc_loader.py                 CSV loading, frame<->second alignment, fall intervals
│   ├── preprocess.py                 resampling + 7 filter candidates (gravity-preserving)
│   ├── features.py                   39 features in 5 groups + 6 selection methods
│   ├── windowize.py                  2.0 s window / 0.25 s stride + window labelling
│   ├── dataset.py                    disk + memory cache of the windowed tensors
│   ├── splits.py                     evaluation pool (70 sequences) and K-fold splits
│   ├── metrics.py                    event aggregation, AUROC, three-level metrics
│   ├── evaluate.py                   unified K-fold / repeated-K-fold evaluator
│   ├── corrupt.py                    sensor degradation: noise / packet loss / block loss
│   └── cache/                        windowed-tensor cache (auto-rebuilt, see §6)
│
├── 03_alg_rule/src/rule_algorithm.py       A1: impact / activity / post-impact rest rule
├── 04_alg_ml/src/ml_algorithm.py           A2: features + RF/XGB
├── 05_alg_temporal/src/temporal_algorithm.py  A3: LSTM / GRU / BiLSTM / TCN
├── 06_alg_hybrid/src/hybrid_algorithm.py   A4: 7-learner stacking with OOF meta-features
│
├── experiments/                    <- the five numbered experiments
│   ├── exp01_preproc.py              Exp.1  filter + resampling comparison
│   ├── exp02_features.py             Exp.2  feature groups + feature selection
│   ├── exp03_alg_compare.py          Exp.3  A1-A4 comparison (primary result)
│   ├── exp04_robustness.py           Exp.4  degradation robustness
│   ├── exp05_preproc_sensitivity.py  Exp.5  preprocessing x algorithm coupling
│   ├── _explore_acc.py               dataset exploration (one-off, informs the design)
│   ├── _diag_a4.py                   fusion-strategy diagnostic for A4 (scratch)
│   ├── _diag_noise_feat.py           per-feature AUROC decay under noise
│   └── _summary.py                   prints the result JSONs as Markdown tables
│
└── results/                        <- created by the experiments (one JSON per experiment)
```

The tree above is the complete repository - there are no other top-level directories and no
code outside it. Two conventions are worth explaining: the preprocessing and windowing stages
live in `00_common/` instead of their own `01_preproc/` and `02_features/` folders, because
all four algorithms share them and duplicating the code per algorithm would have made the
"single variable" comparison meaningless; and no separate reporting package exists, because
`experiments/_summary.py` already turns every result JSON into a table.

---

## 2. Environment

Developed and verified on Windows with Python 3.12.

| Package | Version used | Purpose |
|---|---|---|
| numpy | 1.26.4 | all array computation |
| scipy | 1.16.0 | Butterworth / median / Savitzky-Golay filters |
| scikit-learn | 1.5.1 | RF, LR, RFE, feature selection, GroupKFold |
| xgboost | 2.1.1 | gradient-boosted trees (A2, A4 meta-learner) |
| torch | 2.5.1 | A3 and the temporal base learners of A4 |
| pandas | 2.x | CSV loading in `acc_loader` |
| joblib | 1.x | fold-level multiprocessing, model serialisation |

Only the CPU is required and no web framework is needed anywhere in this repository.
Nothing needs to be installed if the packages above are already present.

```bash
# every command below is run from the project root
cd Fall_Acc_Single_1
```

---

## 3. Dataset and label convention

`data/urfd/csv/` holds 70 URFD sequences: **30 fall (`fall-01..30`)** and
**40 activities of daily living (`adl-01..40`)**. The data is treated as read-only;
nothing in this project modifies it.

Acceleration CSV columns are `t_ms, |a|, ax, ay, az` (units **g**). Sampling is irregular
(nominal ~16 ms, jitter 5-37 ms), with a median effective rate of **62.5 Hz**. `|a|` is
recomputed from the three axes rather than trusted from the file.

Frame alignment: `{seq}-data.csv` maps `frame@30fps <-> t_ms` and both timelines share the
same origin, so `t = frame / 30`.

Fall interval (identical convention for every algorithm):
`[first frame with label=0, first frame with label=1 + 30 frames]`, clipped to the
accelerometer recording range. A 2 s window is positive iff it overlaps any fall interval
("any-hit" semantics). ADL sequences contain no fall interval and therefore no positive window.

---

## 4. Reproduction

Run the steps in order. Each experiment writes one JSON into `results/` and
prints the same table to stdout.

### Step 1 - dataset sanity check (seconds)

```bash
python experiments/_explore_acc.py
```

Reports sampling interval statistics, sequence durations, sample counts, and the
separability of fall vs ADL by a single peak threshold. Purely descriptive; no model.

### Step 2 - Experiment 1: filtering and resampling (~1.5 min)

```bash
python experiments/exp01_preproc.py --jobs 5
```

Runs 4 resampling strategies and 7 filter methods, each with the same fixed downstream
model (A2), and additionally computes three classifier-independent signal metrics:
peak retention, quiet-noise ratio, and phase lag.

- `--jobs N` parallelises the 5 folds; `N=5` is a good default.

### Step 3 - Experiment 2: feature groups and feature selection (~3 min)

```bash
python experiments/exp02_features.py --jobs 5
```

Reads the Exp.1 result to pick the best preprocessing configuration automatically
(override with `--resample 60 --filt butter10`). Runs the 5 single feature groups,
all 10 pairwise combinations, the full set, 6 selection methods, then a descriptive
single-feature AUROC ranking and a full-model importance table.

### Step 4 - Experiment 3: A1-A4 comparison (~30 s for A1+A2, ~3 min at 3 epochs, ~40-60 min at 20 epochs)

```bash
# fast subset: rule and feature models only
python experiments/exp03_alg_compare.py --algs a1,a2 --seeds 2026,2027,2028 --jobs 5

# full protocol (3 seeds x stratified 5-fold, all four algorithms)
python experiments/exp03_alg_compare.py --algs a1,a2,a3,a4 --seeds 2026,2027,2028 \
        --epochs 20 --jobs 5
```

`--skip-extra` drops the three internal analyses (A1 sub-scores, A3 per-architecture,
A4 base-learner/fuser breakdown) when only the headline table is needed.

**Careful: Exp.3 has no `--out` flag and always writes `results/exp03_alg.json`.** Running a
subset therefore overwrites the previous subset's file. Copy the JSON aside before running a
second subset, or the summary will only show the last one. (Exp.4 does have `--out`, which is
why the A3/A4 degradation run above can live side by side with the A1/A2 one.)

### Step 5 - Experiment 4: degradation robustness (~1 min for A1+A2, ~15 min with A3+A4)

```bash
python experiments/exp04_robustness.py --algs a1,a2 --jobs 8
python experiments/exp04_robustness.py --algs a3,a4 --jobs 4 --out exp04_robustness_deep.json
```

A1/A2 cover all 8 conditions; A3/A4 cover the 4 key conditions. Degradation is injected
into the **test set only**; the training set stays clean.

### Step 6 - Experiment 5: preprocessing sensitivity (~6 min)

```bash
python experiments/exp05_preproc_sensitivity.py --jobs 4 --a3-epochs 3
```

Places A1, A2 and (on three configurations) A3 on the same set of preprocessing
configurations to show that preprocessing and algorithm are **not** separable stages.
Drop `--a3-epochs` for the default epoch count; 3 keeps the sweep short.

### Step 7 - result tables

```bash
python experiments/_summary.py
```

Prints all five experiments as Markdown tables (Exp.1, Exp.2 groups/selection/top-15,
Exp.3, Exp.4 per algorithm x condition, Exp.5 as two config x algorithm matrices).
Missing result files are skipped rather than raising.

That is the full reproduction. `results/` is created automatically on the first run.

---

## 5. Expected results

Reference values reproduced by the protocol above. `AUROC` is sequence-level
(score = maximum window probability in the sequence) and is the primary metric.

| Algorithm | Sequence AUROC | Window Recall | Event Recall | Event Accuracy |
|---|---|---|---|---|
| A1 threshold rule | 0.6496 | 0.026 | 0.067 | 0.067 |
| A2 features + ML | 0.9853 +- 0.0010 | 0.817 | 0.967 | 0.826 |
| A3 temporal nets | 0.9225 | 0.808 | 0.933 | 0.737 |
| A4 hybrid stacking | 0.9225 | 0.817 | 0.967 | 0.808 |

Measured on this machine. A1/A2 are 3-seed stratified 5-fold means and are reproducible;
A3/A4 rows are a **single seed with `--epochs 3`** used to confirm the deep path runs, so
they are a lower bound, not the converged figure - re-run with `--epochs 20 --seeds 2026,2027,2028`
for the real numbers. Everything is evaluated on the preprocessing configuration that Exp.1
selects automatically, which here is `resample 60 Hz + butter5`.

**The A1 number is not a property of A1 - it is a property of A1 paired with `butter5`.**
Exp.5 sweeps the filter and shows A1 collapsing monotonically as the cutoff drops:

| config | raw | none | butter20 | butter10 | butter5 | butter3 |
|---|---|---|---|---|---|---|
| A1 seq AUROC | 0.9825 | 0.9483 | 0.9050 | 0.7788 | 0.6496 | 0.6071 |
| A1 event recall | 0.847 | 0.700 | 0.633 | 0.500 | 0.067 | 0.000 |
| A2 seq AUROC | 0.9922 | 0.9767 | 0.9642 | 0.9642 | 0.9867 | 0.9450 |

The mechanism is measurable in Exp.1's `peak_retention` column: `butter5` keeps only 0.41 of
the raw impact peak, and A1's decision rule is an absolute-g threshold on exactly that peak.
Exp.1 picks `butter5` because it maximises A2's sequence AUROC - so the pipeline's automatic
configuration choice is *hostile* to A1. Read the headline table as "A2 beats A1 under the
configuration A2 prefers", not as "A1 is a weak detector": on the raw signal A1 reaches 0.98.
This is the point Exp.5 exists to make - preprocessing and algorithm are not separable stages,
and a single-configuration leaderboard can invert a conclusion.

Acceptance checklist:

- every experiment exits with code 0 and writes its JSON into `results/`;
- `python experiments/_summary.py` prints all five experiments with no `KeyError`;
- re-running `exp03_alg_compare.py --algs a1 --seeds 2026` twice gives a bit-identical
  `seq_auroc` (the protocol is deterministic for a fixed seed; A1 is learning-free, so all
  three seeds give exactly 0.6496);
- `results/exp05_sensitivity.json` reproduces the A1 monotone decay in the table above,
  which is the cheapest single check that the whole upstream chain is intact.

---

## 6. Cache and generated artefacts

`00_common/dataset.py` caches the windowed tensors in `00_common/cache/`, one `.npz` per
preprocessing configuration, keyed by an MD5 fingerprint of the config. Running Exp.1
therefore creates several cache files.

- **The cache is not shipped.** It is purely derived data and is rebuilt automatically on
  demand - measured at about 2.7 s and 2 MB per configuration on this machine, so Exp.1
  (11 configurations) spends roughly 30 s rebuilding before the modelling starts.
- Because the key is the config fingerprint, changing a configuration never overwrites
  another configuration's cache, which is exactly what makes the single-variable
  comparisons valid.
- The cache is therefore also safe to delete at any time, for example to shrink the tree
  before archiving it; a full run only pays the rebuild cost once.
- `results/` is created and filled by the experiments; `00_common/cache/` is the only other
  generated directory.

---

## 7. Design constraints worth knowing

**Every candidate filter preserves gravity.** High-pass and band-pass are excluded by
design: the absolute magnitude is the physical basis of both the impact-peak criterion
and the free-fall (`|a| -> 0`) criterion. Removing the 1 g baseline destroys the meaning
of "3-10 g impact" and `|a| < 0.6 g weightless`, which is precisely A1's decision rule.

**All filters are zero-phase (non-causal).** This introduces no lag for offline analysis,
but it **cannot be used on a live stream as-is**: a real-time version would have to take
an extra 0.25 s of context on each side of a window and delay evaluation by 0.25 s,
accepting a fixed 0.25 s detection latency in exchange for near-identical results. This
trade-off is noted here because it is the reason the reported latencies are offline
latencies, not deployment latencies.

**Feature selection is fitted inside the training fold only**, and A4's meta-features are
generated out-of-fold with `GroupKFold` **grouped by sequence**, so windows of one sequence
never appear on both sides of an estimate.

**Results of different algorithms are never pooled** across folds or conditions;
`exp04` buckets by `(algorithm, condition)` for this reason.

---

## 8. Known limitations

- URFD publishes no subject identifiers, so a subject-independent **LOSO** split cannot be
  constructed.
- The sensor path has a single dataset, so cross-dataset **LODO** generalisation cannot be
  tested. Multi-seed repeated K-fold plus degradation robustness stand in for it, and this
  substitution is stated explicitly rather than papered over.
- Degradation is applied on the window tensors, not by re-simulating a sensor front end;
  it models signal-level corruption (thermal noise, packet loss, block loss), not a
  specific hardware profile.
- A3/A4 are CPU-only and slow; the full 3-seed protocol with all four algorithms is a
  long run. Use `--algs a1,a2` or `--skip-extra` while iterating.

---

## 9. Timings

End-to-end measurements on this machine (CPU only, `--jobs` as shown in §4):

| step | measured |
|---|---|
| Step 1 dataset exploration | 3 s |
| Step 2 Exp.1 | 1 min 30 s |
| Step 3 Exp.2 | 2 min 44 s |
| Step 4 Exp.3, A1+A2, 3 seeds | 32 s (A1 6 s, A2 22 s) |
| Step 4 Exp.3, A3+A4, 1 seed, 3 epochs | 2 min 3 s (A3 24 s, A4 94 s) |
| Step 5 Exp.4, A1+A2, 8 conditions | 56 s |
| Step 6 Exp.5, 7 configs, A3 at 3 epochs | 6 min 0 s |
| Step 7 summary | under 1 s |

All timings are with the tensor cache warm. On a cold start (no `00_common/cache/`) add
about 2.7 s per distinct preprocessing configuration - Exp.1 touches 11 configurations,
so budget an extra ~30 s there; the other experiments reuse configs created by Exp.1.

Only the A3/A4 rows at the full 20-epoch, 3-seed protocol are estimates scaled from the
per-fold cost rather than measurements; everything else above was timed.
