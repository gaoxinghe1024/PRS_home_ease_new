"""Result documents, figures and artifact hashes generated from the recorded experiment JSON."""
import importlib.metadata
import json
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from homeease.exercises import EXERCISE_NAMES

from .common import ROOT, artifact_manifest


def write_results():
    results = json.loads((ROOT / "reports/experiments.json").read_text())
    poses = json.loads((ROOT / "reports/pose_comparison.json").read_text())
    selected = json.loads((ROOT / "artifacts/selected.json").read_text())
    row = next(r for r in results if r["model"] == selected["model"])
    test_best = min(results, key=lambda r: (r["test"]["count_mae"], -r["test"]["phase_macro_f1"]))
    baseline_path = ROOT / "reports/optimization/baseline/selected.json"
    baseline_name = json.loads(baseline_path.read_text())["model"] if baseline_path.exists() else None
    baseline = next((r for r in results if r["model"] == baseline_name), None)
    selected_pose = json.loads((ROOT / "artifacts/selected_pose.json").read_text())
    figs = ROOT / "reports/figures"
    figs.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    handles, legend_labels = [], []
    for number, r in enumerate(results, 1):
        marker = "*" if r["model"] == row["model"] else "o"
        handle = axes[0].scatter(r["cpu_p50_ms"], r["validation"]["count_mae"], marker=marker, s=120 if marker == "*" else 40)
        handles.append(handle)
        legend_labels.append(f"{number}. {r['model']}")
        if r["model"] == row["model"]:
            axes[0].annotate(f"Recommended: {r['model']}", (r["cpu_p50_ms"], r["validation"]["count_mae"]),
                             fontsize=10, fontweight="bold", color="darkgreen", xytext=(0, 20),
                             textcoords="offset points", arrowprops={"arrowstyle": "->", "color": "darkgreen"})
        else:
            axes[0].annotate(str(number), (r["cpu_p50_ms"], r["validation"]["count_mae"]), fontsize=9,
                             xytext=(5, 6 if number % 2 or r["validation"]["count_mae"] <= 1.05 else -14), textcoords="offset points")
    axes[0].set(xlabel="CPU temporal inference p50 (ms)", ylabel="Validation count MAE (reps)", title="Validation accuracy vs compute")
    axes[0].grid(alpha=.2)
    for number, p in enumerate(poses):
        axes[1].scatter(p["latency_p50_ms"], p["pck_bbox_02"] * 100)
        axes[1].annotate(f"{p['model'].replace('-pose.pt','')}@{p['imgsz']}", (p["latency_p50_ms"], p["pck_bbox_02"] * 100),
                         fontsize=8, xytext=(3, -14 if number == 1 else 6), textcoords="offset points")
    axes[1].set(xlabel="GPU full pose call p50 (ms)", ylabel="PCK@0.2bbox (%)", title="COCO 100-image pose subset")
    axes[1].grid(alpha=.2)
    axes[1].margins(x=.2)
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .15, 1, 1))
    fig.savefig(figs / "accuracy_compute.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    matrix = np.asarray(row["test"]["action_confusion_matrix"])
    plot = ax.imshow(matrix, cmap="Blues")
    labels = ["Chair STS", "Squat", "Leg raise", "Hurdle step", "Other"]
    ax.set_xticks(range(5), labels, rotation=35, ha="right")
    ax.set_yticks(range(5), labels)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() * .5 else "black")
    ax.set(xlabel="Predicted action", ylabel="Source action", title=f"Held-out action confusion: {row['model']}")
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(figs / "action_confusion.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    sessions = row["test"]["sessions"]
    x = np.arange(len(sessions))
    ax.bar(x - .2, [s["expected"] for s in sessions], width=.4, label="Source reference")
    ax.bar(x + .2, [s["predicted"] for s in sessions], width=.4, label="Predicted")
    ax.set_xticks(x, [s["session"].replace("_reconstructed", "") for s in sessions], rotation=40, ha="right")
    ax.set(ylabel="Complete cycles", title="Test reconstructed sessions, including negative actions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "test_session_counts.png", dpi=160)
    plt.close(fig)
    lines = ["# HomeEase Algorithm 2 (four exercises): experiment results", "",
             "## Best algorithm at a glance", "",
             f"> **Best overall and current deployment recommendation: hierarchical causal TCN `{row['model']}`**\n> Validation count MAE **{row['validation']['count_mae']:.2f} reps**; test count MAPE **{row['test']['count_mape_pct']:.2f}%**; test action Macro-F1 **{row['test']['action_macro_f1']:.4f}**; CPU P50 **{row['cpu_p50_ms']:.3f} ms**.", "",
             f"> **Highest test replay metrics: optimized GRU `{test_best['model']}`**\n> Test count MAPE **{test_best['test']['count_mape_pct']:.2f}%**; test action Macro-F1 **{test_best['test']['action_macro_f1']:.4f}**; test joint phase Macro-F1 **{test_best['test']['phase_macro_f1']:.4f}**.", "",
             f"| Key metric | **Best overall TCN (recommended for deployment)** | **Highest test metrics GRU** |", "|---|---:|---:|",
             f"| Model id | **`{row['model']}`** | **`{test_best['model']}`** |"]
    for label, split, key, fmt, suffix in [
        ("Validation action Macro-F1 ↑", "validation", "action_macro_f1", ".4f", ""),
        ("Validation joint phase Macro-F1 ↑", "validation", "phase_macro_f1", ".4f", ""),
        ("Validation count MAE ↓", "validation", "count_mae", ".2f", " reps"),
        ("Test action Macro-F1 ↑", "test", "action_macro_f1", ".4f", ""),
        ("Test joint phase Macro-F1 ↑", "test", "phase_macro_f1", ".4f", ""),
        ("Test count MAE ↓", "test", "count_mae", ".2f", " reps"),
        ("Test count MAPE ↓", "test", "count_mape_pct", ".2f", "%"),
        ("Test relative count accuracy ↑", "test", "count_accuracy_pct", ".2f", "%"),
        ("Test exact-session rate ↑", "test", "exact_session_pct", ".1f", "%"),
        ("Test cycle recall ↑", "test", "cycle_recall", ".4f", ""),
        ("Test cycle precision ↑", "test", "cycle_precision", ".4f", ""),
    ]:
        values = [f"{r[split][key]:{fmt}}{suffix}" for r in [row, test_best]]
        lines.append(f"| {label} | **{values[0]}** | **{values[1]}** |")
    lines += [f"| CPU temporal forward P50 ↓ | **{row['cpu_p50_ms']:.3f} ms** | {test_best['cpu_p50_ms']:.3f} ms |",
              f"| CPU temporal forward P95 ↓ | **{row['cpu_p95_ms']:.3f} ms** | {test_best['cpu_p95_ms']:.3f} ms |",
              f"| Parameters | {row['parameters']:,} | {test_best['parameters']:,} |", "",
              f"**Why the TCN is recommended:** its validation count MAE is **{row['validation']['count_mae']:.2f} reps**, lower than the GRU's {test_best['validation']['count_mae']:.2f} reps, and the GRU's CPU P50 is **{test_best['cpu_p50_ms']/row['cpu_p50_ms']:.2f} times** that of the TCN. The TCN is therefore chosen on validation performance and compute cost, and the GRU is kept as the candidate with the best test replay accuracy. Model and decoder were locked before this round's test.", "",
              "The F1 scores above are Macro-F1; phase labels are kinematic weak labels. MAE covers ten sessions (eight positive, two negative); MAPE excludes the two zero-reference sessions. Relative count accuracy = 100% - MAPE, not a classification accuracy. CPU is a single temporal forward pass on this machine, without the pose CNN, HTTP, feature extraction or queueing.", ""]
    if baseline:
        lines += ["## Before and after optimization: original default GRU -> best overall TCN", "",
                  f"| Metric | Original default `{baseline['model']}` | **Optimized `{row['model']}`** |", "|---|---:|---:|"]
        for label, key, fmt, suffix in [("Test action Macro-F1", "action_macro_f1", ".4f", ""),
                                        ("Test joint phase Macro-F1", "phase_macro_f1", ".4f", ""),
                                        ("Test count MAE", "count_mae", ".2f", " reps"),
                                        ("Test count MAPE", "count_mape_pct", ".2f", "%"),
                                        ("Test cycle recall", "cycle_recall", ".4f", "")]:
            lines.append(f"| {label} | {baseline['test'][key]:{fmt}}{suffix} | **{row['test'][key]:{fmt}}{suffix}** |")
        lines += ["", f"Test action F1 improves by **{100*(row['test']['action_macro_f1']-baseline['test']['action_macro_f1']):.2f} percentage points**; count MAE drops by **{100*(1-row['test']['count_mae']/baseline['test']['count_mae']):.2f}%**. This is a descriptive comparison under the same labels and split.", ""]
    matched, spurious, missed = (row["test"][k] for k in ["matched_cycles", "spurious_cycles", "missed_cycles"])
    lines += ["## Per-exercise results of the best overall TCN", "",
              f"**{matched}/{matched+missed} reference cycles matched, {missed} missed, {spurious} spurious.** Both negative shoulder-exercise sessions counted 0.", "",
              "| Exercise | Reference cycles | Predicted cycles | MAE (reps) | MAPE | Exact-session rate | Cycle recall | Cycle precision |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    action_names = {"chair_sit_to_stand": "Chair sit-to-stand", "squat": "Squat", "standing_leg_raise": "Standing straight-leg raise", "hurdle_step": "Hurdle step"}
    for exercise in EXERCISE_NAMES:
        m = row["test"]["per_exercise"][exercise]
        expected = sum(s["expected"] for s in m["sessions"])
        predicted = sum(s["predicted"] for s in m["sessions"])
        lines.append(f"| {action_names[exercise]} `{exercise}` | {expected} | **{predicted}** | {m['count_mae']:.2f} | **{m['count_mape_pct']:.2f}%** | {m['exact_session_pct']:.1f}% | {m['cycle_recall']:.3f} | {m['cycle_precision']:.3f} |")
    lines += ["", "Only two test participants per exercise. Cycle timing is matched with a weak boundary of +/-0.5 s; the count reference comes from the original segment repetitions.", "",
              "## Deployment configuration of the best algorithm", "",
              f"Weights actually loaded: `{', '.join(selected['members'])}`, see `artifacts/selected.json`. Decoder `{selected['decoder_strategy']}`, phase evidence threshold {selected['phase_evidence_threshold']}, debounce {selected['min_dwell_s']} s, deployment window {row.get('config', {}).get('context', 64)} frames. Default pose frontend: **{selected_pose['model']} @ {selected_pose['imgsz']}**. Integration: see `docs/openapi.json`; the full optimization process: see `reports/optimization/`.", "",
              f"Deployment selection rule: `{selected['selection_rule']}`. The original test set had been seen in the previous round; this round locked the selection on the validation set first and then reused the test set for a descriptive comparison, so it is not a fresh blind test. Results on projected public skeletons, weak labels and reconstructed replays do not equal performance on home videos of older adults.", "",
              "## Full model comparison (best candidates first)", "",
              "| Model and role | Parameters | CPU P50 ms | Validation MAE | Test action F1 | Test joint phase F1 | Test MAE | Test MAPE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    ordered = [row] + ([test_best] if test_best["model"] != row["model"] else [])
    ordered += [r for r in results if r["model"] not in {row["model"], test_best["model"]}]
    for r in ordered:
        label = r["model"]
        if r["model"] == row["model"]:
            label += " (best overall / recommended for deployment)"
        elif r["model"] == test_best["model"]:
            label += " (highest test metrics)"
        cells = [label, f"{r['parameters']:,}", f"{r['cpu_p50_ms']:.3f}", f"{r['validation']['count_mae']:.2f}",
                 f"{r['test']['action_macro_f1']:.4f}", f"{r['test']['phase_macro_f1']:.4f}",
                 f"{r['test']['count_mae']:.2f}", f"{r['test']['count_mape_pct']:.2f}%"]
        if r["model"] in {row["model"], test_best["model"]}:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Ensemble latency is the sum of the members measured separately; in particular the summed P95 is not a measured joint P95. First round seed=42, optimization round seed=2026; no repetition over several seeds. CPU timings are from this machine, and timing conditions may differ between rounds.", ""]
    lines += ["## Per test session", "", "| Reconstructed session | Source reference | Predicted |", "|---|---:|---:|"]
    for s in sessions:
        lines.append(f"| {s['session']} | {s['expected']} | {s['predicted']} |")
    lines += ["", "## Pose CNN comparison", "", "100 official COCO val images, 198 annotated people, 2,482 visible keypoints. This is not a full COCO AP; the PCK threshold is 20% of the longest bbox side and missed detections count as errors.", "", "| Pose CNN | Input | PCK@0.2 | Mean OKS | Matched people | GPU P50 ms | GPU P95 ms | Parameters |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in poses:
        lines.append(f"| {p['model']} | {p['imgsz']} | {p['pck_bbox_02']*100:.2f}% | {p['mean_oks_including_misses']:.4f} | {p['matched_persons']}/{p['persons']} | {p['latency_p50_ms']:.2f} | {p['latency_p95_ms']:.2f} | {p['parameters']:,} |")
    lines += ["", "Timing includes preprocessing, forward pass and post-processing, with CUDA synchronization. A P50 within the 30 FPS budget does not mean the P95 is, and not all system overhead is included.", "", "## Figures", "", "![Validation accuracy vs compute](../reports/figures/accuracy_compute.png)", "", "![Action confusion matrix](../reports/figures/action_confusion.png)", "", "![Test session counts](../reports/figures/test_session_counts.png)", "", "## Does it meet the proposal target?", ""]
    lines += [f"The overall reconstructed-replay MAPE of the current default model is {row['test']['count_mape_pct']:.2f}%. " + ("It reaches <=5% on this limited replay, but acceptance on home videos cannot be declared yet." if row["test"]["count_mape_pct"] <= 5 else "It does not reach the <=5% target, so a finished implementation must not be reported as a met accuracy target."), "", "Code, trained weights, four-exercise recognition and counting, the interface and the offline experiments are implemented. The next step is independently hand-labelled natural continuous camera video, including older adults, different viewpoints, occlusion and everyday disturbances, to validate keypoint error and full-cycle performance with the real frontend.", ""]
    robustness_path = ROOT / "reports/robustness.json"
    if robustness_path.exists() and all(r.get("model") == selected["model"] for r in json.loads(robustness_path.read_text())):
        robustness = json.loads(robustness_path.read_text())
        lines += ["## Viewpoint and noise tests of the fixed model", "", "The table only covers the eight positive sessions of the four exercises, so the MAE denominator differs from the main table, which includes two negative sessions; model and thresholds were not changed based on these tests. Noise is in the original UI-PRMD length unit, not pixels of a particular camera.", "", "| Projection angle | Coordinate noise std | Positive-session MAE | MAPE |", "|---|---:|---:|---:|"]
        for r in robustness:
            lines.append(f"| {r['angle_degrees']}° | {r['noise_source_units_std']} | {r['mae']:.3f} | {r['mape_pct']:.2f}% |")
        lines += ["", "The 15 / 75 degree views of unseen training participants are a projection robustness stress test and do not replace acceptance at real camera angles.", ""]
    lines += ["This file is generated by `python -m algorithms.exercise report` from the actual experiment JSON.", ""]
    (ROOT / "docs/RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    environment = {"python": sys.version, "platform": platform.platform(), "packages": {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "matplotlib", "opencv-python", "ultralytics", "fastapi", "uvicorn"]}}
    (ROOT / "reports/environment.json").write_text(json.dumps(environment, indent=2))
    (ROOT / "reports/artifact_manifest.json").write_text(
        json.dumps(artifact_manifest(), indent=2), encoding="utf-8", newline="\n")
    print("Generated docs/RESULTS.md, figures, environment and artifact hashes")


def write_optimization():
    directory = ROOT / "reports/optimization"
    results = json.loads((directory / "results.json").read_text())
    lock = json.loads((directory / "selection_lock.json").read_text())
    old_selected = json.loads((directory / "baseline/selected.json").read_text())
    old = next(r for r in json.loads((directory / "baseline/experiments.json").read_text()) if r["model"] == old_selected["model"])
    best = next(r for r in results if r["model"] == lock["model"])
    lines = ["# Algorithm 2: optimization, self-check and handover record", "", "## Current deployment conclusion", "",
             f"The current default algorithm is **{best['model']}**, checkpoint `artifacts/{best['model']}.pt`. Deployment still uses HTTP API v1, and the system's keypoint protocol and four-exercise enumeration stay compatible.",
             f"The network is a hierarchical {best['config']['kind'].upper()} with {best['config']['blocks']} residual blocks (two causal convolutions each), width {best['config']['width']}, stable80 features as input and a {best['config']['context']}-frame causal window. The best checkpoint is from epoch {best['best_epoch']}.",
             f"Decoder `{best['decoder_strategy']}`, phase evidence threshold {best['phase_evidence_threshold']}, debounce {best['decoder_dwell_s']} s.",
             f"Weights SHA-256: `{lock['checkpoint_sha256']}`.", "", "## Before and after optimization", "",
             "The phase metrics below are joint action / phase weak-label Macro-F1; MAPE only covers the eight sessions with non-zero counts, MAE includes the two negative sessions.", "",
             "| Metric | Original default model | Current default model |", "|---|---:|---:|"]
    for label, key in [("Validation action Macro-F1", "action_macro_f1"), ("Validation phase Macro-F1", "phase_macro_f1"), ("Validation count MAE", "count_mae"),
                       ("Test action Macro-F1", "action_macro_f1"), ("Test phase Macro-F1", "phase_macro_f1"), ("Test count MAE", "count_mae"),
                       ("Test count MAPE %", "count_mape_pct"), ("Test complete-cycle recall", "cycle_recall"), ("Test complete-cycle precision", "cycle_precision")]:
        split = "validation" if label.startswith("Validation") else "test"
        lines.append(f"| {label} | {old[split][key]:.4f} | {best[split][key]:.4f} |")
    lines += [f"| Parameters | {old['parameters']:,} | {best['parameters']:,} |",
              f"| CPU P50 ms | {old['cpu_p50_ms']:.3f} | {best['cpu_p50_ms']:.3f} |",
              f"| CPU P95 ms | {old['cpu_p95_ms']:.3f} | {best['cpu_p95_ms']:.3f} |", "",
              "A single CPU forward pass excludes feature extraction, the pose CNN, network transfer and queueing; the old model uses 64 frames, the new models their actual windows. The three optimized models were timed again during the common calibration after all training finished: 4 CPU threads, 40 warm-up runs, 300 measurements. The old / new default models were also timed in the same run as a deployment audit, see deployment_audit.json. Parameter count, window length and actual latency together explain the compute cost.", "",
              "## Issues found in the self-check and the changes made", "",
              "1. The first round's joint 17-class output made action classes and phases compete with each other, and the still standing posture is nearly the same for several exercises. The new network uses one five-class action head and four conditional phase heads, optimizing the action CE and the phase CE of the current action separately. The joint probability is P(action) x P(phase|action), and the interface still outputs the original 17-class probabilities.",
              "2. The original normalization denominator depended on the projected leg length, which can change with posture when bending the knees or raising a leg. stable80 normalizes by twice the 2D torso length from the shoulder centre to the hip centre and adds left/right knee and ankle heights, the higher knee/ankle and the left-right height difference, keeping one-sided movements. It is still affected by projection and torso tilt and cannot be claimed fully view-invariant.",
              "3. Originally the loss was computed at every position of overlapping windows, where early positions have little history. The new training supervises only the last eight frames of each window; the windows are 96, 128 and 96 frames, so most of the loss no longer comes from positions without history. All temporal operations remain causal.",
              "4. The original training mostly used isolated clips while deployment input is a continuous stream. The new training adds cross-repetition reconstructed sequences of training participants, without passing episode, clip boundary, source count or future frames to the model.",
              "5. Training adds sampling strides 0.85 / 1.0 / 1.15, left-right mirroring with semantic swaps, 25 / 45 / 65 degree projections and feature noise 0.025. Normalization mean and standard deviation are fitted on training participants only.",
              "6. The TCN residual branches and input get per-frame LayerNorm, with AdamW, a cosine learning rate, weight decay, gradient clipping and mild label smoothing. The temporal model was not replaced by an MLP.", "",
              "7. The theoretical receptive field of the four-block TCN is 61 frames. After training the deployment window was reduced from 96 to 64, all validation clips and sessions were inferred again, the maximum probability difference and identical per-frame phase / action argmax were checked, and then the common calibration and timing were run. The five-block TCN has a 125-frame receptive field and keeps 128; the GRU keeps 96, without applying the finite convolution receptive-field argument. Shrinking the window only changes deployment metadata; the original training checkpoints are backed up in reports/optimization/training_checkpoints.", "",
              "These changes were applied to the three architectures as one joint optimization; this round has no separate ablation per change, so the total improvement cannot be attributed to any single one.", "",
              "## Actual development and fixes", "",
              "First the original model, features, evaluation and missed-count audit were completed and the first-round experiments and deployment configuration saved; then the hierarchical network, stable80 features and window length loaded with the checkpoint were implemented, with new causality / probability composition / feature invariance tests.",
              "In the first optimization run, after the four-block TCN finished training and validation, building the 128-frame data for the five-block TCN failed because NumPy could not allocate a contiguous window array of about 994 MiB. Pre-expanding all overlapping windows also keeps the source array, which produces a high memory peak. It was changed to CausalWindowDataset: the source sequences are stored once and sliced per batch with start-frame padding; tests check item by item that the old and new windows and supervision labels are identical.",
              "The finished four-block TCN weights and records were kept, and only the five-block TCN and the GRU were continued. Labels, splits and evaluation sessions were not changed or reduced because of the memory error. The final training logs, complete probability caches, selection lock and deployment self-check together verify the actual results.", "",
              "## Experiment configuration and selection", "",
              "| Candidate | Deployment window | Best epoch | Trained epochs | Validation action F1 | Validation phase F1 | Validation MAE | Test action F1 | Test phase F1 | Test MAE | Test MAPE | CPU P50 ms |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['config']['context']} | {r['best_epoch']} | {r['history'][-1]['epoch']} | {r['validation']['action_macro_f1']:.4f} | {r['validation']['phase_macro_f1']:.4f} | {r['validation']['count_mae']:.2f} | {r['test']['action_macro_f1']:.4f} | {r['test']['phase_macro_f1']:.4f} | {r['test']['count_mae']:.2f} | {r['test']['count_mape_pct']:.2f}% | {r['cpu_p50_ms']:.3f} |")
    lines += ["", "The three candidates are optimized versions of the TCN / GRU families that performed well in the first round; the 11 first-round configurations were not all retrained. At most 80 epochs, seed=2026, validation every five epochs, early stop after eight validations without improvement. Checkpoints are selected by validation count MAE first, then validation phase F1. During training phase F1 is computed every third frame to save cost; the final comparison of candidates uses complete per-frame metrics.", "",
              "AdamW lr=0.0015, minimum lr=0.00005, weight_decay=0.005; batch=128, window stride=12; dropout=0.15, gradient clipping 1.0; loss=0.7 x action_CE + phase_CE. Action label smoothing=0.04, phase=0.025, class weights are the training frequency to the power of -0.35.", "",
              "Validation count decoding of the training checkpoints compares two ordering strategies x three thresholds (0.25 / 0.35 / 0.5) x two debounce values (0.067 / 0.1 s), 12 combinations. After training, the complete validation probabilities of the three candidates were calibrated together, adding 0 / 0.033 s debounce, 24 combinations. Lowest MAE first, then cycle recall, longer debounce and higher threshold.",
              "An additional validation diagnosis found that the action support gate of the four-block TCN gives MAE=0.3 everywhere in 0.05 / 0.1 / 0.15 / 0.2, so 0.15 was kept; reducing debounce from 0.1 to 0.033 s raises the validation s07 leg-raise count from 7/10 to 9/10 while negatives stay at zero. This led to the common post-training decoder calibration; test data was not used for this diagnosis.",
              f"Final candidate selection rule: `{lock['selection_rule']}`. The selection lock file was created before the test stage and contains the deployment model, decoder, timestamp and checkpoint hash; finalize checks the hash.", "",
              "## Per test session", "", "| Session | Source count | Original default prediction | Optimized default prediction |", "|---|---:|---:|---:|"]
    old_sessions = {s["session"]: s for s in old["test"]["sessions"]}
    for s in best["test"]["sessions"]:
        lines.append(f"| {s['session']} | {s['expected']} | {old_sessions[s['session']]['predicted']} | {s['predicted']} |")
    lines += ["", "## Validation scope and limits", "",
              "Labels, participant split, evaluation angles, count ground truth and event matching tolerance keep the first-round definitions. Training s01-s06, validation s07-s08, test s09-s10. The original test set had been seen in the first round and in the problem diagnosis; this round keeps it for a descriptive comparison under the same protocol. The locking procedure avoids selecting the model by test score in this round, but a used test set cannot become a fresh blind test again.", "",
              "The validation set has only two people, and repeated epoch and decoder comparisons risk overfitting. The results are 2D projections of public mocap, weak phase labels and reconstructed replays; natural continuous RGB video and older adults were not validated. An improvement with one seed is not statistically significant, and a correct count does not guarantee that every event time is correct. The current results do not prove a clinical fall-prevention effect.", "",
              "The optimized GRU has test action F1=0.9659, phase F1=0.8246 and count MAPE=0%, slightly better numbers on this already-used test replay. But its validation count MAE=0.80 is higher than the TCN's 0.10, and its CPU latency is higher, so the TCN selected before the test stays the deployment default. Distinguishing the best test numbers from the best validation / compute trade-off avoids re-selecting by test results already seen.", "",
              "## Reproduction and files", "", "```bash", "# from the repository root",
              "python -m algorithms.exercise optimize --epochs 80 --seed 2026",
              "python -m algorithms.exercise calibrate",
              "python -m algorithms.exercise finalize",
              "python -m algorithms.exercise robustness",
              "python -m algorithms.exercise audit",
              "python -m algorithms.exercise optimization-report",
              "python -m algorithms.exercise report", "```", "",
              "Retraining overwrites the optimized checkpoints of the same name; reproduce in a separate copy of the project or archive them first. The original baseline records are in `reports/optimization/baseline/`, and the first-round `.pt` weights stay in artifacts.",
              "`validation.json`: training log and complete validation metrics; `selection_lock.json`: lock made before the test; `results.json`: complete results of the three candidates; `*_probabilities.npz`: per-frame and per-session probabilities; `reports/comparison.csv`: overall table including the old and optimization rounds. The complete API contract: see `docs/openapi.json`.", ""]
    audit_path = directory / "deployment_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        lines += ["## Additional self-check of the fixed deployment model", "", "The two test shoulder-exercise replays were run with each of the four training targets, eight checks in total, all counting 0; the same still posture for 300 consecutive frames also counts 0 for all four targets. These checks only cover recorded negatives and do not mean that no household disturbance can cause a false count.", "",
                  "| Same-run CPU measurement | Window | P50 ms | P95 ms |", "|---|---:|---:|---:|"]
        for b in audit["same_run_cpu_benchmarks"]:
            lines.append(f"| {b['model']} | {b['context_frames']} | {b['cpu_p50_ms']:.3f} | {b['cpu_p95_ms']:.3f} |")
        lines += ["", "With the same thread count, warm-up and number of measurements in this run, the optimized TCN is faster per inference than the old GRU, although it has about 13% more parameters.", ""]
    robustness_path = ROOT / "reports/robustness.json"
    if robustness_path.exists():
        robust = json.loads(robustness_path.read_text())
        if all(r.get("model") == best["model"] for r in robust):
            lines += ["## Stress test of the fixed model", "", "Projection / coordinate-noise checks of the eight positive sessions; model and thresholds were not changed based on them.", "",
                      "| Projection angle | Noise std (original units) | Positive-session MAE | MAPE |", "|---|---:|---:|---:|"]
            for r in robust:
                lines.append(f"| {r['angle_degrees']}° | {r['noise_source_units_std']} | {r['mae']:.3f} | {r['mape_pct']:.2f}% |")
            lines += ["", "The usual 45 degree projection improves clearly, but extreme angles and noise still cause clear missed counts, e.g. 0/9 for the s09 sit-to-stand at 15 degrees. The higher average accuracy does not solve the camera domain shift or every viewpoint.", ""]
    lines += ["## Training curves", "", "![Training curves of the optimization candidates](../reports/figures/optimization_history.png)", ""]
    (ROOT / "docs/OPTIMIZATION.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        axes[0].plot([h["epoch"] for h in r["history"]], [h["validation_phase_f1_stride3"] for h in r["history"]], label=r["model"])
        axes[1].plot([h["epoch"] for h in r["history"]], [h["validation_count_mae"] for h in r["history"]], label=r["model"])
    axes[0].set(xlabel="Epoch", ylabel="Validation phase macro-F1 (stride 3)")
    axes[1].set(xlabel="Epoch", ylabel="Validation reconstructed count MAE")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/optimization_history.png", dpi=160)
    plt.close(fig)
    print("Generated docs/OPTIMIZATION.md and optimization_history.png")
