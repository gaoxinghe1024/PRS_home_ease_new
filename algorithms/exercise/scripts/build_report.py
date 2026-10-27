"""Generate auditable Markdown and scientific figures from actual JSON results."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.exercises import EXERCISE_NAMES


def main():
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
              f"Weights actually loaded: `{', '.join(selected['members'])}`, see `artifacts/selected.json`. Decoder `{selected['decoder_strategy']}`, phase evidence threshold {selected['phase_evidence_threshold']}, debounce {selected['min_dwell_s']} s, deployment window {row.get('config', {}).get('context', 64)} frames. Default pose frontend: **{selected_pose['model']} @ {selected_pose['imgsz']}**. Integration: see [API.md](API.md); the full optimization process: see [OPTIMIZATION.md](OPTIMIZATION.md).", "",
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
    lines += ["This file is generated by `scripts/build_report.py` from the actual experiment JSON. Development method and evaluation limits: see [DEVELOPMENT.md](DEVELOPMENT.md).", ""]
    (ROOT / "docs/RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    environment = {"python": sys.version, "platform": platform.platform(), "packages": {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "matplotlib", "opencv-python", "ultralytics", "fastapi", "uvicorn"]}}
    (ROOT / "reports/environment.json").write_text(json.dumps(environment, indent=2))
    manifest = [{"file": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((ROOT / "artifacts").glob("*")) if p.is_file()]
    (ROOT / "reports/artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Generated docs/RESULTS.md, figures, environment and artifact hashes")


if __name__ == "__main__":
    main()
