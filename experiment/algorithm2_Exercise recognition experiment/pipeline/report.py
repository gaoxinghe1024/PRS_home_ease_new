"""Export experiment metrics, plots and artifact hashes."""
import importlib.metadata
import json
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT, artifact_manifest


def write_results():
    results = json.loads((ROOT / "reports/experiments.json").read_text())
    poses = json.loads((ROOT / "reports/pose_comparison.json").read_text())
    selected = json.loads((ROOT / "artifacts/selected.json").read_text())
    row = next(r for r in results if r["model"] == selected["model"])
    baseline_path = ROOT / "reports/optimization/baseline/selected.json"
    baseline_name = json.loads(baseline_path.read_text())["model"] if baseline_path.exists() else None
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
    summary = {
        "selected_model": selected,
        "selected_pose": selected_pose,
        "baseline_model": baseline_name,
        "validation": row["validation"],
        "test": row["test"],
        "pose_comparison": poses,
    }
    robustness_path = ROOT / "reports/robustness.json"
    if robustness_path.exists():
        robustness = json.loads(robustness_path.read_text())
        if all(r.get("model") == selected["model"] for r in robustness):
            summary["robustness"] = robustness
    _write_json(ROOT / "reports/results.json", summary)
    packages = ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "matplotlib",
                "opencv-python", "ultralytics", "fastapi", "uvicorn")
    environment = {"python": sys.version, "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in packages}}
    _write_json(ROOT / "reports/environment.json", environment)
    _write_json(ROOT / "reports/artifact_manifest.json", artifact_manifest(ROOT))
    print("Generated result JSON, figures and artifact hashes")


def write_optimization():
    directory = ROOT / "reports/optimization"
    results = json.loads((directory / "results.json").read_text())
    lock = json.loads((directory / "selection_lock.json").read_text())
    old_selected = json.loads((directory / "baseline/selected.json").read_text())
    old_results = json.loads((directory / "baseline/experiments.json").read_text())
    old = next(r for r in old_results if r["model"] == old_selected["model"])
    best = next(r for r in results if r["model"] == lock["model"])
    fields = ("model", "parameters", "cpu_p50_ms", "cpu_p95_ms", "validation", "test")
    summary = {
        "selection_lock": lock,
        "baseline": {key: old[key] for key in fields},
        "selected": {key: best[key] for key in fields},
        "candidates": [r["model"] for r in results],
    }
    audit_path = directory / "deployment_audit.json"
    if audit_path.exists():
        summary["deployment_audit"] = json.loads(audit_path.read_text())
    _write_json(directory / "summary.json", summary)
    (ROOT / "reports/figures").mkdir(parents=True, exist_ok=True)
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
    print("Generated optimization JSON and training curves")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8", newline="\n")
