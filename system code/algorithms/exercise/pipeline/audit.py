"""Post-training checks that never feed back into a model or a threshold."""
import hashlib
import json

import numpy as np
import torch

from homeease.engine import Predictor
from homeease.exercises import EXERCISE_NAMES, EXERCISES
from homeease.features import extract_sequence
from homeease.models import build_model

from .common import ROOT
from .data import project
from .metrics import count_probs, cpu_benchmark, infer, replay


def fixed_model():
    predictor = Predictor(device="cuda" if torch.cuda.is_available() else "cpu")
    records = json.loads((ROOT / "data/processed/records.json").read_text())
    skeletons = np.load(ROOT / "data/processed/skeletons.npz")
    output = {"model": predictor.selected["model"], "negative_all_targets": [], "static_all_targets": [], "same_run_cpu_benchmarks": []}
    selected = predictor.selected
    def probabilities(pose):
        return np.mean([infer(m["model"], extract_sequence(pose, mode=m["mode"]), m["mean"], m["std"],
                              predictor.device, context=m["context"]) for m in predictor.members], axis=0)
    def count(probs, exercise):
        return count_probs(probs, selected["min_dwell_s"], exercise=exercise, strategy=selected["decoder_strategy"],
                           threshold=selected["phase_evidence_threshold"]).count
    for session in replay(skeletons, records, "test", "m07"):
        p = probabilities(project(session["skel"]))
        for exercise in EXERCISE_NAMES:
            output["negative_all_targets"].append({"source": session["id"], "target": exercise, "count": count(p, exercise), "expected": 0})
    pose = project(skeletons["m01_s09_e01_skel"])[0]
    p = probabilities(np.repeat(pose[None], 300, axis=0))
    for exercise in EXERCISE_NAMES:
        output["static_all_targets"].append({"target": exercise, "count": count(p, exercise), "expected": 0})
    # Compare models under the same timing settings.
    old = json.loads((ROOT / "reports/optimization/baseline/selected.json").read_text())
    for filename in old["members"] + selected["members"]:
        package = torch.load(ROOT / "artifacts" / filename, map_location="cpu", weights_only=True)
        model = build_model(package["config"], package["input_dim"])
        model.load_state_dict(package["state_dict"])
        output["same_run_cpu_benchmarks"].append({"model": filename.removesuffix(".pt"), "context_frames": package["context_frames"],
                                                   **cpu_benchmark(model, package["input_dim"], package["context_frames"])})
    (ROOT / "reports/optimization/deployment_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    assert all(r["count"] == 0 for r in output["negative_all_targets"] + output["static_all_targets"]), "False cycles found; report and investigate"


def robustness():
    skeletons = np.load(ROOT / "data/processed/skeletons.npz")
    records = json.loads((ROOT / "data/processed/records.json").read_text())
    predictor = Predictor(device="cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(123)
    outcomes = []
    for angle, noise in [(15, 0), (45, 0), (75, 0), (45, 1)]:
        details = []
        for exercise, config in EXERCISES.items():
            for source in replay(skeletons, records, "test", config["movement"]):
                poses = project(source["skel"], angle)
                if noise:
                    poses[:, :, :2] += rng.normal(0, noise, poses[:, :, :2].shape)
                probabilities = np.mean([infer(m["model"], extract_sequence(poses, mode=m["mode"]),
                                               m["mean"], m["std"], predictor.device, context=m["context"])
                                         for m in predictor.members], axis=0)
                counter = count_probs(probabilities, predictor.selected["min_dwell_s"], exercise=exercise,
                                      strategy=predictor.selected["decoder_strategy"],
                                      threshold=predictor.selected["phase_evidence_threshold"])
                details.append({"session": source["id"], "exercise": exercise,
                                "reference": source["expected_count"], "count": counter.count})
        outcomes.append({"model": predictor.selected["model"], "angle_degrees": angle, "noise_source_units_std": noise,
                         "mae": float(np.mean([abs(d["count"] - d["reference"]) for d in details])),
                         "mape_pct": float(np.mean([abs(d["count"] - d["reference"]) / d["reference"] * 100 for d in details])),
                         "details": details})
        print(outcomes[-1], flush=True)
    (ROOT / "reports/robustness.json").write_text(json.dumps(outcomes, indent=2))


def sources():
    directory = ROOT / "data/raw/uiprmd"
    manifest = json.loads((directory / "manifest.json").read_text())
    for entry in manifest["files"]:
        raw = (directory / entry["file"]).read_bytes()
        assert len(raw) == entry["bytes"], entry["file"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], entry["file"]
        assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == entry["git_blob_sha"], entry["file"]
    result = {"verified_source_files": len(manifest["files"]), "source_bytes": sum(e["bytes"] for e in manifest["files"]),
              "source_commit": manifest["commit"], "sha256_and_git_blob_sha": "passed"}
    (ROOT / "reports/source_integrity.json").write_text(json.dumps(result, indent=2))
    print(result)
