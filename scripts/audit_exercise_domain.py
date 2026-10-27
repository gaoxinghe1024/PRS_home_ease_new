"""Compare observed poses with original training distributions, without counting."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithm2"))
from homeease.exercises import EXERCISES
from homeease.features import stable_features
from pipeline.data import project

REQUIRED = [0, 1, 6, 7, 8, 9, 10, 11]


def measurements(poses):
    features = np.array([stable_features(p) for p in poses])
    return {"higher_knee_y": features[:, 36], "higher_ankle_y": features[:, 37],
            "mean_elbow_y": features[:, [5, 7]].mean(1),
            "mean_wrist_y": features[:, [9, 11]].mean(1)}


def summary(values):
    return {name: {"p05": float(np.quantile(x, .05)), "p50": float(np.median(x)),
                   "p95": float(np.quantile(x, .95))} for name, x in values.items()}


def audit(record):
    exercise = record["state"]["exercise"]
    if exercise not in {"standing_leg_raise", "hurdle_step"}:
        raise ValueError("This comparison is for the two leg exercises.")
    valid = []
    for frame in record["frames"]:
        pose = np.asarray(frame["keypoints"], dtype=np.float32)
        if pose.shape == (17, 3):
            pose = pose[5:17]
        if (pose.shape == (12, 3) and np.isfinite(pose).all()
                and np.min(pose[REQUIRED, 2]) >= .3
                and frame.get("person_id", "primary") == "primary"):
            valid.append(pose)
    if not valid:
        raise ValueError("No visible single-person poses in the trace.")
    records = json.loads((ROOT / "algorithm2/data/processed/records.json").read_text())
    skeletons = np.load(ROOT / "algorithm2/data/processed/skeletons.npz")
    reference = {0: [], 2: []}
    clips = 0
    for source in records:
        if source["split"] != "train" or source["incorrect"] or source["movement"] != EXERCISES[exercise]["movement"]:
            continue
        poses = project(skeletons[f"{source['id']}_skel"])
        phases = skeletons[f"{source['id']}_labels"] % 4
        for phase in reference:
            reference[phase].extend(poses[phases == phase])
        clips += 1
    return {"exercise": exercise, "observed_frames": len(record["frames"]),
        "visible_single_person_frames": len(valid), "truncated": record.get("truncated", False),
        "training_clips": clips, "projection_degrees": 45,
        "coordinates": "hip_centered_y_down_divided_by_twice_torso_length",
        "quantile_levels": [.05, .5, .95],
        "training_rest": summary(measurements(reference[0])),
        "training_hold": summary(measurements(reference[2])),
        "observed_all_visible_phases": summary(measurements(valid)),
        "limitations": ["observed_frames_have_no_manual_phase_labels",
                        "different_distributions_are_not_proof_of_a_single_cause",
                        "these_quantiles_are_not_counting_thresholds"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.input.read_text(encoding="utf-8")))
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
