"""Audit training-only mirror averaging without changing weights or count rules."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithm2"))
from homeease.counter import CycleCounter
from homeease.engine import ExerciseSession, Predictor
from homeease.exercises import decode
from homeease.features import extract_sequence
from pipeline.optimize import predict, sessions_for
from pipeline.data import project

TARGETS = ("standing_leg_raise", "hurdle_step")
SWAP = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]


def batch_probabilities(predictor, poses):
    results = []
    for member in predictor.members:
        features = extract_sequence(poses, mode=member["mode"])
        results.append(predict(member["model"], features, member["mean"], member["std"],
                               member["context"], predictor.device))
    return np.mean(results, axis=0)


def count(probabilities, exercise, selected):
    counter = CycleCounter(min_dwell_s=selected["min_dwell_s"])
    for i, p in enumerate(probabilities):
        canonical = decode(p, exercise, "training")[1]
        counter.update_probabilities(canonical, i / 30, selected["phase_evidence_threshold"],
                                     infer_return=True)
    return counter.count


def compare(predictor, poses, source, identifier):
    reflected = poses[:, SWAP].copy()
    reflected[:, :, 0] *= -1
    original = batch_probabilities(predictor, poses)
    average = (original + batch_probabilities(predictor, reflected)) / 2
    return [{"source": source, "id": identifier, "target": target,
             "positive": source == target,
             "original_count": count(original, target, predictor.selected),
             "mirror_count": count(average, target, predictor.selected)} for target in TARGETS]


def replay_trace(predictor, path):
    record = json.loads(path.read_text(encoding="utf-8"))
    results = {}
    for target in TARGETS:
        counts = {}
        for views in ("original", "mirror_average"):
            session = ExerciseSession(predictor, exercise=target, action_scope="training")
            if views == "original":
                session.stream = predictor.stream()
                session.inference_views = "original"
            for frame in record["frames"]:
                session.update(frame["timestamp_ms"], frame["keypoints"], frame.get("person_id", "primary"))
            counts[views] = session.snapshot()["count"]
        results[target] = counts
    return {"source_exercise": record["state"]["exercise"], "frames": len(record["frames"]),
            "truncated": record.get("truncated", False),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "counts": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/exercise_views_validation.json")
    args = parser.parse_args()
    predictor = Predictor(device=args.device)
    records = json.loads((ROOT / "algorithm2/data/processed/records.json").read_text())
    rows, static = [], []
    with np.load(ROOT / "algorithm2/data/processed/skeletons.npz") as skeletons:
        for split in ("validation", "test"):
            for source in sessions_for(records, skeletons, split):
                poses = project(source["skel"])
                rows.extend(compare(predictor, poses, source["exercise"], source["id"]))
                if source["exercise"] in TARGETS:
                    static.extend(compare(predictor, np.repeat(poses[:1], 180, axis=0),
                                          "stationary", source["id"]))
                print(f'{split}: {source["id"]}', flush=True)
    traces = [replay_trace(predictor, path) for path in args.trace]
    negative = [r for r in rows + static if not r["positive"]]
    positive = [r for r in rows if r["positive"]]
    report = {
        "model": predictor.selected["model"], "action_scope": "training",
        "public_inference": "batched causal windows, original EMA and 30 Hz",
        "trace_inference": "actual streaming SDK with original input interruptions",
        "positive_sessions": len(positive),
        "positive_counts": [{k: r[k] for k in ("id", "target", "original_count", "mirror_count")} for r in positive],
        "negative_checks": len(negative),
        "negative_cycles": sum(r["mirror_count"] for r in negative),
        "public_checks": rows, "static_checks": static, "captured_traces": traces,
        "limitation": "Captured traces can be truncated and are not independent live accuracy validation."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("positive_sessions", "positive_counts", "negative_checks", "negative_cycles", "captured_traces")}))
    if report["negative_cycles"] or any(r["mirror_count"] < r["original_count"] for r in positive):
        raise SystemExit("View augmentation regressed the checked public cases")


if __name__ == "__main__":
    main()
