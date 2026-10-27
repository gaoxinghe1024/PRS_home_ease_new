"""Compare deployment phase policies through the SDK with identical posteriors."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithm2"))
from homeease.engine import ExerciseSession, MirroredPhaseStream, Predictor
from pipeline.optimize import sessions_for
from pipeline.data import project
from validate_exercise_views import batch_probabilities

TARGETS = ("standing_leg_raise", "hurdle_step")
SWAP = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]


class RecordedStream:
    def __init__(self, probabilities=None, stream=None):
        self.probabilities = probabilities if probabilities is not None else {}
        self.stream = stream

    def reset(self):
        if self.stream is not None:
            self.stream.reset()

    def update(self, pose, timestamp):
        key = round(timestamp, 6)
        if self.stream is not None:
            self.probabilities[key] = self.stream.update(pose, timestamp)
        return self.probabilities[key].copy()


def replay(predictor, frames, target, policy, stream):
    session = ExerciseSession(predictor, exercise=target, action_scope="training", phase_policy=policy)
    session.stream = stream
    for frame in frames:
        session.update(frame["timestamp_ms"], frame["keypoints"], frame.get("person_id", "primary"))
    return session.snapshot()


def compare(predictor, frames, probabilities, source, identifier):
    rows = []
    for target in TARGETS:
        states = {policy: replay(predictor, frames, target, policy if target == "hurdle_step" else "original", RecordedStream(probabilities))
                  for policy in ("original", "action_gap_hold_v1")}
        rows.append({"id": identifier, "source": source, "target": target,
                     "positive": source == target,
                     "original_count": states["original"]["count"],
                     "continuity_count": states["action_gap_hold_v1"]["count"],
                     "cycles": states["action_gap_hold_v1"]["cycles"]})
    return rows


def public_input(predictor, poses):
    reflected = poses[:, SWAP].copy()
    reflected[:, :, 0] *= -1
    posteriors = (batch_probabilities(predictor, poses) + batch_probabilities(predictor, reflected)) / 2
    frames = [{"timestamp_ms": i / 30 * 1000, "keypoints": pose} for i, pose in enumerate(poses)]
    probabilities = {round(i / 30, 6): p for i, p in enumerate(posteriors)}
    return frames, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/exercise_continuity_validation.json")
    args = parser.parse_args()
    predictor = Predictor(device=args.device)
    records = json.loads((ROOT / "algorithm2/data/processed/records.json").read_text())
    public, static, captured = [], [], []
    with np.load(ROOT / "algorithm2/data/processed/skeletons.npz") as skeletons:
        for split in ("validation", "test"):
            for source in sessions_for(records, skeletons, split):
                poses = project(source["skel"])
                frames, probabilities = public_input(predictor, poses)
                public.extend(compare(predictor, frames, probabilities, source["exercise"], source["id"]))
                if source["exercise"] in TARGETS:
                    frames, probabilities = public_input(predictor, np.repeat(poses[:1], 180, axis=0))
                    static.extend(compare(predictor, frames, probabilities, "stationary", source["id"]))
                print(f'{split}: {source["id"]}', flush=True)
    for path in args.trace:
        record = json.loads(path.read_text(encoding="utf-8"))
        stream = RecordedStream(stream=MirroredPhaseStream(predictor))
        replay(predictor, record["frames"], record["state"]["exercise"], "original", stream)
        captured.append({"input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "source": record["state"]["exercise"], "saved_system_count": record["state"]["count"],
                         "frames": len(record["frames"]), "truncated": record.get("truncated", False),
                         "first_timestamp_ms": record["frames"][0]["timestamp_ms"],
                         "checks": compare(predictor, record["frames"], stream.probabilities,
                                           record["state"]["exercise"], path.stem)})
        print(f'trace: {path.stem}', flush=True)
    positive = [row for row in public if row["positive"]]
    negative = [row for row in public + static if not row["positive"]]
    captured_new_cross_action_cycles = sum(max(row["continuity_count"] - row["original_count"], 0)
        for trace in captured for row in trace["checks"] if not row["positive"])
    report = {"model": predictor.selected["model"], "action_scope": "training",
              "phase_policy": "action_gap_hold_v1", "policy_exercises": ["hurdle_step"], "max_action_gap_hold_s": .15,
              "method": "SDK replay; identical causal mirror-averaged model posteriors for both policies",
              "positive_sessions": len(positive),
              "positive_count_regressions": sum(row["continuity_count"] < row["original_count"] for row in positive),
              "negative_checks": len(negative), "negative_cycles": sum(row["continuity_count"] for row in negative),
              "captured_new_cross_action_cycles": captured_new_cross_action_cycles,
              "public_checks": public, "static_checks": static, "captured_traces": captured,
              "limitation": "Captured input is truncated, has no per-cycle labels, and is not live accuracy validation. "
                            "Cross-action errors already present in live input remain unresolved. "
                            "Reported 7-9 repetitions are approximate whole-session counts, not trace labels."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("positive_sessions", "positive_count_regressions", "negative_checks", "negative_cycles")}))
    if report["positive_count_regressions"] or report["negative_cycles"] or captured_new_cross_action_cycles:
        raise SystemExit("Phase continuity regressed the checked cases")


if __name__ == "__main__":
    main()
