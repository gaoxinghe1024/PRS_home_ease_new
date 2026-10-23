"""Replay exported camera landmarks through the algorithm 2 SDK."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithm2"))

from homeease.engine import ExerciseSession, Predictor
from homeease.exercises import EXERCISES


def diagnose(record, exercises, device="cpu", action_scope=None, phase_policy=None, counting_policy=None):
    predictor = Predictor(device=device)
    results = {}
    for exercise in exercises:
        scope = action_scope or record.get("state", {}).get("action_scope", "open")
        policy = phase_policy or record.get("state", {}).get("phase_policy", "original")
        if scope != "training" or exercise != "hurdle_step":
            policy = "original"
        counting = counting_policy or record.get("state", {}).get("counting_policy", "original")
        if scope != "training" or exercise not in {"chair_sit_to_stand", "standing_leg_raise"}:
            counting = "original"
        session = ExerciseSession(predictor, exercise=exercise, action_scope=scope, phase_policy=policy,
                                  counting_policy=counting)
        phases, qualities, stages = Counter(), Counter(), Counter()
        peak_probabilities = {}
        transitions = []
        previous = None
        for frame in record["frames"]:
            observed = session.update(frame["timestamp_ms"], frame["keypoints"], frame.get("person_id", "primary"))
            phases[observed["phase"]] += 1
            qualities[observed["quality"]] += 1
            stages[str(session.counter.stage)] += 1
            for name, probability in observed["phase_probabilities"].items():
                peak_probabilities[name] = max(peak_probabilities.get(name, 0), probability)
            current = (session.counter.stage, observed["quality"], observed["count"])
            if current != previous:
                transitions.append({"timestamp_ms": frame["timestamp_ms"], "stage": current[0],
                    "quality": current[1], "count": current[2], "phase": observed["phase"],
                    "phase_probabilities": observed["phase_probabilities"],
                    "recognized_exercise": observed["recognized_exercise"],
                    "exercise_probabilities": observed["exercise_probabilities"],
                    "missing_joints": [i for i, point in enumerate(frame["keypoints"])
                                       if i in (5, 6, 11, 12, 13, 14, 15, 16) and point[2] < .3]})
                previous = current
        state = session.snapshot()
        results[exercise] = {key: state[key] for key in
            ("action_scope", "inference_views", "phase_policy", "counting_policy", "count", "quality", "phase", "phase_probabilities", "valid_duration_s", "elapsed_s")}
        results[exercise]["interruption_counts"] = dict(Counter(i["reason"] for i in state["interruptions"]))
        results[exercise].update(phase_observations=dict(phases), quality_observations=dict(qualities),
            counter_stage_observations=dict(stages), peak_phase_probabilities=peak_probabilities,
            counter_transitions=transitions)
        original = record.get("state", {})
        if not record.get("truncated") and original.get("exercise") == exercise:
            # Controls are not input frames, so only compare unpaused sessions.
            paused = any(i["reason"] in {"paused", "waiting_for_input"} for i in original.get("interruptions", []))
            same_mode = (scope == original.get("action_scope", "open")
                and state["inference_views"] == original.get("inference_views", "original")
                and state["phase_policy"] == original.get("phase_policy", "original")
                and counting == original.get("counting_policy", "original"))
            if not paused and same_mode:
                results[exercise]["system_count"] = original["count"]
                results[exercise]["count_matches_system"] = state["count"] == original["count"]
    return {"model_id": predictor.selected["model"], "frames": len(record["frames"]),
            "truncated": record.get("truncated", False), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--exercise", choices=["all", *EXERCISES], default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-scope", choices=["open", "training"])
    parser.add_argument("--phase-policy", choices=["original", "action_gap_hold_v1"])
    parser.add_argument("--counting-policy", choices=["original", "pose_cycle_v1", "pose_cycle_v2"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = json.loads(args.input.read_text(encoding="utf-8"))
    report = diagnose(record, list(EXERCISES) if args.exercise == "all" else [args.exercise], args.device,
                      args.action_scope, args.phase_policy, args.counting_policy)
    encoded = json.dumps(report, ensure_ascii=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
