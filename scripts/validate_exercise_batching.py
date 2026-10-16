"""Compare sequential and batched mirror inference on identical causal input."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithm2"))
from homeease.engine import ExerciseSession, MirroredPhaseStream, Predictor
from scripts.optimize_top import sessions_for
from scripts.prepare_data import project

TARGETS = ("standing_leg_raise", "hurdle_step")
BEHAVIOR = ("count", "counter_stage", "phase", "recognized_exercise", "quality", "cycles", "interruptions",
            "active_duration_s", "valid_duration_s")


class SequentialMirror:
    def __init__(self, predictor):
        self.original, self.mirrored = predictor.stream(), predictor.stream()

    def reset(self):
        self.original.reset()
        self.mirrored.reset()

    def update(self, pose, timestamp):
        reflected = pose[[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]].copy()
        reflected[:, 0] *= -1
        return (self.original.update(pose, timestamp) + self.mirrored.update(reflected, timestamp)) / 2


class RecordedStream:
    def __init__(self, stream):
        self.stream = stream
        self.rows = []

    def reset(self):
        self.stream.reset()

    def update_many(self, observations):
        method = getattr(self.stream, "update_many", None)
        probabilities = (method(observations) if method is not None else
                         [self.stream.update(pose, t) for pose, t in observations])
        self.rows.extend((t, p.copy()) for (_, t), p in zip(observations, probabilities))
        return probabilities


class CachedStream:
    def __init__(self, rows):
        self.rows = iter(rows)

    def reset(self):
        pass

    def update(self, pose, timestamp):
        t, p = next(self.rows)
        assert t == timestamp
        return p.copy()


def stats(values):
    return {"p50_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95))} if values else {}


def replay(predictor, frames, target, stream):
    session = ExerciseSession(predictor, exercise=target, action_scope="training")
    session.stream = stream
    snapshots, times = [], []
    for frame in frames:
        started = time.perf_counter()
        state = session.update(frame["timestamp_ms"], frame["keypoints"], frame.get("person_id", "primary"))
        duration = (time.perf_counter() - started) * 1000
        if state["quality"] == "good":
            times.append(duration)
        snapshots.append({key: state[key] for key in BEHAVIOR})
    return session.snapshot(), snapshots, stats(times[10:])


def compare(predictor, frames, source, identifier, atol):
    runs, audit = {}, {}
    target = source if source in TARGETS else TARGETS[0]
    for name, cls in [("sequential", SequentialMirror), ("batched", MirroredPhaseStream)]:
        stream = RecordedStream(cls(predictor))
        state, snapshots, timing = replay(predictor, frames, target, stream)
        runs[name] = (state, snapshots, stream.rows)
        audit[name] = {"valid_frame_phase_update": timing}
    reference, actual = runs["sequential"][2], runs["batched"][2]
    assert [t for t, _ in reference] == [t for t, _ in actual]
    a, b = np.array([p for _, p in reference]), np.array([p for _, p in actual])
    error = float(np.max(np.abs(a - b))) if len(a) else 0
    assert error <= atol, (identifier, error)
    checks = []
    for exercise in TARGETS:
        states, snapshots = {}, {}
        for name, (_, _, rows) in runs.items():
            states[name], snapshots[name], _ = replay(predictor, frames, exercise, CachedStream(rows))
        assert snapshots["sequential"] == snapshots["batched"], (identifier, exercise, "behavior differs")
        checks.append({"target": exercise, "positive": source == exercise,
                       "sequential_count": states["sequential"]["count"], "batched_count": states["batched"]["count"],
                       "all_frame_behavior_equal": True, "cycles_equal": True})
    return {"id": identifier, "source": source, "frames": len(frames), "model_ticks": len(reference),
            "max_probability_error": error, "timings": audit, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--skip-public", action="store_true")
    parser.add_argument("--probability-atol", type=float, default=2e-6)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/exercise_batching_validation.json")
    args = parser.parse_args()
    predictor = Predictor(device=args.device)
    public, static, captured = [], [], []
    if not args.skip_public:
        records = json.loads((ROOT / "algorithm2/data/processed/records.json").read_text())
        with np.load(ROOT / "algorithm2/data/processed/skeletons.npz") as skeletons:
            for split in ("validation", "test"):
                for source in sessions_for(records, skeletons, split):
                    poses = project(source["skel"])
                    # Keep real 30 Hz timestamps, including the between-observation ticks.
                    frames = [{"timestamp_ms": i / 30 * 1000, "keypoints": poses[i]} for i in range(0, len(poses), 4)]
                    public.append(compare(predictor, frames, source["exercise"], source["id"], args.probability_atol))
                    if source["exercise"] in TARGETS:
                        frames = [{"timestamp_ms": i / 30 * 1000, "keypoints": poses[0]} for i in range(0, 180, 4)]
                        static.append(compare(predictor, frames, "stationary", source["id"], args.probability_atol))
                    print(f'{split}: {source["id"]}', flush=True)
    for path in args.trace:
        record = json.loads(path.read_text(encoding="utf-8"))
        row = compare(predictor, record["frames"], record["state"]["exercise"], path.stem, args.probability_atol)
        row.update(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   truncated=record.get("truncated", False), saved_system_count=record["state"]["count"])
        captured.append(row)
        print(f'trace: {path.stem}', flush=True)
    report = {"model": predictor.selected["model"], "device": args.device, "probability_atol": args.probability_atol,
              "method": "Both paths use identical real observations and causal 30 Hz SDK resampling. "
                        "Only already-available ticks/views share model calls; counters still consume every tick in order.",
              "all_frame_behavior_equal": True,
              "public_checks": public, "static_checks": static, "captured_traces": captured,
              "limitation": "Performance excludes pose, IPC, storage and camera scheduling. "
                            "Unchanged saved-input counts do not validate live accuracy or recover missed camera frames."}
    rows = public + static + captured
    report["max_probability_error"] = max((r["max_probability_error"] for r in rows), default=0)
    report["public_positive_checks"] = sum(c["positive"] for r in public for c in r["checks"])
    negative = [c for r in public + static for c in r["checks"] if not c["positive"]]
    report["public_negative_checks"] = len(negative)
    report["public_negative_cycles"] = sum(c["batched_count"] for c in negative)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["all_frame_behavior_equal", "max_probability_error",
                     "public_positive_checks", "public_negative_checks", "public_negative_cycles"]}), flush=True)


if __name__ == "__main__":
    main()
