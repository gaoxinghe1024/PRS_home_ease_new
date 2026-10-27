"""Offline measurement: windows, inference, replay counting and the metric definitions."""
import copy
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from homeease.counter import PHASES, CycleCounter
from homeease.exercises import EXERCISES, EXERCISE_NAMES, MOVEMENT_TO_INDEX, decode


def windows(x, context=64):
    padded = np.pad(x, ((context - 1, 0), (0, 0)), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, context, axis=0).transpose(0, 2, 1).copy()


def infer(model, x, mean, std, device, context=64):
    data = windows((x - mean) / std, context)
    output = []
    with torch.inference_mode():
        for start in range(0, len(data), 256):
            output.append(model(torch.tensor(data[start:start + 256], device=device))[:, -1].softmax(-1).cpu().numpy())
    return np.concatenate(output)


def benchmark(model, dimension):
    model = copy.deepcopy(model).cpu().eval()
    sample = torch.zeros(1, 64, dimension)
    with torch.inference_mode():
        for _ in range(30):
            model(sample)
        elapsed = []
        for _ in range(200):
            start = time.perf_counter()
            model(sample)
            elapsed.append((time.perf_counter() - start) * 1000)
    return {"cpu_p50_ms": float(np.median(elapsed)), "cpu_p95_ms": float(np.quantile(elapsed, .95)),
            "parameters": sum(p.numel() for p in model.parameters()), "weight_kib": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024}


def cpu_benchmark(model, dimension, context):
    model = copy.deepcopy(model).cpu().eval()
    sample = torch.zeros(1, context, dimension)
    elapsed = []
    with torch.inference_mode():
        for _ in range(40):
            model(sample)
        for _ in range(300):
            start = time.perf_counter()
            model(sample)
            elapsed.append((time.perf_counter() - start) * 1000)
    parameters = sum(p.numel() for p in model.parameters())
    return {"cpu_p50_ms": float(np.median(elapsed)), "cpu_p95_ms": float(np.quantile(elapsed, .95)),
            "parameters": parameters, "weight_kib": parameters * 4 / 1024}


def action_predictions(p):
    return np.column_stack([p[:, i * 4:i * 4 + 4].sum(1) for i in range(4)] + [p[:, 16]]).argmax(1)


def replay(skeletons, records, split, movement="m05"):
    sessions = []
    for subject in sorted({r["subject"] for r in records if r["split"] == split}):
        clips = [r for r in records if r["subject"] == subject and r["movement"] == movement and not r["incorrect"]]
        merged, truth = [], []
        for r in sorted(clips, key=lambda r: r["episode"]):
            sk = skeletons[f"{r['id']}_skel"].copy()
            ankle = (sk[0, 16] + sk[0, 20]) / 2
            sk[:, :, 0] -= ankle[0]
            sk[:, :, 2] -= ankle[2]
            merged.append(sk)
            truth.append(skeletons[f"{r['id']}_labels"])
        if merged:
            exercise = next((k for k, v in EXERCISES.items() if v["movement"] == movement), "other")
            sessions.append({"id": f"s{subject:02}_{movement}_reconstructed", "skel": np.concatenate(merged),
                             "exercise": exercise, "labels": np.concatenate(truth),
                             "expected_count": len(clips) - 1 if movement == "m05" else len(clips) if movement in MOVEMENT_TO_INDEX else 0})
    return sessions


def count_probs(probabilities, dwell=.067, confidence=0, exercise="chair_sit_to_stand", strategy="argmax", threshold=.35):
    counter = CycleCounter(min_dwell_s=dwell)
    for i, p in enumerate(probabilities):
        if len(p) == 17:
            phase, canonical, _ = decode(p, exercise)
        else:
            phase, canonical = int(p.argmax()), p
        if strategy in ["ordered_posterior", "ordered_transition"]:
            counter.update_probabilities(canonical, i / 30, threshold, infer_return=strategy == "ordered_transition")
        else:
            counter.update(PHASES[phase] if p.max() >= confidence else "other", i / 30)
    return counter


def match_events(predicted, reference, tolerance=.5):
    unused = set(range(len(reference)))
    matched = 0
    for event in predicted:
        options = [i for i in unused if abs(reference[i]["end_s"] - event["end_s"]) <= tolerance]
        if options:
            best = min(options, key=lambda i: abs(reference[i]["end_s"] - event["end_s"]))
            unused.remove(best)
            matched += 1
    return matched, len(predicted) - matched, len(unused)


def count_metrics(session_probs, sessions, dwell=.067, confidence=0, strategy="argmax", threshold=.35):
    errors, relative, exact, details = [], [], [], []
    tp = fp = fn = 0
    for prob, session in zip(session_probs, sessions):
        exercise = session["exercise"] if session["exercise"] != "other" else "chair_sit_to_stand"
        if prob.shape[1] == 17 and confidence == 0:
            if session.get("_probability_source") is not prob:
                session["_probability_source"] = prob
                session["_canonical_probabilities"] = np.asarray([decode(p, exercise)[1] for p in prob])
            counting_probabilities = session["_canonical_probabilities"]
        else:
            counting_probabilities = prob
        counter = count_probs(counting_probabilities, dwell, confidence, exercise, strategy, threshold)
        expected = session["expected_count"]
        error = abs(counter.count - expected)
        errors.append(error)
        if expected:
            relative.append(error / expected)
        exact.append(error == 0)
        if "_reference_events" not in session:
            target = EXERCISE_NAMES.index(exercise)
            phase_labels = np.where(session["labels"] // 4 == target, session["labels"] % 4, 4)
            session["_reference_events"] = count_probs(np.eye(5)[phase_labels], .033, exercise=exercise).events
        reference = session["_reference_events"]
        a, b, c = match_events(counter.events, reference)
        tp, fp, fn = tp + a, fp + b, fn + c
        details.append({"session": session["id"], "exercise": session["exercise"], "expected": expected, "predicted": counter.count,
                        "events": counter.events})
    return {"count_mae": float(np.mean(errors)), "count_mape_pct": float(100 * np.mean(relative)),
            "count_accuracy_pct": float(100 * max(0, 1 - np.mean(relative))), "exact_session_pct": float(100 * np.mean(exact)),
            "cycle_recall": tp / max(tp + fn, 1), "cycle_precision": tp / max(tp + fp, 1),
            "matched_cycles": tp, "spurious_cycles": fp, "missed_cycles": fn, "sessions": details}


def evaluate(probs, truth, session_probs, sessions, dwell, strategy="argmax", threshold=.35):
    per_exercise = {}
    for exercise in EXERCISE_NAMES:
        selected = [(p, s) for p, s in zip(session_probs, sessions) if s["exercise"] == exercise]
        per_exercise[exercise] = count_metrics([p for p, _ in selected], [s for _, s in selected], dwell, strategy=strategy, threshold=threshold)
    negative = [int(s["exercise"] == "other") for s in sessions]
    phase_names = [f"{e}/{phase}" for e, cfg in EXERCISES.items() for phase in cfg["phases"]] + ["other"]
    return {"phase_macro_f1": f1_score(truth, probs.argmax(1), average="macro", labels=range(17), zero_division=0),
            "action_macro_f1": f1_score(truth // 4, action_predictions(probs), average="macro"),
            "phase_report": classification_report(truth, probs.argmax(1), labels=range(17), target_names=phase_names, output_dict=True, zero_division=0),
            "action_confusion_matrix": confusion_matrix(truth // 4, action_predictions(probs), labels=range(5)).tolist(),
            "confusion_matrix": confusion_matrix(truth, probs.argmax(1), labels=range(17)).tolist(),
            **count_metrics(session_probs, sessions, dwell, strategy=strategy, threshold=threshold), "per_exercise": per_exercise}
