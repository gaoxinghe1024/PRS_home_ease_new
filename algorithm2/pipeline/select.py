"""Consistent decoder evaluation over every first-round checkpoint, validation-calibrated."""
import csv
import json

import numpy as np
import torch

from homeease.features import extract_sequence
from homeease.models import build_model

from .common import ROOT
from .config import CONFIGS, MOVEMENTS
from .data import project
from .metrics import benchmark, count_metrics, evaluate, infer, replay


def select(preview=False):
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = json.loads((ROOT / "data/processed/records.json").read_text())
    skeletons = np.load(ROOT / "data/processed/skeletons.npz")
    sessions = {s: sum([replay(skeletons, records, s, m) for m in MOVEMENTS], []) for s in ["validation", "test"]}
    history = {r["model"]: r for r in json.loads((ROOT / "reports/experiments.json").read_text())}
    results, predictions = [], {}
    for cfg in CONFIGS:
        path = ROOT / "artifacts" / f"{cfg['name']}.pt"
        if not path.exists():
            if preview:
                continue
            raise FileNotFoundError(path)
        package = torch.load(path, weights_only=True, map_location="cpu")
        if package.get("num_classes") != 17:
            if preview:
                continue
            raise ValueError(f"Incomplete 4-exercise checkpoint: {path}")
        model = build_model(package["config"], package["input_dim"]).to(device).eval()
        model.load_state_dict(package["state_dict"])
        mean, std = np.array(package["mean"], dtype=np.float32), np.array(package["std"], dtype=np.float32)
        mode, name = package["config"]["features"], cfg["name"]
        predictions[name] = {}
        for split in ["validation", "test"]:
            probs, truth = [], []
            for r in records:
                if r["split"] == split:
                    x = extract_sequence(project(skeletons[f"{r['id']}_skel"]), mode=mode)
                    probs.append(infer(model, x, mean, std, device))
                    truth.append(skeletons[f"{r['id']}_labels"])
            session_probs = [infer(model, extract_sequence(project(s["skel"]), mode=mode), mean, std, device) for s in sessions[split]]
            predictions[name][split] = {"phase": np.concatenate(probs), "truth": np.concatenate(truth), "sessions": session_probs}
        # Decoder dwell is calibrated once per model on the validation recordings.
        options = []
        for strategy, threshold in [("argmax", .35)] + [(strategy, t) for strategy in ["ordered_posterior", "ordered_transition"] for t in [.25, .35, .5]]:
            for dwell in [0, .033, .067, .1]:
                metrics = count_metrics(predictions[name]["validation"]["sessions"], sessions["validation"], dwell, strategy=strategy, threshold=threshold)
                options.append({"strategy": strategy, "threshold": threshold, "dwell_s": dwell, "mae": metrics["count_mae"], "cycle_recall": metrics["cycle_recall"]})
        choice = min(options, key=lambda o: (o["mae"], -o["cycle_recall"], -o["dwell_s"], -o["threshold"]))
        row = {"model": name, "config": package["config"], "decoder_dwell_s": choice["dwell_s"], "decoder_strategy": choice["strategy"], "phase_evidence_threshold": choice["threshold"], "decoder_validation_grid": options,
               "training_s": history.get(name, {}).get("training_s"), "history": history.get(name, {}).get("history", []),
               **benchmark(model, package["input_dim"])}
        for split in ["validation", "test"]:
            p = predictions[name][split]
            row[split] = evaluate(p["phase"], p["truth"], p["sessions"], sessions[split], choice["dwell_s"], choice["strategy"], choice["threshold"])
        results.append(row)
        print(f"{name}: val MAE={row['validation']['count_mae']:.2f}, test MAE={row['test']['count_mae']:.2f}, action F1={row['test']['action_macro_f1']:.3f}, dwell={choice['dwell_s']}", flush=True)
    if preview:
        (ROOT / "reports/decoder_preview.json").write_text(json.dumps(results, indent=2))
        return
    ranking = sorted(results, key=lambda r: (r["validation"]["count_mae"], -r["validation"]["phase_macro_f1"]))
    members = [r["model"] for r in ranking[:2]]
    ensemble_predictions = {}
    for split in ["validation", "test"]:
        ensemble_predictions[split] = {"phase": np.mean([predictions[m][split]["phase"] for m in members], axis=0),
                                      "truth": predictions[members[0]][split]["truth"],
                                      "sessions": [np.mean(p, axis=0) for p in zip(*[predictions[m][split]["sessions"] for m in members])]}
    options = [{"dwell_s": d, "strategy": strategy, "threshold": threshold,
                **count_metrics(ensemble_predictions["validation"]["sessions"], sessions["validation"], d, strategy=strategy, threshold=threshold)}
               for strategy, threshold in [("argmax", .35)] + [(strategy, t) for strategy in ["ordered_posterior", "ordered_transition"] for t in [.25, .35, .5]] for d in [0, .033, .067, .1]]
    best = min(options, key=lambda o: (o["count_mae"], -o["cycle_recall"], -o["dwell_s"], -o["threshold"]))
    ensemble = {"model": "soft_vote_top2", "members": members, "decoder_dwell_s": best["dwell_s"], "decoder_strategy": best["strategy"], "phase_evidence_threshold": best["threshold"],
                "parameters": sum(r["parameters"] for r in ranking[:2]), "weight_kib": sum(r["weight_kib"] for r in ranking[:2]),
                "cpu_p50_ms": sum(r["cpu_p50_ms"] for r in ranking[:2]), "cpu_p95_ms": sum(r["cpu_p95_ms"] for r in ranking[:2]),
                "latency_note": "sum of independently benchmarked member CPU times; not a measured joint p95"}
    for split in ["validation", "test"]:
        p = ensemble_predictions[split]
        ensemble[split] = evaluate(p["phase"], p["truth"], p["sessions"], sessions[split], best["dwell_s"], best["strategy"], best["threshold"])
    results.append(ensemble)
    best_mae = min(r["validation"]["count_mae"] for r in results)
    eligible = [r for r in results if r["validation"]["count_mae"] <= best_mae + .25]
    best_f1 = max(r["validation"]["phase_macro_f1"] for r in eligible)
    eligible = [r for r in eligible if r["validation"]["phase_macro_f1"] >= best_f1 - .02]
    selected = min(eligible, key=lambda r: r["cpu_p50_ms"])
    (ROOT / "artifacts/selected.json").write_text(json.dumps({"api_version": "v1", "model": selected["model"],
        "members": [f"{m}.pt" for m in selected.get("members", [selected["model"]])], "pose_model": "yolo11n-pose.pt",
        "confidence_threshold": 0, "min_dwell_s": selected["decoder_dwell_s"], "decoder": "conditioned_target_action_support_015",
        "decoder_strategy": selected["decoder_strategy"], "phase_evidence_threshold": selected["phase_evidence_threshold"],
        "selection_rule": "validation MAE <= best+.25, joint phase F1 >= eligible best-.02, minimum CPU p50"}, indent=2))
    (ROOT / "reports/experiments.json").write_text(json.dumps(results, indent=2))
    fields = ["model", "parameters", "cpu_p50_ms", "cpu_p95_ms", "weight_kib", "val_phase_f1", "val_mae", "test_action_f1", "test_phase_f1", "test_mae", "test_mape_pct", "test_cycle_recall"]
    with (ROOT / "reports/comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({**{k: r[k] for k in fields[:5]}, "val_phase_f1": r["validation"]["phase_macro_f1"], "val_mae": r["validation"]["count_mae"],
                             "test_action_f1": r["test"]["action_macro_f1"], "test_phase_f1": r["test"]["phase_macro_f1"],
                             "test_mae": r["test"]["count_mae"], "test_mape_pct": r["test"]["count_mape_pct"], "test_cycle_recall": r["test"]["cycle_recall"]})
    # Cached probability sequences keep the decoder audits reproducible.
    cached = {f"{r['model']}_{split}_{i}": p for r in results[:-1] for split in ["validation", "test"] for i, p in enumerate(predictions[r['model']][split]["sessions"])}
    np.savez_compressed(ROOT / "reports/session_probabilities.npz", **cached)
    print("SELECTED", selected["model"], "dwell", selected["decoder_dwell_s"], flush=True)
