"""Subject-disjoint training and validation-only model selection over the first-round grid."""
import csv
import json
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from homeease.features import FEATURE_DIMS, extract_sequence
from homeease.models import build_model

from .common import ROOT, seeds
from .config import CONFIGS, FPS, LABEL_SOURCE, PROJECTION_ANGLES
from .data import project
from .metrics import benchmark, count_metrics, infer, replay, windows


def train(epochs=25, seed=42, only=None):
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = json.loads((ROOT / "data/processed/records.json").read_text())
    skeletons = np.load(ROOT / "data/processed/skeletons.npz")
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    report = ROOT / "reports"
    report.mkdir(exist_ok=True)
    sessions = {s: sum([replay(skeletons, records, s, movement) for movement in ["m05", "m01", "m06", "m02", "m07"]], [])
                for s in ["validation", "test"]}
    results = []
    probabilities = {}
    configs = [c for c in CONFIGS if not only or c["name"] in only]
    for config in configs:
        name = config["name"]
        config = {"features": "hybrid", "dropout": .1, **config, "seed": seed}
        mode = config["features"]
        seeds(seed)
        training = [(extract_sequence(project(skeletons[f"{r['id']}_skel"], angle), mode=mode), skeletons[f"{r['id']}_labels"])
                    for r in records if r["split"] == "train" for angle in PROJECTION_ANGLES]
        # Statistics come from training participants only.
        values = np.concatenate([x for x, _ in training])
        mean, std = values.mean(0), np.maximum(values.std(0), .1)
        xx, yy = [], []
        for x, y in training:
            w = windows((x - mean) / std)
            yw = windows(y[:, None]).squeeze(-1)
            for i in sorted(set(range(0, len(x), 16)) | {len(x) - 1}):
                xx.append(w[i])
                labels = yw[i].copy()
                labels[:max(0, 63 - i)] = -100
                yy.append(labels)
        loader = DataLoader(TensorDataset(torch.tensor(np.asarray(xx)), torch.tensor(np.asarray(yy))),
                            batch_size=64, shuffle=True, num_workers=0)
        model = build_model(config, FEATURE_DIMS[mode]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        labels_all = np.concatenate([y for _, y in training])
        class_weights = torch.tensor((1 / np.maximum(np.bincount(labels_all, minlength=17), 1)) ** .5, dtype=torch.float32, device=device)
        class_weights /= class_weights.mean()
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        val_clips = [(extract_sequence(project(skeletons[f"{r['id']}_skel"]), mode=mode), skeletons[f"{r['id']}_labels"])
                     for r in records if r["split"] == "validation"]
        best_score, history, best_state = -1, [], None
        started = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                prediction = model(x + torch.randn_like(x) * .015)
                loss = loss_fn(prediction.transpose(1, 2), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                model.eval()
                val_probs = [infer(model, x, mean, std, device) for x, _ in val_clips]
                score = f1_score(np.concatenate([y for _, y in val_clips]), np.concatenate(val_probs).argmax(1), average="macro")
                history.append({"epoch": epoch + 1, "train_loss": total_loss / len(loader), "validation_macro_f1": score})
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"{name} epoch {epoch+1:02}: loss={total_loss/len(loader):.4f}, val_f1={score:.4f}", flush=True)
        training_s = time.perf_counter() - started
        model.load_state_dict(best_state)
        model.eval()
        measured = benchmark(model, FEATURE_DIMS[mode])
        package = {"config": config, "state_dict": best_state, "mean": mean.tolist(), "std": std.tolist(),
                   "context_frames": 64, "fps": FPS, "input_dim": FEATURE_DIMS[mode], "label_source": LABEL_SOURCE, "num_classes": 17}
        torch.save(package, output / f"{name}.pt")
        entry = {"model": name, "config": config, "training_s": training_s, "history": history, **measured}
        probabilities[name] = {}
        for split in ["validation", "test"]:
            split_records = [r for r in records if r["split"] == split]
            probs, truth = [], []
            for r in split_records:
                x = extract_sequence(project(skeletons[f"{r['id']}_skel"]), mode=mode)
                probs.append(infer(model, x, mean, std, device))
                truth.append(skeletons[f"{r['id']}_labels"])
            phase_prob, phase_truth = np.concatenate(probs), np.concatenate(truth)
            session_probs = [infer(model, extract_sequence(project(s["skel"]), mode=mode), mean, std, device) for s in sessions[split]]
            probabilities[name][split] = {"phase": phase_prob, "truth": phase_truth, "sessions": session_probs}
            entry[split] = {"phase_macro_f1": f1_score(phase_truth, phase_prob.argmax(1), average="macro"),
                            "phase_report": classification_report(phase_truth, phase_prob.argmax(1), output_dict=True, zero_division=0),
                            "action_macro_f1": f1_score(phase_truth // 4, phase_prob.argmax(1) // 4, average="macro"),
                            "confusion_matrix": confusion_matrix(phase_truth, phase_prob.argmax(1), labels=list(range(17))).tolist(),
                            **count_metrics(session_probs, sessions[split])}
        results.append(entry)
        (report / "experiments.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"FINISHED {name}: test F1={entry['test']['phase_macro_f1']:.4f}, MAE={entry['test']['count_mae']:.2f}, CPU={measured['cpu_p50_ms']:.3f}ms", flush=True)
    # Ensemble membership and weights come from validation alone.
    ranking = sorted(results, key=lambda r: (r["validation"]["count_mae"], -r["validation"]["phase_macro_f1"]))
    members = [r["model"] for r in ranking[:2]]
    ensemble = {"model": "soft_vote_top2", "members": members,
                "parameters": sum(r["parameters"] for r in ranking[:2]),
                "cpu_p50_ms": sum(r["cpu_p50_ms"] for r in ranking[:2]),
                "cpu_p95_ms": sum(r["cpu_p95_ms"] for r in ranking[:2]),
                "weight_kib": sum(r["weight_kib"] for r in ranking[:2]), "latency_note": "sum of member benchmark latencies"}
    for split in ["validation", "test"]:
        phase = np.mean([probabilities[m][split]["phase"] for m in members], axis=0)
        truth = probabilities[members[0]][split]["truth"]
        merged = [np.mean(p, axis=0) for p in zip(*[probabilities[m][split]["sessions"] for m in members])]
        ensemble[split] = {"phase_macro_f1": f1_score(truth, phase.argmax(1), average="macro"),
                           "action_macro_f1": f1_score(truth // 4, phase.argmax(1) // 4, average="macro"),
                           **count_metrics(merged, sessions[split])}
    results.append(ensemble)
    # Stay within .02 phase F1 and .25 count MAE of the best, then take the cheapest CPU model.
    best_mae = min(r["validation"]["count_mae"] for r in results)
    candidates = [r for r in results if r["validation"]["count_mae"] <= best_mae + .25]
    best_f1 = max(r["validation"]["phase_macro_f1"] for r in candidates)
    candidates = [r for r in candidates if r["validation"]["phase_macro_f1"] >= best_f1 - .02]
    chosen = min(candidates, key=lambda r: r["cpu_p50_ms"])
    selected_members = chosen.get("members", [chosen["model"]])
    selected = {"api_version": "v1", "model": chosen["model"], "members": [f"{m}.pt" for m in selected_members],
                "pose_model": "yolo11n-pose.pt", "confidence_threshold": 0.0, "min_dwell_s": .067,
                "selection_rule": "validation MAE within .25 of best, phase F1 within .02 of best remaining, minimum CPU p50"}
    (output / "selected.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    (report / "experiments.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (report / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["model", "parameters", "cpu_p50_ms", "cpu_p95_ms", "weight_kib", "val_f1", "val_mae", "test_f1", "test_mae", "test_mape_pct", "test_cycle_recall"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({**{k: r[k] for k in fields[:5]}, "val_f1": r["validation"]["phase_macro_f1"],
                             "val_mae": r["validation"]["count_mae"], "test_f1": r["test"]["phase_macro_f1"],
                             "test_mae": r["test"]["count_mae"], "test_mape_pct": r["test"]["count_mape_pct"],
                             "test_cycle_recall": r["test"]["cycle_recall"]})
    print(f"SELECTED {chosen['model']}", flush=True)
