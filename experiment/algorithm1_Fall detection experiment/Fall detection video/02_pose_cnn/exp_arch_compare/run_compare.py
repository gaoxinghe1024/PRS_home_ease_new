"""Three-architecture comparison driver."""

import os
import sys
import json
import argparse
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
ARCHS = ["p1", "p2", "p3"]
ARCH_NAMES = {
    "p1": "P1 SmallUNet (U-Net encoder-decoder)",
    "p2": "P2 CPM3Stage (3-stage iterative refinement + intermediate supervision)",
    "p3": "P3 SimpleBaseline (slim ResNet + 3x deconv head)",
}


def train_one(arch, epochs, mode):
    cmd = [
        sys.executable,
        os.path.join(HERE, "..", "train.py"),
        "--arch",
        arch,
        "--epochs",
        str(epochs),
    ]
    if mode == "full":
        cmd += ["--workers", "4"]
    print(f"\n===== training {arch} ({mode}, {epochs} epochs) =====", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise RuntimeError(f"{arch} training failed, exit={r.returncode}")
    hist_path = os.path.join(HERE, "..", "outputs", arch, "history.json")
    with open(hist_path, encoding="utf-8") as f:
        hist = json.load(f)
    hist["_wall_sec"] = round(time.time() - t0, 1)
    return hist


def summarize(results):
    archs_done = [a for a in ARCHS if a in results]
    rows = []
    for arch in archs_done:
        h = results[arch]
        hist = h["history"]
        best = max(hist, key=lambda r: r["val_pck02"])
        rows.append(
            {
                "arch": arch,
                "name": ARCH_NAMES[arch],
                "params_m": None,
                "best_epoch": best["epoch"],
                "val_pck02": round(best["val_pck02"], 4),
                "val_pck05": round(best["val_pck05"], 4),
                "final_loss": round(hist[-1]["train_loss"], 4),
                "train_wall_sec": h["_wall_sec"],
            }
        )
    from models import build_model, count_params

    for r in rows:
        m, _ = build_model(r["arch"])
        r["params_m"] = round(count_params(m) / 1000000.0, 2)
    comp = {"mode": results["_mode"], "epochs": results["_epochs"], "rows": rows}
    with open(os.path.join(HERE, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comp, f, ensure_ascii=False, indent=1)
    best_arch = max(rows, key=lambda r: r["val_pck02"])["arch"]
    score_txt = "; ".join((f"{r['arch']}: PCK@0.2={r['val_pck02']:.4f}" for r in rows))
    lines = [
        f"# Architecture comparison results "
        f"(mode={results['_mode']}, epochs={results['_epochs']})\n",
        "> Single-variable principle: identical data, hyperparameters and seed; "
        "the only difference is the network architecture.",
        "> Metrics: PCK@0.2 (primary) and PCK@0.5 (secondary); "
        "the best epoch is selected on validation PCK@0.2.\n",
        "| Architecture | Params | Best epoch | PCK@0.2 | PCK@0.5 | Train loss | Train time |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['params_m']}M | {r['best_epoch']} | **{r['val_pck02']:.4f}** | {r['val_pck05']:.4f} | {r['final_loss']:.4f} | {r['train_wall_sec'] / 60:.1f}min |"
        )
    lines += [
        "",
        f"**Conclusion: {best_arch} is best on PCK@0.2** ({score_txt}).",
        "",
        "## Learning curve (val PCK@0.2 per epoch)",
        "",
    ]
    for arch in archs_done:
        hist = results[arch]["history"]
        curve = " ".join((f"{r['val_pck02']:.3f}" for r in hist))
        lines.append(f"- **{arch}**: {curve}")
    lines.append("")
    with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return best_arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val_split", "full"], default="val_split")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--archs", default="p1,p2,p3")
    a = ap.parse_args()
    epochs = a.epochs or (30 if a.mode == "full" else 8)
    results = {"_mode": a.mode, "_epochs": epochs}
    for arch in [s.strip() for s in a.archs.split(",")]:
        results[arch] = train_one(arch, epochs, a.mode)
    best_arch = summarize(results)
    print(
        f"\n>>> best architecture: {best_arch} (for deployment copy outputs/{best_arch}/best.pt -> outputs/best.pt for export_infer)"
    )


if __name__ == "__main__":
    main()
