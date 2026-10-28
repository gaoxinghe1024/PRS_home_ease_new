"""Full pipeline driver: architecture selection, skeletons, algorithm comparison."""

import os
import sys
import json
import shutil
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
POSE = os.path.join(ROOT, "02_pose_cnn")
COMMON = os.path.join(ROOT, "00_common")


def pick_best_arch():
    cmp_path = os.path.join(POSE, "exp_arch_compare", "comparison.json")
    if not os.path.exists(cmp_path):
        raise SystemExit(
            f"comparison results not found {cmp_path}; run exp_arch_compare/run_compare.py first"
        )
    with open(cmp_path, encoding="utf-8") as f:
        comp = json.load(f)
    rows = comp["rows"]
    best = max(rows, key=lambda r: r["val_pck02"])
    print(f"best architecture: {best['arch']} (PCK@0.2={best['val_pck02']})")
    return best["arch"]


def stage1_extract(arch):
    src = os.path.join(POSE, "outputs", arch, "best.pt")
    dst = os.path.join(POSE, "outputs", "best.pt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"best.pt: {src} → {dst}")
    sys.path.insert(0, COMMON)
    import splits

    available = [
        d
        for d in ("urfd", "mc", "le2i")
        if any(s[0] == d for s in splits.unified_pool())
    ]
    print(f"[stage1] datasets present: {available}", flush=True)
    for ds in available:
        cmd = [sys.executable, os.path.join(COMMON, "pose_extract.py"), ds]
        print(f"\n===== skeleton extraction {ds} =====", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(f"{ds} skeleton extraction failed")


def stage2_compare():
    cmd = [sys.executable, os.path.join(ROOT, "experiments", "exp01_alg_compare.py")]
    print("\n===== A1/A2/A3 algorithm comparison =====", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit("algorithm comparison failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pose", action="store_true", help="skip skeleton extraction")
    ap.add_argument(
        "--arch",
        default=None,
        help="manually specify the architecture (overrides auto-selection)",
    )
    a = ap.parse_args()
    arch = a.arch or pick_best_arch()
    if not a.skip_pose:
        stage1_extract(arch)
    stage2_compare()
    print("\nmain pipeline done -> experiments/exp01_results.json")


if __name__ == "__main__":
    main()
