# -*- coding: utf-8 -*-
"""Flatten the JSON output of all five experiments into Markdown tables on stdout."""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
R = os.path.join(ROOT, "results")


def load(n):
    p = os.path.join(R, n)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float) and np.isnan(x):
        return "nan"
    return f"{x:.{nd}f}"


def main():
    e1 = load("exp01_preproc.json")
    e2 = load("exp02_features.json")
    e3 = load("exp03_alg.json")
    e4 = load("exp04_robustness.json")
    e5 = load("exp05_sensitivity.json")

    if e1:
        print("\n## Experiment 1: preprocessing\n")
        print("| config | seq AUROC | win Acc | win Rec | seq Acc | seq Rec | ev Rec | ev Prec | ev Acc | peak ret. | quiet noise |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for grp, tag in (("E1a_resample", "resample"), ("E1b_filter", "filter")):
            for k, v in e1[grp].items():
                w, s, ev = v["window"], v["sequence"], v["event"]
                sq = v.get("signal") or {}
                print(f"| {tag}: {k} | {f(v['auroc_seq'],4)} | {f(w['acc'])} | {f(w['recall'])} | "
                      f"{f(s['acc'])} | {f(s['recall'])} | {f(ev['event_recall'])} | "
                      f"{f(ev['event_precision'])} | {f(ev['event_accuracy'])} | "
                      f"{f(sq.get('peak_retention'))} | {f(sq.get('quiet_noise_ratio'))} |")

    if e2:
        print("\n## Experiment 2: feature groups\n")
        print("| feature group | dims | seq AUROC | win Acc | win Rec | ev Rec | ev Acc |")
        print("|---|---|---|---|---|---|---|")
        for k, v in e2["E2a_families"].items():
            w, ev = v["window"], v["event"]
            print(f"| {k} | {v.get('n_feat')} | {f(v['auroc_seq'],4)} | {f(w['acc'])} | "
                  f"{f(w['recall'])} | {f(ev['event_recall'])} | {f(ev['event_accuracy'])} |")
        print("\n## Experiment 2: feature selection\n")
        print("| method | seq AUROC | win Acc | win Rec | ev Rec | ev Prec |")
        print("|---|---|---|---|---|---|")
        for k, v in e2["E2b_selection"].items():
            w, ev = v["window"], v["event"]
            print(f"| {k} | {f(v['auroc_seq'],4)} | {f(w['acc'])} | {f(w['recall'])} | "
                  f"{f(ev['event_recall'])} | {f(ev['event_precision'])} |")
        print("\n### Single-feature AUROC, top 15")
        for r in e2["E2c_univariate"][:15]:
            print(f"- {r['feature']}: {r['auroc']:.4f}")
        print("\n### Importance, top 15")
        for r in e2["E2d_importance"][:15]:
            print(f"- {r['feature']}: {r['importance']:.4f}")

    if e3:
        print("\n## Experiment 3: algorithm comparison\n")
        print("| algorithm | seq AUROC | win Acc | win Rec | win F1 | seq Acc | seq Rec | seq F1 | ev Acc | ev Rec | ev Prec | ev F1 | delay s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, v in e3["algorithms"].items():
            s = v["summary"]
            g = lambda x: f(s[x]["mean"])
            print(f"| {v['label']} | {f(s['auroc_seq']['mean'],4)}+-{f(s['auroc_seq']['std'],3)} | "
                  f"{g('win_acc')} | {g('win_recall')} | {g('win_f1')} | {g('seq_acc')} | {g('seq_recall')} | "
                  f"{g('seq_f1')} | {g('ev_acc')} | {g('ev_recall')} | {g('ev_prec')} | {g('ev_f1')} | {g('delay_s')} |")
        ex = e3.get("extra", {})
        if "a3_per_net" in ex:
            print("\nA3 internal architectures:", {k: round(v, 4) for k, v in ex["a3_per_net"].items()})
        if "a1_subscores" in ex:
            print("A1 sub-scores:", {k: round(v, 4) for k, v in ex["a1_subscores"].items()})
        if "a4_detail" in ex:
            d = ex["a4_detail"]
            print("A4 base-learner OOF:", {k: round(v, 4) for k, v in d["base_oof_auroc"].items()})
            print("A4 fuser inner CV:", {k: round(v, 4) for k, v in d["fusion_cv"].items()},
                  "-> selected", d["selected"])

    # exp04 keys by (algorithm, condition); conditions are never pooled across algorithms
    if e4:
        print("\n## Experiment 4: robustness\n")
        print("| algorithm | condition | seq AUROC | win Acc | win Rec | seq Rec | ev Rec | ev Prec | fp/h | delay s |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for alg, conds in e4.get("algorithms", {}).items():
            for k, v in conds.items():
                print(f"| {alg} | {k} | {f(v.get('auroc_seq'), 4)} | {f(v.get('win_acc'))} | "
                      f"{f(v.get('win_recall'))} | {f(v.get('seq_recall'))} | {f(v.get('ev_recall'))} | "
                      f"{f(v.get('ev_prec'))} | {f(v.get('fp_per_hour'), 2)} | {f(v.get('delay_mean_s'), 2)} |")

    # exp05 is a config x algorithm matrix: one table per metric
    if e5:
        cfgs = e5.get("configs", {})
        algs = sorted({a for c in cfgs.values() for a in c})
        for metric, title in (("auroc_seq", "sequence AUROC"), ("ev_recall", "event recall")):
            print(f"\n## Experiment 5: preprocessing sensitivity ({title})\n")
            print("| config | " + " | ".join(algs) + " |")
            print("|---" * (len(algs) + 1) + "|")
            for name, per_alg in cfgs.items():
                cells = [f(per_alg.get(a, {}).get(metric), 4) for a in algs]
                print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
