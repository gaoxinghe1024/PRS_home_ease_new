# -*- coding: utf-8 -*-
"""Experiment 3: temporal classification models and ensemble strategies (A1-A4).

Single-variable principle: all four algorithms share the SAME preprocessed + windowed
cache (preprocessing config from Experiment 1, or set on the CLI); the only difference
is the temporal decision algorithm.

  A1 threshold rule    -- learning-free baseline (impact peak / motion intensity /
                          post-impact rest, three weighted conditions)
  A2 features + ML     -- 39 handcrafted features + RF/XGB
  A3 temporal DNN      -- LSTM/GRU/BiLSTM/TCN soft voting (raw waveform, end to end)
  A4 hybrid ensemble   -- 7 heterogeneous base learners (3 feature + 4 temporal paths)
                          with an automatically selected fuser

Protocol:
  * primary protocol = stratified 5-fold CV over 3 seeds (2026/2027/2028), reported as mean +- std
  * metrics          = sequence-level AUROC (primary) + window/sequence/event Acc and Recall

Additional analyses:
  * inside A3: each network architecture evaluated separately
  * inside A4: OOF AUROC of the 7 base learners + inner-CV score of each fuser candidate
  * inside A1: ablation of the three sub-scores

Usage: python exp03_alg_compare.py [--algs a1,a2,a3,a4] [--seeds 2026,2027,2028] [--epochs 20] [--jobs 5]
Output -> results/exp03_alg.json
"""
import os
import sys
import json
import time
import argparse
import functools
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for sub in ("00_common", "03_alg_rule/src", "04_alg_ml/src",
            "05_alg_temporal/src", "06_alg_hybrid/src"):
    sys.path.insert(0, os.path.join(ROOT, *sub.split("/")))

import metrics as M
import dataset
import evaluate
import splits
import windowize

FS_DEFAULT = 60.0
ALG_NAMES = {"a1": "A1_threshold_rule", "a2": "A2_feature_ml",
             "a3": "A3_temporal_nets", "a4": "A4_hybrid_stacking"}


def cfg_from_exp01():
    import preprocess
    valid = set(preprocess.FILTER_CHOICES)
    p = os.path.join(ROOT, "results", "exp01_preproc.json")
    if not os.path.isfile(p):
        return {"resample": 60.0, "filt": "butter10"}
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    cand = []
    for k, v in d.get("E1a_resample", {}).items():
        if v.get("auroc_seq") is None:
            continue
        cand.append((v["auroc_seq"], (float(k) if k != "raw" else "raw"), "butter10"))
    for k, v in d.get("E1b_filter", {}).items():
        if v.get("auroc_seq") is None or k not in valid:
            continue
        cand.append((v["auroc_seq"], True, k))
    if not cand:
        return {"resample": 60.0, "filt": "butter10"}
    cand.sort(key=lambda x: -x[0])
    if isinstance(cand[0][2], str):
        return {"resample": 60.0, "filt": cand[0][2]}
    return {"resample": cand[0][2], "filt": "butter10"}


def build_factory(name, cfg, seed, epochs):
    """Return a () -> algorithm-instance factory (module-level function + partial, so joblib can pickle it)."""
    fs = cfg["resample"] if cfg["resample"] != "raw" else 62.5
    if name == "a1":
        from rule_algorithm import RuleAlgorithm
        return functools.partial(RuleAlgorithm, fs=float(fs))
    if name == "a2":
        from ml_algorithm import MLAlgorithm
        return functools.partial(MLAlgorithm, fs=float(fs), seed=seed)
    if name == "a3":
        from temporal_algorithm import TemporalAlgorithm
        return functools.partial(TemporalAlgorithm, fs=float(fs), seed=seed, epochs=epochs)
    if name == "a4":
        from hybrid_algorithm import HybridAlgorithm
        return functools.partial(HybridAlgorithm, fs=float(fs), seed=seed, epochs=epochs)
    raise ValueError(name)


def keys_of(res):
    return {
        "auroc_seq": res["auroc_seq"],
        "win_acc": res["window"]["acc"], "win_recall": res["window"]["recall"],
        "win_f1": res["window"]["f1"], "win_auc": res["window"]["auroc"],
        "seq_acc": res["sequence"]["acc"], "seq_recall": res["sequence"]["recall"],
        "seq_prec": res["sequence"]["precision"], "seq_f1": res["sequence"]["f1"],
        "ev_acc": res["event"]["event_accuracy"], "ev_recall": res["event"]["event_recall"],
        "ev_prec": res["event"]["event_precision"], "ev_f1": res["event"]["event_f1"],
        "delay_s": res["delay_mean_s"], "fp_per_hour": res["fp_per_hour"],
        "thr_mean": res["thr_mean"],
    }


# ---------------------------------------------------------------------------
# Inside A3: per-network evaluation
# ---------------------------------------------------------------------------
def a3_per_net(cfg, seed=0, epochs=20):
    from temporal_algorithm import TemporalAlgorithm
    sm = dataset.load_all(cfg)
    folds = splits.stratified_folds(5, seed)
    fs = cfg["resample"] if cfg["resample"] != "raw" else 62.5
    acc = {}
    for fi, te in enumerate(folds):
        tr = [s for s in sm if s not in set(te)]
        d = dataset.pack(sm, tr)
        alg = TemporalAlgorithm(fs=float(fs), seed=seed, epochs=epochs)
        alg.fit(d)
        nets = list(alg.nets)
        seqs = {k: [[], []] for k in nets}          # k -> [scores, labels]
        vot = [[], []]
        for s in te:
            dd = sm[s]
            if len(dd["wins"]) == 0:
                continue
            pn = alg.predict_per_net(dd["wins"])
            for k in nets:
                seqs[k][0].append(float(np.max(pn[k])))
                seqs[k][1].append(1 if dd["has_fall"] else 0)
            pv = np.mean([pn[k] for k in nets], axis=0)
            vot[0].append(float(np.max(pv)))
            vot[1].append(1 if dd["has_fall"] else 0)
        for k in nets + ["vote"]:
            src = vot if k == "vote" else seqs[k]
            acc.setdefault(k, []).append(M.auc_score(src[0], src[1]))
        print(f"  A3 internals: fold{fi+1}/{len(folds)} done", flush=True)
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Inside A4: base-learner OOF and fuser selection
# ---------------------------------------------------------------------------
def a4_detail(cfg, seed=0, epochs=20):
    from hybrid_algorithm import HybridAlgorithm
    from sklearn.model_selection import GroupKFold
    sm = dataset.load_all(cfg)
    d = dataset.pack(sm, list(sm.keys()))
    fs = cfg["resample"] if cfg["resample"] != "raw" else 62.5
    alg = HybridAlgorithm(fs=float(fs), seed=seed, epochs=epochs)
    fold = splits.stratified_folds(5, seed)[0]
    tr = [s for s in sm if s not in set(fold)]
    dtr = dataset.pack(sm, tr)
    alg.fit(dtr)
    y = dtr["labels"]
    base_auc = {n: float(M.auc_score(alg.oof_[:, j], y))
                for j, n in enumerate(alg.oof_names_)}
    mean_auc = float(M.auc_score(alg.oof_.mean(1), y))
    return {"base_oof_auroc": base_auc, "oof_mean_auroc": mean_auc,
            "fusion_cv": alg.fusion_cv_, "selected": alg.meta_kind_}


# ---------------------------------------------------------------------------
# Inside A1: sub-score ablation
# ---------------------------------------------------------------------------
def a1_subscores(cfg):
    from rule_algorithm import RuleAlgorithm, window_sub_scores
    sm = dataset.load_all(cfg)
    X, y = [], []
    for s, d in sm.items():
        if len(d["wins"]) == 0:
            continue
        X.append(np.array([window_sub_scores(w) for w in d["wins"]]))
        y.append(windowize.window_labels(d["t0_s"], d["intervals"], cfg["win_s"]))
    X = np.concatenate(X); y = np.concatenate(y)
    out = {name: float(M.auc_score(X[:, j], y))
           for j, name in enumerate(("impact", "active", "calm"))}
    out["combined(0.5/0.2/0.3)"] = float(M.auc_score(
        X @ np.array([0.50, 0.20, 0.30]), y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algs", default="a1,a2,a3,a4")
    ap.add_argument("--seeds", default="2026,2027,2028")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--resample", default=None)
    ap.add_argument("--filt", default=None)
    ap.add_argument("--skip-extra", action="store_true")
    a = ap.parse_args()

    c0 = cfg_from_exp01()
    if a.resample is not None:
        c0["resample"] = float(a.resample) if a.resample != "raw" else "raw"
    if a.filt is not None:
        c0["filt"] = a.filt
    cfg = {"resample": c0["resample"], "filt": c0["filt"],
           "win_s": windowize.WIN_S, "step_s": windowize.STEP_S}
    seeds = [int(s) for s in a.seeds.split(",")]
    algs = [x.strip() for x in a.algs.split(",") if x.strip()]
    print(f"preprocessing config: {cfg}\nseeds: {seeds}\nalgs: {algs}\n", flush=True)

    out = {"cfg": cfg, "seeds": seeds, "algorithms": {}, "extra": {}}

    for name in algs:
        t0 = time.time()
        per_seed = {}
        for sd in seeds:
            res = evaluate.run_cv(build_factory(name, cfg, sd, a.epochs), cfg, n_jobs=a.jobs)
            per_seed[str(sd)] = keys_of(res)
            print(f"  {name} seed={sd} seq_AUROC={res['auroc_seq']:.4f} "
                  f"win_recall={res['window']['recall']:.3f} event_recall={res['event']['event_recall']:.3f}",
                  flush=True)
        ks = list(per_seed[str(seeds[0])].keys())
        summ = {}
        for k in ks:
            vals = [per_seed[str(s)][k] for s in seeds]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            summ[k] = {"mean": float(np.mean(vals)) if vals else None,
                       "std": float(np.std(vals)) if vals else None}
        out["algorithms"][name] = {"label": ALG_NAMES[name], "per_seed": per_seed,
                                   "summary": summ,
                                   "elapsed_s": round(time.time() - t0, 1)}
        print(f"  ==> {name} seq_AUROC = {summ['auroc_seq']['mean']:.4f} "
              f"+- {summ['auroc_seq']['std']:.4f}  ({time.time()-t0:.0f}s)\n", flush=True)

    if not a.skip_extra:
        print("=== A1 sub-score ablation (window-level AUROC, pool-wide, descriptive) ===", flush=True)
        out["extra"]["a1_subscores"] = a1_subscores(cfg)
        for k, v in out["extra"]["a1_subscores"].items():
            print(f"  {k:20s} {v:.4f}")

        if "a3" in algs:
            print("\n=== A3 internal architecture comparison (sequence AUROC, seed=2026) ===", flush=True)
            out["extra"]["a3_per_net"] = a3_per_net(cfg, seed=seeds[0], epochs=a.epochs)
            for k, v in out["extra"]["a3_per_net"].items():
                print(f"  {k:8s} {v:.4f}")

        if "a4" in algs:
            print("\n=== A4 internals: base learners + fuser (fold1) ===", flush=True)
            out["extra"]["a4_detail"] = a4_detail(cfg, seed=seeds[0], epochs=a.epochs)
            for k, v in out["extra"]["a4_detail"]["base_oof_auroc"].items():
                print(f"  base {k:8s} OOF AUROC={v:.4f}")
            print(f"  fusion_cv={out['extra']['a4_detail']['fusion_cv']} "
                  f"-> selected {out['extra']['a4_detail']['selected']}")

    p = os.path.join(ROOT, "results", "exp03_alg.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nresults -> {p}")

    print("\n===== summary (mean+-std, %d-seed stratified 5-fold) =====" % len(seeds))
    for name in algs:
        s = out["algorithms"][name]["summary"]
        print(f"{ALG_NAMES[name]:20s} AUROC={s['auroc_seq']['mean']:.4f}+-{s['auroc_seq']['std']:.4f}  "
              f"win_Rec={s['win_recall']['mean']:.3f} seq_Rec={s['seq_recall']['mean']:.3f}  "
              f"ev_Rec={s['ev_recall']['mean']:.3f} ev_Acc={s['ev_acc']['mean']:.3f}")


if __name__ == "__main__":
    main()
