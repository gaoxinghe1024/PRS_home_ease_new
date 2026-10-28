# -*- coding: utf-8 -*-
"""A4 meta-learner diagnostic (scratch script): compare fusion strategies on the same base learners."""
import sys, os, numpy as np, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "00_common")
sys.path.insert(0, "06_alg_hybrid/src")
import dataset, splits, metrics as M
from hybrid_algorithm import HybridAlgorithm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 3
folds = splits.stratified_folds()
sm = dataset.load_all()

meta_defs = {
    "mean7": None,
    "LR_C1": Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))]),
    "LR_C0.2": Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.2, max_iter=3000, class_weight="balanced"))]),
    "XGB_small": XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.8,
                               min_child_weight=5, reg_lambda=5.0, random_state=0, n_jobs=-1,
                               eval_metric="logloss", verbosity=0),
    "XGB_tiny": XGBClassifier(n_estimators=40, max_depth=2, learning_rate=0.08, subsample=0.8,
                              min_child_weight=8, reg_lambda=8.0, random_state=0, n_jobs=-1,
                              eval_metric="logloss", verbosity=0),
}

acc = {k: {"oof": [], "test": []} for k in meta_defs}
base_oof_auc, names = [], []
t0 = time.time()
for fi, te in enumerate(folds[:NF]):
    tr = [s for s in sm if s not in set(te)]
    d = dataset.pack(sm, tr)
    alg = HybridAlgorithm(fs=60.0, seed=0, epochs=EPOCHS)
    alg.fit(d)
    oof, names = alg.oof_, alg.oof_names_
    y = d["labels"]
    base_oof_auc.append([M.auc_score(oof[:, j], y) for j in range(oof.shape[1])])

    test_seqs = [s for s in te if len(sm[s]["wins"]) > 0]
    Mte = np.stack([np.concatenate([alg.bases[j].predict_proba(sm[s]["wins"]) for s in test_seqs])
                    for j in range(oof.shape[1])], axis=1)
    seq_te = np.concatenate([np.array([s] * len(sm[s]["wins"])) for s in test_seqs])

    for k, md in meta_defs.items():
        if md is None:
            poof, pte = oof.mean(1), Mte.mean(1)
        else:
            md.fit(oof, y)
            poof, pte = md.predict_proba(oof)[:, 1], md.predict_proba(Mte)[:, 1]
        acc[k]["oof"].append(M.auc_score(poof, y))
        ss, sl = [], []
        for s in test_seqs:
            m = seq_te == s
            if m.sum():
                ss.append(float(pte[m].max())); sl.append(1 if sm[s]["has_fall"] else 0)
        acc[k]["test"].append(M.auc_score(ss, sl))
    print(f"fold{fi} done {time.time()-t0:.0f}s", flush=True)

print(f"\n=== A4 fusion strategy comparison (epochs={EPOCHS}, first {NF} folds) ===")
print("base learner OOF AUROC: " + ", ".join(
    f"{n}={np.mean([b[j] for b in base_oof_auc]):.3f}" for j, n in enumerate(names)))
for k in meta_defs:
    print(f"  {k:10s} OOF AUROC={np.mean(acc[k]['oof']):.4f}  test sequence AUROC={np.mean(acc[k]['test']):.4f}  "
          f"per-fold={[round(x,3) for x in acc[k]['test']]}")
