# -*- coding: utf-8 -*-
"""Classifier of the best combination: RF(400) + XGB(400) probability average.

Parameters identical to `04_alg_ml/src/ml_algorithm.py` (A2):
  RF  : n_estimators=400, max_depth=10, min_samples_leaf=2, class_weight='balanced'
  XGB : n_estimators=400, max_depth=5, lr=0.06, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0, scale_pos_weight=(negative/positive)
The **arithmetic mean** of both predict_proba outputs is the window fall probability.

The model (the two fitted RF + XGB estimators) is serialized with joblib to model/best_model.pkl.
"""
import os
import json
import numpy as np

from . import config as C
from . import features as F


class BestFallModel:
    """A2 best combination: RF + XGB probability average."""

    name = "A2_RF+XGB_avg"

    def __init__(self, fs=C.RESAMPLE_FS, seed=C.TRAIN_SEED):
        self.fs = float(fs)
        self.seed = int(seed)
        self.models = {}
        self.meta = {}

    # ---------------------------------------------------------------- training
    def _build(self, spw):
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        m = {}
        rf = dict(C.RF_PARAMS); rf.update(random_state=self.seed, n_jobs=-1)
        m["rf"] = RandomForestClassifier(**rf)
        xg = dict(C.XGB_PARAMS); xg.update(scale_pos_weight=spw, random_state=self.seed,
                                           n_jobs=-1, eval_metric="logloss", verbosity=0)
        m["xgb"] = XGBClassifier(**xg)
        return m

    def set_inference_jobs(self, n_jobs=C.INFER_JOBS):
        """Fix the number of inference threads.

        Streaming detection evaluates **one** window at a time, and `n_jobs=-1` would only create / synchronize a thread
        pool for a single sample (measured about 100 ms per window, mostly scheduling; the real work is 22 ms for the
        400 RF trees + 0.4 ms for XGB). With one thread a window takes about 23 ms, and since predictions are deterministic **the results are bit-identical**.
        """
        for m in self.models.values():
            try:
                if hasattr(m, "n_jobs"):
                    m.n_jobs = n_jobs
                if hasattr(m, "nthread"):
                    m.nthread = n_jobs
                booster = getattr(m, "get_booster", None)
                if callable(booster):
                    booster().set_param({"nthread": int(n_jobs)})
            except Exception:                                  # pragma: no cover
                pass
        self.infer_n_jobs = int(n_jobs)
        return self

    def fit(self, wins, labels, fs=None):
        """wins [N,win_n,4], labels [N] -> self."""
        if fs is not None:
            self.fs = float(fs)
        y = np.asarray(labels, np.int64)
        X = F.extract(wins, self.fs)
        spw = max((y == 0).sum() / max((y == 1).sum(), 1), 1.0)
        self.models = self._build(spw)
        for m in self.models.values():
            m.fit(X, y)
        self.set_inference_jobs()          # training uses all cores, inference goes back to one thread
        self.meta = {
            "algorithm": self.name,
            "fs": self.fs,
            "seed": self.seed,
            "n_features": int(X.shape[1]),
            "feature_groups": list(C.FEATURE_GROUPS),
            "feature_names": F.feature_names(),
            "n_train_windows": int(len(y)),
            "pos_ratio": float((y == 1).mean()) if len(y) else 0.0,
            "scale_pos_weight": float(spw),
            "config": {
                "resample_fs": C.RESAMPLE_FS, "filter": C.FILTER_METHOD,
                "butter_order": C.BUTTER_ORDER, "butter_cutoff": C.BUTTER_CUTOFF,
                "win_s": C.WIN_S, "step_s": C.STEP_S, "merge_gap_win": C.MERGE_GAP_WIN,
                "rf": C.RF_PARAMS, "xgb": C.XGB_PARAMS,
            },
        }
        return self

    # ---------------------------------------------------------------- inference
    def predict_proba(self, wins):
        """wins [N,win_n,4] -> mean fall probability [N]. Empty input returns an empty array."""
        if len(wins) == 0 or not self.models:
            return np.zeros(len(wins), np.float64)
        X = F.extract(wins, self.fs)
        probs = [m.predict_proba(X)[:, 1] for m in self.models.values()]
        return np.mean(probs, axis=0)

    # ------------------------------------------------------------ persistence
    def save(self, path):
        import joblib
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"models": self.models, "meta": self.meta}, path)

    @classmethod
    def load(cls, path):
        import joblib
        blob = joblib.load(path)
        obj = cls(fs=blob.get("meta", {}).get("fs", C.RESAMPLE_FS))
        obj.models = blob["models"]
        obj.meta = blob.get("meta", {})
        obj.set_inference_jobs()           # inference threads are fixed on load
        return obj


def feature_importance(model):
    """Average of the normalized RF and XGB importances, returned as [(name, value)] in descending order."""
    names = F.feature_names()
    imp = []
    if "rf" in model.models:
        imp.append(np.asarray(model.models["rf"].feature_importances_, np.float64))
    if "xgb" in model.models:
        imp.append(np.asarray(model.models["xgb"].feature_importances_, np.float64))
    if not imp:
        return []
    v = np.mean(imp, axis=0)
    v = v / (v.sum() + 1e-12)
    return sorted(zip(names, v.tolist()), key=lambda kv: -kv[1])
