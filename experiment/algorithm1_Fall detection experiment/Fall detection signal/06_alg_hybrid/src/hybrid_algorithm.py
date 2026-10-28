# -*- coding: utf-8 -*-
"""A4 hybrid ensemble (proper stacking: heterogeneous base learners + OOF meta-features
+ a strong meta-learner).

Goal: fuse two complementary representations with *proper* ensemble learning rather
than a plain average.

1) Heterogeneous base-learner pool (7 learners, two complementary representation paths)
   Feature path (handcrafted / statistical representation):
       RF   -- random forest
       XGB  -- gradient-boosted trees
       LR   -- logistic regression (linear, adds diversity)
   Temporal path (raw waveform representation):
       LSTM / GRU / BiLSTM / TCN -- read the [120,4] waveform directly

2) Proper stacking
   - Base-learner meta-features are generated out-of-fold (OOF); the inner split is
     K-fold **grouped by sequence** (GroupKFold), so windows from one sequence can
     never sit on both the training and prediction side of an OOF estimate
     (that would leak).
   - Meta-learner: XGBoost (learns the non-linear optimal combination of 7 evidence streams)

3) Predict: base learners are retrained on the full training set -> 7-dim meta-features
   -> the meta-learner outputs the final probability.

Interface matches A1/A2/A3: fit(train) / predict(wins).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "..", "00_common"))
sys.path.insert(0, COMMON)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "04_alg_ml", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "05_alg_temporal", "src")))

import features as F
from ml_algorithm import extract_window_features
from temporal_algorithm import _TorchModel, MODEL_NAMES

FEAT_BASES = ("rf", "xgb", "lr")
TEMP_BASES = MODEL_NAMES


# ---------------------------------------------------------------------------
# Base-learner wrappers (uniform fit(wins, labels) / predict_proba(wins))
# ---------------------------------------------------------------------------
class _FeatBase:
    def __init__(self, kind, fs, seed, groups, selection, k):
        self.kind, self.fs, self.seed = kind, fs, seed
        self.groups, self.selection, self.k = groups, selection, k
        self.idx = None
        self.model = None

    def _raw(self, wins):
        return extract_window_features(wins, self.fs, self.groups)

    def fit(self, wins, y):
        X = self._raw(wins)
        X, self.idx, _ = F.select_features(X, y, self.selection, self.k, self.seed)
        if self.kind == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=400, max_depth=10, min_samples_leaf=2,
                class_weight="balanced", random_state=self.seed, n_jobs=F.tree_jobs())
        elif self.kind == "xgb":
            from xgboost import XGBClassifier
            spw = max((y == 0).sum() / max((y == 1).sum(), 1), 1.0)
            self.model = XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.9,
                colsample_bytree=0.9, reg_lambda=1.0, scale_pos_weight=spw,
                random_state=self.seed, n_jobs=F.tree_jobs(), eval_metric="logloss", verbosity=0)
        elif self.kind == "lr":
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            self.model = Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(C=1.0, max_iter=3000,
                                          class_weight="balanced"))])
        self.model.fit(X, y)
        return self

    def predict_proba(self, wins):
        if self.model is None:
            return np.zeros(len(wins), np.float64)
        X = self._raw(wins)
        if self.idx is not None:
            X = X[:, self.idx]
        return self.model.predict_proba(X)[:, 1]


class _TempBase:
    def __init__(self, key, seed, epochs):
        self.m = _TorchModel(key, seed=seed, epochs=epochs)

    def fit(self, wins, y):
        self.m.fit(wins, y)
        return self

    def predict_proba(self, wins):
        return self.m.predict_proba(wins)


class HybridAlgorithm:
    """A4: 7 base learners + a fusion layer (the fuser is chosen automatically by inner CV).

    Fusion candidates:
      mean  -- heterogeneous soft-voting average (parameter-free, most stable)
      lr    -- L2 logistic regression (linear fusion)
      xgb   -- regularised XGBoost (non-linear fusion, the meta-learner)
    Selection rule: whichever maximises sequence-grouped inner-CV AUROC on the
    meta-feature matrix.
    """

    name = "A4_hybrid_stacking"

    FUSIONS = ("mean", "lr", "xgb")

    def __init__(self, fs=60.0, seed=0, n_inner=3, groups=F.DEFAULT_GROUPS,
                 selection="all", k=20, feat_bases=FEAT_BASES, temp_bases=TEMP_BASES,
                 epochs=20, fusion="auto"):
        self.fs = fs
        self.seed = seed
        self.n_inner = n_inner
        self.groups = tuple(groups)
        self.selection = selection
        self.k = k
        self.feat_bases = tuple(feat_bases)
        self.temp_bases = tuple(temp_bases)
        self.epochs = epochs
        self.fusion_kind = fusion
        self.bases = []
        self.meta = None
        self.meta_kind_ = None
        self.fusion_cv_ = {}

    # -- base-learner construction ---------------------------------------
    def _new_bases(self):
        b = []
        for i, kind in enumerate(self.feat_bases):
            b.append(_FeatBase(kind, self.fs, self.seed + i, self.groups,
                               self.selection, self.k))
        for i, key in enumerate(self.temp_bases):
            b.append(_TempBase(key, seed=self.seed + 100 + 17 * i, epochs=self.epochs))
        return b

    def _make_meta(self, kind):
        if kind == "mean":
            return None
        if kind == "lr":
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            return Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(C=0.5, max_iter=3000,
                                                       class_weight="balanced"))])
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                             subsample=0.8, min_child_weight=5, reg_lambda=5.0,
                             random_state=self.seed, n_jobs=F.tree_jobs(),
                             eval_metric="logloss", verbosity=0)

    def _apply(self, kind, meta, M):
        if kind == "mean":
            return np.asarray(M).mean(1)
        return meta.predict_proba(M)[:, 1]

    # -- interface --------------------------------------------------------
    def fit(self, train):
        from sklearn.model_selection import GroupKFold
        import metrics as Mmet
        self.fs = train.get("fs", self.fs)
        wins = np.asarray(train["wins"], np.float32)
        y = np.asarray(train["labels"], np.int64)
        seq_id = np.asarray(train.get("seq_id", np.arange(len(y))))

        self.bases = self._new_bases()
        nb = len(self.bases)
        oof = np.zeros((len(y), nb), np.float64)

        # Inner OOF, grouped by sequence to prevent leakage
        n_groups = len(np.unique(seq_id))
        n_split = int(min(self.n_inner, max(n_groups, 2)))
        gkf = GroupKFold(n_splits=n_split)
        oof_parts = []
        for tr, va in gkf.split(wins, y, groups=seq_id):
            if len(np.unique(y[tr])) < 2 or len(va) == 0:
                continue
            for j, base in enumerate(self.bases):
                b = type(base)(**self._base_kwargs(base))
                b.fit(wins[tr], y[tr])
                oof[va, j] = b.predict_proba(wins[va])
            oof_parts.append((tr, va))
        self.oof_ = oof
        self.oof_names_ = list(self.feat_bases) + list(self.temp_bases)
        self.oof_folds_ = oof_parts

        # Fuser selection: grouped CV on the meta-features
        cand = self.FUSIONS if self.fusion_kind == "auto" else (self.fusion_kind,)
        self.fusion_cv_ = {}
        for kind in cand:
            if kind == "mean":
                scores = [Mmet.auc_score(oof[va].mean(1), y[va]) for _, va in oof_parts]
            else:
                # Fit on the remaining OOF folds, validate on the current one
                # (an approximation of nested CV)
                scores = []
                for tr, va in oof_parts:
                    if len(np.unique(y[tr])) < 2:
                        continue
                    md = self._make_meta(kind)
                    md.fit(oof[tr], y[tr])
                    scores.append(Mmet.auc_score(md.predict_proba(oof[va])[:, 1], y[va]))
            self.fusion_cv_[kind] = float(np.nanmean(scores)) if scores else float("nan")
        if self.fusion_kind == "auto":
            valid = {k2: v for k2, v in self.fusion_cv_.items() if v == v}
            self.meta_kind_ = max(valid, key=valid.get) if valid else "mean"
        else:
            self.meta_kind_ = self.fusion_kind

        # Retrain base learners on the full set; fit the chosen fuser on all OOF rows
        for base in self.bases:
            base.fit(wins, y)
        self.meta = self._make_meta(self.meta_kind_)
        if self.meta is not None and len(np.unique(y)) > 1:
            self.meta.fit(oof, y)
        return self

    def _base_kwargs(self, base):
        if isinstance(base, _FeatBase):
            return {"kind": base.kind, "fs": base.fs, "seed": base.seed,
                    "groups": base.groups, "selection": base.selection, "k": base.k}
        return {"key": base.m.key, "seed": base.m.seed, "epochs": base.m.epochs}

    def predict(self, wins):
        if len(wins) == 0:
            return np.zeros(len(wins), np.float64)
        if self.meta is None and self.meta_kind_ != "mean":
            return np.zeros(len(wins), np.float64)
        M = np.stack([b.predict_proba(wins) for b in self.bases], axis=1)
        return self._apply(self.meta_kind_, self.meta, M)

    def predict_oof_meta(self, train):
        """Return the OOF probability matrix of every base learner (used to discuss base-learner contributions)."""
        from sklearn.model_selection import GroupKFold
        wins = np.asarray(train["wins"], np.float32)
        y = np.asarray(train["labels"], np.int64)
        seq_id = np.asarray(train.get("seq_id", np.arange(len(y))))
        bases = self._new_bases()
        oof = np.zeros((len(y), len(bases)), np.float64)
        gkf = GroupKFold(n_splits=int(min(self.n_inner, max(len(np.unique(seq_id)), 2))))
        for tr, va in gkf.split(wins, y, groups=seq_id):
            if len(np.unique(y[tr])) < 2 or len(va) == 0:
                continue
            for j, base in enumerate(bases):
                b = type(base)(**self._base_kwargs(base))
                b.fit(wins[tr], y[tr])
                oof[va, j] = b.predict_proba(wins[va])
        return oof, list(self.feat_bases) + list(self.temp_bases)
