# -*- coding: utf-8 -*-
"""A2 kinematic features + classical machine learning.

Idea: compress each sliding window into a set of statistical features (feature
groups defined in 00_common/features.py) and learn the decision boundary with two
complementary tree models, then average their probabilities:

  RF    random forest (bagging: many trees vote; robust to overfitting, small-sample friendly)
  XGB   gradient-boosted trees (boosting: fit residuals sequentially; usually strongest on tabular data)

Class imbalance: RF uses class_weight='balanced', XGB uses scale_pos_weight.
(URFD has ~17% positive windows at window level; without explicit handling the model
leans to the majority class and recall collapses.)

Feature selection is fitted INSIDE the training fold (leak-free); predict reuses the
same feature index set.

Interface matches A1/A3/A4: fit(train) / predict(wins).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "00_common"))

import features as F


def extract_window_features(wins, fs, groups=F.DEFAULT_GROUPS):
    """Feature entry point reused by A4."""
    return F.extract(wins, fs, groups)


class MLAlgorithm:
    """A2: handcrafted features + RF/XGBoost (feature groups and selection are configurable)."""

    name = "A2_feature_ml"

    def __init__(self, fs=60.0, seed=0, groups=F.DEFAULT_GROUPS,
                 selection="all", k=20, use=("rf", "xgb")):
        self.fs = fs
        self.seed = seed
        self.groups = tuple(groups)
        self.selection = selection
        self.k = k
        self.use = tuple(use)
        self.idx = None
        self.models = {}

    # -- internals -------------------------------------------------------
    def _raw_features(self, wins):
        return F.extract(wins, self.fs, self.groups)

    def _make_models(self, spw=1.0):
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        m = {}
        if "rf" in self.use:
            m["rf"] = RandomForestClassifier(
                n_estimators=400, max_depth=10, min_samples_leaf=2,
                class_weight="balanced", random_state=self.seed, n_jobs=F.tree_jobs())
        if "xgb" in self.use:
            m["xgb"] = XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.06,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                scale_pos_weight=spw, random_state=self.seed, n_jobs=F.tree_jobs(),
                eval_metric="logloss", verbosity=0)
        return m

    # -- interface -------------------------------------------------------
    def fit(self, train):
        self.fs = train.get("fs", self.fs)
        X = self._raw_features(train["wins"])
        y = np.asarray(train["labels"], np.int64)
        X, self.idx, _ = F.select_features(X, y, self.selection, self.k, self.seed)
        self._Xtr_mean, self._Xtr_std = X.mean(0), X.std(0) + 1e-9
        spw = max((y == 0).sum() / max((y == 1).sum(), 1), 1.0)
        self.models = self._make_models(spw)
        for m in self.models.values():
            m.fit(X, y)
        return self

    def predict(self, wins):
        if len(self.models) == 0 or len(wins) == 0:
            return np.zeros(len(wins), np.float64)
        X = self._raw_features(wins)
        if self.idx is not None:
            X = X[:, self.idx]
        probs = [m.predict_proba(X)[:, 1] for m in self.models.values()]
        return np.mean(probs, axis=0)

    def feature_importance(self):
        """Return {feature name: importance} (averaged over normalised RF and XGB), for Experiment 2 analysis."""
        names = F.feature_names(self.groups)
        if self.idx is not None:
            names = [names[i] for i in self.idx]
        imp = []
        if "rf" in self.models:
            imp.append(self.models["rf"].feature_importances_)
        if "xgb" in self.models:
            imp.append(self.models["xgb"].feature_importances_)
        if not imp:
            return {}
        v = np.mean(imp, axis=0)
        v = v / (v.sum() + 1e-12)
        return dict(zip(names, v.tolist()))
