"""Trainable fall models: training, calibration, threshold selection, packaging and
the window scorers that plug trained models into the streaming channels.

Model packages live in ``models/<task>/<model_id>/<version>/`` with ``meta.json``.
A package records its feature schema, window parameters, layout, calibration,
operating thresholds (chosen on validation data only) and training provenance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from homeease.core.config import AppConfig
from homeease.core.paths import models_dir
from homeease.core.util import code_version, dumps, sha256_file
from homeease.ml.fall.accel import AccelRuleScorer, AccelWindow, ScoreResult, feature_vector, quality_of, window_features
from homeease.ml.fall.fusion import TrainedFusionModel
from homeease.ml.fall.vision import VisionRuleScorer, VisionWindow
from homeease.ml.fall.vision import quality_of as vquality
from homeease.schemas.evidence import ScoreSemantics


# ------------------------------------------------------------ sklearn models
def make_sklearn(kind: str, seed: int = 0) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=seed))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    raise ValueError(kind)


# ----------------------------------------------------------- calibration
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any] | None:
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y_val)) < 2 or len(y_val) < 20:
        return None
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(logit(p_val).reshape(-1, 1), y_val)
    return {"method": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]), "n": int(len(y_val)), "positives": int(y_val.sum())}


def apply_platt(p: np.ndarray, cal: dict[str, Any] | None) -> np.ndarray:
    if not cal:
        return p
    z = cal["a"] * logit(np.asarray(p)) + cal["b"]
    return 1.0 / (1.0 + np.exp(-z))


def choose_thresholds(p_val: np.ndarray, y_val: np.ndarray, max_fpr_candidate: float = 0.02, max_fpr_strong: float = 0.005) -> dict[str, Any]:
    """Lowest thresholds whose validation window-FPR stays within the constraint
    ("maximise recall under a false-alarm constraint", slide 14)."""
    neg = p_val[y_val == 0]
    pos = p_val[y_val == 1]
    out: dict[str, Any] = {"rule": f"min threshold with validation window FPR <= {max_fpr_candidate} (candidate) / {max_fpr_strong} (strong)"}
    for name, lim in (("candidate", max_fpr_candidate), ("strong", max_fpr_strong)):
        if len(neg) == 0:
            th = 0.5
        else:
            th = float(np.quantile(neg, 1 - lim)) + 1e-6 if len(neg) > 1 / max(lim, 1e-9) else float(neg.max()) + 1e-6
        th = float(min(max(th, 0.05), 0.995))
        out[name] = th
        out[f"{name}_val_recall"] = float((pos >= th).mean()) if len(pos) else None
        out[f"{name}_val_fpr"] = float((neg >= th).mean()) if len(neg) else None
    out["strong"] = max(out["strong"], out["candidate"])
    return out


# ------------------------------------------------------------- packaging
def save_package(task: str, model_id: str, model: Any, meta: dict[str, Any], kind: str, version: str | None = None) -> Path:
    version = version or time.strftime("%Y%m%d-%H%M%S")
    d = models_dir() / task / model_id / version
    d.mkdir(parents=True, exist_ok=True)
    if kind == "sklearn":
        wp = d / "model.joblib"
        joblib.dump(model, wp)
    elif kind == "torch":
        import torch

        wp = d / "model.pt"
        torch.save({"state_dict": model.state_dict(), "cfg": model.cfg}, wp)
    elif kind == "linear":
        wp = d / "model.json"
        wp.write_text(json.dumps(model), encoding="utf-8")
    else:
        raise ValueError(kind)
    meta = {**meta, "task": task, "model_id": model_id, "version": version, "kind": kind, "weights": wp.name, "sha256": sha256_file(wp), "code_version": code_version(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (d / "meta.json").write_text(dumps(meta, indent=2), encoding="utf-8")
    return d


def load_meta(path: Path) -> dict[str, Any]:
    return json.loads((Path(path) / "meta.json").read_text(encoding="utf-8"))


def load_model(path: Path) -> tuple[Any, dict[str, Any]]:
    path = Path(path)
    meta = load_meta(path)
    wp = path / meta["weights"]
    if sha256_file(wp) != meta["sha256"]:
        raise RuntimeError(f"weights checksum mismatch for {path}")
    if meta["kind"] == "sklearn":
        return joblib.load(wp), meta
    if meta["kind"] == "torch":
        import torch

        from homeease.ml.common.torch_models import build

        ck = torch.load(wp, map_location="cpu", weights_only=True)
        m = build(ck["cfg"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        return m, meta
    if meta["kind"] == "linear":
        return json.loads(wp.read_text(encoding="utf-8")), meta
    raise ValueError(meta["kind"])


def predict_any(model: Any, kind: str, X: np.ndarray) -> np.ndarray:
    if kind == "sklearn":
        return model.predict_proba(X)[:, 1]
    from homeease.ml.common.torch_models import predict

    return predict(model, X)


# --------------------------------------------------------------- scorers
class AccelModelScorer:
    """Trained accelerometer window model inside the streaming AccelChannel."""

    channel = "accel"

    def __init__(self, model: Any, meta: dict[str, Any], cfg: AppConfig) -> None:
        if meta["task"] != "fall_accel":
            raise ValueError(f"package task {meta['task']} is not fall_accel")
        inp = meta["input"]
        if not inp.get("includes_gravity", True):
            raise ValueError("package expects linear acceleration; live collectors send accelerationIncludingGravity")
        self.model, self.meta, self.kind = model, meta, meta["kind"]
        self.model_id, self.model_version = meta["model_id"], meta["version"]
        self.calibration = meta.get("calibration")
        self.calibration_id = self.calibration.get("id") if self.calibration else None
        self.score_semantics = ScoreSemantics.calibrated_probability if self.calibration else ScoreSemantics.uncalibrated_probability
        th = meta["thresholds"]
        self.candidate_threshold, self.strong_threshold = float(th["candidate"]), float(th["strong"])
        self.window_ms = float(inp["window_ms"])
        self.step_ms = float(inp.get("step_ms", 250.0))
        self.post_min_s = float(inp.get("post_min_s", 0.5))
        self.rule = AccelRuleScorer(cfg.fall.accel, cfg.acquisition.accel_target_hz, cfg.quality.max_clock_uncertainty_ms, cfg.quality.saturation_g)
        self.feature_names = meta.get("feature_schema", {}).get("names")
        self.cfg = cfg

    def _x(self, w: AccelWindow) -> np.ndarray | None:
        if self.meta["input"]["representation"] == "features":
            c = self.cfg.fall.accel
            f = window_features(w, c.post_start_s, self.post_min_s, c.post_window_s, saturation_g=self.cfg.quality.saturation_g)
            if not f.get("valid"):
                return None
            return np.nan_to_num(feature_vector(f, self.feature_names)).reshape(1, -1)
        from homeease.ml.fall.windows import accel_seq

        return accel_seq(w, int(self.window_ms / 20.0))[None]

    def score(self, w: AccelWindow, peak_search: tuple[float, float] | None = None, truncated: bool = False) -> ScoreResult:
        base = self.rule.score(w, peak_search, truncated)
        if len(w.t) < 10 or w.mask.mean() < 0.5:
            return ScoreResult(False, 0.0, False, None, base.features, ["insufficient_samples"], 0.0, base.flags)
        x = self._x(w)
        if x is None:
            return ScoreResult(False, 0.0, False, None, base.features, base.reasons, base.quality, base.flags)
        p = float(apply_platt(predict_any(self.model, self.kind, x), self.calibration)[0])
        feats = dict(base.features)
        feats["p_model"] = p
        cand = p >= self.candidate_threshold
        reasons = ["model_positive" if cand else "model_negative"] + [r for r in base.reasons if r in ("impact_detected", "post_impact_low_activity", "truncated_post_window")]
        q, flags = quality_of(w, base.features if base.valid else {}, self.cfg.acquisition.accel_target_hz, self.cfg.quality.max_clock_uncertainty_ms)
        return ScoreResult(True, p, cand, base.peak_time_ms if base.valid else float(w.t[-1]) - 1500.0, feats, reasons, q, flags)


class VisionModelScorer:
    channel = "vision"

    def __init__(self, model: Any, meta: dict[str, Any], cfg: AppConfig) -> None:
        if meta["task"] != "fall_vision":
            raise ValueError(f"package task {meta['task']} is not fall_vision")
        if meta["input"].get("layout") != "coco17-v1":
            raise ValueError("keypoint layout mismatch: package was trained on a different joint layout")
        self.model, self.meta, self.kind = model, meta, meta["kind"]
        self.model_id, self.model_version = meta["model_id"], meta["version"]
        self.calibration = meta.get("calibration")
        self.calibration_id = self.calibration.get("id") if self.calibration else None
        self.score_semantics = ScoreSemantics.calibrated_probability if self.calibration else ScoreSemantics.uncalibrated_probability
        th = meta["thresholds"]
        self.candidate_threshold, self.strong_threshold = float(th["candidate"]), float(th["strong"])
        self.window_ms = float(meta["input"]["window_ms"])
        self.step_ms = float(meta["input"].get("step_ms", 200.0))
        self.fps = float(meta["input"].get("fps", 15.0))
        self.rule = VisionRuleScorer(cfg.fall.vision)
        self.cfg = cfg
        self.feature_names = meta.get("feature_schema", {}).get("names")

    def _x(self, w: VisionWindow) -> np.ndarray:
        from homeease.ml.fall.windows import vision_features, vision_seq

        if self.meta["input"]["representation"] == "features":
            return vision_features(w, self.cfg, self.feature_names).reshape(1, -1)
        return vision_seq(w, int(round(self.window_ms / 1000.0 * self.fps)))[None]

    def score(self, w: VisionWindow, descent_search: tuple[float, float] | None = None, truncated: bool = False) -> ScoreResult:
        base = self.rule.score(w, descent_search, truncated)
        q, flags = vquality(w)
        if w.mask.mean() < 0.2:
            return ScoreResult(False, 0.0, False, None, base.features, ["person_not_visible"], q, flags)
        p = float(apply_platt(predict_any(self.model, self.kind, self._x(w)), self.calibration)[0])
        feats = dict(base.features)
        feats["p_model"] = p
        cand = p >= self.candidate_threshold
        peak = base.peak_time_ms
        if peak is None and cand:
            peak = float(w.t[-1]) - 1000.0
        reasons = ["model_positive" if cand else "model_negative"] + [r for r in base.reasons if r in ("rapid_descent", "persistent_low_posture", "recovered_to_standing", "target_lost_after_drop")]
        return ScoreResult(True, p, cand, peak, feats, reasons, q, flags)

    def assess_contradiction(self, w: VisionWindow, t_ref: float) -> ScoreResult:
        res = self.rule.assess_contradiction(w, t_ref)
        m = self.score(w, descent_search=(t_ref - 2000.0, t_ref + 1500.0))
        if m.candidate:
            return m
        m.contradicts = res.contradicts
        m.reasons = sorted(set(m.reasons + [r for r in res.reasons if r in ("target_upright_no_descent", "recovered_to_standing", "person_not_visible", "target_ambiguous")]))
        if not res.valid:
            m.valid = False
        return m


def load_window_scorer(path: Path, cfg: AppConfig) -> Any:
    model, meta = load_model(path)
    if meta["task"] == "fall_accel":
        return AccelModelScorer(model, meta, cfg)
    if meta["task"] == "fall_vision":
        return VisionModelScorer(model, meta, cfg)
    raise ValueError(f"not a window scorer package: {meta['task']}")


def load_trained_fusion(path: Path) -> TrainedFusionModel:
    model, meta = load_model(path)
    if meta["task"] != "fall_fusion":
        raise ValueError("not a late-fusion package")
    return TrainedFusionModel(model["coef"], model["intercept"], meta["thresholds"]["strong"], meta["thresholds"]["candidate"], meta["model_id"], meta["version"])
