"""Trainable sit-to-stand phase classifiers (RF, LSTM, TCN) and soft voting.

Features are derived causally per frame by ExerciseFeatureExtractor (shared with the
live tracker). Session-state-dependent features (hip_rel, absolute hip height) are not
used by models so that training and live inference see identical inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from homeease.core.config import AppConfig
from homeease.datasets.recording import Recording
from homeease.edge.acquisition.replay import frame_to_obs, score_scale_for
from homeease.edge.tracking import TargetTracker
from homeease.ml.exercise.features import EX_FEATURES, PHASES, ExerciseFeatureExtractor
from homeease.ml.exercise.phase import ModelPhaseClassifier, SoftVotingPhaseClassifier

SEL_BASE = ["thigh_vertical", "knee_angle", "torso_deg"]
SEL_TEMPORAL = SEL_BASE + ["hip_vel_tl_s", "knee_vel_tl_s"]
SCALE = {"thigh_vertical": 1.0, "knee_angle": 1 / 180.0, "torso_deg": 1 / 90.0, "hip_vel_tl_s": 1 / 2.0, "knee_vel_tl_s": 1 / 2.0, "hip_rel": 1.0, "hip_y_tl": 0.1}
SEQ_T = 30
RF_CONTEXT = 8


def project(x: np.ndarray, m: np.ndarray, sel: list[str]) -> np.ndarray:
    """(T, len(EX_FEATURES)) buffer -> (T, len(sel)+1) normalised with a mask channel."""
    idx = [EX_FEATURES.index(s) for s in sel]
    out = np.zeros((len(x), len(sel) + 1), dtype=np.float32)
    for j, (i, s) in enumerate(zip(idx, sel)):
        out[:, j] = np.clip(np.nan_to_num(x[:, i]) * SCALE[s], -5, 5)
    out[:, -1] = m
    out[m == 0, :-1] = 0.0
    return out


def rf_vector(seq: np.ndarray) -> np.ndarray:
    """Current frame + short causal context statistics."""
    ctx = seq[-RF_CONTEXT:]
    valid = ctx[:, -1] > 0
    cur = seq[-1, :-1]
    stats = []
    for j in range(seq.shape[1] - 1):
        v = ctx[valid, j] if valid.any() else np.zeros(1)
        stats += [float(v.mean()), float(v.min()), float(v.max())]
    return np.concatenate([cur, np.asarray(stats), [float(valid.mean())]]).astype(np.float32)


def phase_at(t: float, phases: list[dict[str, Any]]) -> str | None:
    hit = [p["phase"] for p in phases if p["start_ms"] <= t < p["end_ms"]]
    return hit[0] if len(set(hit)) == 1 else None


def extract(rec: Recording, cfg: AppConfig) -> list[tuple[float, dict[str, float] | None, str | None]]:
    fx = ExerciseFeatureExtractor(cfg.acquisition.pose.min_keypoint_score)
    tracker = TargetTracker(cfg.acquisition.zone, min_score=cfg.acquisition.pose.min_keypoint_score)
    phases = rec.labels().get("phases", [])
    out = []
    for f in rec.detections:
        obs = frame_to_obs(f, tracker, f.t_ms, subject_id="x", device_id="c", session_id="c", source_mode="replay", score_scale=score_scale_for(cfg, f.backend))
        feats = fx(obs) if obs.target_visible else None
        if obs.target_visible is False:
            fx.reset_motion()
        out.append((f.t_ms, feats, phase_at(f.t_ms, phases)))
    return out


def build_dataset(recs: list[Recording], cfg: AppConfig, sel: list[str]) -> dict[str, Any]:
    Xs, Xr, y, g = [], [], [], []
    for rec in recs:
        frames = extract(rec, cfg)
        buf = np.zeros((len(frames), len(EX_FEATURES)))
        mask = np.zeros(len(frames))
        for i, (_, feats, _) in enumerate(frames):
            if feats is not None:
                buf[i] = [feats.get(k, np.nan) for k in EX_FEATURES]
                mask[i] = 1.0
        proj = project(buf, mask, sel)
        for i, (_, feats, lab) in enumerate(frames):
            if lab is None or feats is None:
                continue
            lo = max(0, i - SEQ_T + 1)
            seq = proj[lo : i + 1]
            if len(seq) < SEQ_T:
                seq = np.vstack([np.zeros((SEQ_T - len(seq), proj.shape[1]), dtype=np.float32), seq])
            Xs.append(seq)
            Xr.append(rf_vector(seq))
            y.append(PHASES.index(lab))
            g.append(str(rec.manifest.get("group_id")))
    return {"X_seq": np.stack(Xs) if Xs else np.zeros((0, SEQ_T, len(sel) + 1)), "X_rf": np.stack(Xr) if Xr else np.zeros((0, 1)), "y": np.asarray(y, dtype=int), "groups": g}


class _Predictor:
    """Callable (x_buffer (T, 7), mask) -> class probabilities."""

    def __init__(self, model: Any, kind: str, sel: list[str]) -> None:
        self.model, self.kind, self.sel = model, kind, sel

    def __call__(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        seq = project(x, m, self.sel)
        if self.kind == "sklearn":
            p = self.model.predict_proba(rf_vector(seq)[None])[0]
            full = np.zeros(len(PHASES))
            full[self.model.classes_] = p
            return full
        import torch

        with torch.no_grad():
            return torch.softmax(self.model(torch.from_numpy(seq[None].astype(np.float32))), -1).numpy()[0]


def make_classifier(model: Any, kind: str, sel: list[str], model_id: str, version: str) -> ModelPhaseClassifier:
    return ModelPhaseClassifier(_Predictor(model, kind, sel), model_id, version, SEQ_T)


def load_phase_classifier(model_ref: str, cfg: AppConfig) -> Any:
    from homeease.ml.fall.models import load_model
    from homeease.ml.registry import find_package

    if model_ref.startswith("ensemble:"):
        members = [load_phase_classifier(r, cfg) for r in model_ref.split(":", 1)[1].split("+")]
        return SoftVotingPhaseClassifier(members, model_id="phase_ensemble", model_version="+".join(m.model_id for m in members))
    model, meta = load_model(find_package(model_ref, "exercise_phase"))
    if meta["input"].get("layout") != "coco17-v1":
        raise ValueError("phase model trained on a different keypoint layout")
    return make_classifier(model, meta["kind"], meta["feature_schema"]["selected"], meta["model_id"], meta["version"])


__all__ = ["SEL_BASE", "SEL_TEMPORAL", "build_dataset", "make_classifier", "load_phase_classifier", "Path"]
