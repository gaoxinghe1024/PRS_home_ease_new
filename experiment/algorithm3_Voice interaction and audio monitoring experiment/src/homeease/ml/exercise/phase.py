"""Sit-to-stand phase classifiers: geometric rule with hysteresis, and wrappers
for trained models (sklearn / torch) plus soft-voting ensembles.

All classifiers expose ``step(feats) -> (phase, probs)`` and are causal.
UNKNOWN is returned when features are missing; it is never used as a training class.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol

import numpy as np

from homeease.core.config import ExerciseCfg
from homeease.ml.exercise.features import EX_FEATURES, PHASES, UNKNOWN, feature_vector


class PhaseClassifier(Protocol):
    model_id: str
    model_version: str

    def reset(self) -> None: ...

    def step(self, feats: dict[str, float] | None) -> tuple[str, np.ndarray | None]: ...


class RulePhaseClassifier:
    model_id = "phase_rule"
    model_version = "1.0"

    def __init__(self, cfg: ExerciseCfg) -> None:
        self.cfg = cfg
        self.prev = UNKNOWN

    def reset(self) -> None:
        self.prev = UNKNOWN

    def step(self, feats: dict[str, float] | None) -> tuple[str, np.ndarray | None]:
        c = self.cfg
        if feats is None:
            self.prev = UNKNOWN
            return UNKNOWN, None
        tv = feats["thigh_vertical"]
        vel = feats["hip_vel_tl_s"]  # +down
        prev = self.prev
        if feats.get("torso_deg", 0.0) > c.max_torso_deg:
            # Lying or strongly bent: not a sit-to-stand posture (e.g. a fall during a session).
            self.prev = UNKNOWN
            return UNKNOWN, None
        if prev == "SITTING" and tv < c.sit_exit:
            ph = "SITTING"
        elif prev == "STANDING" and tv > c.stand_exit:
            ph = "STANDING"
        elif tv <= c.sit_enter:
            ph = "SITTING"
        elif tv >= c.stand_enter:
            ph = "STANDING"
        else:
            if vel < -c.velocity_tl_s:
                ph = "RISING"
            elif vel > c.velocity_tl_s:
                ph = "DESCENDING"
            elif prev in ("SITTING", "RISING"):
                ph = "RISING"
            elif prev in ("STANDING", "DESCENDING"):
                ph = "DESCENDING"
            else:
                ph = UNKNOWN
        self.prev = ph
        probs = np.zeros(len(PHASES))
        if ph != UNKNOWN:
            probs[PHASES.index(ph)] = 1.0
        return ph, probs


class SequenceBuffer:
    """Causal buffer of the last ``n`` feature vectors with a validity mask."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.buf: deque[np.ndarray] = deque(maxlen=n)
        self.mask: deque[float] = deque(maxlen=n)

    def reset(self) -> None:
        self.buf.clear()
        self.mask.clear()

    def push(self, feats: dict[str, float] | None) -> None:
        if feats is None:
            self.buf.append(np.zeros(len(EX_FEATURES)))
            self.mask.append(0.0)
        else:
            v = feature_vector(feats)
            v = np.nan_to_num(v, nan=0.0)
            self.buf.append(v)
            self.mask.append(1.0)

    def array(self) -> tuple[np.ndarray, np.ndarray]:
        pad = self.n - len(self.buf)
        x = np.vstack([np.zeros((pad, len(EX_FEATURES)))] + [np.vstack(self.buf)] if self.buf else [np.zeros((self.n, len(EX_FEATURES)))])
        m = np.concatenate([np.zeros(pad), np.asarray(self.mask)]) if self.buf else np.zeros(self.n)
        return x[-self.n :], m[-self.n :]


class ModelPhaseClassifier:
    """Wraps a trained phase model package (see ml.exercise.train)."""

    def __init__(self, predictor: Any, model_id: str, model_version: str, context: int) -> None:
        self.predictor = predictor
        self.model_id = model_id
        self.model_version = model_version
        self.seq = SequenceBuffer(context)

    def reset(self) -> None:
        self.seq.reset()

    def step(self, feats: dict[str, float] | None) -> tuple[str, np.ndarray | None]:
        self.seq.push(feats)
        if feats is None:
            return UNKNOWN, None
        x, m = self.seq.array()
        probs = np.asarray(self.predictor(x, m), dtype=float)
        return PHASES[int(np.argmax(probs))], probs


class SoftVotingPhaseClassifier:
    def __init__(self, members: list[Any], weights: list[float] | None = None, model_id: str = "phase_ensemble", model_version: str = "1.0") -> None:
        self.members = members
        self.weights = np.asarray(weights if weights else [1.0] * len(members), dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.model_id = model_id
        self.model_version = model_version

    def reset(self) -> None:
        for m in self.members:
            m.reset()

    def step(self, feats: dict[str, float] | None) -> tuple[str, np.ndarray | None]:
        acc = np.zeros(len(PHASES))
        any_valid = False
        for w, m in zip(self.weights, self.members):
            ph, p = m.step(feats)
            if p is not None:
                acc += w * p
                any_valid = True
        if not any_valid:
            return UNKNOWN, None
        return PHASES[int(np.argmax(acc))], acc
