"""Keep the reported default and exercise discrimination when conditioning training."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.api import create_app
from homeease.counter import CycleCounter
from homeease.exercises import EXERCISE_NAMES, decode


def probabilities(exercise, phase, other_action=None):
    p = np.zeros(17)
    p[16] = .975
    for i, name in enumerate(EXERCISE_NAMES):
        p[4 * i + (phase if name == exercise else 0)] = .01 if name == exercise else .005
    if other_action is not None:
        p[16] -= .8
        p[4 * EXERCISE_NAMES.index(other_action)] += .8
    return p


class Predictor:
    selected = {"min_dwell_s": .033, "model": "context-test", "decoder_strategy": "ordered_transition",
                "phase_evidence_threshold": .25}

    def stream(self):
        class Stream:
            def reset(self):
                pass

            def update(self, pose, timestamp):
                return probabilities("squat", int(round(pose[0, 0])))
        return Stream()


def frames():
    return [{"timestamp_ms": i * 100, "keypoints": [[phase, joint, 1] for joint in range(17)]}
            for i, phase in enumerate([0, 0, 1, 1, 2, 2, 3, 3, 0, 0])]


def test_training_scope_is_per_session_and_default_remains_open():
    with TestClient(create_app(Predictor())) as client:
        for body, expected, scope in [({}, 0, "open"), ({"action_scope": "training"}, 1, "training")]:
            created = client.post("/v1/sessions", json={"exercise": "squat", **body})
            created.raise_for_status()
            sid = created.json()["session_id"]
            result = client.post(f"/v1/sessions/{sid}/frames", json={"request_id": "cycle", "frames": frames()})
            result.raise_for_status()
            assert result.json()["action_scope"] == scope
            assert result.json()["count"] == expected
        assert client.post("/v1/sessions", json={"action_scope": "anything"}).status_code == 422


def test_counter_progress_reports_confirmed_stages_and_control_resets():
    with TestClient(create_app(Predictor())) as client:
        created = client.post("/v1/sessions", json={"exercise": "squat", "action_scope": "training"}).json()
        sid = created["session_id"]
        assert created["counter_stage"] == -1
        observations = frames()
        for batch, expected in enumerate([0, 1, 2, 3, 0]):
            result = client.post(f"/v1/sessions/{sid}/frames", json={
                "request_id": f"phase-{batch}", "frames": observations[batch*2:batch*2+2]}).json()
            assert result["counter_stage"] == expected
            assert result["count"] == (1 if batch == 4 else 0)
        assert client.get(f"/v1/sessions/{sid}").json()["counter_stage"] == 0
        for action in ["pause", "resume", "end"]:
            state = client.post(f"/v1/sessions/{sid}/control", json={"action": action}).json()
            assert state["counter_stage"] == -1
            assert state["count"] == 1


@pytest.mark.parametrize("exercise", EXERCISE_NAMES)
def test_training_preserves_competing_actions_and_raw_probabilities(exercise):
    p = probabilities(exercise, 2)
    original = p.copy()
    assert decode(p, exercise)[0] == 4
    phase, canonical, actions = decode(p, exercise, "training")
    assert phase == 2
    assert canonical[2] == pytest.approx(1)
    assert actions[4] == pytest.approx(.975)
    np.testing.assert_array_equal(p, original)
    competitor = next(name for name in EXERCISE_NAMES if name != exercise)
    assert decode(probabilities(exercise, 2, competitor), exercise, "training")[0] == 4


@pytest.mark.parametrize("interruption", ["occlusion", "person", "gap", "pause"])
def test_training_cannot_finish_a_cycle_across_an_interruption(interruption):
    from homeease.engine import ExerciseSession
    session = ExerciseSession(Predictor(), exercise="squat", action_scope="training")
    observations = frames()
    for frame in observations[:6]:
        session.update(frame["timestamp_ms"], frame["keypoints"])
    person = "primary"
    if interruption == "pause":
        session.control("pause")
        session.control("resume")
    elif interruption == "occlusion":
        session.update(550, [[0, i, 0] for i in range(17)])
    elif interruption == "person":
        person = "new-person"
        session.update(550, observations[5]["keypoints"], person)
    for frame in observations[6:]:
        stamp = frame["timestamp_ms"] + (1000 if interruption == "gap" else 0)
        session.update(stamp, frame["keypoints"], person)
    assert session.counter.count == 0


def test_cached_validation_and_test_counts_preserve_discrimination():
    selected = json.loads((ROOT / "artifacts/selected.json").read_text())

    def count(sequence, target, scope):
        counter = CycleCounter(min_dwell_s=selected["min_dwell_s"])
        for i, p in enumerate(sequence):
            counter.update_probabilities(decode(p, target, scope)[1], i / 30,
                selected["phase_evidence_threshold"], infer_return=True)
        return counter.count

    for split in ("validation", "test"):
        with np.load(ROOT / f"reports/optimization/opt_tcn_b4_w48_{split}_probabilities.npz") as cache:
            for action in range(5):
                for subject in range(2):
                    index = action * 2 + subject
                    sequence = cache[f"session_{index}"]
                    for target in EXERCISE_NAMES:
                        before, after = count(sequence, target, "open"), count(sequence, target, "training")
                        assert after == before, (split, index, target, before, after)
                        if action == 4 or target != EXERCISE_NAMES[action]:
                            assert after == 0, (split, index, target)
