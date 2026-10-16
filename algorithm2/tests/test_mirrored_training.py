"""Training view augmentation must preserve the model and interruption contract."""
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.api import create_app
from homeease.engine import ExerciseSession, MirroredPhaseStream
from homeease.exercises import EXERCISE_NAMES


class RecordingStream:
    def __init__(self):
        self.observations = []
        self.resets = 0

    def reset(self):
        self.observations.clear()
        self.resets += 1

    def update(self, pose, timestamp):
        self.observations.append((pose.copy(), timestamp))
        p = np.ones(17)
        p[0] = abs(float(pose[0, 0])) + 1
        return p / p.sum()


class RecordingPredictor:
    selected = {"min_dwell_s": .033, "model": "view-test", "confidence_threshold": 0}

    def __init__(self):
        self.streams = []

    def stream(self):
        stream = RecordingStream()
        self.streams.append(stream)
        return stream


@pytest.mark.parametrize("exercise", EXERCISE_NAMES)
@pytest.mark.parametrize("scope", ["open", "training"])
def test_view_policy_is_limited_to_two_training_exercises(exercise, scope):
    predictor = RecordingPredictor()
    session = ExerciseSession(predictor, exercise=exercise, action_scope=scope)
    mirrored = scope == "training" and exercise in {"standing_leg_raise", "hurdle_step"}
    assert session.snapshot()["inference_views"] == ("mirror_average" if mirrored else "original")
    assert len(predictor.streams) == (2 if mirrored else 1)


def test_mirror_matches_training_augmentation_without_mutating_input():
    predictor = RecordingPredictor()
    stream = MirroredPhaseStream(predictor)
    pose = np.arange(36, dtype=np.float32).reshape(12, 3)
    before = pose.copy()
    result = stream.update(pose, 3.5)
    original, mirrored = predictor.streams
    expected = pose[[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]].copy()
    expected[:, 0] *= -1
    np.testing.assert_array_equal(original.observations[0][0], pose)
    np.testing.assert_array_equal(mirrored.observations[0][0], expected)
    assert original.observations[0][1] == mirrored.observations[0][1] == 3.5
    np.testing.assert_array_equal(pose, before)
    np.testing.assert_allclose(result, (RecordingStream().update(pose, 3.5)
                                     + RecordingStream().update(expected, 3.5)) / 2)
    assert result.sum() == pytest.approx(1)


@pytest.mark.parametrize("interruption", ["occlusion", "person", "gap", "pause"])
def test_both_histories_reset_on_input_interruption(interruption):
    predictor = RecordingPredictor()
    session = ExerciseSession(predictor, exercise="hurdle_step", action_scope="training")
    pose = [[joint, joint + 1, 1] for joint in range(17)]
    session.update(0, pose)
    assert all(len(stream.observations) == 1 for stream in predictor.streams)
    if interruption == "pause":
        session.control("pause")
        session.control("resume")
    elif interruption == "occlusion":
        session.update(100, [[x, y, 0] for x, y, c in pose])
    else:
        session.update(1000 if interruption == "gap" else 100, pose,
                       "new-person" if interruption == "person" else "primary")
    assert all(stream.resets >= 1 for stream in predictor.streams)
    assert all(len(stream.observations) <= 1 for stream in predictor.streams)
    assert session.snapshot()["count"] == 0


def test_http_exposes_view_policy_and_sessions_do_not_share_histories():
    predictor = RecordingPredictor()
    with TestClient(create_app(predictor)) as client:
        first = client.post("/v1/sessions", json={"exercise": "hurdle_step", "action_scope": "training"}).json()
        second = client.post("/v1/sessions", json={"exercise": "hurdle_step"}).json()
        assert first["inference_views"] == "mirror_average"
        assert first["phase_policy"] == "action_gap_hold_v1"
        assert second["inference_views"] == "original"
        assert second["phase_policy"] == "original"
        pose = [[joint, joint + 1, 1] for joint in range(17)]
        result = client.post(f'/v1/sessions/{first["session_id"]}/frames', json={
            "request_id": "one", "frames": [{"timestamp_ms": 0, "keypoints": pose}]})
        assert result.status_code == 200
        assert result.json()["inference_views"] == "mirror_average"
        assert not predictor.streams[2].observations
