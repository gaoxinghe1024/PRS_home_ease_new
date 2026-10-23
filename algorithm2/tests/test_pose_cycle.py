import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.engine import ExerciseSession
from homeease.api import create_app


class Predictor:
    selected = {"min_dwell_s": .033, "model": "test", "decoder_strategy": "ordered_transition",
                "phase_evidence_threshold": .25}

    def stream(self):
        return self

    def reset(self):
        pass

    def update(self, pose, timestamp):
        p = np.zeros(17)
        p[16] = 1  # No invented model probabilities to repair a missed transition.
        return p


def pose(kind="standing", side=0):
    p = np.array([[0., 0., 1.] for _ in range(17)])
    p[5:17, :2] = [[0, -.8], [.2, -.8], [-.2, -.4], [.4, -.4], [-.2, 0], [.4, 0],
                   [0, 0], [.2, 0], [0, .65], [.2, .65], [0, 1.3], [.2, 1.3]]
    if kind == "seated":
        p[[5, 6, 7, 8, 9, 10, 11, 12], 1] += .6
        p[[13, 14], 0] += .6
    if kind in {"lift", "small", "bent"}:
        p[13 + side, :2] += [.5, -.3]
        p[15 + side, :2] += [1., -.6]
        if kind == "small":
            p[13 + side, :2] = [side * .2 + .15, .6]
            p[15 + side, :2] = [side * .2 + .3, 1.2]
        if kind == "bent":
            p[13 + side, :2] = [side * .2 + .6, 0]
            p[15 + side, :2] = [side * .2 + .6, .6]
    return p


def session(exercise):
    policy = "pose_cycle_v2" if exercise == "chair_sit_to_stand" else "pose_cycle_v1"
    return ExerciseSession(Predictor(), exercise=exercise, action_scope="training", counting_policy=policy)


def feed(s, p, duration=.7):
    start = 0 if s.last_timestamp is None else s.last_timestamp + 1000 / 30
    for i in range(round(duration * 30)):
        s.update(start + i * 1000 / 30, p.tolist())


@pytest.mark.parametrize("exercise,rest,peak", [
    ("chair_sit_to_stand", "seated", "standing"),
    ("standing_leg_raise", "standing", "lift"),
])
def test_visible_complete_cycles_count_once_per_return(exercise, rest, peak):
    s = session(exercise)
    feed(s, pose(rest))
    for _ in range(3):
        feed(s, pose(peak))
        assert s.counter.count == _
        feed(s, pose(rest))
        assert s.counter.count == _ + 1
    feed(s, pose(rest), 5)
    assert s.counter.count == 3
    assert s.snapshot()["exercise_probabilities"]["other"] == 1


def test_legs_alternate_only_after_grounded_return():
    s = session("standing_leg_raise")
    feed(s, pose())
    for side in [0, 1, 0, 1]:
        feed(s, pose("lift", side))
        feed(s, pose())
    assert s.counter.count == 4


def test_switching_legs_without_return_does_not_complete_a_cycle():
    s = session("standing_leg_raise")
    feed(s, pose())
    feed(s, pose("lift", 0))
    feed(s, pose("lift", 1))
    feed(s, pose())
    assert s.counter.count == 0


def test_starting_at_peak_and_half_chair_movement_do_not_count():
    s = session("chair_sit_to_stand")
    feed(s, pose())
    feed(s, pose("seated"))
    half = pose("seated") * .6 + pose() * .4
    feed(s, half)
    feed(s, pose("seated"))
    assert s.counter.count == 0


def test_standing_start_is_recognized_while_waiting_for_seated_baseline():
    s = session("chair_sit_to_stand")
    feed(s, pose(), 3)
    state = s.snapshot()
    assert state["phase"] == state["setup_pose"] == "standing"
    assert state["quality"] == "good"
    assert state["counter_stage"] == -1
    assert state["count"] == 0
    feed(s, pose("seated"))
    assert s.snapshot()["counter_stage"] == 0
    assert s.snapshot()["setup_pose"] == "sitting"
    feed(s, pose())
    feed(s, pose("seated"))
    assert s.counter.count == 1


def test_chair_short_scheduling_gaps_preserve_observed_endpoint_cycle():
    s = session("chair_sit_to_stand")
    stamp = 0
    for kind in ["standing", "seated", "standing", "seated"]:
        for _ in range(6):
            state = s.update(stamp, pose(kind).tolist())
            stamp += [140, 290, 350][_ % 3]
    assert state["count"] == 1
    assert state["max_input_gap_ms"] == 400
    assert not any(i["reason"] == "timestamp_gap" for i in state["interruptions"])


def test_chair_long_gap_still_clears_cycle_and_recognizes_current_posture():
    s = session("chair_sit_to_stand")
    feed(s, pose("seated"))
    feed(s, pose())
    state = s.update(s.last_timestamp + 450, pose().tolist())
    assert state["counter_stage"] == -1
    assert state["setup_pose"] == state["phase"] == "standing"
    assert state["interruptions"][-1]["reason"] == "timestamp_gap"
    feed(s, pose("seated"))
    assert s.counter.count == 0


def test_other_modes_keep_250ms_input_gap_limit():
    for exercise, scope, policy in [("standing_leg_raise", "training", "pose_cycle_v1"),
                                   ("chair_sit_to_stand", "training", "pose_cycle_v1"),
                                   ("chair_sit_to_stand", "open", "original"),
                                   ("squat", "training", "original")]:
        s = ExerciseSession(Predictor(), exercise=exercise, action_scope=scope, counting_policy=policy)
        assert s.snapshot()["max_input_gap_ms"] == 250
        s.update(0, pose().tolist())
        state = s.update(300, pose().tolist())
        assert state["interruptions"][-1]["reason"] == "timestamp_gap"


@pytest.mark.parametrize("kind", ["standing", "small", "bent", "seated"])
def test_static_half_leg_bent_step_and_bilateral_squat_do_not_count(kind):
    s = session("standing_leg_raise")
    feed(s, pose())
    for _ in range(3):
        feed(s, pose(kind))
        feed(s, pose())
    assert s.counter.count == 0


@pytest.mark.parametrize("exercise,rest,peak", [
    ("chair_sit_to_stand", "seated", "standing"),
    ("standing_leg_raise", "standing", "lift"),
])
@pytest.mark.parametrize("interruption", ["occlusion", "gap", "person", "pause"])
def test_input_interruption_invalidates_unfinished_cycle(exercise, rest, peak, interruption):
    s = session(exercise)
    feed(s, pose(rest))
    feed(s, pose(peak))
    if interruption == "pause":
        s.control("pause")
        s.control("resume")
    else:
        p = pose(peak)
        if interruption == "occlusion":
            p[15, 2] = 0
        s.update(s.last_timestamp + (500 if interruption == "gap" else 33.34), p.tolist(),
                 "changed" if interruption == "person" else "primary")
    feed(s, pose(rest))
    assert s.counter.count == 0


def test_http_explicit_policy_is_validated_and_exposes_original_model_phase():
    with TestClient(create_app(Predictor())) as client:
        body = {"exercise": "chair_sit_to_stand", "action_scope": "training", "counting_policy": "pose_cycle_v2"}
        assert client.post("/v1/sessions", json={**body, "exercise": "hurdle_step"}).status_code == 422
        assert client.post("/v1/sessions", json={**body, "action_scope": "open"}).status_code == 422
        assert client.post("/v1/sessions", json={**body, "exercise": "standing_leg_raise"}).status_code == 422
        r = client.post("/v1/sessions", json=body)
        assert r.status_code == 201, r.text
        assert r.json()["counting_policy"] == "pose_cycle_v2"
        assert r.json()["model_phase"] == "other"
        assert client.post("/v1/sessions", json={"exercise": "chair_sit_to_stand"}).json()["counting_policy"] == "original"
