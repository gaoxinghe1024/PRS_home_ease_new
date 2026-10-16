"""Training holds preserve model posteriors and real input interruptions."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from homeease.engine import ExerciseSession
from homeease.exercises import EXERCISE_NAMES


class Predictor:
    selected = {"min_dwell_s": .033, "model": "continuity-test",
                "decoder_strategy": "ordered_transition", "phase_evidence_threshold": .25}

    def stream(self):
        return self

    def reset(self):
        pass

    def update(self, pose, timestamp):
        return self.probabilities.copy()


def feed(session, predictor, start, end, phases, support=1):
    predictor.probabilities = np.zeros(17)
    target = EXERCISE_NAMES.index(session.exercise)
    predictor.probabilities[target * 4:target * 4 + 4] = np.asarray(phases) * support
    predictor.probabilities[((target + 1) % 4) * 4] = 1 - support
    pose = [[i, i + 1, 1.] for i in range(17)]
    for stamp in range(start, end, 33):
        session.update(stamp, pose)


def prepared(policy=None):
    predictor = Predictor()
    session = ExerciseSession(predictor, exercise="hurdle_step", action_scope="training", phase_policy=policy)
    feed(session, predictor, 0, 200, [1, 0, 0, 0])
    feed(session, predictor, 231, 431, [0, 1, 0, 0])
    assert session.counter.stage == 1
    return session, predictor


def test_short_classifier_gap_holds_without_advancing_then_counts_once():
    session, predictor = prepared()
    feed(session, predictor, 462, 562, [0, 0, 1, 0], support=.01)
    assert session.counter.stage == 1
    assert session.counter.count == 0
    assert session.snapshot()["phase_probabilities"]["other"] == 1
    feed(session, predictor, 594, 794, [0, 0, 1, 0])
    feed(session, predictor, 825, 1025, [0, 0, 0, 1])
    feed(session, predictor, 1056, 1356, [1, 0, 0, 0])
    assert session.counter.count == 1
    assert len(session.counter.events) == 1


def test_long_classifier_gap_invalidates_unfinished_cycle():
    session, predictor = prepared()
    feed(session, predictor, 462, 862, [0, 0, 1, 0], support=.01)
    assert session.counter.stage == -1
    feed(session, predictor, 891, 1091, [0, 0, 0, 1])
    feed(session, predictor, 1122, 1422, [1, 0, 0, 0])
    assert session.counter.count == 0


def test_hold_keeps_original_phase_posteriors_without_smoothing():
    predictor = Predictor()
    session = ExerciseSession(predictor, exercise="hurdle_step", action_scope="training")
    feed(session, predictor, 0, 200, [1, 0, 0, 0])
    feed(session, predictor, 231, 531, [.65, .30, .03, .02])
    np.testing.assert_allclose(list(session.snapshot()["phase_probabilities"].values()), [.65, .30, .03, .02, 0])


@pytest.mark.parametrize("interruption", ["occlusion", "gap", "person", "pause"])
def test_action_hold_cannot_bridge_real_input_interruptions(interruption):
    session, predictor = prepared()
    pose = [[i, i + 1, 1.] for i in range(17)]
    if interruption == "occlusion":
        session.update(440, [[x, y, 0] for x, y, c in pose])
    elif interruption == "gap":
        session.update(700, pose)
    elif interruption == "person":
        session.update(440, pose, "another-person")
    else:
        session.control("pause")
        session.control("resume")
    assert session.last_supported_action_tick is None or interruption in {"gap", "person"}
    assert session.counter.stage == -1
    assert session.counter.count == 0


def test_original_policy_preserves_immediate_classifier_interrupt():
    session, predictor = prepared("original")
    feed(session, predictor, 462, 562, [0, 0, 1, 0], support=.01)
    assert session.counter.stage == -1


@pytest.mark.parametrize("exercise", EXERCISE_NAMES)
@pytest.mark.parametrize("scope", ["open", "training"])
def test_policy_is_limited_to_hurdle_training(exercise, scope):
    session = ExerciseSession(Predictor(), exercise=exercise, action_scope=scope)
    expected = "action_gap_hold_v1" if scope == "training" and exercise == "hurdle_step" else "original"
    assert session.snapshot()["phase_policy"] == expected
