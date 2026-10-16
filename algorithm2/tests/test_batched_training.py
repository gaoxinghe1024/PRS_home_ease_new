"""Batching available camera ticks must preserve causal model histories."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from homeease.engine import MirroredPhaseStream, PhaseStream
from homeease.models import build_model


def predictor(kind):
    torch.manual_seed(17)
    torch.set_num_threads(2)
    members = []
    for mode, dim, context in [("stable80", 80, 5), ("hybrid", 64, 9)]:
        model = build_model({"hierarchical": True, "kind": kind, "width": 8, "blocks": 1}, dim).eval()
        members.append({"model": model, "mode": mode, "context": context,
                        "mean": np.zeros(dim, dtype=np.float32), "std": np.ones(dim, dtype=np.float32)})
    result = SimpleNamespace(members=members, device=torch.device("cpu"), lock=threading.RLock())
    result.stream = lambda: PhaseStream(result)
    return result


def sequential(original, mirrored, observations):
    output = []
    for pose, timestamp in observations:
        reflected = pose[[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]].copy()
        reflected[:, 0] *= -1
        output.append((original.update(pose, timestamp) + mirrored.update(reflected, timestamp)) / 2)
    return output


@pytest.mark.parametrize("kind", ["tcn", "gru"])
def test_batch_matches_sequential_across_padding_rollover_and_reset(kind):
    model = predictor(kind)
    batch = MirroredPhaseStream(model)
    original, mirrored = model.stream(), model.stream()
    rng = np.random.default_rng(7)
    poses = rng.normal(size=(30, 12, 3)).astype(np.float32)
    poses[:, :, 2] = 1
    before = poses.copy()
    offset = 0
    for size in [1, 3, 8, 0, 6, 7, 5]:
        if size == 0:
            batch.reset()
            original.reset()
            mirrored.reset()
        observations = [(pose, index / 30) for index, pose in enumerate(poses[offset:offset + size], offset)]
        expected = sequential(original, mirrored, observations)
        actual = batch.update_many(observations)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
        offset += size
    np.testing.assert_array_equal(poses, before)


def test_batch_uses_only_each_ticks_past_and_retains_continuation():
    model = predictor("tcn")
    rng = np.random.default_rng(33)
    poses = rng.normal(size=(8, 12, 3)).astype(np.float32)
    observations = [(pose, index / 30) for index, pose in enumerate(poses)]
    stream = MirroredPhaseStream(model)
    reference = stream.update_many(observations)
    changed = poses.copy()
    changed[4:, :, :2] += 10
    other = MirroredPhaseStream(model).update_many([(pose, index / 30) for index, pose in enumerate(changed)])
    np.testing.assert_array_equal(reference[:4], other[:4])
    continuation = stream.update(poses[0], 8 / 30)
    expected = sequential(model.stream(), model.stream(), observations + [(poses[0], 8 / 30)])[-1]
    np.testing.assert_allclose(continuation, expected, atol=1e-6, rtol=0)
