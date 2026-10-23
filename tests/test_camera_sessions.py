"""Exercise sessions must retain their camera after startup cleanup."""
import time
from unittest.mock import Mock

import numpy as np
import pytest

from care.camera import CameraHub, FrameReader
from care.services import System
from care.storage import Store


class Capture:
    def __init__(self):
        self.closed = False
        self.tick = 0

    def isOpened(self):
        return not self.closed

    def set(self, *args):
        return True

    def read(self):
        time.sleep(.005)
        self.tick += 1
        return True, np.full((48, 64, 3), self.tick % 255, np.uint8)

    def release(self):
        self.closed = True


@pytest.fixture
def camera_system(tmp_path, monkeypatch):
    service = System(Store(tmp_path / "camera.sqlite3"))
    captures = []

    def factory(index, *args):
        capture = Capture()
        captures.append((index, capture))
        return capture

    def call(method, **params):
        if method == "pose_initialize":
            return {}
        assert method == "create"
        return {"session_id": "training", "status": "running",
                "exercise": params["exercise"], "target_reps": params["target_reps"], "count": 0}

    service.cameras = CameraHub(factory)
    service.submit = Mock()
    monkeypatch.setattr(service.modules["exercise"], "call", call)
    try:
        yield service, captures
    finally:
        service.cameras.close()
        service.pool.shutdown(wait=True)


@pytest.mark.parametrize("mode", ["standalone", "shared", "handover"])
def test_exercise_retains_live_camera_after_startup(camera_system, mode):
    service, captures = camera_system
    previous = service.cameras.acquire(0, "video") if mode != "standalone" else None
    unrelated = service.cameras.acquire(1, "other-preview") if mode == "handover" else None
    result = service.start_camera_session("chair_sit_to_stand", 10, 0,
                                          handover="video" if mode == "handover" else None)
    sid = result["session_id"]
    assert sid in service.cameras.users(sid)
    assert "exercise-opening" not in service.cameras.users(sid)
    assert service.cameras.preview(sid).startswith(b"\xff\xd8")
    descriptor = service.submit.call_args.args[2]
    if previous:
        assert descriptor == previous
    assert len([capture for index, capture in captures if index == 0]) == 1
    if unrelated:
        assert service.cameras.descriptor("other-preview") == unrelated
        assert service.cameras.preview("other-preview")
    reader = FrameReader(descriptor)
    try:
        _, sequence, _ = reader.read()
        service.cameras.release("video")
        time.sleep(.03)
        _, later_sequence, _ = reader.read()
        assert later_sequence > sequence
        assert service.cameras.preview(sid)
        assert not next(capture for index, capture in captures if index == 0).closed
    finally:
        reader.close()
    service.cameras.release(sid)
    assert next(capture for index, capture in captures if index == 0).closed


@pytest.mark.parametrize("shared", [False, True])
def test_failed_exercise_initialization_releases_only_temporary_owner(camera_system, monkeypatch, shared):
    service, captures = camera_system
    if shared:
        service.cameras.acquire(0, "video")
    monkeypatch.setattr(service.modules["exercise"], "call", Mock(side_effect=RuntimeError("load failed")))
    with pytest.raises(RuntimeError, match="load failed"):
        service.start_camera_session("chair_sit_to_stand", 10, 0)
    if shared:
        assert service.cameras.status()[0]["users"] == ["video"]
        assert service.cameras.preview("video")
    else:
        assert service.cameras.status() == []
        assert captures[0][1].closed
