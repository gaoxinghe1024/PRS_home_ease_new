import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from care.exercise import ExerciseDiagnostics
from care.workers import ExerciseWorker


def state(**updates):
    return {"session_id": "test", "status": "running", "quality": "good",
            "last_timestamp_ms": 0, "elapsed_s": 0, "interruptions": [], **updates}


def test_capture_diagnostics_survive_control_and_idempotent_retry():
    worker = ExerciseWorker.__new__(ExerciseWorker)
    worker.diagnostics = {"test": ExerciseDiagnostics()}
    points = [[100, 100, 1.0] for _ in range(17)]
    points[15][2] = .2
    body = {"request_id": "frame-0", "frames": [{"timestamp_ms": 0, "keypoints": points}]}
    interrupted = state(quality="insufficient_pose", interruptions=[{"reason": "insufficient_pose", "timestamp_ms": 0}])
    worker._request = lambda *args: interrupted
    first = worker.frames("test", body)
    assert first["capture_diagnostics"]["missing_joints"] == ["left_ankle"]
    retry = worker.frames("test", body)
    assert retry["counting_diagnostics"]["observed_frames"] == 1
    assert retry["counting_diagnostics"]["missing_joint_frames"] == {"left_ankle": 1}
    worker._request = lambda *args: state(status="paused", quality="paused")
    paused = worker.control("test", "pause")
    assert paused["capture_diagnostics"] == first["capture_diagnostics"]
    assert paused["counting_diagnostics"]["uninterrupted_s"] == 0
    assert worker.snapshot("test")["capture_diagnostics"] == paused["capture_diagnostics"]


def test_rejected_frames_do_not_change_diagnostics():
    worker = ExerciseWorker.__new__(ExerciseWorker)
    worker.diagnostics = {"test": ExerciseDiagnostics()}
    def reject(*args):
        raise ValueError("TIMESTAMP_NOT_INCREASING")
    worker._request = reject
    with pytest.raises(ValueError):
        worker.frames("test", {"frames": [{"timestamp_ms": 0, "keypoints": []}]})
    assert worker.diagnostics["test"].observed_frames == 0


def test_transient_interruptions_remain_visible_after_good_frame():
    diagnostic = ExerciseDiagnostics()
    frames = [{"timestamp_ms": t, "keypoints": [[1, 1, 1] for _ in range(17)]} for t in (0, 33, 400, 433)]
    diagnostic.observe(frames)
    result = diagnostic.attach(state(last_timestamp_ms=433, elapsed_s=.433,
        interruptions=[{"reason": "timestamp_gap", "timestamp_ms": 400}]))
    assert result["capture_diagnostics"]["gap_ms"] == 33
    counts = result["counting_diagnostics"]
    assert counts["max_gap_ms"] == 367
    assert counts["interruption_counts"] == {"timestamp_gap": 1}
    assert counts["uninterrupted_s"] == .033


def test_trace_is_bounded_by_time_and_frame_count():
    diagnostic = ExerciseDiagnostics()
    points = [[1, 1, 1] for _ in range(17)]
    diagnostic.observe([{"timestamp_ms": t, "keypoints": points} for t in range(2000)])
    assert len(diagnostic.recent_frames) == 1800
    assert diagnostic.recent_frames[0]["timestamp_ms"] == 200
    diagnostic.observe([{"timestamp_ms": 63000, "keypoints": points, "person_id": "new"}])
    assert list(diagnostic.recent_frames) == [{"timestamp_ms": 63000, "keypoints": points, "person_id": "new"}]


@pytest.mark.parametrize("by_scope", [False, True])
def test_trace_deletion_follows_training_record_selection(tmp_path, monkeypatch, by_scope):
    import care.records as records
    from care.storage import Store
    monkeypatch.setattr(records, "RUNTIME", tmp_path)
    store = Store(tmp_path / "test.sqlite3")
    directory = tmp_path / "exercise-traces"
    directory.mkdir()
    for sid, mode in [("drop", "replay"), ("keep", "live")]:
        store.save_session({"session_id": sid, "mode": mode, "status": "ended", "created_at": "2026-10-15"})
        (directory / f"{sid}.json").write_text("{}", encoding="ascii")
    service = records.Records(store)
    if by_scope:
        service.scope("replay", "2026-10-01", events=False)
    else:
        service.sessions(["drop"])
    assert not (directory / "drop.json").exists()
    assert (directory / "keep.json").exists()


def test_uploaded_video_matches_original_decoded_pixels_and_timestamps(tmp_path):
    path = str(tmp_path / "exercise.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20, (64, 48))
    assert writer.isOpened()
    rng = np.random.default_rng(42)
    for _ in range(3):
        writer.write(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    writer.release()
    reference = cv2.VideoCapture(path)
    expected = []
    while True:
        ok, pixels = reference.read()
        if not ok:
            break
        expected.append(pixels)
    reference.release()
    worker = ExerciseWorker.__new__(ExerciseWorker)
    worker.video_readers = {}
    received = []
    def pose(sid, pixels, stamp, request_id):
        received.append((pixels.copy(), stamp, request_id))
        return state()
    worker._pose_frame = pose
    try:
        assert worker.video_open("test", path)["fps"] == 20
        for _ in expected:
            assert worker.video_frame("test")["frame_shape"] == [48, 64, 3]
        assert worker.video_frame("test") == {"eof": True}
        for i, (pixels, stamp, request_id) in enumerate(received):
            np.testing.assert_array_equal(pixels, expected[i])
            assert stamp == i * 50
            assert request_id == f"video-{i}"
    finally:
        capture = worker.video_readers["test"]["capture"]
        worker.video_detach("test")
    assert not capture.isOpened() and not worker.video_readers


def test_camera_skips_pre_session_frames_instead_of_clamping_timestamps():
    worker = ExerciseWorker.__new__(ExerciseWorker)
    base = time.perf_counter()
    observations = iter([(np.zeros((2, 2, 3), np.uint8), seq, base + dt) for seq, dt in [(1, -.1), (2, -.05), (3, .02)]])
    worker.camera_readers = {"test": SimpleNamespace(read=lambda: next(observations))}
    worker.pose_initialize = lambda: None
    stamps = []
    def pose(sid, pixels, stamp, request_id):
        stamps.append(stamp)
        return state()
    worker._pose_frame = pose
    assert worker.camera_frame("test", {}, base, 0) == {"duplicate": True}
    assert worker.camera_frame("test", {}, base, 0) == {"duplicate": True}
    assert worker.camera_frame("test", {}, base, 0)["capture_sequence"] == 3
    assert len(stamps) == 1 and stamps[0] == pytest.approx(20)
