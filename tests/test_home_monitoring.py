"""Durable home monitoring must not depend on an open preview or exercise."""
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest
from fastapi.testclient import TestClient

from care.services import System
from care.storage import Store


@pytest.fixture
def home(tmp_path):
    service = System(Store(tmp_path / "home.sqlite3"))
    service.modules["video"] = SimpleNamespace(state="ready", close=Mock())
    def start_video(source, mode):
        service.video_mode = mode
        service.video.update(state="running", processing=True)

    service.video_start = Mock(side_effect=start_video)
    service.video_stop = Mock(side_effect=lambda: service.video.update(state="idle", processing=False))
    yield service
    service.pool.shutdown(wait=True)


def test_monitoring_starts_once_and_survives_preview_absence(home):
    home.set_home_monitoring(True, 2)
    assert home.store.home_monitoring() == {"enabled": True, "camera_index": 2}
    home.ensure_home_monitoring()
    home.video_start.assert_called_once_with({"type": "camera", "index": 2}, "live")
    home.ensure_home_monitoring()
    assert home.video_start.call_count == 1
    assert not home.session_metadata


def test_monitoring_preference_recovers_after_service_restart(home):
    home.set_home_monitoring(True, 1)
    restarted = System(home.store)
    try:
        assert restarted.home_monitoring["enabled"]
        assert restarted.home_monitoring["camera_index"] == 1
        assert restarted.store.settings()["audio_consent"] is False
    finally:
        restarted.pool.shutdown(wait=True)


def test_camera_failure_is_visible_and_retry_is_bounded(home):
    home.set_home_monitoring(True)
    home.video_start.side_effect = RuntimeError("camera disconnected")
    home.ensure_home_monitoring()
    home.ensure_home_monitoring()
    assert home.video_start.call_count == 1
    assert home.home_monitoring["error"] == "camera disconnected"
    home.home_retry_at = 0
    home.video_start.side_effect = lambda *args: home.video.update(state="running", processing=True)
    home.ensure_home_monitoring()
    assert home.video_start.call_count == 2
    assert home.home_monitoring["error"] is None


def test_pause_is_durable_and_does_not_restart(home):
    home.set_home_monitoring(True)
    home.ensure_home_monitoring()
    home.set_home_monitoring(False)
    home.home_retry_at = 0
    home.ensure_home_monitoring()
    home.video_stop.assert_called_once()
    assert home.video_start.call_count == 1
    assert home.store.home_monitoring()["enabled"] is False


def test_dead_capture_recovers_and_changing_camera_releases_old_run(home):
    home.set_home_monitoring(True)
    home.ensure_home_monitoring()
    home.video.update(state="finished", processing=False)
    home.home_retry_at = 0
    home.ensure_home_monitoring()
    assert home.video_start.call_count == 2
    home.set_home_monitoring(True, 3)
    home.video_stop.assert_called_once()
    home.ensure_home_monitoring()
    assert home.video_start.call_args.args == ({"type": "camera", "index": 3}, "live")


def test_wait_for_module_and_preserve_exercise_session(home):
    home.modules["video"].state = "loading"
    home.session_metadata["exercise"] = {"active": True, "status": "running"}
    home.set_home_monitoring(True)
    home.ensure_home_monitoring()
    home.video_start.assert_not_called()
    home.modules["video"].state = "ready"
    home.ensure_home_monitoring()
    assert home.session_metadata["exercise"] == {"active": True, "status": "running"}


def test_model_recovery_runs_off_scheduler_and_can_be_paused(home):
    entered, release = threading.Event(), threading.Event()
    def initialize(name):
        entered.set()
        release.wait(timeout=3)
        home.modules[name].state = "ready"
    home.initialize = initialize
    home.modules["video"].state = "unavailable"
    home.set_home_monitoring(True)
    try:
        home.ensure_home_monitoring()
        assert entered.wait(timeout=1)
        pending = home.home_initializing
        home.home_retry_at = 0
        home.ensure_home_monitoring()
        assert home.home_initializing is pending
        home.set_home_monitoring(False)
        assert not home.home_monitoring["enabled"]
    finally:
        release.set()
    pending.result(timeout=2)
    home.ensure_home_monitoring()
    home.video_start.assert_not_called()


def test_failed_poll_leaves_reinitialization_to_home_recovery(home):
    home.set_home_monitoring(True)
    home.video_run = "failed-run"
    home.video_finalized = False
    home.modules["video"].call = Mock(side_effect=RuntimeError("worker exited"))
    with pytest.raises(RuntimeError, match="worker exited"):
        home.poll_video()
    assert home.video_finalized
    assert home.video["state"] == "error"
    home.poll_video()
    assert home.modules["video"].call.call_count == 1


def test_home_routes_and_consent_preserve_profile(home, monkeypatch):
    import care.main as main
    monkeypatch.setattr(main, "system", home)
    client = TestClient(main.app)
    response = client.get("/home")
    assert response.status_code == 200
    assert 'data-household="true"' in response.text
    assert 'id="page-safety"' in response.text
    assert 'id="page-exercise"' in response.text
    assert 'data-household="true"' not in client.get("/").text
    assert client.get("/elder").status_code == 200
    assert client.put("/api/home/monitoring", json={"enabled": True, "camera_index": 11}).status_code == 422
    assert client.put("/api/home/monitoring", json={"enabled": True, "camera_index": 2}).json()["enabled"]
    before = home.store.settings()
    after = client.post("/api/home/audio-consent").json()
    assert after == {**before, "audio_consent": True}
    assert client.post("/api/video/stop").status_code == 200
    assert not home.home_monitoring["enabled"]
