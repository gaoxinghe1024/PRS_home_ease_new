from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from homeease.core.config import AppConfig, load_config
from homeease.datasets.synthetic import SCENARIOS, generate_scenario


@pytest.fixture()
def cfg() -> AppConfig:
    return load_config(None)


@pytest.fixture(scope="session")
def scenario_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Seeded synthetic scenario recordings shared by the acceptance tests."""
    d = tmp_path_factory.mktemp("scenarios")
    for i, name in enumerate(SCENARIOS):
        generate_scenario(name, 42 + i, d)
    return d


def scenario_path(scenario_dir: Path, name: str) -> Path:
    hits = [p for p in scenario_dir.iterdir() if p.name.startswith(f"syn-{name}-")]
    assert hits, name
    return hits[0]


@pytest.fixture(autouse=True)
def _no_host_audio_enumeration(monkeypatch: pytest.MonkeyPatch) -> None:
    """The voice inquiry's start-up check lists the audio devices (PortAudio). Tests must not depend on, or be slowed by,
    the host's audio stack: no device is ever opened, and none is even enumerated unless a test installs its own stub."""
    try:
        import sounddevice
    except Exception:  # the audio extra is optional
        return
    monkeypatch.setattr(sounddevice, "query_devices", lambda *a, **k: [])


@pytest.fixture(autouse=True)
def _drop_log_handlers_on_closed_streams() -> Iterator[None]:
    """`setup_logging` runs once per process and binds a StreamHandler to the stderr of the test that first called it (a CLI test under capsys).
    Once that capture is closed every later WARNING raised "--- Logging error ---" in unrelated tests; remove such dead handlers after each test."""
    yield
    import logging

    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and getattr(getattr(h, "stream", None), "closed", False):
            root.removeHandler(h)
