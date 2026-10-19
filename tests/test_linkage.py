"""One fall seen by the camera (or the sensor) and heard by the microphone is one event, asked about once."""
from datetime import datetime, timedelta, timezone

import pytest

from care.events import LINK_WINDOW_S, Events
from care.storage import Store


@pytest.fixture
def system(tmp_path):
    store = Store(tmp_path / "test.sqlite3")
    return store, Events(store)


def at(seconds=0):
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(timespec="milliseconds")


def open_live(store):
    return [e for e in store.events(state="open") if e["mode"] == "live"]


def impact(sound_id="s1", seconds=0, category="IMPACT"):
    return {"event_id": sound_id, "category": category, "confidence": .8, "t_start_utc": at(seconds), "n_windows": 2}


def test_spoken_help_after_a_camera_fall_joins_the_fall_and_raises_it(system):
    store, service = system
    fall = service.create("vision", {"prob": .9, "threshold": .5}, "live", "camera 0", "video:1:100", at=at())
    before = len(store.notifications())
    joined = service.create("audio", {"intent": "HELP"}, "live", "连续求救监听", "audio:a", at=at(4))
    assert joined["id"] == fall["id"] and len(open_live(store)) == 1
    assert joined["level"] == "EMERGENCY" and joined["escalated_at"]
    assert joined["tag"] == "dual_channel" and joined["channels"] == ["vision", "audio"]
    assert joined["inquiry"]["status"] == "not_started"  # still asked once, now as one incident
    assert joined["timeline"][-1]["action"] == "linked"
    assert len(store.notifications()) == before + 1  # the caregivers hear that it became an emergency


def test_a_camera_fall_after_a_spoken_call_joins_the_call(system):
    store, service = system
    call = service.create("audio", {"intent": "HELP"}, "live", "连续求救监听", "audio:a", at=at())
    joined = service.create("vision", {"prob": .9, "threshold": .5}, "live", "camera 0", "video:1:100", at=at(8))
    assert joined["id"] == call["id"] and joined["level"] == "EMERGENCY" and joined["tag"] == "dual_channel"
    assert len(open_live(store)) == 1


def test_the_same_alarm_reported_again_is_not_linked_twice(system):
    store, service = system
    fall = service.create("vision", {"prob": .9}, "live", "camera 0", "video:1:100", at=at())
    service.create("audio", {"intent": "HELP"}, "live", "listener", "audio:a", at=at(2))
    again = service.create("audio", {"intent": "HELP"}, "live", "listener", "audio:a", at=at(2))
    assert again["id"] == fall["id"] and len(again["linked"]) == 1 and len(open_live(store)) == 1


def test_far_apart_alarms_withdrawn_alarms_and_replays_stay_apart(system):
    store, service = system
    service.create("vision", {"prob": .9}, "live", "camera 0", "video:1:100", at=at())
    late = service.create("audio", {"intent": "HELP"}, "live", "listener", "audio:a", at=at(LINK_WINDOW_S + 15))
    assert late["channel"] == "audio" and len(open_live(store)) == 2

    call = service.create("audio", {"intent": "HELP"}, "live", "listener", "audio:b", at=at(200))
    service.action(call["id"], "withdraw")
    fall = service.create("vision", {"prob": .9}, "live", "camera 0", "video:1:300", at=at(205))
    assert fall["id"] != call["id"] and fall["level"] == "PENDING"

    replay = service.create("audio", {"intent": "HELP"}, "replay", "音频回放", at=at(1))
    assert replay["mode"] == "replay" and replay["channel"] == "audio"


def test_a_second_alarm_of_the_same_channel_is_still_its_own_event(system):
    store, service = system
    first = service.create("vision", {"prob": .9}, "live", "camera 0", "video:1:100", at=at())
    second = service.create("vision", {"prob": .9}, "live", "camera 0", "video:1:105", at=at(5))
    assert first["id"] != second["id"]


def test_sound_labels_are_noted_on_the_fall_but_never_change_it(system):
    store, service = system
    fall = service.create("vision", {"prob": .9}, "live", "camera 0", "video:1:100", at=at())
    noted = service.attach_sound(impact(seconds=-3))
    assert noted["id"] == fall["id"] and noted["level"] == "PENDING" and noted["inquiry"]["status"] == "not_started"
    assert noted["sound_hints"][0]["category"] == "IMPACT" and noted["timeline"][-1]["action"] == "sound_hint"
    assert service.attach_sound(impact(seconds=-3)) is None  # the same sound grows windows: noted once
    assert service.attach_sound(impact("daily", category="DAILY")) is None
    assert service.attach_sound(impact("far", seconds=LINK_WINDOW_S + 20)) is None


def test_a_sound_heard_before_the_camera_decides_is_noted_when_the_fall_arrives(system):
    store, service = system
    store.save_sound({"task": "sounds", "result": {"events": [impact(seconds=-6)], "raw_audio_stored": False}}, "live")
    fall = service.create("vision", {"prob": .9}, "live", "camera 0", "video:1:100", at=at())
    assert [h["category"] for h in fall["sound_hints"]] == ["IMPACT"] and fall["level"] == "PENDING"
