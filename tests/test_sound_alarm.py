"""A sound event of the sound monitor is a second alarm channel for a fall candidate, or a suspected event of its own."""
from datetime import datetime, timedelta, timezone

import pytest

from care.events import Events
from care.storage import Store


@pytest.fixture
def system(tmp_path):
    store = Store(tmp_path / "test.sqlite3")
    return store, Events(store)


def sound(category="HELP", seconds_from_now=0.0, event_id="s1", confidence=0.9):
    start = datetime.now(timezone.utc) + timedelta(seconds=seconds_from_now)
    return {"category": category, "confidence": confidence, "event_id": event_id, "t_start_utc": start.isoformat(timespec="milliseconds")}


def test_a_sound_next_to_a_fall_candidate_makes_it_dual_channel_and_an_emergency(system):
    store, service = system
    fall = service.create("vision", {"prob": .9}, "live", "camera")
    assert fall["level"] == "PENDING" and fall["tag"] == "single_channel"
    event = service.sound_alarm(sound("HELP"))
    assert event["id"] == fall["id"]
    stored = store.event(fall["id"])
    assert stored["level"] == "EMERGENCY" and stored["tag"] == "dual_channel"
    assert stored["channels"] == ["vision", "audio"] and stored.get("escalated_at")
    assert len(store.events(mode="live")) == 1
    assert store.notifications()[0]["level"] == "EMERGENCY"


def test_the_same_sound_again_changes_nothing(system):
    store, service = system
    fall = service.create("accel", {"score": .9}, "live", "sensor")
    service.sound_alarm(sound("IMPACT", event_id="a"))
    first = store.event(fall["id"])
    service.sound_alarm(sound("IMPACT", event_id="a"))
    again = store.event(fall["id"])
    assert again["markers"] == first["markers"] and len(again["timeline"]) == len(first["timeline"])


@pytest.mark.parametrize("category", ["HELP", "IMPACT", "ABNORMAL"])
def test_a_lone_sound_opens_a_suspected_event(system, category):
    store, service = system
    event = service.sound_alarm(sound(category))
    assert event["level"] == "PENDING" and event["channel"] == "audio" and event["state"] == "open"
    assert event["markers"][0]["kind"] == "sound_event" and event["tag"] == "single_channel"
    assert event["evidence"]["sound_category"] == category
    assert event["timeline"][0]["note"]


@pytest.mark.parametrize("category", ["DAILY", "UNKNOWN", None])
def test_other_labels_are_no_alarm(system, category):
    store, service = system
    assert service.sound_alarm(sound(category or "DAILY") | {"category": category}) is None
    assert store.events(mode="live") == []


def test_a_sound_far_from_the_candidate_is_a_separate_event(system):
    store, service = system
    fall = service.create("vision", {"prob": .9}, "live", "camera")
    other = service.sound_alarm(sound("IMPACT", seconds_from_now=40))
    assert other["id"] != fall["id"] and other["level"] == "PENDING"
    assert store.event(fall["id"])["level"] == "PENDING" and store.event(fall["id"])["tag"] == "single_channel"


def test_a_closed_candidate_is_not_touched(system):
    store, service = system
    fall = service.create("vision", {"prob": .9}, "live", "camera")
    service.action(fall["id"], "resolve", outcome="false_alarm", note="test")
    other = service.sound_alarm(sound("HELP"))
    assert other["id"] != fall["id"]
    assert store.event(fall["id"])["state"] == "resolved" and store.event(fall["id"])["level"] == "PENDING"


def test_a_sound_of_a_spoken_call_joins_that_event_instead_of_opening_another(system):
    store, service = system
    call = service.create("audio", {"intent": "HELP"}, "live", "listener")
    service.sound_alarm(sound("HELP"))
    assert len(store.events(mode="live")) == 1
    stored = store.event(call["id"])
    assert stored["level"] == "EMERGENCY" and any(m["kind"] == "sound_event" for m in stored["markers"])


def test_a_withdrawn_event_stays_withdrawn_when_a_sound_joins(system):
    store, service = system
    call = service.create("manual", {"action": "help_button"}, "live", "button")
    service.action(call["id"], "withdraw")
    service.sound_alarm(sound("ABNORMAL"))
    assert store.event(call["id"]).get("withdrawal")


def test_a_replay_event_is_never_joined(system):
    store, service = system
    replay = service.create("vision", {"prob": .9}, "replay", "sample")
    other = service.sound_alarm(sound("HELP"))
    assert other["id"] != replay["id"]
    assert store.event(replay["id"])["level"] == "PENDING"
