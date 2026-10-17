"""The older adult can withdraw a self-triggered alarm within a short window; the event is never closed or downgraded."""
import time
from datetime import datetime, timedelta, timezone

import pytest

from care.events import WITHDRAW_WINDOW_S, Events, is_withdrawn, withdraw_status
from care.services import System
from care.storage import Store


@pytest.fixture
def system(tmp_path):
    store = Store(tmp_path / "test.sqlite3")
    return store, Events(store)


def two_contacts(store):
    settings = store.settings()
    settings["contacts"] = [{"name": "A", "relation": "family", "phone": "111"}, {"name": "B", "relation": "family", "phone": "222"}]
    store.save_settings(settings)


def age(store, event_id, seconds):
    stamp = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat(timespec="milliseconds")
    store.update_event(event_id, lambda e: e.update(detected_at=stamp))


def test_withdraw_keeps_level_and_state_and_tells_the_caregivers(system):
    store, service = system
    event = service.create("manual", {"action": "help_button"}, "live", "button")
    assert withdraw_status(event)["withdrawable"] is True
    before = len(store.notifications())
    result = service.action(event["id"], "withdraw", actor="elder")
    assert result["level"] == "EMERGENCY" and result["state"] == "open"
    assert is_withdrawn(result) and result["withdrawal"]["by"] == "elder"
    assert System.describe(result)["withdrawn"] is True and System.describe(result)["withdrawable"] is False
    assert len(store.notifications()) == before + 1
    assert store.notifications()[0]["title"]  # a caregiver notice exists
    # only a caregiver closes it
    closed = service.action(event["id"], "resolve", outcome="false_alarm", note="pressed by mistake")
    assert closed["state"] == "resolved" and closed["level"] == "EMERGENCY"


def test_withdraw_twice_does_not_notify_twice(system):
    store, service = system
    event = service.create("audio", {"intent": "HELP"}, "live", "listener")
    service.action(event["id"], "withdraw")
    count = len(store.notifications())
    again = service.action(event["id"], "withdraw")
    assert is_withdrawn(again)
    assert len(store.notifications()) == count  # the second request changes nothing and says nothing


def test_withdraw_is_refused_after_the_window_and_for_sensor_events(system):
    store, service = system
    event = service.create("manual", {"action": "help_button"}, "live", "button")
    age(store, event["id"], WITHDRAW_WINDOW_S + 5)
    assert withdraw_status(store.event(event["id"]))["withdrawable"] is False
    with pytest.raises(ValueError):
        service.action(event["id"], "withdraw")
    fall = service.create("vision", {"prob": .9}, "live", "camera")
    with pytest.raises(ValueError):
        service.action(fall["id"], "withdraw")
    assert store.event(fall["id"])["level"] == "PENDING" and not store.event(fall["id"]).get("withdrawal")


def test_withdrawn_alarm_stops_contact_escalation_and_a_caregiver_can_resume_it(system):
    store, service = system
    two_contacts(store)
    event = service.create("manual", {"action": "help_button"}, "live", "button")
    store.update_event(event["id"], lambda e: e.update(next_escalation_at=time.time() - 1))
    service.action(event["id"], "withdraw")
    service.tick()
    assert store.event(event["id"])["contact_index"] == 0  # not escalated
    # a caregiver who decides that it is real escalates by hand and the withdrawal no longer holds
    service.action(event["id"], "escalate")
    assert store.event(event["id"])["contact_index"] == 1
    assert not is_withdrawn(store.event(event["id"]))


def test_an_alarm_that_was_not_withdrawn_still_escalates(system):
    store, service = system
    two_contacts(store)
    event = service.create("manual", {"action": "help_button"}, "live", "button")
    store.update_event(event["id"], lambda e: e.update(next_escalation_at=time.time() - 1))
    service.tick()
    assert store.event(event["id"])["contact_index"] == 1


def test_a_later_spoken_call_for_help_overrides_the_withdrawal(system):
    store, service = system
    event = service.create("manual", {"action": "help_button"}, "live", "button")
    service.action(event["id"], "withdraw")
    service.inquiry_start(event["id"], True)
    service.inquiry_processing(event["id"])
    result = service.inquiry_reply(event["id"], {"status": "RESPONDED", "intent": "HELP"})
    assert not is_withdrawn(result) and result["withdrawal"].get("overridden_at")
    assert result["level"] == "EMERGENCY"


def test_a_closed_event_cannot_be_withdrawn(system):
    store, service = system
    event = service.create("manual", {"action": "help_button"}, "live", "button")
    service.action(event["id"], "resolve", outcome="safe")
    with pytest.raises(ValueError):
        service.action(event["id"], "withdraw")


def escalate_by_inquiry(store, service, event_id, reply=None):
    """The older adult did not answer (or called for help): the inquiry raises the fall alarm to an emergency."""
    service.inquiry_start(event_id, True)
    service.inquiry_reply(event_id, reply or {"status": "NO_RESPONSE"})
    return store.event(event_id)


def age_escalation(store, event_id, seconds):
    stamp = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat(timespec="milliseconds")
    store.update_event(event_id, lambda e: e.update(escalated_at=stamp))


def test_a_fall_alarm_raised_by_the_inquiry_can_be_withdrawn_for_30_s(system):
    store, service = system
    two_contacts(store)
    fall = service.create("vision", {"prob": .9}, "live", "camera")
    assert withdraw_status(store.event(fall["id"]))["withdrawable"] is False  # still a candidate: the inquiry is the way to answer
    event = escalate_by_inquiry(store, service, fall["id"])
    assert event["level"] == "EMERGENCY" and event.get("escalated_at")
    status = withdraw_status(event)
    assert status["withdrawable"] is True and 0 < status["withdraw_left_s"] <= WITHDRAW_WINDOW_S
    service.action(fall["id"], "withdraw")
    after = store.event(fall["id"])
    assert is_withdrawn(after) and after["level"] == "EMERGENCY" and after["state"] == "open"  # never closed or downgraded


def test_the_window_of_an_escalated_fall_alarm_counts_from_the_escalation(system):
    store, service = system
    fall = service.create("vision", {"prob": .9}, "live", "camera")
    age(store, fall["id"], 600)  # detected long ago, escalated just now
    event = escalate_by_inquiry(store, service, fall["id"])
    assert withdraw_status(event)["withdrawable"] is True
    age_escalation(store, fall["id"], WITHDRAW_WINDOW_S + 5)
    assert withdraw_status(store.event(fall["id"]))["withdrawable"] is False
    with pytest.raises(ValueError):
        service.action(fall["id"], "withdraw")
    assert not store.event(fall["id"]).get("withdrawal")


def test_a_withdrawn_fall_alarm_stops_contact_escalation(system):
    store, service = system
    two_contacts(store)
    fall = service.create("vision", {"prob": .9}, "live", "camera")
    escalate_by_inquiry(store, service, fall["id"])
    service.action(fall["id"], "withdraw")
    store.update_event(fall["id"], lambda e: e.update(next_escalation_at=time.time() - 2))
    service.tick()
    assert store.event(fall["id"])["contact_index"] == 0


def test_a_caregiver_escalation_of_a_fall_candidate_is_not_withdrawable(system):
    store, service = system
    two_contacts(store)
    fall = service.create("vision", {"prob": .9}, "live", "camera")
    service.action(fall["id"], "escalate")
    assert store.event(fall["id"])["level"] == "EMERGENCY"
    assert withdraw_status(store.event(fall["id"]))["withdrawable"] is False
    with pytest.raises(ValueError):
        service.action(fall["id"], "withdraw")


def test_a_fall_candidate_that_a_valid_answer_cancelled_is_not_withdrawable(system):
    store, service = system
    fall = service.create("vision", {"prob": .9}, "live", "camera")
    service.inquiry_start(fall["id"], True)
    service.inquiry_reply(fall["id"], {"status": "RESPONDED", "intent": "REASSURANCE", "trusted_reassurance": True, "asr_avg_logprob": -.3})
    event = store.event(fall["id"])
    assert event["level"] == "PENDING" and not event.get("escalated_at")
    assert withdraw_status(event)["withdrawable"] is False


def test_a_later_spoken_call_overrides_the_withdrawal_of_an_escalated_fall_alarm(system):
    store, service = system
    fall = service.create("vision", {"prob": .9}, "live", "camera")
    escalate_by_inquiry(store, service, fall["id"])
    service.action(fall["id"], "withdraw")
    service.inquiry_start(fall["id"], True)
    service.inquiry_reply(fall["id"], {"status": "RESPONDED", "intent": "HELP", "asr_avg_logprob": -.4})
    assert not is_withdrawn(store.event(fall["id"]))
