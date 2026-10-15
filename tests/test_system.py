import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from care.events import Events
from care.inquiry import ANSWERED, HIGH_RISK, NOT_ASKED, voice_counts, voice_inquiry
from care.records import Records
from care.services import System, canonical_csv
from care.storage import Store


@pytest.fixture
def system(tmp_path):
    store = Store(tmp_path / "test.sqlite3")
    return store, Events(store)


def candidate(system, mode="replay"):
    return system[1].create("vision", {"prob": .8, "threshold": .6}, mode)


def test_dedup_and_channel_separation(system):
    store, service = system
    first = service.create("accel", {"score": .9}, "replay", dedup_key="test")
    again = service.create("accel", {"score": .9}, "replay", dedup_key="test")
    assert first["id"] == again["id"]
    second = service.create("vision", {"prob": .9}, "replay")
    assert second["id"] != first["id"]
    assert len(store.events()) == 2
    assert len(store.notifications()) == 2
    assert first["level"] == second["level"] == "PENDING"
    assert first["tag"] == second["tag"] == "single_channel"


@pytest.mark.parametrize("record,expected", [
    ({"status": "NO_RESPONSE"}, "EMERGENCY"),
    ({"status": "RESPONDED", "intent": "HELP"}, "EMERGENCY"),
    ({"status": "RESPONDED", "intent": "REASSURANCE", "asr_avg_logprob": -.1}, "PENDING"),
    ({"status": "audio_unavailable"}, "PENDING"),
])
def test_voice_policy_keeps_event_open(system, record, expected):
    store, service = system
    e = candidate(system)
    service.inquiry_start(e["id"], True)
    service.inquiry_processing(e["id"])
    result = service.inquiry_reply(e["id"], record)
    assert result["level"] == expected
    assert result["state"] == "open"
    assert result["inquiry"]["result"]["escalate"] == (expected == "EMERGENCY")


def test_button_does_not_hide_spoken_help(system):
    _, service = system
    event = candidate(system)
    service.inquiry_start(event["id"], True)
    service.action(event["id"], "safe")
    result = service.inquiry_reply(event["id"], {"status": "RESPONDED", "intent": "HELP"})
    assert result["level"] == "EMERGENCY"


def test_unplayed_prompt_never_escalates(system):
    _, service = system
    event = candidate(system)
    service.inquiry_start(event["id"], False)
    assert service.store.event(event["id"])["inquiry"]["status"] == "audio_unavailable"
    service.tick()
    assert service.store.event(event["id"])["level"] == "PENDING"


def test_reply_drops_transcript_and_resolution_is_explicit(system):
    store, service = system
    e = candidate(system)
    service.inquiry_start(e["id"], True)
    result = service.inquiry_reply(e["id"], {"status": "RESPONDED", "intent": "OTHER", "transcript": "private speech"})
    assert "private speech" not in json.dumps(result)
    result = service.action(e["id"], "resolve", outcome="false_alarm", note="核实为正常活动")
    assert result["state"] == "resolved" and result["response_s"] >= 0
    with pytest.raises(ValueError):
        service.action(e["id"], "escalate")


def test_concurrent_revision_and_reopen_help(system):
    store, service = system
    e = candidate(system)
    service.action(e["id"], "acknowledge", revision=e["revision"])
    with pytest.raises(ValueError):
        service.action(e["id"], "resolve", outcome="safe", revision=e["revision"])
    service.action(e["id"], "resolve", outcome="safe")
    new = service.inquiry_reply(e["id"], {"status": "RESPONDED", "intent": "HELP"})
    assert new["id"] != e["id"] and new["level"] == "EMERGENCY"
    assert store.event(e["id"])["state"] == "resolved"


def test_transport_and_asr_failure_not_silence(system):
    store, service = system
    e = candidate(system)
    service.inquiry_start(e["id"], True)
    service.inquiry_processing(e["id"])
    store.update_event(e["id"], lambda item: item["inquiry"].update(processing_deadline=time.time()-1))
    service.tick()
    updated = store.event(e["id"])
    assert updated["level"] == "PENDING"
    assert updated["inquiry"]["result"]["state"] == "audio_unavailable"


def test_coverage_restart_does_not_claim_offline_time_as_active(system):
    store, _ = system
    store.set_coverage("vision", "active", "实时画面有效")
    stamp = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    with store.connection() as conn:
        conn.execute("UPDATE coverage SET started_at=?", ((datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat(),))
        conn.execute("INSERT INTO runtime_state VALUES (1,?)", (stamp,))
    store.recover_coverage()
    rows = store.coverage()
    active = next(c for c in rows if c["state"] == "active")
    assert active["ended_at"] == stamp
    assert any(c["reason"] == "服务离线区间" for c in rows)


def test_csv_adapter_is_only_format_conversion():
    native = canonical_csv(b"0,1,0,0,1\n" + b"\n".join(f"{i*20},1,0,0,1".encode() for i in range(1,13)))
    generic = canonical_csv(b"t,ax,ay,az\n" + b"\n".join(f"{i*.02},0,0,1".encode() for i in range(13)))
    import numpy as np
    assert np.allclose(np.loadtxt(__import__("io").BytesIO(native), delimiter=","), np.loadtxt(__import__("io").BytesIO(generic), delimiter=","))
    with pytest.raises(ValueError):
        canonical_csv(b"\n".join(f"{i},0,0,1".encode() for i in range(15)))
    with pytest.raises(ValueError):
        canonical_csv(b"t,ax,ay,az\n" + b"\n".join(f"{i},nan,0,1".encode() for i in range(15)))


def test_notice_delivery_has_real_timestamps(system):
    store, _ = system
    n = store.notify(None, {"title": "test"})
    assert store.notifications()[0]["status"] == "queued"
    delivered = store.notifications(deliver=True)[0]
    assert delivered["delivered_at"]
    store.acknowledge_notification(n["id"])
    assert store.notifications()[0]["seen_at"]


def test_recorded_audio_does_not_claim_continuous_coverage(system, monkeypatch):
    store, _ = system
    settings = store.settings()
    settings["audio_consent"] = True
    store.save_settings(settings)
    service = System(store)
    monkeypatch.setattr(service.modules["audio"], "call", lambda *args, **kwargs: {"events": []})
    try:
        service.audio_analyze(b"recording", "sounds", "live")
        service.check_coverage()
        assert service.audio_last_received is None
        current = next(c for c in store.coverage() if c["channel"] == "audio" and c["ended_at"] is None)
        assert current["state"] == "unavailable"
    finally:
        service.close()


def test_camera_session_ends_when_browser_heartbeat_expires(system, monkeypatch):
    store, _ = system
    service = System(store)
    record = {"session_id": "camera-session", "exercise": "squat", "target_reps": 10,
              "status": "paused", "count": 0, "mode": "live", "created_at": datetime.now(timezone.utc).isoformat()}
    store.save_session(record)
    service.session_metadata["camera-session"] = {"active": True, "mode": "live", "shared_camera": True,
                                                    "last_viewer": time.monotonic() - 20}
    ended = []
    def control(sid, action):
        ended.append((sid, action))
        service.session_metadata[sid]["active"] = False
    monkeypatch.setattr(service, "session_control", control)
    try:
        service.video = {"state": "running", "frames": 50}
        service.check_sessions()
        assert ended == [("camera-session", "end")]
        assert service.video["state"] == "running"
    finally:
        service.close()


def test_voice_conclusion_marks_answered_and_high_risk(system):
    """语音问询结论：有回答记「已应答」，问过但没答记「高风险」。"""
    _, service = system
    answered = candidate(system)
    service.inquiry_start(answered["id"], True)
    service.inquiry_processing(answered["id"])
    answered = service.inquiry_reply(answered["id"], {"status": "RESPONDED", "intent": "REASSURANCE", "asr_avg_logprob": -.1})
    voice = voice_inquiry(answered)
    assert (voice["state"], voice["label"]) == (ANSWERED, "已应答")
    assert voice["answered"] is True and voice["high_risk"] is False

    calling_help = candidate(system)
    service.inquiry_start(calling_help["id"], True)
    service.inquiry_processing(calling_help["id"])
    calling_help = service.inquiry_reply(calling_help["id"], {"status": "RESPONDED", "intent": "HELP"})
    assert voice_inquiry(calling_help)["kind"] == "help"

    silent = candidate(system)
    service.inquiry_start(silent["id"], True)
    service.inquiry_processing(silent["id"])
    silent = service.inquiry_reply(silent["id"], {"status": "NO_RESPONSE"})
    voice = voice_inquiry(silent)
    assert (voice["state"], voice["label"]) == (HIGH_RISK, "高风险")
    assert voice["high_risk"] is True and voice["answered"] is False


def test_voice_conclusion_keeps_unasked_separate_from_high_risk(system):
    """尚未问询、正在问询、问询音频不可用，都只记「未问询」，不计入高风险。"""
    store, service = system
    pending = candidate(system)
    assert voice_inquiry(pending)["label"] == "未问询"

    listening = service.inquiry_start(pending["id"], True)
    assert voice_inquiry(listening)["label"] == "未问询"

    processing = service.inquiry_processing(pending["id"])
    assert voice_inquiry(processing)["label"] == "未问询"

    unavailable = service.inquiry_reply(pending["id"], {"status": "audio_unavailable"})
    voice = voice_inquiry(unavailable)
    assert voice["state"] == NOT_ASKED
    assert voice["high_risk"] is False
    assert voice["reason"]

    counts = voice_counts(store.events())
    assert counts[NOT_ASKED] == 1 and counts[HIGH_RISK] == 0 and counts[ANSWERED] == 0


def test_records_delete_is_limited_to_selection(system):
    """删除只作用于勾选的记录，其余事件与通知保持原样。"""
    store, service = system
    kept = service.create("vision", {"prob": .8}, "replay")
    dropped = service.create("accel", {"score": .9}, "replay")
    assert len(store.notifications()) == 2

    removed = Records(store).events([dropped["id"]])
    assert removed["events"] == 1 and removed["notifications"] == 1
    assert [event["id"] for event in store.events()] == [kept["id"]]
    assert len(store.notifications()) == 1

    assert Records(store).events([])["events"] == 0
    assert len(store.events()) == 1


def test_report_scope_clear_uses_the_same_window_as_report(system):
    """清空范围与报告窗口一致：窗口外的事件不受影响。"""
    store, service = system
    service.create("vision", {"prob": .8}, "replay")
    records = Records(store)
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()

    assert records.scope("replay", future)["events"] == 0
    assert records.scope("live", "2000-01-01T00:00:00+00:00")["events"] == 0
    assert len(store.events()) == 1

    assert records.scope("replay", "2000-01-01T00:00:00+00:00")["events"] == 1
    assert store.events() == []
