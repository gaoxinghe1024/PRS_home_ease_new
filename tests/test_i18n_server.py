"""The server's English (CSV export, webhooks) is the same as the pages': care/i18n.py and web/static/i18n.js give identical output."""
import ast
import csv
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from care import i18n
from care.services import System
from care.storage import Store

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
JS = """
global.window = global;
require(process.argv[1] + '/web/static/i18n-en.js');
const {translate} = require(process.argv[1] + '/web/static/i18n.js');
let input = '';
process.stdin.on('data', chunk => input += chunk).on('end', () => process.stdout.write(JSON.stringify(JSON.parse(input).map(translate))));
"""


def server_texts():
    """Every string constant of the server code that contains Chinese, whole (not only its Chinese runs)."""
    for path in sorted((ROOT / "care").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and i18n.CJK.search(node.value):
                yield node.value


def samples():
    table, _ = i18n.phrases()
    keys = [k for k in table if len(k) > 1]
    texts = list(table) + list(server_texts())
    # phrases joined the way messages are built: Chinese punctuation, numbers, names and blanks around them
    for a, b in zip(keys[::7], keys[3::7]):
        texts += [f"{a}\uff0c{b}\u3002", f"  {a} 3 {b}  ", f"{a}\uff08{b}\uff09", f"Room 2 \u00b7 {a}", f"{a}\uff1a12 \u79d2"]
    return texts + ["", "Emergency", "12:30", "\u5ba2\u5385 \u00b7 \u65b0\u4e8b\u4ef6", "\u3000", " \ufeff\u5ba2\u5385\u3000"]


@pytest.mark.skipif(NODE is None, reason="node is needed to run the page translator")
def test_server_and_page_translators_agree():
    texts = samples()
    run = subprocess.run([NODE, "-e", JS, str(ROOT)], input=json.dumps(texts), capture_output=True, text=True, encoding="utf-8", check=True)
    page = json.loads(run.stdout)
    different = [(t, p, i18n.translate(t)) for t, p in zip(texts, page) if i18n.translate(t) != p]
    assert len(texts) > 1000 and not different, different[:10]


def test_text_without_chinese_and_non_text_pass_unchanged():
    for value in ["Emergency", "", "a 3 b", None, 12]:
        assert i18n.translate(value) == value
    assert not i18n.CJK.search(i18n.translate("\u7d27\u6025\u6c42\u52a9"))  # "emergency call for help"


def test_webhooks_carry_english_text_and_the_record_keeps_the_original(tmp_path, monkeypatch):
    store = Store(tmp_path / "test.sqlite3")
    settings = store.settings()
    settings["webhook_url"] = "https://receiver.example/hook"
    store.save_settings(settings)
    store.notify(None, {"title": "\u7d27\u6025\u6c42\u52a9", "message": "\u5ba2\u5385 \u00b7 \u65b0\u4e8b\u4ef6", "mode": "live"},
                 kind="webhook")
    sent = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json, headers):
            sent.append(json)
            return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr("care.services.httpx.Client", Client)
    System.deliver_webhooks(SimpleNamespace(store=store))
    assert len(sent) == 1
    assert not i18n.CJK.search(sent[0]["title"] + sent[0]["message"]), sent[0]
    assert sent[0]["mode"] == "live"
    with store.connection() as conn:
        status, payload = conn.execute("SELECT status, payload FROM notifications WHERE kind='webhook'").fetchone()
    assert status == "delivered" and json.loads(payload)["title"] == "\u7d27\u6025\u6c42\u52a9"


def run_main(tmp_path, code):
    """Import the application in a child process with its own data directory (no algorithm process is started)."""
    env = {**os.environ, "HOMEEASE_SYSTEM_DATA": str(tmp_path / "runtime"), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", check=True).stdout


def test_the_api_document_is_english(tmp_path):
    doc = json.loads(run_main(tmp_path, "import json; from care.main import app; print(json.dumps(app.openapi(), ensure_ascii=False))"))
    assert doc["info"]["title"] == "HomeEase Home Care"
    assert not i18n.CJK.search(json.dumps(doc, ensure_ascii=False))


def test_the_csv_export_is_english(tmp_path):
    code = """
import types
from care import main
from care.inquiry import voice_inquiry
event = {"id": "e1", "created_at": "2026-10-23T08:00:00+00:00", "mode": "live", "level": "EMERGENCY", "outcome": None, "state": "open",
         "withdrawn": True, "inquiry": {"status": "completed", "record": {"status": "NO_RESPONSE"}, "completed_at": "2026-10-23T08:00:12+00:00"}}
asked = {**event, "id": "e2", "withdrawn": False, "inquiry": {"status": "listening"}}
for item in (event, asked):
    item["voice"] = voice_inquiry(item)
session = {"session_id": "s1", "created_at": "2026-10-23T09:00:00+00:00", "mode": "live", "exercise": "squat", "count": 8, "status": "ended"}
main.system = types.SimpleNamespace(report=lambda days, mode: {"events": [event, asked], "sessions": [session]})
import sys
sys.stdout.buffer.write(main.export_csv(7, "live").body)
"""
    text = run_main(tmp_path, code).lstrip("﻿")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][0] == "Type" and [r[0] for r in rows[1:4]] == ["Event", "Event", "Training"]
    assert "withdrawn" in rows[1][8]
    assert not i18n.CJK.search(text), text


def test_defaults_are_english_and_old_chinese_defaults_read_as_english(tmp_path):
    store = Store(tmp_path / "test.sqlite3")
    assert store.settings()["resident"]["room"] == "Living room"
    assert store.exercise_plan()["title"] == "Today's movement plan"
    # a database of an earlier version: its untouched Chinese defaults read as English, what a person typed stays as it is
    settings = store.settings()
    settings["resident"]["room"] = "客厅"
    settings["contacts"] = [{"name": "A", "relation": "家属", "phone": "1"}, {"name": "B", "relation": "女儿", "phone": "2"}]
    store.save_settings(settings)
    store.save_exercise_plan({**store.exercise_plan(), "title": "今天的运动计划"})
    settings = store.settings()
    assert settings["resident"]["room"] == "Living room"
    assert [c["relation"] for c in settings["contacts"]] == ["Family", "女儿"]
    assert store.exercise_plan()["title"] == "Today's movement plan"
    settings["resident"]["room"] = "卧室"
    store.save_settings(settings)
    assert store.settings()["resident"]["room"] == "卧室"
