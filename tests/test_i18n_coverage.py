"""Every Chinese text the pages and the server put in front of a person has an English phrase in web/static/i18n-en.js."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "web" / "static" / "i18n-en.js"
CJK = r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]"
LETTER = re.compile(r"[\u4e00-\u9fff]")
RUN = re.compile(CJK + r"+(?:[ \u00b7A-Za-z0-9./%+\-:]*" + CJK + r"+)*")
NOT_SHOWN = {"i18n.js", "i18n-en.js"}


def table() -> dict:
    text = TABLE.read_text(encoding="utf-8")
    pairs = re.findall(r'^\s*("(?:[^"\\]|\\.)*"):\s*("(?:[^"\\]|\\.)*"),?\s*$', text, flags=re.M)
    return {json.loads(k): json.loads(v) for k, v in pairs}


def decode(text: str) -> str:
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    return re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), text)


def strip_comments(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"(?m)(^|[\s;{}()])//[^\n]*", r"\1", text)


def web_runs():
    files = sorted((ROOT / "web" / "templates").glob("*.html")) + [p for p in sorted((ROOT / "web" / "static").glob("*.js")) if p.name not in NOT_SHOWN]
    for path in files:
        text = strip_comments(decode(path.read_text(encoding="utf-8")))
        text = re.sub(r"\[[^\]]*[\u4e00-\u9fff]-[\u4e00-\u9fff][^\]]*\]", " ", text)  # character classes of a pattern are not text
        for m in RUN.finditer(text):
            yield path.name, m.group(0)


def python_runs():
    for path in sorted((ROOT / "care").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                for m in RUN.finditer(node.value):
                    yield "%s:%d" % (path.relative_to(ROOT).as_posix(), node.lineno), m.group(0)


def uncovered(runs, phrases):
    keys = sorted(phrases, key=len, reverse=True)
    rx = re.compile("|".join(re.escape(k) for k in keys))
    missing = {}
    for where, run in runs:
        if run in phrases or not LETTER.search(rx.sub(" ", run)):
            continue
        missing.setdefault(run, where)
    return missing


def test_the_phrase_table_is_english_only_and_ascii():
    text = TABLE.read_text(encoding="utf-8")
    assert all(ord(c) < 128 for c in text), "i18n-en.js is kept as ASCII with unicode escapes"
    phrases = table()
    assert len(phrases) > 500
    assert not [k for k, v in phrases.items() if re.search(CJK, v)]
    # a counter word (the Chinese for 'items', 'pieces') has no English counterpart and is dropped on purpose
    assert sorted(k for k, v in phrases.items() if not v.strip()) == sorted(["\u4e2a", "\u6761"])


def test_every_text_on_the_pages_has_an_english_phrase():
    missing = uncovered(web_runs(), table())
    assert not missing, "no English phrase for: " + "; ".join("%s (%s)" % (k, v) for k, v in sorted(missing.items())[:40])


def test_every_text_the_server_sends_has_an_english_phrase():
    missing = uncovered(python_runs(), table())
    assert not missing, "no English phrase for: " + "; ".join("%s (%s)" % (k, v) for k, v in sorted(missing.items())[:40])
