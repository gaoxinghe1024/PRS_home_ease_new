"""English text for what the server hands out directly (CSV export, webhook messages).

The pages are translated in the browser by ``web/static/i18n.js``; this module applies the same phrase table (``i18n-en.js``) with the
same rules, line by line, so a text reads the same on a page and in a file. ``tests/test_i18n_server.py`` compares both translators.
"""
import json
import re
from functools import lru_cache

from care.config import ROOT

TABLE = ROOT / "web" / "static" / "i18n-en.js"
CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")
# JavaScript's \s, so that leading and trailing blanks are cut exactly as in the browser.
WS = "[\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]"
PUNCT = {"\uff0c": ", ", "\u3002": ". ", "\uff1a": ": ", "\uff1b": "; ", "\uff1f": "? ", "\uff01": "! ", "\uff08": " (", "\uff09": ") ",
         "\u3001": ", ", "\u300c": "\"", "\u300d": "\"", "\u201c": "\"", "\u201d": "\"", "\u3000": " "}
MARK_A, MARK_B = "\ue000", "\ue001"


@lru_cache(maxsize=1)
def phrases():
    text = TABLE.read_text(encoding="utf-8")
    pairs = re.findall(r'^\s*("(?:[^"\\]|\\.)*"):\s*("(?:[^"\\]|\\.)*"),?\s*$', text, flags=re.M)
    table = {json.loads(k): json.loads(v) for k, v in pairs}
    keys = sorted(table, key=len, reverse=True)  # stable, like Array.prototype.sort: longer phrases first
    return table, re.compile("|".join(re.escape(k) for k in keys)) if keys else None


def _strip(text):
    return re.sub(f"^{WS}+|{WS}+$", "", text)


def translate(text):
    """Chinese text -> English, as the pages show it. Text without Chinese is returned unchanged."""
    if not isinstance(text, str) or not CJK.search(text):
        return text
    table, rx = phrases()
    lead = re.match(f"{WS}*", text).group(0)
    trail = re.search(f"{WS}*$", text).group(0)
    core = text[len(lead):len(text) - len(trail)]
    # a text ending with a colon or comma runs on into the next element ("Status: <span>Off</span>"): keep its blank
    gap = " " if not trail and re.search("[\uff1a\uff0c\uff1b\u3001]$", core) else ""
    if core in table:
        return lead + _strip(table[core]) + gap + trail
    out = rx.sub(lambda m: MARK_A + table[m.group(0)] + MARK_B, core) if rx else core
    # spaces between an English phrase and a neighbouring letter or digit, or the next phrase
    out = re.sub(r"([A-Za-z0-9.,;:!?)])\ue001\ue000(?=[A-Za-z0-9(])", "\\1 \ue001\ue000", out)
    out = re.sub(r"([A-Za-z0-9])\ue000(?=[A-Za-z0-9])", "\\1 \ue000", out)
    out = re.sub(r"([A-Za-z0-9.,;:!?)])\ue001(?=[A-Za-z0-9(])", "\\1 ", out)
    # a phrase in the middle of a sentence starts with a small letter
    out = re.sub(r"([a-z,;] ?)(\ue001?)\ue000([A-Z])(?=[a-z])", lambda m: m.group(1) + m.group(2) + MARK_A + m.group(3).lower(), out)
    out = out.replace(MARK_A, "").replace(MARK_B, "")
    out = re.sub("[" + "".join(PUNCT) + "]", lambda m: PUNCT[m.group(0)], out)
    out = re.sub(" {2,}", " ", out)
    out = re.sub(f"{WS}+([,.;:?!)])", "\\1", out)
    out = re.sub(f"\\({WS}+", "(", out)
    out = re.sub(r"([,;:!?])(?=[A-Za-z])", "\\1 ", out)
    out = re.sub(r"(\.)(?=[A-Z])", "\\1 ", out)
    return lead + _strip(out) + gap + trail
