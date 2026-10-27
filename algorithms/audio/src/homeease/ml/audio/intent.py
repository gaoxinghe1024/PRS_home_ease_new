"""Response intent: HELP / REASSURANCE / OTHER / UNCERTAIN (English and Chinese).

NO_RESPONSE is not an intent: it is a workflow state set when the inquiry window ends
without usable speech. Conservative rules: any help cue wins over reassurance
("I'm fine but I can't get up" -> HELP); negated reassurance counts as HELP.
An intent is evidence only - it never closes or downgrades an event.
"""

from __future__ import annotations

import re
from typing import Any

INTENTS = ["HELP", "REASSURANCE", "OTHER", "UNCERTAIN"]

HELP_EN = [r"\bhelp\b", r"\bi fell\b", r"\bfallen\b", r"can'?t get up", r"cannot get up", r"\bhurts?\b", r"\bpain\b", r"\bambulance\b", r"\bcall (someone|my|a)\b", r"\bnot (ok|okay|fine|alright|all right)\b", r"\bi need\b"]
HELP_ZH = ["\u6551\u547d", "\u5e2e\u5e2e\u6211", "\u5e2e\u6211", "\u9700\u8981\u5e2e\u52a9", "\u6454\u5012", "\u8d77\u4e0d\u6765", "\u75bc", "\u75db", "\u6551\u62a4\u8f66", "\u4e0d\u8212\u670d", "\u4e0d\u597d"]
OK_EN = [r"\bi'?m (fine|ok|okay|alright|all right|good)\b", r"\bi am (fine|ok|okay|alright|all right|good)\b", r"\bno problem\b", r"\bdon'?t worry\b", r"\bnot hurt\b", r"^\s*(fine|ok|okay)\s*$", r"\bi'?m not hurt\b"]
OK_ZH = ["\u6211\u6ca1\u4e8b", "\u6ca1\u4e8b", "\u6211\u5f88\u597d", "\u4e0d\u7528\u62c5\u5fc3", "\u6ca1\u5173\u7cfb", "\u6211\u8fd8\u597d", "\u6ca1\u95ee\u9898", "\u633a\u597d"]


def normalise(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^\w\s'\u4e00-\u9fff]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class RuleIntent:
    name = "rule"
    version = "1.0"

    def predict(self, text: str, asr_conf: float | None = None) -> dict[str, Any]:
        t = normalise(text or "")
        if not t:
            return {"intent": "UNCERTAIN", "reason": "empty_transcript"}
        help_hit = any(re.search(p, t) for p in HELP_EN) or any(k in t for k in HELP_ZH)
        ok_hit = any(re.search(p, t) for p in OK_EN) or any(k in t for k in OK_ZH)
        if "\u4e0d\u597d" in t or re.search(r"\bnot (ok|okay|fine|alright)\b", t):
            ok_hit = False
            help_hit = True
        if asr_conf is not None and asr_conf < -1.2 and not (help_hit or ok_hit):
            return {"intent": "UNCERTAIN", "reason": "low_asr_confidence"}
        if help_hit:
            return {"intent": "HELP", "reason": "help_cue" + ("_overrides_reassurance" if ok_hit else "")}
        if ok_hit:
            return {"intent": "REASSURANCE", "reason": "reassurance_cue"}
        return {"intent": "OTHER", "reason": "no_cue"}


def phrase_corpus() -> list[tuple[str, str]]:
    """Synthetic training phrases (text only) for the supervised baseline."""
    help_ = ["help", "help me", "please help", "i need help", "i fell down", "i can't get up", "my hip hurts", "it hurts", "call an ambulance", "call my daughter",
             "\u6551\u547d", "\u5e2e\u5e2e\u6211", "\u5feb\u6765\u5e2e\u6211", "\u6211\u9700\u8981\u5e2e\u52a9", "\u6211\u6454\u5012\u4e86", "\u6211\u8d77\u4e0d\u6765", "\u597d\u75bc", "\u817f\u5f88\u75db", "\u53eb\u6551\u62a4\u8f66", "\u6211\u4e0d\u8212\u670d"]
    ok = ["i'm fine", "i am fine", "i'm okay", "i'm ok", "i'm alright", "i'm all right", "no problem", "don't worry", "i'm not hurt", "i'm good thanks",
          "\u6211\u6ca1\u4e8b", "\u6ca1\u4e8b\u7684", "\u6211\u5f88\u597d", "\u4e0d\u7528\u62c5\u5fc3", "\u6ca1\u5173\u7cfb", "\u6211\u8fd8\u597d", "\u6ca1\u95ee\u9898", "\u6211\u633a\u597d\u7684", "\u653e\u5fc3\u5427\u6211\u6ca1\u4e8b", "\u4e0d\u7528\u7ba1\u6211"]
    other = ["what time is it", "the weather is nice today", "turn on the television", "where are my glasses", "dinner is ready", "who is at the door",
             "\u73b0\u5728\u51e0\u70b9\u4e86", "\u4eca\u5929\u5929\u6c14\u4e0d\u9519", "\u628a\u7535\u89c6\u6253\u5f00", "\u6211\u7684\u773c\u955c\u5462", "\u5403\u996d\u4e86", "\u8c01\u5728\u95e8\u53e3", "hello", "\u4f60\u597d", "yes", "\u662f\u7684", "no", "\u4e0d\u662f"]
    return [(p, "HELP") for p in help_] + [(p, "REASSURANCE") for p in ok] + [(p, "OTHER") for p in other]


class SupervisedIntent:
    name = "char_ngram_lr"
    version = "1.0"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.pipe = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3)), LogisticRegression(max_iter=2000, class_weight="balanced"))

    def fit(self, pairs: list[tuple[str, str]]) -> "SupervisedIntent":
        self.pipe.fit([normalise(t) for t, _ in pairs], [y for _, y in pairs])
        return self

    def predict(self, text: str, asr_conf: float | None = None) -> dict[str, Any]:
        t = normalise(text or "")
        if not t:
            return {"intent": "UNCERTAIN", "reason": "empty_transcript"}
        probs = self.pipe.predict_proba([t])[0]
        k = int(probs.argmax())
        if probs[k] < 0.45:
            return {"intent": "UNCERTAIN", "reason": "low_classifier_confidence", "p": float(probs[k])}
        return {"intent": str(self.pipe.classes_[k]), "reason": "classifier", "p": float(probs[k])}
