"""Phrase bank: the shipped file satisfies every evaluation rule, and the validator rejects each way of breaking them."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import yaml

from homeease.ml.audio import phrases as P


@pytest.fixture(scope="module")
def bank() -> P.PhraseBank:
    return P.load_phrase_bank()


def _raw() -> dict[str, Any]:
    return yaml.safe_load(P.default_path().read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the shipped file
def test_shipped_bank_is_valid_and_sized(bank: P.PhraseBank) -> None:
    assert bank.schema_version == 1 and P.validate(bank) == []
    c = bank.counts()
    for role in P.ROLES:
        for lang in P.LANGS:
            for intent in P.INTENTS:
                s = c[f"{role}/{lang}/{intent}"]
                assert s["dev_phrases"] + s["test_phrases"] >= P.MIN_PHRASES[role]
                assert s["dev_families"] >= 2 and s["test_families"] >= 2
                if role == "self":
                    assert s["dev_families"] + s["test_families"] >= 8


def test_every_family_lives_in_exactly_one_split_role_intent_and_language(bank: P.PhraseBank) -> None:
    for fam, ents in bank.families().items():
        assert len({(e.split, e.speaker_role, e.intent, e.lang) for e in ents}) == 1, fam
    dev, test = {e.family for e in bank.select(split="dev")}, {e.family for e in bank.select(split="test")}
    assert dev and test and not dev & test


def test_no_text_in_both_splits_and_no_close_paraphrase_across_splits(bank: P.PhraseBank) -> None:
    dev = {(e.lang, P.norm_text(e.text)) for e in bank.select(split="dev")}
    test = {(e.lang, P.norm_text(e.text)) for e in bank.select(split="test")}
    assert not dev & test
    assert P.near_duplicates(bank, threshold=0.6) == []


def test_splits_are_balanced_and_stratified(bank: P.PhraseBank) -> None:
    n_dev, n_test = len(bank.select(split="dev")), len(bank.select(split="test"))
    assert abs(n_dev - n_test) <= 0.1 * len(bank.entries)
    for key, s in bank.counts().items():
        assert abs(s["dev_phrases"] - s["test_phrases"]) <= 2, key


def test_required_content_is_present(bank: P.PhraseBank) -> None:
    def texts(role: str, intent: str, lang: str) -> list[str]:
        return [P.norm_text(e.text) for e in bank.select(role=role, intent=intent, lang=lang)]

    sh_en, sh_zh = texts("self", "HELP", "en"), texts("self", "HELP", "zh")
    assert "ow" in sh_en and any("\u54ce\u54df" in t or "\u597d\u75db" in t for t in sh_zh)  # short cries
    assert any("not okay" in t or "not alright" in t or "not feeling fine" in t for t in sh_en)  # negations
    assert any("ambulance" in t for t in sh_en) and any("\u6551\u62a4\u8f66" in t or "\u6025\u6551\u8f66" in t for t in sh_zh)
    assert "i'm fine but i can't get up" in sh_en  # mixed sentence is HELP
    sr_en, sr_zh = texts("self", "REASSURANCE", "en"), texts("self", "REASSURANCE", "zh")
    assert "i'm okay lah" in sr_en and "\u6ca1\u4e8b\u5566" in sr_zh  # Singapore-flavoured
    so_en = texts("self", "OTHER", "en")
    assert "um" in so_en and "yes" in so_en and "no" in so_en and any(t.startswith(("what", "who", "is it")) for t in so_en)
    assert any(e.lang == "zh" for e in bank.select(role="visitor", intent="HELP"))
    assert any("resting" in t for t in texts("visitor", "REASSURANCE", "en"))


def test_dev_and_test_both_contain_the_hard_types(bank: P.PhraseBank) -> None:
    """Negations, mixed sentences and Singlish are not concentrated in one split."""
    for split in ("dev", "test"):
        en = [P.norm_text(e.text) for e in bank.select(split=split, role="self", intent="HELP", lang="en")]
        assert any(" not " in f" {t} " for t in en), split
        assert any("but" in t for t in en), split
        lah = [e for e in bank.select(split=split, role="self", intent="REASSURANCE", lang="en") if "lah" in e.text.lower()]
        assert lah, split


def test_tv_lines_are_original_and_neutral(bank: P.PhraseBank) -> None:
    tv = bank.select(role="tv")
    assert len(tv) >= 48
    banned = ["trump", "obama", "netflix", "disney", "bbc", "cnn", "star wars", "harry potter"]
    assert not any(b in e.text.lower() for e in tv for b in banned)
    assert all(len(e.text) <= 80 for e in tv)  # generic one-line news / weather / drama sentences


# ---------------------------------------------------------------- the validator itself
def _fails(mutate: Any, match: str) -> None:
    raw = copy.deepcopy(_raw())
    mutate(raw)
    with pytest.raises(P.PhraseBankError, match=match):
        P.parse_phrase_bank(raw)


def test_validator_rejects_duplicates_and_leaks() -> None:
    _fails(lambda r: r["entries"].append(dict(r["entries"][0])), "duplicate")
    def same_text_other_split(r: dict[str, Any]) -> None:
        e = next(x for x in r["entries"] if x["split"] == "dev" and x["speaker_role"] == "self")
        other = next(x for x in r["entries"] if x["split"] == "test" and x["lang"] == e["lang"])
        other["text"] = e["text"].upper() + "!"  # same words after normalisation
    _fails(same_text_other_split, "duplicate text")


def test_validator_rejects_a_family_in_two_splits_or_two_intents() -> None:
    def split_family(r: dict[str, Any]) -> None:
        fam = r["entries"][0]["family"]
        sib = [x for x in r["entries"] if x["family"] == fam]
        sib[-1]["split"] = "test" if sib[0]["split"] == "dev" else "dev"
    _fails(split_family, "mixes")
    def mix_intent(r: dict[str, Any]) -> None:
        fam = r["entries"][0]["family"]
        [x for x in r["entries"] if x["family"] == fam][-1]["intent"] = "OTHER" if r["entries"][0]["intent"] != "OTHER" else "HELP"
    _fails(mix_intent, "mixes")


def test_validator_rejects_count_and_stratum_violations() -> None:
    def drop_stratum(r: dict[str, Any]) -> None:
        r["entries"] = [x for x in r["entries"] if not (x["speaker_role"] == "tv" and x["lang"] == "zh" and x["intent"] == "HELP")]
    _fails(drop_stratum, r"tv/zh/HELP: 0 phrases")
    def no_test_families(r: dict[str, Any]) -> None:
        for x in r["entries"]:
            if x["speaker_role"] == "visitor" and x["lang"] == "en" and x["intent"] == "OTHER":
                x["split"] = "dev"
    _fails(no_test_families, r"visitor/en/OTHER: 0 test families")
    def few_self(r: dict[str, Any]) -> None:
        drop = [x for x in r["entries"] if x["speaker_role"] == "self" and x["lang"] == "en" and x["intent"] == "OTHER"][:6]
        for d in drop:
            r["entries"].remove(d)
    _fails(few_self, r"self/en/OTHER")


def test_validator_rejects_schema_problems() -> None:
    _fails(lambda r: r.update(schema_version=2), "schema_version")
    _fails(lambda r: r["entries"][0].update(intent="MAYBE"), "schema error")
    _fails(lambda r: r["entries"][0].update(speaker_role="robot"), "schema error")
    _fails(lambda r: r["entries"][0].update(split="train"), "schema error")
    _fails(lambda r: r["entries"][0].update(lang="fr"), "schema error")
    _fails(lambda r: r["entries"][0].update(text=" "), "schema error")
    _fails(lambda r: r["entries"][0].update(extra="x"), "schema error")
    _fails(lambda r: r["entries"][0].pop("family"), "schema error")
    _fails(lambda r: r["entries"][0].update(id="bad id!"), "schema error")
    with pytest.raises(P.PhraseBankError, match="not found"):
        P.load_phrase_bank("does/not/exist.yaml")


def test_near_duplicate_lint_flags_close_pairs() -> None:
    raw = copy.deepcopy(_raw())
    d = next(x for x in raw["entries"] if x["split"] == "dev" and x["lang"] == "en" and len(x["text"].split()) >= 3)
    t = next(x for x in raw["entries"] if x["split"] == "test" and x["lang"] == "en")
    t["text"] = d["text"] + " now"
    bank = P.PhraseBank.model_validate(raw)
    assert any(a == d["id"] and b == t["id"] for a, b, _ in P.near_duplicates(bank, 0.6))
