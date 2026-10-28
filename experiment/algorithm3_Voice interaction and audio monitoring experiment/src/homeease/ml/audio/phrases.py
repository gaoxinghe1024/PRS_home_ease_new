"""Phrase bank for the spoken-reply experiments: loader and validation.

`config/audio_phrases.yaml` lists reply phrases with the intent of their WORDS (HELP / REASSURANCE / OTHER),
who says them (the monitored person, a visitor, a TV) and a split. Validation enforces the evaluation rules
before any experiment can use the file:

* a paraphrase family lives in exactly one split, one speaker role, one intent and one language;
* ids and (language, normalised text) are unique, so no phrase can sit in both splits;
* every (speaker_role, intent, lang) stratum has at least two families in dev and two in test;
* minimum phrase counts (self >= 24 and >= 8 families, visitor >= 8, tv >= 8 per intent and language).

`near_duplicates()` is a lint for paraphrases that are too close across splits (word / character-bigram Jaccard).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from homeease.core.paths import project_root

SCHEMA_VERSION = 1
Intent = Literal["HELP", "REASSURANCE", "OTHER"]
Role = Literal["self", "visitor", "tv"]
Lang = Literal["en", "zh"]
Split = Literal["dev", "test"]
INTENTS: tuple[str, ...] = ("HELP", "REASSURANCE", "OTHER")
ROLES: tuple[str, ...] = ("self", "visitor", "tv")
LANGS: tuple[str, ...] = ("en", "zh")
SPLITS: tuple[str, ...] = ("dev", "test")
MIN_PHRASES = {"self": 24, "visitor": 8, "tv": 8}  # per (intent, lang)
MIN_FAMILIES_SELF = 8  # per (intent, lang)
MIN_FAMILIES_PER_SPLIT = 2  # per (role, intent, lang)


class PhraseBankError(ValueError):
    """The phrase bank file violates the schema or an evaluation rule."""


class PhraseEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=3, max_length=60, pattern=r"^[A-Za-z0-9_.\-]+$")
    text: str = Field(min_length=1, max_length=200)
    lang: Lang
    intent: Intent
    speaker_role: Role
    family: str = Field(min_length=1, max_length=60, pattern=r"^[A-Za-z0-9_.\-]+$")
    split: Split

    @field_validator("text")
    @classmethod
    def _printable(cls, v: str) -> str:
        if v != v.strip() or re.search(r"[\x00-\x1f]", v):
            raise ValueError("text must not have leading/trailing blanks or control characters")
        return v


class PhraseBank(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    entries: list[PhraseEntry]

    def select(self, *, split: str | None = None, role: str | None = None, lang: str | None = None, intent: str | None = None) -> list[PhraseEntry]:
        return [e for e in self.entries if (split is None or e.split == split) and (role is None or e.speaker_role == role)
                and (lang is None or e.lang == lang) and (intent is None or e.intent == intent)]

    def families(self, split: str | None = None) -> dict[str, list[PhraseEntry]]:
        out: dict[str, list[PhraseEntry]] = defaultdict(list)
        for e in self.select(split=split):
            out[e.family].append(e)
        return dict(out)

    def counts(self) -> dict[str, dict[str, int]]:
        """{"<role>/<lang>/<intent>": {"dev_phrases", "test_phrases", "dev_families", "test_families"}}"""
        out: dict[str, dict[str, int]] = {}
        for e in self.entries:
            c = out.setdefault(f"{e.speaker_role}/{e.lang}/{e.intent}", {"dev_phrases": 0, "test_phrases": 0, "dev_families": 0, "test_families": 0})
            c[f"{e.split}_phrases"] += 1
        for (role, lang, intent), fams in _strata(self.entries).items():
            for split in SPLITS:
                out[f"{role}/{lang}/{intent}"][f"{split}_families"] = len({f for f, s in fams if s == split})
        return out


def norm_text(text: str) -> str:
    """Comparison form: lower case, no punctuation, single blanks (Chinese characters kept)."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^\w\s'\u4e00-\u9fff]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _strata(entries: Iterable[PhraseEntry]) -> dict[tuple[str, str, str], set[tuple[str, str]]]:
    out: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    for e in entries:
        out[(e.speaker_role, e.lang, e.intent)].add((e.family, e.split))
    return out


def validate(bank: PhraseBank) -> list[str]:
    """All rule violations as readable messages (empty list = valid)."""
    problems: list[str] = []
    if bank.schema_version != SCHEMA_VERSION:
        problems.append(f"schema_version {bank.schema_version} != {SCHEMA_VERSION}")
    seen_id: dict[str, int] = defaultdict(int)
    seen_text: dict[tuple[str, str], str] = {}
    fam_attr: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    for e in bank.entries:
        seen_id[e.id] += 1
        key = (e.lang, norm_text(e.text))
        if not key[1]:
            problems.append(f"{e.id}: text has no content after normalisation")
        elif key in seen_text:
            problems.append(f"duplicate text in {e.lang}: {e.id} repeats {seen_text[key]} ({e.text!r})")
        else:
            seen_text[key] = e.id
        fam_attr[e.family].add((e.speaker_role, e.intent, e.lang, e.split))
    problems += [f"duplicate id {i}" for i, n in seen_id.items() if n > 1]
    for f, attrs in fam_attr.items():
        if len(attrs) > 1:
            problems.append(f"family {f} mixes role/intent/lang/split: {sorted(attrs)}")
    strata = _strata(bank.entries)
    for role in ROLES:
        for lang in LANGS:
            for intent in INTENTS:
                fams = strata.get((role, lang, intent), set())
                n_phr = sum(1 for e in bank.entries if (e.speaker_role, e.lang, e.intent) == (role, lang, intent))
                tag = f"{role}/{lang}/{intent}"
                if n_phr < MIN_PHRASES[role]:
                    problems.append(f"{tag}: {n_phr} phrases < {MIN_PHRASES[role]}")
                if role == "self" and len({f for f, _ in fams}) < MIN_FAMILIES_SELF:
                    problems.append(f"{tag}: {len({f for f, _ in fams})} families < {MIN_FAMILIES_SELF}")
                for split in SPLITS:
                    k = len({f for f, s in fams if s == split})
                    if k < MIN_FAMILIES_PER_SPLIT:
                        problems.append(f"{tag}: {k} {split} families < {MIN_FAMILIES_PER_SPLIT}")
    return problems


def _grams(text: str, lang: str) -> set[str]:
    t = norm_text(text)
    if lang == "zh":
        chars = re.sub(r"\s+", "", t)
        return {chars[i : i + 2] for i in range(max(1, len(chars) - 1))}
    return set(t.split())


def near_duplicates(bank: PhraseBank, threshold: float = 0.8) -> list[tuple[str, str, float]]:
    """Cross-split pairs of the same language whose word (en) / character-bigram (zh) Jaccard reaches ``threshold``."""
    out = []
    dev = bank.select(split="dev")
    test = bank.select(split="test")
    for a in dev:
        ga = _grams(a.text, a.lang)
        for b in test:
            if a.lang != b.lang:
                continue
            gb = _grams(b.text, b.lang)
            j = len(ga & gb) / max(1, len(ga | gb))
            if j >= threshold:
                out.append((a.id, b.id, round(j, 3)))
    return out


def default_path() -> Path:
    return project_root() / "config" / "audio_phrases.yaml"


def parse_phrase_bank(data: Any) -> PhraseBank:
    """Validate an already-parsed mapping (schema and rules); raises PhraseBankError."""
    try:
        bank = PhraseBank.model_validate(data)
    except Exception as e:  # pydantic ValidationError carries every field problem
        raise PhraseBankError(f"phrase bank schema error: {e}") from e
    problems = validate(bank)
    if problems:
        raise PhraseBankError("phrase bank invalid:\n- " + "\n- ".join(problems[:40]) + (f"\n... {len(problems) - 40} more" if len(problems) > 40 else ""))
    return bank


def load_phrase_bank(path: str | Path | None = None) -> PhraseBank:
    p = Path(path) if path else default_path()
    if not p.is_file():
        raise PhraseBankError(f"phrase bank not found: {p}")
    return parse_phrase_bank(yaml.safe_load(p.read_text(encoding="utf-8")))
