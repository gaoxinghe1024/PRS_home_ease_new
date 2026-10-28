"""Class mapping v2 (``esc50-vivae-homeease-v2``): the 4-class sound-event task with a real HELP class.

v1 (``esc50-homeease-v1`` in audio_sets.py) stays untouched so that published numbers remain
reproducible; v2 only ADDS sources and keeps the v1 ESC-50 content mapping verbatim.

Sources and their roles
-----------------------
* ESC-50 (official folds)      IMPACT / ABNORMAL / DAILY exactly as v1. No HELP.
* VIVAE (11 speakers)          HELP = fear and pain vocalisations of moderate/strong/peak intensity.
                               Same-corpus NEGATIVES: anger -> ABNORMAL (loud agitated voice, not a call for
                               help); achievement, pleasure, surprise -> DAILY (everyday non-speech voice).
                               Low-intensity fear/pain are AMBIGUOUS (a quiet whimper is not a call for help
                               and not clearly a daily sound): kept in the manifest with class None, never
                               trained on or scored in the headline numbers.
* FSD50K subset (optional)     HELP = clips labelled Screaming. Same-corpus negatives: Laughter/Giggle/Cough/
                               Sneeze -> DAILY, Thump_and_thud/Knock -> IMPACT, Shatter -> ABNORMAL.
                               Shout/Yell clips without Screaming are NOT used (loud voice, not necessarily
                               distress: unknown label noise).
* LibriSpeech dev-clean-2      read speech -> DAILY (the monitor hears conversation and TV all day; a scream
                               detector that fires on speech is useless). Not older-adult / household speech.

Why the same-corpus negatives: if every HELP clip came from VIVAE and every other class from ESC-50, a
classifier could learn the corpus (microphone, studio, codec) instead of the sound. Here every corpus that
supplies HELP also supplies non-HELP clips of human voice or impacts, loudness is normalised per window, and
the leave-one-corpus-out HELP recall (experiments/audio_suite.py, A2) measures what is left of the confound.

Splits are by group BEFORE windowing: ESC-50 official folds (clips of one source recording share a fold),
VIVAE by speaker, FSD50K by Freesound uploader, LibriSpeech by speaker. No group is ever in two folds OF THE SAME CORPUS
(``check_group_folds``). Across corpora the group strings differ (``esc50:<source file>`` against ``fsd50k:<uploader>``), so one
Freesound recording can sit in two folds when it is both an ESC-50 source file and an FSD50K clip. ``freesound_overlap`` finds those
(none of them HELP; the measured effect on the A2 four-class score is in docs/results.md section 3.1). They are reported, not repaired:
moving a group would change every fold and every published A2 number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from homeease.datasets.adapters.audio_sets import ESC50_MAP, esc50_rows
from homeease.datasets.splits import LeakageError

MAPPING_VERSION_V2 = "esc50-vivae-homeease-v2"
CLASSES = ("HELP", "IMPACT", "ABNORMAL", "DAILY")
SOURCES = ("esc50", "vivae", "fsd50k", "librispeech")
N_FOLDS = 5

VIVAE_HELP_EMOTIONS = ("fear", "pain")
VIVAE_HELP_INTENSITIES = ("moderate", "strong", "peak")  # low fear/pain: ambiguous, excluded (class None)
VIVAE_NEGATIVES = {"anger": "ABNORMAL", "achievement": "DAILY", "pleasure": "DAILY", "surprise": "DAILY"}

FSD50K_MAP = {
    "HELP": ["Screaming"],
    "DAILY": ["Laughter", "Giggle", "Cough", "Sneeze"],
    "IMPACT": ["Thump_and_thud", "Knock"],
    "ABNORMAL": ["Shatter"],
}
# a negative clip must not carry any distress-like label (FSD50K clips are multi-labelled)
FSD50K_NEGATIVE_EXCLUDE = ("Screaming", "Shout", "Yell", "Crying_and_sobbing", "Whimper", "Gasp", "Groan", "Moan")


def map_vivae(emotion: str, intensity: str) -> str | None:
    if emotion in VIVAE_HELP_EMOTIONS:
        return "HELP" if intensity in VIVAE_HELP_INTENSITIES else None
    return VIVAE_NEGATIVES.get(emotion)


def map_fsd50k(labels: list[str]) -> str | None:
    """Class for a multi-labelled FSD50K clip; HELP wins, a negative needs the absence of distress labels."""
    ls = set(labels)
    if "Screaming" in ls:
        return "HELP"
    if ls & set(FSD50K_NEGATIVE_EXCLUDE):
        return None
    for cls in ("IMPACT", "ABNORMAL", "DAILY"):
        if ls & set(FSD50K_MAP[cls]):
            return cls
    return None


def esc50_v2_rows(raw_dir: Path) -> list[dict[str, Any]]:
    """ESC-50 mapped clips with v1 class membership (verbatim) and the official fold."""
    out = []
    for r in esc50_rows(raw_dir):
        if not r["homeease_class"]:
            continue
        out.append({
            "source": "esc50", "path": r["path"], "group": f"esc50:{r['src_file']}", "fold": int(r["fold"]), "homeease_class": r["homeease_class"],
            "subtype": r["category"], "mapping_version": MAPPING_VERSION_V2, "fold_official": int(r["fold"]),
        })
    # The official folds are meant to keep a source recording in one fold, but a few Freesound sources hold
    # two different events that were put into two folds (e.g. 131943: clock_tick in fold 2, clock_alarm in fold 3).
    # For the v2 task the whole source goes to its lowest official fold (v1 / the 3-class run keeps the official folds).
    lowest: dict[str, int] = {}
    for r in out:
        lowest[r["group"]] = min(lowest.get(r["group"], 99), r["fold"])
    for r in out:
        r["fold"] = lowest[r["group"]]
    return out


def librispeech_v2_rows(raw_dir: Path, per_speaker: int = 6, seed: int = 42) -> list[dict[str, Any]]:
    """A few utterances per speaker (DAILY, speech). Pure file scan: no audio is read here."""
    spk: dict[str, list[str]] = {}
    for p in sorted(Path(raw_dir).rglob("*.flac")):
        spk.setdefault(p.parts[-3], []).append(str(p))
    rng = np.random.default_rng(seed)
    out = []
    for s in sorted(spk):
        files = spk[s]
        pick = rng.choice(len(files), size=min(per_speaker, len(files)), replace=False)
        for i in sorted(pick.tolist()):
            out.append({
                "source": "librispeech", "path": files[i], "group": f"librispeech:{s}", "fold": None, "homeease_class": "DAILY",
                "subtype": "read_speech", "mapping_version": MAPPING_VERSION_V2,
            })
    return out


def assign_folds(rows: list[dict[str, Any]], n_folds: int = N_FOLDS, seed: int = 42) -> list[dict[str, Any]]:
    """Give every row without a fold (everything except ESC-50) the fold of its GROUP.

    Groups are placed greedily, largest first, into the fold with the fewest clips of that source, so
    every fold sees every source when the source has at least ``n_folds`` groups. Deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    by_src: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in rows:
        if r.get("fold") is None:
            by_src.setdefault(r["source"], {}).setdefault(r["group"], []).append(r)
    for src in sorted(by_src):
        groups = by_src[src]
        order = sorted(groups, key=lambda g: (-len(groups[g]), float(rng.random())))
        load = [0] * n_folds
        for g in order:
            k = int(np.argmin(load))
            load[k] += len(groups[g])
            for r in groups[g]:
                r["fold"] = k + 1
    return rows


def check_group_folds(rows: list[dict[str, Any]]) -> None:
    """Raise ``LeakageError`` if any group (speaker, uploader, source recording) appears in two folds."""
    seen: dict[str, int] = {}
    for r in rows:
        f = r.get("fold")
        if f is None:
            raise LeakageError(f"row without fold: {r.get('path')}")
        g = r["group"]
        if seen.setdefault(g, int(f)) != int(f):
            raise LeakageError(f"group {g} is in folds {seen[g]} and {f}")


def freesound_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Freesound recordings that occur both as an ESC-50 source file and as an FSD50K clip, with the folds they landed in.

    ``check_group_folds`` only compares the group strings of one corpus, so it cannot see these. ``pairs`` has one entry per (ESC-50 clip, FSD50K clip) pair;
    ``cross_fold`` counts the pairs whose two clips are in different folds (a clip of the test fold then has a sibling of the same recording in training);
    ``rows_by_fold`` counts the clips involved in such cross-fold pairs per fold."""
    fsd = {str(r["freesound_id"]): r for r in rows if r.get("source") == "fsd50k" and r.get("freesound_id") is not None}
    pairs: list[dict[str, Any]] = []
    for r in rows:
        if r.get("source") != "esc50":
            continue
        fid = str(r["group"]).split(":", 1)[1].split(".")[0]
        f = fsd.get(fid)
        if f is not None:
            pairs.append({"freesound_id": fid, "esc50_fold": int(r["fold"]), "fsd50k_fold": int(f["fold"]), "class": r["homeease_class"], "cross_fold": int(r["fold"]) != int(f["fold"])})
    by_fold: dict[int, int] = {}
    for p in pairs:
        if p["cross_fold"]:
            for k in (p["esc50_fold"], p["fsd50k_fold"]):
                by_fold[k] = by_fold.get(k, 0) + 1
    return {"pairs": pairs, "n_pairs": len(pairs), "cross_fold": sum(p["cross_fold"] for p in pairs), "rows_by_fold": dict(sorted(by_fold.items())),
            "help_pairs": sum(p["class"] == "HELP" for p in pairs)}


def combined_rows(data_root: Path, seed: int = 42, sources: tuple[str, ...] = SOURCES) -> list[dict[str, Any]]:
    """All available v2 rows (clips with a class) with group folds. Missing corpora are simply absent."""
    from homeease.datasets.adapters import fsd50k_help, vivae

    root = Path(data_root) / "raw"
    rows: list[dict[str, Any]] = []
    if "esc50" in sources and (root / "esc50" / "meta" / "esc50.csv").is_file():
        rows += esc50_v2_rows(root / "esc50")
    if "vivae" in sources and vivae.is_present(root / "vivae"):
        rows += [r for r in vivae.rows(root / "vivae") if r["homeease_class"]]
    if "fsd50k" in sources and fsd50k_help.is_present(root / "fsd50k_help"):
        rows += [r for r in fsd50k_help.rows(root / "fsd50k_help") if r["homeease_class"]]
    if "librispeech" in sources and (root / "librispeech_mini").is_dir():
        rows += librispeech_v2_rows(root / "librispeech_mini", seed=seed)
    assign_folds(rows, seed=seed)
    check_group_folds(rows)
    return rows


__all__ = [
    "MAPPING_VERSION_V2", "CLASSES", "SOURCES", "N_FOLDS", "map_vivae", "map_fsd50k", "esc50_v2_rows", "librispeech_v2_rows", "assign_folds",
    "check_group_folds", "freesound_overlap", "combined_rows", "ESC50_MAP", "VIVAE_HELP_EMOTIONS", "VIVAE_HELP_INTENSITIES", "VIVAE_NEGATIVES", "FSD50K_MAP",
]
