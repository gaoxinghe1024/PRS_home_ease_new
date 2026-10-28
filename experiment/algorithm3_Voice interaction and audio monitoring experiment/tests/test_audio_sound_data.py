"""HELP-class corpora (VIVAE, FSD50K subset), class mapping v2 and the group splits. No network, no real data: tiny generated files."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sf = pytest.importorskip("soundfile", reason="needs the optional audio extra (uv sync --extra audio)")

from homeease.datasets.adapters import audio_sets, audio_v2, fsd50k_help, vivae
from homeease.datasets.splits import LeakageError

SR = 16000


def write_wav(path: Path, seconds: float = 0.5, sr: int = SR, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sf.write(str(path), (rng.standard_normal(int(seconds * sr)) * 0.1).astype(np.float32), sr)
    return path


# ------------------------------------------------------------------ mapping v1 is frozen, v2 extends it
def test_mapping_v1_is_unchanged() -> None:
    """Published numbers refer to this exact mapping; v2 must never edit it."""
    assert audio_sets.MAPPING_VERSION == "esc50-homeease-v1"
    assert audio_sets.ESC50_MAP == {
        "IMPACT": ["door_wood_knock"],
        "ABNORMAL": ["glass_breaking", "crying_baby", "siren", "clock_alarm"],
        "DAILY": [
            "vacuum_cleaner", "washing_machine", "toilet_flush", "water_drops", "pouring_water", "keyboard_typing", "mouse_click",
            "door_wood_creaks", "can_opening", "clock_tick", "brushing_teeth", "drinking_sipping", "footsteps", "laughing", "snoring",
            "breathing", "coughing", "sneezing", "clapping",
        ],
    }
    assert audio_sets.THREE_CLASS == {"HELP": "HELP", "IMPACT": "ABNORMAL", "ABNORMAL": "ABNORMAL", "DAILY": "DAILY"}
    assert audio_v2.MAPPING_VERSION_V2 != audio_sets.MAPPING_VERSION


def make_esc50(root: Path, rows: list[tuple[str, int, str, str]]) -> Path:
    (root / "meta").mkdir(parents=True)
    with open(root / "meta" / "esc50.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "fold", "target", "category", "esc10", "src_file", "take"])
        for i, (cat, fold, src, take) in enumerate(rows):
            name = f"{fold}-{src}-{take}-{i}.wav"
            w.writerow([name, fold, i, cat, "False", src, take])
            write_wav(root / "audio" / name, 0.3, seed=i)
    return root


def test_v2_keeps_the_v1_esc50_content_and_moves_straddling_sources(tmp_path: Path) -> None:
    root = make_esc50(tmp_path / "esc50", [
        ("door_wood_knock", 1, "100", "A"), ("glass_breaking", 2, "101", "A"), ("footsteps", 3, "102", "A"), ("dog", 4, "103", "A"),  # dog: unmapped
        ("clock_tick", 2, "131943", "A"), ("clock_alarm", 3, "131943", "A"),  # one Freesound source in two official folds
    ])
    v1 = {r["path"]: r["homeease_class"] for r in audio_sets.esc50_rows(root) if r["homeease_class"]}
    v2 = audio_v2.esc50_v2_rows(root)
    assert {r["path"]: r["homeease_class"] for r in v2} == v1  # same clips, same classes
    assert not any(r["subtype"] == "dog" for r in v2)
    straddle = [r for r in v2 if r["group"] == "esc50:131943"]
    assert {r["fold"] for r in straddle} == {2} and {r["fold_official"] for r in straddle} == {2, 3}
    audio_v2.check_group_folds(v2)
    # the v1 rows keep the official folds (the published 3-class run is not touched)
    assert {r["fold"] for r in audio_sets.esc50_rows(root) if r["src_file"] == "131943"} == {2, 3}


def test_vivae_and_fsd50k_class_mapping() -> None:
    assert audio_v2.map_vivae("fear", "peak") == "HELP" and audio_v2.map_vivae("pain", "moderate") == "HELP"
    assert audio_v2.map_vivae("fear", "low") is None and audio_v2.map_vivae("pain", "low") is None  # ambiguous, never trained or scored
    assert audio_v2.map_vivae("anger", "peak") == "ABNORMAL"
    assert {audio_v2.map_vivae(e, "strong") for e in ("achievement", "pleasure", "surprise")} == {"DAILY"}
    assert audio_v2.map_fsd50k(["Screaming", "Human_voice"]) == "HELP"
    assert audio_v2.map_fsd50k(["Screaming", "Shout"]) == "HELP"
    assert audio_v2.map_fsd50k(["Shout", "Yell"]) is None  # loud voice without Screaming: not used
    assert audio_v2.map_fsd50k(["Laughter", "Shout"]) is None  # a negative must not carry distress-like labels
    assert audio_v2.map_fsd50k(["Laughter", "Speech"]) == "DAILY" and audio_v2.map_fsd50k(["Thump_and_thud"]) == "IMPACT" and audio_v2.map_fsd50k(["Shatter"]) == "ABNORMAL"
    assert audio_v2.map_fsd50k(["Guitar"]) is None


# ------------------------------------------------------------------ VIVAE
def make_vivae_files(audio: Path, speakers: int = 6) -> int:
    n = 0
    for s in range(1, speakers + 1):
        for emo in ("fear", "pain", "anger", "achievement", "pleasure", "surprise"):
            for inten in ("low", "peak"):
                write_wav(audio / f"S{s:02d}_{emo}_{inten}_{n % 7 + 1:02d}.wav", 0.4, 44100, seed=n)
                n += 1
    return n


def test_vivae_name_parsing() -> None:
    assert vivae.parse_name("S04_surprise_peak_10.wav") == {"speaker": "S04", "emotion": "surprise", "intensity": "peak", "item": 10}
    assert vivae.parse_name("full_set/S11_fear_low_02.wav")["speaker"] == "S11"
    for bad in ("S04_joy_peak_10.wav", "S4_fear_peak_10.wav", "readme.txt", "S04_fear_peak_10.mp3"):
        assert vivae.parse_name(bad) is None


def test_vivae_rows_groups_and_classes(tmp_path: Path) -> None:
    n = make_vivae_files(tmp_path / "audio")
    assert vivae.is_present(tmp_path) and not vivae.is_present(tmp_path / "nothing")
    rows = vivae.prepare(tmp_path)
    assert len(rows) == n and {r["sr"] for r in rows} == {44100}
    by = {(r["emotion"], r["intensity"]): r["homeease_class"] for r in rows}
    assert by[("fear", "peak")] == "HELP" and by[("pain", "peak")] == "HELP" and by[("fear", "low")] is None
    assert by[("anger", "low")] == "ABNORMAL" and by[("pleasure", "low")] == "DAILY"
    assert all(r["group"] == f"vivae:{r['speaker']}" for r in rows)  # the split unit is the speaker
    assert {r["mapping_version"] for r in rows} == {audio_v2.MAPPING_VERSION_V2}


def test_vivae_download_skips_present_data_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    n = make_vivae_files(tmp_path / "audio")

    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("download_checked must not be called when the extracted corpus is present")

    monkeypatch.setattr(vivae, "download_checked", boom)
    out = vivae.download(tmp_path, expected_clips=n)
    assert out == {"status": "present", "clips": n, "downloaded_bytes": 0}


def test_vivae_download_extracts_full_set_and_deletes_the_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(url: str, dest: Path, **kw: Any) -> dict[str, Any]:
        wav = write_wav(tmp_path / "src.wav", 0.2, 44100)
        with zipfile.ZipFile(dest, "w") as z:
            z.write(wav, "VIVAE/full_set/S01_fear_peak_01.wav")
            z.write(wav, "VIVAE/full_set/S02_anger_low_03.wav")
            z.write(wav, "VIVAE/core_set/S01_fear_peak_01.wav")  # duplicate of a full-set file
            z.writestr("VIVAE/readme.txt", "x")
        return {"url": url, "path": str(dest), "status": "downloaded", "bytes": dest.stat().st_size, "sha256": "abc", "md5": "def"}

    monkeypatch.setattr(vivae, "download_checked", fake_download)
    raw = tmp_path / "raw"
    out = vivae.download(raw, expected_clips=2)
    assert out["status"] == "downloaded" and out["clips"] == 2 and out["downloaded_bytes"] > 0
    assert not (raw / "VIVAE.zip").exists()  # archive removed right after extraction
    assert sorted(p.name for p in (raw / "audio").iterdir()) == ["S01_fear_peak_01.wav", "S02_anger_low_03.wav"]
    log = json.loads((raw / "DOWNLOAD_LOG.json").read_text(encoding="utf-8"))
    assert log["meta"]["license"] == "CC BY-NC 4.0" and log["meta"]["archive_deleted"] is True and log["files"][0]["sha256"] == "abc" and log["files"][0]["url"].startswith("https://zenodo.org/")


def test_vivae_download_failure_is_reported_not_hidden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vivae, "download_checked", lambda url, dest, **k: {"url": url, "path": str(dest), "status": "failed", "error": "offline"})
    out = vivae.download(tmp_path / "raw")
    assert out["status"] == "failed" and "offline" in out["error"]
    assert not (tmp_path / "raw" / "audio").exists() or not list((tmp_path / "raw" / "audio").glob("*.wav"))


# ------------------------------------------------------------------ FSD50K subset
def fake_catalog() -> tuple[dict[str, list[tuple[str, list[str]]]], dict[str, dict[str, dict[str, Any]]]]:
    cc0, by, nc, samp = ("http://creativecommons.org/publicdomain/zero/1.0/", "http://creativecommons.org/licenses/by/3.0/", "http://creativecommons.org/licenses/by-nc/3.0/",
                         "http://creativecommons.org/licenses/sampling+/1.0/")
    gt: dict[str, list[tuple[str, list[str]]]] = {"dev": [], "eval": []}
    info: dict[str, dict[str, dict[str, Any]]] = {"dev": {}, "eval": {}}

    def add(split: str, cid: str, labels: list[str], uploader: str, lic: str) -> None:
        gt[split].append((cid, labels))
        info[split][cid] = {"title": f"clip {cid}", "uploader": uploader, "license": lic}

    for i in range(40):  # screams by 3 uploaders (cap per uploader must bite), one with a Sampling+ licence
        add("dev", f"{1000 + i}", ["Screaming", "Human_voice"], f"up{i % 3}", samp if i == 0 else (cc0 if i % 2 else by))
    for i in range(20):
        add("eval", f"{2000 + i}", ["Laughter", "Human_voice"], f"lau{i}", nc)
    add("eval", "3000", ["Laughter", "Shout"], "lau_x", cc0)  # distress-like co-label: dropped
    add("dev", "3001", ["Guitar"], "g", cc0)
    for i in range(10):
        add("dev", f"{4000 + i}", ["Thump_and_thud"], f"th{i}", by)
    return gt, info


def test_fsd50k_candidates_select_is_filtered_capped_and_deterministic() -> None:
    gt, info = fake_catalog()
    cands = fsd50k_help.candidates(gt, info)
    assert "1000" not in {c["id"] for c in cands["Screaming"]}  # Sampling+ licence excluded
    assert "3000" not in {c["id"] for v in cands.values() for c in v} and "3001" not in {c["id"] for v in cands.values() for c in v}
    assert {c["homeease_class"] for c in cands["Screaming"]} == {"HELP"} and {c["homeease_class"] for c in cands["Thump_and_thud"]} == {"IMPACT"}
    ids = {c["id"] for v in cands.values() for c in v}
    sizes = {i: 100_000 for i in ids}
    sizes["1001"] = 10_000  # below the size band
    sizes["1002"] = 900_000  # above the size band
    a = fsd50k_help.select(cands, sizes, seed=3)
    b = fsd50k_help.select(cands, sizes, seed=3)
    assert [c["id"] for c in a] == [c["id"] for c in b]  # deterministic
    assert "1001" not in {c["id"] for c in a} and "1002" not in {c["id"] for c in a}
    help_up = [c["uploader"] for c in a if c["subtype"] == "Screaming"]
    assert max(help_up.count(u) for u in set(help_up)) <= fsd50k_help.MAX_PER_UPLOADER  # 3 uploaders x cap 8
    assert len(help_up) == 3 * fsd50k_help.MAX_PER_UPLOADER
    assert all(c["bytes"] == sizes[c["id"]] for c in a)
    small = fsd50k_help.select(cands, sizes, seed=3, byte_budget=250_000)
    assert sum(c["bytes"] for c in small) <= 250_000  # total byte budget is a hard cap


def test_fsd50k_rows_from_generated_selection(tmp_path: Path) -> None:
    clips = [
        {"id": "11", "split": "dev", "labels": ["Screaming"], "subtype": "Screaming", "homeease_class": "HELP", "uploader": "alice", "license": "cc0", "title": "t", "bytes": 1, "sha256": "x"},
        {"id": "12", "split": "eval", "labels": ["Cough"], "subtype": "Cough", "homeease_class": "DAILY", "uploader": "bob", "license": "by", "title": "t", "bytes": 1, "sha256": "y"},
    ]
    (tmp_path / "selection.json").write_text(json.dumps({"clips": clips}), encoding="utf-8")
    assert not fsd50k_help.is_present(tmp_path)  # wav files missing
    for c in clips:
        write_wav(tmp_path / "audio" / f"{c['id']}.wav", 0.3, 44100)
    assert fsd50k_help.is_present(tmp_path)
    rows = fsd50k_help.prepare(tmp_path)
    assert [(r["homeease_class"], r["group"]) for r in rows] == [("HELP", "fsd50k:alice"), ("DAILY", "fsd50k:bob")]  # grouped by uploader


def test_fsd50k_download_skips_present_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "selection.json").write_text(json.dumps({"clips": [{"id": "5"}]}), encoding="utf-8")
    write_wav(tmp_path / "audio" / "5.wav")

    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("no network access expected")

    monkeypatch.setattr(fsd50k_help, "download_checked", boom)
    monkeypatch.setattr(fsd50k_help, "get_json", boom)
    assert fsd50k_help.download(tmp_path) == {"status": "present", "clips": 1, "downloaded_bytes": 0}


def test_audio_sets_download_does_not_refetch_extracted_corpora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ESC-50 and LibriSpeech archives were deleted on purpose: with the extracted files present nothing is downloaded."""
    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("fetch must not be called")

    monkeypatch.setattr(audio_sets, "fetch", boom)
    esc = make_esc50(tmp_path / "esc50", [("door_wood_knock", 1, "1", "A"), ("siren", 2, "2", "A")])
    out = audio_sets.download("esc50", esc)
    assert out["status"] == "present" and out["clips"] == 2 and out["downloaded_bytes"] == 0
    li = tmp_path / "libri"
    p = li / "LibriSpeech" / "dev-clean-2" / "84" / "121123" / "84-121123-0000.flac"
    p.parent.mkdir(parents=True)
    sf.write(str(p), np.zeros(1600, dtype=np.float32), SR)
    out = audio_sets.download("librispeech_mini", li)
    assert out["status"] == "present" and out["flac"] == 1
    (esc / "audio" / sorted(p.name for p in (esc / "audio").iterdir())[0]).unlink()  # one clip missing -> not complete -> would fetch
    assert not audio_sets.esc50_present(esc)
    with pytest.raises(AssertionError):
        audio_sets.download("esc50", esc)


# ------------------------------------------------------------------ group splits
def synthetic_rows(seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for i in range(300):  # ESC-50-like: official folds, 2-3 clips per source recording kept in one fold
        src = i // 3
        rows.append({"source": "esc50", "path": f"e{i}", "group": f"esc50:{src}", "fold": src % 5 + 1, "homeease_class": "DAILY"})
    for s in range(11):
        for j in range(int(rng.integers(60, 120))):
            rows.append({"source": "vivae", "path": f"v{s}_{j}", "group": f"vivae:S{s:02d}", "fold": None, "homeease_class": "HELP" if j % 3 == 0 else "DAILY"})
    for u in range(40):
        for j in range(int(rng.integers(1, 9))):
            rows.append({"source": "fsd50k", "path": f"f{u}_{j}", "group": f"fsd50k:u{u}", "fold": None, "homeease_class": "HELP"})
    for s in range(26):
        for j in range(6):
            rows.append({"source": "librispeech", "path": f"l{s}_{j}", "group": f"librispeech:{s}", "fold": None, "homeease_class": "DAILY"})
    return rows


def test_group_folds_never_split_a_speaker_uploader_or_source_recording() -> None:
    rows = audio_v2.assign_folds(synthetic_rows(), seed=7)
    audio_v2.check_group_folds(rows)
    folds_of: dict[str, set[int]] = {}
    for r in rows:
        folds_of.setdefault(r["group"], set()).add(r["fold"])
    assert all(len(v) == 1 for v in folds_of.values())
    for src in ("vivae", "fsd50k", "librispeech"):
        assert {r["fold"] for r in rows if r["source"] == src} == {1, 2, 3, 4, 5}  # every fold sees every corpus
    sizes = [sum(r["fold"] == k for r in rows if r["source"] == "vivae") for k in range(1, 6)]
    assert max(sizes) < 2.2 * min(sizes)  # speakers are spread, not piled into one fold
    again = audio_v2.assign_folds(synthetic_rows(), seed=7)
    assert [r["fold"] for r in again] == [r["fold"] for r in rows]  # deterministic in the seed


def test_a_group_in_two_folds_is_a_leakage_error() -> None:
    rows = audio_v2.assign_folds(synthetic_rows(), seed=7)
    victim = next(r for r in rows if r["source"] == "vivae")
    other = next(r for r in rows if r["group"] == victim["group"] and r is not victim)
    other["fold"] = victim["fold"] % 5 + 1
    with pytest.raises(LeakageError, match=victim["group"]):
        audio_v2.check_group_folds(rows)
    rows2 = audio_v2.assign_folds(synthetic_rows(), seed=7)
    rows2[0]["fold"] = None
    with pytest.raises(LeakageError):
        audio_v2.check_group_folds(rows2)


def test_combined_rows_build_from_generated_corpora(tmp_path: Path) -> None:
    make_esc50(tmp_path / "raw" / "esc50", [("door_wood_knock", f, str(100 + f), "A") for f in range(1, 6)] + [("footsteps", f, str(200 + f), "A") for f in range(1, 6)])
    make_vivae_files(tmp_path / "raw" / "vivae" / "audio", speakers=6)
    p = tmp_path / "raw" / "librispeech_mini" / "LibriSpeech" / "dev-clean-2" / "1" / "2" / "1-2-0000.flac"
    p.parent.mkdir(parents=True)
    sf.write(str(p), np.zeros(1600, dtype=np.float32), SR)
    rows = audio_v2.combined_rows(tmp_path, seed=1)
    assert {r["source"] for r in rows} == {"esc50", "vivae", "librispeech"}  # fsd50k absent -> simply missing, no error
    assert all(r["homeease_class"] in audio_v2.CLASSES for r in rows) and not any(r["homeease_class"] is None for r in rows)
    assert sum(r["homeease_class"] == "HELP" for r in rows) == 6 * 2 * 1  # fear + pain at peak, per speaker (low is excluded)
    audio_v2.check_group_folds(rows)
    only_esc = audio_v2.combined_rows(tmp_path, seed=1, sources=("esc50",))
    assert {r["source"] for r in only_esc} == {"esc50"}


def test_cli_knows_the_new_datasets_and_command() -> None:
    from homeease import cli

    assert {"vivae", "fsd50k_help"} <= set(cli.DATASETS)
    ap = cli.build_parser()
    a = ap.parse_args(["train-sound", "--model-id", "m", "--test-fold", "5"])
    assert a.fn is cli.cmd_train_sound and a.model_id == "m"
    assert ap.parse_args(["data", "download", "--dataset", "vivae"]).dataset == "vivae"
    assert ap.parse_args(["compare", "--suite", "audio", "--only", "A2"]).only == "A2"


@pytest.mark.realdata
def test_real_corpora_give_a_leak_free_split_with_the_documented_sizes() -> None:
    from homeease.core.paths import data_dir

    root = data_dir() / "raw"
    if not (vivae.is_present(root / "vivae", vivae.EXPECTED_CLIPS) and audio_sets.esc50_present(root / "esc50")):
        pytest.skip("VIVAE / ESC-50 not downloaded (homeease data download)")
    rows = audio_v2.combined_rows(data_dir())
    audio_v2.check_group_folds(rows)
    n = lambda src, cls: sum(r["source"] == src and r["homeease_class"] == cls for r in rows)  # noqa: E731
    assert (n("esc50", "IMPACT"), n("esc50", "ABNORMAL"), n("esc50", "DAILY")) == (40, 160, 760)
    assert (n("vivae", "HELP"), n("vivae", "ABNORMAL"), n("vivae", "DAILY")) == (277, 174, 550)
    assert {r["fold"] for r in rows if r["source"] == "vivae"} == {1, 2, 3, 4, 5}
    spk_folds = {r["group"]: r["fold"] for r in rows if r["source"] == "vivae"}
    assert len(spk_folds) == 11  # eleven speakers, each in exactly one fold
    if fsd50k_help.is_present(root / "fsd50k_help"):
        assert n("fsd50k", "HELP") == 130 and len({r["group"] for r in rows if r["source"] == "fsd50k"}) >= 100
