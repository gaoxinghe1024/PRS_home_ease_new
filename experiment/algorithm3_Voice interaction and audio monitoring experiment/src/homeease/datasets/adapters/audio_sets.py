"""Audio dataset adapters: ESC-50 (environmental sounds) and LibriSpeech dev-clean-2 (speech).

ESC-50 (Piczak 2015, https://github.com/karolpiczak/ESC-50): 2000 x 5 s clips, 50 classes,
5 official folds (clips from one source recording stay in one fold); CC BY-NC 3.0.
It contains no "call for help" class: HELP is therefore not evaluated on real data here.

LibriSpeech dev-clean-2 ("mini LibriSpeech", OpenSLR SLR31): read English speech, CC BY 4.0.
Used for VAD mixtures, as "TV/background speech" and for speaker verification trials.
It is not elderly or household speech; results do not transfer directly to the product.

Class mapping (version esc50-homeease-v1, fixed for all comparisons):
"""

from __future__ import annotations

import csv
import io
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from homeease.datasets.download import fetch, write_download_log

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
LIBRI_URL = "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"
MAPPING_VERSION = "esc50-homeease-v1"
ESC50_MAP = {
    "IMPACT": ["door_wood_knock"],  # proxy for knocks / thuds (no fall-thud class in ESC-50)
    "ABNORMAL": ["glass_breaking", "crying_baby", "siren", "clock_alarm"],
    "DAILY": [
        "vacuum_cleaner", "washing_machine", "toilet_flush", "water_drops", "pouring_water", "keyboard_typing", "mouse_click",
        "door_wood_creaks", "can_opening", "clock_tick", "brushing_teeth", "drinking_sipping", "footsteps", "laughing", "snoring",
        "breathing", "coughing", "sneezing", "clapping",
    ],
}
THREE_CLASS = {"HELP": "HELP", "IMPACT": "ABNORMAL", "ABNORMAL": "ABNORMAL", "DAILY": "DAILY"}  # report aggregation, fixed


def esc50_present(raw_dir: Path) -> bool:
    """Extracted ESC-50 is complete when every clip listed in meta/esc50.csv exists."""
    meta = Path(raw_dir) / "meta" / "esc50.csv"
    if not meta.is_file():
        return False
    with open(meta, encoding="utf-8") as fh:
        names = [r["filename"] for r in csv.DictReader(fh)]
    return bool(names) and all((Path(raw_dir) / "audio" / n).is_file() for n in names)


def librispeech_present(raw_dir: Path) -> bool:
    return (Path(raw_dir) / "LibriSpeech").is_dir() and next(Path(raw_dir).rglob("*.flac"), None) is not None


def download(name: str, raw_dir: Path) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if name == "esc50":
        if esc50_present(raw_dir):  # the archive was deleted on purpose after extraction: never fetch it again
            return {"status": "present", "clips": len(list((raw_dir / "audio").glob("*.wav"))), "downloaded_bytes": 0}
        e = fetch(ESC50_URL, raw_dir / "ESC-50-master.zip", timeout=900)
        if e["status"] != "failed" and not (raw_dir / "audio").is_dir():
            with zipfile.ZipFile(raw_dir / "ESC-50-master.zip") as z:
                for n in z.namelist():
                    if n.endswith(".wav") or n.endswith("esc50.csv") or n.endswith("LICENSE"):
                        rel = n.split("ESC-50-master/", 1)[1]
                        out = raw_dir / rel
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(z.read(n))
        write_download_log(raw_dir, [e], {"dataset": "esc50", "license": "CC BY-NC 3.0", "source": ESC50_URL})
        if e["status"] != "failed" and esc50_present(raw_dir):
            (raw_dir / "ESC-50-master.zip").unlink(missing_ok=True)  # keep the extracted clips only (disk budget)
        return {"status": e["status"], "clips": len(list((raw_dir / "audio").glob("*.wav"))) if (raw_dir / "audio").is_dir() else 0}
    if name == "librispeech_mini":
        if librispeech_present(raw_dir):
            return {"status": "present", "flac": len(list(raw_dir.rglob("*.flac"))), "downloaded_bytes": 0}
        e = fetch(LIBRI_URL, raw_dir / "dev-clean-2.tar.gz", timeout=900)
        if e["status"] != "failed" and not (raw_dir / "LibriSpeech").is_dir():
            with tarfile.open(raw_dir / "dev-clean-2.tar.gz") as t:
                t.extractall(raw_dir, filter="data")
        write_download_log(raw_dir, [e], {"dataset": "librispeech_mini", "license": "CC BY 4.0", "source": LIBRI_URL})
        if e["status"] != "failed" and librispeech_present(raw_dir):
            (raw_dir / "dev-clean-2.tar.gz").unlink(missing_ok=True)  # keep the extracted files only (disk budget)
        return {"status": e["status"], "flac": len(list(raw_dir.rglob("*.flac")))}
    raise ValueError(name)


def esc50_rows(raw_dir: Path) -> list[dict[str, Any]]:
    rows = []
    meta = Path(raw_dir) / "meta" / "esc50.csv"
    inv = {c: k for k, cs in ESC50_MAP.items() for c in cs}
    with open(meta, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cat = r["category"]
            rows.append({
                "path": str(Path(raw_dir) / "audio" / r["filename"]), "fold": int(r["fold"]), "category": cat, "esc10": r["esc10"] == "True",
                "src_file": r["src_file"], "homeease_class": inv.get(cat), "mapping_version": MAPPING_VERSION,
            })
    return rows


def librispeech_rows(raw_dir: Path) -> list[dict[str, Any]]:
    import soundfile as sf

    rows = []
    for p in sorted(Path(raw_dir).rglob("*.flac")):
        spk, chap = p.parts[-3], p.parts[-2]
        info = sf.info(str(p))
        rows.append({"path": str(p), "speaker": spk, "chapter": chap, "utt": p.stem, "duration_s": info.frames / info.samplerate, "sr": info.samplerate})
    return rows


def prepare(name: str, raw_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    if name == "esc50":
        return esc50_rows(raw_dir)
    if name == "librispeech_mini":
        return librispeech_rows(raw_dir)
    raise ValueError(name)


__all__ = ["download", "prepare", "esc50_present", "librispeech_present", "esc50_rows", "librispeech_rows", "ESC50_MAP", "THREE_CLASS", "MAPPING_VERSION", "io"]
