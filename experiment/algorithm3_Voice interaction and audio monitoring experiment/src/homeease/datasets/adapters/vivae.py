"""VIVAE: Variably Intense Vocalizations of Affect and Emotion (Holz, Larrouy-Maestri, Poeppel 2022).

Zenodo record 4066235, one 93.5 MB zip (verified on 2026-10-07 through the Zenodo API: size 93460357 bytes,
md5 62383077641fa90f5545e38d7f268e02), licence CC BY-NC 4.0, DOI 10.5281/zenodo.4066235. 1085 files, 44.1 kHz /
16 bit, 11 speakers (S01-S11), 6 emotions x 4 intensities, file name ``S04_surprise_peak_10.wav`` =
speaker_emotion_intensity_item. Non-speech vocalisations, acted in a studio; they are a proxy for screams and
pain/fear cries, NOT recordings of older adults or of a home.

Only the full set is extracted (``audio/<name>.wav``); the archive is deleted right after extraction. A present,
extracted corpus is never downloaded again. The class mapping lives in audio_v2.py (HELP = fear/pain).

Citation: Holz, N., Larrouy-Maestri, P., Poeppel, D. (2022). Emotion, 22(1), 213-225, doi:10.1037/emo0001048.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

from homeease.core.util import get_logger
from homeease.datasets.adapters.audio_v2 import MAPPING_VERSION_V2, map_vivae
from homeease.datasets.download import write_download_log
from homeease.datasets.stream import download_checked

log = get_logger(__name__)

URL = "https://zenodo.org/api/records/4066235/files/VIVAE.zip/content"
BYTES = 93_460_357
MD5 = "62383077641fa90f5545e38d7f268e02"
EXPECTED_CLIPS = 1085
NAME_RE = re.compile(r"^(S\d{2})_(achievement|anger|fear|pain|pleasure|surprise)_(low|moderate|strong|peak)_(\d+)\.wav$")
LICENSE = "CC BY-NC 4.0"


def parse_name(name: str) -> dict[str, Any] | None:
    m = NAME_RE.match(Path(name).name)
    if not m:
        return None
    return {"speaker": m.group(1), "emotion": m.group(2), "intensity": m.group(3), "item": int(m.group(4))}


def is_present(raw_dir: Path, min_clips: int = 1) -> bool:
    a = Path(raw_dir) / "audio"
    return a.is_dir() and sum(1 for _ in a.glob("*.wav")) >= min_clips


def download(raw_dir: Path, expected_clips: int = EXPECTED_CLIPS) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if is_present(raw_dir, expected_clips):
        return {"status": "present", "clips": sum(1 for _ in (raw_dir / "audio").glob("*.wav")), "downloaded_bytes": 0}
    arc = raw_dir / "VIVAE.zip"
    e = download_checked(URL, arc, size=BYTES, md5=MD5)
    if e["status"] == "failed":
        write_download_log(raw_dir, [e], {"dataset": "vivae", "license": LICENSE, "source": URL})
        return {"status": "failed", "error": e.get("error"), "downloaded_bytes": 0}
    out = raw_dir / "audio"
    out.mkdir(exist_ok=True)
    names: set[str] = set()
    with zipfile.ZipFile(arc) as z:
        # the full set first, so a duplicate in the core-set folder never overrides it
        members = sorted((i for i in z.infolist() if parse_name(i.filename)), key=lambda i: ("core" in i.filename.lower(), i.filename))
        for info in members:
            base = Path(info.filename).name
            if base in names:
                continue
            (out / base).write_bytes(z.read(info))
            names.add(base)
    if e["status"] == "downloaded" or arc.is_file():
        arc.unlink(missing_ok=True)  # the archive is not kept: the extracted wav files are the corpus
    meta = {
        "dataset": "vivae", "license": LICENSE, "source": URL, "doi": "10.5281/zenodo.4066235", "archive_deleted": True,
        "clips_extracted": len(names), "clips_expected": expected_clips,
        "citation": "Holz, Larrouy-Maestri, Poeppel (2022) Emotion 22(1):213-225, doi:10.1037/emo0001048",
        "note": "acted non-speech vocalisations (studio); proxy for screams / pain-fear cries, not older-adult or household audio",
    }
    write_download_log(raw_dir, [e], meta)
    status = "downloaded" if len(names) >= expected_clips else "incomplete"
    return {"status": status if e["status"] == "downloaded" else "extracted", "clips": len(names), "downloaded_bytes": e["bytes"] if e["status"] == "downloaded" else 0, "sha256": e.get("sha256")}


def rows(raw_dir: Path) -> list[dict[str, Any]]:
    import soundfile as sf

    out = []
    for p in sorted((Path(raw_dir) / "audio").glob("*.wav")):
        meta = parse_name(p.name)
        if meta is None:
            continue
        info = sf.info(str(p))
        out.append({
            "source": "vivae", "path": str(p), **meta, "group": f"vivae:{meta['speaker']}", "fold": None,
            "duration_s": info.frames / info.samplerate, "sr": info.samplerate,
            "homeease_class": map_vivae(meta["emotion"], meta["intensity"]), "subtype": f"{meta['emotion']}/{meta['intensity']}",
            "mapping_version": MAPPING_VERSION_V2,
        })
    return out


def prepare(raw_dir: Path, out_dir: Path | None = None) -> list[dict[str, Any]]:
    """Manifest rows (all clips, ambiguous ones with ``homeease_class`` None). ``fold`` is assigned in audio_v2.combined_rows."""
    return rows(raw_dir)


__all__ = ["download", "prepare", "rows", "parse_name", "is_present", "URL", "BYTES", "MD5", "EXPECTED_CLIPS"]
