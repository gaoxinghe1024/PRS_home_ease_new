"""A small, selective FSD50K subset for the HELP class (screams) and its same-corpus negatives.

FSD50K (Fonseca et al. 2022, Zenodo record 4060432, 51 k Freesound clips, 200 AudioSet classes) is only
distributed as 24 GB of split archives. This adapter never touches them:

* labels and per-clip metadata (uploader, licence, title) come from the two small Zenodo archives
  ``FSD50K.ground_truth.zip`` (0.33 MB) and ``FSD50K.metadata.zip`` (6.7 MB), both pinned by md5;
* audio comes PER FILE from the Hugging Face mirror ``Fhrozen/FSD50k`` at a pinned revision
  (``resolve/<revision>/clips/<split>/<id>.wav``); each file is verified against the sha256 that the
  mirror's LFS pointer reports (``paths-info`` API). The mirror is third-party: the check proves the file
  is what the mirror advertises, not that it is byte-identical to the Zenodo original.

Selection is deterministic (seed), licence-filtered (CC0, CC BY, CC BY-NC; Sampling+ is skipped), limited to short
clips (the sound window is 2 s) and capped per uploader and in total bytes (``BYTE_BUDGET``). The class mapping
is in audio_v2.py (HELP = clips labelled Screaming; Shout/Yell-only clips are not used). Archives are deleted after the
selection is written; a present subset (``selection.json`` + all wav files) is never downloaded again.

Per-clip attribution (title, uploader, licence URL, Freesound id) is stored in ``selection.json``.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from homeease.core.util import get_logger
from homeease.datasets.adapters.audio_v2 import FSD50K_MAP, MAPPING_VERSION_V2, map_fsd50k
from homeease.datasets.download import write_download_log
from homeease.datasets.stream import download_checked, get_json, total_bytes

log = get_logger(__name__)

ZENODO_RECORD = "4060432"
ZENODO_FILES = {
    "FSD50K.ground_truth.zip": {"bytes": 334_701, "md5": "ca27382c195e37d2269c4c866dd73485"},
    "FSD50K.metadata.zip": {"bytes": 6_700_838, "md5": "b9ea0c829a411c1d42adb9da539ed237"},
}
HF_REPO = "Fhrozen/FSD50k"
HF_REVISION = "ccf1acaff12f3f4a4c10052dddafbb6d22152c9b"
HF_API = f"https://huggingface.co/api/datasets/{HF_REPO}/paths-info/{HF_REVISION}"
HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}"

# subtype -> (label that selects it, quota); order = priority when a clip has several labels
QUOTAS: list[tuple[str, int]] = [
    ("Screaming", 130), ("Thump_and_thud", 25), ("Knock", 20), ("Shatter", 25), ("Laughter", 30), ("Giggle", 10), ("Cough", 20), ("Sneeze", 15),
]
MIN_BYTES, MAX_BYTES = 30_000, 300_000  # about 0.35 s .. 3.4 s of 44.1 kHz / 16 bit mono
MAX_PER_UPLOADER = 8
BYTE_BUDGET = 42 * 1024 * 1024  # clip bytes (metadata archives are 7 MB on top)
ALLOWED_LICENSES = ("publicdomain/zero", "licenses/by/", "licenses/by-nc/")  # Sampling+ is excluded
SEED = 0


def licence_ok(url: str) -> bool:
    return any(s in url for s in ALLOWED_LICENSES)


def subtype_of(labels: list[str]) -> str | None:
    cls = map_fsd50k(labels)
    if cls is None:
        return None
    for lab, _ in QUOTAS:
        if lab in labels and lab in FSD50K_MAP[cls]:
            return lab
    return None


def candidates(ground_truth: dict[str, list[tuple[str, list[str]]]], info: dict[str, dict[str, dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Licence-filtered candidate clips per subtype. ``ground_truth[split] = [(fname, labels)]``, ``info[split][fname] = clip info``."""
    out: dict[str, list[dict[str, Any]]] = {lab: [] for lab, _ in QUOTAS}
    for split, items in ground_truth.items():
        for fname, labels in items:
            sub = subtype_of(labels)
            meta = info.get(split, {}).get(fname)
            if sub is None or meta is None or not licence_ok(str(meta.get("license", ""))):
                continue
            out[sub].append({
                "id": fname, "split": split, "labels": labels, "subtype": sub, "homeease_class": map_fsd50k(labels),
                "uploader": str(meta.get("uploader", "")), "license": str(meta.get("license", "")), "title": str(meta.get("title", ""))[:120],
            })
    return out


def select(cands: dict[str, list[dict[str, Any]]], sizes: dict[str, int], seed: int = SEED, byte_budget: int = BYTE_BUDGET) -> list[dict[str, Any]]:
    """Deterministic selection: shuffled candidates, size band, per-uploader cap, per-subtype quota, total byte budget.

    ``sizes[id]`` is the file size in bytes; candidates without a known size are skipped."""
    rng = np.random.default_rng(seed)
    used: dict[str, int] = {}
    total = 0
    chosen: list[dict[str, Any]] = []
    for lab, quota in QUOTAS:
        pool = [c for c in sorted(cands.get(lab, []), key=lambda c: c["id"])]
        order = rng.permutation(len(pool))
        n = 0
        for i in order:
            c = pool[int(i)]
            b = sizes.get(c["id"])
            if b is None or not MIN_BYTES <= b <= MAX_BYTES or used.get(c["uploader"], 0) >= MAX_PER_UPLOADER:
                continue
            if total + b > byte_budget:
                continue
            chosen.append({**c, "bytes": int(b)})
            used[c["uploader"]] = used.get(c["uploader"], 0) + 1
            total += b
            n += 1
            if n >= quota:
                break
    return chosen


def is_present(raw_dir: Path, min_clips: int = 1) -> bool:
    sel = Path(raw_dir) / "selection.json"
    if not sel.is_file():
        return False
    try:
        items = json.loads(sel.read_text(encoding="utf-8"))["clips"]
    except (OSError, ValueError, KeyError):
        return False
    return len(items) >= min_clips and all((Path(raw_dir) / "audio" / f"{c['id']}.wav").is_file() for c in items)


def _read_ground_truth(z: zipfile.ZipFile) -> dict[str, list[tuple[str, list[str]]]]:
    out = {}
    for split in ("dev", "eval"):
        txt = z.read(f"FSD50K.ground_truth/{split}.csv").decode("utf-8")
        out[split] = [(r["fname"], r["labels"].split(",")) for r in csv.DictReader(io.StringIO(txt))]
    return out


def _read_info(z: zipfile.ZipFile) -> dict[str, dict[str, dict[str, Any]]]:
    return {s: json.loads(z.read(f"FSD50K.metadata/{s}_clips_info_FSD50K.json")) for s in ("dev", "eval")}


def download(raw_dir: Path, seed: int = SEED) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if is_present(raw_dir):
        n = len(json.loads((raw_dir / "selection.json").read_text(encoding="utf-8"))["clips"])
        return {"status": "present", "clips": n, "downloaded_bytes": 0}
    src = raw_dir / "_src"
    entries: list[dict[str, Any]] = []
    for name, pin in ZENODO_FILES.items():
        e = download_checked(f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{name}/content", src / name, size=pin["bytes"], md5=pin["md5"])
        entries.append(e)
        if e["status"] == "failed":
            return {"status": "failed", "error": e.get("error"), "downloaded_bytes": total_bytes(entries)}
    with zipfile.ZipFile(src / "FSD50K.ground_truth.zip") as z:
        gt = _read_ground_truth(z)
    with zipfile.ZipFile(src / "FSD50K.metadata.zip") as z:
        info = _read_info(z)
    cands = candidates(gt, info)
    # file sizes and sha256 for a shuffled pool per subtype (a few requests of 100 paths), not for all 51 k clips
    rng = np.random.default_rng(seed + 1)
    pool: list[dict[str, Any]] = []
    for lab, quota in QUOTAS:
        c = sorted(cands[lab], key=lambda x: x["id"])
        pool += [c[int(i)] for i in rng.permutation(len(c))[: quota * 6]]
    sizes: dict[str, int] = {}
    sha: dict[str, str] = {}
    for i in range(0, len(pool), 100):
        batch = pool[i : i + 100]
        res = get_json(HF_API, data={"paths": [f"clips/{c['split']}/{c['id']}.wav" for c in batch], "expand": False})
        for r in res:
            cid = Path(r["path"]).stem
            sizes[cid] = int(r["size"])
            if r.get("lfs", {}).get("oid"):
                sha[cid] = r["lfs"]["oid"]
    chosen = select({lab: [c for c in pool if c["subtype"] == lab] for lab, _ in QUOTAS}, sizes, seed)
    audio = raw_dir / "audio"
    failed = []
    for c in chosen:
        e = download_checked(f"{HF_RESOLVE}/clips/{c['split']}/{c['id']}.wav", audio / f"{c['id']}.wav", size=c["bytes"], sha256=sha.get(c["id"]))
        entries.append(e)
        if e["status"] == "failed":
            failed.append(c["id"])
        else:
            c["sha256"] = e["sha256"]
    chosen = [c for c in chosen if c["id"] not in failed]
    sel = {
        "mapping_version": MAPPING_VERSION_V2, "seed": seed, "hf_repo": HF_REPO, "hf_revision": HF_REVISION, "zenodo_record": ZENODO_RECORD,
        "quotas": dict(QUOTAS), "size_band_bytes": [MIN_BYTES, MAX_BYTES], "max_per_uploader": MAX_PER_UPLOADER, "clips": chosen,
    }
    (raw_dir / "selection.json").write_text(json.dumps(sel, indent=1, ensure_ascii=False), encoding="utf-8")
    for name in ZENODO_FILES:
        (src / name).unlink(missing_ok=True)
    try:
        src.rmdir()
    except OSError:
        pass
    meta = {
        "dataset": "fsd50k_help", "license": "per clip: CC0 / CC BY / CC BY-NC (see selection.json: uploader, title, licence URL, Freesound id)",
        "source": f"https://zenodo.org/records/{ZENODO_RECORD} (labels, metadata) + https://huggingface.co/datasets/{HF_REPO}/tree/{HF_REVISION} (audio)",
        "archives_deleted": list(ZENODO_FILES), "clips": len(chosen), "clips_failed": failed, "downloaded_bytes_total": total_bytes(entries),
        "citation": "Fonseca, Favory, Pons, Font, Serra (2022) FSD50K: an open dataset of human-labeled sound events, IEEE/ACM TASLP 30, doi:10.1109/TASLP.2021.3133208",
        "note": "third-party Hugging Face mirror; per-file sha256 = the mirror's LFS oid",
    }
    write_download_log(raw_dir, entries, meta)
    return {"status": "downloaded" if not failed else "incomplete", "clips": len(chosen), "failed": failed, "downloaded_bytes": total_bytes(entries)}


def rows(raw_dir: Path) -> list[dict[str, Any]]:
    import soundfile as sf

    sel = json.loads((Path(raw_dir) / "selection.json").read_text(encoding="utf-8"))
    out = []
    for c in sel["clips"]:
        p = Path(raw_dir) / "audio" / f"{c['id']}.wav"
        if not p.is_file():
            continue
        info = sf.info(str(p))
        out.append({
            "source": "fsd50k", "path": str(p), "freesound_id": c["id"], "group": f"fsd50k:{c['uploader']}", "fold": None, "uploader": c["uploader"],
            "labels": c["labels"], "license": c["license"], "duration_s": info.frames / info.samplerate, "sr": info.samplerate,
            "homeease_class": map_fsd50k(c["labels"]), "subtype": c["subtype"], "mapping_version": MAPPING_VERSION_V2,
        })
    return out


def prepare(raw_dir: Path, out_dir: Path | None = None) -> list[dict[str, Any]]:
    return rows(raw_dir)


__all__ = ["download", "prepare", "rows", "is_present", "candidates", "select", "subtype_of", "licence_ok", "QUOTAS"]
