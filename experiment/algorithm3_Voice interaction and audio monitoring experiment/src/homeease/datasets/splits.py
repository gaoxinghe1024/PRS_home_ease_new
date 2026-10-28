"""Grouped splits made BEFORE windowing (by participant or original recording).

Multiple views, resampled versions, windows and augmentations of one group always
stay in one partition. Splits are saved with a content hash and checked for leakage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from homeease.core.paths import data_dir


class LeakageError(RuntimeError):
    pass


def split_hash(split: dict[str, Any]) -> str:
    parts: dict[str, Any] = {}
    for k, v in split.items():
        if k == "folds" and isinstance(v, list):
            parts[k] = [{kk: (sorted(vv) if isinstance(vv, list) else vv) for kk, vv in f.items()} for f in v]
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            parts[k] = sorted(v)
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def check_no_leakage(split: dict[str, list[str]]) -> None:
    keys = [k for k in ("train", "val", "calibration", "test") if k in split]
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            inter = set(split[a]) & set(split[b])
            if inter:
                raise LeakageError(f"groups in both {a} and {b}: {sorted(inter)[:5]}")


def stratified_group_split(groups_by_stratum: dict[str, list[str]], fractions: tuple[float, float, float], seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for stratum in sorted(groups_by_stratum):
        g = sorted(set(groups_by_stratum[stratum]))
        rng.shuffle(g)
        n = len(g)
        n_test = max(1, int(round(n * fractions[2]))) if n >= 3 else 0
        n_val = max(1, int(round(n * fractions[1]))) if n - n_test >= 2 else 0
        out["test"] += g[:n_test]
        out["val"] += g[n_test : n_test + n_val]
        out["train"] += g[n_test + n_val :]
    check_no_leakage(out)
    return out


def grouped_kfold(groups: list[str], strata: dict[str, str], k: int, seed: int) -> list[dict[str, list[str]]]:
    """k folds; fold i uses part i as test, part (i+1)%k as validation, the rest as train."""
    rng = np.random.default_rng(seed)
    by: dict[str, list[str]] = {}
    for g in sorted(set(groups)):
        by.setdefault(strata.get(g, "_"), []).append(g)
    parts: list[list[str]] = [[] for _ in range(k)]
    for s in sorted(by):
        g = by[s]
        rng.shuffle(g)
        for i, x in enumerate(g):
            parts[i % k].append(x)
    folds = []
    for i in range(k):
        test, val = parts[i], parts[(i + 1) % k]
        train = [x for j, p in enumerate(parts) if j not in (i, (i + 1) % k) for x in p]
        f = {"train": sorted(train), "val": sorted(val), "test": sorted(test), "fold": i}
        check_no_leakage(f)  # type: ignore[arg-type]
        folds.append(f)
    return folds


def save_split(name: str, split: dict[str, Any], meta: dict[str, Any]) -> Path:
    d = data_dir() / "splits"
    d.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "hash": split_hash(split), "meta": meta, **split}
    p = d / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load_split(name: str) -> dict[str, Any]:
    p = data_dir() / "splits" / f"{name}.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    check_no_leakage(s)
    if split_hash(s) != s["hash"]:
        raise LeakageError(f"split file {p} was modified (hash mismatch)")
    return s


def weda_split(manifest_rows: list[dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    strata: dict[str, list[str]] = {}
    for r in manifest_rows:
        strata.setdefault(r["age_group"], []).append(r["group_id"])
    return stratified_group_split(strata, (0.6, 0.2, 0.2), seed)
