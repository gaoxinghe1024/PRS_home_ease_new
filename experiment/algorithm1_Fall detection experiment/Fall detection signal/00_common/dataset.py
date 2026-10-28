# -*- coding: utf-8 -*-
"""Shared upstream cache (keeps the upstream fixed so downstream can be compared).

Materialises the "preprocess + windowize" result on disk for every downstream
algorithm and experiment to read:

      raw CSV --[preprocess]--> canonical signal --[windowize]--> window tensor
                                                                     |
                                          00_common/cache/windows_<cfg>.npz
                                          (windows + time base + labels + meta)

Contract:
  * all downstream algorithms of the same cfg read the SAME cache -> upstream is
    identical for everyone (single-variable principle);
  * the cache key includes a preprocessing-config fingerprint, so switching config
    never overwrites another config's cache;
  * this module only reads raw CSVs and never modifies anything under data/.
"""
import os
import json
import hashlib
import pickle
import numpy as np

import paths
import acc_loader
import preprocess
import windowize

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

DEFAULT_CFG = {"resample": 60.0, "filt": "butter10", "win_s": windowize.WIN_S,
               "step_s": windowize.STEP_S}


def cfg_id(cfg):
    s = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def cache_path(cfg):
    return os.path.join(CACHE_DIR, f"windows_{cfg_id(cfg)}.npz")


def build_sequence(seq, cfg):
    """Single sequence -> windowed result (not persisted)."""
    sig = preprocess.build_signal(seq, resample=cfg["resample"], filt=cfg["filt"])
    fs = sig["fs"]
    w4 = np.stack([sig["ax"], sig["ay"], sig["az"], sig["mag"]], axis=1)
    wins, t0_idx = windowize.make_windows(w4, fs, cfg["win_s"], cfg["step_s"])
    t0_s = windowize.window_start_times(sig["t_s"], t0_idx)
    dur = float(sig["t_s"][-1] - sig["t_s"][0])
    iv = acc_loader.fall_intervals_s(seq, clip_to=(0.0, dur))
    lab = windowize.window_labels(t0_s, iv, cfg["win_s"])
    return {"seq": seq, "wins": wins.astype(np.float32), "t0_s": t0_s,
            "labels": lab, "intervals": iv, "dur_s": dur, "has_fall": len(iv) > 0,
            "fs": float(fs), "win_s": float(cfg["win_s"])}


_MEM = {}


def load_all(cfg=None, use_cache=True):
    """Return {seq: dict}, backed by an in-memory and an on-disk cache."""
    cfg = dict(DEFAULT_CFG if cfg is None else cfg)
    key = cfg_id(cfg)
    if key in _MEM:
        return _MEM[key]

    path = cache_path(cfg)
    if use_cache and os.path.isfile(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("cfg") == cfg:
            _MEM[key] = data["seqs"]
            return _MEM[key]

    falls, adls = acc_loader.list_sequences()
    seqs = {}
    for s in falls + adls:
        seqs[s] = build_sequence(s, cfg)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"cfg": cfg, "cfg_id": key, "seqs": seqs}, f, protocol=4)
    _MEM[key] = seqs
    return seqs


def pack(seqs_map, seq_list):
    """Flatten several sequences into the arrays needed for train/test.

    Returns a dict:
      wins [N,win_n,4] / t0_s [N] / labels [N] / seq_id [N]
      meta {seq: (intervals, dur, has_fall)}
    """
    W, T0, L, SID = [], [], [], []
    meta = {}
    for s in seq_list:
        d = seqs_map[s]
        n = len(d["wins"])
        if n == 0:
            meta[s] = (d["intervals"], d["dur_s"], d["has_fall"])
            continue
        W.append(d["wins"]); T0.append(d["t0_s"]); L.append(d["labels"])
        SID += [s] * n
        meta[s] = (d["intervals"], d["dur_s"], d["has_fall"])
    if W:
        return {"wins": np.concatenate(W), "t0_s": np.concatenate(T0),
                "labels": np.concatenate(L), "seq_id": np.array(SID), "meta": meta}
    return {"wins": np.zeros((0, 1, 4), np.float32), "t0_s": np.array([]),
            "labels": np.array([]), "seq_id": np.array([]), "meta": meta}


if __name__ == "__main__":
    m = load_all()
    tot = sum(len(d["wins"]) for d in m.values())
    print(f"cfg={DEFAULT_CFG}")
    print(f"sequences {len(m)} / windows {tot}")
    for s in ("fall-01", "fall-10", "adl-01"):
        d = m[s]
        print(f"  {s}: wins={d['wins'].shape} pos_wins={int(d['labels'].sum())} "
              f"dur={d['dur_s']:.2f}s GT={[(round(a,2),round(b,2)) for a,b in d['intervals']]}")
