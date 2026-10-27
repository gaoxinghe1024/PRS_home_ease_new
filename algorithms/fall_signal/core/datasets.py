# -*- coding: utf-8 -*-
"""Data source: read-only reference to the URFD sequences of the parent project `Fall_Acc_Single/data/urfd/csv`.

**No data is copied** into this folder. Once the system runs:
  - sample data mode: replay / detect any sequence listed by this module;
  - users can also upload their own CSV (see preprocess.read_acc_csv).

When deployed without the parent project, set the environment variable `FALL_SYSTEM_DATA_DIR` to a folder
of `{seq}-acc.csv` files.
"""
import os
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEMS_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PARENT_ROOT = os.path.abspath(os.path.join(SYSTEMS_ROOT, ".."))

ENV_DIR = "FALL_SYSTEM_DATA_DIR"


def acc_csv_dir():
    d = os.environ.get(ENV_DIR)
    if d and os.path.isdir(d):
        return d
    return os.path.join(PARENT_ROOT, "data", "urfd", "csv")


def is_available():
    return os.path.isdir(acc_csv_dir())


def list_sequences():
    """Return {'fall': [...], 'adl': [...]} sorted by number. The sequence names carry the class."""
    d = acc_csv_dir()
    if not os.path.isdir(d):
        return {"fall": [], "adl": []}
    falls = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(d, "fall-*-acc.csv")))
    adls = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(d, "adl-*-acc.csv")))
    return {"fall": falls, "adl": adls}


def acc_csv(seq):
    p = os.path.join(acc_csv_dir(), f"{seq}-acc.csv")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Sequence not found: {seq}")
    return p


def sequence_meta(seq):
    """Sequence metadata (no annotation file; the class is inferred from the name only, as a demo hint)."""
    kind = "fall" if str(seq).startswith("fall") else ("adl" if str(seq).startswith("adl") else "unknown")
    return {"seq": seq, "kind": kind,
            "hint": "URFD fall sequence" if kind == "fall" else ("URFD daily-activity sequence" if kind == "adl" else "Custom sequence")}
