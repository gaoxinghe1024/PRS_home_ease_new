"""Shared paths and random seeds."""
import hashlib
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def seeds(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def artifact_manifest(root=ROOT):
    """Hash JSON with LF endings and binary weights byte for byte."""
    entries = []
    for path in sorted((root / "artifacts").glob("*")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        entry = {"file": path.relative_to(root).as_posix()}
        if path.suffix.lower() == ".json":
            raw = raw.replace(b"\r\n", b"\n")
            entry["text_eol"] = "lf"
        entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        entries.append(entry)
    return entries
