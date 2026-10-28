"""Dataset path registry for URFD, MCF and Le2i."""

import os
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
URFD = os.path.join(ROOT, "data", "urfd")
KAGGLE_CACHE = os.path.join(HERE, "cache", "kaggle")


def _find_kaggle_dir(slug_dirname):
    candidates = []
    home = os.path.expanduser("~")
    for base in (os.path.join(home, ".cache", "kagglehub", "datasets"), KAGGLE_CACHE):
        candidates += glob.glob(os.path.join(base, "**", slug_dirname), recursive=True)
    for c in candidates:
        if os.path.isdir(c) and len(os.listdir(c)):
            v = os.path.join(c, "versions")
            if os.path.isdir(v):
                vers = sorted(os.listdir(v))
                if vers:
                    return os.path.join(v, vers[-1])
            return c
    return None


def urfd_root():
    assert os.path.isdir(URFD), f"URFD not found: {URFD}"
    return URFD


def mc_root():
    p = _find_kaggle_dir("multiple-cameras-fall-dataset")
    if p is None:
        raise FileNotFoundError("MCF not downloaded: run download_kaggle.py")
    return p


def le2i_root():
    p = _find_kaggle_dir("falldataset-imvia")
    if p is None:
        raise FileNotFoundError("Le2i not downloaded: run download_kaggle.py")
    return p
