"""Unified evaluation pool, grouped 5-fold CV and leave-one-dataset-out."""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import urfd_loader, mc_loader, le2i_loader


def group_key(dataset, seq):
    if dataset == "urfd":
        return f"urfd/{seq}"
    if dataset == "mc":
        return f"mc/{seq}"
    if dataset == "le2i":
        return f"le2i/{seq}"
    raise ValueError(dataset)


_WARNED = set()


def _optional_sequences(dataset, loader):
    """Enumerate an optional dataset, skipping it with a warning if not downloaded."""
    try:
        if dataset == "mc":
            return loader.list_sequences()
        return [s for s, _ in loader.list_videos()]
    except FileNotFoundError as exc:
        if dataset not in _WARNED:
            _WARNED.add(dataset)
            print(f"[skip] {dataset} unavailable: {exc}", file=sys.stderr)
        return []


def unified_pool():
    """URFD is mandatory; MCF and Le2i are used only when present on disk."""
    pool = []
    falls, adls = urfd_loader.list_sequences()
    for s in falls + adls:
        iv = urfd_loader.fall_intervals_s(s)
        pool.append(("urfd", s, len(iv) > 0, len(iv)))
    for ds, mod in (("mc", mc_loader), ("le2i", le2i_loader)):
        for s in _optional_sequences(ds, mod):
            iv = mod.fall_intervals_s(s)
            pool.append((ds, s, len(iv) > 0, len(iv)))
    return pool


def group_5fold(seed=2026):
    pool = unified_pool()
    groups = sorted({group_key(d, s) for d, s, *_ in pool})
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    return [groups[i::5] for i in range(5)]


def leave_one_dataset_out():
    pool = unified_pool()
    out = []
    for ds in ("urfd", "mc", "le2i"):
        gs = sorted({group_key(d, s) for d, s, *_ in pool if d == ds})
        if gs:
            out.append((ds, gs))
    return out


if __name__ == "__main__":
    pool = unified_pool()
    n_fall = sum((1 for *_, hf, _ in pool if hf))
    print(
        f"unified pool: {len(pool)} sequences, with fall {n_fall}, without fall {len(pool) - n_fall}"
    )
    from collections import Counter

    print("by dataset:", dict(Counter((d for d, *_ in pool))))
    folds = group_5fold()
    print(
        f"grouped 5-fold: {[len(f) for f in folds]} (total groups {sum((len(f) for f in folds))})"
    )
    loo = leave_one_dataset_out()
    print("LODO fold sizes:", [(ds, len(gs)) for ds, gs in loo])
