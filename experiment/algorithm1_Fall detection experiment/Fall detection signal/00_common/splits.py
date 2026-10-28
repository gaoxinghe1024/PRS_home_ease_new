# -*- coding: utf-8 -*-
"""Evaluation pool and data splits (sensor path).

Pool: 70 URFD accelerometer sequences (30 fall / 40 ADL).

Split schemes:
  1) stratified K-fold cross-validation (K=5, stratified by class; ~6 fall + 8 ADL
     per fold) -- the primary protocol
  2) repeated K-fold over multiple seeds (stability check, reported as mean +- std)
  3) robustness protocol (see experiments/exp04_robustness.py): inject Gaussian
     noise / random packet loss into the test set and measure performance decay

Note: URFD does not publish subject IDs, so a subject-independent (LOSO) split
cannot be built. The sensor path also has a single dataset, so cross-dataset LODO
is impossible. This path therefore substitutes "multi-seed repeated K-fold +
noise/packet-loss robustness" for the generalisation test, and states so explicitly
in the report.
"""
import numpy as np

import acc_loader

K_FOLDS = 5
SEEDS = (2026, 2027, 2028)


def unified_pool():
    """Return [(seq, has_fall, n_fall_intervals)] for all 70 sequences."""
    falls, adls = acc_loader.list_sequences()
    pool = []
    for s in falls + adls:
        iv = acc_loader.fall_intervals_s(s)
        pool.append((s, len(iv) > 0, len(iv)))
    return pool


def pool_summary():
    pool = unified_pool()
    n_fall = sum(1 for *_, hf, _ in pool if hf)
    return {"n_seq": len(pool), "n_fall": n_fall, "n_adl": len(pool) - n_fall}


def stratified_folds(k=K_FOLDS, seed=2026):
    """Class-stratified K folds. Returns [[seq,...] x k]."""
    from sklearn.model_selection import StratifiedKFold
    pool = unified_pool()
    seqs = np.array([s for s, *_ in pool])
    y = np.array([1 if hf else 0 for _, hf, _ in pool])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [[] for _ in range(k)]
    for i, (_, te) in enumerate(skf.split(seqs, y)):
        folds[i] = list(seqs[te])
    return folds


def repeated_folds(k=K_FOLDS, seeds=SEEDS):
    """Multi-seed repeated K-fold. Returns [(seed, folds)]."""
    return [(s, stratified_folds(k, s)) for s in seeds]


def split_train_test(folds, fold_i):
    """Fold fold_i -> (train_seqs, test_seqs)."""
    test = set(folds[fold_i])
    pool = unified_pool()
    train = [s for s, *_ in pool if s not in test]
    return train, list(folds[fold_i])


if __name__ == "__main__":
    print("pool:", pool_summary())
    f = stratified_folds()
    for i, fl in enumerate(f):
        n_f = sum(1 for s in fl if s.startswith("fall"))
        print(f"  fold{i}: {len(fl)} sequences (fall {n_f} / adl {len(fl)-n_f})")
