"""Training and evaluation core of the 4-class sound-event model (HELP / IMPACT / ABNORMAL / DAILY).

Used by experiment A2 (``experiments/audio_suite.py``) and by ``homeease train-sound``. Everything here follows the
same recipe the live monitor uses, so what is evaluated is what is deployed:

* one example = one ``WINDOW_S`` window (the monitor scores exactly such windows). A clip contributes its loudest window
  (evaluation) and one extra jittered window (training only); clips shorter than the window are placed in it and the rest
  is noise at -50 dB re the clip (no digital silence: ESC-50 and VIVAE would otherwise differ by exact zeros);
* the window is loudness-normalised with ``sound_package.normalize_window`` (RMS), the same function the loaded package
  applies to live windows, so recording level cannot identify a corpus;
* splits are by GROUP (audio_v2.py: ESC-50 source recording, VIVAE speaker, FSD50K uploader, LibriSpeech speaker) and are
  fixed before any window is cut; the extra training windows never leave their fold;
* cut-offs below which a window is UNKNOWN and the CNN temperature are chosen on DEV predictions only (out-of-fold
  predictions of the non-test folds), never on the test fold.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from homeease.core.util import get_logger
from homeease.datasets.adapters.audio_v2 import CLASSES, N_FOLDS
from homeease.ml.audio.features import SR, load_audio
from homeease.ml.audio.sound import clip_features, logmel_patch, train_cnn
from homeease.ml.audio.sound_package import normalize_window
from homeease.ml.common import metrics as M

log = get_logger(__name__)

WINDOW_S = 2.0
PATCH_FRAMES = 200  # 198 log-mel frames in 2 s, padded by two zero frames
NORMALIZATION = "rms"
DITHER_DB = -50.0
CANDIDATES = ("mfcc_logmel_stats+rf", "mfcc_logmel_stats+logreg", "logmel_small_cnn", "yamnet_embedding+logreg")
UNKNOWN = -1


# --------------------------------------------------------------------------- windows
def cut_window(x: np.ndarray, n: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    """``mode='eval'``: loudest window / centred clip; ``'aug'``: random active window / random placement."""
    if len(x) >= n:
        hop = max(n // 8, 1)
        starts = np.unique(np.append(np.arange(0, len(x) - n + 1, hop), len(x) - n))
        c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
        e = c[starts + n] - c[starts]
        if mode == "eval" or e.max() <= 0:
            s = int(starts[int(np.argmax(e))])
        else:
            s = int(starts[int(rng.choice(np.flatnonzero(e >= 0.25 * e.max())))])
        return x[s : s + n].astype(np.float32)
    w = np.zeros(n, dtype=np.float32)
    off = (n - len(x)) // 2 if mode == "eval" else int(rng.integers(0, n - len(x) + 1))
    w[off : off + len(x)] = x
    return w


def make_window(x: np.ndarray, rng: np.random.Generator, mode: str, window_s: float = WINDOW_S) -> np.ndarray:
    w = cut_window(np.asarray(x, dtype=np.float32), int(round(window_s * SR)), rng, mode)
    r = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
    if r > 0:
        w = w + (rng.standard_normal(len(w)) * r * 10 ** (DITHER_DB / 20.0)).astype(np.float32)
    return normalize_window(w, NORMALIZATION)


@dataclass
class WindowSet:
    feat: np.ndarray
    patch: np.ndarray | None
    y: np.ndarray
    fold: np.ndarray
    source: np.ndarray
    group: np.ndarray
    subtype: np.ndarray
    aug: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    emb: np.ndarray | None = None  # YAMNet window embeddings (optional candidate)

    def __len__(self) -> int:
        return len(self.y)

    def take(self, mask: np.ndarray) -> "WindowSet":
        m = np.asarray(mask)
        return WindowSet(self.feat[m], None if self.patch is None else self.patch[m], self.y[m], self.fold[m], self.source[m], self.group[m], self.subtype[m], self.aug[m], self.meta,
                         None if self.emb is None else self.emb[m])


def build_windows(rows: Sequence[dict[str, Any]], seed: int = 42, with_patch: bool = True, with_yamnet: bool = False, n_aug: int = 1, workers: int = 8, window_s: float = WINDOW_S) -> WindowSet:
    """Cut and featurise windows for ``audio_v2.combined_rows`` rows (in memory; nothing is written to disk)."""
    yam = None
    if with_yamnet:
        from homeease.ml.audio.embed import shared_embedder

        yam = shared_embedder()

    def one(i: int) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None, bool]]:
        r = rows[i]
        x = load_audio(r["path"])
        out = []
        for j, (mode, aug) in enumerate([("eval", False)] + [("aug", True)] * n_aug):
            w = make_window(x, np.random.default_rng([seed, i, j]), mode, window_s)
            out.append((clip_features(w), logmel_patch(w, PATCH_FRAMES) if with_patch else None, yam.window_embedding(w) if yam else None, aug))
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, range(len(rows))))
    feat, patch, emb, y, fold, src, grp, sub, aug = [], [], [], [], [], [], [], [], []
    for r, wins in zip(rows, res):
        for f, p, e, a in wins:
            feat.append(f)
            patch.append(p)
            emb.append(e)
            y.append(CLASSES.index(r["homeease_class"]))
            fold.append(int(r["fold"]))
            src.append(r["source"])
            grp.append(r["group"])
            sub.append(r.get("subtype", ""))
            aug.append(a)
    return WindowSet(np.stack(feat).astype(np.float32), np.stack(patch).astype(np.float32) if with_patch else None, np.asarray(y), np.asarray(fold), np.asarray(src), np.asarray(grp), np.asarray(sub), np.asarray(aug),
                     {"window_s": window_s, "normalization": NORMALIZATION, "patch_frames": PATCH_FRAMES, "seed": seed, "n_aug": n_aug},
                     np.stack(emb).astype(np.float32) if with_yamnet else None)


# --------------------------------------------------------------------------- candidates
@dataclass
class Fitted:
    name: str
    kind: str  # sklearn | torch
    representation: str  # clip_features | logmel_patch
    model: Any
    info: dict[str, Any] = field(default_factory=dict)
    temperature: float = 1.0

    def logits(self, ws: WindowSet) -> np.ndarray | None:
        if self.kind != "torch":
            return None
        import torch

        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(ws), 512):
                out.append(self.model(torch.from_numpy(ws.patch[i : i + 512])).numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros((0, len(CLASSES)))

    def proba(self, ws: WindowSet) -> np.ndarray:
        if self.kind == "torch":
            return softmax(self.logits(ws) / self.temperature)
        p = np.asarray(self.model.predict_proba(ws.emb if self.representation == "yamnet_embedding" else ws.feat), dtype=np.float64)
        out = np.zeros((len(ws), len(CLASSES)))
        for col, c in enumerate(self.model.classes_):  # a class absent from training keeps probability 0
            out[:, int(c)] = p[:, col]
        return out


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_candidate(name: str, tr: WindowSet, va: WindowSet | None, seed: int, epochs: int | None = None) -> Fitted:
    """Fit one candidate. Classical models use train+val; the CNN uses ``va`` for early stopping, or a fixed ``epochs`` when ``va`` is None."""
    if name == "yamnet_embedding+logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if tr.emb is None:
            raise ValueError("the YAMNet candidate needs embeddings (build_windows(with_yamnet=True))")
        X = tr.emb if va is None else np.concatenate([tr.emb, va.emb])
        y = tr.y if va is None else np.concatenate([tr.y, va.y])
        # C fixed a priori (strong L2 for a 2048-d linear probe on ~2k windows); it is not tuned on any fold
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=0.05, random_state=seed))
        m.fit(X, y)
        return Fitted(name, "sklearn", "yamnet_embedding", m, {"n_train": int(len(y))})
    if name.startswith("mfcc_logmel_stats+"):
        from homeease.ml.fall.models import make_sklearn

        X = tr.feat if va is None else np.concatenate([tr.feat, va.feat])
        y = tr.y if va is None else np.concatenate([tr.y, va.y])
        m = make_sklearn(name.split("+")[1], seed)
        m.fit(X, y)
        return Fitted(name, "sklearn", "clip_features", m, {"n_train": int(len(y))})
    if name == "logmel_small_cnn":
        if tr.patch is None:
            raise ValueError("the CNN candidate needs log-mel patches (build_windows(with_patch=True))")
        empty = np.zeros((0, *tr.patch.shape[1:]), dtype=np.float32)
        Xva, yva = (empty, np.zeros(0, dtype=int)) if va is None else (va.patch, va.y)
        net, info = train_cnn(tr.patch, tr.y, Xva, yva, len(CLASSES), seed, epochs=epochs or 25)
        return Fitted(name, "torch", "logmel_patch", net, {"n_train": int(len(tr)), "best_epoch": int(info["best_epoch"]), "seconds": info["seconds"]})
    raise ValueError(f"unknown candidate {name!r}")


def cv_predict(ws: WindowSet, name: str, seed: int, folds: Sequence[int], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Grouped CV over ``folds``: test fold k, validation fold = next fold in ``folds``, train = the rest (plus training-only extra windows).

    Returns probabilities / logits for the evaluation (non-augmented) windows of ``folds`` and per-fold info."""
    folds = list(folds)
    ev = np.flatnonzero(np.isin(ws.fold, folds) & ~ws.aug)
    probs = np.full((len(ws), len(CLASSES)), np.nan)
    logits = np.full((len(ws), len(CLASSES)), np.nan)
    per_fold = []
    for i, k in enumerate(folds):
        v = folds[(i + 1) % len(folds)]
        te = (ws.fold == k) & ~ws.aug
        va = (ws.fold == v) & ~ws.aug
        tr = np.isin(ws.fold, [f for f in folds if f not in (k, v)])
        fitted = fit_candidate(name, ws.take(tr), ws.take(va), seed)
        t = ws.take(te)
        probs[te] = fitted.proba(t)
        lg = fitted.logits(t)
        if lg is not None:
            logits[te] = lg
        per_fold.append({"fold": k, "val_fold": v, "n_train": int(tr.sum()), "n_test": int(te.sum()), **{kk: vv for kk, vv in fitted.info.items() if kk != "n_train"}})
        if progress:
            progress(f"{name} fold {k}: trained on {int(tr.sum())} windows, tested on {int(te.sum())}")
    return {"index": ev, "probs": probs[ev], "logits": logits[ev] if not np.isnan(logits[ev]).any() else None, "per_fold": per_fold}


# --------------------------------------------------------------------------- decisions and metrics
def decide(probs: np.ndarray, cuts: Sequence[float] | float | None = None) -> np.ndarray:
    """Argmax class index, or UNKNOWN (-1) when the top probability is below that class's cut-off."""
    k = probs.argmax(axis=1)
    if cuts is None:
        return k
    c = np.broadcast_to(np.asarray(cuts, dtype=float), (len(CLASSES),))
    return np.where(probs[np.arange(len(k)), k] >= c[k], k, UNKNOWN)


def report(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Confusion matrix (rows: truth; columns: the four classes + UNKNOWN), class-wise P/R/F1 with Wilson intervals, macro-F1.

    UNKNOWN counts as a miss for the true class' recall and is not a prediction of any class for precision."""
    K = len(CLASSES)
    cm = np.zeros((K, K + 1), dtype=int)
    for a, b in zip(y, pred):
        cm[int(a), K if b == UNKNOWN else int(b)] += 1
    per, f1s = {}, []
    for i, c in enumerate(CLASSES):
        tp, support, predicted = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
        p = tp / predicted if predicted else None
        r = tp / support if support else None
        f1 = None if (p is None or r is None or p + r == 0) else 2 * p * r / (p + r)
        per[c] = {"precision": p, "recall": r, "f1": f1, "support": support, "predicted": predicted, "recall_ci95": M.wilson(tp, support), "precision_ci95": M.wilson(tp, predicted)}
        if support:
            f1s.append(f1 or 0.0)
    n, known = int(cm.sum()), int(cm[:, :K].sum())
    return {
        "labels": list(CLASSES) + ["UNKNOWN"], "confusion": cm.tolist(), "per_class": per, "macro_f1": float(np.mean(f1s)) if f1s else None, "n": n,
        "accuracy": float(np.trace(cm[:, :K]) / n) if n else None, "coverage": known / n if n else None,
        "accepted_accuracy": float(np.trace(cm[:, :K]) / known) if known else None, "unknown_rate": 1 - known / n if n else None,
    }


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float | None:
    return report(y, pred)["macro_f1"]


def boot_ci(groups: np.ndarray, fn: Callable[[np.ndarray], float | None], n_boot: int = 500, seed: int = 0) -> tuple[float, float] | None:
    """Percentile interval resampling whole groups (speakers, uploaders, source recordings), never single windows."""
    ug, inv = np.unique(groups, return_inverse=True)
    if len(ug) < 5:
        return None
    parts = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = fn(np.concatenate([parts[i] for i in rng.integers(0, len(ug), len(ug))]))
        if v is not None and np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) >= n_boot // 2 else None


def per_source(ws: WindowSet, y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Class-wise recall and 'predicted HELP' rate per corpus: how much of the result is the corpus rather than the sound."""
    out = {}
    h = CLASSES.index("HELP")
    for s in sorted(set(ws.source.tolist())):
        m = ws.source == s
        row = {"n": int(m.sum()), "recall": {}, "n_by_class": {}}
        for i, c in enumerate(CLASSES):
            mm = m & (y == i)
            if mm.any():
                k = int((pred[mm] == i).sum())
                row["recall"][c] = {"value": k / int(mm.sum()), "k": k, "n": int(mm.sum()), "ci95": M.wilson(k, int(mm.sum()))}
                row["n_by_class"][c] = int(mm.sum())
        neg = m & (y != h)
        if neg.any():
            k = int((pred[neg] == h).sum())
            row["false_help_rate"] = {"value": k / int(neg.sum()), "k": k, "n": int(neg.sum()), "ci95": M.wilson(k, int(neg.sum()))}
        npred = int((m & (pred == h)).sum())
        if (m & (y == h)).any() and npred:  # HELP precision inside one corpus: the corpus cannot be the cue here
            tp = int((m & (pred == h) & (y == h)).sum())
            row["help_precision"] = {"value": tp / npred, "k": tp, "n": npred, "ci95": M.wilson(tp, npred)}
        out[s] = row
    return out


def choose_cutoffs(probs: np.ndarray, y: np.ndarray, target_precision: float = 0.90, min_accept: int = 10) -> dict[str, Any]:
    """Per-class (and one global) cut-off below which a window is UNKNOWN, chosen on DEV predictions only.

    Rule: the lowest cut-off at which the accepted predictions of the class reach ``target_precision``. If the target is never
    reached (the class cannot be separated well enough), the class falls back to the cut-off with the best dev F1 of that class
    (``target_met`` False, ``fallback`` "max_dev_f1"): demanding the highest precision instead would silence the class
    (first attempt of this work package: HELP cut-off 0.99, 6% test recall). A class predicted fewer than ``min_accept`` times
    keeps the default 0.5. Never call this with test-fold predictions."""
    grid = np.round(np.arange(0.30, 0.9901, 0.01), 2)
    k, top = probs.argmax(axis=1), probs.max(axis=1)
    per: dict[str, Any] = {}
    for i, c in enumerate(CLASSES):
        sup = int((y == i).sum())
        best, fallback = None, None
        for t in grid:
            acc = (k == i) & (top >= t)
            n = int(acc.sum())
            if n < min_accept:
                break
            tp = int((y[acc] == i).sum())
            prec, rec = tp / n, (tp / sup if sup else 0.0)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            point = (float(t), prec, n, rec, f1)
            if fallback is None or f1 > fallback[4]:
                fallback = point
            if prec >= target_precision:
                best = point
                break
        sel = best or fallback
        per[c] = {
            "cutoff": sel[0] if sel else 0.5, "target_met": best is not None, "fallback": None if best or not sel else "max_dev_f1", "dev_precision": sel[1] if sel else None,
            "dev_accepted": sel[2] if sel else 0, "dev_recall": sel[3] if sel else None, "dev_f1": sel[4] if sel else None, "dev_support": sup,
        }
    glob = None
    for t in grid:
        acc = top >= t
        if int(acc.sum()) < min_accept:
            break
        if float((k[acc] == y[acc]).mean()) >= target_precision:
            glob = float(t)
            break
    return {"per_class": per, "global": glob if glob is not None else 0.5, "global_target_met": glob is not None, "target_precision": target_precision, "dev_windows": int(len(y))}


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature that minimises the negative log-likelihood on dev logits (post-hoc calibration of the CNN)."""
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    def nll(lt: float) -> float:
        z = logits / np.exp(lt)
        return float(-(z[np.arange(len(y)), y] - logsumexp(z, axis=1)).mean())

    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.25), np.log(8.0)), method="bounded").x))


def loco(ws: WindowSet, name: str, held: str, seed: int) -> dict[str, Any]:
    """Leave one corpus out: train on every other corpus (folds 1-4, fold 5 as validation), score the held-out corpus entirely.

    Reports argmax decisions (no cut-offs) so the number is about generalisation across corpora / recording channels."""
    pool = ws.source != held
    te = (ws.source == held) & ~ws.aug
    tr = pool & (ws.fold != N_FOLDS)
    va = pool & (ws.fold == N_FOLDS) & ~ws.aug
    fitted = fit_candidate(name, ws.take(tr), ws.take(va), seed)
    t = ws.take(te)
    probs = fitted.proba(t)
    pred = decide(probs)
    rep = report(t.y, pred)
    h = CLASSES.index("HELP")
    sub: dict[str, Any] = {}
    for s in sorted(set(t.subtype[t.y == h].tolist())):
        m = (t.y == h) & (t.subtype == s)
        k = int((pred[m] == h).sum())
        sub[s] = {"recall": k / int(m.sum()), "k": k, "n": int(m.sum())}
    return {
        "held_out": held, "candidate": name, "n_train": int(tr.sum()), "n_test": int(te.sum()), "train_sources": sorted(set(ws.source[tr].tolist())), "report": rep,
        "help_recall": rep["per_class"]["HELP"]["recall"], "help_recall_ci95": rep["per_class"]["HELP"]["recall_ci95"], "help_support": rep["per_class"]["HELP"]["support"],
        "false_help": per_source(t, t.y, pred)[held].get("false_help_rate"), "help_recall_by_subtype": sub,
    }
