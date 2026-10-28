"""The deployable sound-event model: choose on DEV, fit on all non-test groups, score the test fold once, save as a package.

``train_sound_model`` is what ``homeease train-sound`` runs. The evaluation experiment A2 (experiments/audio_suite.py) uses the
building blocks of ``sound_train`` directly. Packages go to ``models/audio_sound/<id>/<version>/`` (gitignored) via
``sound_package.save_sound_package`` and are reloaded through the production loader before this function returns.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from homeease.core.util import get_logger
from homeease.datasets.adapters.audio_v2 import CLASSES, MAPPING_VERSION_V2, N_FOLDS
from homeease.ml.audio import sound_train as ST
from homeease.ml.audio.features import SR, load_audio

log = get_logger(__name__)


def split_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Clips and groups per source / class / fold (what a reader needs to judge sample sizes)."""
    out: dict[str, Any] = {"clips": len(rows), "by_source_class": {}, "by_fold": {}, "groups": {}}
    for r in rows:
        out["by_source_class"].setdefault(r["source"], {}).setdefault(r["homeease_class"], 0)
        out["by_source_class"][r["source"]][r["homeease_class"]] += 1
        out["by_fold"].setdefault(str(r["fold"]), {}).setdefault(r["homeease_class"], 0)
        out["by_fold"][str(r["fold"])][r["homeease_class"]] += 1
    for s in sorted({r["source"] for r in rows}):
        out["groups"][s] = len({r["group"] for r in rows if r["source"] == s})
    return out


def demo_scene(rows: Sequence[dict[str, Any]], fold: int, seed: int = 0, gap_s: float = 7.0) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """A 16 kHz scene of REAL clips from one held-out fold in a quiet noise bed (in memory only), for an end-to-end plumbing check.

    One clip each of: VIVAE HELP, FSD50K HELP, ESC-50 IMPACT, ESC-50 ABNORMAL, ESC-50 DAILY, LibriSpeech speech. Not an accuracy test."""
    rng = np.random.default_rng(seed)
    want = [("vivae", "HELP"), ("fsd50k", "HELP"), ("esc50", "IMPACT"), ("esc50", "ABNORMAL"), ("esc50", "DAILY"), ("librispeech", "DAILY")]

    def bed(s: float) -> np.ndarray:
        return (rng.standard_normal(int(s * SR)) * 10 ** (-58 / 20)).astype(np.float32)

    parts, truth, t = [bed(gap_s)], [], gap_s
    for src, cls in want:
        cand = [r for r in rows if r["fold"] == fold and r["source"] == src and r["homeease_class"] == cls]
        if not cand:
            continue
        r = cand[int(rng.integers(len(cand)))]
        x = load_audio(r["path"])[: int(4 * SR)]
        x = (x / (np.abs(x).max() + 1e-9) * 0.3).astype(np.float32)
        parts += [x, bed(gap_s)]
        truth.append({"t_start_s": round(t, 2), "t_end_s": round(t + len(x) / SR, 2), "class": cls, "source": src, "subtype": r.get("subtype")})
        t += len(x) / SR + gap_s
    return np.concatenate(parts), truth


def run_scene(model_ref: str, x: np.ndarray) -> dict[str, Any]:
    """Load a saved package through the production loader and drive SoundMonitor / Function3Pipeline on a scene (virtual clock)."""
    from homeease.core.clock import VirtualClock
    from homeease.core.config import AudioCfg
    from homeease.ml.audio.monitor import SoundMonitor
    from homeease.ml.audio.pipeline import Function3Pipeline
    from homeease.ml.audio.sound_package import load_sound_classifier

    clf = load_sound_classifier(model_ref)
    cfg, clock = AudioCfg(), VirtualClock(1000.0)
    mon = SoundMonitor(cfg.sound_monitor, clf, clock)
    rep = Function3Pipeline(cfg, clock, mon).process_array(x, SR, t0_mono_ms=1000.0)
    ev = [{"category": e.category, "t_start_s": round((e.t_start_mono_ms - 1000.0) / 1000.0, 2), "t_end_s": round((e.t_end_mono_ms - 1000.0) / 1000.0, 2), "confidence": round(e.confidence, 3)} for e in rep.events]
    st = mon.status()
    return {"model": mon.model_id, "status": st["status"], "min_conf": mon.min_conf, "min_conf_by_class": mon.min_conf_by_class, "events": ev, "windows_scored": st["windows_scored"], "windows_by_label": st["windows_by_label"]}


def train_sound_model(
    rows: Sequence[dict[str, Any]],
    *,
    model_id: str,
    version: str | None = None,
    candidates: Sequence[str] = ST.CANDIDATES,
    test_fold: int = N_FOLDS,
    seed: int = 42,
    target_precision: float = 0.90,
    root: Any = None,
    ws: ST.WindowSet | None = None,
    run_id: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """1. dev = every fold except ``test_fold``; out-of-fold predictions on dev for every candidate (grouped CV, ``cv_predict``);
    2. candidate = best dev macro-F1; CNN temperature and per-class UNKNOWN cut-offs from ITS dev predictions (precision >= target);
    3. refit on ALL dev folds (the CNN for the median early-stopping epoch of step 1), predict the untouched test fold once;
    4. save (immutable version), reload through the production loader and check that it reproduces the evaluated probabilities."""
    from homeease.ml.audio.sound_package import load_sound_classifier, save_sound_package

    say = progress or (lambda m: None)
    dev_folds = [f for f in range(1, N_FOLDS + 1) if f != test_fold]
    if ws is None:
        ws = ST.build_windows(rows, seed, with_patch="logmel_small_cnn" in candidates, with_yamnet=any(c.startswith("yamnet") for c in candidates))
    dev: dict[str, dict[str, Any]] = {}
    for name in candidates:
        r = ST.cv_predict(ws, name, seed, dev_folds, say)
        y = ws.y[r["index"]]
        dev[name] = {"res": r, "y": y, "macro_f1": ST.macro_f1(y, ST.decide(r["probs"]))}
        say(f"dev macro-F1 {name}: {dev[name]['macro_f1']:.3f}")
    best = max(dev, key=lambda n: dev[n]["macro_f1"] or 0.0)
    r, y_dev = dev[best]["res"], dev[best]["y"]
    temperature, epochs, probs_dev = 1.0, None, r["probs"]
    if r["logits"] is not None:
        temperature = ST.fit_temperature(r["logits"], y_dev)
        probs_dev = ST.softmax(r["logits"] / temperature)
        epochs = max(8, int(round(float(np.median([f["best_epoch"] for f in r["per_fold"]])))) + 1)
    cut = ST.choose_cutoffs(probs_dev, y_dev, target_precision)
    cuts = np.array([cut["per_class"][c]["cutoff"] for c in CLASSES])

    fitted = ST.fit_candidate(best, ws.take(np.isin(ws.fold, dev_folds)), None, seed, epochs)
    fitted.temperature = temperature
    te = (ws.fold == test_fold) & ~ws.aug
    t = ws.take(te)
    probs_te = fitted.proba(t)
    pred_arg, pred_cut = ST.decide(probs_te), ST.decide(probs_te, cuts)
    rep_arg, rep_cut = ST.report(t.y, pred_arg), ST.report(t.y, pred_cut)
    h = CLASSES.index("HELP")
    ci_f1 = ST.boot_ci(t.group, lambda i: ST.macro_f1(t.y[i], pred_cut[i]), seed=seed)
    ci_hr = ST.boot_ci(t.group, lambda i: float((pred_cut[i][t.y[i] == h] == h).mean()) if (t.y[i] == h).any() else None, seed=seed)
    test = {
        "fold": test_fold, "n_windows": int(len(t)), "groups": int(len(set(t.group.tolist()))), "argmax": rep_arg, "with_cutoffs": rep_cut,
        "macro_f1_ci95_group_bootstrap": ci_f1, "help_recall_ci95_group_bootstrap": ci_hr, "per_source": ST.per_source(t, t.y, pred_cut),
    }

    input_spec: dict[str, Any] = {"window_s": ST.WINDOW_S, "representation": fitted.representation, "normalization": ST.NORMALIZATION}
    if fitted.representation == "logmel_patch":
        input_spec["patch_frames"] = ST.PATCH_FRAMES
    if fitted.representation == "yamnet_embedding":
        from homeease.ml.audio.embed import shared_embedder

        input_spec.update(embedder="yamnet", embedder_sha256=shared_embedder().sha256)
    thresholds = {
        "min_conf": cut["global"], "per_class": {c: cut["per_class"][c]["cutoff"] for c in CLASSES},
        "chosen_on": "dev: out-of-fold predictions of folds " + ",".join(map(str, dev_folds)), "precision_target": target_precision, "detail": cut,
    }
    training = {
        "mapping_version": MAPPING_VERSION_V2, "candidate": best, "seed": seed,
        "split": {"dev_folds": dev_folds, "test_fold": test_fold, "by": "group: ESC-50 source recording, VIVAE speaker, FSD50K uploader, LibriSpeech speaker"},
        "samples": split_summary(rows), "dev_macro_f1_by_candidate": {n: d["macro_f1"] for n, d in dev.items()},
        "test": {
            "fold": test_fold, "n_windows": int(len(t)), "macro_f1_with_cutoffs": rep_cut["macro_f1"], "macro_f1_argmax": rep_arg["macro_f1"], "macro_f1_ci95": ci_f1,
            "help_recall_with_cutoffs": rep_cut["per_class"]["HELP"]["recall"], "help_precision_with_cutoffs": rep_cut["per_class"]["HELP"]["precision"], "coverage": rep_cut["coverage"],
        },
        "run_id": run_id, "intended_use": "assistive hint only; trained on public clean / acted corpora, not validated on household or older-adult audio",
    }
    pkg = save_sound_package(fitted.model, fitted.kind, CLASSES, input_spec, model_id, version, thresholds, {"method": "temperature", "T": temperature} if fitted.kind == "torch" else None, training, root)

    # the saved package must reproduce what was evaluated: re-cut the very same test windows and compare
    clf = load_sound_classifier(str(pkg))
    n_per_row = 1 + int(ws.meta.get("n_aug", 1))
    te_idx = np.flatnonzero(te)
    diffs = []
    for j in range(0, len(te_idx), max(1, len(te_idx) // 12)):
        row_i = int(te_idx[j]) // n_per_row
        w = ST.make_window(load_audio(rows[row_i]["path"]), np.random.default_rng([seed, row_i, 0]), "eval")
        diffs.append(float(np.abs(clf.predict_proba(w) - probs_te[j]).max()))
    return {
        "package": str(pkg), "candidate": best, "temperature": temperature, "epochs": epochs, "cutoffs": cut, "thresholds": thresholds, "test": test,
        "dev": {n: {"macro_f1": d["macro_f1"], "n_windows": int(len(d["y"])), "per_fold": d["res"]["per_fold"]} for n, d in dev.items()},
        "reload_check": {"windows_compared": len(diffs), "max_abs_probability_difference": max(diffs) if diffs else None, "classes": clf.classes, "min_conf_by_class": clf.min_conf_by_class},
        "samples": training["samples"],
    }
