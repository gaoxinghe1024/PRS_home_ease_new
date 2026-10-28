"""``homeease train-sound``: train, threshold-select and save the 4-class sound-event model, with a run report.

Reproducible from the data cards: needs the extracted ESC-50 and the VIVAE corpus (the FSD50K subset, LibriSpeech and the YAMNet
ONNX file are used when present). The package lands in ``models/audio_sound/<model_id>/<version>/`` (gitignored); the run
(config, metrics, report) lands in ``artifacts/experiments/<run_id>/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from homeease.core.paths import data_dir, models_dir, project_root
from homeease.core.util import get_logger
from homeease.experiments.runner import Run, fmt, md_table

log = get_logger(__name__)


def _shown(p: str | Path) -> str:
    """Project-relative path for reports: an absolute path would publish the machine's directory layout."""
    try:
        return Path(p).resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return str(p)


class PackageExists(Exception):
    """The requested package version is already on disk. Packages are immutable: pick a new version."""


def existing_package(model_id: str, version: str | None) -> Path | None:
    """models/audio_sound/<model_id>/<version>/ if it already holds a package (version None = a fresh timestamp, never exists)."""
    if not version:
        return None
    d = models_dir() / "audio_sound" / model_id / version
    return d if (d / "meta.json").exists() else None


def run_train_sound(model_id: str, version: str | None, candidates: list[str] | None, test_fold: int, seed: int, precision_target: float) -> dict[str, Any]:
    # checked first: the training takes minutes and the save would otherwise fail at the very end, leaving a half-written run folder
    if (d := existing_package(model_id, version)) is not None:
        raise PackageExists(f"sound package {model_id}@{version} already exists ({d}); packages are immutable. Use another --version, or omit it for a timestamp.")
    from homeease.datasets.adapters import audio_v2
    from homeease.ml.audio import sound_train as ST
    from homeease.ml.audio.sound_model import demo_scene, run_scene, train_sound_model

    rows = audio_v2.combined_rows(data_dir(), seed)
    yam_ok = (models_dir() / "audio" / "yamnet" / "yamnet.onnx").is_file()
    cands = candidates or [c for c in ST.CANDIDATES if yam_ok or not c.startswith("yamnet")]
    cfg = {"model_id": model_id, "version": version, "candidates": cands, "test_fold": test_fold, "seed": seed, "precision_target": precision_target, "mapping": audio_v2.MAPPING_VERSION_V2,
           "window_s": ST.WINDOW_S, "normalization": ST.NORMALIZATION}
    run = Run("audio", "A2b_sound_model", cfg, "ESC-50 + VIVAE + FSD50K subset + LibriSpeech (group folds, mapping v2)", seed)
    if not any(r["source"] in ("vivae", "fsd50k") for r in rows):
        return run.finish("data_pending", {}, missing=["a HELP corpus: homeease data download --dataset vivae"])
    if not any(r["source"] == "esc50" for r in rows):
        return run.finish("data_pending", {}, missing=["ESC-50: homeease data download --dataset esc50"])
    res = train_sound_model(rows, model_id=model_id, version=version, candidates=cands, test_fold=test_fold, seed=seed, target_precision=precision_target, run_id=run.run_id, progress=lambda m: print(m, flush=True))
    x, truth = demo_scene(rows, test_fold, seed)
    scene = run_scene(res["package"], x)
    res["scene_check"] = {"truth": truth, **scene, "note": "plumbing check on real held-out-fold clips in a quiet bed: not an accuracy measure"}
    del x  # the scene lived in memory only
    t = res["test"]
    rc = t["with_cutoffs"]
    pcs = rc["per_class"]
    md = f"Package: `{_shown(res['package'])}`\n\nCandidate chosen on dev: **{res['candidate']}** (dev macro-F1 per candidate: " + ", ".join(f"{n} {d['macro_f1']:.3f}" for n, d in res["dev"].items()) + ")\n\n"
    md += f"Temperature {res['temperature']:.2f}; UNKNOWN cut-offs (dev precision target {precision_target:.2f}, chosen on out-of-fold dev predictions): " + ", ".join(f"{c} {v['cutoff']:.2f}{'' if v['target_met'] else ' (target not reachable: cut-off with the best dev F1, dev precision ' + fmt(v['dev_precision']) + ', dev recall ' + fmt(v['dev_recall']) + ')'}" for c, v in res["cutoffs"]["per_class"].items()) + f"; global {res['cutoffs']['global']:.2f}\n\n"
    md += f"## Test fold {t['fold']} ({t['n_windows']} windows, {t['groups']} groups; untouched by training and threshold choice)\n\n"
    md += md_table([{"class": c, "support": pcs[c]["support"], "precision": pcs[c]["precision"], "precision_ci95": pcs[c]["precision_ci95"], "recall": pcs[c]["recall"], "recall_ci95": pcs[c]["recall_ci95"], "f1": pcs[c]["f1"]} for c in audio_v2.CLASSES],
                   ["class", "support", "precision", "precision_ci95", "recall", "recall_ci95", "f1"])
    md += f"\nmacro-F1 with cut-offs {fmt(rc['macro_f1'])} (95% group-bootstrap CI {fmt(t['macro_f1_ci95_group_bootstrap'])}); argmax without cut-offs {fmt(t['argmax']['macro_f1'])}; coverage (not UNKNOWN) {fmt(rc['coverage'])}; accuracy on accepted windows {fmt(rc['accepted_accuracy'])}\n\n"
    md += "Confusion (rows: truth; columns: HELP, IMPACT, ABNORMAL, DAILY, UNKNOWN):\n\n" + md_table([{"truth": c, **{p: rc["confusion"][i][j] for j, p in enumerate(rc["labels"])}} for i, c in enumerate(audio_v2.CLASSES)], ["truth"] + rc["labels"])
    prow = []
    for src, v in t["per_source"].items():
        prow.append({
            "corpus": src, "windows": v["n"], "HELP_recall": v["recall"].get("HELP", {}).get("value"), "HELP_n": v["recall"].get("HELP", {}).get("n"),
            "false_HELP_rate": v.get("false_help_rate", {}).get("value"), "non_HELP_n": v.get("false_help_rate", {}).get("n"),
        })
    md += "\nTest fold per corpus (decisions with the dev cut-offs; UNKNOWN counts as not HELP):\n\n" + md_table(prow, ["corpus", "windows", "HELP_recall", "HELP_n", "false_HELP_rate", "non_HELP_n"])
    md += f"\nReload check through the production loader: max |p_saved - p_evaluated| = {res['reload_check']['max_abs_probability_difference']:.2e} over {res['reload_check']['windows_compared']} windows.\n"
    sc = res["scene_check"]
    md += f"\nScene check (SoundMonitor on real test-fold clips, status {sc['status']}): " + "; ".join(f"{e['category']} @ {e['t_start_s']}s" for e in sc["events"]) + "\nTruth: " + "; ".join(f"{q['class']} ({q['source']}) @ {q['t_start_s']}s" for q in truth) + "\n"
    md += "\nTrained on clean public corpora; not validated on household or older-adult audio. The labels are assistive hints and never change an event state.\n"
    return run.finish("completed", res, f"{res['candidate']}: test-fold macro-F1 {fmt(rc['macro_f1'])}, HELP recall {fmt(pcs['HELP']['recall'])} / precision {fmt(pcs['HELP']['precision'])} (n={pcs['HELP']['support']} HELP windows) at dev cut-offs; package {_shown(res['package'])}", md)
