"""Analyses added after the review of the Function 3 results: the numbers behind the caveats in docs/results.md section 3.

    python scripts/f3_review_analyses.py                    # compute every section, write artifacts/experiments/post_review_analyses.json, print a summary
    python scripts/f3_review_analyses.py --only help_prevalence,unseen_phrases     # a subset (the JSON keeps the other sections)
    python scripts/f3_review_analyses.py --list

Each section is a small function that reads saved run folders (artifacts/experiments), the extracted
corpora, or the shipped model, and returns plain numbers with their sample sizes. Sections whose input is missing return ``{"status": "not_evaluated", ...}``
instead of a number.

Sections (what question each answers, and the finding that raised it):

* ``help_prevalence``   HELP precision is a property of the benchmark's HELP share (17.6% of the test fold). What does the saved confusion imply at lower prevalence?
* ``a6_clip_level``     the zero-event rows of A6 (household sounds, silence, real speech) count trials, but the independent units are clips / probes.
* ``a6_resampling``     the speaker-layer FAR with speakers (not trials) resampled; how well determined the dev threshold is; dev vs eval speech length.
* ``a6_lexicon``        A6's intent-layer protection re-scored with the rule-v2 cues switched off that fire only on test wording.
* ``unseen_phrases``    rule v2 on wording its author never saw (config/audio_unseen_probe.yaml), and how close the dev and test phrases are.
* ``padding``           the 2-s window of a short clip is mostly padding: is the padding a cue for HELP, and what does the shipped model do with a padded short clip?
* ``freesound_overlap`` the same Freesound recording in ESC-50 and FSD50K, in different folds; ``duplicate_effect`` (slow) measures what dropping them does.
* ``a2b_two_attempts``  the two train-sound attempts (the cut-off fallback rule was changed after the first test result).
* ``asr_determinism`` / ``piper_determinism``  where the run-to-run variation of A3b / A6 comes from, and that it can be switched off.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runs as A  # noqa: E402

OUT = A.EVIDENCE_DIR / "post_review_analyses.json"
SEED = 42


def _wilson(k: int, n: int) -> list[float]:
    from homeease.ml.common import metrics as M

    return [float(x) for x in M.wilson(int(k), int(n))]


def _missing(*what: str) -> dict[str, Any]:
    return {"status": "not_evaluated", "missing": list(what)}


# ---------------------------------------------------------------------------------------------------------------- A2b
def sec_help_prevalence() -> dict[str, Any]:
    f = A.Runs().folder("A2b")
    if f is None:
        return _missing("run A2b")
    m = json.loads((f / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    cm = np.array(m["test"]["with_cutoffs"]["confusion"])  # rows: truth HELP, IMPACT, ABNORMAL, DAILY; columns: HELP, IMPACT, ABNORMAL, DAILY, UNKNOWN
    support = cm.sum(1)
    n_help, n_all = int(support[0]), int(cm.sum())
    tp, fp, neg = int(cm[0, 0]), int(cm[1:, 0].sum()), int(support[1:].sum())
    tpr, fpr = tp / n_help, fp / neg
    prevalence = [0.176, 0.05, 0.01, 0.001, 0.0001]
    out: dict[str, Any] = {
        "status": "completed", "run": f.name, "test_windows": n_all, "help_windows": n_help, "help_share": n_help / n_all, "recall": tpr,
        "false_help_k": fp, "false_help_n": neg, "false_help_rate": fpr, "false_help_wilson95": _wilson(fp, neg), "precision": tp / (tp + fp),
        "at_prevalence": [{"prevalence": p, "precision": tpr * p / (tpr * p + fpr * (1 - p))} for p in prevalence],
    }
    src = m["test"].get("per_source", {})
    if "vivae" in src and "false_help_rate" in src["vivae"]:
        v = src["vivae"]["false_help_rate"]
        out["vivae_false_help"] = {"k": v.get("k"), "n": v.get("n"), "rate": v.get("value")}
    return out


def sec_a2b_two_attempts() -> dict[str, Any]:
    runs = A.Runs()
    a, b = runs.folder("A2b_first"), runs.folder("A2b")
    if a is None or b is None:
        return _missing("runs A2b_first and A2b")
    out: dict[str, Any] = {"status": "completed"}
    for key, f in (("first", a), ("second", b)):
        m = json.loads((f / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        t = m["test"]["with_cutoffs"]
        out[key] = {"run": f.name, "help_cutoff": m["cutoffs"]["per_class"]["HELP"]["cutoff"], "test_help_recall": t["per_class"]["HELP"]["recall"], "test_macro_f1": t["macro_f1"]}
    return out


# ---------------------------------------------------------------------------------------------------------------- A6
def _a6_folder() -> Path | None:
    return A.Runs().folder("A6")


def sec_a6_clip_level() -> dict[str, Any]:
    f = _a6_folder()
    if f is None:
        return _missing("run A6")
    m = json.loads((f / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    out: dict[str, Any] = {"status": "completed", "run": f.name}
    for name in ("household_noise_only", "silence", "enrolled_says_help"):  # one clip = one trial row here; the other scenarios have a speaker-bootstrap column
        d = m["scenarios"][name]
        op = d["operating_points"]["far_0.1pct"]
        out[name] = {"clips": d["clips"], "trials": d["trials"], "eligible_k": op["k"], "wilson_on_trials": op["wilson95"], "wilson_on_clips": _wilson(op["k"] if d["trials"] == d["clips"] else 0, d["clips"]), "trials_per_clip": d["trials"] / d["clips"]}
    rs = m["real_speech_check"]
    far = rs["at_tts_dev_thresholds"]
    out["real_speech"] = {"speakers": rs["speakers"], "probes": rs["probes"], "genuine_trials": rs["genuine_trials"], "impostor_trials": rs["impostor_trials"], "eer": rs["eer"],
                          "far_k_zero_wilson_on_trials": _wilson(0, rs["impostor_trials"]), "far_zero_wilson_on_probes": _wilson(0, rs["probes"]), "eer_resolution": 1.0 / rs["genuine_trials"]}
    rows = [json.loads(line) for line in (f / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    noise = [r for r in rows if r["source"] == "noise"]
    out["noise_transcripts"] = {"clips": len(noise), "with_a_transcript": sum(1 for r in noise if r["transcript"]), "responded": sum(1 for r in noise if r["status"] == "RESPONDED")}
    out["real_speech"]["far_points"] = {k: far[k] for k in far} if isinstance(far, dict) else None
    return out


def sec_a6_resampling() -> dict[str, Any]:
    f = _a6_folder()
    if f is None or not (f / "scores.npz").is_file():
        return _missing("run A6 with scores.npz")
    import yaml

    from homeease.experiments import audio_cancel as AC
    from homeease.ml.audio.speaker import tar_at_far  # noqa: F401  (kept for readers: the dev threshold is the FAR point of the dev impostor scores)

    dev, ev = AC.load_tables(f)
    cfg = yaml.safe_load((f / "config.yaml").read_text(encoding="utf-8"))
    m = json.loads((f / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    min_s = float(cfg["min_speech_s"])
    thr = float(m["thresholds"]["far_0.1pct"])
    rows = ev.rows
    src = np.array([r["source"] for r in rows])
    voices = np.array([r["voice"] for r in rows])
    sp = np.array([r["speech_s"] for r in rows])
    sel = np.where((src == "self_reas") & (sp >= min_s))[0]
    S, V, U = ev.S[sel], voices[sel], ev.users
    imp = (V[:, None] != U[None, :]) & np.isfinite(S)
    acc = (S >= thr) & imp
    k, n = int(acc.sum()), int(imp.sum())
    vs = np.unique(V)
    idx = {v: np.where(V == v)[0] for v in vs}
    K = np.zeros((len(vs), len(U)))
    N = np.zeros((len(vs), len(U)))
    for a, v in enumerate(vs):
        K[a], N[a] = acc[idx[v]].sum(0), imp[idx[v]].sum(0)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(2000):  # two-way speaker bootstrap: the speakers of the clips and the enrolled users are resampled independently
        wr = rng.multinomial(len(vs), np.full(len(vs), 1 / len(vs))).astype(float)
        wc = rng.multinomial(len(U), np.full(len(U), 1 / len(U))).astype(float)
        den = wr @ N @ wc
        if den > 0:
            vals.append((wr @ K @ wc) / den)
    lo, hi = (float(x) for x in np.percentile(vals, [2.5, 97.5]))
    # how well is the dev threshold determined? 215 probes x 149 other enrolments give the 32,035 "independent" impostor trials
    ok = dev.probe_speech_s >= min_s
    Sd, pk, spk = dev.S[ok], dev.probe_spk[ok], dev.speakers
    lab = pk[:, None] == spk[None, :]

    def threshold(S_: np.ndarray, lab_: np.ndarray) -> float:
        i = np.sort(S_[~lab_])[::-1]
        return float(i[int(np.floor(0.001 * len(i)))]) + 1e-9

    t0 = threshold(Sd, lab)
    ts = []
    rng = np.random.default_rng(0)
    for _ in range(500):
        r, c = rng.integers(0, Sd.shape[0], Sd.shape[0]), rng.integers(0, Sd.shape[1], Sd.shape[1])
        lb = pk[r][:, None] == spk[c][None, :]
        if (~lb).sum() > 1000:
            ts.append(threshold(Sd[np.ix_(r, c)], lb))
    tlo, thi = (float(x) for x in np.percentile(ts, [2.5, 97.5]))

    def eval_far(t: float) -> float:
        return float((((S >= t) & imp).sum()) / imp.sum())

    esp = sp[src == "self_reas"]
    dsp = np.asarray(dev.probe_speech_s)
    return {
        "status": "completed", "run": f.name, "min_speech_s": min_s,
        "eval_far": {"k": k, "n": n, "rate": k / n, "wilson_on_trials": _wilson(k, n), "speaker_bootstrap95": [lo, hi], "speakers_resampled": int(len(vs)), "users_resampled": int(len(U)), "draws": len(vals),
                     "target_inside_speaker_bootstrap": bool(lo <= 0.001 <= hi)},
        "dev_threshold": {"point": thr, "recomputed_point": t0, "bootstrap95": [tlo, thi], "eval_far_at_low": eval_far(tlo), "eval_far_at_high": eval_far(thi), "independent_dev_probes": int(ok.sum()),
                          "dev_probes": int(len(dsp)), "dev_impostor_trials": int((~lab).sum()), "draws": len(ts)},
        "speech_length": {"dev_probes": int(len(dsp)), "dev_median_s": float(np.median(dsp)), "dev_share_ge_min": float(np.mean(dsp >= min_s)),
                          "eval_clips": int(len(esp)), "eval_median_s": float(np.median(esp)), "eval_share_ge_min": float(np.mean(esp >= min_s))},
    }


def sec_a6_lexicon() -> dict[str, Any]:
    f = _a6_folder()
    if f is None or not (f / "predictions.jsonl").is_file():
        return _missing("run A6 with predictions.jsonl")
    import collections

    from homeease.experiments.audio_intent import _RuleWithoutCues, lexicon_audit
    from homeease.ml.audio.intent_v2 import RuleIntentV2
    from homeease.ml.audio.phrases import load_phrase_bank

    bank = load_phrase_bank()
    aud = lexicon_audit(bank)
    full, abl = RuleIntentV2(), _RuleWithoutCues(set(aud["only_test"]))
    rows = [json.loads(line) for line in (f / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    def count(src: str, rule: Any) -> dict[str, int]:
        c = collections.Counter(rule.predict(r["transcript"], r["asr_conf"])["intent"] for r in rows if r["source"] == src and r["status"] == "RESPONDED")
        return {k: int(c.get(k, 0)) for k in ("HELP", "REASSURANCE", "OTHER", "UNCERTAIN")}

    out: dict[str, Any] = {"status": "completed", "run": f.name, "cues_fired": aud["n_cues_fired"], "cues_only_test": len(aud["only_test"]), "clips": {}}
    for src in ("self_reas", "help", "visitor", "tv", "replay"):
        out["clips"][src] = {"n": sum(1 for r in rows if r["source"] == src), "full": count(src, full), "without_test_only_cues": count(src, abl)}
    out["saved_intent_equals_rescored"] = sum(1 for r in rows if r["status"] == "RESPONDED" and full.predict(r["transcript"], r["asr_conf"])["intent"] != r["intent"]) == 0
    out["help_phrases_distinct"] = len({r.get("phrase_id") for r in rows if r["source"] == "help"}) if any("phrase_id" in r for r in rows) else None
    return out


# ---------------------------------------------------------------------------------------------------------------- intent text
def sec_unseen_phrases() -> dict[str, Any]:
    import yaml

    from homeease.ml.audio.intent_v2 import RuleIntentV2
    from homeease.ml.audio.phrases import load_phrase_bank, near_duplicates, norm_text

    spec = yaml.safe_load((ROOT / "config" / "audio_unseen_probe.yaml").read_text(encoding="utf-8"))
    bank = load_phrase_bank()
    known = {norm_text(e.text) for e in bank.entries}
    hp, rp = spec["help"], spec["reassurance"]
    assert not any(norm_text(p) in known for p in hp + rp), "an unseen-probe phrase is in the phrase bank"
    rule = RuleIntentV2()
    h = [(p, rule.predict(p)["intent"]) for p in hp]
    s = [(p, rule.predict(p)["intent"]) for p in rp]
    nd = {str(th): len(near_duplicates(bank, th)) for th in (0.8, 0.6, 0.5)}
    return {
        "status": "completed", "help_phrases": len(hp), "help_read_as_help": sum(i == "HELP" for _, i in h), "help_read_as_help_wilson95": _wilson(sum(i == "HELP" for _, i in h), len(hp)),
        "help_missed": [p for p, i in h if i != "HELP"], "reassurance_phrases": len(rp), "reassurance_read_as_reassurance": sum(i == "REASSURANCE" for _, i in s),
        "reassurance_read_as_help": sum(i == "HELP" for _, i in s), "reassurance_read_as_help_phrases": [p for p, i in s if i == "HELP"],
        "cross_split_near_duplicate_pairs_by_jaccard": nd, "bank_entries": len(bank.entries),
    }


# ---------------------------------------------------------------------------------------------------------------- sound data
def _rows() -> list[dict[str, Any]] | None:
    from homeease.core.paths import data_dir
    from homeease.datasets.adapters import audio_v2

    rows = audio_v2.combined_rows(data_dir(), SEED)
    return rows if any(r["source"] == "vivae" for r in rows) and any(r["source"] == "esc50" for r in rows) else None


def sec_freesound_overlap() -> dict[str, Any]:
    from homeease.datasets.adapters import audio_v2

    rows = _rows()
    if rows is None:
        return _missing("ESC-50 and VIVAE under data/raw")
    o = audio_v2.freesound_overlap(rows)
    n5 = sum(1 for r in rows if r["fold"] == 5)
    return {"status": "completed", "pairs": o["n_pairs"], "cross_fold_pairs": o["cross_fold"], "help_pairs": o["help_pairs"], "clips_in_cross_fold_pairs_by_fold": {str(k): v for k, v in o["rows_by_fold"].items()},
            "test_fold_windows": n5, "test_fold_clips_with_a_training_sibling": o["rows_by_fold"].get(5, 0)}


def sec_padding() -> dict[str, Any]:
    """Padding of the 2-s window as a cue, and the shipped model on padded short clips."""
    rows = _rows()
    if rows is None:
        return _missing("ESC-50 and VIVAE under data/raw")
    from concurrent.futures import ThreadPoolExecutor

    from sklearn.metrics import roc_auc_score

    from homeease.core.paths import models_dir
    from homeease.ml.audio import sound_train as ST
    from homeease.ml.audio.features import SR, load_audio
    from homeease.ml.audio.sound_package import load_sound_classifier

    def quiet_share(i: int) -> float:
        w = ST.make_window(load_audio(rows[i]["path"]), np.random.default_rng([SEED, i, 0]), "eval")
        fr = w[: len(w) // 160 * 160].reshape(-1, 160).astype(np.float64)
        e = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
        return float(np.mean(e < e.max() - 35))  # frames more than 35 dB below the loudest: padding / digital near-silence

    with ThreadPoolExecutor(8) as ex:
        q = np.array(list(ex.map(quiet_share, range(len(rows)))))
    y = np.array([r["homeease_class"] for r in rows])
    src = np.array([r["source"] for r in rows])
    is_help = y == "HELP"
    out: dict[str, Any] = {"status": "completed", "windows": len(rows), "quiet_share_median": {}}
    for s in ("esc50", "vivae", "fsd50k", "librispeech"):
        m = src == s
        out["quiet_share_median"][s] = {"n": int(m.sum()), "all": float(np.median(q[m])), "help": float(np.median(q[m & is_help])) if (m & is_help).any() else None, "non_help": float(np.median(q[m & ~is_help]))}
    both = np.isin(src, ["esc50", "librispeech"])
    out["auc_quiet_share_help_vs_esc50_librispeech_non_help"] = float(roc_auc_score(np.r_[np.ones(is_help.sum()), np.zeros(both.sum())], np.r_[q[is_help], q[both]]))
    out["auc_quiet_share_within_corpus"] = {s: float(roc_auc_score(is_help[src == s], q[src == s])) for s in ("vivae", "fsd50k")}
    out["auc_quiet_share_pooled"] = float(roc_auc_score(is_help, q))
    pkg = models_dir() / "audio_sound" / "sound_events" / "v2"
    if not (pkg / "meta.json").is_file():
        out["model_probe"] = _missing("models/audio_sound/sound_events/v2")
        return out
    clf = load_sound_classifier(str(pkg))
    classes, cuts = list(clf.classes), clf.min_conf_by_class

    def label(w: np.ndarray) -> str:
        p = clf.predict_proba(w)
        k = int(np.argmax(p))
        return classes[k] if p[k] >= cuts.get(classes[k], 0.5) else "UNKNOWN"

    def loudest(x: np.ndarray, n: int) -> np.ndarray:
        c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
        starts = np.arange(0, max(1, len(x) - n + 1), 160)
        e = c[np.minimum(starts + n, len(x))] - c[starts]
        s0 = int(starts[int(np.argmax(e))])
        return x[s0:s0 + n]

    probe: dict[str, Any] = {"crop_s": 0.85, "description": "the same fold-5 non-HELP clips as the evaluation window (loudest 2 s) and, cut to their loudest 0.85 s, centred in the same padded 2-s window"}
    for s in ("esc50", "librispeech"):
        nat, crop = [], []
        for i, r in enumerate(rows):
            if r["fold"] != 5 or r["source"] != s:
                continue
            x = load_audio(r["path"])
            nat.append(label(ST.make_window(x, np.random.default_rng([SEED, i, 0]), "eval")))
            crop.append(label(ST.make_window(loudest(x, int(0.85 * SR)), np.random.default_rng([SEED, i, 0]), "eval")))
        n = len(nat)
        probe[s] = {"n": n, **{f"{tag}_{c.lower()}": int(sum(1 for l in lab if l == c)) for tag, lab in (("natural", nat), ("padded_short", crop)) for c in ("HELP", "UNKNOWN", "DAILY", "IMPACT", "ABNORMAL")},
                    "natural_help_wilson95": _wilson(sum(1 for l in nat if l == "HELP"), n), "padded_short_help_wilson95": _wilson(sum(1 for l in crop if l == "HELP"), n)}
    out["model_probe"] = probe
    return out


def sec_duplicate_effect() -> dict[str, Any]:
    """Slow (about 10 minutes): macro-F1 of the A2 four-class YAMNet + logistic regression with the duplicate ESC-50 clips dropped, against random drops of the same class mix."""
    rows = _rows()
    if rows is None:
        return _missing("ESC-50 and VIVAE under data/raw")
    import collections

    from homeease.ml.audio import sound_train as ST

    fsd = {str(r["freesound_id"]) for r in rows if r["source"] == "fsd50k"}
    dup = [i for i, r in enumerate(rows) if r["source"] == "esc50" and r["group"].split(":")[1] in fsd]
    cnt = collections.Counter(rows[i]["homeease_class"] for i in dup)
    ws_all = ST.build_windows(rows, SEED, with_patch=False, with_yamnet=True)  # windows depend on (seed, row index): built once

    def macro(drop: list[int]) -> float:
        keep = np.ones(len(rows), bool)
        keep[list(drop)] = False
        ws = ws_all.take(np.repeat(keep, 2))  # two windows (evaluation, augmented) per clip
        r = ST.cv_predict(ws, "yamnet_embedding+logreg", SEED, range(1, 6))
        sub = ws.take(np.isin(np.arange(len(ws)), r["index"]))
        return float(ST.report(sub.y, ST.decide(r["probs"]))["macro_f1"])

    base, real = macro([]), macro(dup)
    rng = np.random.default_rng(0)
    by_cls = {c: [i for i, r in enumerate(rows) if r["source"] == "esc50" and r["homeease_class"] == c and i not in set(dup)] for c in cnt}
    deltas = [macro([int(i) for c, k in cnt.items() for i in rng.choice(by_cls[c], k, replace=False)]) - base for _ in range(8)]
    return {"status": "completed", "dropped_clips": len(dup), "dropped_by_class": dict(cnt), "macro_f1_all": base, "macro_f1_duplicates_out": real, "delta": real - base,
            "random_drop_deltas": {"n": len(deltas), "mean": float(np.mean(deltas)), "min": float(min(deltas)), "max": float(max(deltas))}}


def sec_asr_determinism() -> dict[str, Any]:
    """Slow (about 15 minutes): do identical Whisper calls give identical text? Default decoding (random temperature fallback) against temperature 0.0, on appliance-noise clips."""
    import os

    os.environ.pop("HOMEEASE_ASR_TEMPERATURE", None)
    from homeease.core.paths import models_dir

    if not (models_dir() / "audio" / "faster-whisper-tiny" / "model.bin").is_file():
        return _missing("models/audio/faster-whisper-tiny")
    from homeease.experiments.audio_intent import _esc_pools
    from homeease.ml.audio.asr import WhisperASR
    from homeease.ml.audio.features import load_audio

    paths = _esc_pools()["test"][:40]
    default, fixed = WhisperASR("faster-whisper-tiny"), WhisperASR("faster-whisper-tiny", temperature=0.0)
    for a in (default, fixed):
        a.transcribe(np.zeros(16000, dtype=np.float32), "en")
    differ_default = differ_fixed = 0
    examples = []
    for p in paths:
        x = load_audio(p)[: 5 * 16000]
        x = np.concatenate([np.random.default_rng(0).normal(0, 3e-4, 4800), x, np.random.default_rng(1).normal(0, 3e-4, 4800)]).astype(np.float32)
        outs = [default.transcribe(x, None)["text"] for _ in range(4)]
        outs0 = [fixed.transcribe(x, None)["text"] for _ in range(4)]
        differ_default += len(set(outs)) > 1
        differ_fixed += len(set(outs0)) > 1
        if len(set(outs)) > 1 and len(examples) < 3:
            examples.append(outs)
    return {"status": "completed", "clips": len(paths), "repeats": 4, "clips_that_differ_default_decoding": int(differ_default), "clips_that_differ_temperature_0": int(differ_fixed),
            "examples_default": examples, "model": "faster-whisper-tiny", "input": "ESC-50 DAILY clips of the A3b test pool (5 s, padded), identical samples every call, one process"}


def sec_piper_determinism() -> dict[str, Any]:
    """Is Piper's synthesis repeatable? Default sampling against noise_scale = noise_w_scale = 0."""
    from homeease.ml.audio import multi_tts as MT

    ok, why = MT.availability()
    if not ok:
        return _missing(str(why))
    text = "I am fine, do not worry."
    out: dict[str, Any] = {"status": "completed", "speaker": 5, "text": text}
    for tag, det in (("default_sampling", False), ("deterministic", True)):
        e = MT.PiperSpeakers(deterministic=det)
        a, b = e.synthesize(text, 5), e.synthesize(text, 5)
        out[tag] = {"same_length": bool(len(a) == len(b)), "bit_identical": bool(len(a) == len(b) and np.array_equal(a, b)), "samples": [int(len(a)), int(len(b))]}
    return out


SECTIONS: dict[str, Callable[[], dict[str, Any]]] = {
    "help_prevalence": sec_help_prevalence, "a2b_two_attempts": sec_a2b_two_attempts, "a6_clip_level": sec_a6_clip_level, "a6_resampling": sec_a6_resampling, "a6_lexicon": sec_a6_lexicon,
    "unseen_phrases": sec_unseen_phrases, "freesound_overlap": sec_freesound_overlap, "padding": sec_padding, "piper_determinism": sec_piper_determinism,
    "duplicate_effect": sec_duplicate_effect, "asr_determinism": sec_asr_determinism,
}
SLOW = {"duplicate_effect", "asr_determinism"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", help="comma-separated sections")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip-slow", action="store_true", help="leave out duplicate_effect and asr_determinism (10-15 minutes each)")
    args = ap.parse_args(argv)
    if args.list:
        for k, fn in SECTIONS.items():
            print(f"{k:20s}{' (slow)' if k in SLOW else ''}  {(fn.__doc__ or '').strip().splitlines()[0] if fn.__doc__ else ''}")
        return 0
    names = args.only.split(",") if args.only else [k for k in SECTIONS if not (args.skip_slow and k in SLOW)]
    unknown = [n for n in names if n not in SECTIONS]
    if unknown:
        print(f"unknown section(s): {unknown}", file=sys.stderr)
        return 2
    cur = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {}
    for n in names:
        print(f"== {n}", flush=True)
        cur[n] = SECTIONS[n]()
        print(json.dumps(cur[n], indent=1, default=str)[:1800])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cur, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
