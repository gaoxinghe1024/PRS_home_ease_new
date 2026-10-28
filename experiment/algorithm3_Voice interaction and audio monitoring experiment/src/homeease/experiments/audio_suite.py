"""Algorithm 3 research suite (A1-A6).

Data: LibriSpeech dev-clean-2 (real read speech, 26 speakers), ESC-50 (real isolated
environmental clips), and short English/Chinese replies rendered with the macOS
text-to-speech engine (synthetic, as planned in proposal 4.2). Clean isolated clips and
TTS do not represent real household audio; household false-alarm rates are not claimed.
A2 additionally has a 4-class run (HELP / IMPACT / ABNORMAL / DAILY) with VIVAE and a small FSD50K
subset as HELP corpora (class mapping v2, datasets/adapters/audio_v2.py).
A6 measures the cancel-eligibility EVIDENCE (voiceprint match + reassurance) and how often it would be a false cancel;
it uses 300 synthetic Piper voices (experiments/audio_cancel.py) and never acts on an event.
"""

from __future__ import annotations

import platform
import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from homeease.core.paths import data_dir, models_dir
from homeease.core.util import get_logger
from homeease.experiments.runner import Run, md_table, write_suite_index
from homeease.ml.common import metrics as M

log = get_logger(__name__)


def _libri() -> dict[str, list[str]]:
    root = data_dir() / "raw" / "librispeech_mini"
    spk: dict[str, list[str]] = {}
    for p in sorted(root.rglob("*.flac")):
        spk.setdefault(p.parts[-3], []).append(str(p))
    return spk


def _stable_seed(path: str) -> int:
    """Per-file seed that is the same in every process and on every machine.

    The first version used the built-in ``hash(path)``: Python salts str hashes per process (PYTHONHASHSEED), so the noise clip and the
    noise / loudspeaker realisation of A4 changed with every run, and a rerun on another machine could not reproduce the published
    noisy-condition EERs. The file name (not the absolute path, which differs between machines) is unique per LibriSpeech utterance.
    """
    return zlib.crc32(Path(path).name.encode("utf-8"))


def _esc50() -> list[dict[str, Any]]:
    from homeease.datasets.adapters.audio_sets import esc50_rows

    raw = data_dir() / "raw" / "esc50"
    return esc50_rows(raw) if (raw / "meta" / "esc50.csv").is_file() else []


# ======================================================================= A1
def exp_vad(seed: int = 42) -> dict[str, Any]:
    from homeease.ml.audio.features import SR, load_audio, mix_at_snr
    from homeease.ml.audio.vad import EnergyVAD, SileroVAD, WebRTCVAD

    spk, esc = _libri(), _esc50()
    run = Run("audio", "A1_vad", {"mixtures": 45, "seed": seed}, "LibriSpeech dev-clean-2 speech + ESC-50 non-speech (real clips, constructed mixtures)", seed)
    if not spk or not esc:
        return run.finish("data_pending", {}, missing=["LibriSpeech dev-clean-2 and ESC-50 (homeease data download)"])
    rng = np.random.default_rng(seed)
    daily = [r["path"] for r in esc if r["homeease_class"] == "DAILY"]
    impact = [r["path"] for r in esc if r["homeease_class"] in ("IMPACT", "ABNORMAL")]
    utts = [u for v in spk.values() for u in v]
    vads = {"energy": EnergyVAD(), "webrtc": WebRTCVAD(2), "silero": SileroVAD()}
    rows: dict[str, dict[str, list[int]]] = {}
    for i in range(45):
        cond = ["clean", "household_noise_10dB", "impact_sounds_in_gaps"][i % 3]
        sig, lab = [], []
        for _ in range(3):
            gap = rng.uniform(1.0, 2.5)
            if cond == "impact_sounds_in_gaps":
                g = load_audio(impact[int(rng.integers(len(impact)))])[: int(gap * SR)]
                g = np.pad(g, (0, max(0, int(gap * SR) - len(g)))) * 0.5
            else:
                g = rng.normal(0, 1e-4, int(gap * SR)).astype(np.float32)
            sig.append(g)
            lab.append(np.zeros(len(g), dtype=np.int8))
            u = load_audio(utts[int(rng.integers(len(utts)))])
            sig.append(u)
            lab.append(np.ones(len(u), dtype=np.int8))
        x = np.concatenate(sig)
        y = np.concatenate(lab)
        if cond == "household_noise_10dB":
            x = mix_at_snr(x, load_audio(daily[int(rng.integers(len(daily)))]), 10.0, rng)
        x = (x / (np.abs(x).max() + 1e-9) * 0.7).astype(np.float32)
        yf = y[: (len(y) // 160) * 160].reshape(-1, 160).mean(axis=1) > 0.5
        for name, v in vads.items():
            pf = v.frames(x)
            n = min(len(pf), len(yf))
            d = rows.setdefault(f"{name}|{cond}", {"y": [], "p": []})
            d["y"] += yf[:n].astype(int).tolist()
            d["p"] += pf[:n].astype(int).tolist()
    entries = []
    for key, d in sorted(rows.items()):
        name, cond = key.split("|")
        b = M.binary_counts(d["y"], d["p"])
        entries.append({"config": name, "condition": cond, "precision": b["precision"], "recall": b["recall"], "f1": b["f1"], "frames": b["n"]})
    for name in vads:
        ys = sum((rows[k]["y"] for k in rows if k.startswith(name + "|")), [])
        ps = sum((rows[k]["p"] for k in rows if k.startswith(name + "|")), [])
        b = M.binary_counts(ys, ps)
        entries.append({"config": name, "condition": "ALL", "precision": b["precision"], "recall": b["recall"], "f1": b["f1"], "frames": b["n"]})
    rep = md_table(entries, ["config", "condition", "precision", "recall", "f1", "frames"])
    rep += "\nFrame truth = utterance span (LibriSpeech utterances contain short internal pauses, so recall is slightly understated for all VADs). VAD only gates the speech branch; impact sounds are classified by the separate sound branch.\n"
    allrows = [e for e in entries if e["condition"] == "ALL"]
    head = "; ".join(f"{e['config']} F1 {e['f1']:.3f}" for e in allrows)
    return run.finish("completed", {"entries": entries}, head, rep)


# ======================================================================= A2
def exp_sound(seed: int = 42) -> dict[str, Any]:
    from homeease.datasets.adapters.audio_sets import MAPPING_VERSION, THREE_CLASS
    from homeease.ml.audio.features import load_audio
    from homeease.ml.audio.sound import clip_features, logmel_patch, train_cnn
    from homeease.ml.common.torch_models import predict
    from homeease.ml.fall.models import make_sklearn

    esc = [r for r in _esc50() if r["homeease_class"]]
    run = Run("audio", "A2_sound_events", {"mapping": MAPPING_VERSION, "mapping_4class": "esc50-vivae-homeease-v2", "seed": seed}, "ESC-50 mapped subset (real isolated clips), official 5 folds; 4-class run adds VIVAE + FSD50K subset + LibriSpeech (group folds)", seed)
    if not esc:
        return run.finish("data_pending", {}, missing=["ESC-50 (homeease data download --dataset esc50)"])
    classes = ["IMPACT", "ABNORMAL", "DAILY"]
    X_feat, X_patch, y, folds = [], [], [], []
    for r in esc:
        x = load_audio(r["path"])
        X_feat.append(clip_features(x))
        X_patch.append(logmel_patch(x))
        y.append(classes.index(r["homeease_class"]))
        folds.append(r["fold"])
    X_feat, X_patch, y, folds = np.stack(X_feat), np.stack(X_patch), np.asarray(y), np.asarray(folds)
    preds: dict[str, list[tuple[int, int]]] = {"mfcc_logmel_stats+rf": [], "mfcc_logmel_stats+logreg": [], "logmel_small_cnn": []}
    for k in range(1, 6):
        te = folds == k
        va = folds == (k % 5) + 1
        tr = ~(te | va)
        for name, kind in (("mfcc_logmel_stats+rf", "rf"), ("mfcc_logmel_stats+logreg", "logreg")):
            m = make_sklearn(kind, seed)
            m.fit(X_feat[tr | va], y[tr | va])
            if hasattr(m, "n_jobs"):
                m.n_jobs = 1
            preds[name] += list(zip(y[te].tolist(), m.predict(X_feat[te]).tolist()))
        cnn, _ = train_cnn(X_patch[tr], y[tr], X_patch[va], y[va], len(classes), seed, epochs=25)
        p = predict(cnn, X_patch[te], multiclass=True)
        preds["logmel_small_cnn"] += list(zip(y[te].tolist(), p.argmax(1).tolist()))
    entries = []
    for name, pr in preds.items():
        yt = [classes[a] for a, _ in pr]
        yp = [classes[b] for _, b in pr]
        rep4 = M.multiclass_report(yt, yp, classes)
        rep3 = M.multiclass_report([THREE_CLASS[c] for c in yt], [THREE_CLASS[c] for c in yp], ["ABNORMAL", "DAILY"])
        entries.append({"config": name, "macro_f1": rep4["macro_f1"], "impact_recall": rep4["per_class"]["IMPACT"]["recall"], "abnormal_recall": rep4["per_class"]["ABNORMAL"]["recall"], "daily_precision": rep4["per_class"]["DAILY"]["precision"], "macro_f1_3class_agg": rep3["macro_f1"], "report": rep4})
    rep = md_table(entries, ["config", "macro_f1", "impact_recall", "abnormal_recall", "daily_precision", "macro_f1_3class_agg"])
    rep += f"\nClass mapping {MAPPING_VERSION} (see datasets/adapters/audio_sets.py). HELP is **not evaluated** in this 3-class run: ESC-50 has no call-for-help class (the 4-class run below uses VIVAE / FSD50K). Isolated clips overstate accuracy relative to continuous household audio. Sound labels are assistive hints only.\n"
    best = max(entries, key=lambda e: e["macro_f1"] or 0)
    four = _sound_four_class(seed)
    rep = "# 3-class run (mapping v1, ESC-50 official folds, unchanged)\n\n" + rep + "\n" + four["md"]
    head = f"3-class: best macro-F1 {best['config']} {best['macro_f1']:.3f} (IMPACT/ABNORMAL/DAILY; HELP not evaluated). 4-class: {four['headline']}"
    return run.finish("completed", {"entries": entries, "classes": classes, "clips": int(len(y)), "class_counts": np.bincount(y).tolist(), "four_class": four["metrics"]}, head, rep)


def _sound_four_class(seed: int) -> dict[str, Any]:
    """4-class sound events with a real HELP class (mapping v2): grouped 5-fold CV of every candidate on 2 s windows,
    class-wise P/R with intervals, per-fold spread, per-corpus breakdown and leave-one-corpus-out HELP recall."""
    from homeease.datasets.adapters import audio_v2
    from homeease.ml.audio import sound_train as ST
    from homeease.ml.audio.sound_model import split_summary

    rows = audio_v2.combined_rows(data_dir(), seed)
    srcs = sorted({r["source"] for r in rows})
    if "vivae" not in srcs and "fsd50k" not in srcs:
        return {
            "metrics": {"status": "data_pending", "missing": ["VIVAE (homeease data download --dataset vivae)"]}, "headline": "HELP data_pending (no HELP corpus downloaded)",
            "md": "# 4-class run (mapping v2)\n\nstatus: **data_pending**. No HELP corpus on disk; run `homeease data download --dataset vivae` (93 MB, CC BY-NC 4.0). No numbers are given.\n",
        }
    yam_ok = (models_dir() / "audio" / "yamnet" / "yamnet.onnx").is_file()
    cands = [c for c in ST.CANDIDATES if yam_ok or not c.startswith("yamnet")]
    ws = ST.build_windows(rows, seed, with_patch=True, with_yamnet=yam_ok)
    out: dict[str, Any] = {
        "status": "completed", "mapping_version": audio_v2.MAPPING_VERSION_V2, "window_s": ST.WINDOW_S, "normalization": ST.NORMALIZATION, "samples": split_summary(rows), "candidates": {},
    }
    if not yam_ok:
        out["not_evaluated"] = {"yamnet_embedding+logreg": "models/audio/yamnet/yamnet.onnx missing (homeease models fetch --audio)"}
    for cand in cands:
        r = ST.cv_predict(ws, cand, seed, range(1, audio_v2.N_FOLDS + 1))
        sub = ws.take(np.isin(np.arange(len(ws)), r["index"]))
        pred = ST.decide(r["probs"])
        rep4 = ST.report(sub.y, pred)
        per_fold = []
        for k in range(1, audio_v2.N_FOLDS + 1):
            m = sub.fold == k
            fr = ST.report(sub.y[m], pred[m])
            per_fold.append({"fold": k, "n": int(m.sum()), "macro_f1": fr["macro_f1"], "help_recall": fr["per_class"]["HELP"]["recall"], "help_support": fr["per_class"]["HELP"]["support"]})
        f1s = [f["macro_f1"] for f in per_fold if f["macro_f1"] is not None]
        out["candidates"][cand] = {
            "report": rep4, "per_fold": per_fold, "macro_f1_fold_mean": float(np.mean(f1s)), "macro_f1_fold_std": float(np.std(f1s)),
            "macro_f1_ci95_group_bootstrap": ST.boot_ci(sub.group, lambda i: ST.macro_f1(sub.y[i], pred[i]), seed=seed),
            "per_source": ST.per_source(sub, sub.y, pred), "n_windows": int(len(sub)), "help_recall_ci95": rep4["per_class"]["HELP"]["recall_ci95"], "training_info": r["per_fold"],
        }
        log.info("A2 4-class %s macro-F1 %.3f", cand, rep4["macro_f1"])
    loco_sources = [s for s in ("vivae", "fsd50k") if s in srcs]
    if len(loco_sources) < 2:
        out["loco"] = {"status": "not_evaluated", "reason": "needs two corpora that both contain HELP (VIVAE and the FSD50K subset); only " + ", ".join(loco_sources) + " present"}
    else:
        out["loco"] = {"status": "completed", "runs": {c: {s: ST.loco(ws, c, s, seed) for s in loco_sources} for c in cands}}
    best = max(out["candidates"], key=lambda c: out["candidates"][c]["report"]["macro_f1"] or 0)
    out["best_by_cv_macro_f1"] = best
    b = out["candidates"][best]
    pc = b["report"]["per_class"]["HELP"]
    ci = b["macro_f1_ci95_group_bootstrap"]
    head = f"best macro-F1 {best} {b['report']['macro_f1']:.3f}" + (f" (95% CI {ci[0]:.3f}-{ci[1]:.3f})" if ci else "") + f", HELP recall {pc['recall']:.3f} / precision {pc['precision']:.3f} on {pc['support']} HELP windows"
    if out["loco"]["status"] == "completed":
        head += "; leave-one-corpus-out HELP recall " + ", ".join(f"{s} {v['help_recall']:.3f} (n={v['help_support']})" for s, v in out["loco"]["runs"][best].items())
    else:
        head += "; leave-one-corpus-out not_evaluated"
    return {"metrics": out, "headline": head, "md": _four_class_md(out)}


def _four_class_md(o: dict[str, Any]) -> str:
    cl = ["HELP", "IMPACT", "ABNORMAL", "DAILY"]
    md = f"# 4-class run (mapping `{o['mapping_version']}`, {o['window_s']:.0f} s windows, {o['normalization']}-normalised, grouped 5-fold CV)\n\n"
    sm = o["samples"]
    md += "Clips per corpus and class (each clip gives one evaluation window; groups = ESC-50 source recording, VIVAE speaker, FSD50K uploader, LibriSpeech speaker):\n\n"
    md += md_table([{"corpus": s, "groups": sm["groups"].get(s), **{c: sm["by_source_class"][s].get(c, 0) for c in cl}} for s in sorted(sm["by_source_class"])], ["corpus", "groups"] + cl)
    md += "\n## Candidates (pooled over the 5 test folds; argmax decisions, UNKNOWN is not used here)\n\n"
    rows = []
    for c, d in o["candidates"].items():
        r = d["report"]["per_class"]
        rows.append({
            "candidate": c, "macro_f1": d["report"]["macro_f1"], "macro_f1_ci95": d["macro_f1_ci95_group_bootstrap"], "fold_mean": d["macro_f1_fold_mean"], "fold_std": d["macro_f1_fold_std"],
            "HELP_P": r["HELP"]["precision"], "HELP_R": r["HELP"]["recall"], "HELP_R_ci95": d["help_recall_ci95"], "IMPACT_R": r["IMPACT"]["recall"], "ABNORMAL_R": r["ABNORMAL"]["recall"],
            "DAILY_R": r["DAILY"]["recall"], "n": d["n_windows"],
        })
    md += md_table(rows, ["candidate", "macro_f1", "macro_f1_ci95", "fold_mean", "fold_std", "HELP_P", "HELP_R", "HELP_R_ci95", "IMPACT_R", "ABNORMAL_R", "DAILY_R", "n"])
    md += "\nmacro_f1_ci95 = percentile interval of a bootstrap over whole groups (not windows); fold_mean / fold_std = mean and standard deviation of the 5 per-fold macro-F1 values; HELP_R_ci95 = Wilson interval.\n"
    for c, why in o.get("not_evaluated", {}).items():
        md += f"\n`{c}`: **not_evaluated** ({why}).\n"
    best = o["best_by_cv_macro_f1"]
    d = o["candidates"][best]
    pcs = d["report"]["per_class"]
    md += f"\n## Best candidate by CV macro-F1: `{best}`\n\n(selected on the same test folds, so its score is slightly optimistic; the deployable model's candidate is chosen on dev folds only, see `homeease train-sound`.)\n\n"
    md += "Class-wise (Wilson 95% intervals):\n\n" + md_table(
        [{"class": c, "support": pcs[c]["support"], "precision": pcs[c]["precision"], "precision_ci95": pcs[c]["precision_ci95"], "recall": pcs[c]["recall"], "recall_ci95": pcs[c]["recall_ci95"], "f1": pcs[c]["f1"]} for c in cl],
        ["class", "support", "precision", "precision_ci95", "recall", "recall_ci95", "f1"],
    )
    md += "\nConfusion matrix (rows: truth, columns: predicted):\n\n" + md_table([{"truth": c, **{p: d["report"]["confusion"][i][j] for j, p in enumerate(cl)}} for i, c in enumerate(cl)], ["truth"] + cl)
    md += "\nPer fold: " + "; ".join(f"fold {f['fold']}: macro-F1 {f['macro_f1']:.3f}, HELP recall {f['help_recall']:.2f} (n={f['help_support']})" for f in d["per_fold"]) + "\n"
    md += "\n### Per corpus (confound check: how much of the result is the corpus rather than the sound)\n\n"
    prow = []
    for s, v in d["per_source"].items():
        prow.append({
            "corpus": s, "windows": v["n"], "HELP_recall": v["recall"].get("HELP", {}).get("value"), "HELP_n": v["recall"].get("HELP", {}).get("n"),
            "HELP_precision_in_corpus": v.get("help_precision", {}).get("value"), "false_HELP_rate": v.get("false_help_rate", {}).get("value"),
            "false_HELP_ci95": v.get("false_help_rate", {}).get("ci95"), "non_HELP_n": v.get("false_help_rate", {}).get("n"),
        })
    md += md_table(prow, ["corpus", "windows", "HELP_recall", "HELP_n", "HELP_precision_in_corpus", "false_HELP_rate", "false_HELP_ci95", "non_HELP_n"])
    md += (
        "\nfalse_HELP_rate = share of that corpus' non-HELP windows predicted HELP. VIVAE holds HELP (fear/pain) AND non-HELP voice (anger, achievement, pleasure, surprise) from the same speakers and "
        "microphone, so HELP_precision_in_corpus / false_HELP_rate there cannot be explained by the corpus. The LibriSpeech row shows what read speech does to the HELP class.\n"
    )
    md += "\n## Leave-one-corpus-out (train without the corpus, score all of it; argmax decisions)\n\n"
    if o["loco"]["status"] != "completed":
        md += f"**not_evaluated**: {o['loco']['reason']}.\n"
    else:
        lrows = []
        for c, byc in o["loco"]["runs"].items():
            for s, v in byc.items():
                lrows.append({
                    "candidate": c, "held_out": s, "trained_on": ",".join(v["train_sources"]), "HELP_recall": v["help_recall"], "HELP_recall_ci95": v["help_recall_ci95"], "HELP_n": v["help_support"],
                    "false_HELP_rate": (v["false_help"] or {}).get("value"), "non_HELP_n": (v["false_help"] or {}).get("n"), "macro_f1": v["report"]["macro_f1"],
                })
        md += md_table(lrows, ["candidate", "held_out", "trained_on", "HELP_recall", "HELP_recall_ci95", "HELP_n", "false_HELP_rate", "non_HELP_n", "macro_f1"])
        md += (
            "\nA HELP recall far below the in-corpus recall means the classifier leans on the corpus / channel. The FSD50K 'Screaming' label also covers film and game effects (for example monster screams), "
            "so it is a looser notion of a human call for help than VIVAE fear/pain.\n"
        )
    md += (
        "\nAll corpora are clean, short, public clips (acted vocalisations, Freesound uploads, read speech): these numbers say nothing about older adults, real distress or a home with TV, "
        "appliances and reverberation. HELP is an assistive hint and never changes an event state.\n"
    )
    return md


# ======================================================================= A3
TEST_PHRASES = {
    "HELP": {"en": ["Help, I fell", "Please help me", "I cannot stand up"], "zh": ["\u6551\u6551\u6211", "\u6211\u8d77\u4e0d\u6765\u4e86", "\u5feb\u5e2e\u5e2e\u6211"]},
    "REASSURANCE": {"en": ["I'm okay, really", "I am fine", "Don't worry, I'm alright"], "zh": ["\u6211\u6ca1\u4e8b\u7684", "\u6211\u5f88\u597d\uff0c\u4e0d\u7528\u62c5\u5fc3", "\u6ca1\u4e8b\u6ca1\u4e8b"]},
    "OTHER": {"en": ["Can you turn off the light", "I'm watching television"], "zh": ["\u4eca\u5929\u7684\u65b0\u95fb\u5f88\u6709\u610f\u601d", "\u6211\u5728\u770b\u7535\u89c6"]},
}


def tts(text: str, voice: str, out: Path | None = None) -> np.ndarray | None:
    """One reply through the TTS abstraction (macOS say or Windows SAPI); None when no engine or voice is available.
    `out` is kept for old callers and unused: nothing is written to disk."""
    from homeease.ml.audio import tts as tts_mod

    try:
        return tts_mod.synthesize(text, voice, "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en")
    except tts_mod.TTSError as e:
        log.warning("TTS unavailable for voice %r: %s", voice, e)
        return None


def _a3_voice_plan(backend: str) -> dict[str, Any]:
    """Voices of the legacy A3 protocol per TTS backend. 'say' = the original macOS plan (config unchanged);
    'sapi' = Zira/Huihui as self, David as proxy and TV, and for Mandarin (one installed voice) an EMULATED proxy voice."""
    if backend == "say":
        return {"self": {"en": "Samantha", "zh": "Tingting"}, "proxy": {"en": "Daniel", "zh": "Grandpa (\u4e2d\u6587\uff08\u4e2d\u56fd\u5927\u9646\uff09)"}, "tv": "Fred", "proxy_zh_ratio": None}
    return {"self": {"en": "Zira", "zh": "Huihui"}, "proxy": {"en": "David", "zh": "Huihui"}, "tv": "David", "proxy_zh_ratio": 0.84}


def exp_intent(seed: int = 42) -> dict[str, Any]:
    import time as _time

    from homeease.ml.audio import tts as tts_mod
    from homeease.ml.audio.asr import WhisperASR
    from homeease.ml.audio.features import SR, load_audio, mix_at_snr, simulate_loudspeaker
    from homeease.ml.audio.intent import RuleIntent, SupervisedIntent, phrase_corpus
    from homeease.ml.audio.vad import EnergyVAD, segments

    backend = tts_mod.native_backend_name()
    plan = _a3_voice_plan(backend)
    cfg: dict[str, Any] = {"seed": seed, "asr": ["faster-whisper-tiny", "faster-whisper-base"], "test_phrases": TEST_PHRASES, "voices": {"self": ["Samantha", "Tingting"], "proxy": ["Daniel", "Grandpa (zh_CN)"]}}
    if backend == "say":  # exactly the original configuration, so the published run id's config hash still reproduces
        name, source = "A3_response_intent", "macOS TTS replies (synthetic voices) + LibriSpeech (TV speech proxy) + ESC-50 noise"
    else:  # never mixed with the macOS rows: own run name, platform and voices in the config and the data description
        cfg["voices"] = {"self": [plan["self"]["en"], plan["self"]["zh"]], "proxy": [plan["proxy"]["en"], f"{plan['proxy']['zh']} (emulated, resampling x{plan['proxy_zh_ratio']})"], "tv": plan["tv"]}
        cfg["platform"], cfg["tts_backend"] = platform.system(), backend
        name = f"A3_response_intent_{backend}"
        source = f"{platform.system()} {backend.upper()} TTS replies (synthetic voices; Mandarin proxy voice EMULATED) + LibriSpeech (TV speech proxy) + ESC-50 noise"
    run = Run("audio", name, cfg, source, seed)
    if backend == "none":
        return run.finish("not_evaluated", {}, missing=["a text-to-speech engine (macOS `say` or Windows SAPI) or own recorded replies"])
    asrs = {}
    for m in ("faster-whisper-tiny", "faster-whisper-base"):
        try:
            asrs[m] = WhisperASR(m)
        except FileNotFoundError:
            pass
    if not asrs:
        return run.finish("data_pending", {}, missing=["faster-whisper model files under models/audio/"])
    rng = np.random.default_rng(seed)
    spk, esc = _libri(), _esc50()
    daily = [r["path"] for r in esc if r["homeease_class"] == "DAILY"]
    self_voice, other_voice = plan["self"], plan["proxy"]
    rule, sup = RuleIntent(), SupervisedIntent().fit(phrase_corpus())
    clips: list[dict[str, Any]] = []

    def clip(bucket: str, expected: str, x: np.ndarray | None, lang: str, text: str) -> None:
        if x is not None:
            clips.append({"bucket": bucket, "expected": expected, "lang": lang, "text": text, "audio": x})

    for intent, by_lang in TEST_PHRASES.items():
        for lang, phrases in by_lang.items():
            for i, ph in enumerate(phrases):
                x = tts(ph, self_voice[lang])
                bucket = {"HELP": "self_call_for_help", "REASSURANCE": "self_response", "OTHER": "self_other"}[intent]
                clip(bucket, intent, x, lang, ph)
                if x is not None and daily:
                    clip("ambient_noise_5dB", intent, mix_at_snr(x, load_audio(daily[int(rng.integers(len(daily)))]), 5.0, rng), lang, ph)
    for lang, ph in (("en", "She's fine, she's just resting"), ("zh", "\u5979\u6ca1\u4e8b\uff0c\u5979\u5728\u4f11\u606f")):
        px = tts(ph, other_voice[lang])
        if px is not None and lang == "zh" and plan["proxy_zh_ratio"]:
            px = tts_mod.emulate_voice(px, plan["proxy_zh_ratio"])  # only one Mandarin voice installed: emulate a second speaker
        clip("proxy_answer", "REASSURANCE", px, lang, ph)
    utts = [u for v in spk.values() for u in v]
    for i in range(4):
        clip("tv_speech", "OTHER", simulate_loudspeaker(load_audio(utts[int(rng.integers(len(utts)))])[: 6 * SR], rng=rng), "en", "(LibriSpeech through loudspeaker simulation)")
    tv_help = tts("Help! Somebody help me!", plan["tv"])
    clip("tv_says_help", "HELP", simulate_loudspeaker(tv_help, rng=rng) if tv_help is not None else None, "en", "TV drama line 'help'")
    for i in range(3):
        clip("no_response", "NO_RESPONSE", rng.normal(0, 2e-4, 5 * SR).astype(np.float32), "en", "(silence)")
    labels = ["HELP", "REASSURANCE", "OTHER", "UNCERTAIN", "NO_RESPONSE"]
    cases: list[dict[str, Any]] = []
    entries = []
    for asr_name, asr in asrs.items():
        lat = []
        for c in clips:
            row = {k: v for k, v in c.items() if k != "audio"}
            row["asr"] = asr_name
            if not segments(EnergyVAD().frames(c["audio"])):
                row.update(transcript="", rule="NO_RESPONSE", supervised="NO_RESPONSE")
            else:
                t0 = _time.perf_counter()
                res = asr.transcribe(c["audio"], c["lang"])
                lat.append(_time.perf_counter() - t0)
                row.update(transcript=res["text"], rule=rule.predict(res["text"], res["avg_logprob"])["intent"], supervised=sup.predict(res["text"])["intent"])
            cases.append(row)
        sub = [r for r in cases if r["asr"] == asr_name]
        for model in ("rule", "supervised"):
            rep = M.multiclass_report([r["expected"] for r in sub], [r[model] for r in sub], labels)
            entries.append({"config": f"{asr_name} + {model}", "macro_f1": rep["macro_f1"], "accuracy": rep["accuracy"], "n": rep["n"], "asr_latency_s_median": float(np.median(lat)) if lat else None, "report": rep})
    buckets = sorted({c["bucket"] for c in cases})
    per_bucket = []
    for asr_name in asrs:
        for b in buckets:
            sel = [c for c in cases if c["asr"] == asr_name and c["bucket"] == b]
            per_bucket.append({"asr": asr_name, "bucket": b, "n": len(sel), "rule_correct": sum(c["rule"] == c["expected"] for c in sel), "supervised_correct": sum(c["supervised"] == c["expected"] for c in sel)})
    run.predictions(cases)
    rep_md = md_table([{k: v for k, v in e.items() if k != "report"} for e in entries], ["config", "macro_f1", "accuracy", "n", "asr_latency_s_median"]) + "\n## Per interference bucket\n\n" + md_table(per_bucket, ["asr", "bucket", "n", "rule_correct", "supervised_correct"])
    rep_md += "\nKey safety observation: proxy answers and TV speech produce REASSURANCE/HELP intents that did not come from the monitored person; this is why a reply never cancels an MVP emergency (A5). Voices are synthetic TTS; real older-adult speech, dialects and microphones are not covered.\n"
    head = "; ".join(f"{e['config']} macro-F1 {e['macro_f1']:.3f}" for e in entries)
    return run.finish("completed", {"entries": entries, "per_bucket": per_bucket}, head, rep_md)


# ======================================================================= A4
def exp_speaker(seed: int = 42) -> dict[str, Any]:
    from homeease.ml.audio.features import load_audio, mix_at_snr, simulate_loudspeaker
    from homeease.ml.audio.speaker import WeSpeakerEmbedder, eer, tar_at_far

    spk, esc = _libri(), _esc50()
    run = Run("audio", "A4_speaker_verification", {"seed": seed, "enroll_utts": 3, "max_trials_per_speaker": 25}, "LibriSpeech dev-clean-2 (real read speech, 26 speakers), speaker-disjoint dev/eval halves", seed)
    if not spk:
        return run.finish("data_pending", {}, missing=["LibriSpeech dev-clean-2"])
    rng = np.random.default_rng(seed)
    speakers = sorted(spk)
    rng.shuffle(speakers)
    dev, ev = speakers[: len(speakers) // 2], speakers[len(speakers) // 2 :]
    daily = [r["path"] for r in esc if r["homeease_class"] == "DAILY"]
    entries = []
    for model_name in ("voxceleb_resnet34_LM.onnx", "cnceleb_resnet34_LM.onnx"):
        try:
            emb = WeSpeakerEmbedder(model_name)
        except FileNotFoundError as e:
            entries.append({"config": model_name, "status": f"missing: {e}"})
            continue
        for cond in ("clean", "noise_5dB", "replay_sim"):
            cache: dict[str, Any] = {}

            def E(path: str) -> np.ndarray:
                if path not in cache:
                    x = load_audio(path)
                    if cond == "noise_5dB" and daily:
                        x = mix_at_snr(x, load_audio(daily[_stable_seed(path) % len(daily)]), 5.0, np.random.default_rng(_stable_seed(path)))
                    elif cond == "replay_sim":
                        x = simulate_loudspeaker(x, rng=np.random.default_rng(_stable_seed(path)))
                    cache[path] = emb.embed(x)
                return cache[path]

            res = {}
            for part, group in (("dev", dev), ("eval", ev)):
                enroll = {s: np.mean([emb.embed(load_audio(p)) for p in spk[s][:3]], axis=0) for s in group}
                scores, labels = [], []
                for s in group:
                    for p in spk[s][3 : 3 + 25]:
                        e = E(p)
                        for s2, m in enroll.items():
                            scores.append(float(np.dot(e, m) / (np.linalg.norm(m) + 1e-9)))
                            labels.append(1 if s2 == s else 0)
                res[part] = (np.asarray(scores), np.asarray(labels))
            th_dev = tar_at_far(*res["dev"], far=0.001)
            s_ev, y_ev = res["eval"]
            th = th_dev.get("threshold")
            tar_ev = float(np.mean(s_ev[y_ev == 1] > th)) if th is not None else None
            far_ev = float(np.mean(s_ev[y_ev == 0] > th)) if th is not None else None
            e_ev = eer(s_ev, y_ev)
            entries.append({
                "config": model_name.replace(".onnx", ""), "condition": cond, "eer_eval": e_ev["eer"], "tar_at_dev_far0.1pct_threshold": tar_ev,
                "far_eval_at_that_threshold": far_ev, "genuine_trials_eval": int(y_ev.sum()), "impostor_trials_eval": int((1 - y_ev).sum()),
                "far_0.1pct_reliable": th_dev.get("reliable"),
            })
    rep = md_table([e for e in entries if "eer_eval" in e], ["config", "condition", "eer_eval", "tar_at_dev_far0.1pct_threshold", "far_eval_at_that_threshold", "genuine_trials_eval", "impostor_trials_eval", "far_0.1pct_reliable"])
    rep += "\nThreshold for FAR=0.1% chosen on the dev speakers, applied to disjoint eval speakers. With a few thousand impostor trials FAR=0.1% is only coarsely estimable; zero false accepts here would NOT prove household safety (TV, visitors, recordings of the user). The cnceleb model is trained on Chinese speech and is evaluated on English speech here (domain shift). Verification never unlocks automatic cancellation in this MVP.\n"
    ok = [e for e in entries if "eer_eval" in e]
    head = "; ".join(f"{e['config']}/{e['condition']}: EER {e['eer_eval']:.3f}" for e in ok[:6])
    return run.finish("completed" if ok else "data_pending", {"entries": entries, "dev_speakers": len(dev), "eval_speakers": len(ev)}, head, rep)


# ======================================================================= A5
def exp_alarm_behaviour(seed: int = 42) -> dict[str, Any]:
    from homeease.core.config import load_config
    from homeease.experiments.headless import HeadlessSystem
    from homeease.ml.audio.inquiry import AudioInquiryService

    run = Run("audio", "A5_alarm_behaviour", {"seed": seed}, "headless system (virtual clock) + scripted inquiry outcomes", seed)
    from homeease.ml.audio.cancel_gate import CancelGate, VoiceprintEnrollment

    cfg = load_config(None)
    cfg.audio.enabled = True
    outcomes = []
    voice = np.random.default_rng(3).normal(size=32).astype(np.float32)
    voice /= np.linalg.norm(voice)

    class ScriptedVoice:  # speaker model stand-in: every reply "sounds like" the enrolled person
        model_id = "scripted"

        def embed(self, audio: Any) -> np.ndarray:
            return voice

    # the last case scripts the strongest evidence the gate can give: enrolled voice + reassurance + confident ASR => cancel_eligible True
    for reply, transcript in (("reassurance", "I'm fine, don't worry"), ("help", "help me please"), ("no_response", None), ("audio_failure", "ERR"), ("reassurance_voiceprint_match", "I'm fine, don't worry")):
        sysm = HeadlessSystem(cfg)
        try:
            ev = sysm.rt.events.help_request(sysm.rt.subject_id, "user", "1", f"a5-{reply}", "replay")
            sysm.step(sysm.clock.mono_ms())
            n_before = len(sysm.rt.notifier.test_sink.received)

            class FakeASR:
                def transcribe(self, audio: Any, lang: Any) -> dict[str, Any]:
                    return {"text": transcript, "avg_logprob": -0.2, "model": "scripted"}

            def recorder(sec: float, sr: int = 16000) -> np.ndarray:
                if reply == "audio_failure":
                    raise RuntimeError("microphone unavailable")
                if reply == "no_response":
                    return np.zeros(int(sec * sr), dtype=np.float32)
                t = np.arange(int(sec * sr)) / sr
                return (0.3 * np.sin(2 * np.pi * 220 * t) * (t > 1) * (t < 3)).astype(np.float32)

            matched = reply == "reassurance_voiceprint_match"
            svc = AudioInquiryService(sysm.rt, recorder=recorder, asr=FakeASR(), tts=lambda text, lang: True, cancel_gate=CancelGate(0.5, embedder=ScriptedVoice()) if matched else None,
                                      enrollment=VoiceprintEnrollment.from_embeddings([voice] * 3, "scripted", 9.0) if matched else None)
            svc._state["consent"] = True
            rec = svc.run_inquiry(ev.id)
            sysm.advance(2.0)
            d = sysm.rt.events.detail(ev.id)
            outcomes.append({"reply": reply, "inquiry_status": rec.get("status"), "intent": rec.get("intent"), "cancel_eligible": rec.get("cancel_eligible"), "level_after": d["decision_level"], "status_after": d["lifecycle_status"], "notifications_before_inquiry": n_before, "closed_by_voice": d["lifecycle_status"] == "RESOLVED"})
        finally:
            sysm.close()
    ok = all(not o["closed_by_voice"] and o["level_after"] == "EMERGENCY" and o["status_after"] == "OPEN" for o in outcomes) and all(o["notifications_before_inquiry"] > 0 for o in outcomes)
    ok = ok and any(o["cancel_eligible"] is True for o in outcomes)  # the case that scripts cancel_eligible=True really produced it
    rep = md_table(outcomes, ["reply", "inquiry_status", "intent", "cancel_eligible", "level_after", "status_after", "notifications_before_inquiry", "closed_by_voice"])
    rep += ("\nPolicy check: an emergency stays EMERGENCY/OPEN after a spoken reassurance, after silence and after an audio failure, and ALSO when the voiceprint gate reports cancel_eligible=True "
            "(enrolled voice, reassurance, confident ASR): that flag is evidence only and is never acted on. Caregiver notifications were already dispatched before the inquiry ran. "
            "(ASR and speaker model are scripted here to isolate the policy; A3 measures real ASR/intent, A6 the rates of the evidence.)\n")
    return run.finish("completed", {"outcomes": outcomes, "policy_ok": ok}, f"policy {'PASSED' if ok else 'FAILED'}: voice replies never closed or downgraded an emergency", rep)


# ======================================================================= A6
def exp_cancel_eligibility(seed: int = 42, **kw: Any) -> dict[str, Any]:
    """A6: true-cancel and FALSE-cancel rates of the voiceprint cancel-eligibility evidence (implementation: experiments/audio_cancel.py).
    Evidence only: the product never cancels on it (audio.auto_cancel_emergency is fixed to false)."""
    from homeease.experiments.audio_cancel import exp_cancel_eligibility as impl

    return impl(seed, **kw)


def run_suite(only: list[str] | None = None, seed: int = 42) -> list[dict[str, Any]]:
    from homeease.experiments.audio_intent import exp_intent_bank

    # A3 = the original reply-intent protocol (TTS replies, now through the platform TTS); A3b = the phrase-bank protocol with a dev/test split.
    # `--only A3` runs both, `--only A3b` only the second.
    plan = [("A1", exp_vad), ("A2", exp_sound), ("A3", exp_intent), ("A3b", exp_intent_bank), ("A4", exp_speaker), ("A5", exp_alarm_behaviour), ("A6", exp_cancel_eligibility)]
    out = []
    for key, fn in plan:
        if only and key not in only and key.rstrip("b") not in only:
            continue
        try:
            r = fn(seed)
        except Exception as e:  # pragma: no cover
            log.exception("%s failed", key)
            r = Run("audio", key, {}, "n/a", seed).fail(e)
        out.append({"id": key, "experiment": r["name"], "status": r["status"], "data": r["data_source"], "headline": r["headline"][:300], "run_id": r["provenance"]["run_id"]})
    write_suite_index("audio", out)
    return out
