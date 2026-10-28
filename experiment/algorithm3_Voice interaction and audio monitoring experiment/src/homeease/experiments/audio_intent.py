"""A3b: spoken-reply recognition on the phrase bank (config/audio_phrases.yaml), platform independent.

Replies are rendered with the TTS abstraction (Windows SAPI or macOS say), so every number is for SYNTHETIC speech from a
handful of voices, never for older adults. The legacy A3 protocol (experiments/audio_suite.exp_intent) stays as it was.

Protocol (fixed before the test split was looked at; the code enforces the order):
1. The phrase bank is split by paraphrase family into dev and test (validated in `ml/audio/phrases.py`).
2. DEV: a grid of ASR settings x intent methods is scored on the dev split only:
   ASR model (tiny, base) x decoding bias (none, hotwords, initial prompt) x language (forced to the prompt language,
   Whisper auto-ID, auto-ID restricted to en/zh); intent method (v1 rule as published, v1 rule on Simplified script, v2 rule,
   supervised char n-gram trained on the legacy corpus, supervised trained on dev phrases, conservative combination).
   The supervised-on-dev method is scored on dev with leave-family-group-out folds, never on phrases it was trained on.
3. SELECTION (rule written here, applied by `select_config`): among DEPLOYABLE grid cells whose median ASR latency is within the budget,
   highest dev macro-F1 over HELP/REASSURANCE/OTHER; ties within 0.005 go to fewer missed self calls for help, then lower latency.
   "Forced language" cells are ORACLE cells here (the test clip's language is told to the ASR; in the product only the prompt language is
   known and people answer in the other language too), so they are scored and reported but never selected.
4. TEST: scored once for the production baseline, the selected cell and its one-factor neighbours. Nothing on the test split
   influences any choice. Test noise comes from different ESC-50 folds and test TV speech from different LibriSpeech speakers.

Safety counts (reported with every table): self HELP missed (a call for help not recognised as HELP) and TV / visitor speech
recognised as REASSURANCE (evidence that must never count without a voiceprint match). Silence is decided by the energy VAD
exactly like the inquiry service; NO_RESPONSE and UNCERTAIN are never read as safe.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np

from homeease.core.paths import data_dir
from homeease.core.util import get_logger
from homeease.experiments.runner import Run, md_table
from homeease.ml.audio import tts as tts_mod
from homeease.ml.audio.intent import RuleIntent, SupervisedIntent, phrase_corpus
from homeease.ml.audio.intent_v2 import HOTWORDS, PROMPTS, CombinedIntent, RuleIntentV1Simplified, RuleIntentV2, TextIntentLR
from homeease.ml.audio.phrases import PhraseBank, load_phrase_bank
from homeease.ml.common import metrics as M

log = get_logger(__name__)

LABELS = ["HELP", "REASSURANCE", "OTHER", "UNCERTAIN", "NO_RESPONSE"]
SEMANTIC = ["HELP", "REASSURANCE", "OTHER"]
METHODS = ["rule_v1", "rule_v1_t2s", "rule_v2", "sup_legacy", "sup_dev", "combined"]
MODELS = ("faster-whisper-tiny", "faster-whisper-base")
BIASES = ("none", "hotwords", "prompt")
LANG_MODES = ("forced", "auto", "auto_en_zh")
BASELINE = ("faster-whisper-tiny", "none", "auto", "rule_v1")  # what the inquiry service did before this step
LATENCY_BUDGET_S = 3.0  # median ASR time per reply; the inquiry window is 10 s and ASR runs after it
TIE = 0.005
PAD_S = 0.3
SR = 16000
ROLE_PREFS = {
    "sapi": {"self": {"en": ["Zira"], "zh": ["Huihui"]}, "visitor": {"en": ["David"], "zh": []}, "tv": {"en": ["David"], "zh": []}},
    "say": {"self": {"en": ["Samantha"], "zh": ["Tingting"]}, "visitor": {"en": ["Daniel"], "zh": ["Grandpa (\u4e2d\u6587\uff08\u4e2d\u56fd\u5927\u9646\uff09)"]}, "tv": {"en": ["Fred"], "zh": ["Grandpa (\u4e2d\u6587\uff08\u4e2d\u56fd\u5927\u9646\uff09)"]}},
}
EMULATION_RATIO = {"visitor": 0.84, "tv": 1.14}  # resampling pitch/speed shift of the self voice when no second real voice exists


@dataclass(frozen=True)
class ASRCfg:
    model: str
    bias: str
    lang_mode: str

    @property
    def key(self) -> str:
        return f"{self.model.replace('faster-whisper-', '')}|{self.bias}|{self.lang_mode}"


def full_grid() -> list[ASRCfg]:
    return [ASRCfg(m, b, l) for m, b, l in product(MODELS, BIASES, LANG_MODES)]


# --------------------------------------------------------------------------- voices
@dataclass(frozen=True)
class VoiceChoice:
    name: str  # installed voice used for rendering
    ratio: float | None  # None = real voice; a number = EMULATED by resampling the self voice
    label: str

    @property
    def emulated(self) -> bool:
        return self.ratio is not None


def plan_voices(backend: str, installed: list[tts_mod.Voice] | None = None) -> dict[tuple[str, str], VoiceChoice]:
    """role x lang -> voice. A role without a distinct real voice falls back to the self voice with an EMULATED shift.
    ``installed`` replaces the engine's voice list (tests)."""
    be = tts_mod.get_backend(backend)
    prefs = ROLE_PREFS.get(be.name, {})
    voices = tts_mod.available_voices(be.name) if installed is None else installed
    plan: dict[tuple[str, str], VoiceChoice] = {}
    for lang in ("en", "zh"):
        self_v = None
        for role in ("self", "visitor", "tv"):
            chosen = None
            for cand in prefs.get(role, {}).get(lang, []):
                chosen = next((v for v in voices if v.name.lower() == cand.lower() or cand.lower() in v.name.lower().split() or v.short.lower() == cand.lower()), None)
                if chosen:
                    break
            if role == "self":
                chosen = chosen or next((v for v in voices if v.lang == lang), None)
                if chosen is None:
                    raise tts_mod.TTSError(f"no {lang} voice installed for the self role")
                self_v = chosen
                plan[(role, lang)] = VoiceChoice(chosen.name, None, chosen.short)
            elif chosen is not None and chosen.name != self_v.name:  # type: ignore[union-attr]
                plan[(role, lang)] = VoiceChoice(chosen.name, None, chosen.short)
            else:
                r = EMULATION_RATIO[role]
                plan[(role, lang)] = VoiceChoice(self_v.name, r, f"{self_v.short} (emulated x{r})")  # type: ignore[union-attr]
    return plan


# --------------------------------------------------------------------------- data pools
def _libri_pool(seed: int) -> dict[str, list[dict[str, Any]]]:
    """LibriSpeech utterances without help/reassurance words in the reference transcript, by split (speaker-disjoint)."""
    import re

    root = data_dir() / "raw" / "librispeech_mini"
    if not root.is_dir():
        return {"dev": [], "test": []}
    cue = re.compile(r"\b(help|fine|ok|okay|hurt|hurts|pain|fell|fallen|ambulance|worry|alright|doctor|sorry)\b", re.I)
    by_spk: dict[str, list[dict[str, Any]]] = {}
    for tf in sorted(root.rglob("*.trans.txt")):
        for line in tf.read_text(encoding="utf-8").splitlines():
            uid, _, words = line.partition(" ")
            p = tf.parent / f"{uid}.flac"
            if p.is_file() and not cue.search(words):
                by_spk.setdefault(uid.split("-")[0], []).append({"path": str(p), "ref": words.lower(), "speaker": uid.split("-")[0]})
    speakers = sorted(by_spk)
    out: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
    rng = np.random.default_rng(seed)
    for i, s in enumerate(speakers):
        split = "dev" if i % 2 == 0 else "test"
        utts = by_spk[s]
        out[split].append(utts[int(rng.integers(len(utts)))])
    return out


def _esc_pools() -> dict[str, list[str]]:
    """ESC-50 DAILY clips for noise mixing / hallucination probe: dev = folds 1-3, test = folds 4-5; sources in both are dropped."""
    from homeease.datasets.adapters.audio_sets import esc50_rows

    raw = data_dir() / "raw" / "esc50"
    if not (raw / "meta" / "esc50.csv").is_file():
        return {"dev": [], "test": []}
    rows = [r for r in esc50_rows(raw) if r["homeease_class"] == "DAILY"]
    src_dev = {r["src_file"] for r in rows if r["fold"] <= 3}
    src_test = {r["src_file"] for r in rows if r["fold"] >= 4}
    both = src_dev & src_test
    return {"dev": [r["path"] for r in rows if r["fold"] <= 3 and r["src_file"] not in both],
            "test": [r["path"] for r in rows if r["fold"] >= 4 and r["src_file"] not in both]}


# --------------------------------------------------------------------------- clips
def _pad(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = int(PAD_S * SR)
    return np.concatenate([rng.normal(0, 3e-4, n), x, rng.normal(0, 3e-4, n)]).astype(np.float32)


def _speech_present(audio: np.ndarray) -> bool:
    from homeease.ml.audio.vad import EnergyVAD, segments

    return bool(segments(EnergyVAD().frames(audio)))


def build_clips(bank: PhraseBank, split: str, synth: Callable[[list[tuple[str, str, str]]], list[np.ndarray]], plan: dict[tuple[str, str], VoiceChoice],
                noise_paths: list[str], libri: list[dict[str, Any]], seed: int, load: Callable[[str], np.ndarray] | None = None) -> list[dict[str, Any]]:
    """Render every phrase of `split`, add noise variants, TV speech, silence and the noise-only probe. In-memory audio only."""
    from homeease.ml.audio.features import load_audio, mix_at_snr, simulate_loudspeaker
    from homeease.ml.audio.tts import emulate_voice

    load = load or load_audio
    rng = np.random.default_rng(seed + (0 if split == "dev" else 1000))
    ents = sorted(bank.select(split=split), key=lambda e: e.id)
    jobs = [(e.text, plan[(e.speaker_role, e.lang)].name, e.lang) for e in ents]
    audio: list[np.ndarray] = []
    for i in range(0, len(jobs), 100):  # one engine call per 100 phrases
        audio += synth(jobs[i : i + 100])
    clips: list[dict[str, Any]] = []
    noise = [load(p) for p in rng.permutation(noise_paths)[: max(4, len(ents) // 6)]] if noise_paths else []
    n_self = 0
    for e, x in zip(ents, audio):
        vc = plan[(e.speaker_role, e.lang)]
        if vc.ratio is not None:
            x = emulate_voice(x, vc.ratio)
        if e.speaker_role == "tv":
            x = simulate_loudspeaker(x, rng=rng)
        base = {"split": split, "role": e.speaker_role, "lang": e.lang, "expected": e.intent, "text": e.text, "group": e.family, "family": e.family,
                "voice": vc.label, "emulated": vc.emulated, "probe": False}
        bucket = {"self": "self_reply", "visitor": "visitor_answer"}.get(e.speaker_role) or ("tv_says_help" if e.intent == "HELP" else "tv_speech")
        clips.append({**base, "cid": e.id, "bucket": bucket, "audio": _pad(x, rng)})
        if e.speaker_role == "self":
            if noise and n_self % 3 == 0:  # every third self phrase also at 5 dB SNR (household noise, different clips per split)
                clips.append({**base, "cid": e.id + "+n5", "bucket": "self_noise_5dB", "audio": mix_at_snr(_pad(x, rng), noise[int(rng.integers(len(noise)))], 5.0, rng)})
            n_self += 1
    for j, u in enumerate(libri[:8]):
        x = load(u["path"])[: 6 * SR]
        clips.append({"cid": f"libri-{u['speaker']}", "split": split, "bucket": "tv_speech", "role": "tv", "lang": "en", "expected": "OTHER", "text": u["ref"], "group": f"spk-{u['speaker']}",
                      "family": None, "voice": "LibriSpeech (real read speech) + loudspeaker sim", "emulated": False, "probe": False,
                      "audio": _pad(simulate_loudspeaker(x, rng=rng), rng)})
    levels = (1e-4, 3e-4, 1e-3, 2e-3) if split == "dev" else (2e-4, 5e-4, 1.5e-3, 3e-3)
    for j, lv in enumerate(levels):
        clips.append({"cid": f"silence-{j}", "split": split, "bucket": "silence", "role": "none", "lang": "en", "expected": "NO_RESPONSE", "text": None, "group": f"silence-{split}-{j}",
                      "family": None, "voice": "none", "emulated": False, "probe": False, "audio": rng.normal(0, lv, 5 * SR).astype(np.float32)})
    for j, p in enumerate(rng.permutation(noise_paths)[:12] if noise_paths else []):
        clips.append({"cid": f"noise-{j}", "split": split, "bucket": "noise_only", "role": "none", "lang": "en", "expected": "NO_RESPONSE", "text": None, "group": f"noise-{Path(p).stem}",
                      "family": None, "voice": "ESC-50 DAILY", "emulated": False, "probe": True, "audio": _pad(load(p)[: 5 * SR], rng)})
    for c in clips:
        c["speech"] = _speech_present(c["audio"])
    return clips


# --------------------------------------------------------------------------- ASR
def decode_args(cfg: ASRCfg, lang: str) -> tuple[str | None, dict[str, Any]]:
    language = lang if cfg.lang_mode == "forced" else None
    kw: dict[str, Any] = {}
    if cfg.lang_mode == "auto_en_zh":
        kw["restrict_languages"] = ("en", "zh")
    which = lang if cfg.lang_mode == "forced" else "auto"
    if cfg.bias == "hotwords":
        kw["hotwords"] = HOTWORDS[which]
    elif cfg.bias == "prompt":
        kw["initial_prompt"] = PROMPTS[which]
    return language, kw


def transcribe_clips(clips: list[dict[str, Any]], asr: Any, cfg: ASRCfg) -> list[dict[str, Any] | None]:
    out: list[dict[str, Any] | None] = []
    for c in clips:
        if not c["speech"]:
            out.append(None)
            continue
        language, kw = decode_args(cfg, c["lang"])
        t0 = time.perf_counter()
        r = asr.transcribe(c["audio"], language, **kw)
        out.append({"text": r["text"], "avg_logprob": r.get("avg_logprob"), "language": r.get("language"), "latency_s": time.perf_counter() - t0})
    return out


# --------------------------------------------------------------------------- intent methods
class IntentBank:
    """All intent methods; the dev-trained supervised model is scored out-of-fold on dev and fitted on all dev phrases for test."""

    def __init__(self, bank: PhraseBank, n_folds: int = 5) -> None:
        self.rule_v1, self.rule_v1s, self.rule_v2 = RuleIntent(), RuleIntentV1Simplified(), RuleIntentV2()
        self.sup_legacy = SupervisedIntent().fit(phrase_corpus())
        dev = sorted(bank.select(split="dev"), key=lambda e: e.id)
        fams = sorted({e.family for e in dev})
        self.fold_of = {f: i % n_folds for i, f in enumerate(fams)}
        self.sup_all = TextIntentLR().fit([(e.text, e.intent) for e in dev])
        self.sup_fold = [TextIntentLR().fit([(e.text, e.intent) for e in dev if self.fold_of[e.family] != k]) for k in range(n_folds)]

    def sup_for(self, clip: dict[str, Any]) -> Any:
        if clip["split"] == "dev" and clip.get("family") in self.fold_of:
            return self.sup_fold[self.fold_of[clip["family"]]]
        return self.sup_all

    def predict(self, method: str, clip: dict[str, Any], text: str, lp: float | None) -> str:
        if method == "rule_v1":
            return self.rule_v1.predict(text, lp)["intent"]
        if method == "rule_v1_t2s":
            return self.rule_v1s.predict(text, lp)["intent"]
        if method == "rule_v2":
            return self.rule_v2.predict(text, lp)["intent"]
        if method == "sup_legacy":
            return self.sup_legacy.predict(text, lp)["intent"]
        if method == "sup_dev":
            return self.sup_for(clip).predict(text, lp)["intent"]
        if method == "combined":
            return CombinedIntent(self.rule_v2, self.sup_for(clip)).predict(text, lp)["intent"]
        raise ValueError(method)

    def predict_all(self, clips: list[dict[str, Any]], trs: list[dict[str, Any] | None], method: str) -> list[str]:
        return ["NO_RESPONSE" if tr is None else self.predict(method, c, tr["text"], tr["avg_logprob"]) for c, tr in zip(clips, trs)]

    def predict_text(self, clips: list[dict[str, Any]], method: str) -> list[str | None]:
        """Reference-text upper bound: no ASR, no audio (only phrases that have a reference text)."""
        return [self.predict(method, c, c["text"], None) if c.get("text") and c["bucket"] != "silence" else None for c in clips]


# --------------------------------------------------------------------------- metrics
def _macro3(y: np.ndarray, p: np.ndarray) -> float:
    f1s = []
    for k in SEMANTIC:
        tp = int(((y == k) & (p == k)).sum())
        fp = int(((y != k) & (p == k)).sum())
        fn = int(((y == k) & (p != k)).sum())
        if tp + fn == 0:
            continue
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def score(clips: list[dict[str, Any]], preds: list[str], trs: list[dict[str, Any] | None] | None = None, ci: bool = False) -> dict[str, Any]:
    main = [i for i, c in enumerate(clips) if not c["probe"] and preds[i] is not None]
    y = np.array([clips[i]["expected"] for i in main])
    p = np.array([preds[i] for i in main])
    rep = M.multiclass_report(list(y), list(p), LABELS)
    for k in LABELS:
        pc = rep["per_class"][k]
        tp = int(((y == k) & (p == k)).sum())
        pc["recall_ci"] = M.wilson(tp, pc["support"]) if pc["support"] else None
        n_pred = int((p == k).sum())
        pc["precision_ci"] = M.wilson(tp, n_pred) if n_pred else None
        pc["n_pred"] = n_pred
    role = np.array([clips[i]["role"] for i in main])
    self_help = (role == "self") & (y == "HELP")
    nonself = np.isin(role, ["tv", "visitor"])
    safety = {
        "self_help_n": int(self_help.sum()),
        "self_help_missed": int((self_help & (p != "HELP")).sum()),
        "self_help_as_reassurance": int((self_help & (p == "REASSURANCE")).sum()),
        "nonself_n": int(nonself.sum()),
        "nonself_reassurance_accepted": int((nonself & (p == "REASSURANCE")).sum()),
        "nonself_reassurance_but_words_not_reassuring": int((nonself & (p == "REASSURANCE") & (y != "REASSURANCE")).sum()),
        "nonself_words_reassuring_n": int((nonself & (y == "REASSURANCE")).sum()),
        "self_false_reassurance": int(((role == "self") & (y != "REASSURANCE") & (p == "REASSURANCE")).sum()),
        "self_not_reassurance_n": int(((role == "self") & (y != "REASSURANCE")).sum()),
        "noresp_missed_as_speech": int(((y == "NO_RESPONSE") & (p != "NO_RESPONSE")).sum()),
        # an explicit HELP may raise PENDING to EMERGENCY, so words that are not HELP but read as HELP are false escalations (safe direction, costs a false alarm)
        "false_help_n": int((~np.isin(y, ["HELP", "NO_RESPONSE"])).sum()),
        "false_help": int((~np.isin(y, ["HELP", "NO_RESPONSE"]) & (p == "HELP")).sum()),
    }
    safety["false_help_ci"] = M.wilson(safety["false_help"], safety["false_help_n"]) if safety["false_help_n"] else None
    safety["self_help_missed_ci"] = M.wilson(safety["self_help_missed"], safety["self_help_n"]) if safety["self_help_n"] else None
    safety["nonself_reassurance_accepted_ci"] = M.wilson(safety["nonself_reassurance_accepted"], safety["nonself_n"]) if safety["nonself_n"] else None
    buckets: dict[str, dict[str, Any]] = {}
    for b in sorted({clips[i]["bucket"] for i in main}):
        idx = [j for j, i in enumerate(main) if clips[i]["bucket"] == b]
        k = int(sum(p[j] == y[j] for j in idx))
        buckets[b] = {"n": len(idx), "correct": k, "accuracy": k / len(idx), "ci": M.wilson(k, len(idx))}
    out: dict[str, Any] = {"n": len(main), "macro_f1": _macro3(y, p), "accuracy": rep["accuracy"], "report": rep, "safety": safety, "buckets": buckets}
    if ci:
        groups = [clips[i]["group"] for i in main]
        out["macro_f1_ci"] = M.grouped_bootstrap(groups, lambda idx: _macro3(y[idx], p[idx]), n_boot=1000, seed=7)
    if trs is not None:
        lat = [t["latency_s"] for t in trs if t is not None]
        out["latency_s"] = {"n": len(lat), "median": float(np.median(lat)) if lat else None, "p95": float(np.percentile(lat, 95)) if lat else None}
    return out


def probe_stats(clips: list[dict[str, Any]], trs: list[dict[str, Any] | None], preds: list[str]) -> dict[str, Any]:
    idx = [i for i, c in enumerate(clips) if c["probe"]]
    passed = [i for i in idx if trs[i] is not None]
    return {"n": len(idx), "passed_vad": len(passed), "help": sum(preds[i] == "HELP" for i in passed), "reassurance": sum(preds[i] == "REASSURANCE" for i in passed),
            "transcripts": [trs[i]["text"][:40] for i in passed if trs[i] and trs[i]["text"]][:6]}


# --------------------------------------------------------------------------- grid, selection
def evaluate_grid(clips: list[dict[str, Any]], asrs: dict[str, Any], cfgs: list[ASRCfg], ib: IntentBank, methods: list[str] = METHODS) -> dict[str, Any]:
    """{cfg.key: {"trs": [...], "scores": {method: score}, "preds": {method: [...]}}}"""
    out: dict[str, Any] = {}
    for cfg in cfgs:
        t0 = time.time()
        trs = transcribe_clips(clips, asrs[cfg.model], cfg)
        preds = {m: ib.predict_all(clips, trs, m) for m in methods}
        out[cfg.key] = {"cfg": cfg, "trs": trs, "preds": preds, "scores": {m: score(clips, preds[m], trs) for m in methods}}
        log.info("A3b %s %s done in %.0fs", clips[0]["split"], cfg.key, time.time() - t0)
    return out


def select_config(dev: dict[str, Any], latency_budget_s: float = LATENCY_BUDGET_S) -> tuple[tuple[str, str] | None, list[dict[str, Any]]]:
    """The pre-declared selection rule (module docstring, step 3). Returns ((cfg key, method), ranked rows)."""
    rows = []
    for key, r in dev.items():
        for m, s in r["scores"].items():
            lat = (s.get("latency_s") or {}).get("median")
            rows.append({"cfg": key, "method": m, "macro_f1": s["macro_f1"], "self_help_missed": s["safety"]["self_help_missed"], "latency_median_s": lat,
                         "oracle_language": key.split("|")[2] == "forced", "eligible": lat is not None and lat <= latency_budget_s and key.split("|")[2] != "forced"})
    elig = [r for r in rows if r["eligible"]]
    if not elig:
        return None, rows
    best = max(r["macro_f1"] for r in elig)
    near = [r for r in elig if r["macro_f1"] >= best - TIE]
    near.sort(key=lambda r: (r["self_help_missed"], r["latency_median_s"], r["cfg"], r["method"]))
    pick = near[0]
    rows.sort(key=lambda r: -r["macro_f1"])
    return (pick["cfg"], pick["method"]), rows


def neighbours(key: str) -> list[ASRCfg]:
    """One-factor neighbours of a grid cell (model, bias, language mode)."""
    model, bias, lm = key.split("|")
    full = "faster-whisper-" + model
    out = [ASRCfg("faster-whisper-base" if full.endswith("tiny") else "faster-whisper-tiny", bias, lm)]
    out += [ASRCfg(full, b, lm) for b in BIASES if b != bias]
    out += [ASRCfg(full, bias, l) for l in LANG_MODES if l != lm]
    return out


def cfg_from_key(key: str) -> ASRCfg:
    model, bias, lm = key.split("|")
    return ASRCfg("faster-whisper-" + model, bias, lm)


# --------------------------------------------------------------------------- report helpers
def _ci(v: Any) -> str:
    return "n/a" if v is None else f"[{v[0]:.2f}, {v[1]:.2f}]"


def _frac(k: int, n: int, ci: Any = None) -> str:
    return "n/a" if n == 0 else f"{k}/{n} ({k / n:.2f}{', CI ' + _ci(ci) if ci else ''})"


def _cm_md(rep: dict[str, Any]) -> str:
    labels = rep["labels"]
    rows = [{"true \\ pred": l, **{p: rep["confusion"][i][j] for j, p in enumerate(labels)}, "n": sum(rep["confusion"][i])} for i, l in enumerate(labels) if sum(rep["confusion"][i])]
    return md_table(rows, ["true \\ pred"] + labels + ["n"])


def _class_md(s: dict[str, Any]) -> str:
    rows = []
    for k in LABELS:
        pc = s["report"]["per_class"][k]
        if pc["support"] == 0 and pc["n_pred"] == 0:
            continue
        rows.append({"class": k, "support": pc["support"], "precision": pc["precision"], "precision_CI": _ci(pc["precision_ci"]), "recall": pc["recall"], "recall_CI": _ci(pc["recall_ci"]), "f1": pc["f1"]})
    return md_table(rows, ["class", "support", "precision", "precision_CI", "recall", "recall_CI", "f1"])


def _bucket_md(s: dict[str, Any]) -> str:
    return md_table([{"bucket": b, "n": v["n"], "correct": v["correct"], "accuracy": v["accuracy"], "CI": _ci(v["ci"])} for b, v in s["buckets"].items()], ["bucket", "n", "correct", "accuracy", "CI"])


def _safety_md(s: dict[str, Any]) -> str:
    sf = s["safety"]
    rows = [
        {"safety count": "self call for help NOT recognised as HELP", "value": _frac(sf["self_help_missed"], sf["self_help_n"], sf["self_help_missed_ci"])},
        {"safety count": "  of which recognised as REASSURANCE", "value": _frac(sf["self_help_as_reassurance"], sf["self_help_n"])},
        {"safety count": "TV / visitor speech recognised as REASSURANCE (would be false evidence without a voiceprint)", "value": _frac(sf["nonself_reassurance_accepted"], sf["nonself_n"], sf["nonself_reassurance_accepted_ci"])},
        {"safety count": "  of which the words are NOT reassuring (recognition error)", "value": _frac(sf["nonself_reassurance_but_words_not_reassuring"], sf["nonself_n"] - sf["nonself_words_reassuring_n"])},
        {"safety count": "self reply that is NOT reassuring but recognised as REASSURANCE", "value": _frac(sf["self_false_reassurance"], sf["self_not_reassurance_n"])},
        {"safety count": "words that are not a call for help recognised as HELP (false escalation, any speaker)", "value": _frac(sf["false_help"], sf["false_help_n"], sf["false_help_ci"])},
    ]
    return md_table(rows, ["safety count", "value"])


def _grid_row(key: str, method: str, s: dict[str, Any]) -> dict[str, Any]:
    sf = s["safety"]
    return {"asr (model|bias|language)": key, "intent": method, "macro_f1": s["macro_f1"], "accuracy": s["accuracy"], "n": s["n"],
            "self_help_missed": f"{sf['self_help_missed']}/{sf['self_help_n']}", "nonself_reassurance": f"{sf['nonself_reassurance_accepted']}/{sf['nonself_n']}",
            "asr_median_s": (s.get("latency_s") or {}).get("median"), "asr_p95_s": (s.get("latency_s") or {}).get("p95")}


GRID_COLS = ["asr (model|bias|language)", "intent", "macro_f1", "accuracy", "n", "self_help_missed", "nonself_reassurance", "asr_median_s", "asr_p95_s"]


def _marginals(dev: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    cells = [(key, m, s["macro_f1"]) for key, r in dev.items() for m, s in r["scores"].items()]
    for name, pick in (("ASR model", lambda k, m: k.split("|")[0]), ("decoding bias", lambda k, m: k.split("|")[1]), ("language", lambda k, m: k.split("|")[2]), ("intent method", lambda k, m: m)):
        lv: dict[str, list[float]] = {}
        for k, m, f in cells:
            lv.setdefault(pick(k, m), []).append(f)
        for level, vals in lv.items():
            rows.append({"factor": name, "level": level, "mean_dev_macro_f1": float(np.mean(vals)), "cells": len(vals)})
    return rows


# --------------------------------------------------------------------------- hygiene audit of the hand-written rule
def lexicon_audit(bank: PhraseBank) -> dict[str, Any]:
    """Which rule_v2 cues fire on dev wording, on test wording, or ONLY on test wording.

    The v2 lexicon was written by the same author who wrote the phrase bank, so test wording was in sight when it was written; this
    is the check of how much of it is visibly shaped by the test split. A cue that fires on test phrases and on no dev phrase is
    suspect. It describes the rule only (text, no ASR) and feeds no decision."""
    rule = RuleIntentV2()
    fire: dict[str, dict[str, int]] = {}
    for e in bank.entries:
        h, o = rule.cues(e.text)
        for c in set(h) | set(o):
            fire.setdefault(c, {"dev": 0, "test": 0})[e.split] += 1  # harm words are separate cues ("harm:fell", "not_hurt")
    only_test = sorted(k for k, v in fire.items() if v["test"] > 0 and v["dev"] == 0)
    only_dev = sorted(k for k, v in fire.items() if v["dev"] > 0 and v["test"] == 0)
    return {"n_cues_fired": len(fire), "only_test": only_test, "only_dev": only_dev, "share_only_test": len(only_test) / max(1, len(fire)), "by_cue": fire}


class _RuleWithoutCues(RuleIntentV2):
    """rule_v2 with some cues switched off (diagnostic for the hand-written lexicon, see `lexicon_ablation`)."""

    def __init__(self, drop: set[str]) -> None:
        self.drop = drop

    def cues(self, text: str) -> tuple[list[str], list[str]]:
        h, o = super().cues(text)
        return [c for c in h if c not in self.drop], [c for c in o if c not in self.drop]


def _rescore_rows(rows: list[dict[str, Any]], rule: Any, field: str) -> dict[str, Any]:
    """Score saved prediction rows (labels + text only) with ``rule`` on the reference text ("said") or on the ASR transcript ("heard")."""
    clips, preds = [], []
    for r in rows:
        text = r["said"] if field == "said" else r["heard"]
        silent = r["bucket"] == "silence" or (field == "heard" and text is None)
        preds.append("NO_RESPONSE" if silent else rule.predict(text or "")["intent"])
        clips.append({"bucket": r["bucket"], "role": r["role"], "expected": r["expected"], "probe": r["bucket"] == "noise_only", "group": r["cid"], "lang": r["lang"]})
    sc = score(clips, preds)
    return {"macro_f1": sc["macro_f1"], "self_help_missed": sc["safety"]["self_help_missed"], "self_help_n": sc["safety"]["self_help_n"], "n": sc["n"]}


def lexicon_ablation(rows: list[dict[str, Any]], bank: PhraseBank, asr_key: str) -> dict[str, Any]:
    """How much of the rule_v2 score depends on cues that exist only because of one split's wording.

    For each split, the cues that fire on that split's phrases and on no phrase of the other split are switched off and the saved
    transcripts of ``asr_key`` are re-scored (``avg_logprob`` is not saved, so the low-confidence rule is inactive in both columns).
    A diagnostic about the rule, run after the splits were scored: it selects nothing."""
    aud = lexicon_audit(bank)
    out: dict[str, Any] = {"asr": asr_key, "note": "post-selection diagnostic; selects nothing"}
    for split, dropped in (("dev", aud["only_dev"]), ("test", aud["only_test"])):
        rs = [r for r in rows if r["split"] == split and r["asr"] == asr_key]
        if not rs:
            continue
        full, abl = RuleIntentV2(), _RuleWithoutCues(set(dropped))
        out[split] = {"dropped": dropped, **{f"{f}_{k}": _rescore_rows(rs, rule, f) for f in ("said", "heard") for k, rule in (("full", full), ("ablated", abl))}}
    return out


def ablation_md(abl: dict[str, Any]) -> str:
    rows = []
    for split in ("dev", "test"):
        if split not in abl:
            continue
        a = abl[split]
        for what, f in (("reference text (no ASR)", "said"), (f"ASR transcript ({abl['asr']})", "heard")):
            full, ab = a[f"{f}_full"], a[f"{f}_ablated"]
            rows.append({"split": split, "input": what, "cues switched off": len(a["dropped"]), "macro_f1 full": full["macro_f1"], "macro_f1 without": ab["macro_f1"],
                         "self HELP missed full": f"{full['self_help_missed']}/{full['self_help_n']}", "self HELP missed without": f"{ab['self_help_missed']}/{ab['self_help_n']}"})
    return ("\n### Rule v2 without the cues that only one split's wording triggers (post-selection diagnostic)\n\n"
            + md_table(rows, ["split", "input", "cues switched off", "macro_f1 full", "macro_f1 without", "self HELP missed full", "self HELP missed without"])
            + "\nRead as a range: the true generalisation of a lexicon written without sight of the data lies between 'without' (too pessimistic: ordinary words such as pain / slipped are removed) and 'full' (optimistic).\n")


def rescore_run(run_dir: str | Path, asr_key: str = "tiny|none|auto") -> tuple[dict[str, Any], str]:
    """Post-hoc ablation of a saved A3b run (`predictions.jsonl` holds labels and text only)."""
    rows = [json.loads(line) for line in (Path(run_dir) / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    abl = lexicon_ablation(rows, load_phrase_bank(), asr_key)
    return abl, ablation_md(abl)


# --------------------------------------------------------------------------- the protocol
def run_protocol(bank: PhraseBank, synth: Callable[[list[tuple[str, str, str]]], list[np.ndarray]], plan: dict[tuple[str, str], VoiceChoice], asr_factory: Callable[[str], Any],
                 noise: dict[str, list[str]], libri: dict[str, list[dict[str, Any]]], seed: int = 42, dev_only: bool = False, grid: list[ASRCfg] | None = None,
                 load: Callable[[str], np.ndarray] | None = None, on_dev_done: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Pure protocol (injectable TTS / ASR / pools); returns metrics + markdown pieces. Test split untouched until selection is done."""
    grid = grid or full_grid()
    ib = IntentBank(bank)
    asrs = {m: asr_factory(m) for m in sorted({c.model for c in grid} | {BASELINE[0]})}
    for a in asrs.values():  # warm-up: model load / first-call cost must not land in the latency numbers
        a.transcribe(np.zeros(SR, dtype=np.float32), "en")
    dev_clips = build_clips(bank, "dev", synth, plan, noise["dev"], libri["dev"], seed, load)
    dev = evaluate_grid(dev_clips, asrs, grid, ib)
    pick, ranked = select_config(dev)
    if on_dev_done:  # the choice is written down before the test split is even rendered
        on_dev_done({"selected": pick, "dev_ranked": ranked[:30], "grid": [c.key for c in grid]})
    base_key = ASRCfg(BASELINE[0], BASELINE[1], BASELINE[2]).key
    result: dict[str, Any] = {"dev_clips": len(dev_clips), "selected": pick, "dev_ranked": ranked, "dev_marginals": _marginals(dev), "baseline": BASELINE, "dev": dev, "dev_clip_rows": dev_clips}
    result["lexicon_audit"] = lexicon_audit(bank)
    result["dev_text_ceiling"] = {m: score(dev_clips, [p if p is not None else "NO_RESPONSE" for p in ib.predict_text(dev_clips, m)], None) for m in METHODS}
    if dev_only or pick is None:
        result["test"] = None
        return result
    # ---- TEST: nothing below feeds back into any choice
    test_cfgs: dict[str, ASRCfg] = {base_key: cfg_from_key(base_key), pick[0]: cfg_from_key(pick[0])}
    for n in neighbours(pick[0]):
        test_cfgs.setdefault(n.key, n)
    for c in test_cfgs.values():
        if c.model not in asrs:
            asrs[c.model] = asr_factory(c.model)
            asrs[c.model].transcribe(np.zeros(SR, dtype=np.float32), "en")
    test_clips = build_clips(bank, "test", synth, plan, noise["test"], libri["test"], seed, load)
    test = evaluate_grid(test_clips, asrs, list(test_cfgs.values()), ib)
    result["test"] = test
    result["test_clip_rows"] = test_clips
    result["test_text_ceiling"] = {m: score(test_clips, [p if p is not None else "NO_RESPONSE" for p in ib.predict_text(test_clips, m)], None) for m in METHODS}
    result["test_selected"] = score(test_clips, test[pick[0]]["preds"][pick[1]], test[pick[0]]["trs"], ci=True)
    result["test_baseline"] = score(test_clips, test[base_key]["preds"][BASELINE[3]], test[base_key]["trs"], ci=True)
    result["test_probe"] = {k: probe_stats(test_clips, test[k]["trs"], test[k]["preds"]["rule_v2"]) for k in test}
    result["dev_probe"] = {k: probe_stats(dev_clips, dev[k]["trs"], dev[k]["preds"]["rule_v2"]) for k in dev}
    return result


def _ablation_rows(test: dict[str, Any], pick: tuple[str, str]) -> list[dict[str, Any]]:
    """Test rows around the selected cell: change one factor at a time (ASR factors) or the intent method."""
    rows = []
    key, method = pick
    rows.append({"change": "selected (dev)", **_grid_row(key, method, test[key]["scores"][method])})
    for k in test:
        if k != key:
            diff = [n for n, a, b in zip(("model", "bias", "language"), k.split("|"), key.split("|")) if a != b]
            rows.append({"change": "ASR " + ",".join(diff) if len(diff) == 1 else "ASR other", **_grid_row(k, method, test[k]["scores"][method])})
    for m in METHODS:
        if m != method:
            rows.append({"change": "intent method", **_grid_row(key, m, test[key]["scores"][m])})
    return rows


def render_report(res: dict[str, Any], profile: dict[str, Any], bank: PhraseBank) -> tuple[str, str, dict[str, Any]]:
    """(headline, markdown, json-safe metrics)"""
    pick = res["selected"]
    md = "## Setup\n\n"
    md += f"- platform: **{profile['platform']}**, TTS backend **{profile['backend']}**\n"
    md += md_table([{"role": r, "lang": l, "voice": v["label"], "emulated": "YES (resampling shift)" if v["emulated"] else "no"} for (r, l), v in sorted(profile["voices"].items())], ["role", "lang", "voice", "emulated"])
    md += "\nVoices are synthetic. Emulated voices are the self voice with a pitch/speed shift (not a second real speaker). TV speech goes through a loudspeaker simulation; LibriSpeech utterances are real read speech through the same simulation.\n"
    cnt = md_table([{"split": s, "phrases": len(bank.select(split=s)), "families": len(bank.families(s))} for s in ("dev", "test")], ["split", "phrases", "families"])
    md += f"\n### Phrase bank\n\n{cnt}\n"
    md += f"\n### Clips\n\ndev: {res['dev_clips']} clips" + (f", test: {len(res['test_clip_rows'])} clips" if res.get('test') else " (dev-only run)") + ".\n"
    md += "\n## Selection on DEV only\n\nRule: highest dev macro-F1 over HELP/REASSURANCE/OTHER among DEPLOYABLE cells (language not forced) with median ASR latency <= %.1f s; ties within %.3f go to fewer missed self calls for help, then lower latency.\n" % (LATENCY_BUDGET_S, TIE)
    if pick is None:
        md += "\nNo cell was eligible (latency budget) - nothing selected.\n"
    else:
        md += f"\n**Selected: ASR `{pick[0]}` + intent `{pick[1]}`**\n"
    top = [{"asr (model|bias|language)": r["cfg"] + ("  [oracle language]" if r["oracle_language"] else ""), "intent": r["method"], "macro_f1": r["macro_f1"], "self_help_missed": r["self_help_missed"], "asr_median_s": r["latency_median_s"]} for r in res["dev_ranked"][:12]]
    md += "\nTop 12 dev cells (cells marked oracle tell the ASR the clip's language; they are reference only and cannot be selected):\n\n" + md_table(top, ["asr (model|bias|language)", "intent", "macro_f1", "self_help_missed", "asr_median_s"])
    base_key = ASRCfg(BASELINE[0], BASELINE[1], BASELINE[2]).key
    md += "\nProduction baseline before this step on dev:\n\n" + md_table([_grid_row(base_key, BASELINE[3], res["dev"][base_key]["scores"][BASELINE[3]])], GRID_COLS)
    md += "\nMain effects on dev (mean dev macro-F1 over all cells of the other factors):\n\n" + md_table(res["dev_marginals"], ["factor", "level", "mean_dev_macro_f1", "cells"])
    md += "\nReference-text upper bound on dev (no ASR; how much the intent method alone can do):\n\n"
    md += md_table([{"intent": m, "macro_f1": s["macro_f1"], "n": s["n"]} for m, s in res["dev_text_ceiling"].items()], ["intent", "macro_f1", "n"])
    aud = res["lexicon_audit"]
    md += ("\n### Hygiene audit of the hand-written v2 rule\n\nThe v2 lexicon was written by the author of the phrase bank, with both splits in sight. So its TEST numbers (and the text upper bound) are NOT a blind estimate; "
           f"the dev/test gap cannot reveal overfitting of the rule either. Cues that fire on test phrases and on no dev phrase: {len(aud['only_test'])} of {aud['n_cues_fired']} cues that fire at all "
           f"({', '.join(aud['only_test']) or 'none'}). The v1 rule (written by the teammate before this bank existed) and the supervised models trained on the legacy corpus are blind to the test wording; sup_dev is trained on dev phrases only.\n")
    metrics: dict[str, Any] = {"lexicon_audit": {k: v for k, v in aud.items() if k != "by_cue"}, "selected": pick, "baseline": list(BASELINE), "dev_grid": [{"cfg": k, "method": m, **{kk: vv for kk, vv in _grid_row(k, m, s).items()}} for k, r in res["dev"].items() for m, s in r["scores"].items()],
                               "dev_marginals": res["dev_marginals"], "dev_text_ceiling": {m: s["macro_f1"] for m, s in res["dev_text_ceiling"].items()}, "profile": {"platform": profile["platform"], "backend": profile["backend"]}}
    if not res.get("test"):
        return (f"dev-only run; selected {pick}", md, metrics)
    t = res["test"]
    ts, tb = res["test_selected"], res["test_baseline"]
    md += "\n\n## TEST (reported once)\n\nThe test split was not used for any choice above.\n"
    md += "\n### Baseline vs selected\n\n"
    md += md_table([
        {"config": f"baseline: {base_key} + {BASELINE[3]}", **{k: v for k, v in _grid_row(base_key, BASELINE[3], t[base_key]["scores"][BASELINE[3]]).items() if k in GRID_COLS[2:]}, "macro_f1_CI": _ci(tb.get("macro_f1_ci"))},
        {"config": f"selected: {pick[0]} + {pick[1]}", **{k: v for k, v in _grid_row(pick[0], pick[1], t[pick[0]]["scores"][pick[1]]).items() if k in GRID_COLS[2:]}, "macro_f1_CI": _ci(ts.get("macro_f1_ci"))},
    ], ["config", "macro_f1", "macro_f1_CI", "accuracy", "n", "self_help_missed", "nonself_reassurance", "asr_median_s", "asr_p95_s"])
    md += "\nCI = family/speaker-grouped bootstrap (1000 resamples). Rates below carry Wilson intervals.\n"
    md += "\n### One factor at a time around the selected cell (test)\n\n" + md_table(_ablation_rows(t, pick), ["change"] + GRID_COLS)
    md += f"\n### Selected cell on test: per-class precision / recall / F1\n\n{_class_md(ts)}\n### Confusion matrix (rows = truth)\n\n{_cm_md(ts['report'])}\n### Accuracy per interference bucket\n\n{_bucket_md(ts)}\n### Safety counts\n\n{_safety_md(ts)}"
    md += f"\n### Baseline on test: per-class and safety counts\n\n{_class_md(tb)}\n{_safety_md(tb)}\n### Baseline confusion matrix\n\n{_cm_md(tb['report'])}\n"
    md += "\n### Reference-text upper bound on test (no ASR)\n\n" + md_table([{"intent": m, "macro_f1": s["macro_f1"], "n": s["n"]} for m, s in res["test_text_ceiling"].items()], ["intent", "macro_f1", "n"])
    md += "\n### Hallucination probe: ESC-50 appliance noise reaching the ASR (expected: no HELP / REASSURANCE)\n\n"
    md += md_table([{"asr": k, "noise clips": v["n"], "passed VAD": v["passed_vad"], "read as HELP": v["help"], "read as REASSURANCE": v["reassurance"], "examples": " | ".join(v["transcripts"][:3])} for k, v in res["test_probe"].items()], ["asr", "noise clips", "passed VAD", "read as HELP", "read as REASSURANCE", "examples"])
    md += "\n### Transcript examples (selected cell, test, first 12 errors)\n\n"
    errs = [(c, tr, p) for c, tr, p in zip(res["test_clip_rows"], t[pick[0]]["trs"], t[pick[0]]["preds"][pick[1]]) if not c["probe"] and p != c["expected"]][:12]
    md += md_table([{"bucket": c["bucket"], "lang": c["lang"], "said": c["text"], "heard": (tr or {}).get("text"), "expected": c["expected"], "got": p} for c, tr, p in errs], ["bucket", "lang", "said", "heard", "expected", "got"])
    metrics["test"] = {"grid": [{"cfg": k, "method": m, **_grid_row(k, m, s)} for k, r in t.items() for m, s in r["scores"].items()],
                       "selected": {"cfg": pick[0], "method": pick[1], "macro_f1": ts["macro_f1"], "macro_f1_ci": ts.get("macro_f1_ci"), "report": ts["report"], "buckets": ts["buckets"], "safety": ts["safety"], "latency_s": ts.get("latency_s")},
                       "baseline": {"cfg": base_key, "method": BASELINE[3], "macro_f1": tb["macro_f1"], "macro_f1_ci": tb.get("macro_f1_ci"), "report": tb["report"], "buckets": tb["buckets"], "safety": tb["safety"], "latency_s": tb.get("latency_s")},
                       "text_ceiling": {m: s["macro_f1"] for m, s in res["test_text_ceiling"].items()}, "probe": res["test_probe"]}
    head = (f"selected on dev: {pick[0]} + {pick[1]}; TEST macro-F1 {ts['macro_f1']:.3f} {_ci(ts.get('macro_f1_ci'))} vs baseline {tb['macro_f1']:.3f} {_ci(tb.get('macro_f1_ci'))}; "
            f"self HELP missed {ts['safety']['self_help_missed']}/{ts['safety']['self_help_n']} (baseline {tb['safety']['self_help_missed']}/{tb['safety']['self_help_n']}); "
            f"TV/visitor read as REASSURANCE {ts['safety']['nonself_reassurance_accepted']}/{ts['safety']['nonself_n']}")
    return head, md, metrics


# --------------------------------------------------------------------------- suite entry
def exp_intent_bank(seed: int = 42, dev_only: bool = False) -> dict[str, Any]:
    """A3b on this machine (needs a TTS backend, the phrase bank and faster-whisper models); data_pending / not_evaluated otherwise."""
    import platform

    backend = tts_mod.native_backend_name()
    cfg = {"seed": seed, "protocol": "A3b_phrase_bank", "platform": platform.system(), "tts_backend": backend, "grid": [c.key for c in full_grid()], "methods": METHODS,
           "baseline": list(BASELINE), "latency_budget_s": LATENCY_BUDGET_S, "dev_only": dev_only}
    name = f"A3b_response_intent_bank_{backend}" + ("_devonly" if dev_only else "")
    run = Run("audio", name, cfg, f"phrase bank config/audio_phrases.yaml rendered with {backend} voices ({platform.system()}) + LibriSpeech (TV speech) + ESC-50 noise", seed)
    if backend == "none":
        return run.finish("not_evaluated", {}, missing=["a text-to-speech engine (Windows SAPI or macOS say)"])
    try:
        bank = load_phrase_bank()
    except Exception as e:
        return run.finish("failed", {"error": repr(e)}, headline=f"phrase bank invalid: {e}")
    from homeease.ml.audio.asr import WhisperASR
    from homeease.core.paths import models_dir

    missing = [m for m in MODELS if not (models_dir() / "audio" / m / "model.bin").is_file()]
    if missing:
        return run.finish("data_pending", {}, missing=[f"faster-whisper model files under models/audio/: {', '.join(missing)}"])
    noise, libri = _esc_pools(), _libri_pool(seed)
    if not noise["dev"] or not noise["test"] or not libri["dev"]:
        return run.finish("data_pending", {}, missing=["ESC-50 and LibriSpeech dev-clean-2 (homeease data download)"])
    plan = plan_voices(backend)
    profile = {"platform": platform.system(), "backend": backend, "voices": {k: {"label": v.label, "emulated": v.emulated} for k, v in plan.items()}}
    def _save_selection(sel: dict[str, Any]) -> None:
        (run.dir / "dev_selection.json").write_text(json.dumps(sel, indent=2, default=str), encoding="utf-8")

    res = run_protocol(bank, lambda jobs: tts_mod.synthesize_many(jobs, backend), plan, WhisperASR, noise, libri, seed, dev_only, on_dev_done=_save_selection)
    head, md, metrics = render_report(res, profile, bank)
    rows = []
    for split, key in (("dev", "dev_clip_rows"), ("test", "test_clip_rows")):
        if key not in res:
            continue
        keep = [k for k in {res["selected"][0] if res["selected"] else None, ASRCfg(BASELINE[0], BASELINE[1], BASELINE[2]).key} if k]
        data = res["dev"] if split == "dev" else res["test"]
        for k in keep:
            for i, c in enumerate(res[key]):
                tr = data[k]["trs"][i]
                rows.append({"split": split, "cid": c["cid"], "bucket": c["bucket"], "role": c["role"], "lang": c["lang"], "voice": c["voice"], "expected": c["expected"], "said": c["text"], "asr": k,
                             "heard": (tr or {}).get("text"), "latency_s": (tr or {}).get("latency_s"), **{f"pred_{m}": data[k]["preds"][m][i] for m in METHODS}})
    abl_keys = {r["asr"] for r in rows}
    abl = {k: lexicon_ablation(rows, bank, k) for k in sorted(abl_keys)}
    for k, v in abl.items():
        md += ablation_md(v)
    metrics["lexicon_ablation"] = abl
    run.predictions(rows)  # labels and text only; the in-memory audio is never written
    return run.finish("completed", metrics, head, md)


if __name__ == "__main__":  # python -m homeease.experiments.audio_intent rescore <run_dir> [asr key]
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "rescore":
        print(rescore_run(sys.argv[2], *(sys.argv[3:4]))[1])
    else:
        print(__doc__)
