"""A6: cancel-eligibility of a spoken reply (voiceprint match + reassurance intent) and its FALSE-CANCEL rate.

Figure 3-3 draws "person responds and voiceprint matches -> auto-cancel false alarm". The approved policy keeps the proposal text:
a reply never changes an event. `ml/audio/cancel_gate.py` therefore only produces `cancel_eligible` EVIDENCE, and this experiment
measures how often that evidence would be wrong, i.e. how often an automatic cancel would have fired when it should not have.

Data: Piper en_US-libritts_r-medium renders the SAME words with hundreds of different synthetic voices (`ml/audio/multi_tts.py`);
the phrase bank (`config/audio_phrases.yaml`, step 3) supplies the words. Everything is SYNTHETIC and CLEAN speech: the numbers do not
describe real households or older adults.

Protocol (fixed here before any eval clip exists; the code enforces the order dev -> thresholds -> eval):
1. Speakers are split into disjoint DEV and EVAL sets. Words are split by paraphrase family (the phrase bank's dev / test split).
2. DEV (speaker layer only): every dev speaker enrols with three fixed sentences (embedding mean), then dev-split reassurance phrases
   are scored against every dev enrollment. `calibrate_threshold` picks the decision threshold at FAR 0.1% (proposal 10.1(3)) and at
   a looser FAR 1%. Only clips with at least `min_speech_s` of speech take part, as in the gate. Nothing else is tuned.
3. EVAL: for every eval speaker (the "household user") the full chain runs exactly like the inquiry service: energy VAD -> faster-whisper
   tiny -> reply intent rule v2 -> voiceprint gate, in these scenarios:
     true_cancel_clean / _noise5dB       the enrolled speaker says a reassurance (the case that SHOULD be eligible)
     other_person_same_words / _noise5dB another eval speaker says the same reassurance phrases ("a visitor answers I'm fine")
     visitor_on_behalf                   a visitor-role phrase ("she is fine") by another person
     tv_radio_speech                     tv-role reassurance lines through a loudspeaker simulation, other speaker
     replay_of_enrolled_voice            the enrolled user's OWN reassurance replayed through the loudspeaker simulation
     household_noise_only / silence      ESC-50 household sounds, near-silence (no speech)
     enrolled_says_help                  the enrolled speaker says a call for help (must never be eligible)
   A "trial" is one clip scored against one enrolled user (a clip from another person is scored against many users, which gives the
   impostor trials that FAR 0.1% needs). Counts, Wilson intervals (which treat trials as independent and are therefore too narrow) and
   identity-level bootstrap intervals (resampling speakers, not trials) are reported, together with the intent-only rate: what a bare
   spoken reply without any voiceprint would have cancelled.
4. A real-speech transfer check (LibriSpeech, read speech, speaker layer only) applies the TTS-calibrated thresholds to real voices.

Piper's sampling noise is not seedable, so a rerun gives slightly different clips. The per-clip table (labels, transcripts, scores; no
audio) is saved and `rescore` re-analyses it. Without Piper (package or model missing) the experiment falls back to the installed
SAPI voices, labels itself PARTIAL (a handful of speakers cannot estimate a false-accept rate) and reports no cancel_eligible rate.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from homeease.core.paths import data_dir, models_dir
from homeease.core.util import get_logger
from homeease.experiments.runner import Run, md_table
from homeease.ml.audio import cancel_gate as cg
from homeease.ml.audio.features import SR, load_audio, mix_at_snr, simulate_loudspeaker
from homeease.ml.audio.phrases import PhraseBank, load_phrase_bank
from homeease.ml.audio.speaker import eer as eer_fn
from homeease.ml.common import metrics as M

log = get_logger(__name__)

FARS = (0.001, 0.01)  # proposal 10.1(3) operating point and a looser one
OP_NAMES = {0.001: "far_0.1pct", 0.01: "far_1pct"}
OP_LABEL = {0.001: "FAR 0.1%", 0.01: "FAR 1%"}
PAD_S = 0.3
N_ENROLL = 3
# Fixed enrollment script (text-independent verification: the reply phrases are different words). 3-4 s of speech each.
ENROLL_TEXTS = (
    "The weather report says it will be warm and cloudy this afternoon.",
    "Please put the blue cup on the kitchen table before dinner.",
    "My name is on the list and I live on the third floor of this building.",
    "We usually have tea and toast at about seven in the morning.",
    "Today is a quiet day and the neighbours are away on holiday.",
)
SRC_ID = {"self_reas": 1, "help": 2, "visitor": 3, "tv": 4}  # fixed numbers for per-clip seeds (Python's str hash is randomised per process)
SOURCES = ("self_reas", "self_reas_n5", "help", "visitor", "tv", "replay", "noise", "silence")
# (name, source, kind): own = the clip is scored against the enrolled user who spoke it; others = against every other eval user; all = every user
SCENARIOS: tuple[tuple[str, str, str], ...] = (
    ("true_cancel_clean", "self_reas", "own"),
    ("true_cancel_noise5dB", "self_reas_n5", "own"),
    ("other_person_same_words", "self_reas", "others"),
    ("other_person_noise5dB", "self_reas_n5", "others"),
    ("visitor_on_behalf", "visitor", "others"),
    ("tv_radio_speech", "tv", "others"),
    ("replay_of_enrolled_voice", "replay", "own"),
    ("household_noise_only", "noise", "all"),
    ("silence", "silence", "all"),
    ("enrolled_says_help", "help", "own"),
)
SHOULD_CANCEL = {"true_cancel_clean", "true_cancel_noise5dB"}  # all other scenarios are FALSE-cancel scenarios (eligible there is wrong)
DURATION_EDGES = (0.0, 0.75, 1.0, 1.5, 2.0, 99.0)


# ================================================================================ plan
@dataclass(frozen=True)
class Plan:
    seed: int
    dev: tuple[int, ...]
    eval: tuple[int, ...]
    dev_probes: dict[int, tuple[str, ...]]  # speaker -> phrase ids (dev split, self / REASSURANCE)
    eval_phrases: dict[str, dict[int, tuple[str, ...]]]  # source -> speaker -> phrase ids (test split)
    n_noise_only: int
    n_silence: int

    def phrase_ids(self) -> tuple[set[str], set[str]]:
        dev = {p for v in self.dev_probes.values() for p in v}
        ev = {p for src in self.eval_phrases.values() for v in src.values() for p in v}
        return dev, ev


def make_plan(bank: PhraseBank, n_speakers_total: int, n_dev: int, n_eval: int, seed: int, probes_dev: int = 4, n_noise_only: int = 100, n_silence: int = 20) -> Plan:
    """Disjoint speaker split and per-speaker phrase draws. Dev speakers only ever see dev-split phrases, eval speakers only test-split phrases."""
    if n_dev + n_eval > n_speakers_total:
        raise ValueError(f"need {n_dev + n_eval} speakers, the engine has {n_speakers_total}")
    perm = np.random.default_rng(seed).permutation(n_speakers_total)
    dev = tuple(sorted(int(s) for s in perm[:n_dev]))
    ev = tuple(sorted(int(s) for s in perm[n_dev : n_dev + n_eval]))
    ids = lambda split, role, intent: sorted(e.id for e in bank.select(split=split, role=role, lang="en", intent=intent))  # noqa: E731
    pools = {"self_reas": ids("test", "self", "REASSURANCE"), "help": ids("test", "self", "HELP"), "visitor": ids("test", "visitor", "REASSURANCE"), "tv": ids("test", "tv", "REASSURANCE")}
    dev_pool = ids("dev", "self", "REASSURANCE")
    if not all(pools.values()) or not dev_pool:
        raise ValueError("phrase bank lacks English phrases for A6")

    def draw(pool: list[str], spk: int, tag: int, k: int) -> tuple[str, ...]:
        r = np.random.default_rng([seed, spk, tag])
        return tuple(pool[int(i)] for i in r.permutation(len(pool))[: min(k, len(pool))])

    dev_probes = {s: draw(dev_pool, s, 1, probes_dev) for s in dev}
    ev_ph: dict[str, dict[int, tuple[str, ...]]] = {k: {} for k in SOURCES}
    for s in ev:
        ev_ph["self_reas"][s] = draw(pools["self_reas"], s, 2, 4)
        ev_ph["self_reas_n5"][s] = ev_ph["self_reas"][s][:2]  # the same words with noise, so clean vs noisy is a paired comparison
        ev_ph["replay"][s] = ev_ph["self_reas"][s][:2]  # replay of the user's own earlier recording of these phrases
        ev_ph["help"][s] = draw(pools["help"], s, 3, 3)
        ev_ph["visitor"][s] = draw(pools["visitor"], s, 4, 2)
        ev_ph["tv"][s] = draw(pools["tv"], s, 5, 2)
    return Plan(seed, dev, ev, dev_probes, ev_ph, n_noise_only, n_silence)


# ================================================================================ product chain
def reply_chain(audio: np.ndarray, asr: Any, intent_model: Any) -> dict[str, Any]:
    """The steps of `AudioInquiryService._handle_reply` up to the intent: energy VAD -> ASR -> intent. NO_RESPONSE and a failing ASR are never 'safe'."""
    from homeease.ml.audio.vad import EnergyVAD, segments

    if not segments(EnergyVAD().frames(audio)):
        return {"status": "NO_RESPONSE", "intent": None, "intent_reason": None, "asr_conf": None, "transcript": ""}
    try:
        res = asr.transcribe(audio, None)
    except Exception as e:  # noqa: BLE001 - a failing ASR is a result, not a crash
        return {"status": "UNCERTAIN", "intent": "UNCERTAIN", "intent_reason": f"asr_failed:{type(e).__name__}", "asr_conf": None, "transcript": ""}
    it = intent_model.predict(res["text"], res.get("avg_logprob"))
    return {"status": "RESPONDED", "intent": it["intent"], "intent_reason": it["reason"], "asr_conf": res.get("avg_logprob"), "transcript": (res["text"] or "")[:160]}


def pad_clip(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = int(PAD_S * SR)
    return np.concatenate([rng.normal(0, 3e-4, n), x, rng.normal(0, 3e-4, n)]).astype(np.float32)


@dataclass
class Ctx:
    synth: Callable[[str, int], np.ndarray]
    embedder: Any
    asr: Any
    intent: Any
    noise_eval: list[np.ndarray]  # ESC-50 DAILY clips of the test folds (dev is clean speech only)
    min_speech_s: float = cg.DEFAULT_MIN_SPEECH_S
    min_asr_logprob: float = cg.DEFAULT_MIN_ASR_LOGPROB
    vad: Any = None
    texts: dict[str, str] = field(default_factory=dict)  # phrase id -> text


def embed_crop(embedder: Any, audio: np.ndarray, vad: Any = None, min_s: float = 0.3) -> tuple[np.ndarray | None, float]:
    """(L2-normalised embedding of the VAD speech, seconds of speech). The crop is zeroed after use; None when there is almost no speech."""
    crop, sp = cg.speech_crop(audio, vad)
    try:
        if sp < min_s or len(crop) == 0:
            return None, sp
        e = np.asarray(embedder.embed(crop), dtype=np.float64).reshape(-1)
        return (e / (np.linalg.norm(e) + 1e-12)).astype(np.float32), sp
    finally:
        crop[:] = 0.0


def enroll_speaker(ctx: Ctx, spk: int) -> cg.VoiceprintEnrollment | None:
    """Three fixed sentences -> enrollment. None when the enrollment is refused (counted by the caller)."""
    utts = [ctx.synth(t, spk) for t in ENROLL_TEXTS[:N_ENROLL]]
    try:
        return cg.VoiceprintEnrollment.from_audio(ctx.embedder, utts, ctx.vad)
    except cg.EnrollmentError as e:
        log.warning("enrollment of speaker %s refused: %s", spk, e)
        return None
    finally:
        for u in utts:
            u[:] = 0.0


# ================================================================================ dev
@dataclass
class DevTable:
    speakers: np.ndarray  # (D,) enrolled dev speakers
    probe_spk: np.ndarray  # (P,)
    probe_speech_s: np.ndarray  # (P,)
    S: np.ndarray  # (P, D) cosine of each probe embedding with each dev enrollment (NaN: no embedding)
    refused: int = 0


def collect_dev(plan: Plan, ctx: Ctx) -> DevTable:
    t0 = time.time()
    enr: dict[int, cg.VoiceprintEnrollment] = {}
    refused = 0
    for s in plan.dev:
        e = enroll_speaker(ctx, s)
        if e is None:
            refused += 1
        else:
            enr[s] = e
    spk = np.asarray(sorted(enr), dtype=int)
    mat = np.stack([enr[int(s)].embedding for s in spk]) if len(spk) else np.zeros((0, 1), dtype=np.float32)
    ps, psp, rows = [], [], []
    for s in spk:
        for k, pid in enumerate(plan.dev_probes[int(s)]):
            rng = np.random.default_rng([plan.seed, int(s), 100 + k])
            x = pad_clip(ctx.synth(ctx.texts[pid], int(s)), rng)
            e, sp = embed_crop(ctx.embedder, x, ctx.vad)
            x[:] = 0.0
            ps.append(int(s))
            psp.append(sp)
            rows.append(np.full(len(spk), np.nan, dtype=np.float32) if e is None else (mat @ e).astype(np.float32))
    log.info("A6 dev: %d speakers, %d probes in %.0f s", len(spk), len(ps), time.time() - t0)
    return DevTable(spk, np.asarray(ps, dtype=int), np.asarray(psp, dtype=float), np.stack(rows) if rows else np.zeros((0, len(spk)), dtype=np.float32), refused)


def dev_trials(d: DevTable, min_speech_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Flattened (scores, labels) of the dev probes that have enough speech; label 1 = the probe speaker is the enrolled one."""
    ok = d.probe_speech_s >= min_speech_s
    S = d.S[ok]
    lab = (d.probe_spk[ok][:, None] == d.speakers[None, :]).astype(int)
    keep = np.isfinite(S)
    return S[keep].astype(float), lab[keep]


# ================================================================================ eval
@dataclass
class EvalTable:
    users: np.ndarray  # (U,) enrolled eval speakers
    rows: list[dict[str, Any]]  # one per clip (no audio)
    S: np.ndarray  # (C, U) cosine to each enrollment (NaN: no embedding)
    refused: int = 0
    audit: dict[str, Any] = field(default_factory=dict)


def collect_eval(plan: Plan, ctx: Ctx, thresholds: dict[float, float | None], audit_share: float = 0.08) -> EvalTable:
    """Run the product chain on every eval clip. `thresholds` (from dev) only feed the consistency audit of the vectorised analysis."""
    t0 = time.time()
    enr: dict[int, cg.VoiceprintEnrollment] = {}
    refused = 0
    for s in plan.eval:
        e = enroll_speaker(ctx, s)
        if e is None:
            refused += 1
        else:
            enr[s] = e
    users = np.asarray(sorted(enr), dtype=int)
    uidx = {int(s): i for i, s in enumerate(users)}
    mat = np.stack([enr[int(s)].embedding for s in users]) if len(users) else np.zeros((0, 1), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    Srows: list[np.ndarray] = []
    rng_audit = np.random.default_rng(plan.seed + 7)
    audit = {"n": 0, "mismatch": 0, "threshold": thresholds.get(0.001)}
    thr_audit = thresholds.get(0.001)
    gate = cg.CancelGate(thr_audit, ctx.min_speech_s, ctx.min_asr_logprob, embedder=ctx.embedder, vad=ctx.vad) if thr_audit is not None else None

    def add(source: str, voice: int, pid: str | None, text: str, audio: np.ndarray, kind: str, group: str) -> None:
        ch = reply_chain(audio, ctx.asr, ctx.intent)
        e, sp = embed_crop(ctx.embedder, audio, ctx.vad)
        row_s = np.full(len(users), np.nan, dtype=np.float32) if e is None else (mat @ e).astype(np.float32)
        cid = f"{source}-{len(rows):05d}"
        rows.append({"cid": cid, "source": source, "voice": voice, "phrase_id": pid, "said": text, "group": group, **ch, "speech_s": round(sp, 3), "embedded": e is not None})
        Srows.append(row_s)
        if gate is not None and len(users) > 1 and rng_audit.random() < audit_share:  # the product gate on this very clip versus the vectorised analysis
            u = int(users[int(rng_audit.integers(len(users)))]) if kind != "own" else voice
            if kind == "others" and u == voice:
                u = int(users[(uidx[u] + 1) % len(users)])
            intent_res = None if ch["intent"] is None else {"intent": ch["intent"], "reason": ch["intent_reason"]}
            dec = gate.evaluate(audio, intent_res, enr[u], ch["asr_conf"])
            base = ch["intent"] == "REASSURANCE" and sp >= ctx.min_speech_s and ch["asr_conf"] is not None and ch["asr_conf"] >= ctx.min_asr_logprob
            vec = bool(base and np.isfinite(row_s[uidx[u]]) and row_s[uidx[u]] >= thr_audit)
            audit["n"] += 1
            audit["mismatch"] += int(dec.cancel_eligible != vec)

    for n, s in enumerate(users):
        s = int(s)
        raws: dict[str, list[tuple[str, np.ndarray]]] = {k: [] for k in ("self_reas", "help", "visitor", "tv")}
        for src in ("self_reas", "help", "visitor", "tv"):
            for k, pid in enumerate(plan.eval_phrases[src].get(s, ())):
                raws[src].append((pid, ctx.synth(ctx.texts[pid], s)))
        for src in ("self_reas", "help", "visitor", "tv"):
            for k, (pid, raw) in enumerate(raws[src]):
                rng = np.random.default_rng([plan.seed, s, 200 + k, SRC_ID[src]])
                x = simulate_loudspeaker(raw, rng=rng) if src == "tv" else raw
                clip = pad_clip(x, rng)
                add(src, s, pid, ctx.texts[pid], clip, "own" if src in ("self_reas", "help") else "others", f"spk-{s}")
                clip[:] = 0.0
                if src == "self_reas" and pid in plan.eval_phrases["self_reas_n5"][s]:
                    noisy = mix_at_snr(pad_clip(raw, rng), ctx.noise_eval[int(rng.integers(len(ctx.noise_eval)))], 5.0, rng) if ctx.noise_eval else None
                    if noisy is not None:
                        add("self_reas_n5", s, pid, ctx.texts[pid], noisy, "own", f"spk-{s}")
                        noisy[:] = 0.0
                if src == "self_reas" and pid in plan.eval_phrases["replay"][s]:
                    rep = pad_clip(simulate_loudspeaker(raw, rng=rng), rng)
                    add("replay", s, pid, ctx.texts[pid], rep, "own", f"spk-{s}")
                    rep[:] = 0.0
        for v in raws.values():
            for _, raw in v:
                raw[:] = 0.0
        if (n + 1) % 25 == 0:
            log.info("A6 eval: %d/%d speakers, %d clips, %.0f s", n + 1, len(users), len(rows), time.time() - t0)
    rng = np.random.default_rng(plan.seed + 11)
    for k in range(plan.n_noise_only):
        if not ctx.noise_eval:
            break
        x = pad_clip(ctx.noise_eval[k % len(ctx.noise_eval)][: 5 * SR], rng)
        add("noise", -1, None, "(household sound, no speech)", x, "all", f"noise-{k}")
        x[:] = 0.0
    levels = (2e-4, 5e-4, 1.5e-3, 3e-3)
    for k in range(plan.n_silence):
        x = rng.normal(0, levels[k % len(levels)], 5 * SR).astype(np.float32)
        add("silence", -1, None, "(silence)", x, "all", f"silence-{k}")
    log.info("A6 eval: %d clips in %.0f s; gate audit %d clips, %d mismatches", len(rows), time.time() - t0, audit["n"], audit["mismatch"])
    return EvalTable(users, rows, np.stack(Srows) if Srows else np.zeros((0, len(users)), dtype=np.float32), refused, audit)


# ================================================================================ statistics
def rate_ci(k: int, n: int) -> dict[str, Any]:
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "wilson95": M.wilson(int(k), int(n)) if n else None}


def identity_bootstrap(K: np.ndarray, N: np.ndarray, paired: bool, n_boot: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    """95% interval of sum(K)/sum(N) when whole IDENTITIES are resampled (speakers, not trials). K, N: (row groups, users) counts of eligible / all trials.
    paired: rows and columns are the same identities (a user's own clips; only the diagonal is filled), resampled together;
    otherwise rows (speakers of the clips, or single clips for noise) and columns (enrolled users) are resampled independently."""
    R, C = K.shape
    if R < 3 or C < 3:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    kd, nd = (np.diag(K), np.diag(N)) if paired and R == C else (None, None)
    for _ in range(n_boot):
        wc = rng.multinomial(C, np.full(C, 1.0 / C)).astype(float)
        if kd is not None:
            num, den = float(wc @ kd), float(wc @ nd)
        else:
            wr = rng.multinomial(R, np.full(R, 1.0 / R)).astype(float)
            num, den = float(wr @ K @ wc), float(wr @ N @ wc)
        if den > 0:
            vals.append(num / den)
    if len(vals) < n_boot * 0.5:
        return None
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def trial_mask(kind: str, voices: np.ndarray, users: np.ndarray) -> np.ndarray:
    """(clips, users) boolean: which clip-user pairs are trials of this scenario kind."""
    if kind == "own":
        return voices[:, None] == users[None, :]
    if kind == "others":
        return (voices[:, None] != users[None, :]) & (voices[:, None] >= 0)
    return np.ones((len(voices), len(users)), dtype=bool)


def base_flags(rows: Sequence[dict[str, Any]], min_speech_s: float, min_asr_logprob: float) -> dict[str, np.ndarray]:
    """Per-clip parts of the gate that do not depend on who the enrolled user is."""
    intent_ok = np.array([r["intent"] == "REASSURANCE" for r in rows], dtype=bool)
    speech_ok = np.array([r["speech_s"] >= min_speech_s for r in rows], dtype=bool)
    asr_ok = np.array([r["asr_conf"] is not None and r["asr_conf"] >= min_asr_logprob for r in rows], dtype=bool)
    return {"intent": intent_ok, "speech": speech_ok, "asr": asr_ok, "all_but_voiceprint": intent_ok & speech_ok & asr_ok}


def scenario_stats(ev: EvalTable, flags: dict[str, np.ndarray], thresholds: dict[float, float | None], seed: int) -> dict[str, Any]:
    rows = ev.rows
    voices = np.array([r["voice"] for r in rows], dtype=int)
    src = np.array([r["source"] for r in rows])
    out: dict[str, Any] = {}
    for name, source, kind in SCENARIOS:
        sel = np.where(src == source)[0]
        d: dict[str, Any] = {"source": source, "kind": kind, "should_cancel": name in SHOULD_CANCEL, "clips": int(len(sel))}
        if len(sel) == 0:
            out[name] = {**d, "status": "no_clips"}
            continue
        mask = trial_mask(kind, voices[sel], ev.users)
        n_trials = int(mask.sum())
        d["trials"] = n_trials
        d["intent_only"] = rate_ci(int((flags["intent"][sel][:, None] & mask).sum()), n_trials)
        d["all_but_voiceprint"] = rate_ci(int((flags["all_but_voiceprint"][sel][:, None] & mask).sum()), n_trials)
        S = ev.S[sel]
        row_ids = voices[sel] if kind != "all" else np.arange(len(sel))
        uniq, inv = np.unique(row_ids, return_inverse=True)
        ind = np.zeros((len(sel), len(uniq)), dtype=float)
        ind[np.arange(len(sel)), inv] = 1.0
        d["operating_points"] = {}
        for far, thr in thresholds.items():
            if thr is None:
                d["operating_points"][OP_NAMES[far]] = {"threshold": None, "status": "no_threshold"}
                continue
            with np.errstate(invalid="ignore"):  # NaN = no embedding: never above a threshold
                vp = np.isfinite(S) & (S >= thr)
            elig = flags["all_but_voiceprint"][sel][:, None] & vp & mask
            voice_only = vp & mask & flags["speech"][sel][:, None]
            if kind == "own":  # one identity per user: k / n of that user's own clips
                K, Nn = np.diag(elig.sum(axis=0).astype(float)), np.diag(mask.sum(axis=0).astype(float))
            else:
                K, Nn = ind.T @ elig.astype(float), ind.T @ mask.astype(float)
            op = rate_ci(int(elig.sum()), n_trials)
            op["identity_bootstrap95"] = identity_bootstrap(K, Nn, paired=(kind == "own"), seed=seed)
            op["voiceprint_only"] = rate_ci(int(voice_only.sum()), n_trials)  # speaker score above the threshold on clips with enough speech, whatever was said
            op["threshold"] = thr
            d["operating_points"][OP_NAMES[far]] = op
        out[name] = d
    return out


def reason_breakdown(ev: EvalTable, thresholds: dict[float, float | None], min_speech_s: float, min_asr: float, per_scenario: int = 3000, seed: int = 0) -> dict[str, Any]:
    """Which reason code stops the evidence, using the gate's own `decide` on a sample of trials per scenario (strict operating point)."""
    thr = thresholds.get(0.001)
    voices = np.array([r["voice"] for r in ev.rows], dtype=int)
    src = np.array([r["source"] for r in ev.rows])
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for name, source, kind in SCENARIOS:
        sel = np.where(src == source)[0]
        if len(sel) == 0:
            continue
        pairs = np.argwhere(trial_mask(kind, voices[sel], ev.users))
        if len(pairs) > per_scenario:
            pairs = pairs[rng.choice(len(pairs), per_scenario, replace=False)]
        cnt: Counter[str] = Counter()
        elig = 0
        for i, j in pairs:
            r = ev.rows[sel[i]]
            sc = ev.S[sel[i], j]
            dec = cg.decide(enrolled=True, threshold=thr, speech_s=r["speech_s"], min_speech_s=min_speech_s, intent=r["intent"] if r["status"] == "RESPONDED" else None,
                            asr_conf=r["asr_conf"], min_asr_logprob=min_asr, score=float(sc) if np.isfinite(sc) else None)
            elig += int(dec.cancel_eligible)
            for c in dec.reasons:
                cnt[c] += 1
        out[name] = {"sampled_trials": int(len(pairs)), "eligible": elig, "reason_counts": dict(cnt)}
    return out


def duration_buckets(scores: np.ndarray, labels: np.ndarray, speech_s: np.ndarray, edges: Sequence[float] = DURATION_EDGES) -> list[dict[str, Any]]:
    """Speaker-layer EER per length of the probe's speech (arrays per TRIAL). EER is reported only with at least 30 genuine and 300 impostor trials."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (speech_s >= lo) & (speech_s < hi)
        g, i = int((labels[m] == 1).sum()), int((labels[m] == 0).sum())
        e = eer_fn(scores[m], labels[m]) if g >= 30 and i >= 300 else {"eer": None}
        out.append({"speech_s": f"{lo:g}-{hi:g}" if hi < 90 else f">={lo:g}", "genuine": g, "impostor": i, "eer": e["eer"]})
    return out


def score_summary(x: np.ndarray) -> dict[str, Any]:
    if len(x) == 0:
        return {"n": 0}
    p = np.percentile(x, [1, 50, 99, 99.9])
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()), "p1": float(p[0]), "median": float(p[1]), "p99": float(p[2]), "p99.9": float(p[3])}


def speaker_layer_eval(ev: EvalTable, flags: dict[str, np.ndarray], thresholds: dict[float, float | None], seed: int) -> dict[str, Any]:
    rows = ev.rows
    src = np.array([r["source"] for r in rows])
    voices = np.array([r["voice"] for r in rows], dtype=int)
    sp = np.array([r["speech_s"] for r in rows], dtype=float)
    sel = np.where(src == "self_reas")[0]
    S = ev.S[sel]
    own = voices[sel][:, None] == ev.users[None, :]
    fin = np.isfinite(S)
    scores, labels = S[fin].astype(float), own[fin].astype(int)
    spt = np.broadcast_to(sp[sel][:, None], S.shape)[fin]
    out: dict[str, Any] = {"all_lengths": {"genuine": int((labels == 1).sum()), "impostor": int((labels == 0).sum()), "eer": eer_fn(scores, labels)["eer"] if len(scores) else None},
                           "by_duration": duration_buckets(scores, labels, spt)}
    long = flags["speech"][sel][:, None] & fin
    s2, y2 = S[long].astype(float), own[long].astype(int)
    out["with_enough_speech"] = {"genuine": int((y2 == 1).sum()), "impostor": int((y2 == 0).sum()), "eer": eer_fn(s2, y2)["eer"] if len(s2) else None,
                                 "genuine_scores": score_summary(s2[y2 == 1]), "impostor_scores": score_summary(s2[y2 == 0])}
    out["at_dev_thresholds"] = {}
    for far, thr in thresholds.items():
        if thr is None:
            continue
        gen, imp = s2[y2 == 1], s2[y2 == 0]
        out["at_dev_thresholds"][OP_NAMES[far]] = {"threshold": thr, "tar": rate_ci(int((gen >= thr).sum()), len(gen)), "far": rate_ci(int((imp >= thr).sum()), len(imp))}
    return out


# ================================================================================ real speech transfer check
def libri_speakers() -> dict[str, list[str]]:
    root = data_dir() / "raw" / "librispeech_mini"
    spk: dict[str, list[str]] = {}
    for p in sorted(root.rglob("*.flac")):
        spk.setdefault(p.parts[-3], []).append(str(p))
    return spk


def real_speech_check(embedder: Any, thresholds: dict[float, float | None], seed: int, min_speech_s: float, loader: Callable[[str], np.ndarray] = load_audio,
                      speakers: dict[str, list[str]] | None = None, probes_per_speaker: int = 8) -> dict[str, Any]:
    """Speaker layer only: do the TTS-calibrated thresholds work on REAL read speech (LibriSpeech)? Enrollment = 3 utterances (first 4 s of speech),
    probes = 2.5 s windows of other utterances. Reports TAR / FAR with Wilson intervals at the thresholds chosen on TTS dev speakers; nothing is tuned on it."""
    spk = speakers if speakers is not None else libri_speakers()
    rng = np.random.default_rng(seed)
    names = sorted(s for s, u in spk.items() if len(u) >= N_ENROLL + 2)
    if len(names) < 4:
        return {"status": "data_pending", "missing": ["LibriSpeech dev-clean-2 (homeease data download --dataset librispeech)"]}
    enr, probes = {}, []
    for s in names:
        utts = list(rng.permutation(spk[s]))
        embs = []
        for p in utts[:N_ENROLL]:
            crop, sp = cg.speech_crop(loader(p))
            x = crop[: 4 * SR].copy()
            crop[:] = 0.0
            e = embedder.embed(x)
            x[:] = 0.0
            embs.append(np.asarray(e, dtype=np.float32))
        try:
            enr[s] = cg.VoiceprintEnrollment.from_embeddings(embs, getattr(embedder, "model_id", "?"))
        except cg.EnrollmentError:
            continue
        for p in utts[N_ENROLL : N_ENROLL + probes_per_speaker]:
            crop, sp = cg.speech_crop(loader(p))
            seg = int(2.5 * SR)
            off = int(rng.integers(0, max(1, len(crop) - seg + 1)))
            x = crop[off : off + seg].copy()
            crop[:] = 0.0
            if len(x) < min_speech_s * SR:
                continue
            e = np.asarray(embedder.embed(x), dtype=np.float64)
            x[:] = 0.0
            probes.append((s, e / (np.linalg.norm(e) + 1e-12)))
    ids = sorted(enr)
    mat = np.stack([enr[s].embedding for s in ids]).astype(np.float64)
    scores, labels = [], []
    for s, e in probes:
        sc = mat @ e
        scores.extend(sc.tolist())
        labels.extend([1 if t == s else 0 for t in ids])
    scores_a, labels_a = np.asarray(scores), np.asarray(labels)
    gen, imp = scores_a[labels_a == 1], scores_a[labels_a == 0]
    out: dict[str, Any] = {"status": "completed", "speakers": len(ids), "probes": len(probes), "genuine_trials": int(len(gen)), "impostor_trials": int(len(imp)),
                           "eer": eer_fn(scores_a, labels_a)["eer"], "genuine_scores": score_summary(gen), "impostor_scores": score_summary(imp), "at_tts_dev_thresholds": {}}
    for far, thr in thresholds.items():
        if thr is None:
            continue
        out["at_tts_dev_thresholds"][OP_NAMES[far]] = {"threshold": thr, "tar": rate_ci(int((gen >= thr).sum()), len(gen)), "far": rate_ci(int((imp >= thr).sum()), len(imp))}
    return out


# ================================================================================ analysis
def calibrate_dev(dev: DevTable, min_speech_s: float) -> dict[str, Any]:
    scores, labels = dev_trials(dev, min_speech_s)
    cals = {OP_NAMES[f]: cg.calibrate_threshold(scores, labels, f).to_dict() if len(scores) and labels.min() == 0 and labels.max() == 1 else None for f in FARS}
    sp_t = np.broadcast_to(dev.probe_speech_s[:, None], dev.S.shape)
    fin = np.isfinite(dev.S)
    all_lab = (dev.probe_spk[:, None] == dev.speakers[None, :]).astype(int)
    return {"min_speech_s": min_speech_s, "n_speakers": int(len(dev.speakers)), "n_probes": int(len(dev.probe_spk)), "probes_with_enough_speech": int((dev.probe_speech_s >= min_speech_s).sum()),
            "calibrations": cals, "eer_dev": eer_fn(scores, labels)["eer"] if len(scores) and labels.min() == 0 and labels.max() == 1 else None,
            "by_duration": duration_buckets(dev.S[fin].astype(float), all_lab[fin], sp_t[fin]) if fin.any() else [], "refused_enrollments": dev.refused}


def thresholds_from(dev_cal: dict[str, Any]) -> dict[float, float | None]:
    return {f: (dev_cal["calibrations"][OP_NAMES[f]] or {}).get("threshold") for f in FARS}


def analyse(dev: DevTable, ev: EvalTable, min_speech_s: float, min_asr_logprob: float, seed: int, partial: bool = False) -> dict[str, Any]:
    if partial or len(dev.speakers) == 0:
        dev_cal = {"min_speech_s": min_speech_s, "n_speakers": int(len(dev.speakers)), "calibrations": {OP_NAMES[f]: None for f in FARS}, "note": "no dev speakers: no threshold can be calibrated"}
    else:
        dev_cal = calibrate_dev(dev, min_speech_s)
    thr = thresholds_from(dev_cal)
    flags = base_flags(ev.rows, min_speech_s, min_asr_logprob)
    return {
        "partial": bool(partial), "dev": dev_cal, "thresholds": {OP_NAMES[f]: t for f, t in thr.items()}, "eval_users": int(len(ev.users)), "eval_clips": int(len(ev.rows)),
        "refused_enrollments": {"dev": dev.refused, "eval": ev.refused}, "speaker_layer": speaker_layer_eval(ev, flags, thr, seed) if len(ev.users) > 1 else {},
        "scenarios": scenario_stats(ev, flags, thr, seed), "reasons": reason_breakdown(ev, thr, min_speech_s, min_asr_logprob, seed=seed), "gate_audit": ev.audit,
        "reliability": reliability(dev, ev, thr, min_speech_s),
    }


def reliability(dev: DevTable, ev: EvalTable, thr: dict[float, float | None], min_speech_s: float) -> dict[str, Any]:
    _, lab = dev_trials(dev, min_speech_s) if len(dev.speakers) else (np.zeros(0), np.zeros(0, dtype=int))
    n_imp = int((lab == 0).sum())
    out = {"dev_impostor_trials": n_imp, "dev_speakers": int(len(dev.speakers)), "eval_speakers": int(len(ev.users))}
    for f in FARS:
        out[OP_NAMES[f]] = {"needs_impostor_trials": int(10 / f), "have_dev_impostor_trials": n_imp, "reliable": n_imp >= 10 / f, "min_measurable_rate": (1.0 / n_imp) if n_imp else None}
    out["note"] = ("A false-accept rate of 0.1% needs at least 10,000 impostor trials to mean anything. Trials here are pairs of clips and enrolled users built from a limited set of "
                   "speakers, so they are not independent: the identity-bootstrap intervals (which resample speakers) are the more honest ones, and they are wider than the Wilson intervals.")
    return out


# ================================================================================ report
def pct(r: dict[str, Any] | None, ci: bool = True) -> str:
    if not r or r.get("rate") is None:
        return "n/a"
    s = f"{100 * r['rate']:.2f}% ({r['k']}/{r['n']})"
    if ci and r.get("wilson95"):
        s += f" [{100 * r['wilson95'][0]:.2f}, {100 * r['wilson95'][1]:.2f}]"
    return s


def boot(r: dict[str, Any] | None) -> str:
    b = (r or {}).get("identity_bootstrap95")
    if r and r.get("k") == 0:
        return "0 events (interval degenerate; use the Wilson upper bound)"
    return "n/a" if not b else f"[{100 * b[0]:.2f}, {100 * b[1]:.2f}]"


def render(m: dict[str, Any], info: dict[str, Any]) -> tuple[str, str]:
    """(headline, markdown) from the metrics dict of `analyse` plus run info."""
    sc = m["scenarios"]
    thr = m["thresholds"]
    partial = m["partial"]
    md = ""
    if partial:
        md += ("> **PARTIAL RUN.** Piper (904 speakers) was not available, so only a handful of installed voices were used. That is far too few speakers to estimate a false-accept rate "
               "or to calibrate a threshold: NO cancel_eligible rate is reported, only intent-only rates and raw speaker scores.\n\n")
    md += (f"Engine: {info['engine'].get('engine')} ({info['engine'].get('n_speakers')} speakers); dev speakers {m['dev'].get('n_speakers')}, eval speakers {m['eval_users']}, eval clips {m['eval_clips']}; "
           f"ASR faster-whisper tiny, intent rule v2, speaker model {info['speaker_model']}; min speech {info['min_speech_s']} s, min ASR avg_logprob {info['min_asr_logprob']}.\n")
    md += "\nAll voices are synthetic and clean (read-style TTS); no older adults, no real household acoustics. Rates are rates of an EVIDENCE flag that is never acted on.\n"
    if not partial:
        md += "\n## Thresholds (calibrated on dev speakers only)\n\n"
        rows = []
        for f in FARS:
            c = m["dev"]["calibrations"].get(OP_NAMES[f]) or {}
            rows.append({"operating point": OP_LABEL[f], "threshold": c.get("threshold"), "dev FAR achieved": c.get("achieved_far"), "dev TAR": c.get("tar"), "dev impostor trials": c.get("n_impostor"),
                         "dev genuine trials": c.get("n_genuine"), "reliable": c.get("reliable")})
        md += md_table(rows, ["operating point", "threshold", "dev FAR achieved", "dev TAR", "dev impostor trials", "dev genuine trials", "reliable"])
        md += f"\nDev EER (speaker layer, clips with enough speech): {m['dev'].get('eer_dev')}. {m['reliability']['note']}\n"
        md += "\nSpeaker-layer EER by length of speech, dev speakers (justifies the minimum speech length; EER shown only with >= 30 genuine and >= 300 impostor trials):\n\n"
        md += md_table(m["dev"]["by_duration"], ["speech_s", "genuine", "impostor", "eer"])
    sl = m.get("speaker_layer") or {}
    if sl:
        md += "\n## Speaker layer on the eval speakers (disjoint from dev)\n\n"
        w = sl["with_enough_speech"]
        md += (f"EER {w['eer']} on {w['genuine']} genuine / {w['impostor']} impostor trials (clips with at least the minimum speech); genuine scores mean {w['genuine_scores'].get('mean', float('nan')):.3f}, "
               f"impostor scores mean {w['impostor_scores'].get('mean', float('nan')):.3f}, impostor 99.9th percentile {w['impostor_scores'].get('p99.9', float('nan')):.3f}.\n\n")
        md += md_table(sl["by_duration"], ["speech_s", "genuine", "impostor", "eer"])
        if sl["at_dev_thresholds"]:
            md += "\nAt the dev thresholds:\n\n" + md_table([{"operating point": k, "threshold": v["threshold"], "TAR": pct(v["tar"]), "FAR": pct(v["far"])} for k, v in sl["at_dev_thresholds"].items()], ["operating point", "threshold", "TAR", "FAR"])
    md += "\n## Scenarios\n\nTrial = one clip scored against one enrolled user. 'intent only' = what a bare spoken reply without voiceprint would have cancelled (intent == REASSURANCE). "
    md += "'eligible' = all gate conditions (speech length, reassurance intent, ASR confidence, voiceprint). Intervals in brackets are Wilson 95% (trials treated as independent: too narrow); 'speaker-bootstrap' resamples speakers.\n\n"
    for f in FARS:
        key = OP_NAMES[f]
        if partial or thr.get(key) is None:
            continue
        md += f"\n### Operating point {OP_LABEL[f]} (threshold {thr[key]:.3f})\n\n"
        rows = []
        for name, source, kind in SCENARIOS:
            d = sc.get(name, {})
            op = (d.get("operating_points") or {}).get(key) or {}
            rows.append({"scenario": name, "should cancel": "yes" if d.get("should_cancel") else "no (false cancel)", "clips": d.get("clips"), "trials": d.get("trials"), "intent only": pct(d.get("intent_only")),
                         "eligible": pct(op), "speaker-bootstrap 95%": boot(op), "voiceprint alone passes": pct(op.get("voiceprint_only"), ci=False)})
        md += md_table(rows, ["scenario", "should cancel", "clips", "trials", "intent only", "eligible", "speaker-bootstrap 95%", "voiceprint alone passes"])
    if partial:
        md += md_table([{"scenario": n, "clips": sc[n].get("clips"), "trials": sc[n].get("trials"), "intent only": pct(sc[n].get("intent_only")), "eligible": "not evaluated (no calibrated threshold)"} for n, _, _ in SCENARIOS if n in sc],
                       ["scenario", "clips", "trials", "intent only", "eligible"])
    md += "\n## Why trials failed (gate reason codes, strict operating point, up to 3000 sampled trials per scenario)\n\n"
    rr = []
    for name, _, _ in SCENARIOS:
        d = m["reasons"].get(name)
        if d:
            rr.append({"scenario": name, "sampled": d["sampled_trials"], "eligible": d["eligible"], **{c: d["reason_counts"].get(c, 0) for c in ("too_short", "intent_unavailable", "intent_not_reassurance", "low_asr_confidence", "speaker_mismatch", "no_threshold")}})
    md += md_table(rr, ["scenario", "sampled", "eligible", "too_short", "intent_unavailable", "intent_not_reassurance", "low_asr_confidence", "speaker_mismatch", "no_threshold"])
    md += "\nA trial can fail for several reasons at once; intent_unavailable = no speech was found (NO_RESPONSE) or the ASR failed.\n"
    a = m["gate_audit"]
    md += f"\nConsistency audit: the product gate (`CancelGate.evaluate` on the audio) was run on {a.get('n', 0)} random trials and disagreed with the vectorised analysis in {a.get('mismatch', 0)} of them.\n"
    rs = info.get("real_speech")
    if rs:
        md += "\n## Real-speech transfer check (LibriSpeech, read English, speaker layer only)\n\n"
        if rs.get("status") != "completed":
            md += f"status: **{rs.get('status')}**; missing: {rs.get('missing')}\n"
        else:
            md += (f"{rs['speakers']} real speakers, {rs['probes']} probes of 2.5 s; {rs['genuine_trials']} genuine / {rs['impostor_trials']} impostor trials; EER {rs['eer']:.4f}. "
                   "The thresholds below were chosen on TTS dev speakers and never tuned here.\n\n")
            md += md_table([{"operating point": k, "threshold": v["threshold"], "TAR": pct(v["tar"]), "FAR": pct(v["far"])} for k, v in rs["at_tts_dev_thresholds"].items()], ["operating point", "threshold", "TAR", "FAR"])
            md += f"\nReal impostor scores: mean {rs['impostor_scores'].get('mean', float('nan')):.3f}, 99th percentile {rs['impostor_scores'].get('p99', float('nan')):.3f}; genuine mean {rs['genuine_scores'].get('mean', float('nan')):.3f}.\n"
    md += "\n## Reading these numbers\n\n"
    md += ("* Cancel-eligible is EVIDENCE: no code path lets it change an event (A5 and tests/test_cancel_policy.py check that an EMERGENCY stays EMERGENCY/OPEN whatever it says).\n"
           "* The voiceprint cannot tell a live reply from a replay of the same voice: replay_of_enrolled_voice measures how much the loudspeaker channel alone changes the score.\n"
           "* Synthetic voices from one engine are more separable (or less) than real people in a real room; the real-speech check above is the only real-voice evidence and it is read, clean English.\n"
           "* English only. Mandarin is not evaluated: no multi-speaker Mandarin voice is available within the download budget (one SAPI voice, Huihui).\n")
    c = sc.get("true_cancel_clean", {}).get("operating_points", {}).get("far_0.1pct")
    f_oth = sc.get("other_person_same_words", {}).get("operating_points", {}).get("far_0.1pct")
    f_tv = sc.get("tv_radio_speech", {}).get("operating_points", {}).get("far_0.1pct")
    f_rep = sc.get("replay_of_enrolled_voice", {}).get("operating_points", {}).get("far_0.1pct")
    f_help = sc.get("enrolled_says_help", {}).get("operating_points", {}).get("far_0.1pct")
    if partial or not c:
        head = ("PARTIAL: too few speakers for a false-accept estimate; intent-only rates: " + "; ".join(f"{n} {pct(sc[n].get('intent_only'), ci=False)}" for n in ("other_person_same_words", "tv_radio_speech") if n in sc))
    else:
        io_o = sc["other_person_same_words"]["intent_only"]
        head = (f"At speaker threshold FAR 0.1% (calibrated on {m['dev']['n_speakers']} dev speakers): true-cancel {pct(c, ci=False)}; other person saying the same words eligible {pct(f_oth, ci=False)} "
                f"(intent-only {pct(io_o, ci=False)}); TV/radio {pct(f_tv, ci=False)}; replay of the enrolled voice {pct(f_rep, ci=False)}; enrolled speaker saying HELP {pct(f_help, ci=False)}. "
                f"{m['eval_users']} eval speakers, synthetic clean speech.")
    return head, md


# ================================================================================ runner
def load_texts(bank: PhraseBank) -> dict[str, str]:
    return {e.id: e.text for e in bank.entries}


def _save_tables(run: Run, dev: DevTable, ev: EvalTable) -> None:
    """Per-clip rows (labels, transcripts, numbers) and score matrices. No audio and no embeddings."""
    run.predictions(ev.rows)  # predictions.jsonl = the clip table
    np.savez_compressed(run.dir / "scores.npz", dev_speakers=dev.speakers, dev_probe_spk=dev.probe_spk, dev_probe_speech_s=dev.probe_speech_s, dev_S=dev.S, eval_users=ev.users, eval_S=ev.S,
                        dev_refused=dev.refused, eval_refused=ev.refused, audit_json=np.array(json.dumps(ev.audit)))


def load_tables(run_dir: str | Path) -> tuple[DevTable, EvalTable]:
    d = Path(run_dir)
    z = np.load(d / "scores.npz")
    rows = [json.loads(line) for line in (d / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    audit: dict[str, Any] = json.loads(str(z["audit_json"])) if "audit_json" in z.files else {}
    if not audit and (d / "metrics.json").is_file():  # runs saved before the audit was stored in the tables keep it in metrics.json
        audit = json.loads((d / "metrics.json").read_text(encoding="utf-8")).get("metrics", {}).get("gate_audit", {})
    return (DevTable(z["dev_speakers"], z["dev_probe_spk"], z["dev_probe_speech_s"], z["dev_S"], int(z["dev_refused"])), EvalTable(z["eval_users"], rows, z["eval_S"], int(z["eval_refused"]), audit))


def rescore_run(run_dir: str | Path) -> tuple[dict[str, Any], str]:
    """Re-analyse a saved run from its clip table (no synthesis, no ASR). Uses the settings stored in the run's config.yaml."""
    d = Path(run_dir)
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    dev, ev = load_tables(d)
    m = analyse(dev, ev, float(cfg["min_speech_s"]), float(cfg["min_asr_logprob"]), int(cfg["seed"]), partial=bool(cfg.get("partial", False)))
    real = None  # the real-speech check is not part of the saved tables; reuse the result of the original run when its metrics.json is there
    if (d / "metrics.json").is_file():
        real = json.loads((d / "metrics.json").read_text(encoding="utf-8")).get("metrics", {}).get("real_speech_check")
        if real is not None:
            m["real_speech_check"] = real
    head, md = render(m, {"engine": cfg.get("engine", {}), "speaker_model": cfg.get("speaker_model"), "min_speech_s": cfg["min_speech_s"], "min_asr_logprob": cfg["min_asr_logprob"], "real_speech": real})
    return m, head + "\n\n" + md


def sapi_speakers() -> tuple[Callable[[str, int], np.ndarray], list[str]] | None:
    """Fallback: the installed SAPI voices as 'speakers' (3 on the target machine)."""
    from homeease.ml.audio import tts as tts_mod

    try:
        voices = [v for v in tts_mod.available_voices() if v.lang in ("en", "zh")]
    except Exception:  # noqa: BLE001
        return None
    if not voices:
        return None
    names = [v.name for v in voices]
    return (lambda text, i: tts_mod.synthesize(text, names[i], voices[i].lang)), names


def exp_cancel_eligibility(seed: int = 42, n_dev: int = 150, n_eval: int = 150, probes_dev: int = 4, n_noise_only: int = 100, n_silence: int = 20, with_real_check: bool = True) -> dict[str, Any]:
    """A6. `n_dev` / `n_eval` are speaker counts (the defaults give ~90,000 impostor trials per clip set); smaller values are for smoke tests."""
    from homeease.ml.audio import multi_tts
    from homeease.ml.audio.intent_v2 import RuleIntentV2
    from homeease.experiments.audio_intent import _esc_pools

    cfg: dict[str, Any] = {"seed": seed, "n_dev": n_dev, "n_eval": n_eval, "probes_dev": probes_dev, "n_noise_only": n_noise_only, "n_silence": n_silence, "fars": list(FARS),
                           "min_speech_s": cg.DEFAULT_MIN_SPEECH_S, "min_asr_logprob": cg.DEFAULT_MIN_ASR_LOGPROB, "speaker_model": "voxceleb_resnet34_LM.onnx", "asr": "faster-whisper-tiny",
                           "intent": "rule_v2", "enroll_utterances": N_ENROLL, "scenarios": [s[0] for s in SCENARIOS], "snr_db": 5.0}
    run = Run("audio", "A6_cancel_eligibility", cfg, "Piper libritts_r (904 synthetic voices) saying the phrase bank; ESC-50 household noise; LibriSpeech dev-clean-2 for a real-speech check", seed)
    missing = []
    if not (models_dir() / "audio" / "wespeaker" / cfg["speaker_model"]).is_file():
        missing.append("WeSpeaker voxceleb_resnet34_LM.onnx (homeease models fetch --audio)")
    if not (models_dir() / "audio" / "faster-whisper-tiny" / "model.bin").is_file():
        missing.append("faster-whisper-tiny (homeease models fetch --audio)")
    noise = _esc_pools()
    if not noise["dev"] or not noise["test"]:
        missing.append("ESC-50 (homeease data download --dataset esc50)")
    if missing:
        return run.finish("data_pending", {}, missing=missing)
    try:
        bank = load_phrase_bank()
    except Exception as e:  # noqa: BLE001
        return run.finish("failed", {"error": repr(e)}, headline=f"phrase bank invalid: {e}")
    from homeease.ml.audio.asr import WhisperASR
    from homeease.ml.audio.speaker import WeSpeakerEmbedder

    ok, why = multi_tts.availability()
    partial = False
    if ok:
        engine = multi_tts.PiperSpeakers()
        synth: Callable[[str, int], np.ndarray] = engine.synthesize
        n_total, info_engine = engine.n_speakers, engine.describe()
    else:
        fb = sapi_speakers()
        if fb is None:
            return run.finish("data_pending", {}, missing=[why or "multi-speaker TTS", "or an installed system TTS (Windows SAPI / macOS say)"])
        synth, names = fb
        n_total, n_dev, n_eval, partial = len(names), 0, len(names), True
        info_engine = {"engine": "system TTS voices (PARTIAL: Piper unavailable)", "n_speakers": len(names), "voices": names, "piper_unavailable_because": why}
        log.warning("A6 PARTIAL: Piper unavailable (%s); using %d system voices", why, len(names))
    cfg.update(engine=info_engine, partial=partial)
    (run.dir / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(cfg, default=str)), sort_keys=False, allow_unicode=True), encoding="utf-8")
    plan = make_plan(bank, n_total, n_dev, n_eval, seed, probes_dev, n_noise_only, n_silence)
    rng = np.random.default_rng(seed)
    load = lambda paths, k: [load_audio(p)[: 5 * SR] for p in rng.permutation(paths)[:k]]  # noqa: E731
    ctx = Ctx(synth, WeSpeakerEmbedder(cfg["speaker_model"]), WhisperASR("faster-whisper-tiny"), RuleIntentV2(), load(noise["test"], max(60, n_noise_only)),
              cfg["min_speech_s"], cfg["min_asr_logprob"], None, load_texts(bank))
    dev = collect_dev(plan, ctx) if plan.dev else DevTable(np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0), np.zeros((0, 0), dtype=np.float32))
    dev_cal = calibrate_dev(dev, ctx.min_speech_s) if len(dev.speakers) else {"calibrations": {OP_NAMES[f]: None for f in FARS}}
    ev = collect_eval(plan, ctx, thresholds_from(dev_cal))
    _save_tables(run, dev, ev)  # before any analysis, so a failure there never costs the synthesis + ASR time (see `rescore_run`)
    m = analyse(dev, ev, ctx.min_speech_s, ctx.min_asr_logprob, seed, partial)
    real: dict[str, Any] | None = None
    if with_real_check and not partial:
        try:
            real = real_speech_check(ctx.embedder, thresholds_from(m["dev"]), seed, ctx.min_speech_s)
        except Exception as e:  # noqa: BLE001 - the transfer check is a side result
            log.exception("real-speech check failed")
            real = {"status": "failed", "missing": [repr(e)[:200]]}
    if real is not None:
        m["real_speech_check"] = real
    head, md = render(m, {"engine": info_engine, "speaker_model": cfg["speaker_model"], "min_speech_s": ctx.min_speech_s, "min_asr_logprob": ctx.min_asr_logprob, "real_speech": real})
    return run.finish("synthetic_only" if partial else "completed", m, ("PARTIAL RUN. " if partial else "") + head, md)


if __name__ == "__main__":  # python -m homeease.experiments.audio_cancel rescore <run_dir>
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "rescore":
        print(rescore_run(sys.argv[2])[1])
    else:
        print(__doc__)
