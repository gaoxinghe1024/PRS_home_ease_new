"""Scripted, seeded audio scenes for the Function 3 demo and the offline end-to-end checks (no microphone, no human).

A scene is a household background (ESC-50 washing-machine hum, quiet) with sounds placed on a timeline: a thud, a
cry, breaking glass, a TV speaker, and the spoken reply of the person to the voice inquiry (Windows SAPI / macOS ``say``
voices). Every choice (which clip, when, how loud) comes from ``numpy.random.default_rng([seed, crc32(name)])``, so a
scene is bit-identical on every run on the same data and TTS engine.

Hygiene
* Clips are taken from FOLD 5 of the class mapping v2 split, the fold that the shipped ``sound_events`` model never saw
  (``homeease train-sound``: folds 1-4 train, fold 5 test). A scene is a plumbing demo, not an accuracy measurement.
* Everything is clean, short, acted or synthetic (TTS); a scene says nothing about older adults or real homes.
* Scenes need data that may be missing (ESC-50, VIVAE, FSD50K subset, LibriSpeech, a TTS voice). A scene whose inputs
  are missing is skipped with a message that names the missing input; nothing is invented.
* The scene files are synthetic test inputs in the gitignored ``data/synthetic/audio_scenes/`` (about 0.8 MB each); the
  JSON sidecar next to a WAV holds ground truth and sources, never audio. The product never writes audio anywhere.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from homeease.core.util import get_logger
from homeease.ml.audio.features import SR, load_audio

log = get_logger(__name__)

SCENE_VERSION = "f3-scenes/1"
DEFAULT_SEED = 42
HELD_OUT_FOLD = 5  # the fold of the v2 split that the shipped sound model was neither trained nor tuned on
PROMPT_S = 4.0  # the spoken prompt takes about this long before listening starts (measured with SAPI: ~3.8 s)
INQUIRY_WINDOW_S = 10.0
BACKGROUND_DBFS = -48.0
DITHER_DBFS = -62.0  # no digital silence anywhere (a perfectly silent window is not a household sound)
CROSSFADE_S = 0.5
FADE_S = 0.01

# loudness (RMS over the active frames of a clip) per role
LEVELS_DBFS = {"daily": -34.0, "glass": -22.0, "thud": -20.0, "scream": -24.0, "tv_speech": -32.0, "knock": -24.0, "reply": -26.0}

LICENCE = {"esc50": "CC BY-NC 3.0", "vivae": "CC BY-NC 4.0", "librispeech": "CC BY 4.0", "fsd50k": "per clip, see data/raw/fsd50k_help/selection.json"}
HINTS = {
    "esc50": "homeease data download --dataset esc50",
    "vivae": "homeease data download --dataset vivae",
    "fsd50k": "homeease data download --dataset fsd50k_help",
    "librispeech": "homeease data download --dataset librispeech_mini",
}
# role -> ordered (source, subtype prefix) preferences; the first non-empty group wins, a seeded draw picks inside it
ROLES: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "background": [("esc50", ("washing_machine",))],
    "daily": [("esc50", ("water_drops", "footsteps", "keyboard_typing", "door_wood_creaks", "pouring_water", "clock_tick"))],
    "glass": [("esc50", ("glass_breaking",)), ("fsd50k", ("Shatter",))],
    "thud": [("fsd50k", ("Thump_and_thud",)), ("esc50", ("door_wood_knock",))],
    "knock": [("esc50", ("door_wood_knock",)), ("fsd50k", ("Knock",))],
    "scream": [("vivae", ("pain/strong", "pain/peak", "fear/strong", "fear/peak")), ("fsd50k", ("Screaming",))],
    "tv_speech": [("librispeech", ("read_speech",))],
}


class SceneUnavailable(RuntimeError):
    """A scene cannot be built here; the message names the missing input and how to get it."""


@dataclass(frozen=True)
class Clip:
    source: str  # esc50 | vivae | fsd50k | librispeech
    path: str
    subtype: str
    fold: int | None
    group: str = ""
    license: str | None = None


class Catalog(Protocol):
    """Where the scene builder gets sound clips from (the real data on disk, or a stub in the tests)."""

    def candidates(self, role: str) -> list[Clip]: ...

    def load(self, clip: Clip) -> np.ndarray: ...  # mono float32 at 16 kHz


class DiskCatalog:
    """Clips from ``data/raw`` as listed by the class mapping v2 (``datasets.adapters.audio_v2``). Prefers the held-out fold."""

    def __init__(self, data_root: Path | None = None, fold: int = HELD_OUT_FOLD, rows: Sequence[dict[str, Any]] | None = None) -> None:
        self.fold = fold
        if rows is None:
            from homeease.core.paths import data_dir
            from homeease.datasets.adapters.audio_v2 import combined_rows

            rows = combined_rows(Path(data_root) if data_root else data_dir())
        self.rows = [dict(r) for r in rows]
        self.sources = sorted({r["source"] for r in self.rows})

    def candidates(self, role: str) -> list[Clip]:
        for source, prefixes in ROLES[role]:
            pool = [r for r in self.rows if r["source"] == source and any(str(r.get("subtype", "")).startswith(p) for p in prefixes)]
            held = [r for r in pool if r.get("fold") == self.fold]
            use = held or pool  # fall back to any fold, flagged in the sidecar (held_out=false)
            if use:
                return [Clip(r["source"], str(r["path"]), str(r.get("subtype", "")), r.get("fold"), str(r.get("group", "")), r.get("license")) for r in sorted(use, key=lambda r: str(r["path"]))]
        return []

    def load(self, clip: Clip) -> np.ndarray:
        return load_audio(clip.path, SR)


# ---------------------------------------------------------------------------------- signal helpers
def rms_db(x: np.ndarray) -> float:
    return float(10.0 * np.log10(float(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)) if len(x) else -120.0


def active_rms_db(x: np.ndarray, frame: int = SR // 100, span_db: float = 30.0) -> float:
    """RMS over the frames within ``span_db`` of the loudest frame: the level of the sound itself, not of the silence around it."""
    n = len(x) // frame
    if n == 0:
        return rms_db(x)
    fr = np.asarray(x[: n * frame], dtype=np.float64).reshape(n, frame)
    lv = 10.0 * np.log10(np.mean(fr * fr, axis=1) + 1e-12)
    keep = lv >= lv.max() - span_db
    return float(10.0 * np.log10(float(np.mean(fr[keep] ** 2)) + 1e-12))


def scale_to(x: np.ndarray, target_dbfs: float, max_peak: float = 0.9) -> np.ndarray:
    """Scale to a target active level, never above ``max_peak`` (a loud impulsive clip is limited by its peak)."""
    x = np.asarray(x, dtype=np.float32)
    if not np.any(x):
        raise SceneUnavailable("a source clip is digital silence")
    g = 10.0 ** ((target_dbfs - active_rms_db(x)) / 20.0)
    peak = float(np.max(np.abs(x))) * g
    if peak > max_peak:
        g *= max_peak / peak
    return (x * g).astype(np.float32)


def fade(x: np.ndarray, seconds: float = FADE_S) -> np.ndarray:
    n = min(int(seconds * SR), len(x) // 2)
    if n < 2:
        return x
    y = x.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def tile(clip: np.ndarray, seconds: float, cross_s: float = CROSSFADE_S) -> np.ndarray:
    """Repeat a clip to ``seconds`` with equal-power cross-fades (a 5 s ESC-50 clip becomes a continuous background)."""
    n, c = int(seconds * SR), int(cross_s * SR)
    if len(clip) <= 2 * c:
        raise SceneUnavailable("background clip is too short to tile")
    out = np.zeros(n, dtype=np.float32)
    up = np.sin(np.linspace(0.0, np.pi / 2, c, dtype=np.float32))
    down = np.cos(np.linspace(0.0, np.pi / 2, c, dtype=np.float32))
    step = len(clip) - c
    pos = k = 0
    while pos < n:  # overlap-add: the tail of one copy fades out while the head of the next fades in
        seg = np.asarray(clip, dtype=np.float32).copy()
        if k > 0:
            seg[:c] *= up
        seg[-c:] *= down
        end = min(pos + len(seg), n)
        out[pos:end] += seg[: end - pos]
        pos += step
        k += 1
    return out


# ---------------------------------------------------------------------------------- scene specs
@dataclass(frozen=True)
class Item:
    kind: str  # a role of ROLES, or "reply"
    t0: float  # seconds; a seeded jitter of +-JITTER_S is added
    expect: str | None = None  # the category a perfect sound branch would report (None: not a sound-branch expectation)


@dataclass(frozen=True)
class SceneSpec:
    description: str
    duration_s: float
    items: tuple[Item, ...]
    fall_at_s: float | None = None  # a scripted suspected fall (the audio never creates it) that starts the voice inquiry
    reply: tuple[str, str, str, str] | None = None  # (voice, text, role, language)


JITTER_S = 0.4
SCENES: dict[str, SceneSpec] = {
    "quiet_home": SceneSpec("Washing-machine hum with one everyday sound. Nothing should be reported as HELP, IMPACT or ABNORMAL.", 20.0, (Item("daily", 9.0, "DAILY"),)),
    "glass_break": SceneSpec("Breaking glass in a quiet home.", 20.0, (Item("glass", 8.0, "ABNORMAL"),)),
    "tv_and_knock": SceneSpec("A TV speaker talks for ten seconds (speech, not a person answering), then someone knocks.", 24.0, (Item("tv_speech", 3.0, None), Item("knock", 18.0, "IMPACT"))),
    "fall_help_call": SceneSpec("A thud, a cry, and the person calls for help when asked 'Do you need help?' (suspected fall at 4 s).", 24.0, (Item("thud", 4.0, "IMPACT"), Item("scream", 5.6, "HELP"), Item("reply", 0.0, None)),
                                fall_at_s=4.0, reply=("Zira", "Help me, I fell and I cannot get up.", "self", "en")),
    "fall_reassured": SceneSpec("A thud; the person answers 'I'm fine' when asked (suspected fall at 4 s).", 24.0, (Item("thud", 4.0, "IMPACT"), Item("reply", 0.0, None)),
                                fall_at_s=4.0, reply=("Zira", "I'm fine, don't worry.", "self", "en")),
    "fall_visitor_reply": SceneSpec("A thud; a different voice (a visitor) answers 'she is fine' when asked (suspected fall at 4 s).", 24.0, (Item("thud", 4.0, "IMPACT"), Item("reply", 0.0, None)),
                                    fall_at_s=4.0, reply=("David", "Don't worry, she is fine.", "visitor", "en")),
}
SCENE_NAMES = tuple(SCENES)
ENROLL_TEXTS = (
    "Good morning, this is my voice and I am setting up the speaker check.",
    "The weather is mild today and I would like a cup of tea after lunch.",
    "My granddaughter visits on Sunday and we usually cook something together.",
)

TTS = Callable[[str, str, str], np.ndarray]  # (text, voice, lang) -> mono float32 at 16 kHz


def default_tts(text: str, voice: str, lang: str) -> np.ndarray:
    from homeease.ml.audio import tts as tts_mod

    try:
        return tts_mod.synthesize(text, voice, lang)
    except tts_mod.TTSError as e:  # TTSUnavailable is a TTSError
        raise SceneUnavailable(f"needs a text-to-speech voice ({voice}): {str(e)[:120]} (Windows SAPI or macOS say; see docs/function3_voice_audio.md)") from e


@dataclass
class Scene:
    name: str
    seed: int
    description: str
    audio: np.ndarray = field(repr=False)  # mono float32, 16 kHz
    truth: list[dict[str, Any]]
    inquiry: dict[str, Any] | None
    sources: list[dict[str, Any]]
    gain_db: float = 0.0

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SR

    def meta(self) -> dict[str, Any]:
        """Everything about the scene except the samples (JSON-safe)."""
        return {"schema": SCENE_VERSION, "name": self.name, "seed": self.seed, "description": self.description, "sample_rate": SR, "duration_s": round(self.duration_s, 3),
                "background_dbfs": BACKGROUND_DBFS, "scene_gain_db": round(self.gain_db, 2), "truth": self.truth, "inquiry": self.inquiry, "sources": self.sources,
                "note": "synthetic scene built from clean public clips and TTS; a plumbing demo, not household or older-adult audio; sources come from the held-out fold of the sound model (held_out=true)"}


def _pick(rng: np.random.Generator, clips: list[Clip]) -> Clip:
    return clips[int(rng.integers(len(clips)))]


def _file_sha256(path: str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def build_scene(name: str, catalog: Catalog, seed: int = DEFAULT_SEED, tts: TTS | None = None) -> Scene:
    """Build one scene. Raises ``SceneUnavailable`` (with the missing input) instead of returning a partial scene."""
    if name not in SCENES:
        raise KeyError(f"unknown scene {name!r}; available: {', '.join(SCENE_NAMES)}")
    spec = SCENES[name]
    rng = np.random.default_rng([int(seed), zlib.crc32(name.encode("utf-8"))])
    n = int(round(spec.duration_s * SR))
    sources: list[dict[str, Any]] = []

    def use(role: str) -> tuple[Clip, np.ndarray]:
        cands = catalog.candidates(role)
        if not cands:
            wanted = " or ".join(f"{s} {'/'.join(p)}" for s, p in ROLES[role])
            hints = "; ".join(dict.fromkeys(HINTS[s] for s, _ in ROLES[role]))
            raise SceneUnavailable(f"no '{role}' clip available (needs {wanted}); run: {hints}")
        clip = _pick(rng, cands)
        x = catalog.load(clip)
        sources.append({"role": role, "source": clip.source, "file": Path(clip.path).name, "subtype": clip.subtype, "fold": clip.fold, "group": clip.group,
                        "held_out": clip.fold == HELD_OUT_FOLD, "license": clip.license or LICENCE.get(clip.source), "sha256": _file_sha256(clip.path)})
        return clip, x

    _, bg = use("background")
    x = tile(scale_to(bg, BACKGROUND_DBFS, max_peak=0.5), spec.duration_s)
    x += (rng.standard_normal(n) * 10.0 ** (DITHER_DBFS / 20.0)).astype(np.float32)
    truth: list[dict[str, Any]] = []
    inquiry: dict[str, Any] | None = None
    inq_start = (spec.fall_at_s + PROMPT_S) if spec.fall_at_s is not None else None

    for it in spec.items:
        if it.kind == "reply":
            assert spec.reply is not None and inq_start is not None
            voice, text, role, lang = spec.reply
            sig = (tts or default_tts)(text, voice, lang)
            sig = fade(scale_to(np.asarray(sig, dtype=np.float32), LEVELS_DBFS["reply"]))
            t0 = inq_start + 1.0 + float(rng.uniform(0.0, 0.5))
            sources.append({"role": "reply", "source": "tts", "voice": voice, "text": text, "speaker_role": role, "language": lang, "license": "synthesised on the fly by the operating-system voice"})
            truth.append({"kind": "reply", "t0_s": round(t0, 3), "t1_s": round(t0 + len(sig) / SR, 3), "expect": None, "voice": voice, "text": text, "speaker_role": role})
            inquiry = {"fall_at_s": spec.fall_at_s, "prompt_s": PROMPT_S, "inquiry_start_s": round(inq_start, 3), "window_s": INQUIRY_WINDOW_S, "level": "emergency",
                       "reply_voice": voice, "reply_text": text, "reply_role": role, "reply_t0_s": round(t0, 3),
                       "note": "the fall event is scripted: audio never creates or cancels an event; the reply is read from the scene timeline like a microphone window"}
        else:
            _, clip_x = use(it.kind)
            t0 = max(0.0, it.t0 + float(rng.uniform(-JITTER_S, JITTER_S)))
            sig = fade(scale_to(clip_x[: int(12 * SR)], LEVELS_DBFS[it.kind]))
            truth.append({"kind": it.kind, "t0_s": round(t0, 3), "t1_s": round(min(t0 + len(sig) / SR, spec.duration_s), 3), "expect": it.expect})
        i = int(round(t0 * SR))
        if i + len(sig) > n:
            sig = sig[: max(n - i, 0)]
        x[i:i + len(sig)] += sig

    peak = float(np.max(np.abs(x)))
    gain_db = 0.0
    if peak > 0.98:  # overlapping items summed past full scale: one gain for the whole scene, recorded
        g = 0.98 / peak
        x *= g
        gain_db = float(20.0 * np.log10(g))
    return Scene(name, int(seed), spec.description, x.astype(np.float32), truth, inquiry, sources, gain_db)


# ---------------------------------------------------------------------------------- files
def to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def write_scene(scene: Scene, out_dir: Path) -> Path:
    """Write ``<name>-s<seed>.wav`` (16-bit PCM) and ``<name>-s<seed>.json`` (ground truth and sources, no audio). Returns the WAV path."""
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{scene.name}-s{scene.seed}"
    wav = out_dir / f"{stem}.wav"
    sf.write(str(wav), to_pcm16(scene.audio), SR, subtype="PCM_16")
    meta = scene.meta()
    meta["wav_sha256"] = hashlib.sha256(wav.read_bytes()).hexdigest()
    meta["wav_bytes"] = wav.stat().st_size
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return wav


def read_sidecar(wav_path: Path) -> dict[str, Any] | None:
    p = Path(wav_path).with_suffix(".json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and str(d.get("schema", "")).startswith("f3-scenes/") else None


def generate_scenes(names: Sequence[str] | None, seed: int, out_dir: Path, catalog: Catalog, tts: TTS | None = None) -> tuple[list[Path], dict[str, str]]:
    """Build and write scenes; returns (written WAV paths, {scene: reason it was skipped}). Never raises for missing data."""
    written: list[Path] = []
    skipped: dict[str, str] = {}
    for name in (list(names) if names else list(SCENE_NAMES)):
        try:
            written.append(write_scene(build_scene(name, catalog, seed, tts), out_dir))
        except SceneUnavailable as e:
            skipped[name] = str(e)
    return written, skipped


__all__ = ["Catalog", "Clip", "DiskCatalog", "Scene", "SceneSpec", "SceneUnavailable", "SCENES", "SCENE_NAMES", "ENROLL_TEXTS", "DEFAULT_SEED", "HELD_OUT_FOLD", "build_scene", "write_scene",
           "read_sidecar", "generate_scenes", "default_tts", "active_rms_db", "scale_to", "tile", "to_pcm16"]
