"""Training, thresholds, package round trip and the monitor on a synthetic scene for the 4-class sound model.

Fixtures are generated tones / noise bursts (seeded); nothing here downloads data, loads a real model or opens a microphone.
These tests prove that the pipeline is leak-free and that a saved package behaves like the evaluated model; they say nothing about recognition accuracy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sf = pytest.importorskip("soundfile", reason="needs the optional audio extra (uv sync --extra audio)")

from homeease.core.clock import VirtualClock
from homeease.core.config import SoundMonitorCfg
from homeease.datasets.adapters import audio_v2
from homeease.ml.audio import sound_model as SM
from homeease.ml.audio import sound_train as ST
from homeease.ml.audio.monitor import SoundMonitor
from homeease.ml.audio.sound_package import ModelLoadError, load_sound_classifier, normalize_window, save_sound_package

SR = 16000
CLASSES = list(audio_v2.CLASSES)


def synth(cls: str, rng: np.random.Generator, seconds: float = 1.2) -> np.ndarray:
    """Four crude, separable sounds standing in for the four classes."""
    t = np.arange(int(seconds * SR)) / SR
    if cls == "HELP":  # high harmonic call with vibrato
        x = sum(np.sin(2 * np.pi * (900 + 40 * k) * t + 3 * np.sin(2 * np.pi * 6 * t)) / k for k in range(1, 4))
    elif cls == "IMPACT":  # decaying noise burst
        x = rng.standard_normal(len(t)) * np.exp(-t / 0.05)
    elif cls == "ABNORMAL":  # mid-pitched beeps
        x = np.sin(2 * np.pi * 450 * t) * (np.sin(2 * np.pi * 3 * t) > 0)
    else:  # low rumble
        x = np.convolve(rng.standard_normal(len(t)), np.ones(40) / 40, mode="same")
    x = x / (np.abs(x).max() + 1e-9) * (0.2 + 0.4 * rng.random())
    return (x + rng.standard_normal(len(t)) * 0.003).astype(np.float32)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    d = tmp_path_factory.mktemp("synth_corpus")
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for i in range(10 * 4 * 2):  # 10 groups per class (2 clips each) spread over 5 folds, 2 sources
        cls = CLASSES[i % 4]
        grp = i // 8
        p = d / f"c{i}.wav"
        sf.write(str(p), synth(cls, rng), SR)
        src = "vivae" if cls == "HELP" or i % 8 < 4 else "esc50"
        rows.append({"source": src, "path": str(p), "group": f"{src}:g{cls}{grp}", "fold": (grp + CLASSES.index(cls)) % 5 + 1, "homeease_class": cls, "subtype": cls.lower()})
    audio_v2.check_group_folds(rows)
    return rows


@pytest.fixture(scope="module")
def ws(corpus: list[dict[str, Any]]) -> ST.WindowSet:
    return ST.build_windows(corpus, seed=3, with_patch=True, workers=2)


# ------------------------------------------------------------------ windows
def test_cut_window_takes_the_loudest_part_and_places_short_clips() -> None:
    rng = np.random.default_rng(0)
    n = int(2.0 * SR)
    x = np.zeros(5 * SR, dtype=np.float32)
    x[int(3.1 * SR):int(3.6 * SR)] = 0.5
    w = ST.cut_window(x, n, rng, "eval")
    assert len(w) == n and np.abs(w).max() == pytest.approx(0.5)  # the burst is inside the chosen window
    short = np.ones(int(0.5 * SR), dtype=np.float32)
    c = ST.cut_window(short, n, rng, "eval")
    assert len(c) == n and c[: int(0.7 * SR)].sum() == 0 and c.sum() == pytest.approx(len(short))  # centred, rest empty
    starts = {int(np.argmax(ST.cut_window(short, n, np.random.default_rng(s), "aug"))) for s in range(8)}
    assert len(starts) > 1  # training-only placement jitters


def test_windows_are_level_normalised_so_recording_gain_cannot_identify_a_corpus() -> None:
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    x = synth("HELP", np.random.default_rng(5))
    quiet, loud = ST.make_window(x * 0.05, rng_a, "eval"), ST.make_window(x * 4.0, rng_b, "eval")
    assert np.sqrt(np.mean(quiet**2)) == pytest.approx(0.1, rel=0.02) and np.sqrt(np.mean(loud**2)) == pytest.approx(0.1, rel=0.02)
    np.testing.assert_allclose(quiet, loud, atol=2e-3)  # same window whatever the gain (dither scales with the level too)


def test_window_set_keeps_extra_windows_in_the_fold_of_their_clip(ws: ST.WindowSet, corpus: list[dict[str, Any]]) -> None:
    assert len(ws) == 2 * len(corpus) and ws.feat.shape[0] == len(ws) and ws.patch.shape[1:] == (1, 64, ST.PATCH_FRAMES)
    assert np.isfinite(ws.feat).all() and np.isfinite(ws.patch).all()
    by_group: dict[str, set[int]] = {}
    for g, f in zip(ws.group, ws.fold):
        by_group.setdefault(str(g), set()).add(int(f))
    assert all(len(v) == 1 for v in by_group.values())  # windows (original and augmented) of one group share one fold
    assert int(ws.aug.sum()) == len(corpus)


# ------------------------------------------------------------------ metrics, thresholds
def test_report_counts_unknown_as_a_miss_and_not_as_a_prediction() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 2, 3])
    pred = np.array([0, 0, ST.UNKNOWN, 1, 1, 1, 2, 0])
    r = ST.report(y, pred)
    assert r["n"] == 8 and r["coverage"] == pytest.approx(7 / 8)
    h = r["per_class"]["HELP"]
    assert (h["support"], h["predicted"]) == (4, 3) and h["recall"] == pytest.approx(2 / 4) and h["precision"] == pytest.approx(2 / 3)  # UNKNOWN: not recalled, not predicted
    assert r["confusion"][0] == [2, 1, 0, 0, 1] and sum(map(sum, r["confusion"])) == 8
    assert r["per_class"]["DAILY"]["recall"] == 0.0 and r["per_class"]["HELP"]["recall_ci95"][0] < 0.5 < r["per_class"]["HELP"]["recall_ci95"][1]


def test_decide_uses_per_class_cutoffs() -> None:
    p = np.array([[0.60, 0.20, 0.10, 0.10], [0.60, 0.20, 0.10, 0.10], [0.10, 0.70, 0.10, 0.10]])
    assert ST.decide(p).tolist() == [0, 0, 1]
    assert ST.decide(p, [0.7, 0.5, 0.5, 0.5]).tolist() == [ST.UNKNOWN, ST.UNKNOWN, 1]  # HELP needs 0.7
    assert ST.decide(p, 0.65).tolist() == [ST.UNKNOWN, ST.UNKNOWN, 1]


def test_choose_cutoffs_reaches_the_precision_target_on_the_given_data_only() -> None:
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 4, n)
    conf = rng.uniform(0.3, 0.99, n)
    right = rng.random(n) < conf  # calibrated by construction: accuracy grows with confidence
    pred = np.where(right, y, (y + 1) % 4)
    probs = np.full((n, 4), 0.0)
    probs[np.arange(n), pred] = conf
    probs += ((1 - conf) / 3)[:, None] * (probs == 0)
    cut = ST.choose_cutoffs(probs, y, 0.9)
    k, top = probs.argmax(1), probs.max(1)
    for i, c in enumerate(CLASSES):
        info = cut["per_class"][c]
        acc = (k == i) & (top >= info["cutoff"])
        assert info["target_met"] and (y[acc] == i).mean() >= 0.9 and info["dev_accepted"] == int(acc.sum())
    assert cut["dev_windows"] == n and cut["target_precision"] == 0.9
    hopeless = ST.choose_cutoffs(np.tile([0.3, 0.3, 0.2, 0.2], (50, 1)), rng.integers(0, 4, 50), 0.9)
    assert not hopeless["per_class"]["HELP"]["target_met"]  # an unreachable target is reported, not faked
    assert hopeless["per_class"]["HELP"]["fallback"] == "max_dev_f1"


def test_unreachable_precision_falls_back_to_the_best_f1_cutoff_not_to_silence() -> None:
    """Precision tops out at 0.6 here: demanding the highest precision would put the cut-off at the grid edge and silence the class."""
    rng = np.random.default_rng(1)
    n = 800
    y = rng.integers(0, 4, n)
    conf = rng.uniform(0.3, 0.99, n)
    right = rng.random(n) < 0.2 + 0.4 * conf  # accuracy never exceeds 0.6
    pred = np.where(right, y, (y + 1) % 4)
    probs = np.zeros((n, 4))
    probs[np.arange(n), pred] = conf
    probs += ((1 - conf) / 3)[:, None] * (probs == 0)
    cut = ST.choose_cutoffs(probs, y, 0.9)
    k, top = probs.argmax(1), probs.max(1)
    for i, c in enumerate(CLASSES):
        info = cut["per_class"][c]
        assert not info["target_met"] and info["fallback"] == "max_dev_f1"
        f1 = {}
        for t in np.round(np.arange(0.30, 0.9901, 0.01), 2):
            acc = (k == i) & (top >= t)
            if acc.sum() < 10:
                break
            p_, r_ = (y[acc] == i).mean(), (y[acc] == i).sum() / (y == i).sum()
            f1[float(t)] = 2 * p_ * r_ / (p_ + r_)
        assert info["cutoff"] == max(f1, key=f1.get)  # type: ignore[arg-type]
        assert info["cutoff"] < 0.9 and info["dev_recall"] > 0.3  # the class still fires


def test_temperature_fit_softens_overconfident_logits() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 800)
    logits = rng.standard_normal((800, 4)) * 2
    logits[np.arange(800), y] += 2.0
    logits *= 4.0  # same ranking, absurdly confident
    assert ST.fit_temperature(logits, y) > 2.0


def test_bootstrap_resamples_groups_not_windows() -> None:
    groups = np.repeat(np.arange(20), 5)
    ci = ST.boot_ci(groups, lambda i: float(i.size), n_boot=200, seed=0)
    assert ci is not None and ci[0] <= 100 <= ci[1]
    assert ST.boot_ci(np.repeat(np.arange(3), 5), lambda i: 1.0) is None  # too few groups for an honest interval


# ------------------------------------------------------------------ end to end on the synthetic corpus
def test_train_sound_model_is_leak_free_and_the_saved_package_reproduces_the_evaluation(corpus: list[dict[str, Any]], ws: ST.WindowSet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {"fit_folds": [], "cutoff_n": []}
    real_fit, real_cut = ST.fit_candidate, ST.choose_cutoffs

    def spy_fit(name: str, tr: ST.WindowSet, va: Any, seed: int, epochs: int | None = None) -> Any:
        seen["fit_folds"].append(set(np.unique(tr.fold).tolist()) | (set(np.unique(va.fold).tolist()) if va is not None else set()))
        return real_fit(name, tr, va, seed, epochs)

    def spy_cut(probs: np.ndarray, y: np.ndarray, *a: Any, **k: Any) -> Any:
        seen["cutoff_n"].append(len(y))
        return real_cut(probs, y, *a, **k)

    monkeypatch.setattr(ST, "fit_candidate", spy_fit)
    monkeypatch.setattr(ST, "choose_cutoffs", spy_cut)
    res = SM.train_sound_model(corpus, model_id="unit_sound", version="v1", candidates=("mfcc_logmel_stats+logreg", "mfcc_logmel_stats+rf"), test_fold=5, seed=3, ws=ws, root=tmp_path, run_id="unit")
    assert all(5 not in folds for folds in seen["fit_folds"])  # the test fold is in no training, validation or final fit
    n_dev_eval = int(((ws.fold != 5) & ~ws.aug).sum())
    assert seen["cutoff_n"] == [n_dev_eval] and res["cutoffs"]["dev_windows"] == n_dev_eval  # thresholds from dev predictions only
    assert res["test"]["n_windows"] == int(((ws.fold == 5) & ~ws.aug).sum())
    assert res["candidate"] in ("mfcc_logmel_stats+logreg", "mfcc_logmel_stats+rf")

    pkg = Path(res["package"])
    assert pkg == tmp_path / "audio_sound" / "unit_sound" / "v1"
    assert sorted(p.name for p in pkg.iterdir()) == ["meta.json", "model.joblib"]  # weights + metadata, never audio, features or embeddings of a recording
    meta = json.loads((pkg / "meta.json").read_text(encoding="utf-8"))
    assert meta["classes"] == CLASSES and meta["input"]["normalization"] == "rms" and meta["input"]["window_s"] == 2.0
    assert meta["thresholds"]["per_class"].keys() == set(CLASSES) and "dev" in meta["thresholds"]["chosen_on"]
    assert meta["training"]["split"]["test_fold"] == 5 and meta["training"]["run_id"] == "unit"
    assert ".wav" not in json.dumps(meta) and str(tmp_path) not in json.dumps(meta["training"])  # no paths of the training clips
    rc = res["reload_check"]
    assert rc["windows_compared"] >= 5 and rc["max_abs_probability_difference"] < 1e-3  # the loaded package gives the evaluated probabilities
    assert rc["min_conf_by_class"] == meta["thresholds"]["per_class"]

    # the saved package drives SoundMonitor on a scene (virtual clock): classes come out, HELP needs its own cut-off
    rng = np.random.default_rng(11)
    gap = (rng.standard_normal(6 * SR) * 10 ** (-58 / 20)).astype(np.float32)
    scene = np.concatenate([gap, synth("HELP", rng) * 0.8, synth("HELP", rng) * 0.8, gap, synth("IMPACT", rng), gap])
    out = SM.run_scene(str(pkg), scene)
    assert out["status"] == "ok" and out["windows_scored"] > 0 and set(out["min_conf_by_class"]) == set(CLASSES)
    assert all(e["category"] in (*CLASSES, "UNKNOWN") for e in out["events"])


def test_candidate_selection_ignores_the_test_fold(corpus: list[dict[str, Any]], ws: ST.WindowSet, tmp_path: Path) -> None:
    """Corrupting the labels of the test fold must not change the chosen candidate, thresholds or dev scores."""
    kw = dict(model_id="a", candidates=("mfcc_logmel_stats+logreg",), test_fold=5, seed=3, root=tmp_path)
    a = SM.train_sound_model(corpus, ws=ws, version="v1", **kw)  # type: ignore[arg-type]
    ws2 = ws.take(np.ones(len(ws), dtype=bool))
    ws2.y = ws2.y.copy()
    ws2.y[ws2.fold == 5] = (ws2.y[ws2.fold == 5] + 1) % 4
    b = SM.train_sound_model(corpus, ws=ws2, version="v2", **kw)  # type: ignore[arg-type]
    assert a["dev"]["mfcc_logmel_stats+logreg"]["macro_f1"] == b["dev"]["mfcc_logmel_stats+logreg"]["macro_f1"]
    assert a["thresholds"]["per_class"] == b["thresholds"]["per_class"]
    assert a["test"]["with_cutoffs"]["macro_f1"] != b["test"]["with_cutoffs"]["macro_f1"]  # only the test score reacts


def test_cnn_candidate_trains_calibrates_and_round_trips(ws: ST.WindowSet, corpus: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    torch.manual_seed(0)
    monkeypatch.setattr(ST, "train_cnn", lambda *a, **k: _quick_cnn(*a, **k))
    res = SM.train_sound_model(corpus, model_id="unit_cnn", version="v1", candidates=("logmel_small_cnn",), test_fold=5, seed=3, ws=ws, root=tmp_path)
    meta = json.loads((Path(res["package"]) / "meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "torch" and meta["input"]["representation"] == "logmel_patch" and meta["input"]["patch_frames"] == ST.PATCH_FRAMES
    assert meta["calibration"]["method"] == "temperature" and meta["calibration"]["T"] == pytest.approx(res["temperature"])
    assert res["reload_check"]["max_abs_probability_difference"] < 1e-3
    clf = load_sound_classifier(res["package"])
    p = clf.predict_proba(synth("HELP", np.random.default_rng(1), 2.0))
    assert p.shape == (4,) and p.sum() == pytest.approx(1.0)


def _quick_cnn(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, n_out: int, seed: int = 0, epochs: int = 30) -> Any:
    from homeease.ml.audio.sound import train_cnn as real

    return real(Xtr, ytr, Xva, yva, n_out, seed, epochs=3)


# ------------------------------------------------------------------ package + monitor behaviour
def _fit_tiny() -> Any:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = np.concatenate([ST.clip_features(ST.make_window(synth(c, rng), rng, "eval"))[None] for c in CLASSES for _ in range(8)])
    y = np.repeat(np.arange(4), 8)
    return LogisticRegression(max_iter=500).fit(X, y)


def test_package_round_trip_with_per_class_cutoffs_and_stub_model(tmp_path: Path) -> None:
    d = save_sound_package(_fit_tiny(), "sklearn", CLASSES, {"window_s": 2.0, "representation": "clip_features", "normalization": "rms"}, "stub_lr", version="v1",
                           thresholds={"min_conf": 0.5, "per_class": {"HELP": 0.8, "IMPACT": 0.4}, "chosen_on": "dev"}, root=tmp_path)
    clf = load_sound_classifier(str(d))
    assert clf.min_conf == 0.5 and clf.min_conf_by_class == {"HELP": 0.8, "IMPACT": 0.4}
    mon = SoundMonitor(SoundMonitorCfg(), clf, VirtualClock())
    assert mon.min_conf_by_class == {"HELP": 0.8, "IMPACT": 0.4} and mon.status()["min_conf_by_class"] == {"HELP": 0.8, "IMPACT": 0.4}
    mon_cfg = SoundMonitor(SoundMonitorCfg(min_conf=0.99), clf, VirtualClock())
    assert mon_cfg.min_conf_by_class == {} and mon_cfg.min_conf == 0.99  # an explicit config value is a global override
    again = load_sound_classifier(str(d))
    w = synth("IMPACT", np.random.default_rng(2), 2.0)
    np.testing.assert_allclose(clf.predict_proba(w), again.predict_proba(w))
    np.testing.assert_allclose(clf.predict_proba(w * 5.0), clf.predict_proba(w * 0.3), atol=2e-3)  # rms normalisation: mic level is irrelevant


class _Fixed:
    model_id, classes, window_s = "fixed", ["HELP", "IMPACT", "ABNORMAL", "DAILY"], 2.0

    def __init__(self, probs: list[float], by_class: dict[str, float] | None = None, min_conf: float | None = None) -> None:
        self.p, self.min_conf_by_class, self.min_conf = np.array(probs), by_class, min_conf

    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        return self.p


@pytest.mark.parametrize("probs,by_class,expect", [
    ([0.60, 0.20, 0.10, 0.10], {"HELP": 0.7}, "UNKNOWN"),  # below the HELP cut-off
    ([0.75, 0.15, 0.05, 0.05], {"HELP": 0.7}, "HELP"),
    ([0.10, 0.60, 0.20, 0.10], {"HELP": 0.7}, "IMPACT"),  # classes without an entry use the global value (0.5)
    ([0.10, 0.45, 0.25, 0.20], {"HELP": 0.7}, "UNKNOWN"),
])
def test_monitor_applies_per_class_cutoffs(probs: list[float], by_class: dict[str, float], expect: str) -> None:
    mon = SoundMonitor(SoundMonitorCfg(), _Fixed(probs, by_class), VirtualClock())
    mon.submit_window(np.zeros(32000, dtype=np.float32), 0.0, 2000.0)
    assert mon.status()["windows_by_label"] == {expect: 1}


def test_unknown_is_never_a_class_and_per_class_entries_for_unknown_classes_are_ignored() -> None:
    mon = SoundMonitor(SoundMonitorCfg(), _Fixed([0.6, 0.2, 0.1, 0.1], {"HELP": 0.3, "SIREN": 0.9, "UNKNOWN": 0.1}), VirtualClock())
    assert mon.min_conf_by_class == {"HELP": 0.3}
    bad = _Fixed([0.6, 0.2, 0.1, 0.1])
    bad.classes = ["HELP", "UNKNOWN", "ABNORMAL", "DAILY"]
    with pytest.raises(ValueError):
        SoundMonitor(SoundMonitorCfg(), bad, VirtualClock())


# ------------------------------------------------------------------ pretrained embedding packages (stub embedder, no model file)
class _FakeEmbedder:
    sha256 = "a" * 64

    def window_embedding(self, x: np.ndarray) -> np.ndarray:
        e = np.zeros(2048, dtype=np.float32)
        e[0] = float(np.sqrt(np.mean(x**2))) * 10
        return e


def test_yamnet_embedding_package_pins_its_embedder(tmp_path: Path) -> None:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 2048)).astype(np.float32)
    clf = LogisticRegression(max_iter=200).fit(X, np.repeat(np.arange(4), 10))
    spec = {"window_s": 2.0, "representation": "yamnet_embedding", "normalization": "rms", "embedder": "yamnet", "embedder_sha256": "a" * 64}
    d = save_sound_package(clf, "sklearn", CLASSES, spec, "emb", version="v1", root=tmp_path)
    loaded = load_sound_classifier(str(d), embedder=_FakeEmbedder())
    p = loaded.predict_proba(synth("HELP", rng, 2.0))
    assert p.shape == (4,) and p.sum() == pytest.approx(1.0)
    wrong = _FakeEmbedder()
    wrong.sha256 = "b" * 64
    with pytest.raises(ModelLoadError, match="checksum"):
        load_sound_classifier(str(d), embedder=wrong)  # a different embedder file is refused, not run
    with pytest.raises(ModelLoadError):
        save_sound_package(clf, "sklearn", CLASSES, {"window_s": 2.0, "representation": "yamnet_embedding"}, "emb2", version="v1", root=tmp_path)
    assert not (tmp_path / "audio_sound" / "emb2").exists()


@pytest.mark.realdata
def test_yamnet_onnx_embedding_shape_when_the_model_is_present() -> None:
    from homeease.core.paths import models_dir
    from homeease.ml.audio.embed import EMBED_DIM, YamnetEmbedder

    if not (models_dir() / "audio" / "yamnet" / "yamnet.onnx").is_file():
        pytest.skip("YAMNet ONNX not fetched (homeease models fetch --audio)")
    e = YamnetEmbedder().window_embedding(synth("HELP", np.random.default_rng(0), 2.0))
    assert e.shape == (EMBED_DIM,) and np.isfinite(e).all()


def test_normalize_window_is_idempotent() -> None:
    x = synth("DAILY", np.random.default_rng(3), 2.0)
    for mode in ("rms", "peak"):
        once = normalize_window(x, mode)
        np.testing.assert_allclose(normalize_window(once, mode), once, atol=1e-6)
    assert normalize_window(x, "none") is x


# ------------------------------------------------------------------ experiment A2, 4-class part
def test_a2_four_class_reports_data_pending_without_a_help_corpus(corpus: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    from homeease.experiments import audio_suite

    only_esc = [r for r in corpus if r["source"] == "esc50"]
    monkeypatch.setattr(audio_v2, "combined_rows", lambda *a, **k: only_esc)
    out = audio_suite._sound_four_class(1)
    assert out["metrics"]["status"] == "data_pending" and "VIVAE" in out["metrics"]["missing"][0]
    assert "data_pending" in out["md"] and "No numbers are given" in out["md"]


def test_a2_four_class_runs_and_marks_leave_one_corpus_out_not_evaluated_with_a_single_help_corpus(corpus: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    from homeease.experiments import audio_suite

    monkeypatch.setattr(audio_v2, "combined_rows", lambda *a, **k: corpus)
    monkeypatch.setattr(ST, "CANDIDATES", ("mfcc_logmel_stats+logreg",))
    out = audio_suite._sound_four_class(1)
    m = out["metrics"]
    assert m["status"] == "completed" and list(m["candidates"]) == ["mfcc_logmel_stats+logreg"]
    assert m["loco"]["status"] == "not_evaluated" and "VIVAE and the FSD50K subset" in m["loco"]["reason"]
    d = m["candidates"]["mfcc_logmel_stats+logreg"]
    assert d["n_windows"] == len(corpus) and len(d["per_fold"]) == 5 and sum(f["n"] for f in d["per_fold"]) == len(corpus)
    assert set(d["per_source"]) == {"esc50", "vivae"} and d["report"]["per_class"]["HELP"]["support"] == sum(r["homeease_class"] == "HELP" for r in corpus)
    assert "not_evaluated" in out["md"] and "Wilson" in out["md"] and "leave-one-corpus-out" in out["headline"]


def test_a2_four_class_leave_one_corpus_out_trains_without_the_held_out_corpus(corpus: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    from homeease.experiments import audio_suite

    rows = [dict(r) for r in corpus]
    for r in rows:  # half of the HELP groups become a second corpus
        if r["homeease_class"] == "HELP" and int(r["group"].rsplit("HELP", 1)[1]) % 2:
            r["source"], r["group"] = "fsd50k", r["group"].replace("vivae:", "fsd50k:")
    monkeypatch.setattr(audio_v2, "combined_rows", lambda *a, **k: rows)
    monkeypatch.setattr(ST, "CANDIDATES", ("mfcc_logmel_stats+logreg",))
    seen: list[set[str]] = []
    real = ST.fit_candidate
    monkeypatch.setattr(ST, "fit_candidate", lambda name, tr, va, seed, epochs=None: (seen.append(set(tr.source.tolist()) | (set(va.source.tolist()) if va is not None else set())), real(name, tr, va, seed, epochs))[1])
    out = audio_suite._sound_four_class(1)
    runs = out["metrics"]["loco"]["runs"]["mfcc_logmel_stats+logreg"]
    assert set(runs) == {"vivae", "fsd50k"}
    assert runs["fsd50k"]["train_sources"] == ["esc50", "vivae"] and runs["vivae"]["train_sources"] == ["esc50", "fsd50k"]
    loco_fits = seen[-2:]  # one fit per held-out corpus (classical models fit once on train+val)
    assert "vivae" not in loco_fits[0] and "fsd50k" not in loco_fits[1]
    assert runs["fsd50k"]["n_test"] == sum(r["source"] == "fsd50k" for r in rows) and runs["fsd50k"]["help_support"] == runs["fsd50k"]["n_test"]
