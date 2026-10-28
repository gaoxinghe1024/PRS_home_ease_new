"""Command line of the Algorithm 3 experiments: `python -m homeease <command>`.

    models fetch --audio [--audio-tts]     pretrained models (pinned revisions, sha256 recorded in models/audio/MODELS.json)
    data download --dataset <name>         esc50, librispeech_mini, vivae, fsd50k_help
    compare --suite audio [--only A1,A4]   experiments A1 to A6
    train-sound                            the deployable four-class sound-event model
    audio scenes | audio replay            scripted scenes and an offline replay of a WAV file
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from homeease.core.paths import models_dir, resolve
from homeease.core.util import dumps, setup_logging, sha256_file

EXIT_MISSING = 3
EXIT_PENDING = 4

DATASETS = {
    "esc50": "data/raw/esc50",
    "librispeech_mini": "data/raw/librispeech_mini",
    "vivae": "data/raw/vivae",  # fear / pain vocalisations, 93 MB, CC BY-NC 4.0
    "fsd50k_help": "data/raw/fsd50k_help",  # per-file FSD50K subset (screams and same-corpus negatives), about 50 MB
}

# Revisions are pinned so that results stay reproducible.
AUDIO_MODELS = {
    "faster-whisper-tiny": {"repo": "Systran/faster-whisper-tiny", "revision": "d90ca5fe260221311c53c58e660288d3deb8d356", "dest": "audio/faster-whisper-tiny", "check": "model.bin", "license": "MIT"},
    "faster-whisper-base": {"repo": "Systran/faster-whisper-base", "revision": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66", "dest": "audio/faster-whisper-base", "check": "model.bin", "license": "MIT"},
    "wespeaker-voxceleb-resnet34-LM": {"repo": "Wespeaker/wespeaker-voxceleb-resnet34-LM", "revision": "f0c48c298fd835726c27956a5d617bad7115627e", "file": "voxceleb_resnet34_LM.onnx", "dest": "audio/wespeaker", "check": "voxceleb_resnet34_LM.onnx", "license": "CC-BY-4.0"},
    "wespeaker-cnceleb-resnet34-LM": {"repo": "Wespeaker/wespeaker-cnceleb-resnet34-LM", "revision": "ed25c1ecbd9e8d3c7944df95430da02bd3cdcf7a", "file": "cnceleb_resnet34_LM.onnx", "dest": "audio/wespeaker", "check": "cnceleb_resnet34_LM.onnx", "license": "Apache-2.0"},
    # sound-event embedding (candidate of experiment A2): Google YAMNet, tf2onnx conversion, 16 MB, waveform in, 1024-d embeddings out
    "yamnet-onnx": {"repo": "audiomagic/yamnet-onnx", "revision": "f25b741c2f0bdc6d7e6db24b5fddda23347dbafd", "file": "yamnet.onnx", "dest": "audio/yamnet", "check": "yamnet.onnx", "license": "Apache-2.0"},
    # multi-speaker TTS for experiment A6 only (many speakers saying the same words): Piper en_US-libritts_r-medium, 904 speakers, 78.6 MB.
    # Opt-in (--audio-tts) and needs the optional `audio-tts` extra (piper-tts, GPL-3.0).
    "piper-libritts-r-medium": {"repo": "rhasspy/piper-voices", "revision": "c10ece1aade47bb51c153c893d14e5bf8e5b7117", "file": "en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx", "dest": "audio/piper",
                                "check": "en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx", "license": "CC-BY-4.0 (LibriTTS-R training data)", "group": "tts",
                                "expected_sha256": "10bb85e071d616fcf4071f369f1799d0491492ab3c5d552ec19fb548fac13195"},
    "piper-libritts-r-medium-config": {"repo": "rhasspy/piper-voices", "revision": "c10ece1aade47bb51c153c893d14e5bf8e5b7117", "file": "en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json", "dest": "audio/piper",
                                       "check": "en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json", "license": "CC-BY-4.0 (LibriTTS-R training data)", "group": "tts"},
}


def fetch_audio_models(force: bool = False, groups: tuple[str, ...] = ()) -> int:
    """faster-whisper (ASR) and WeSpeaker (speaker verification) from the Hugging Face Hub. Silero VAD ships inside the `silero-vad` package.
    Entries with a "group" (the Piper voice model, group "tts") are only fetched when that group is asked for."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("install the audio extra first: pip install -e .[audio]", file=sys.stderr)
        return EXIT_MISSING
    manifest_p = models_dir() / "audio" / "MODELS.json"
    manifest: dict[str, Any] = json.loads(manifest_p.read_text()) if manifest_p.is_file() else {}
    rc = 0
    for k, m in AUDIO_MODELS.items():
        if m.get("group") and m["group"] not in groups:
            continue
        dest = models_dir() / m["dest"]
        target = dest / m["check"]
        if not target.is_file() or force:
            try:
                print(f"downloading {k} @ {m['revision'][:8]} ...")
                if "file" in m:
                    hf_hub_download(m["repo"], m["file"], revision=m["revision"], local_dir=str(dest))
                else:
                    snapshot_download(m["repo"], revision=m["revision"], local_dir=str(dest))
            except Exception as e:  # network or hub changes must be visible
                print(f"FAILED {k}: {e}", file=sys.stderr)
                rc = EXIT_MISSING
                continue
        digest = sha256_file(target)
        if m.get("expected_sha256") and digest != m["expected_sha256"]:  # upstream LFS oid: a mismatch means a corrupt or replaced file
            print(f"FAILED {k}: sha256 {digest[:16]} != expected {m['expected_sha256'][:16]}", file=sys.stderr)
            rc = EXIT_MISSING
            continue
        prev = manifest.get(k, {}).get("sha256")
        if prev and prev != digest:
            print(f"WARNING {k}: checksum changed ({prev[:12]} -> {digest[:12]}); review before use", file=sys.stderr)
        manifest[k] = {"repo": m["repo"], "revision": m["revision"], "path": f"{m['dest']}/{m['check']}", "sha256": digest, "license": m["license"], "bytes": target.stat().st_size}
        print(f"ok {k} sha256={digest[:16]}")
    manifest_p.parent.mkdir(parents=True, exist_ok=True)
    manifest_p.write_text(dumps(manifest, indent=2))
    return rc


def cmd_models_fetch(args: argparse.Namespace) -> int:
    if not (args.audio or args.audio_tts):
        print("nothing to fetch: pass --audio (and --audio-tts for experiment A6)", file=sys.stderr)
        return EXIT_MISSING
    return fetch_audio_models(args.force, ("tts",) if args.audio_tts else ())


def cmd_data_download(args: argparse.Namespace) -> int:
    raw = resolve(args.raw_dir or DATASETS[args.dataset])
    assert raw is not None
    if args.dataset in ("esc50", "librispeech_mini"):
        from homeease.datasets.adapters import audio_sets

        out = audio_sets.download(args.dataset, raw)
    elif args.dataset == "vivae":
        from homeease.datasets.adapters import vivae

        out = vivae.download(raw)
    else:
        from homeease.datasets.adapters import fsd50k_help

        out = fsd50k_help.download(raw)
    print(dumps(out, indent=2))
    return 0 if not out.get("failed") and out.get("status") != "failed" else EXIT_MISSING


def cmd_train_sound(args: argparse.Namespace) -> int:
    from homeease.experiments.train_sound import PackageExists, run_train_sound

    setup_logging(to_file=False)
    try:
        res = run_train_sound(args.model_id, args.version, args.candidates.split(",") if args.candidates else None, args.test_fold, args.seed, args.precision_target)
    except PackageExists as e:
        print(str(e), file=sys.stderr)
        return 2
    print(dumps({"status": res["status"], "run_id": res["provenance"]["run_id"], "headline": res["headline"], "missing": res["missing"]}, indent=2))
    return 0 if res["status"] == "completed" else (EXIT_PENDING if res["status"] in ("data_pending", "not_evaluated") else 1)


def cmd_compare(args: argparse.Namespace) -> int:
    setup_logging(to_file=False)
    if args.suite != "audio":
        print("only the audio suite is part of this package", file=sys.stderr)
        return EXIT_MISSING
    from homeease.experiments.audio_suite import run_suite

    entries = run_suite(args.only.split(",") if args.only else None, args.seed)
    for e in entries:
        print(f"[{e['status']:>14}] {e['id']:<8} {e['experiment']:<24} {e['headline'][:160]}")
    return 1 if "failed" in {e["status"] for e in entries} else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="homeease", description="Algorithm 3 experiments: voice interaction and audio monitoring")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("models", help="model files")
    ms = p.add_subparsers(dest="sub", required=True)
    f = ms.add_parser("fetch", help="download the audio models with checksums")
    f.add_argument("--audio", action="store_true", help="faster-whisper tiny and base, WeSpeaker x2, YAMNet")
    f.add_argument("--audio-tts", action="store_true", help="also the 78.6 MB Piper multi-speaker voice model for experiment A6 (implies --audio)")
    f.add_argument("--force", action="store_true")
    f.set_defaults(fn=cmd_models_fetch)

    p = sub.add_parser("data", help="dataset download")
    ds = p.add_subparsers(dest="sub", required=True)
    d = ds.add_parser("download")
    d.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    d.add_argument("--raw-dir", default=None)
    d.set_defaults(fn=cmd_data_download)

    p = sub.add_parser("compare", help="run the comparison suite")
    p.add_argument("--suite", required=True, choices=["audio"])
    p.add_argument("--only", default=None, help="comma-separated experiment ids, e.g. A1,A4")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("train-sound", help="train the 4-class sound-event model (HELP/IMPACT/ABNORMAL/DAILY) and save models/audio_sound/<id>/<version>/")
    p.add_argument("--model-id", default="sound_events", help="package id")
    p.add_argument("--version", default=None, help="package version (default: timestamp); versions are immutable")
    p.add_argument("--candidates", default=None, help="comma-separated subset of: mfcc_logmel_stats+rf, mfcc_logmel_stats+logreg, logmel_small_cnn, yamnet_embedding+logreg")
    p.add_argument("--test-fold", type=int, default=5, choices=[1, 2, 3, 4, 5], help="fold kept out of training and threshold selection")
    p.add_argument("--precision-target", type=float, default=0.90, help="per-class precision the UNKNOWN cut-off must reach on dev data")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_train_sound)

    from homeease import cli_audio

    cli_audio.register(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
