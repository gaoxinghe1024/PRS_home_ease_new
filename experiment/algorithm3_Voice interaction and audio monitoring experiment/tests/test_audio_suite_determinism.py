"""Experiment seeds must not depend on the process (A4 once used the salted built-in hash(), so its noisy conditions changed with every run)."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import zlib
from pathlib import Path

from homeease.experiments import audio_suite

EXPERIMENTS = Path(audio_suite.__file__).resolve().parent


def test_stable_seed_depends_on_the_file_name_only() -> None:
    a = audio_suite._stable_seed("D:\\data\\raw\\librispeech_mini\\84\\121123\\84-121123-0001.flac")
    b = audio_suite._stable_seed("/Users/x/data/raw/librispeech_mini/84/121123/84-121123-0001.flac")
    assert a == b == zlib.crc32(b"84-121123-0001.flac")
    assert a != audio_suite._stable_seed("/x/84-121123-0002.flac")
    assert 0 <= a < 2**32  # a valid numpy seed


def test_stable_seed_is_the_same_under_different_hash_salts() -> None:
    code = "from homeease.experiments.audio_suite import _stable_seed; print(_stable_seed('/a/b/84-121123-0001.flac'))"
    out = []
    for salt in ("1", "2", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": salt}
        out.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True, timeout=120).stdout.strip())
    assert len(set(out)) == 1 and out[0] == str(zlib.crc32(b"84-121123-0001.flac"))


def test_audio_experiments_never_seed_from_the_builtin_hash() -> None:
    """Static guard: no call of the built-in hash() in the audio experiment modules (it is salted per process)."""
    offenders = []
    for f in sorted(EXPERIMENTS.glob("audio*.py")) + [EXPERIMENTS / "train_sound.py"]:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash":
                offenders.append(f"{f.name}:{node.lineno}")
    assert not offenders, offenders
