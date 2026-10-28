"""Run-folder helper: the newest completed run of each audio experiment, with its metrics.json loaded lazily."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "artifacts" / "experiments"
EVIDENCE_DIR = RUNS_DIR  # the result folders in this package are the committed runs

TOLERANCE = 0.0005  # the brief's tolerance; a print with fewer decimals must be the correctly rounded value instead

# experiment key -> marker in the run folder name (the newest folder containing the marker, with status completed, is used)
MARKERS: dict[str, str] = {
    "A1": "_audio_A1_vad_",
    "A2": "_audio_A2_sound_events_",
    "A2b": "_audio_A2b_sound_model_",
    "A3": "_audio_A3_response_intent_sapi_",
    "A3b": "_audio_A3b_response_intent_bank_",
    "A4": "_audio_A4_speaker_verification_",
    "A5": "_audio_A5_alarm_behaviour_",
    "A6": "_audio_A6_cancel_eligibility_",
}
# earlier runs that the results document quotes by name next to the newest one
FIXED_RUNS: dict[str, str] = {
    "A6first": "20261008-113720_audio_A6_cancel_eligibility_cac405cb21",
    "A2b_first": "20261008-002335_audio_A2b_sound_model_2b3d962b34",
    # the A6 run whose two thresholds are the demo defaults (src/homeease/ml/audio/replay.py DEMO_THRESHOLD)
    "A6demo": "20261008-132958_audio_A6_cancel_eligibility_cac405cb21",
    # the first A2 run on this machine and the first A3b run: the text compares the newest run with them (CNN run-to-run variation, hallucination probe)
    "A2first": "20261007-235935_audio_A2_sound_events_083bdeb9b0",
    "A3bfirst": "20261008-082808_audio_A3b_response_intent_bank_sapi_cee2ad886d",
    "A3first": "20261008-082700_audio_A3_response_intent_sapi_9fe9479c15",
    # A4: two runs before the seeding fix (salted hash(): the noisy conditions differ between them) and the first of the two runs after it (identical to the newest)
    "A4salt1": "20261008-143327_audio_A4_speaker_verification_646d18511c",
    "A4salt2": "20261008-185033_audio_A4_speaker_verification_646d18511c",
    "A4det1": "20261008-193346_audio_A4_speaker_verification_646d18511c",
}


# ---------------------------------------------------------------------------------------------------------------- run folders
class Runs:
    """The newest completed run folder of each experiment, with its metrics.json loaded lazily."""

    def __init__(self, pins: dict[str, str] | None = None) -> None:
        self.pins = {**FIXED_RUNS, **(pins or {})}
        self._cache: dict[str, dict[str, Any] | None] = {}

    def folder(self, exp: str) -> Path | None:
        """The newest completed run folder of an experiment (or the pinned one)."""
        if exp in self.pins:
            for base in (RUNS_DIR, EVIDENCE_DIR):
                p = base / self.pins[exp]
                if (p / "metrics.json").is_file():
                    return p
            return None
        marker = MARKERS[exp]
        best: Path | None = None
        names = sorted({p.name for base in (RUNS_DIR, EVIDENCE_DIR) if base.is_dir() for p in base.glob(f"*{marker}*")})  # by name = by start time
        for name in names:
            for base in (RUNS_DIR, EVIDENCE_DIR):  # a local folder wins over its snapshot
                p = base / name
                f = p / "metrics.json"
                if not f.is_file():
                    continue
                try:
                    st = json.loads(f.read_text(encoding="utf-8")).get("status")
                except (OSError, ValueError):
                    continue
                if st == "completed":
                    best = p
                break
        return best

    def load(self, exp: str) -> dict[str, Any] | None:
        if exp not in self._cache:
            f = self.folder(exp)
            self._cache[exp] = json.loads((f / "metrics.json").read_text(encoding="utf-8")) if f else None
        return self._cache[exp]

    def metrics(self, exp: str) -> dict[str, Any]:
        d = self.load(exp)
        if d is None:
            raise RunMissing(exp)
        return d["metrics"]

    def name(self, exp: str) -> str:
        f = self.folder(exp)
        return f.name if f else "(not present)"


class RunMissing(Exception):
    pass

