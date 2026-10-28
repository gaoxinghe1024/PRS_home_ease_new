import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("HOMEEASE_SYSTEM_DATA", str(ROOT / "runtime"))).resolve()
DB_PATH = RUNTIME / "homeease.sqlite3"
UPLOADS = RUNTIME / "uploads"
LOGS = RUNTIME / "logs"
VIDEO = ROOT / "algorithms" / "fall_video"
SIGNAL = ROOT / "algorithms" / "fall_signal"
EXERCISE = ROOT / "algorithms" / "exercise"
AUDIO = ROOT / "algorithms" / "audio"
EXERCISES = {
    "chair_sit_to_stand": "Sit-to-stand",
    "squat": "Squat",
    "standing_leg_raise": "Standing leg raise",
    "hurdle_step": "Hurdle step",
}
# Algorithm channels that run automatically in always-on mode. A switch only decides whether input is handed to the
# original algorithm; no model, threshold or decision rule is involved, and a channel that is off receives no input at all.
ALGORITHM_CHANNELS = {
    "resident": "Always on in background",
    "vision": "Video fall detection",
    "accel": "Accelerometer fall detection",
}
# Exercise channel keys reuse the original Algorithm 2 movement ids, so the frontend and EXERCISES line up.
ALGORITHM_KEYS = tuple(ALGORITHM_CHANNELS) + tuple(EXERCISES)


def prepare():
    for directory in (RUNTIME, UPLOADS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
