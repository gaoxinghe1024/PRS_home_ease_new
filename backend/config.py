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
# 后台常驻时自动运行的算法通道。只决定「是否交给原算法运行」，不涉及任何
# 模型、阈值或判定条件；未开启的通道算法完全不会收到输入。
ALGORITHM_CHANNELS = {
    "resident": "Always on in background",
    "vision": "Video fall detection",
    "accel": "Accelerometer fall detection",
}
# 运动检测通道的键名沿用原算法2的动作编号，便于前端与 EXERCISES 对齐。
ALGORITHM_KEYS = tuple(ALGORITHM_CHANNELS) + tuple(EXERCISES)


def prepare():
    for directory in (RUNTIME, UPLOADS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
