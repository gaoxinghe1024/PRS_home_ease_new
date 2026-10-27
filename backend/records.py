"""删除跌倒事件与运动训练记录，连同关联通知、录像片段与回放派生文件。

与应用内的查看路径保持一致：

* 录像片段 ``runtime/video_bridge/systems/records/clips/<clip>``
* 回放派生文件 ``runtime/playback/<clip>``
* 原视频引擎自己的明细 ``runtime/video_bridge/systems/records/falls.db``

删除是**逐条精确**的：只处理与所选记录一一对应的片段，不清空整个目录，
不触碰居民资料、联系人、声音历史与监测覆盖记录。所有文件操作都限制在数据目录内。
调用方（``care/services.py``）负责取报告范围与记录编号；本模块不做范围判断。
"""

import json
import logging
import sqlite3
from pathlib import Path

from backend.config import RUNTIME

log = logging.getLogger(__name__)

CLIPS = RUNTIME / "video_bridge" / "systems" / "records" / "clips"
ENGINE_DB = RUNTIME / "video_bridge" / "systems" / "records" / "falls.db"
PLAYBACK = RUNTIME / "playback"


def _inside(path):
    """只允许删除数据目录内的真实文件（拒绝符号链接与越界路径）。"""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if not resolved.is_relative_to(RUNTIME) or resolved == RUNTIME:
        return False
    return path.is_file() and not path.is_symlink()


def clip_names(events):
    """事件里登记的录像片段名（去重，保留顺序）。"""
    names = []
    for event in events:
        clip = (event.get("evidence") or {}).get("clip")
        if isinstance(clip, str) and clip and Path(clip).name == clip:
            if clip not in names:
                names.append(clip)
    return names


def drop_evidence(names):
    """删除片段文件、回放派生文件与引擎明细，返回各自删除的数量（尽力而为）。"""
    removed = {"clips": 0, "playback": 0, "engine_rows": 0}
    for name in names:
        for directory, key in ((CLIPS, "clips"), (PLAYBACK, "playback")):
            path = directory / name
            if _inside(path):
                try:
                    path.unlink()
                    removed[key] += 1
                except OSError:
                    log.warning("无法删除 %s", path)
    if names and ENGINE_DB.is_file():
        placeholders = ",".join("?" * len(names))
        try:
            with sqlite3.connect(ENGINE_DB, timeout=5) as conn:
                removed["engine_rows"] = conn.execute(
                    f"DELETE FROM falls WHERE clip IN ({placeholders})", names).rowcount
        except sqlite3.Error:
            # 视频工作进程可能正占用该库；片段文件已删除，明细留待下次清理。
            log.warning("无法清理引擎明细 %s", ENGINE_DB)
    return removed


def drop_exercise_traces(ids):
    for sid in ids:
        if Path(sid).name != sid:
            continue
        path = RUNTIME / "exercise-traces" / f"{sid}.json"
        if _inside(path):
            try:
                path.unlink()
            except OSError:
                log.warning("无法删除训练诊断 %s", path)


class Records:
    """记录删除：事件与训练各自独立，通知随事件一并删除。"""

    def __init__(self, store):
        self.store = store

    def events(self, ids):
        ids = [str(one) for one in dict.fromkeys(ids) if one]
        if not ids:
            return {"events": 0, "notifications": 0, "clips": 0, "playback": 0, "engine_rows": 0}
        placeholders = ",".join("?" * len(ids))
        with self.store.connection() as conn:
            rows = conn.execute(f"SELECT payload FROM events WHERE id IN ({placeholders})", ids).fetchall()
            found = [json.loads(row[0]) for row in rows]
            notifications = conn.execute(
                f"DELETE FROM notifications WHERE event_id IN ({placeholders})", ids).rowcount
            deleted = conn.execute(f"DELETE FROM events WHERE id IN ({placeholders})", ids).rowcount
        removed = drop_evidence(clip_names(found))
        return {"events": deleted, "notifications": notifications, **removed}

    def sessions(self, ids):
        ids = [str(one) for one in dict.fromkeys(ids) if one]
        if not ids:
            return {"sessions": 0}
        placeholders = ",".join("?" * len(ids))
        with self.store.connection() as conn:
            found = [row[0] for row in conn.execute(f"SELECT id FROM sessions WHERE id IN ({placeholders})", ids)]
            deleted = conn.execute(f"DELETE FROM sessions WHERE id IN ({placeholders})", ids).rowcount
        drop_exercise_traces(found)
        return {"sessions": deleted}

    def scope(self, mode, since, events=True, sessions=True):
        """清空某个报告范围：默认同时删除事件与训练记录。"""
        result = {"events": 0, "notifications": 0, "sessions": 0, "clips": 0, "playback": 0, "engine_rows": 0}
        if events:
            with self.store.connection() as conn:
                rows = conn.execute("SELECT payload FROM events WHERE mode=? AND created_at>=?", (mode, since)).fetchall()
                found = [json.loads(row[0]) for row in rows]
                result["notifications"] = conn.execute(
                    "DELETE FROM notifications WHERE event_id IN (SELECT id FROM events WHERE mode=? AND created_at>=?)",
                    (mode, since)).rowcount
                result["events"] = conn.execute("DELETE FROM events WHERE mode=? AND created_at>=?", (mode, since)).rowcount
            result.update({key: value for key, value in drop_evidence(clip_names(found)).items()})
        if sessions:
            with self.store.connection() as conn:
                found = [row[0] for row in conn.execute("SELECT id FROM sessions WHERE mode=? AND created_at>=?", (mode, since))]
                result["sessions"] = conn.execute("DELETE FROM sessions WHERE mode=? AND created_at>=?", (mode, since)).rowcount
            drop_exercise_traces(found)
        return result
