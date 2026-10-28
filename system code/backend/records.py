"""Delete fall events and exercise sessions together with their notifications, video clips and playback copies.

The paths match the ones the application reads from:

* video clips ``runtime/video_bridge/systems/records/clips/<clip>``
* playback copies ``runtime/playback/<clip>``
* the video engine's own records ``runtime/video_bridge/systems/records/falls.db``

Deletion is **exact**: only the clips that belong to the selected records are removed, no directory is emptied,
and resident data, contacts, sound history and coverage records are never touched. All file operations stay inside the data directory.
The caller (``backend/services.py``) resolves the report range and the record ids; this module does not check ranges.
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
    """Only allow deleting real files inside the data directory (no symbolic links, no paths outside it)."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if not resolved.is_relative_to(RUNTIME) or resolved == RUNTIME:
        return False
    return path.is_file() and not path.is_symlink()


def clip_names(events):
    """Clip names registered in an event (deduplicated, in order)."""
    names = []
    for event in events:
        clip = (event.get("evidence") or {}).get("clip")
        if isinstance(clip, str) and clip and Path(clip).name == clip:
            if clip not in names:
                names.append(clip)
    return names


def drop_evidence(names):
    """Delete clip files, playback copies and engine records; return how many of each were removed (best effort)."""
    removed = {"clips": 0, "playback": 0, "engine_rows": 0}
    for name in names:
        for directory, key in ((CLIPS, "clips"), (PLAYBACK, "playback")):
            path = directory / name
            if _inside(path):
                try:
                    path.unlink()
                    removed[key] += 1
                except OSError:
                    log.warning("Cannot delete %s", path)
    if names and ENGINE_DB.is_file():
        placeholders = ",".join("?" * len(names))
        try:
            with sqlite3.connect(ENGINE_DB, timeout=5) as conn:
                removed["engine_rows"] = conn.execute(
                    f"DELETE FROM falls WHERE clip IN ({placeholders})", names).rowcount
        except sqlite3.Error:
            # The video worker may hold this database; the clip files are gone, the records are cleaned up next time.
            log.warning("Cannot clear the engine details %s", ENGINE_DB)
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
                log.warning("Could not delete training diagnostics %s", path)


class Records:
    """Record deletion: events and sessions are independent; notifications are deleted with their event."""

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
        """Clear a report range: by default both events and exercise sessions are deleted."""
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
