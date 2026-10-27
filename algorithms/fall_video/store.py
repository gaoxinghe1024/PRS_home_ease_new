# -*- coding: utf-8 -*-
"""
store.py - persistent storage of fall records (SQLite)
================================================================
Records and report data are stored in SQLite (systems/records/falls.db)
with the standard library sqlite3; no extra dependency.

Table falls:
  id          INTEGER  primary key (auto-increment)
  fall_time   TEXT     fall time (local system time, YYYY-MM-DD HH:MM:SS)
  epoch       REAL     timestamp of the same moment (for sorting / filtering)
  prob        REAL     probability at the trigger
  peak        REAL     peak probability of the event
  clip        TEXT     clip file name (3 seconds, in records/clips/)
  duration_s  REAL     clip duration (seconds)
  frames      INTEGER  number of frames in the clip
  fps         REAL     nominal clip frame rate
  threshold   REAL     decision threshold at the trigger
  source      TEXT     input source (camera 0 / urfd/fall-01 ...)
  alarm       INTEGER  alarm flag (1 = alarm)
  created_at  TEXT     time the record was stored

Thread safety: one connection per operation + an in-process RLock; SQLite also locks the file.
"""
from __future__ import annotations

import os
import json
import sqlite3
import threading
from datetime import datetime

DB_NAME = "falls.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS falls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fall_time   TEXT    NOT NULL,
    epoch       REAL,
    prob        REAL,
    peak        REAL,
    clip        TEXT,
    duration_s  REAL,
    frames      INTEGER,
    fps         REAL,
    threshold   REAL,
    source      TEXT,
    alarm       INTEGER DEFAULT 1,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_falls_epoch ON falls(epoch DESC);
"""

_COLUMNS = ("fall_time", "epoch", "prob", "peak", "clip", "duration_s",
            "frames", "fps", "threshold", "source", "alarm", "created_at")


class RecordStore:
    def __init__(self, records_dir: str):
        os.makedirs(records_dir, exist_ok=True)
        self.db_path = os.path.join(records_dir, DB_NAME)
        self._lock = threading.RLock()
        self._init_db()

    # ------------------------------------------------------------ basics
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=15)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._lock, self._conn() as c:
            c.executescript(_SCHEMA)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": int(row["id"]),
            "time": row["fall_time"],
            "epoch": float(row["epoch"] or 0.0),
            "prob": float(row["prob"] or 0.0),
            "peak": float(row["peak"] or 0.0),
            "clip": row["clip"] or "",
            "clip_url": "/api/clip/" + (row["clip"] or ""),
            "duration_s": float(row["duration_s"] or 0.0),
            "frames": int(row["frames"] or 0),
            "fps": float(row["fps"] or 0.0),
            "threshold": float(row["threshold"] or 0.0),
            "source": row["source"] or "",
            "alarm": bool(row["alarm"]),
            "created_at": row["created_at"] or "",
        }

    # ------------------------------------------------------------ add / query / delete
    def add(self, entry: dict) -> dict:
        """Store one fall record and return the complete record with its database id."""
        with self._lock, self._conn() as c:
            cur = c.execute(
                f"INSERT INTO falls ({','.join(_COLUMNS)}) "
                f"VALUES ({','.join('?' * len(_COLUMNS))})",
                (entry.get("time"), entry.get("epoch"), entry.get("prob"),
                 entry.get("peak"), entry.get("clip"), entry.get("duration_s"),
                 entry.get("frames"), entry.get("fps"), entry.get("threshold"),
                 entry.get("source"), 1 if entry.get("alarm", True) else 0,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            rid = cur.lastrowid
            row = c.execute("SELECT * FROM falls WHERE id = ?", (rid,)).fetchone()
        return self._row_to_dict(row)

    def list(self, limit: int | None = None, newest_first: bool = True) -> list[dict]:
        order = "DESC" if newest_first else "ASC"
        sql = f"SELECT * FROM falls ORDER BY epoch {order}, id {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock, self._conn() as c:
            rows = c.execute(sql).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def latest(self) -> dict | None:
        rows = self.list(limit=1, newest_first=True)
        return rows[0] if rows else None

    def count(self) -> int:
        with self._lock, self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM falls").fetchone()[0])

    def delete(self, rid) -> dict | None:
        """Delete one record and return it (so its clip file can be deleted too); None if it does not exist."""
        with self._lock, self._conn() as c:
            row = c.execute("SELECT * FROM falls WHERE id = ?", (rid,)).fetchone()
            if row is None:
                return None
            c.execute("DELETE FROM falls WHERE id = ?", (rid,))
        return self._row_to_dict(row)

    def clear(self) -> list[dict]:
        """Delete everything and return the deleted records (so their clip files can be deleted too)."""
        with self._lock:
            old = self.list(newest_first=False)
            with self._conn() as c:
                c.execute("DELETE FROM falls")
                try:
                    c.execute("DELETE FROM sqlite_sequence WHERE name='falls'")
                except sqlite3.Error:
                    pass
        return old

    def stats(self) -> dict:
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) n, MAX(peak) peak, MIN(epoch) first_epoch, "
                "MAX(epoch) last_epoch FROM falls").fetchone()
        return {"count": int(row["n"] or 0),
                "max_peak": float(row["peak"]) if row["peak"] is not None else None,
                "first_epoch": row["first_epoch"],
                "last_epoch": row["last_epoch"]}

    # ------------------------------------------------------------ migration
    def migrate_from_json(self, json_path: str) -> int:
        """Import the old falls.json into SQLite once (renamed to .bak afterwards)."""
        if not os.path.isfile(json_path):
            return 0
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return 0
        if not isinstance(data, list) or not data:
            return 0
        n = 0
        for r in data:
            try:
                self.add({
                    "time": r.get("time"),
                    "epoch": r.get("epoch"),
                    "prob": r.get("prob"),
                    "peak": r.get("peak"),
                    "clip": r.get("clip"),
                    "duration_s": r.get("duration_s"),
                    "frames": r.get("frames"),
                    "fps": r.get("fps"),
                    "threshold": r.get("threshold"),
                    "source": r.get("source"),
                    "alarm": r.get("alarm", True),
                })
                n += 1
            except Exception:
                continue
        try:
            os.replace(json_path, json_path + ".bak")
        except OSError:
            pass
        return n
