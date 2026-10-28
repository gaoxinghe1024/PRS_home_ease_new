"""Database engine, sessions and migrations (SQLite with WAL, FK, busy timeout)."""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from homeease.core.paths import project_root

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _sqlite_url(url: str) -> str:
    """Resolve relative sqlite paths against the project root."""
    prefix = "sqlite:///"
    if url.startswith(prefix) and not url.startswith("sqlite:////") and ":memory:" not in url:
        rel = url[len(prefix):]
        p = Path(rel)
        if not p.is_absolute():
            p = project_root() / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return prefix + str(p)
    return url


def make_engine(url: str) -> Engine:
    url = _sqlite_url(url)
    kwargs: dict[str, Any] = {"future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 15}
        if ":memory:" in url:
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn: Any, _rec: Any) -> None:  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            if ":memory:" not in url:
                cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=15000")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

    return engine


class Database:
    """Owns the engine, a session factory and a process-wide domain lock.

    The domain lock serialises event / notification / exercise state
    transitions across the runtime thread, the dispatcher and API workers.
    """

    def __init__(self, url: str) -> None:
        self.url = _sqlite_url(url)
        self.engine = make_engine(self.url)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        self.lock = threading.RLock()

    @contextlib.contextmanager
    def session(self) -> Iterator[Session]:
        s = self.SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    @contextlib.contextmanager
    def tx(self) -> Iterator[Session]:
        """Serialised domain transaction (lock + commit/rollback)."""
        with self.lock:
            with self.session() as s:
                yield s

    def migrate(self) -> None:
        upgrade(self.url, self.engine)

    def dispose(self) -> None:
        self.engine.dispose()


def alembic_config(url: str) -> Any:
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(url: str, engine: Engine | None = None) -> None:
    from alembic import command

    cfg = alembic_config(url)
    if engine is not None:
        with engine.begin() as conn:
            cfg.attributes["connection"] = conn
            command.upgrade(cfg, "head")
    else:
        command.upgrade(cfg, "head")
