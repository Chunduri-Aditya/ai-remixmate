"""
scripts/core/db.py — SQLAlchemy Core engine factory.

Persistence topology
--------------------
``DATABASE_URL`` unset (local dev, CI, existing installs)
    One SQLite engine **per store**, keyed by that store's on-disk path. The
    files are byte for byte the ones jobs.py / crates.py / feedback_store.py
    wrote before this layer existed (data/jobs.db, data/remixmate.db,
    data/feedback.db). The test suite's ``monkeypatch.setattr(mod, "_DB_PATH",
    tmp)`` fixtures keep redirecting because ``get_engine`` resolves the path at
    call time, not at import.

``DATABASE_URL`` set (hosted: ``postgresql+psycopg://…``)
    One shared engine for every store — all six tables live in one database.

This keeps a per-store engine when ``DATABASE_URL`` is unset on purpose:
collapsing to a single engine there would defeat the per-store path override
the Phase 0 storage-isolation fixtures rely on.

Usage
-----
    from scripts.core import db
    engine = db.get_engine(str(_DB_PATH))
    with engine.begin() as conn:            # write: commits on exit
        conn.execute(text("INSERT ..."), params)
    with closing(engine.connect()) as conn: # read: no implicit commit
        rows = conn.execute(text("SELECT ...")).fetchall()

psycopg is imported by SQLAlchemy only when a ``postgresql+psycopg`` engine is
actually built, so an install that never sets ``DATABASE_URL`` never loads it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import NullPool

_DATABASE_URL_ENV = "DATABASE_URL"


def database_url() -> str | None:
    """The configured database URL, or ``None`` for per-store SQLite."""
    return os.environ.get(_DATABASE_URL_ENV, "").strip() or None


@lru_cache(maxsize=None)
def _engine_for(url: str) -> Engine:
    """Build (once) the engine for a fully-formed URL.

    Cached on the URL string: every ``get_engine`` call for the same SQLite path
    — or any call at all once ``DATABASE_URL`` is set — reuses one engine.
    """
    if url.startswith("sqlite"):
        return create_engine(
            url,
            future=True,
            # NullPool: each .connect() opens a fresh sqlite3 connection and
            # closes it on exit, matching the pre-existing connect-per-call
            # style and stopping hundreds of per-tmp-path test engines from
            # each pinning an idle handle.
            poolclass=NullPool,
            connect_args={"check_same_thread": False},
        )
    return create_engine(url, future=True, pool_pre_ping=True)


def get_engine(sqlite_path: str | os.PathLike[str]) -> Engine:
    """Return the engine for a store.

    ``sqlite_path`` is that store's SQLite file. It is used only when
    ``DATABASE_URL`` is unset; when it is set every store shares one engine and
    the path is ignored.
    """
    url = database_url()
    if url:
        return _engine_for(url)
    path = Path(os.fspath(sqlite_path))
    # The pre-existing per-module _get_conn() each did this before opening the
    # file; ensure_directories() is not called at import time.
    path.parent.mkdir(parents=True, exist_ok=True)
    return _engine_for(f"sqlite:///{path}")


def is_sqlite(engine: Engine) -> bool:
    """True when this engine talks to SQLite (guards PRAGMA-only statements)."""
    return engine.dialect.name == "sqlite"
