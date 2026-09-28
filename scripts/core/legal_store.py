"""
scripts/core/legal_store.py — DMCA reports + flagged-user registry (Phase 8).

SQLAlchemy Core over the same engine factory as crates/feedback: SQLite at
``data/legal.db`` by default, shared Postgres when ``DATABASE_URL`` is set.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from sqlalchemy import Connection, inspect, text

from scripts.core import db
from scripts.core.paths import DATA_DIR

_DB_PATH = DATA_DIR / "legal.db"


def _schema_ddl(dialect_name: str) -> tuple[str, ...]:
    ts = "DOUBLE PRECISION" if dialect_name == "postgresql" else "REAL"
    return (
        f"""CREATE TABLE IF NOT EXISTS dmca_reports (
    report_id       TEXT PRIMARY KEY,
    reporter_name   TEXT NOT NULL,
    reporter_email  TEXT NOT NULL,
    copyright_work  TEXT NOT NULL,
    infringing_url  TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    created_at      {ts} NOT NULL
)""",
        f"""CREATE TABLE IF NOT EXISTS flagged_users (
    user_id     TEXT PRIMARY KEY,
    reason      TEXT NOT NULL DEFAULT '',
    flagged_by  TEXT NOT NULL DEFAULT '',
    flagged_at  {ts} NOT NULL
)""",
    )


def _conn() -> Connection:
    engine = db.get_engine(str(_DB_PATH))
    c = engine.connect()
    if db.is_sqlite(engine):
        c.exec_driver_sql("PRAGMA foreign_keys = ON")
    return c


def _ensure_schema(conn: Connection) -> None:
    dialect = conn.engine.dialect.name
    for stmt in _schema_ddl(dialect):
        conn.execute(text(stmt))


def init_schema() -> None:
    """Create and commit both legal tables. Called once at startup; idempotent."""
    with _conn() as c:
        _ensure_schema(c)
        c.commit()


def create_dmca_report(
    *,
    reporter_name: str,
    reporter_email: str,
    copyright_work: str,
    infringing_url: str,
    description: str = "",
) -> str:
    """Insert a DMCA report row; return its ``report_id``."""
    report_id = str(uuid.uuid4())
    with _conn() as c:
        _ensure_schema(c)
        c.execute(
            text(
                """
                INSERT INTO dmca_reports
                    (report_id, reporter_name, reporter_email, copyright_work,
                     infringing_url, description, created_at)
                VALUES
                    (:report_id, :reporter_name, :reporter_email, :copyright_work,
                     :infringing_url, :description, :created_at)
                """
            ),
            {
                "report_id": report_id,
                "reporter_name": reporter_name.strip(),
                "reporter_email": reporter_email.strip(),
                "copyright_work": copyright_work.strip(),
                "infringing_url": infringing_url.strip(),
                "description": (description or "").strip(),
                "created_at": time.time(),
            },
        )
        c.commit()
    return report_id


def get_dmca_report(report_id: str) -> Optional[dict[str, Any]]:
    with _conn() as c:
        _ensure_schema(c)
        row = c.execute(
            text("SELECT * FROM dmca_reports WHERE report_id = :id"),
            {"id": report_id},
        ).fetchone()
    return dict(row._mapping) if row else None


def flag_user(user_id: str, *, reason: str = "", flagged_by: str = "") -> None:
    """Upsert a flagged-user row. Flagged users get 403 from ``require_user``."""
    uid = (user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    with _conn() as c:
        _ensure_schema(c)
        c.execute(
            text(
                """
                INSERT INTO flagged_users (user_id, reason, flagged_by, flagged_at)
                VALUES (:user_id, :reason, :flagged_by, :flagged_at)
                ON CONFLICT(user_id) DO UPDATE SET
                    reason     = excluded.reason,
                    flagged_by = excluded.flagged_by,
                    flagged_at = excluded.flagged_at
                """
            ),
            {
                "user_id": uid,
                "reason": (reason or "").strip(),
                "flagged_by": (flagged_by or "").strip(),
                "flagged_at": time.time(),
            },
        )
        c.commit()


def unflag_user(user_id: str) -> bool:
    with _conn() as c:
        _ensure_schema(c)
        cur = c.execute(
            text("DELETE FROM flagged_users WHERE user_id = :id"),
            {"id": user_id},
        )
        c.commit()
        return cur.rowcount > 0


def is_flagged(user_id: str) -> bool:
    """Plain SELECT, no DDL: the schema comes from ``init_schema()`` at startup.

    A database that has never had ``init_schema()`` or ``flag_user()`` run
    against it holds no flags, so the missing table reads as "not flagged"
    (a catalog lookup, not a CREATE) instead of a 500 on every gated route.
    """
    if not user_id:
        return False
    with _conn() as c:
        if not inspect(c).has_table("flagged_users"):
            return False
        row = c.execute(
            text("SELECT 1 FROM flagged_users WHERE user_id = :id LIMIT 1"),
            {"id": user_id},
        ).fetchone()
    return row is not None
