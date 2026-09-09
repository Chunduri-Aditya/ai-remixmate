"""
scripts/core/crates.py — Library crate and tag management.

A "crate" is a named group of songs — the DJ equivalent of a playlist.
Tags and favorites are per-song labels that persist across sessions.

All state lives behind a SQLAlchemy Core engine (scripts.core.db): SQLite at
data/remixmate.db by default, Postgres when DATABASE_URL is set.

Public API
----------
    # Crates
    create_crate(name)                    → crate_id
    delete_crate(crate_id)                → bool
    list_crates()                         → list[dict]
    add_to_crate(crate_id, song_name)     → bool
    remove_from_crate(crate_id, name)     → bool
    crate_songs(crate_id)                 → list[str]
    rename_crate(crate_id, new_name)      → bool

    # Tags
    add_tag(song_name, tag)               → bool
    remove_tag(song_name, tag)            → bool
    song_tags(song_name)                  → list[str]
    songs_by_tag(tag)                     → list[str]
    all_tags()                            → list[str]

    # Favorites
    set_favorite(song_name, value=True)   → None
    is_favorite(song_name)                → bool
    list_favorites()                      → list[str]
"""

from __future__ import annotations

import time
import uuid
from typing import List

from sqlalchemy import Connection, inspect, text
from sqlalchemy.exc import IntegrityError

from scripts.core import db
from scripts.core.paths import DATA_DIR

_DB_PATH = DATA_DIR / "remixmate.db"
_DEFAULT_USER = "local"

# Phase 3: every row is owned by a user id (contextvar, "local" until Phase 4
# wires auth). Reads filter `WHERE user_id = ?`; writes stamp the acting user.
#
# LIMITATION — the table-level uniqueness here is still user-blind:
#   crates.name UNIQUE            → two users cannot both name a crate "Bangers"
#   song_tags PK (song_name,tag)  → two users cannot tag the same song alike
#   song_favorites PK (song_name) → one user's favorite ON CONFLICT-overwrites
#                                   the other's row
# Harmless while get_user_id() is always "local" (single writer). Phase 4 must
# rebuild these three keys to include user_id before real multi-user. Guarded by
# tests/test_user_scope.py::test_crates_uniqueness_is_still_user_blind_phase4.

_TABLES = ("crates", "crate_songs", "song_tags", "song_favorites")


def _schema_ddl(dialect_name: str) -> tuple[str, ...]:
    """One CREATE TABLE per statement (SQLAlchemy executes one at a time, unlike
    sqlite3.executescript). Timestamp columns hold ``time.time()`` epoch values,
    so on Postgres they need DOUBLE PRECISION — float4 ``REAL`` would collide
    adjacent values and scramble the ``ORDER BY`` clauses below."""
    ts = "DOUBLE PRECISION" if dialect_name == "postgresql" else "REAL"
    return (
        f"""CREATE TABLE IF NOT EXISTS crates (
    crate_id   TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    created_at {ts} NOT NULL,
    user_id    TEXT NOT NULL DEFAULT 'local'
)""",
        f"""CREATE TABLE IF NOT EXISTS crate_songs (
    crate_id   TEXT NOT NULL REFERENCES crates(crate_id) ON DELETE CASCADE,
    song_name  TEXT NOT NULL,
    added_at   {ts} NOT NULL,
    user_id    TEXT NOT NULL DEFAULT 'local',
    PRIMARY KEY (crate_id, song_name)
)""",
        f"""CREATE TABLE IF NOT EXISTS song_tags (
    song_name  TEXT NOT NULL,
    tag        TEXT NOT NULL,
    created_at {ts} NOT NULL,
    user_id    TEXT NOT NULL DEFAULT 'local',
    PRIMARY KEY (song_name, tag)
)""",
        f"""CREATE TABLE IF NOT EXISTS song_favorites (
    song_name    TEXT PRIMARY KEY,
    favorited_at {ts} NOT NULL,
    user_id      TEXT NOT NULL DEFAULT 'local'
)""",
    )


def _acting_user() -> str:
    """Acting user id (contextvar, "local" until Phase 4). Lazy import so this
    core module never pulls in the API package at load time."""
    try:
        from scripts.api.auth import get_user_id

        return get_user_id()
    except Exception:
        return _DEFAULT_USER


def _conn() -> Connection:
    """A SQLAlchemy Connection to the crate store — SQLite file when
    ``DATABASE_URL`` is unset, the shared Postgres engine when it is set. Used
    with ``with _conn() as c:`` and an explicit ``c.commit()`` after writes.
    Kept as the module's seam (tests read the schema through it)."""
    engine = db.get_engine(str(_DB_PATH))
    c = engine.connect()
    if db.is_sqlite(engine):
        # ON DELETE CASCADE on crate_songs needs this per connection; Postgres
        # enforces foreign keys without it. exec_driver_sql skips text() parsing.
        c.exec_driver_sql("PRAGMA foreign_keys = ON")
    return c


def _init() -> None:
    engine = db.get_engine(str(_DB_PATH))
    with engine.begin() as c:
        for stmt in _schema_ddl(engine.dialect.name):
            c.execute(text(stmt))
        # Migration: a remixmate.db written before Phase 3 has no user_id
        # column. ADD COLUMN with a DEFAULT backfills every existing row to
        # "local", so single-user crates/tags/favorites stay visible. inspect()
        # reads the live columns on either dialect (PRAGMA is SQLite-only); it
        # must run on this connection so it sees the CREATEs above, which have
        # not committed yet on a transactional-DDL dialect like Postgres.
        insp = inspect(c)
        for table in _TABLES:
            cols = {col["name"] for col in insp.get_columns(table)}
            if "user_id" not in cols:
                c.execute(
                    text(
                        f"ALTER TABLE {table} "
                        "ADD COLUMN user_id TEXT NOT NULL DEFAULT 'local'"
                    )
                )


_init()


# ---------------------------------------------------------------------------
# Crates
# ---------------------------------------------------------------------------

def create_crate(name: str) -> str:
    """Create a new crate with the given name. Returns the new crate_id."""
    crate_id = str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            text(
                "INSERT INTO crates (crate_id, name, created_at, user_id) "
                "VALUES (:id, :name, :ts, :user)"
            ),
            {"id": crate_id, "name": name.strip(), "ts": time.time(), "user": _acting_user()},
        )
        c.commit()
    return crate_id


def delete_crate(crate_id: str) -> bool:
    """Delete a crate and all its song memberships. Returns True if deleted."""
    with _conn() as c:
        cur = c.execute(
            text("DELETE FROM crates WHERE crate_id = :id AND user_id = :user"),
            {"id": crate_id, "user": _acting_user()},
        )
        c.commit()
        return cur.rowcount > 0


def list_crates() -> List[dict]:
    """Return the acting user's crates, newest first."""
    user = _acting_user()
    with _conn() as c:
        rows = c.execute(
            text(
                "SELECT crate_id, name, created_at FROM crates "
                "WHERE user_id = :user ORDER BY created_at DESC"
            ),
            {"user": user},
        ).fetchall()
        crates = [dict(r._mapping) for r in rows]
        # Annotate each crate with its song count
        for crate in crates:
            crate["song_count"] = c.execute(
                text(
                    "SELECT COUNT(*) FROM crate_songs "
                    "WHERE crate_id = :id AND user_id = :user"
                ),
                {"id": crate["crate_id"], "user": user},
            ).scalar()
    return crates


def rename_crate(crate_id: str, new_name: str) -> bool:
    """Rename a crate. Returns True if the crate was found and updated."""
    with _conn() as c:
        cur = c.execute(
            text("UPDATE crates SET name = :name WHERE crate_id = :id AND user_id = :user"),
            {"name": new_name.strip(), "id": crate_id, "user": _acting_user()},
        )
        c.commit()
        return cur.rowcount > 0


def add_to_crate(crate_id: str, song_name: str) -> bool:
    """Add a song to a crate. Returns False if already present."""
    try:
        with _conn() as c:
            c.execute(
                text(
                    "INSERT INTO crate_songs (crate_id, song_name, added_at, user_id) "
                    "VALUES (:id, :song, :ts, :user)"
                ),
                {"id": crate_id, "song": song_name, "ts": time.time(), "user": _acting_user()},
            )
            c.commit()
        return True
    except IntegrityError:
        return False


def remove_from_crate(crate_id: str, song_name: str) -> bool:
    """Remove a song from a crate. Returns True if it was present."""
    with _conn() as c:
        cur = c.execute(
            text(
                "DELETE FROM crate_songs "
                "WHERE crate_id = :id AND song_name = :song AND user_id = :user"
            ),
            {"id": crate_id, "song": song_name, "user": _acting_user()},
        )
        c.commit()
        return cur.rowcount > 0


def crate_songs(crate_id: str) -> List[str]:
    """Return the song names in a crate, ordered by when they were added."""
    with _conn() as c:
        rows = c.execute(
            text(
                "SELECT song_name FROM crate_songs "
                "WHERE crate_id = :id AND user_id = :user ORDER BY added_at"
            ),
            {"id": crate_id, "user": _acting_user()},
        ).fetchall()
    return [r.song_name for r in rows]


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def add_tag(song_name: str, tag: str) -> bool:
    """Add a tag to a song. Returns False if the tag already exists."""
    tag = tag.strip().lower()
    if not tag:
        return False
    try:
        with _conn() as c:
            c.execute(
                text(
                    "INSERT INTO song_tags (song_name, tag, created_at, user_id) "
                    "VALUES (:song, :tag, :ts, :user)"
                ),
                {"song": song_name, "tag": tag, "ts": time.time(), "user": _acting_user()},
            )
            c.commit()
        return True
    except IntegrityError:
        return False


def remove_tag(song_name: str, tag: str) -> bool:
    """Remove a tag from a song. Returns True if it was present."""
    with _conn() as c:
        cur = c.execute(
            text(
                "DELETE FROM song_tags "
                "WHERE song_name = :song AND tag = :tag AND user_id = :user"
            ),
            {"song": song_name, "tag": tag.strip().lower(), "user": _acting_user()},
        )
        c.commit()
        return cur.rowcount > 0


def song_tags(song_name: str) -> List[str]:
    """Return all tags for a song, alphabetically sorted."""
    with _conn() as c:
        rows = c.execute(
            text(
                "SELECT tag FROM song_tags "
                "WHERE song_name = :song AND user_id = :user ORDER BY tag"
            ),
            {"song": song_name, "user": _acting_user()},
        ).fetchall()
    return [r.tag for r in rows]


def songs_by_tag(tag: str) -> List[str]:
    """Return all songs that have a given tag."""
    with _conn() as c:
        rows = c.execute(
            text(
                "SELECT DISTINCT song_name FROM song_tags "
                "WHERE tag = :tag AND user_id = :user ORDER BY song_name"
            ),
            {"tag": tag.strip().lower(), "user": _acting_user()},
        ).fetchall()
    return [r.song_name for r in rows]


def all_tags() -> List[str]:
    """Return every distinct tag the acting user has applied."""
    with _conn() as c:
        rows = c.execute(
            text("SELECT DISTINCT tag FROM song_tags WHERE user_id = :user ORDER BY tag"),
            {"user": _acting_user()},
        ).fetchall()
    return [r.tag for r in rows]


# ---------------------------------------------------------------------------
# Favorites
# ---------------------------------------------------------------------------

def set_favorite(song_name: str, value: bool = True) -> None:
    """Mark or unmark a song as a favorite."""
    with _conn() as c:
        if value:
            # ON CONFLICT DO UPDATE is valid on both SQLite (3.24+) and Postgres;
            # INSERT OR REPLACE was SQLite-only.
            c.execute(
                text(
                    "INSERT INTO song_favorites (song_name, favorited_at, user_id) "
                    "VALUES (:song, :ts, :user) "
                    "ON CONFLICT(song_name) DO UPDATE SET "
                    "favorited_at = excluded.favorited_at, user_id = excluded.user_id"
                ),
                {"song": song_name, "ts": time.time(), "user": _acting_user()},
            )
        else:
            c.execute(
                text("DELETE FROM song_favorites WHERE song_name = :song AND user_id = :user"),
                {"song": song_name, "user": _acting_user()},
            )
        c.commit()


def is_favorite(song_name: str) -> bool:
    """Return True if the song is favorited."""
    with _conn() as c:
        row = c.execute(
            text("SELECT 1 FROM song_favorites WHERE song_name = :song AND user_id = :user"),
            {"song": song_name, "user": _acting_user()},
        ).fetchone()
    return row is not None


def list_favorites() -> List[str]:
    """Return all favorited song names, most recently favorited first."""
    with _conn() as c:
        rows = c.execute(
            text(
                "SELECT song_name FROM song_favorites "
                "WHERE user_id = :user ORDER BY favorited_at DESC"
            ),
            {"user": _acting_user()},
        ).fetchall()
    return [r.song_name for r in rows]


def purge_user_library(user_id: str) -> dict:
    """Delete all crates/tags/favorites for ``user_id``. Returns per-table counts."""
    counts = {"crates": 0, "crate_songs": 0, "song_tags": 0, "song_favorites": 0}
    with _conn() as c:
        for table in ("crate_songs", "crates", "song_tags", "song_favorites"):
            cur = c.execute(
                text(f"DELETE FROM {table} WHERE user_id = :user"),
                {"user": user_id},
            )
            counts[table] = cur.rowcount
        c.commit()
    return counts


def export_user_library(user_id: str) -> dict:
    """Snapshot crates + favorites + tags for account export."""
    with _conn() as c:
        crates = [
            dict(r._mapping)
            for r in c.execute(
                text(
                    "SELECT crate_id, name, created_at FROM crates "
                    "WHERE user_id = :user ORDER BY created_at DESC"
                ),
                {"user": user_id},
            ).fetchall()
        ]
        for crate in crates:
            songs = c.execute(
                text(
                    "SELECT song_name FROM crate_songs "
                    "WHERE crate_id = :id AND user_id = :user"
                ),
                {"id": crate["crate_id"], "user": user_id},
            ).fetchall()
            crate["songs"] = [r.song_name for r in songs]
        favorites = [
            r.song_name
            for r in c.execute(
                text(
                    "SELECT song_name FROM song_favorites "
                    "WHERE user_id = :user ORDER BY favorited_at DESC"
                ),
                {"user": user_id},
            ).fetchall()
        ]
        tags = [
            dict(r._mapping)
            for r in c.execute(
                text(
                    "SELECT song_name, tag FROM song_tags WHERE user_id = :user"
                ),
                {"user": user_id},
            ).fetchall()
        ]
    return {"crates": crates, "favorites": favorites, "tags": tags}
