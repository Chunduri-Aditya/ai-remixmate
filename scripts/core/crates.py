"""
scripts/core/crates.py — Library crate and tag management.

A "crate" is a named group of songs — the DJ equivalent of a playlist.
Tags and favorites are per-song labels that persist across sessions.

All state lives in a SQLite database (data/remixmate.db).

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

import sqlite3
import time
import uuid
from pathlib import Path
from typing import List, Optional

from scripts.core.paths import DATA_DIR

_DB_PATH = DATA_DIR / "remixmate.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS crates (
    crate_id   TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS crate_songs (
    crate_id   TEXT NOT NULL REFERENCES crates(crate_id) ON DELETE CASCADE,
    song_name  TEXT NOT NULL,
    added_at   REAL NOT NULL,
    PRIMARY KEY (crate_id, song_name)
);

CREATE TABLE IF NOT EXISTS song_tags (
    song_name  TEXT NOT NULL,
    tag        TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (song_name, tag)
);

CREATE TABLE IF NOT EXISTS song_favorites (
    song_name    TEXT PRIMARY KEY,
    favorited_at REAL NOT NULL
);
"""


def _conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def _init() -> None:
    with _conn() as c:
        c.executescript(_SCHEMA)
        c.commit()


_init()


# ---------------------------------------------------------------------------
# Crates
# ---------------------------------------------------------------------------

def create_crate(name: str) -> str:
    """Create a new crate with the given name. Returns the new crate_id."""
    crate_id = str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            "INSERT INTO crates (crate_id, name, created_at) VALUES (?, ?, ?)",
            (crate_id, name.strip(), time.time()),
        )
        c.commit()
    return crate_id


def delete_crate(crate_id: str) -> bool:
    """Delete a crate and all its song memberships. Returns True if deleted."""
    with _conn() as c:
        cur = c.execute("DELETE FROM crates WHERE crate_id = ?", (crate_id,))
        c.commit()
        return cur.rowcount > 0


def list_crates() -> List[dict]:
    """Return all crates ordered by creation time (newest first)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT crate_id, name, created_at FROM crates ORDER BY created_at DESC"
        ).fetchall()
    crates = [dict(r) for r in rows]
    # Annotate each crate with its song count
    with _conn() as c:
        for crate in crates:
            count = c.execute(
                "SELECT COUNT(*) FROM crate_songs WHERE crate_id = ?",
                (crate["crate_id"],),
            ).fetchone()[0]
            crate["song_count"] = count
    return crates


def rename_crate(crate_id: str, new_name: str) -> bool:
    """Rename a crate. Returns True if the crate was found and updated."""
    with _conn() as c:
        cur = c.execute(
            "UPDATE crates SET name = ? WHERE crate_id = ?",
            (new_name.strip(), crate_id),
        )
        c.commit()
        return cur.rowcount > 0


def add_to_crate(crate_id: str, song_name: str) -> bool:
    """Add a song to a crate. Returns False if already present."""
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO crate_songs (crate_id, song_name, added_at) VALUES (?, ?, ?)",
                (crate_id, song_name, time.time()),
            )
            c.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def remove_from_crate(crate_id: str, song_name: str) -> bool:
    """Remove a song from a crate. Returns True if it was present."""
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM crate_songs WHERE crate_id = ? AND song_name = ?",
            (crate_id, song_name),
        )
        c.commit()
        return cur.rowcount > 0


def crate_songs(crate_id: str) -> List[str]:
    """Return the song names in a crate, ordered by when they were added."""
    with _conn() as c:
        rows = c.execute(
            "SELECT song_name FROM crate_songs WHERE crate_id = ? ORDER BY added_at",
            (crate_id,),
        ).fetchall()
    return [r["song_name"] for r in rows]


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
                "INSERT INTO song_tags (song_name, tag, created_at) VALUES (?, ?, ?)",
                (song_name, tag, time.time()),
            )
            c.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def remove_tag(song_name: str, tag: str) -> bool:
    """Remove a tag from a song. Returns True if it was present."""
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM song_tags WHERE song_name = ? AND tag = ?",
            (song_name, tag.strip().lower()),
        )
        c.commit()
        return cur.rowcount > 0


def song_tags(song_name: str) -> List[str]:
    """Return all tags for a song, alphabetically sorted."""
    with _conn() as c:
        rows = c.execute(
            "SELECT tag FROM song_tags WHERE song_name = ? ORDER BY tag",
            (song_name,),
        ).fetchall()
    return [r["tag"] for r in rows]


def songs_by_tag(tag: str) -> List[str]:
    """Return all songs that have a given tag."""
    with _conn() as c:
        rows = c.execute(
            "SELECT DISTINCT song_name FROM song_tags WHERE tag = ? ORDER BY song_name",
            (tag.strip().lower(),),
        ).fetchall()
    return [r["song_name"] for r in rows]


def all_tags() -> List[str]:
    """Return every distinct tag in use across the entire library."""
    with _conn() as c:
        rows = c.execute(
            "SELECT DISTINCT tag FROM song_tags ORDER BY tag"
        ).fetchall()
    return [r["tag"] for r in rows]


# ---------------------------------------------------------------------------
# Favorites
# ---------------------------------------------------------------------------

def set_favorite(song_name: str, value: bool = True) -> None:
    """Mark or unmark a song as a favorite."""
    with _conn() as c:
        if value:
            c.execute(
                "INSERT OR REPLACE INTO song_favorites (song_name, favorited_at) VALUES (?, ?)",
                (song_name, time.time()),
            )
        else:
            c.execute("DELETE FROM song_favorites WHERE song_name = ?", (song_name,))
        c.commit()


def is_favorite(song_name: str) -> bool:
    """Return True if the song is favorited."""
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM song_favorites WHERE song_name = ?", (song_name,)
        ).fetchone()
    return row is not None


def list_favorites() -> List[str]:
    """Return all favorited song names, most recently favorited first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT song_name FROM song_favorites ORDER BY favorited_at DESC"
        ).fetchall()
    return [r["song_name"] for r in rows]
