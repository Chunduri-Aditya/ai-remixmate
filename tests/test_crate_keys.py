"""
tests/test_crate_keys.py — B2.6: per user keys on crates, song_tags, song_favorites.

Phase 3 stamped every row with ``user_id`` but left the table keys user blind:
``crates.name UNIQUE``, ``song_tags PK (song_name, tag)``,
``song_favorites PK (song_name)``. B2.6 rebuilds them inside ``_init()``:

* crates          UNIQUE (user_id, name)
* song_tags       PK (user_id, song_name, tag)
* song_favorites  PK (user_id, song_name)

SQLite: create new table, copy, drop, rename. Postgres: drop and add the
constraint. The migration is idempotent, and an existing file with ``local``
rows keeps every row. The crate 409 stops echoing the driver's SQL.

The Postgres branch is not exercised here (no server in the test environment).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from scripts.api.auth import DEFAULT_USER, current_user

_TABLES = ("crates", "crate_songs", "song_tags", "song_favorites")

# Tokens a 409 body must not carry: anything that reads as the driver's message
# or the statement text (sqlite3 and psycopg wording, SQLAlchemy's "[SQL: ...]").
_SQL_LEAK_TOKENS = ("sql", "insert", "constraint", "sqlite", "psycopg", "integrityerror")


@pytest.fixture(autouse=True)
def _isolate_user():
    current_user.set(DEFAULT_USER)
    yield
    current_user.set(DEFAULT_USER)


@pytest.fixture
def crate_store(tmp_path, monkeypatch):
    """A throwaway crate store on a fresh SQLite file, initialised by _init()."""
    from scripts.core import crates

    monkeypatch.setattr(crates, "_DB_PATH", tmp_path / "remixmate.db")
    crates._init()
    return crates


# ---------------------------------------------------------------------------
# Schema helpers (raw sqlite3 on the file, independent of the module's engine)
# ---------------------------------------------------------------------------

def _keys(db_path: Path) -> dict:
    """Return {'unique': {table: {frozenset(cols), ...}}, 'pk': {table: [cols]},
    'tables': {names}} read straight from the SQLite file."""
    conn = sqlite3.connect(str(db_path))
    try:
        out: dict = {"unique": {}, "pk": {}, "tables": set()}
        out["tables"] = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }
        for table in _TABLES:
            uniques = set()
            for row in conn.execute(f"PRAGMA index_list({table})").fetchall():
                # (seq, name, unique, origin, partial)
                if not row[2]:
                    continue
                cols = [r[2] for r in conn.execute(f'PRAGMA index_info("{row[1]}")').fetchall()]
                uniques.add(frozenset(cols))
            out["unique"][table] = uniques
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            # (cid, name, type, notnull, dflt_value, pk) — pk is the 1-based position
            out["pk"][table] = [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])]
        return out
    finally:
        conn.close()


def _assert_new_keys(db_path: Path) -> None:
    keys = _keys(db_path)
    assert frozenset({"user_id", "name"}) in keys["unique"]["crates"], keys["unique"]["crates"]
    assert frozenset({"name"}) not in keys["unique"]["crates"], "crates.name is still globally UNIQUE"
    assert set(keys["pk"]["song_tags"]) == {"user_id", "song_name", "tag"}, keys["pk"]["song_tags"]
    assert set(keys["pk"]["song_favorites"]) == {"user_id", "song_name"}, keys["pk"]["song_favorites"]
    # create new, copy, drop, rename leaves no *_new / *_old table behind
    assert keys["tables"] == set(_TABLES), keys["tables"]


def _schema_sql(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
    finally:
        conn.close()


def _all_rows(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        return {t: sorted(conn.execute(f"SELECT * FROM {t}").fetchall()) for t in _TABLES}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Two users, same names: the inverse of the Phase 3 user blind pin
# ---------------------------------------------------------------------------

def test_two_users_each_own_a_crate_named_mix(crate_store):
    c = crate_store

    current_user.set("u1")
    mix_u1 = c.create_crate("Mix")

    current_user.set("u2")
    try:
        mix_u2 = c.create_crate("Mix")
    except IntegrityError:
        pytest.fail("crates.name is still user blind: u2 cannot own a crate named 'Mix'")

    assert mix_u1 != mix_u2
    assert [x["name"] for x in c.list_crates()] == ["Mix"]
    current_user.set("u1")
    assert [x["name"] for x in c.list_crates()] == ["Mix"]

    # deny side: one user still cannot hold two crates with the same name
    with pytest.raises(IntegrityError):
        c.create_crate("Mix")


def test_two_users_each_favorite_the_same_song_without_clobbering(crate_store):
    c = crate_store

    current_user.set("u1")
    c.set_favorite("song-x", True)
    current_user.set("u2")
    c.set_favorite("song-x", True)

    # u2's upsert must not have moved u1's row to u2
    current_user.set("u1")
    assert c.is_favorite("song-x") is True, "u2's favorite overwrote u1's row"
    assert c.list_favorites() == ["song-x"]
    current_user.set("u2")
    assert c.is_favorite("song-x") is True
    assert c.list_favorites() == ["song-x"]

    # un-favoriting as u2 leaves u1's favorite alone
    c.set_favorite("song-x", False)
    assert c.is_favorite("song-x") is False
    current_user.set("u1")
    assert c.is_favorite("song-x") is True


def test_two_users_each_tag_the_same_song_alike(crate_store):
    c = crate_store

    current_user.set("u1")
    assert c.add_tag("song-x", "peak") is True
    current_user.set("u2")
    assert c.add_tag("song-x", "peak") is True, "song_tags PK is still user blind"

    assert c.song_tags("song-x") == ["peak"]
    assert c.songs_by_tag("peak") == ["song-x"]
    current_user.set("u1")
    assert c.song_tags("song-x") == ["peak"]

    # deny side: a repeat within one user is still a False, not a raise
    assert c.add_tag("song-x", "peak") is False

    current_user.set("u2")
    assert c.remove_tag("song-x", "peak") is True
    current_user.set("u1")
    assert c.song_tags("song-x") == ["peak"]


# ---------------------------------------------------------------------------
# Schema: the keys themselves, and the migration is idempotent
# ---------------------------------------------------------------------------

def test_fresh_schema_carries_per_user_keys(crate_store, tmp_path):
    _assert_new_keys(tmp_path / "remixmate.db")


def test_init_is_idempotent(crate_store, tmp_path):
    """Running _init() again on an already migrated file changes neither the
    schema text nor a single row, and leaves no scratch table behind."""
    c = crate_store
    db = tmp_path / "remixmate.db"

    current_user.set("u1")
    cid = c.create_crate("Mix")
    c.add_to_crate(cid, "s1")
    c.add_tag("s1", "peak")
    c.set_favorite("s1", True)
    current_user.set("u2")
    try:
        c.create_crate("Mix")
    except IntegrityError:
        pytest.fail("crates.name is still user blind: u2 cannot own a crate named 'Mix'")
    c.set_favorite("s1", True)

    schema_before = _schema_sql(db)
    rows_before = _all_rows(db)

    c._init()
    c._init()

    assert _schema_sql(db) == schema_before
    assert _all_rows(db) == rows_before
    _assert_new_keys(db)


# ---------------------------------------------------------------------------
# Migration: an old file built by hand, with local rows, survives
# ---------------------------------------------------------------------------

_PHASE3_DDL = """
CREATE TABLE crates (
    crate_id   TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    user_id    TEXT NOT NULL DEFAULT 'local'
);
CREATE TABLE crate_songs (
    crate_id   TEXT NOT NULL REFERENCES crates(crate_id) ON DELETE CASCADE,
    song_name  TEXT NOT NULL,
    added_at   REAL NOT NULL,
    user_id    TEXT NOT NULL DEFAULT 'local',
    PRIMARY KEY (crate_id, song_name)
);
CREATE TABLE song_tags (
    song_name  TEXT NOT NULL,
    tag        TEXT NOT NULL,
    created_at REAL NOT NULL,
    user_id    TEXT NOT NULL DEFAULT 'local',
    PRIMARY KEY (song_name, tag)
);
CREATE TABLE song_favorites (
    song_name    TEXT PRIMARY KEY,
    favorited_at REAL NOT NULL,
    user_id      TEXT NOT NULL DEFAULT 'local'
);
"""

_PRE_PHASE3_DDL = """
CREATE TABLE crates (crate_id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL);
CREATE TABLE crate_songs (crate_id TEXT NOT NULL REFERENCES crates(crate_id) ON DELETE CASCADE,
    song_name TEXT NOT NULL, added_at REAL NOT NULL, PRIMARY KEY (crate_id, song_name));
CREATE TABLE song_tags (song_name TEXT NOT NULL, tag TEXT NOT NULL,
    created_at REAL NOT NULL, PRIMARY KEY (song_name, tag));
CREATE TABLE song_favorites (song_name TEXT PRIMARY KEY, favorited_at REAL NOT NULL);
"""


def _build_legacy(db: Path, ddl: str, with_user_id: bool) -> None:
    conn = sqlite3.connect(str(db))
    conn.executescript(ddl)
    if with_user_id:
        conn.execute("INSERT INTO crates VALUES ('c1', 'old-crate', 1.0, 'local')")
        conn.execute("INSERT INTO crates VALUES ('c2', 'theirs', 2.0, 'u9')")
        conn.execute("INSERT INTO crate_songs VALUES ('c1', 's1', 1.0, 'local')")
        conn.execute("INSERT INTO crate_songs VALUES ('c1', 's2', 2.0, 'local')")
        conn.execute("INSERT INTO song_tags VALUES ('s1', 'peak', 1.0, 'local')")
        conn.execute("INSERT INTO song_favorites VALUES ('fav-song', 1.0, 'local')")
        conn.execute("INSERT INTO song_favorites VALUES ('other', 2.0, 'u9')")
    else:
        conn.execute("INSERT INTO crates VALUES ('c1', 'old-crate', 1.0)")
        conn.execute("INSERT INTO crate_songs VALUES ('c1', 's1', 1.0)")
        conn.execute("INSERT INTO crate_songs VALUES ('c1', 's2', 2.0)")
        conn.execute("INSERT INTO song_tags VALUES ('s1', 'peak', 1.0)")
        conn.execute("INSERT INTO song_favorites VALUES ('fav-song', 1.0)")
    conn.commit()
    conn.close()


def test_migration_from_phase3_schema_keeps_local_rows(tmp_path, monkeypatch):
    """The Phase 3 shape (user_id present, keys user blind) with local rows in
    all four tables, plus one other user's rows. After _init(): every row is
    still there under its owner, the keys are per user, the crate_songs
    cascade still works, and a second user can now take the name 'old-crate'.
    crate_songs is the trap: DROP TABLE crates with foreign keys on cascades
    into it."""
    from scripts.core import crates

    db = tmp_path / "legacy_remixmate.db"
    _build_legacy(db, _PHASE3_DDL, with_user_id=True)

    monkeypatch.setattr(crates, "_DB_PATH", db)
    crates._init()

    _assert_new_keys(db)

    current_user.set("local")
    assert {x["name"] for x in crates.list_crates()} == {"old-crate"}
    assert crates.crate_songs("c1") == ["s1", "s2"]
    assert crates.song_tags("s1") == ["peak"]
    assert crates.list_favorites() == ["fav-song"]

    current_user.set("u9")
    assert {x["name"] for x in crates.list_crates()} == {"theirs"}
    assert crates.list_favorites() == ["other"]
    # the copy kept the original values, not just the row count
    conn = sqlite3.connect(str(db))
    assert conn.execute("SELECT created_at FROM crates WHERE crate_id='c1'").fetchone()[0] == 1.0
    conn.close()

    # the migrated file behaves like a fresh one
    crates.create_crate("old-crate")
    assert {x["name"] for x in crates.list_crates()} == {"theirs", "old-crate"}
    crates.set_favorite("fav-song", True)
    current_user.set("local")
    assert crates.is_favorite("fav-song") is True

    # ON DELETE CASCADE survived the rebuild
    assert crates.delete_crate("c1") is True
    conn = sqlite3.connect(str(db))
    assert conn.execute("SELECT COUNT(*) FROM crate_songs WHERE crate_id='c1'").fetchone()[0] == 0
    conn.close()


def test_migration_from_pre_phase3_schema_backfills_and_rekeys(tmp_path, monkeypatch):
    """A file older than Phase 3 (no user_id anywhere) goes through both
    steps in one _init(): ADD COLUMN backfills 'local', then the rebuild puts
    user_id into the keys. Rows in all four tables survive."""
    from scripts.core import crates

    db = tmp_path / "ancient_remixmate.db"
    _build_legacy(db, _PRE_PHASE3_DDL, with_user_id=False)

    monkeypatch.setattr(crates, "_DB_PATH", db)
    crates._init()

    _assert_new_keys(db)

    conn = sqlite3.connect(str(db))
    owners = {
        t: {r[0] for r in conn.execute(f"SELECT DISTINCT user_id FROM {t}").fetchall()}
        for t in _TABLES
    }
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in _TABLES}
    conn.close()
    assert owners == {t: {"local"} for t in _TABLES}, owners
    assert counts == {"crates": 1, "crate_songs": 2, "song_tags": 1, "song_favorites": 1}, counts

    current_user.set("local")
    assert crates.crate_songs("c1") == ["s1", "s2"]
    assert crates.list_favorites() == ["fav-song"]


# ---------------------------------------------------------------------------
# Route: the 409 no longer echoes SQL
# ---------------------------------------------------------------------------

@pytest.fixture
def client(crate_store):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from scripts.api.routers.crates import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_crate_409_does_not_echo_sql(client):
    assert client.post("/crates", json={"name": "Mix"}).status_code == 201

    r = client.post("/crates", json={"name": "Mix"})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    lowered = detail.lower()
    leaked = [tok for tok in _SQL_LEAK_TOKENS if tok in lowered]
    assert not leaked, f"409 detail echoes the driver or SQL ({leaked}): {detail!r}"
