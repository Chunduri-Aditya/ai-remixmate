"""
tests/test_user_scope.py — Phase 3 slice 3a: per-user output paths + job store.

Covers the four failure modes the deployment plan names for the contextvar
migration:

* allow  — a job created as ``u2`` writes to ``outputs/u2/`` and is visible to
           ``u2`` only.
* deny   — ``u1``'s jobs are invisible to ``u2``; the active-job cap counts
           per user.
* deny   — the ``/outputs`` serving route is anchored on the caller's own
           output root, so ``u2`` cannot read ``u1``'s mix, and a ``..`` in the
           session id is rejected.
* TRIPWIRE — a job submitted as ``u1`` must land under ``outputs/u1/``. This
           fails if ``jobs._wrapper`` stops re-``set``-ting ``current_user``
           (the ThreadPoolExecutor does not copy context), because the worker
           thread would fall back to the default ``"local"``.

Legacy job rows (written before this migration, no ``meta.user_id``) count as
the ``"local"`` user so single-user history stays visible.
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from scripts.api import jobs as job_store
from scripts.api.auth import DEFAULT_USER, current_user
from scripts.api.schemas import JobStatus, JobType


@pytest.fixture(autouse=True)
def _isolate_user():
    """Every test starts and ends as the default user."""
    current_user.set(DEFAULT_USER)
    yield
    current_user.set(DEFAULT_USER)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A throwaway job store: fresh SQLite file, empty in-memory dict.

    _DB_PATH is patched to a tmp file; scripts.core.db keys its SQLite engine on
    that path, so the code under test uses its own _get_conn unchanged.
    """
    db = tmp_path / "jobs.db"
    monkeypatch.setattr(job_store, "_DB_PATH", str(db))
    monkeypatch.setattr(job_store, "_jobs", {})
    job_store._init_db()
    return db


# ---------------------------------------------------------------------------
# allow / deny — the job store
# ---------------------------------------------------------------------------

def test_job_is_stamped_with_and_scoped_to_its_creator(store):
    current_user.set("u2")
    jid = job_store.create_job(JobType.DJ_REMIX, {"song": "x"})

    assert job_store.get_job(jid)["meta"]["user_id"] == "u2"

    # u2 sees it (scope defaults to the acting user) ...
    assert jid in {j["job_id"] for j in job_store.list_jobs(limit=100)}
    # ... u1 does not ...
    assert jid not in {j["job_id"] for j in job_store.list_jobs(limit=100, user_id="u1")}
    # ... and the unscoped operator view does.
    assert jid in {j["job_id"] for j in job_store.list_jobs(limit=100, user_id="*")}


def test_filtered_query_is_also_user_scoped(store):
    current_user.set("u1")
    mine = job_store.create_job(JobType.DJ_REMIX, {"song": "a"})
    job_store.update_job(mine, status=JobStatus.DONE)

    current_user.set("u2")
    theirs = job_store.create_job(JobType.DJ_REMIX, {"song": "b"})
    job_store.update_job(theirs, status=JobStatus.DONE)

    # u2 asking for completed remixes gets only its own, on the read-through path.
    got = {j["job_id"] for j in job_store.list_jobs(
        limit=100, job_type="dj_remix", status="done")}
    assert got == {theirs}


def test_active_job_cap_counts_per_user(store):
    from scripts.api.routers._helpers import _MAX_ACTIVE_JOBS, _check_job_cap

    current_user.set("u1")
    for i in range(_MAX_ACTIVE_JOBS):
        jid = job_store.create_job(JobType.ANALYZE, {"n": i})
        job_store.update_job(jid, status=JobStatus.RUNNING)

    # u1 is at the cap ...
    current_user.set("u1")
    with pytest.raises(Exception) as exc:
        _check_job_cap()
    assert "429" in str(exc.value) or "Too many" in str(exc.value)

    # ... u2 is not affected by u1's load.
    current_user.set("u2")
    _check_job_cap()  # must not raise


def test_legacy_rows_without_user_id_belong_to_local(store):
    conn = sqlite3.connect(str(store))
    conn.execute(
        "INSERT INTO jobs (job_id, status, job_type, created_at, progress, message,"
        " result_json, meta_json) VALUES (?,?,?,?,0.0,'',?,'{}')",
        ("legacy-1", "done", "dj_remix", 1000.0, json.dumps(None)),
    )
    conn.commit()
    conn.close()

    seen_local = {j["job_id"] for j in job_store.list_jobs(
        limit=100, job_type="dj_remix", status="done", user_id="local")}
    seen_u1 = {j["job_id"] for j in job_store.list_jobs(
        limit=100, job_type="dj_remix", status="done", user_id="u1")}

    assert "legacy-1" in seen_local
    assert "legacy-1" not in seen_u1


def test_running_jobs_roll_back_to_failed_on_restart(store):
    """A crashed process cannot leave a job RUNNING: _load_from_db rewrites any
    RUNNING/PENDING row to FAILED because the executor queue died with the old
    process. Guards scripts/api/jobs.py::_load_from_db across the SQLAlchemy
    swap — the row must come back FAILED both in memory and on disk.
    """
    from contextlib import closing

    from sqlalchemy import text

    with closing(job_store._get_conn()) as conn:
        conn.execute(
            text(
                "INSERT INTO jobs (job_id, status, job_type, created_at, progress,"
                " message, meta_json) VALUES (:i, 'running', 'dj_remix', :t, 0.4,"
                " 'Running', '{}')"
            ),
            {"i": "orphan", "t": time.time()},
        )
        conn.commit()

    job_store._jobs.clear()
    job_store._load_from_db()

    rolled = job_store.get_job("orphan")
    assert rolled is not None, "orphan row was not reloaded from the store"
    assert job_store._status_value(rolled["status"]) == "failed"
    assert "restart" in rolled["message"].lower()

    with closing(job_store._get_conn()) as conn:
        persisted = conn.execute(
            text("SELECT status FROM jobs WHERE job_id = 'orphan'")
        ).scalar()
    assert job_store._status_value(persisted) == "failed", persisted


# ---------------------------------------------------------------------------
# deny — the /outputs serving route
# ---------------------------------------------------------------------------

def test_serving_route_is_anchored_on_the_callers_own_output_root():
    from fastapi import HTTPException

    import scripts.core.paths as paths
    from scripts.api.routers.library import stream_output

    current_user.set("u1")
    d = paths.user_outputs_dir() / "sess1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "mix.wav").write_bytes(b"RIFF0000WAVE")

    # allow: the owner can read it
    current_user.set("u1")
    resp = stream_output("sess1", "mix.wav")
    assert getattr(resp, "path", getattr(resp, "media_type", None))  # a FileResponse

    # deny: another user resolves against outputs/u2/ and 404s
    current_user.set("u2")
    with pytest.raises(HTTPException) as exc:
        stream_output("sess1", "mix.wav")
    assert exc.value.status_code == 404

    # deny: a traversal component is rejected outright
    current_user.set("u1")
    with pytest.raises(HTTPException) as exc:
        stream_output("../u1", "mix.wav")
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# TRIPWIRE — the worker thread must re-establish the user context
# ---------------------------------------------------------------------------

def test_tripwire_job_artifact_lands_under_the_submitters_user_dir():
    import scripts.core.paths as paths
    from scripts.api.auth import get_user_id as _worker_user
    from scripts.api.jobs import create_job, get_job, submit_job

    captured: dict = {}

    def _task(job_id: str, **kwargs):
        # Runs in the ThreadPoolExecutor thread. If _wrapper does not re-set
        # current_user from the job meta, this reads the default "local".
        out = paths.user_outputs_dir()
        out.mkdir(parents=True, exist_ok=True)
        artifact = out / "mix.wav"
        artifact.write_bytes(b"RIFF0000WAVE")
        captured["artifact"] = artifact
        captured["worker_user"] = _worker_user()
        return {"out": str(artifact)}

    current_user.set("u1")
    jid = create_job(JobType.DJ_REMIX, {"song": "x"})
    submit_job(jid, _task)

    deadline = time.time() + 5
    while time.time() < deadline:
        if get_job(jid)["status"] in (JobStatus.DONE, JobStatus.FAILED):
            break
        time.sleep(0.05)

    assert get_job(jid)["status"] == JobStatus.DONE, get_job(jid).get("error")
    assert captured["worker_user"] == "u1"
    assert captured["artifact"].parent.name == "u1"
    assert captured["artifact"].parent.name not in ("local", "u2")


# ===========================================================================
# Phase 3 slice 3b — stem-split gate · audit · spectral freeze · feedback · crates
# ===========================================================================


def _row_conn(db) -> sqlite3.Connection:
    c = sqlite3.connect(str(db))
    c.row_factory = sqlite3.Row
    return c


# --- the /stems/split Demucs gate ------------------------------------------

@pytest.fixture
def stems_client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # the *aggregated* router — proves the gate is live in the real assembly,
    # not just on stems.router in isolation (routes.py must include it).
    from scripts.api.routes import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_stem_split_is_503_unless_enabled(stems_client, monkeypatch):
    """The two Demucs routes 503 by default; the FLAC-only compress routes
    stay open. Deletes the dependency → this goes green with the box exposed."""
    monkeypatch.delenv("REMIXMATE_STEM_SPLIT_ENABLED", raising=False)

    assert stems_client.post("/stems/split", json={"song": "x"}).status_code == 503
    assert stems_client.post("/stems/split-batch", json={}).status_code == 503

    # compress = FLAC encode, never gated — reaches its own handler (422 on the
    # missing ?song query param, not 503)
    assert stems_client.post("/stems/compress").status_code == 422
    assert stems_client.post("/stems/compress-batch").status_code != 503


@pytest.mark.parametrize("truthy", ["1", "true", "yes"])
def test_stem_split_enabled_reaches_handler(stems_client, monkeypatch, truthy):
    monkeypatch.setenv("REMIXMATE_STEM_SPLIT_ENABLED", truthy)
    # past the gate: now it fails on the unknown song / job cap, never 503
    r = stems_client.post("/stems/split", json={"song": "no-such-song"})
    assert r.status_code != 503


@pytest.mark.parametrize("falsy", ["", "0", "false", "off"])
def test_stem_split_flag_falsy_values_stay_disabled(stems_client, monkeypatch, falsy):
    monkeypatch.setenv("REMIXMATE_STEM_SPLIT_ENABLED", falsy)
    assert stems_client.post("/stems/split", json={"song": "x"}).status_code == 503


# --- audit.py stamps the acting user -------------------------------------------

def test_audit_stamps_the_acting_user(tmp_path, monkeypatch):
    from scripts.core import audit

    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "audit.jsonl")

    current_user.set("u7")
    audit.log_audit("remix_complete", resource="song-x")
    current_user.set(DEFAULT_USER)
    audit.log_audit("remix_complete", resource="song-y")
    audit.log_audit("op", resource="song-z", user="operator")  # explicit still wins

    users = {r["resource"]: r["user"] for r in audit.read_audit()}
    assert users == {"song-x": "u7", "song-y": "local", "song-z": "operator"}


# --- spectral_matching freeze in prod ----------------------------------------

@pytest.fixture
def spectral(tmp_path, monkeypatch):
    from scripts.core import spectral_matching as sm

    monkeypatch.setattr(sm, "_data_dir", lambda: tmp_path)
    return sm


def test_spectral_learning_frozen_is_a_noop(spectral, monkeypatch):
    """Non-vacuous: the render is recorded WHILE learning is on, so the history
    entry exists; the frozen apply_feedback must decline it *because of the
    flag*, not because there is nothing to train on. Drop the _learning_enabled
    guard in apply_feedback → it trains on sess-frozen and the model bytes move."""
    sm = spectral
    sims = [0.40 + 0.02 * i for i in range(sm.N_BANDS)]
    render = {"sims": sims, "entry_bar": 8, "score": 0.9}

    # learning ON — the allow side genuinely writes the model
    monkeypatch.setenv("REMIXMATE_SPECTRAL_LEARNING", "true")
    sm.record_render("sess-on", render)
    assert "sess-on" in json.loads(sm.history_path().read_text())
    before = sm.model_path().read_bytes() if sm.model_path().exists() else b""
    out = sm.apply_feedback("sess-on", liked=False)
    assert out["updated"] is True
    assert sm.model_path().read_bytes() != before

    # a render captured while ON, then the deployment freezes learning
    sm.record_render("sess-frozen", render)
    monkeypatch.setenv("REMIXMATE_SPECTRAL_LEARNING", "false")
    frozen_model = sm.model_path().read_bytes()
    frozen_history = sm.history_path().read_text()

    res = sm.apply_feedback("sess-frozen", liked=False)  # history for it EXISTS
    assert res["updated"] is False
    assert res["reason"] == "spectral learning frozen on this deployment"
    assert sm.model_path().read_bytes() == frozen_model

    # record_render is a no-op too — sess-3 never reaches the history file
    sm.record_render("sess-3", render)
    assert sm.history_path().read_text() == frozen_history


# --- feedback_store per-user events ------------------------------------------

@pytest.fixture
def fstore(tmp_path, monkeypatch):
    from scripts.core import feedback_store as fs

    monkeypatch.setattr(fs, "_DB_PATH", tmp_path / "feedback.db")
    monkeypatch.setattr(fs, "_events", [])
    fs._init_db()
    return fs


def test_feedback_events_are_user_scoped(fstore):
    fs = fstore

    current_user.set("u1")
    fs.log_event("thumbs_up", "A", "B")
    fs.log_event("skip", "A", "C")
    current_user.set("u2")
    fs.log_event("thumbs_down", "X", "Y")

    current_user.set("u1")
    assert len(fs.list_events()) == 2
    assert fs.summarize()["total_events"] == 2

    current_user.set("u2")
    assert {e["song_a"] for e in fs.list_events()} == {"X"}  # exactly its own, not empty
    assert fs.summarize()["total_events"] == 1

    assert len(fs.list_events(user_id="*")) == 3  # operator / offline analysis

    # persisted with the owner, not just filtered in memory
    from contextlib import closing

    from sqlalchemy import text

    with closing(fs._get_conn()) as conn:
        rows = conn.execute(
            text("SELECT song_a, user_id FROM feedback_events ORDER BY created_at")
        ).fetchall()
    assert [(r.song_a, r.user_id) for r in rows] == [
        ("A", "u1"), ("A", "u1"), ("X", "u2"),
    ]


def test_feedback_migration_adds_user_id_and_backfills(tmp_path, monkeypatch):
    """The P0 fixture always hands a fresh schema, so the migration branch never
    runs in the normal suite. Build the pre-Phase-3 table by hand and prove
    _init_db() adds the column and backfills existing rows to 'local'."""
    from scripts.core import feedback_store as fs

    db = tmp_path / "legacy_feedback.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE feedback_events ("
        " id TEXT PRIMARY KEY, event_type TEXT NOT NULL, song_a TEXT NOT NULL,"
        " song_b TEXT, job_id TEXT, position_sec REAL,"
        " context_json TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL)"
    )
    conn.execute(
        "INSERT INTO feedback_events (id, event_type, song_a, created_at)"
        " VALUES ('e1', 'skip', 'A', 1.0)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(fs, "_DB_PATH", db)
    fs._init_db()

    c = _row_conn(db)
    cols = {r[1] for r in c.execute("PRAGMA table_info(feedback_events)")}
    assert "user_id" in cols
    assert c.execute("SELECT user_id FROM feedback_events WHERE id='e1'").fetchone()[0] == "local"


# --- crates / tags / favorites per-user -------------------------------------

@pytest.fixture
def crate_store(tmp_path, monkeypatch):
    from scripts.core import crates

    monkeypatch.setattr(crates, "_DB_PATH", tmp_path / "remixmate.db")
    crates._init()
    return crates


def test_crates_tags_favorites_are_user_scoped(crate_store):
    c = crate_store

    current_user.set("u1")
    cid = c.create_crate("Bangers-u1")
    c.add_to_crate(cid, "song-1")
    c.add_tag("song-1", "peak")
    c.set_favorite("song-1", True)

    current_user.set("u2")
    assert c.list_crates() == []
    assert c.list_favorites() == []
    assert c.is_favorite("song-1") is False       # u1's favorite is invisible
    assert c.song_tags("song-1") == []
    assert c.songs_by_tag("peak") == []           # reverse lookup is scoped too
    assert c.all_tags() == []
    assert c.crate_songs(cid) == []               # cannot read into u1's crate
    assert c.delete_crate(cid) is False           # cannot delete it
    assert c.rename_crate(cid, "hijack") is False  # cannot rename it
    c.create_crate("Bangers-u2")
    assert {x["name"] for x in c.list_crates()} == {"Bangers-u2"}

    current_user.set("u1")
    assert {x["name"] for x in c.list_crates()} == {"Bangers-u1"}
    assert c.list_favorites() == ["song-1"]
    assert c.song_tags("song-1") == ["peak"]
    assert c.crate_songs(cid) == ["song-1"]


def test_crates_conflict_paths_return_false_not_raise(crate_store):
    """The IntegrityError catches in add_to_crate / add_tag and the ON CONFLICT
    in set_favorite are the dialect-sensitive spots of the SQLAlchemy swap
    (sqlite3.IntegrityError -> sqlalchemy.exc.IntegrityError; INSERT OR REPLACE
    -> ON CONFLICT DO UPDATE). A duplicate add must come back False; a repeat
    favorite must update in place, never raise.
    """
    c = crate_store
    current_user.set("u1")
    cid = c.create_crate("dupes")

    assert c.add_to_crate(cid, "s1") is True
    assert c.add_to_crate(cid, "s1") is False        # IntegrityError -> False
    assert c.crate_songs(cid) == ["s1"]

    assert c.add_tag("s1", "peak") is True
    assert c.add_tag("s1", "peak") is False
    assert c.song_tags("s1") == ["peak"]

    c.set_favorite("s1", True)
    c.set_favorite("s1", True)                        # ON CONFLICT DO UPDATE, no raise
    assert c.list_favorites() == ["s1"]
    c.set_favorite("s1", False)
    assert c.is_favorite("s1") is False


def test_crates_migration_adds_user_id_and_backfills(tmp_path, monkeypatch):
    from scripts.core import crates

    db = tmp_path / "legacy_remixmate.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE crates (crate_id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,"
        " created_at REAL NOT NULL);"
        "CREATE TABLE crate_songs (crate_id TEXT NOT NULL, song_name TEXT NOT NULL,"
        " added_at REAL NOT NULL, PRIMARY KEY (crate_id, song_name));"
        "CREATE TABLE song_tags (song_name TEXT NOT NULL, tag TEXT NOT NULL,"
        " created_at REAL NOT NULL, PRIMARY KEY (song_name, tag));"
        "CREATE TABLE song_favorites (song_name TEXT PRIMARY KEY, favorited_at REAL NOT NULL);"
    )
    conn.execute("INSERT INTO crates VALUES ('c1', 'old-crate', 1.0)")
    conn.execute("INSERT INTO song_favorites VALUES ('fav-song', 1.0)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(crates, "_DB_PATH", db)
    crates._init()

    current_user.set("local")
    assert {x["name"] for x in crates.list_crates()} == {"old-crate"}
    assert crates.list_favorites() == ["fav-song"]
    assert _row_conn(db).execute(
        "SELECT user_id FROM crates WHERE crate_id='c1'"
    ).fetchone()[0] == "local"


def test_crates_uniqueness_is_still_user_blind_phase4(crate_store):
    """Phase 3 adds the user_id column but NOT per-user uniqueness. These
    asserts pin that limitation: crates.name is globally UNIQUE and
    song_favorites' PK is (song_name) alone. Phase 4 must rebuild both keys to
    include user_id — when it does, flip these and tick DEPLOYMENT_PLAN P4."""
    c = crate_store

    with c._conn() as conn:
        indexes = conn.exec_driver_sql("PRAGMA index_list(crates)").fetchall()
        for row in indexes:
            idx_cols = [
                r[2] for r in conn.exec_driver_sql(f"PRAGMA index_info({row[1]})").fetchall()
            ]
            assert idx_cols not in (["user_id", "name"], ["name", "user_id"])

        fav_pk = [
            r[1]
            for r in conn.exec_driver_sql("PRAGMA table_info(song_favorites)").fetchall()
            if r[5]
        ]
    assert fav_pk == ["song_name"]
