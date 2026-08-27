"""
tests/test_job_list_filters.py — server-side filtering for GET /jobs.

Regression guard for the Mix Vault "Vault is empty" bug, which had two layers:

1. The page fetched the unfiltered job list and filtered client-side, but /jobs
   returns only the newest `limit` jobs. On a real store (~700 analyze/download
   rows) every completed remix fell outside that window.
2. Even filtering server-side was not enough, because the in-memory job dict only
   holds the newest 500 rows (see _load_from_db). A filtered lookup answered from
   memory still omits older jobs, so filtered queries read through to SQLite.

There is one extra wrinkle throughout: the store holds two serializations of the
same enums. Older rows persisted the repr ('JobType.DJ_REMIX', 'JobStatus.DONE');
newer rows hold the value ('dj_remix', 'done'). A raw string comparison matches
only the newer set and silently drops the rest.
"""

import json
import sqlite3

import pytest

from scripts.api import jobs as job_store

# How many rows the in-memory dict is allowed to hold, mirroring _load_from_db.
MEMORY_CAP = 500


def _write(conn, job_id, job_type, status, created_at):
    conn.execute(
        "INSERT INTO jobs (job_id, status, job_type, created_at, progress, message,"
        " result_json, meta_json) VALUES (?,?,?,?,0.0,'',?,'{}')",
        (job_id, status, job_type, created_at, json.dumps(None)),
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    """
    A store shaped like the real one: more rows on disk than the in-memory dict
    can hold, with completed remixes deliberately placed *below* the memory cap
    so that anything answering from memory alone cannot see them.

    Returns (expected_total_remixes, expected_visible_from_memory).
    """
    db = tmp_path / "jobs.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(job_store._DB_SCHEMA)

    # 10 completed remixes, oldest in the table (created_at 1000-1009):
    # 6 in the legacy repr form, 4 in the current value form.
    for i in range(6):
        _write(conn, f"legacy-{i}", "JobType.DJ_REMIX", "JobStatus.DONE", 1000 + i)
    for i in range(4):
        _write(conn, f"current-{i}", "dj_remix", "done", 1006 + i)

    # 600 newer unrelated jobs — more than the 500-row memory cap, so the
    # remixes above are pushed out of memory entirely.
    for i in range(600):
        _write(conn, f"noise-{i}", "analyze", "failed", 9000 + i)

    conn.commit()

    def _conn():
        c = sqlite3.connect(str(db))
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr(job_store, "_DB_PATH", str(db))
    monkeypatch.setattr(job_store, "_get_conn", _conn)

    # Populate the in-memory dict the way _load_from_db would: newest 500 only.
    rows = _conn().execute(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (MEMORY_CAP,)
    ).fetchall()
    mem = {}
    for row in rows:
        job = job_store._row_to_dict(row)
        job["status"] = job_store._status_value(job["status"])
        job["job_type"] = job_store._type_value(job["job_type"])
        mem[job["job_id"]] = job
    monkeypatch.setattr(job_store, "_jobs", mem)

    # Precondition: the fixture is only meaningful if memory really cannot see them.
    visible_in_memory = sum(
        1
        for j in mem.values()
        if j["job_type"] == "dj_remix" and j["status"] == "done"
    )
    assert visible_in_memory == 0, (
        "fixture is not exercising the bug — the remixes must sit below the memory cap"
    )
    return 10


def test_completed_remixes_found_below_the_memory_cap(store):
    """The bug itself: these 10 are unreachable from the in-memory dict."""
    got = job_store.list_jobs(limit=20, job_type="dj_remix", status="done")

    assert len(got) == 10, (
        "expected all 10 completed remixes; a lookup answered from the in-memory "
        "dict returns 0 because they sit below the 500-row load cap"
    )
    assert {j["job_id"] for j in got} == (
        {f"legacy-{i}" for i in range(6)} | {f"current-{i}" for i in range(4)}
    )


def test_both_enum_serializations_match(store):
    """A raw string comparison would return 4 of 10 and look plausible."""
    got = job_store.list_jobs(limit=100, job_type="dj_remix", status="done")
    ids = {j["job_id"] for j in got}

    assert len({i for i in ids if i.startswith("current-")}) == 4, (
        "value-serialized rows ('dj_remix'/'done') must match"
    )
    assert len({i for i in ids if i.startswith("legacy-")}) == 6, (
        "repr-serialized rows ('JobType.DJ_REMIX'/'JobStatus.DONE') must match too — "
        "this is the half a naive == comparison drops"
    )


def test_filters_accept_either_serialization_as_input(store):
    """Callers passing the repr form get the same result as the value form."""
    by_value = job_store.list_jobs(limit=100, job_type="dj_remix", status="done")
    by_repr = job_store.list_jobs(
        limit=100, job_type="JobType.DJ_REMIX", status="JobStatus.DONE"
    )
    assert {j["job_id"] for j in by_value} == {j["job_id"] for j in by_repr}


def test_limit_still_caps_the_filtered_set(store):
    """Filtering must not defeat the limit — newest-first, capped."""
    got = job_store.list_jobs(limit=3, job_type="dj_remix", status="done")
    assert len(got) == 3
    # created_at 1007-1009 are the newest remixes, so they win the cap.
    assert {j["job_id"] for j in got} == {"current-3", "current-2", "current-1"}


def test_filtering_on_only_one_axis_works(store):
    """job_type alone and status alone each filter independently."""
    assert len(job_store.list_jobs(limit=1000, job_type="analyze")) == 600
    assert len(job_store.list_jobs(limit=1000, status="done")) == 10


def test_nonmatching_filter_returns_empty(store):
    assert job_store.list_jobs(limit=1000, job_type="dj_chain", status="done") == []


# --- allow side: a filter mutated to refuse everything would pass a deny-only suite ---


def test_unfiltered_list_is_unchanged_and_still_memory_backed(store):
    """
    No filters means the original fast path: the in-memory dict, capped at 500.
    This is the allow test — a list_jobs that returned nothing would pass every
    assertion above that only checks for absence.
    """
    assert len(job_store.list_jobs(limit=10)) == 10
    assert len(job_store.list_jobs(limit=1000)) == MEMORY_CAP


def test_db_failure_falls_back_to_memory_rather_than_raising(store, monkeypatch):
    """A broken DB degrades to the memory view; it must not 500 the endpoint."""
    def _boom():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(job_store, "_get_conn", _boom)
    got = job_store.list_jobs(limit=1000, job_type="analyze", status="failed")
    # Memory holds 500 noise jobs, all analyze/failed.
    assert len(got) == MEMORY_CAP
