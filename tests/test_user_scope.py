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
    """A throwaway job store: fresh SQLite file, empty in-memory dict."""
    db = tmp_path / "jobs.db"
    conn = sqlite3.connect(str(db))
    conn.execute(job_store._DB_SCHEMA)
    conn.commit()
    conn.close()

    def _conn():
        c = sqlite3.connect(str(db))
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr(job_store, "_DB_PATH", str(db))
    monkeypatch.setattr(job_store, "_get_conn", _conn)
    monkeypatch.setattr(job_store, "_jobs", {})
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
