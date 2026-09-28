"""
tests/test_account_lifecycle.py — account family (plan B2.7).

Written before the build: every test here asserts behaviour the code does
not have yet and must go red for that reason, not for an import error.

B2.7 Account lifecycle
    ``DELETE /account`` first calls ``job_store.cancel_user_jobs(uid)`` and
    then ``job_store.mark_purged(uid)`` before any other purge step, so a
    worker that finishes after the delete neither persists the row back nor
    writes output; purges ``audit.jsonl`` lines for the uid by atomic
    rewrite; continues past a failing step, collecting errors, instead of
    leaving a half purge; reports ``clean`` truthfully: true only when every
    step succeeded (including the Supabase purge when it is configured),
    otherwise false with a ``reason``.

    ``GET /account/export`` adds ``feedback.json`` and ``jobs.json``
    (metadata only, the caller's rows only) to the existing three entries.

Interface pinned by these tests (the spec fixes the behaviour, the tests
fix the names the build must expose on the DELETE body):
  * ``clean``: bool (already present).
  * ``reason``: non empty string, present whenever ``clean`` is false.
  * ``errors``: list, empty on a clean run, one entry per failed step.
  * ``status``: ``"partial"`` when at least one step failed and at least
    one succeeded.

Router tests mount the real router behind a two line middleware that sets
``current_user`` from an ``X-Test-User`` header (the same probe as
tests/test_tenancy_jobs.py: the contextvar reaches the sync handler thread
under TestClient), so u1 and u2 are real requests.
"""

from __future__ import annotations

import io
import json
import sqlite3
import threading
import zipfile
from typing import Callable

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed (required by TestClient)")

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from scripts.api import jobs as job_store
from scripts.api.auth import DEFAULT_USER, current_user
from scripts.api.routers import account
from scripts.api.schemas import JobStatus, JobType
from scripts.core import audit, crates, feedback_store
from scripts.core import paths as paths_mod

USER_HEADER = "X-Test-User"
U1 = "u1"
U2 = "u2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_user(monkeypatch):
    current_user.set(DEFAULT_USER)
    # Supabase is never configured here unless a test sets it explicitly, and
    # even then the network call is stubbed. Env only: config.local.yaml is
    # never read or written by this module.
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    yield
    current_user.set(DEFAULT_USER)


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Throwaway job, feedback, crate stores, outputs root and audit log."""
    from scripts.core import db as db_mod

    db_mod._engine_for.cache_clear()

    jobs_db = tmp_path / "jobs.db"
    monkeypatch.setattr(job_store, "_DB_PATH", str(jobs_db))
    monkeypatch.setattr(job_store, "_jobs", {})
    monkeypatch.setattr(job_store, "_cancelled", set())
    monkeypatch.setattr(job_store, "_queued", set())
    if hasattr(job_store, "_purged"):
        monkeypatch.setattr(job_store, "_purged", set())
    monkeypatch.setattr(job_store, "_sse_hook", None)
    job_store._init_db()

    monkeypatch.setattr(feedback_store, "_DB_PATH", tmp_path / "feedback.db")
    monkeypatch.setattr(feedback_store, "_events", [])
    feedback_store._init_db()

    monkeypatch.setattr(crates, "_DB_PATH", tmp_path / "remixmate.db")
    crates._init()

    out_root = tmp_path / "outputs"
    out_root.mkdir()
    monkeypatch.setattr(paths_mod, "OUTPUTS_DIR", out_root)

    audit_path = tmp_path / "data" / "audit.jsonl"
    audit_path.parent.mkdir(parents=True)
    monkeypatch.setattr(audit, "_AUDIT_PATH", audit_path)

    return {
        "jobs_db": jobs_db,
        "feedback_db": tmp_path / "feedback.db",
        "outputs": out_root,
        "audit": audit_path,
    }


class _CaptureExecutor:
    """Executor double that never runs anything: the test drives the wrapper."""

    def __init__(self) -> None:
        self.calls: list[Callable[[], None]] = []

    def submit(self, fn: Callable[[], None]) -> None:
        self.calls.append(fn)


@pytest.fixture
def capture_executor(monkeypatch):
    ex = _CaptureExecutor()
    monkeypatch.setattr(job_store, "_executor", ex)
    return ex


@pytest.fixture
def client():
    app = FastAPI()

    @app.middleware("http")
    async def _set_user(request: Request, call_next):
        current_user.set(request.headers.get(USER_HEADER, DEFAULT_USER))
        return await call_next(request)

    app.include_router(account.router)
    # raise_server_exceptions=False: an unhandled exception inside the handler
    # is the missing behaviour ("continue past a failing step"), so it must
    # surface as a 500 the assertion reads, not as a traceback out of the client.
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _as(uid: str) -> dict[str, str]:
    return {USER_HEADER: uid}


def _job_row(db_path, job_id: str) -> dict | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT status, result_json, meta_json FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "status": row[0],
        "result": json.loads(row[1] or "null"),
        "meta": json.loads(row[2] or "{}"),
    }


def _feedback_rows(db_path, uid: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM feedback_events WHERE user_id = ?", (uid,)
        ).fetchone()[0]
    finally:
        conn.close()


def _write_audit_lines(path, *entries: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _seed_user(uid: str, stores) -> dict:
    """One crate, one favorite, one feedback event, one done job and one
    manifest on disk, all owned by ``uid``. Returns the ids."""
    current_user.set(uid)
    crates.create_crate(f"Mix-{uid}")
    crates.set_favorite("Song A", True)
    ev = feedback_store.log_event("thumbs_up", "Song A", song_b="Song B")
    jid = job_store.create_job(JobType.DJ_REMIX, {"song_a": "Song A", "song_b": "Song B"})
    job_store.update_job(jid, status=JobStatus.DONE, result={"output": f"dj_{uid}/mix.wav"})
    sess = stores["outputs"] / uid / f"dj_{uid}"
    sess.mkdir(parents=True)
    (sess / "manifest.json").write_text(json.dumps({"credits": []}), encoding="utf-8")
    (sess / "mix.wav").write_bytes(b"RIFF")
    current_user.set(DEFAULT_USER)
    return {"event": ev, "job": jid}


# ---------------------------------------------------------------------------
# DELETE /account: running job
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_delete_with_running_job_leaves_no_row_and_no_files_after_task_completes(
    stores, capture_executor, client
):
    """A worker that finishes after DELETE /account must not write the row or
    its output back. Sequence: job RUNNING (real thread, blocked inside the
    task) -> DELETE -> task released -> nothing of the user remains."""
    started = threading.Event()
    release = threading.Event()

    def _task(job_id: str):
        out = paths_mod.user_outputs_dir() / "dj_live"
        out.mkdir(parents=True, exist_ok=True)
        (out / "partial.wav").write_bytes(b"RIFF")   # exists at delete time
        started.set()
        release.wait(5)
        # A cooperative task (B2.5: task_dj_chain polls is_cancelled between
        # songs) stops writing once its owner is gone.
        if job_store.is_cancelled(job_id):
            return None
        out.mkdir(parents=True, exist_ok=True)       # the purge removed the dir
        (out / "mix.wav").write_bytes(b"RIFF")       # would land after the purge
        return {"output": "dj_live/mix.wav"}

    current_user.set(U1)
    jid = job_store.create_job(JobType.DJ_REMIX, {"song_a": "A", "song_b": "B"})
    job_store.submit_job(jid, _task)
    current_user.set(DEFAULT_USER)
    (wrapper,) = capture_executor.calls

    worker = threading.Thread(target=wrapper, name="test-worker")
    worker.start()
    assert started.wait(5), "task never started"
    assert job_store._status_value(job_store.get_job(jid)["status"]) == "running"
    assert (stores["outputs"] / U1 / "dj_live" / "partial.wav").exists()

    res = client.delete("/account", headers=_as(U1))
    assert res.status_code == 200, res.text

    release.set()
    worker.join(5)
    assert not worker.is_alive(), "worker did not finish"

    row = _job_row(stores["jobs_db"], jid)
    assert row is None, f"job row came back after the task completed: {row!r}"
    assert job_store.get_job(jid) is None
    current_user.set(U1)
    assert job_store.list_jobs(user_id=U1) == []
    user_dir = stores["outputs"] / U1
    leftovers = sorted(str(p.relative_to(stores["outputs"])) for p in user_dir.rglob("*")) if user_dir.exists() else []
    assert leftovers == [], f"files left after the task completed: {leftovers}"


@pytest.mark.integration
def test_delete_calls_cancel_user_jobs_then_mark_purged_before_any_other_step(
    stores, client, monkeypatch
):
    """Order pin: cancel_user_jobs, then mark_purged, then everything else."""
    order: list[str] = []

    real_cancel = job_store.cancel_user_jobs
    real_mark = getattr(job_store, "mark_purged", None)
    real_purge_jobs = job_store.purge_user_jobs
    real_purge_lib = crates.purge_user_library
    real_purge_fb = feedback_store.purge_user_events
    real_rmtree = account.shutil.rmtree

    def _spy(name, fn):
        def _inner(*a, **k):
            order.append(name)
            return fn(*a, **k) if fn is not None else None
        return _inner

    monkeypatch.setattr(job_store, "cancel_user_jobs", _spy("cancel_user_jobs", real_cancel))
    monkeypatch.setattr(job_store, "mark_purged", _spy("mark_purged", real_mark), raising=False)
    monkeypatch.setattr(job_store, "purge_user_jobs", _spy("purge_user_jobs", real_purge_jobs))
    monkeypatch.setattr(crates, "purge_user_library", _spy("purge_user_library", real_purge_lib))
    monkeypatch.setattr(feedback_store, "purge_user_events", _spy("purge_user_events", real_purge_fb))
    monkeypatch.setattr(account.shutil, "rmtree", _spy("rmtree", real_rmtree))

    _seed_user(U1, stores)
    res = client.delete("/account", headers=_as(U1))
    assert res.status_code == 200, res.text

    assert order[:2] == ["cancel_user_jobs", "mark_purged"], f"call order was {order}"
    for step in ("purge_user_jobs", "purge_user_library", "purge_user_events", "rmtree"):
        assert step in order, f"{step} was not called; order was {order}"


# ---------------------------------------------------------------------------
# DELETE /account: audit purge
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_delete_purges_audit_lines_for_the_uid_and_keeps_other_users(stores, client):
    u2_line = {"ts": "t2", "epoch": 2.0, "action": "remix_complete", "resource": "x",
               "user": U2, "job_id": "j2", "meta": {}}
    _write_audit_lines(
        stores["audit"],
        {"ts": "t1", "epoch": 1.0, "action": "download_start", "resource": "a",
         "user": U1, "job_id": "j1", "meta": {}},
        u2_line,
        {"ts": "t3", "epoch": 3.0, "action": "library_delete", "resource": "b",
         "user": U1, "job_id": None, "meta": {"n": 1}},
    )
    before = stores["audit"].read_text(encoding="utf-8")
    assert before.count("\n") == 3   # control: the seed landed

    res = client.delete("/account", headers=_as(U1))
    assert res.status_code == 200, res.text

    assert stores["audit"].exists(), "audit.jsonl was removed instead of rewritten"
    lines = [ln for ln in stores["audit"].read_text(encoding="utf-8").splitlines() if ln.strip()]
    users = [json.loads(ln).get("user") for ln in lines]
    assert U1 not in users, f"audit lines for {U1} survived the delete: {lines}"
    assert users == [U2], f"other users' lines were not preserved verbatim: {lines}"
    assert json.loads(lines[0]) == u2_line
    # Atomic rewrite leaves no temp file beside the log.
    siblings = sorted(p.name for p in stores["audit"].parent.iterdir())
    assert siblings == ["audit.jsonl"], f"leftover files next to audit.jsonl: {siblings}"


@pytest.mark.integration
def test_delete_with_no_audit_file_is_still_clean(stores, client):
    """Empty case: no audit.jsonl on disk is not an error."""
    assert not stores["audit"].exists()
    res = client.delete("/account", headers=_as(U1))
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["errors"] == [], body
    assert body["clean"] is True, body


# ---------------------------------------------------------------------------
# DELETE /account: partial purge and clean reporting
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_crate_purge_error_still_purges_feedback_and_reports_partial(
    stores, client, monkeypatch
):
    ids = _seed_user(U1, stores)
    _write_audit_lines(stores["audit"], {"ts": "t", "epoch": 1.0, "action": "x",
                                         "resource": None, "user": U1, "job_id": None, "meta": {}})
    assert _feedback_rows(stores["feedback_db"], U1) == 1   # control: the seed landed

    def _boom(user_id: str):
        raise RuntimeError("crate store unavailable")

    monkeypatch.setattr(crates, "purge_user_library", _boom)

    res = client.delete("/account", headers=_as(U1))
    assert res.status_code == 200, (
        f"a failing crate purge must not abort the delete; got {res.status_code}: {res.text[:300]}"
    )
    body = res.json()

    # Steps after the failing one still ran.
    assert _feedback_rows(stores["feedback_db"], U1) == 0, "feedback rows survived the delete"
    current_user.set(U1)
    assert feedback_store.list_events(user_id=U1) == []
    assert _job_row(stores["jobs_db"], ids["job"]) is None, "job row survived the delete"
    assert not (stores["outputs"] / U1).exists() or not any((stores["outputs"] / U1).iterdir())
    if stores["audit"].exists():
        assert U1 not in stores["audit"].read_text(encoding="utf-8")

    # The report is truthful.
    assert body["clean"] is False, body
    assert body.get("status") == "partial", body
    assert isinstance(body.get("reason"), str) and body["reason"], body
    assert isinstance(body.get("errors"), list) and body["errors"], body
    assert any("crate" in str(e).lower() for e in body["errors"]), body["errors"]


@pytest.mark.integration
def test_supabase_purge_failure_makes_clean_false_with_reason(stores, client, monkeypatch):
    """Supabase configured (fake env, network stubbed) and the admin delete
    fails: every local step succeeds, yet clean must be false with a reason."""
    import urllib.error
    import urllib.request

    monkeypatch.setenv("SUPABASE_URL", "https://project.test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
    calls: list[str] = []

    def _urlopen(req, *a, **k):
        calls.append(getattr(req, "full_url", str(req)))
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    _seed_user(U1, stores)
    res = client.delete("/account", headers=_as(U1))
    assert res.status_code == 200, res.text
    body = res.json()

    assert calls and calls[0].startswith("https://project.test/auth/v1/admin/users/"), calls
    assert body["supabase_deleted"] is False
    # Local purge did succeed...
    current_user.set(U1)
    assert crates.list_crates() == []
    assert job_store.list_jobs(user_id=U1) == []
    # ...and the report still says not clean, because the account survives upstream.
    assert body["clean"] is False, body
    assert isinstance(body.get("reason"), str) and "supabase" in body["reason"].lower(), body


@pytest.mark.integration
def test_delete_with_every_step_succeeding_reports_clean_true_and_no_errors(stores, client):
    """Allow control for the deny cases above: Supabase unconfigured, every
    local step succeeds, so clean is true and errors is empty."""
    ids = _seed_user(U1, stores)
    _seed_user(U2, stores)

    res = client.delete("/account", headers=_as(U1))
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["errors"] == [], body
    assert body["clean"] is True, body
    assert body["supabase_deleted"] is False

    # u1 gone
    assert _job_row(stores["jobs_db"], ids["job"]) is None
    assert _feedback_rows(stores["feedback_db"], U1) == 0
    # u2 untouched
    assert _feedback_rows(stores["feedback_db"], U2) == 1
    current_user.set(U2)
    assert [c["name"] for c in crates.list_crates()] == [f"Mix-{U2}"]
    assert (stores["outputs"] / U2 / f"dj_{U2}" / "mix.wav").exists()


# ---------------------------------------------------------------------------
# GET /account/export
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_export_zip_lists_five_files_scoped_to_the_caller(stores, client):
    u1 = _seed_user(U1, stores)
    u2 = _seed_user(U2, stores)

    res = client.get("/account/export", headers=_as(U1))
    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith("application/zip")

    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        names = sorted(zf.namelist())
        assert names == sorted(
            ["library.json", "manifests.json", "meta.json", "feedback.json", "jobs.json"]
        ), names

        feedback = json.loads(zf.read("feedback.json"))
        jobs = json.loads(zf.read("jobs.json"))
        library = json.loads(zf.read("library.json"))
        manifests = json.loads(zf.read("manifests.json"))

    # feedback.json: the caller's events, nobody else's.
    assert isinstance(feedback, list)
    fb_ids = {e["id"] for e in feedback}
    assert u1["event"] in fb_ids, feedback
    assert u2["event"] not in fb_ids, feedback
    assert {e.get("user_id") for e in feedback} <= {U1}, feedback

    # jobs.json: metadata only, the caller's jobs, nobody else's.
    assert isinstance(jobs, list)
    job_ids = {j["job_id"] for j in jobs}
    assert u1["job"] in job_ids, jobs
    assert u2["job"] not in job_ids, jobs
    for j in jobs:
        assert {"job_id", "job_type", "status"} <= set(j), j
        assert j.get("meta", {}).get("user_id", U1) == U1, j
    # No file payloads inside the metadata (a manifest or audio would be bytes
    # or nested dicts far larger than a record). Records stay small.
    assert len(json.dumps(jobs)) < 4096, "jobs.json is not metadata only"

    # The three pre existing entries still carry the caller's data.
    assert [c["name"] for c in library["crates"]] == [f"Mix-{U1}"]
    assert [m["path"] for m in manifests] == [f"dj_{U1}/manifest.json"]


@pytest.mark.integration
def test_export_with_nothing_to_export_still_lists_five_files(stores, client):
    """Empty case: a fresh user still gets every entry, each an empty list."""
    res = client.get("/account/export", headers=_as("fresh-user"))
    assert res.status_code == 200, res.text
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        names = set(zf.namelist())
        assert {"feedback.json", "jobs.json"} <= names, sorted(names)
        assert json.loads(zf.read("feedback.json")) == []
        assert json.loads(zf.read("jobs.json")) == []
