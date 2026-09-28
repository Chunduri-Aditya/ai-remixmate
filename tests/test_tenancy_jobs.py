"""
tests/test_tenancy_jobs.py — jobs-sse family (plan B2.1, B2.2, B2.5 and the
job store hooks the account, admin and lifespan families build on).

Written before the build: every test here asserts behaviour the code does
not have yet and must go red for that reason, not for an import error.

B2.1 Job ownership
    GET and DELETE /jobs/{job_id} return 404 when ``job["meta"]["user_id"]``
    differs from ``get_user_id()``; POST /feedback/event applies the same
    check before backfilling ``context`` from a job.

B2.2 SSE scoping
    ``broadcaster.subscribe(user_id)`` records the owner per queue; a job
    payload reaches only the queue whose user matches ``payload["meta"]
    ["user_id"]``; heartbeats reach everyone and carry only ``active_jobs``
    (the caller's own count) and ``clients``; ``detect()`` runs once at
    startup off the event loop; a ``closing`` event ends every generator
    within 1 s; a full queue closes that subscriber's stream.

    Interface pinned by these tests (the spec fixes the behaviour, the tests
    fix the names the build must expose):
      * ``events.send_heartbeat()`` (or ``broadcaster.send_heartbeat()``):
        one heartbeat tick, awaited by ``_heartbeat_loop`` on each interval.
      * ``events.closing`` (or ``broadcaster.closing``): an ``asyncio.Event``.
      * ``_sse_generator(queue)`` keeps its one argument signature.

B2.5 Job cap and cancel
    ``_check_job_cap`` counts PENDING plus RUNNING; the ``_wrapper`` except
    branch keeps CANCELLED when the job was cancelled; ``job_store.
    is_cancelled(job_id)`` exists.

Hooks for later families
    ``job_store.cancel_user_jobs(uid)``, ``job_store.mark_purged(uid)`` (the
    wrapper skips persist and output for a purged user), ``job_store.
    get_job_from_db(job_id)``, ``job_store.mark_expired(session_id)``.

Router tests mount the real router behind a two line middleware that sets
``current_user`` from an ``X-Test-User`` header (probed: the contextvar does
reach the sync handler thread under TestClient), so u1 and u2 are real
requests, not direct handler calls.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sqlite3
import time
from typing import Any, Callable

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed (required by TestClient)")

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from scripts.api import jobs as job_store
from scripts.api.auth import DEFAULT_USER, current_user
from scripts.api.schemas import JobStatus, JobType

USER_HEADER = "X-Test-User"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_user():
    current_user.set(DEFAULT_USER)
    yield
    current_user.set(DEFAULT_USER)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Throwaway job store: fresh SQLite file, empty dict, empty cancel set,
    no SSE hook (another test may have registered one on the module)."""
    db = tmp_path / "jobs.db"
    monkeypatch.setattr(job_store, "_DB_PATH", str(db))
    monkeypatch.setattr(job_store, "_jobs", {})
    monkeypatch.setattr(job_store, "_cancelled", set())
    monkeypatch.setattr(job_store, "_sse_hook", None)
    job_store._init_db()
    return db


class _CaptureExecutor:
    """Executor double that never runs anything: jobs stay PENDING."""

    def __init__(self) -> None:
        self.calls: list[Callable[[], None]] = []

    def submit(self, fn: Callable[[], None]) -> None:
        self.calls.append(fn)


class _InlineExecutor:
    """Executor double that runs the wrapper synchronously in the caller."""

    def submit(self, fn: Callable[[], None]) -> None:
        fn()


@pytest.fixture
def capture_executor(monkeypatch):
    ex = _CaptureExecutor()
    monkeypatch.setattr(job_store, "_executor", ex)
    return ex


@pytest.fixture
def inline_executor(monkeypatch):
    ex = _InlineExecutor()
    monkeypatch.setattr(job_store, "_executor", ex)
    return ex


def _app_with_user_header(*routers) -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def _set_user(request: Request, call_next):
        current_user.set(request.headers.get(USER_HEADER, DEFAULT_USER))
        return await call_next(request)

    for r in routers:
        app.include_router(r)
    return app


def _db_row(db_path, job_id: str) -> dict | None:
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


def _status(job: dict) -> str:
    return job_store._status_value(job["status"])


# ---------------------------------------------------------------------------
# B2.1 Job ownership
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestJobOwnership:
    @pytest.fixture(autouse=True)
    def client(self, store):
        from scripts.api.routers.jobs import router

        current_user.set("u1")
        self.job_id = job_store.create_job(JobType.DJ_REMIX, {"song": "x"})
        current_user.set(DEFAULT_USER)
        assert job_store.get_job(self.job_id)["meta"]["user_id"] == "u1"

        app = _app_with_user_header(router)
        with TestClient(app, raise_server_exceptions=True) as c:
            self._client = c
            yield c

    def test_other_user_get_is_404(self):
        r = self._client.get(f"/jobs/{self.job_id}", headers={USER_HEADER: "u2"})
        assert r.status_code == 404, r.text

    def test_other_user_delete_is_404_and_does_not_cancel(self):
        r = self._client.delete(f"/jobs/{self.job_id}", headers={USER_HEADER: "u2"})
        assert r.status_code == 404, r.text
        # The deny must be a real deny: u1's job is still queued.
        assert _status(job_store.get_job(self.job_id)) == "pending"

    def test_owner_get_is_200(self):
        r = self._client.get(f"/jobs/{self.job_id}", headers={USER_HEADER: "u1"})
        assert r.status_code == 200, r.text
        assert r.json()["job_id"] == self.job_id

    def test_owner_delete_is_200(self):
        r = self._client.delete(f"/jobs/{self.job_id}", headers={USER_HEADER: "u1"})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "cancelled"
        assert _status(job_store.get_job(self.job_id)) == "cancelled"

    def test_unknown_job_is_404_for_owner_too(self):
        r = self._client.get("/jobs/does-not-exist", headers={USER_HEADER: "u1"})
        assert r.status_code == 404


@pytest.mark.integration
class TestFeedbackBackfillOwnership:
    """POST /feedback/event only backfills ``context`` from a job the caller owns."""

    @pytest.fixture(autouse=True)
    def client(self, store, monkeypatch):
        from scripts.api.routers import feedback as feedback_mod
        from scripts.core import feedback_store as fs

        current_user.set("u1")
        self.job_id = job_store.create_job(JobType.DJ_REMIX, {"transition_bars": 16, "preset": "club"})
        current_user.set(DEFAULT_USER)

        self.logged: list[dict[str, Any]] = []

        def _spy_log_event(**kwargs):
            self.logged.append(kwargs)
            return "e" * 36

        monkeypatch.setattr(fs, "log_event", _spy_log_event)
        # The router imported the module, not the function, so patching the
        # module attribute is enough. Confirm the seam before trusting it.
        assert feedback_mod.feedback_store is fs

        app = _app_with_user_header(feedback_mod.router)
        with TestClient(app, raise_server_exceptions=True) as c:
            self._client = c
            yield c

    def _post(self, user: str):
        return self._client.post(
            "/feedback/event",
            json={"event_type": "play_start", "song_a": "A", "song_b": "B", "job_id": self.job_id},
            headers={USER_HEADER: user},
        )

    def test_other_user_gets_no_context_from_the_job(self):
        r = self._post("u2")
        assert r.status_code == 200, r.text
        assert len(self.logged) == 1
        ctx = self.logged[0]["context"]
        assert not ctx, f"u2 must not see u1's job meta, got context={ctx!r}"

    def test_owner_gets_context_backfilled(self):
        r = self._post("u1")
        assert r.status_code == 200, r.text
        assert len(self.logged) == 1
        ctx = self.logged[0]["context"] or {}
        assert ctx.get("transition_bars") == 16
        assert ctx.get("preset") == "club"


# ---------------------------------------------------------------------------
# B2.2 SSE scoping
# ---------------------------------------------------------------------------

def _frame(item: Any) -> dict:
    return json.loads(item) if isinstance(item, str) else item


def _drain_now(q: asyncio.Queue) -> list[dict]:
    out = []
    while True:
        try:
            out.append(_frame(q.get_nowait()))
        except asyncio.QueueEmpty:
            return out


@pytest.fixture
def events(monkeypatch):
    """Fresh broadcaster singleton per test, and a fresh ``closing`` event
    wherever the build put it, so no asyncio primitive is bound to an
    earlier test's loop."""
    from scripts.api.routers import events as events_mod

    fresh = events_mod.SSEBroadcaster()
    monkeypatch.setattr(events_mod, "broadcaster", fresh)
    for holder in (events_mod, fresh):
        if isinstance(getattr(holder, "closing", None), asyncio.Event):
            monkeypatch.setattr(holder, "closing", asyncio.Event())
    return events_mod


async def _subscribe(events_mod, user_id: str) -> asyncio.Queue:
    try:
        return await events_mod.broadcaster.subscribe(user_id)
    except TypeError as exc:
        pytest.fail(f"broadcaster.subscribe(user_id) does not accept a user id: {exc}")


def _closing_event(events_mod) -> asyncio.Event:
    ev = getattr(events_mod, "closing", None) or getattr(events_mod.broadcaster, "closing", None)
    assert isinstance(ev, asyncio.Event), "no `closing` asyncio.Event on events module or broadcaster"
    return ev


def _send_heartbeat(events_mod):
    fn = getattr(events_mod, "send_heartbeat", None) or getattr(events_mod.broadcaster, "send_heartbeat", None)
    assert fn is not None, "no send_heartbeat() one shot on events module or broadcaster"
    return fn()


@pytest.mark.asyncio
async def test_job_payload_reaches_only_its_owner(events):
    q1 = await _subscribe(events, "u1")
    q2 = await _subscribe(events, "u2")

    await events.broadcaster.broadcast(
        "job_updated", {"job_id": "j1", "status": "RUNNING", "meta": {"user_id": "u1"}}
    )

    got1 = _drain_now(q1)
    got2 = _drain_now(q2)
    assert [f["data"]["job_id"] for f in got1] == ["j1"]
    assert got2 == [], f"u2 received u1's job payload: {got2!r}"


@pytest.mark.asyncio
async def test_payload_without_owner_reaches_everyone(events):
    """Allow side of the filter: library and system frames carry no meta and
    still fan out to every subscriber."""
    q1 = await _subscribe(events, "u1")
    q2 = await _subscribe(events, "u2")

    await events.broadcaster.broadcast("library_changed", {"ts": "now"})

    assert [f["type"] for f in _drain_now(q1)] == ["library_changed"]
    assert [f["type"] for f in _drain_now(q2)] == ["library_changed"]


@pytest.mark.asyncio
async def test_heartbeat_reaches_both_with_only_own_active_jobs_and_clients(events, store, capture_executor):
    # u1 has two queued jobs, u2 has none.
    current_user.set("u1")
    for _ in range(2):
        jid = job_store.create_job(JobType.ANALYZE, {})
        job_store.submit_job(jid, lambda job_id: None)
    current_user.set(DEFAULT_USER)

    q1 = await _subscribe(events, "u1")
    q2 = await _subscribe(events, "u2")

    await _send_heartbeat(events)

    hb1 = [f for f in _drain_now(q1) if f["type"] == "heartbeat"]
    hb2 = [f for f in _drain_now(q2) if f["type"] == "heartbeat"]
    assert len(hb1) == 1, hb1
    assert len(hb2) == 1, hb2

    d1, d2 = hb1[0]["data"], hb2[0]["data"]
    assert set(d1) == {"active_jobs", "clients"}, f"heartbeat carries extra fields: {sorted(d1)}"
    assert set(d2) == {"active_jobs", "clients"}, f"heartbeat carries extra fields: {sorted(d2)}"
    assert d1["active_jobs"] == 2
    assert d2["active_jobs"] == 0
    assert d1["clients"] == 2 and d2["clients"] == 2

    for raw in (json.dumps(hb1[0]), json.dumps(hb2[0])):
        low = raw.lower()
        for leak in ("hostname", "cpu", "ram_gb", "gpu", "machine_profile"):
            assert leak not in low, f"heartbeat leaks {leak!r}: {raw}"


def _on_event_loop_thread() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


@pytest.mark.integration
def test_detect_runs_once_at_startup_off_the_event_loop(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from scripts.api import main as main_mod
    from scripts.api.routers import events as events_mod
    from scripts.system import detect_machine

    # Throwaway executor so the lifespan shutdown does not close the shared one.
    monkeypatch.setattr(job_store, "_executor", ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(job_store, "_sse_hook", None)

    dummy_values = {str: "x", int: 1, float: 1.0, bool: False}
    kwargs = {}
    for f in dataclasses.fields(detect_machine.MachineProfile):
        t = f.type if isinstance(f.type, type) else str
        kwargs[f.name] = dummy_values.get(t, None)
    profile = detect_machine.MachineProfile(**kwargs)

    calls: list[bool] = []

    def _spy():
        calls.append(_on_event_loop_thread())
        return profile

    monkeypatch.setattr(detect_machine, "detect", _spy)
    for mod in (events_mod, main_mod):
        if hasattr(mod, "detect"):
            monkeypatch.setattr(mod, "detect", _spy)

    with TestClient(main_mod.app) as c:
        deadline = time.monotonic() + 2.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.05)
        c.get("/events/health")
        c.get("/events/health")

    assert len(calls) == 1, f"detect() called {len(calls)} times during startup (want exactly 1)"
    assert calls == [False], "detect() ran on the event loop thread"


async def _consume(gen, into: list[str]) -> None:
    async for c in gen:
        into.append(c)


async def _wait_closed(tasks: list[asyncio.Task], timeout: float, what: str) -> None:
    """Fail unless every consumer task finished within ``timeout``.

    Uses ``asyncio.wait`` rather than ``wait_for``: the generator under test
    catches CancelledError, so a ``wait_for`` timeout that cancels it returns
    cleanly on 3.12 and the check would pass with the stream still open.
    """
    _, pending = await asyncio.wait(tasks, timeout=timeout)
    if pending:
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        pytest.fail(f"{what}: {len(pending)} SSE generator(s) still open after {timeout}s")
    for t in tasks:
        t.result()   # surface any exception from the consumer


async def _run_generator(gen, timeout: float) -> list[str]:
    chunks: list[str] = []
    task = asyncio.create_task(_consume(gen, chunks))
    await _wait_closed([task], timeout, "single stream")
    return chunks


@pytest.mark.asyncio
async def test_closing_event_ends_every_generator_within_1s(events):
    q1 = await _subscribe(events, "u1")
    q2 = await _subscribe(events, "u2")
    closing = _closing_event(events)

    chunks1: list[str] = []
    chunks2: list[str] = []
    tasks = [
        asyncio.create_task(_consume(events._sse_generator(q1), chunks1)),
        asyncio.create_task(_consume(events._sse_generator(q2), chunks2)),
    ]
    await asyncio.sleep(0.05)   # let both generators reach their wait
    assert not any(t.done() for t in tasks), "generators ended before closing was set"

    closing.set()
    await _wait_closed(tasks, 1.0, "after closing.set()")
    assert chunks1 and chunks2, "generator yielded nothing (ack frame expected)"


@pytest.mark.asyncio
async def test_full_queue_drops_subscriber_and_closes_its_stream(events):
    slow = await _subscribe(events, "u1")
    healthy = await _subscribe(events, "u1")

    # Fill the slow client's queue to the brim before it consumes anything.
    while True:
        try:
            slow.put_nowait(json.dumps({"type": "job_updated", "data": {"filler": True}}))
        except asyncio.QueueFull:
            break
    assert slow.full()

    await events.broadcaster.broadcast(
        "job_updated", {"job_id": "j-overflow", "meta": {"user_id": "u1"}}
    )

    assert events.broadcaster.client_count == 1, "slow subscriber was not dropped"
    assert [f["data"]["job_id"] for f in _drain_now(healthy)] == ["j-overflow"]

    # The dropped client's generator must end (not sit on 25 s pings) once it
    # drains what it had.
    chunks = await _run_generator(events._sse_generator(slow), 1.0)
    assert any("filler" in c for c in chunks)


# ---------------------------------------------------------------------------
# B2.5 Job cap and cancel
# ---------------------------------------------------------------------------

def test_job_cap_counts_pending_plus_running(store, capture_executor):
    from fastapi import HTTPException

    from scripts.api.routers._helpers import _MAX_ACTIVE_JOBS, _check_job_cap

    assert _MAX_ACTIVE_JOBS == 4
    current_user.set("u1")
    for i in range(_MAX_ACTIVE_JOBS - 1):
        jid = job_store.create_job(JobType.ANALYZE, {"n": i})
        job_store.submit_job(jid, lambda job_id: None)
    assert all(_status(j) == "pending" for j in job_store.list_jobs(limit=10))

    _check_job_cap()   # 3 pending: allowed

    jid = job_store.create_job(JobType.ANALYZE, {"n": 3})
    job_store.submit_job(jid, lambda job_id: None)
    assert len(capture_executor.calls) == 4, "executor double did not capture every submit"

    with pytest.raises(HTTPException) as exc:
        _check_job_cap()   # 4 pending: the 5th submit is refused
    assert exc.value.status_code == 429


def test_finished_jobs_do_not_count_toward_the_cap(store, capture_executor):
    from scripts.api.routers._helpers import _MAX_ACTIVE_JOBS, _check_job_cap

    current_user.set("u1")
    for i in range(_MAX_ACTIVE_JOBS + 2):
        jid = job_store.create_job(JobType.ANALYZE, {"n": i})
        job_store.update_job(jid, status=JobStatus.DONE)
    jid = job_store.create_job(JobType.ANALYZE, {"n": "c"})
    job_store.cancel_job(jid)
    _check_job_cap()   # must not raise


def test_wrapper_keeps_cancelled_when_task_raises_after_cancel(store, inline_executor):
    current_user.set("u1")
    jid = job_store.create_job(JobType.ANALYZE, {})

    def _task(job_id: str):
        assert job_store.cancel_job(job_id) is True
        raise RuntimeError("worker noticed the cancel and bailed out")

    job_store.submit_job(jid, _task)

    job = job_store.get_job(jid)
    assert _status(job) == "cancelled", f"except branch overwrote CANCELLED with {job['status']!r}"
    assert _db_row(store, jid)["status"].split(".")[-1].lower() == "cancelled"


def test_is_cancelled_exists_and_tracks_cancel(store, capture_executor):
    assert hasattr(job_store, "is_cancelled"), "job_store.is_cancelled(job_id) is missing"
    current_user.set("u1")
    jid = job_store.create_job(JobType.DJ_CHAIN if hasattr(JobType, "DJ_CHAIN") else JobType.ANALYZE, {})
    job_store.submit_job(jid, lambda job_id: None)

    assert job_store.is_cancelled(jid) is False
    assert job_store.cancel_job(jid) is True
    assert job_store.is_cancelled(jid) is True
    assert job_store.is_cancelled("no-such-job") is False


# ---------------------------------------------------------------------------
# Hooks for later families
# ---------------------------------------------------------------------------

def test_cancel_user_jobs_cancels_only_that_users_active_jobs(store, capture_executor):
    assert hasattr(job_store, "cancel_user_jobs"), "job_store.cancel_user_jobs(uid) is missing"

    current_user.set("u1")
    pending = job_store.create_job(JobType.ANALYZE, {})
    job_store.submit_job(pending, lambda job_id: None)
    running = job_store.create_job(JobType.ANALYZE, {})
    job_store.update_job(running, status=JobStatus.RUNNING)
    done = job_store.create_job(JobType.ANALYZE, {})
    job_store.update_job(done, status=JobStatus.DONE)

    current_user.set("u2")
    other = job_store.create_job(JobType.ANALYZE, {})
    job_store.submit_job(other, lambda job_id: None)
    current_user.set(DEFAULT_USER)

    n = job_store.cancel_user_jobs("u1")

    assert n == 2
    assert _status(job_store.get_job(pending)) == "cancelled"
    assert _status(job_store.get_job(running)) == "cancelled"
    assert _status(job_store.get_job(done)) == "done"
    assert _status(job_store.get_job(other)) == "pending"


def test_mark_purged_makes_wrapper_skip_output_and_persist(store, capture_executor):
    assert hasattr(job_store, "mark_purged"), "job_store.mark_purged(uid) is missing"

    current_user.set("u1")
    jid = job_store.create_job(JobType.ANALYZE, {})
    ran: list[str] = []
    job_store.submit_job(jid, lambda job_id: ran.append(job_id) or {"output": "x.wav"})
    (wrapper,) = capture_executor.calls
    before = _db_row(store, jid)
    assert before["status"].split(".")[-1].lower() == "pending"

    job_store.mark_purged("u1")
    wrapper()   # the worker picks the job up after the purge

    assert ran == [], "task body ran for a purged user (would write output)"
    after = _db_row(store, jid)
    assert after is None or after["status"] == before["status"], (
        f"wrapper persisted a status change for a purged user: {after!r}"
    )


def test_mark_purged_mid_flight_skips_completion_persist(store, inline_executor):
    assert hasattr(job_store, "mark_purged"), "job_store.mark_purged(uid) is missing"

    current_user.set("u1")
    jid = job_store.create_job(JobType.ANALYZE, {})

    def _task(job_id: str):
        # DELETE /account lands while this job is running.
        job_store.mark_purged("u1")
        return {"output": "mix.wav"}

    job_store.submit_job(jid, _task)

    row = _db_row(store, jid)
    assert row is None or row["status"].split(".")[-1].lower() != "done", (
        f"completion was persisted for a purged user: {row!r}"
    )
    assert row is None or not (row["result"] or {}).get("output"), (
        f"result.output was persisted for a purged user: {row!r}"
    )


def test_get_job_from_db_reads_a_row_not_in_memory(store):
    assert hasattr(job_store, "get_job_from_db"), "job_store.get_job_from_db(job_id) is missing"

    current_user.set("u1")
    jid = job_store.create_job(JobType.DJ_REMIX, {"song": "x"})
    job_store.update_job(jid, status=JobStatus.DONE, result={"output": "m.wav"})
    job_store._jobs.clear()   # evicted from memory, row still on disk

    assert job_store.get_job(jid) is None
    row = job_store.get_job_from_db(jid)
    assert row is not None
    assert row["job_id"] == jid
    assert row["meta"]["user_id"] == "u1"
    assert row["result"] == {"output": "m.wav"}
    assert job_store.get_job_from_db("no-such-job") is None


def test_mark_expired_flags_every_row_of_that_session(store):
    assert hasattr(job_store, "mark_expired"), "job_store.mark_expired(session_id) is missing"

    current_user.set("u1")
    swept = job_store.create_job(JobType.DJ_REMIX, {})
    job_store.update_job(swept, status=JobStatus.DONE, result={
        "session_id": "s-swept", "output": "dj_s-swept/mix.wav",
        "stream_url": "/outputs/dj_s-swept/mix.wav",
    })
    kept = job_store.create_job(JobType.DJ_REMIX, {})
    job_store.update_job(kept, status=JobStatus.DONE, result={
        "session_id": "s-kept", "output": "dj_s-kept/mix.wav",
        "stream_url": "/outputs/dj_s-kept/mix.wav",
    })

    n = job_store.mark_expired("s-swept")

    assert n == 1
    assert job_store.get_job(swept)["result"]["expired"] is True
    assert _db_row(store, swept)["result"]["expired"] is True
    assert "expired" not in (job_store.get_job(kept)["result"] or {})
    assert "expired" not in (_db_row(store, kept)["result"] or {})
