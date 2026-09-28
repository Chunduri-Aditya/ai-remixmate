"""
tests/test_lifespan.py — Round C "lifespan" family (spec B2.13 call, B2.15, B2.16, B2.17).

Defines success for:

* B2.13 call   — ``legal_store.init_schema()`` runs once at lifespan startup.
* B2.15 sweep  — a background task sweeps every ``outputs/<uid>/`` after a
                 start delay (``scripts.api.main.SWEEP_START_DELAY_SEC``, default
                 60) and then every ``cfg.retention.sweep_interval_hours``
                 (default 6, new key in config.yaml and RetentionConfig); a swept
                 session marks its job rows ``result.expired = true`` via
                 ``job_store.mark_expired``; previews also sweep the acting user.
* B2.16 ready  — ``/health/ready`` pings each store engine with ``SELECT 1``,
                 reports ``catalog_songs`` and returns 503 ``reason: catalog_empty``
                 when the count is 0 and ``REMIXMATE_REQUIRE_CATALOG=1``.
* B2.17 close  — lifespan shutdown sets the SSE ``closing`` event before it
                 drains the executor; an open stream ends and shutdown finishes
                 under 2 s.

Every lifespan run here swaps ``job_store._executor`` for a fake so the real
pool survives for the rest of the suite, and stubs the heartbeat so no task
leaks across event loops. OUTPUTS_DIR and DATA_DIR are already the throwaway
tree conftest points them at; each test uses its own uid so nothing collides.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.api import jobs as job_store
from scripts.api import main as main_mod
from scripts.api.auth import DEFAULT_USER, current_user
from scripts.api.routers import events as events_mod
from scripts.api.routers import system as system_mod
from scripts.api.schemas import JobStatus
from scripts.core import crates as crates_mod
from scripts.core import db as db_mod
from scripts.core import feedback_store as feedback_mod
from scripts.core import legal_store as legal_mod
from scripts.core import mix_retention
from scripts.core import paths as paths_mod
from scripts.core.config import RetentionConfig, cfg

ROOT = Path(__file__).resolve().parents[1]
_DAY = 86_400


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mk_old_session(root: Path, name: str, *, age_days: float = 99.0) -> Path:
    """A rendered session dir whose newest mtime is ``age_days`` old (TTL bait)."""
    d = root / name
    d.mkdir(parents=True)
    (d / "mix.wav").write_bytes(b"RIFF")
    old = time.time() - age_days * _DAY
    for p in (d / "mix.wav", d):
        os.utime(p, (old, old))
    return d


def _seed_done_job(owner: str, session_id: str) -> str:
    """Persist a DONE job (memory + DB) whose result carries ``session_id``."""
    jid = str(uuid.uuid4())
    job = {
        "job_id":      jid,
        "status":      JobStatus.DONE,
        "job_type":    "dj_remix",
        "created_at":  time.time(),
        "started_at":  None,
        "finished_at": time.time(),
        "progress":    1.0,
        "message":     "Done",
        "result":      {"session_id": session_id, "stream_url": f"/outputs/{session_id}/mix.wav"},
        "error":       None,
        "eta_sec":     None,
        "meta":        {"user_id": owner, "session_id": session_id},
    }
    with job_store._lock:
        job_store._jobs[jid] = job
        with closing(job_store._get_conn()) as conn:
            job_store._upsert_row(conn, job)
    return jid


class _FakeExecutor:
    """Stands in for the ThreadPoolExecutor during a lifespan run.

    Records whether the SSE ``closing`` event was already set when the lifespan
    asked the executor to drain (B2.17 ordering).
    """

    def __init__(self) -> None:
        self.shutdown_calls: list[tuple] = []
        self.closing_set_at_shutdown: bool | None = None

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.closing_set_at_shutdown = events_mod.closing.is_set()
        self.shutdown_calls.append((wait, cancel_futures))

    def submit(self, *a, **k):  # pragma: no cover - guard
        raise AssertionError("lifespan tests never submit jobs")


async def _until(pred, timeout: float, step: float = 0.05) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if pred():
            return True
        await asyncio.sleep(step)
    return pred()


def _ready_reason(body) -> str:
    """The failure reason a 503 readiness body carries, whichever shape it uses."""
    if isinstance(body, dict):
        if isinstance(body.get("reason"), str):
            return body["reason"]
        detail = body.get("detail")
        if isinstance(detail, dict) and isinstance(detail.get("reason"), str):
            return detail["reason"]
        if isinstance(detail, str):
            return detail
    return ""


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Fresh job store per test; DATABASE_URL unset; auth context reset."""
    monkeypatch.setattr(job_store, "_DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(legal_mod, "_DB_PATH", tmp_path / "legal.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REMIXMATE_REQUIRE_CATALOG", raising=False)
    db_mod._engine_for.cache_clear()
    job_store._jobs.clear()
    job_store._init_db()
    current_user.set(DEFAULT_USER)
    events_mod.closing.clear()
    yield
    events_mod.closing.clear()
    job_store._jobs.clear()
    current_user.set(DEFAULT_USER)
    db_mod._engine_for.cache_clear()


@pytest.fixture
def fake_executor(monkeypatch):
    ex = _FakeExecutor()
    monkeypatch.setattr(job_store, "_executor", ex)
    # The heartbeat is not under test; keep it off so no task outlives the loop.
    monkeypatch.setattr(events_mod, "start_heartbeat", lambda: None)
    return ex


# ---------------------------------------------------------------------------
# B2.15 — config key
# ---------------------------------------------------------------------------

class TestSweepIntervalConfig:
    def test_dataclass_default_is_6_hours(self):
        assert getattr(RetentionConfig(), "sweep_interval_hours", None) == 6

    def test_config_yaml_carries_the_key(self):
        data = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        assert data["retention"].get("sweep_interval_hours") == 6

    def test_start_delay_default_is_60_seconds(self):
        assert getattr(main_mod, "SWEEP_START_DELAY_SEC", None) == 60


# ---------------------------------------------------------------------------
# B2.13 call — legal schema at startup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lifespan_calls_legal_init_schema_once(monkeypatch, fake_executor):
    calls: list[int] = []
    monkeypatch.setattr(legal_mod, "init_schema", lambda: calls.append(1))

    async with main_mod.lifespan(main_mod.app):
        assert calls == [1], "init_schema must run during startup, before yield"
    assert calls == [1]


# ---------------------------------------------------------------------------
# B2.15 — per user sweep marks the job; previews sweep too
# ---------------------------------------------------------------------------

class TestSweepMarksJob:
    def test_sweep_current_user_marks_swept_session_expired(self):
        uid = f"lifespan_{uuid.uuid4().hex[:8]}"
        sid = "dj_stale"
        user_root = paths_mod.OUTPUTS_DIR / uid
        _mk_old_session(user_root, sid)
        jid = _seed_done_job(uid, sid)
        # control: a job for a session that is NOT swept stays unexpired
        _mk_old_session(user_root, "dj_fresh", age_days=0.0)
        keep_jid = _seed_done_job(uid, "dj_fresh")

        token = current_user.set(uid)
        try:
            res = mix_retention.sweep_current_user(dry_run=False)
        finally:
            current_user.reset(token)

        assert res.removed == [sid]
        assert not (user_root / sid).exists()
        assert job_store.get_job(jid)["result"].get("expired") is True
        assert job_store.get_job_from_db(jid)["result"].get("expired") is True
        assert job_store.get_job(keep_jid)["result"].get("expired") is not True

    def test_dry_run_sweep_marks_nothing(self):
        uid = f"lifespan_{uuid.uuid4().hex[:8]}"
        sid = "dj_stale"
        _mk_old_session(paths_mod.OUTPUTS_DIR / uid, sid)
        jid = _seed_done_job(uid, sid)
        token = current_user.set(uid)
        try:
            res = mix_retention.sweep_current_user(dry_run=True)
        finally:
            current_user.reset(token)
        assert res.removed == [sid] and (paths_mod.OUTPUTS_DIR / uid / sid).exists()
        assert job_store.get_job(jid)["result"].get("expired") is not True

    def test_preview_task_sweeps_the_acting_user(self, monkeypatch):
        """previews also trigger the per user sweep: a stale session dir of the
        acting user is gone once task_remix_preview returns."""
        from scripts.api.task_modules import remix as remix_mod

        uid = f"lifespan_{uuid.uuid4().hex[:8]}"
        user_root = paths_mod.OUTPUTS_DIR / uid
        _mk_old_session(user_root, "dj_stale")
        monkeypatch.setattr(cfg.retention, "sweep_after_render", True, raising=False)

        _sr = 8000
        struct = SimpleNamespace(bpm=120.0, camelot="8A")
        plan = SimpleNamespace(transition_bars=8, harmonic_score=0.5, exit_bar_a=4,
                               entry_bar_b=1, tempo_shift_ratio=1.0)
        fake_eng = SimpleNamespace(
            name="fake",
            analyze=lambda audio, sr_: struct,
            plan_transition=lambda a, b, transition_bars: plan,
            DJEngine=lambda sr: SimpleNamespace(
                render=lambda *a, **k: np.zeros(sr // 2, dtype=np.float32)
            ),
        )
        monkeypatch.setattr(remix_mod, "_resolve_engine", lambda: fake_eng)
        monkeypatch.setattr(
            remix_mod, "load_source_audio",
            lambda name, sr=None, mono=True: (np.zeros(sr or 8000, dtype=np.float32), sr or 8000),
        )
        import scripts.core.spectral_matching as sm
        monkeypatch.setattr(sm, "apply_spectral_matching", lambda *a, **k: None)

        token = current_user.set(uid)
        try:
            out = remix_mod.task_remix_preview("job-preview", "a", "b", transition_bars=8)
        finally:
            current_user.reset(token)

        assert (user_root / out["session_id"] / "preview.wav").exists()
        assert not (user_root / "dj_stale").exists(), "preview must sweep the acting user"


# ---------------------------------------------------------------------------
# B2.15 — lifespan background task
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lifespan_sweep_covers_every_user_dir_and_marks_job(monkeypatch, fake_executor):
    monkeypatch.setattr(main_mod, "SWEEP_START_DELAY_SEC", 0.1, raising=False)
    monkeypatch.setattr(cfg.retention, "sweep_interval_hours", 1.0, raising=False)  # no repeat in test

    uid_a = f"lifespan_a_{uuid.uuid4().hex[:8]}"
    uid_b = f"lifespan_b_{uuid.uuid4().hex[:8]}"
    dir_a = _mk_old_session(paths_mod.OUTPUTS_DIR / uid_a, "dj_old_a")
    dir_b = _mk_old_session(paths_mod.OUTPUTS_DIR / uid_b, "chain_old_b")
    jid = _seed_done_job(uid_a, "dj_old_a")
    current_user.set("someone_else")  # the sweep must not depend on the acting user

    async with main_mod.lifespan(main_mod.app):
        swept = await _until(lambda: not dir_a.exists() and not dir_b.exists(), timeout=2.0)
        assert swept, f"start delay sweep did not run: a={dir_a.exists()} b={dir_b.exists()}"
        assert job_store.get_job_from_db(jid)["result"].get("expired") is True


@pytest.mark.asyncio
async def test_lifespan_sweep_repeats_on_the_configured_interval(monkeypatch, fake_executor):
    monkeypatch.setattr(main_mod, "SWEEP_START_DELAY_SEC", 0.1, raising=False)
    monkeypatch.setattr(cfg.retention, "sweep_interval_hours", 0.1 / 3600, raising=False)  # 0.1 s

    uid = f"lifespan_{uuid.uuid4().hex[:8]}"
    user_root = paths_mod.OUTPUTS_DIR / uid
    first = _mk_old_session(user_root, "dj_first")

    async with main_mod.lifespan(main_mod.app):
        assert await _until(lambda: not first.exists(), timeout=2.0), "first sweep did not run"
        second = _mk_old_session(user_root, "dj_second")
        assert await _until(lambda: not second.exists(), timeout=2.0), \
            "sweep did not repeat on the 0.1 s interval"


# ---------------------------------------------------------------------------
# B2.17 — shutdown ordering and an open stream
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_shutdown_sets_closing_before_executor_drain_and_ends_open_stream(fake_executor):
    cm = main_mod.lifespan(main_mod.app)
    await cm.__aenter__()

    queue = await events_mod.broadcaster.subscribe("stream_user")

    async def consume() -> int:
        n = 0
        async for _chunk in events_mod._sse_generator(queue):
            n += 1
        return n

    stream_task = asyncio.create_task(consume())
    try:
        await asyncio.sleep(0.3)                 # ack consumed; generator now ticking
        assert not stream_task.done()
        assert not events_mod.closing.is_set()

        t0 = time.perf_counter()
        await cm.__aexit__(None, None, None)
        elapsed = time.perf_counter() - t0

        assert fake_executor.shutdown_calls, "lifespan shutdown must drain the executor"
        assert fake_executor.closing_set_at_shutdown is True, \
            "SSE closing must be set BEFORE the executor is drained"
        assert elapsed < 2.0, f"shutdown took {elapsed:.2f}s"
        frames = await asyncio.wait_for(stream_task, timeout=2.0)
        assert frames >= 1                       # at least the ack went out
    finally:
        if not stream_task.done():
            stream_task.cancel()
        await events_mod.broadcaster.unsubscribe(queue)


# ---------------------------------------------------------------------------
# B2.16 — readiness
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    # No `with`: routes only, lifespan not started.
    return TestClient(main_mod.app)


@pytest.fixture
def tmp_library(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    lib.mkdir()
    monkeypatch.setattr(system_mod, "LIBRARY_DIR", lib)
    monkeypatch.setattr(paths_mod, "LIBRARY_DIR", lib)
    import scripts.core.library as library_mod
    monkeypatch.setattr(library_mod, "_DEFAULT_LIBRARY_DIR", lib, raising=False)
    return lib


class TestReadiness:
    @pytest.mark.parametrize(
        "store", [job_store, crates_mod, feedback_mod, legal_mod],
        ids=["jobs", "crates", "feedback", "legal"],
    )
    def test_dead_store_db_path_is_503(self, store, tmp_path, monkeypatch, client):
        dead = tmp_path / "dead"
        dead.mkdir()                             # a directory: sqlite cannot open it
        monkeypatch.setattr(store, "_DB_PATH", dead)
        db_mod._engine_for.cache_clear()

        r = client.get("/health/ready")
        assert r.status_code == 503, r.text

    def test_healthy_stores_and_catalog_is_200(self, tmp_library, client):
        (tmp_library / "song_one").mkdir()
        r = client.get("/health/ready")
        assert r.status_code == 200, r.text
        assert r.json().get("catalog_songs") == 1

    def test_empty_catalog_with_flag_is_503_catalog_empty(self, tmp_library, monkeypatch, client):
        monkeypatch.setenv("REMIXMATE_REQUIRE_CATALOG", "1")
        r = client.get("/health/ready")
        assert r.status_code == 503, r.text
        assert _ready_reason(r.json()) == "catalog_empty", r.text

    def test_empty_catalog_without_flag_is_200_with_zero(self, tmp_library, monkeypatch, client):
        monkeypatch.delenv("REMIXMATE_REQUIRE_CATALOG", raising=False)
        r = client.get("/health/ready")
        assert r.status_code == 200, r.text
        assert r.json().get("catalog_songs") == 0

    def test_nonempty_catalog_with_flag_is_200(self, tmp_library, monkeypatch, client):
        """allow control for the catalog gate"""
        (tmp_library / "song_one").mkdir()
        monkeypatch.setenv("REMIXMATE_REQUIRE_CATALOG", "1")
        r = client.get("/health/ready")
        assert r.status_code == 200, r.text
        assert r.json().get("catalog_songs") == 1
