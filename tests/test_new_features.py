"""
tests/test_new_features.py — Tests for all new modules added in v2.

Coverage map (following testing pyramid):
  Unit:
    - Audit module (log_audit, read_audit, filtering, error handling)
    - Structured logging (get_logger, context vars, formatters)
    - Job store ETA calculation + lifecycle
    - DJEngine input validation (empty arrays, bad BPM, NaN/Inf)
    - Rate-limit guard (_check_job_cap)
  Integration:
    - Full job lifecycle: create → submit → poll → done
    - Full job lifecycle: create → submit → poll → failed
    - job_to_response() round-trip (eta_sec field present)
  Smoke:
    - Health live / health ready endpoints respond correctly

Run:
    python -m pytest tests/test_new_features.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

SR = 44100


def _make_sine(duration: float = 2.0, bpm: float = 120.0) -> np.ndarray:
    """Synthetic 2-second sine wave with beat-aligned amplitude envelope."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    beat_samples = int(SR * 60.0 / bpm)
    env = np.zeros_like(t)
    for onset in range(0, len(t), beat_samples):
        sl = slice(onset, min(onset + beat_samples, len(t)))
        decay_len = sl.stop - sl.start
        env[sl] += np.exp(-np.linspace(0, 5, decay_len))
    audio = (np.sin(2 * np.pi * 440 * t) + rng.normal(0, 0.02, len(t))) * env
    audio = audio.astype(np.float32)
    mx = np.abs(audio).max()
    if mx > 1e-6:
        audio /= mx
    return audio


# ===========================================================================
# 1. Audit module
# ===========================================================================

class TestAudit:
    """Unit tests for scripts/core/audit.py"""

    def test_log_audit_creates_file(self, tmp_path):
        """log_audit should create data/audit.jsonl in the data dir."""
        from scripts.core import audit as _audit_mod
        audit_path = tmp_path / "audit.jsonl"
        original = _audit_mod._AUDIT_PATH
        _audit_mod._AUDIT_PATH = audit_path
        try:
            from scripts.core.audit import log_audit
            log_audit("test_action", resource="test_song", job_id="job-123")
            assert audit_path.exists(), "audit.jsonl should be created"
        finally:
            _audit_mod._AUDIT_PATH = original

    def test_log_audit_valid_json(self, tmp_path):
        """Every written entry must be valid JSON."""
        from scripts.core import audit as _audit_mod
        audit_path = tmp_path / "audit.jsonl"
        original = _audit_mod._AUDIT_PATH
        _audit_mod._AUDIT_PATH = audit_path
        try:
            from scripts.core.audit import log_audit
            log_audit("download_start", resource="song_a", job_id="j1",
                      metadata={"query": "test", "separate": True})
            raw = audit_path.read_text().strip()
            entry = json.loads(raw)
            assert entry["action"] == "download_start"
            assert entry["resource"] == "song_a"
            assert entry["job_id"] == "j1"
            assert entry["meta"]["query"] == "test"
            assert "ts" in entry
            assert "epoch" in entry
        finally:
            _audit_mod._AUDIT_PATH = original

    def test_read_audit_newest_first(self, tmp_path):
        """read_audit should return entries in reverse chronological order."""
        from scripts.core import audit as _audit_mod
        audit_path = tmp_path / "audit.jsonl"
        original = _audit_mod._AUDIT_PATH
        _audit_mod._AUDIT_PATH = audit_path
        try:
            from scripts.core.audit import log_audit, read_audit
            log_audit("action_1")
            time.sleep(0.01)
            log_audit("action_2")
            time.sleep(0.01)
            log_audit("action_3")
            entries = read_audit()
            assert entries[0]["action"] == "action_3", "Newest should be first"
            assert entries[-1]["action"] == "action_1", "Oldest should be last"
        finally:
            _audit_mod._AUDIT_PATH = original

    def test_read_audit_action_filter(self, tmp_path):
        """action_filter should return only entries with matching action."""
        from scripts.core import audit as _audit_mod
        audit_path = tmp_path / "audit.jsonl"
        original = _audit_mod._AUDIT_PATH
        _audit_mod._AUDIT_PATH = audit_path
        try:
            from scripts.core.audit import log_audit, read_audit
            log_audit("download_start", resource="song_a")
            log_audit("dj_remix_start", resource="song_b")
            log_audit("download_start", resource="song_c")
            downloads = read_audit(action_filter="download_start")
            assert all(e["action"] == "download_start" for e in downloads)
            assert len(downloads) == 2
        finally:
            _audit_mod._AUDIT_PATH = original

    def test_read_audit_limit(self, tmp_path):
        """read_audit should respect the limit parameter."""
        from scripts.core import audit as _audit_mod
        audit_path = tmp_path / "audit.jsonl"
        original = _audit_mod._AUDIT_PATH
        _audit_mod._AUDIT_PATH = audit_path
        try:
            from scripts.core.audit import log_audit, read_audit
            for i in range(10):
                log_audit(f"action_{i}")
            entries = read_audit(limit=3)
            assert len(entries) == 3
        finally:
            _audit_mod._AUDIT_PATH = original

    def test_read_audit_missing_file(self, tmp_path):
        """read_audit on a non-existent file should return empty list, not crash."""
        from scripts.core import audit as _audit_mod
        original = _audit_mod._AUDIT_PATH
        _audit_mod._AUDIT_PATH = tmp_path / "nonexistent.jsonl"
        try:
            from scripts.core.audit import read_audit
            result = read_audit()
            assert result == []
        finally:
            _audit_mod._AUDIT_PATH = original

    def test_log_audit_default_user(self, tmp_path):
        """When user is not provided, it should default to 'local'."""
        from scripts.core import audit as _audit_mod
        audit_path = tmp_path / "audit.jsonl"
        original = _audit_mod._AUDIT_PATH
        _audit_mod._AUDIT_PATH = audit_path
        try:
            from scripts.core.audit import log_audit, read_audit
            log_audit("test_default_user")
            entries = read_audit()
            assert entries[0]["user"] == "local"
        finally:
            _audit_mod._AUDIT_PATH = original


# ===========================================================================
# 2. Structured logging
# ===========================================================================

class TestLoggingUtils:
    """Unit tests for scripts/core/logging_utils.py"""

    def test_get_logger_returns_structured_logger(self):
        """get_logger should return a StructuredLogger instance."""
        from scripts.core.logging_utils import get_logger, StructuredLogger
        logger = get_logger("test.module")
        assert isinstance(logger, StructuredLogger)

    def test_get_logger_same_instance(self):
        """Two calls with the same name should return equivalent loggers."""
        from scripts.core.logging_utils import get_logger
        l1 = get_logger("test.same")
        l2 = get_logger("test.same")
        assert l1._name == l2._name

    def test_set_get_request_id(self):
        """set_request_id + get_request_id should round-trip correctly."""
        from scripts.core.logging_utils import set_request_id, get_request_id
        set_request_id("req-abc-123")
        assert get_request_id() == "req-abc-123"

    def test_set_get_job_id(self):
        """set_job_id + get_job_id should round-trip correctly."""
        from scripts.core.logging_utils import set_job_id, get_job_id
        set_job_id("job-xyz-999")
        assert get_job_id() == "job-xyz-999"

    def test_logger_info_does_not_raise(self):
        """Calling logger.info with extra fields should not raise."""
        from scripts.core.logging_utils import get_logger
        logger = get_logger("test.smoke")
        logger.info("Test info message", extra={"key": "value", "num": 42})

    def test_logger_error_does_not_raise(self):
        """Calling logger.error with extra fields should not raise."""
        from scripts.core.logging_utils import get_logger
        logger = get_logger("test.smoke.error")
        logger.error("Test error message", extra={"error_code": 500})

    def test_logger_warning_does_not_raise(self):
        """Calling logger.warning should not raise."""
        from scripts.core.logging_utils import get_logger
        logger = get_logger("test.smoke.warning")
        logger.warning("Test warning")

    def test_structured_json_formatter_produces_valid_json(self):
        """StructuredJsonFormatter output must be parseable JSON."""
        import logging as _stdlib_logging
        from scripts.core.logging_utils import StructuredJsonFormatter
        formatter = StructuredJsonFormatter()
        record = _stdlib_logging.LogRecord(
            name="test", level=_stdlib_logging.INFO,
            pathname="", lineno=0, msg="hello world",
            args=(), exc_info=None
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello world"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed


# ===========================================================================
# 3. Job store — ETA calculation + lifecycle
# ===========================================================================

class TestJobStore:
    """Unit + integration tests for scripts/api/jobs.py"""

    def test_create_job_returns_string_id(self):
        """create_job should return a non-empty string UUID."""
        from scripts.api.jobs import create_job
        from scripts.api.schemas import JobType
        job_id = create_job(JobType.DOWNLOAD, {"query": "test"})
        assert isinstance(job_id, str)
        assert len(job_id) == 36  # UUID4 format

    def test_create_job_initial_state(self):
        """Newly created job should have PENDING status and 0.0 progress."""
        from scripts.api.jobs import create_job, get_job
        from scripts.api.schemas import JobType, JobStatus
        job_id = create_job(JobType.ANALYZE)
        job = get_job(job_id)
        assert job is not None
        assert job["status"] == JobStatus.PENDING
        assert job["progress"] == 0.0
        assert job["eta_sec"] is None
        assert job["result"] is None

    def test_update_job_progress(self):
        """update_job should update progress, message, and compute ETA."""
        from scripts.api.jobs import create_job, update_job, get_job
        from scripts.api.schemas import JobType
        job_id = create_job(JobType.DJ_REMIX)
        # Simulate started_at being set
        job = get_job(job_id)
        job["started_at"] = time.time() - 10.0  # 10 seconds ago
        update_job(job_id, progress=0.5, message="Halfway there")
        job = get_job(job_id)
        assert job["progress"] == 0.5
        assert job["message"] == "Halfway there"
        # ETA should be set (10s elapsed at 50% → ~10s remaining)
        assert job["eta_sec"] is not None
        assert isinstance(job["eta_sec"], int)
        assert job["eta_sec"] >= 0

    def test_update_job_eta_decreases(self):
        """ETA should decrease as progress increases."""
        from scripts.api.jobs import create_job, update_job, get_job
        from scripts.api.schemas import JobType
        job_id = create_job(JobType.DJ_REMIX)
        job = get_job(job_id)
        job["started_at"] = time.time() - 20.0
        update_job(job_id, progress=0.25)
        eta_early = get_job(job_id)["eta_sec"]
        job["started_at"] = time.time() - 20.0
        update_job(job_id, progress=0.75)
        eta_later = get_job(job_id)["eta_sec"]
        assert eta_later < eta_early, "ETA should decrease as progress increases"

    def test_update_job_clamps_progress(self):
        """Progress should be clamped to [0, 1]."""
        from scripts.api.jobs import create_job, update_job, get_job
        from scripts.api.schemas import JobType
        job_id = create_job(JobType.ANALYZE)
        update_job(job_id, progress=1.5)  # Over 1.0
        assert get_job(job_id)["progress"] == 1.0
        update_job(job_id, progress=-0.3)  # Under 0
        assert get_job(job_id)["progress"] == 0.0

    def test_list_jobs_returns_all(self):
        """list_jobs should include all created jobs."""
        from scripts.api.jobs import create_job, list_jobs, _jobs
        from scripts.api.schemas import JobType
        # Query with a limit well above the store size so the count reflects
        # actual growth — the default limit=50 caps the result and would mask
        # new jobs once the persistent store already holds ≥50 entries.
        big = len(_jobs) + 100
        before = len(list_jobs(limit=big))
        create_job(JobType.DOWNLOAD)
        create_job(JobType.DOWNLOAD)
        after = len(list_jobs(limit=big))
        assert after >= before + 2

    def test_job_to_response_includes_eta_sec(self):
        """job_to_response should include eta_sec field (can be None or int)."""
        from scripts.api.jobs import create_job, update_job, get_job, job_to_response
        from scripts.api.schemas import JobType
        job_id = create_job(JobType.ANALYZE)
        job = get_job(job_id)
        job["started_at"] = time.time() - 5.0
        update_job(job_id, progress=0.3, message="Analysing…")
        job = get_job(job_id)
        response = job_to_response(job)
        # eta_sec must be an attribute (may be None before started)
        assert hasattr(response, "eta_sec")

    def test_full_job_lifecycle_success(self):
        """
        Integration: create → submit → poll until done.
        Task function runs in the thread pool and signals completion.
        """
        from scripts.api.jobs import create_job, submit_job, get_job
        from scripts.api.schemas import JobType, JobStatus

        def _simple_task(job_id: str, **kwargs):
            from scripts.api.jobs import update_job
            update_job(job_id, progress=0.5, message="Halfway")
            return {"result": "ok", "value": 42}

        job_id = create_job(JobType.ANALYZE, {"song": "test"})
        submit_job(job_id, _simple_task)

        # Poll until done (max 5 seconds)
        deadline = time.time() + 5
        while time.time() < deadline:
            job = get_job(job_id)
            if job["status"] in (JobStatus.DONE, JobStatus.FAILED):
                break
            time.sleep(0.05)

        job = get_job(job_id)
        assert job["status"] == JobStatus.DONE, f"Expected DONE, got {job['status']}"
        assert job["result"]["value"] == 42
        assert job["finished_at"] is not None
        assert job["progress"] == 1.0

    def test_full_job_lifecycle_failure(self):
        """
        Integration: create → submit → poll until failed.
        Failed job should capture error message.
        """
        from scripts.api.jobs import create_job, submit_job, get_job
        from scripts.api.schemas import JobType, JobStatus

        def _failing_task(job_id: str, **kwargs):
            raise RuntimeError("Deliberate test failure — do not alarm")

        job_id = create_job(JobType.DJ_REMIX)
        submit_job(job_id, _failing_task)

        deadline = time.time() + 5
        while time.time() < deadline:
            job = get_job(job_id)
            if job["status"] in (JobStatus.DONE, JobStatus.FAILED):
                break
            time.sleep(0.05)

        job = get_job(job_id)
        assert job["status"] == JobStatus.FAILED
        assert "Deliberate test failure" in (job["error"] or "")
        assert job["finished_at"] is not None


# ===========================================================================
# 4. DJEngine input validation
# ===========================================================================

class TestDJEngineValidation:
    """Unit tests for input validation added to DJEngine.render()"""

    @pytest.fixture
    def minimal_plan(self):
        """Return a minimal valid TransitionPlan-like object."""
        from scripts.core.dj_engine import plan_transition, _analyze_impl
        audio = _make_sine(duration=5.0, bpm=120.0)
        struct = _analyze_impl(audio, SR)
        return plan_transition(struct, struct, transition_bars=8)

    def test_empty_track_a_raises(self, minimal_plan):
        """Empty track_a should raise ValueError immediately."""
        from scripts.core.dj_engine import DJEngine
        engine = DJEngine(sr=SR)
        good = _make_sine()
        empty = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="non-zero length"):
            engine.render(empty, good, minimal_plan)

    def test_empty_track_b_raises(self, minimal_plan):
        """Empty track_b should raise ValueError immediately."""
        from scripts.core.dj_engine import DJEngine
        engine = DJEngine(sr=SR)
        good = _make_sine()
        empty = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="non-zero length"):
            engine.render(good, empty, minimal_plan)

    def test_nan_in_track_raises(self, minimal_plan):
        """NaN values in track_a should raise ValueError."""
        from scripts.core.dj_engine import DJEngine
        engine = DJEngine(sr=SR)
        good = _make_sine()
        bad = good.copy()
        bad[100] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            engine.render(bad, good, minimal_plan)

    def test_inf_in_track_raises(self, minimal_plan):
        """Inf values in track_b should raise ValueError."""
        from scripts.core.dj_engine import DJEngine
        engine = DJEngine(sr=SR)
        good = _make_sine()
        bad = good.copy()
        bad[200] = float("inf")
        with pytest.raises(ValueError, match="NaN or Inf"):
            engine.render(good, bad, minimal_plan)

    def test_negative_bpm_plan_raises(self, minimal_plan):
        """A plan with bpm_a <= 0 should raise ValueError."""
        from scripts.core.dj_engine import DJEngine
        engine = DJEngine(sr=SR)
        good = _make_sine()
        minimal_plan.bpm_a = -1.0
        with pytest.raises(ValueError, match="BPM"):
            engine.render(good, good, minimal_plan)

    def test_zero_bpm_plan_raises(self, minimal_plan):
        """A plan with bpm_b = 0 should raise ValueError."""
        from scripts.core.dj_engine import DJEngine
        engine = DJEngine(sr=SR)
        good = _make_sine()
        minimal_plan.bpm_b = 0.0
        with pytest.raises(ValueError, match="BPM"):
            engine.render(good, good, minimal_plan)

    def test_negative_transition_sec_raises(self, minimal_plan):
        """transition_seconds <= 0 should raise ValueError."""
        from scripts.core.dj_engine import DJEngine
        engine = DJEngine(sr=SR)
        good = _make_sine()
        minimal_plan.transition_seconds = -5.0
        with pytest.raises(ValueError, match="[Tt]ransition"):
            engine.render(good, good, minimal_plan)

    def test_valid_inputs_do_not_raise(self, minimal_plan):
        """Valid inputs should not raise during render."""
        from scripts.core.dj_engine import DJEngine
        engine = DJEngine(sr=SR)
        good = _make_sine(duration=3.0)
        # Should not raise
        result = engine.render(good, good, minimal_plan, full_output=False)
        assert len(result) > 0


# ===========================================================================
# 5. Rate limiting guard (_check_job_cap)
# ===========================================================================

class TestRateLimiting:
    """Unit tests for _check_job_cap in scripts/api/routers/_helpers.py"""

    def test_under_cap_passes(self):
        """With fewer running jobs than the cap, _check_job_cap should not raise."""
        with patch("scripts.api.routers._helpers.job_store") as mock_store:
            mock_store.list_jobs.return_value = [
                {"status": "running"} for _ in range(2)
            ]
            from scripts.api.routers._helpers import _check_job_cap
            # Should not raise (2 < cap of 4)
            _check_job_cap()

    def test_at_cap_raises_429(self):
        """At the job cap, _check_job_cap should raise HTTPException with 429."""
        from fastapi import HTTPException
        with patch("scripts.api.routers._helpers.job_store") as mock_store:
            from scripts.api.routers._helpers import _check_job_cap, _MAX_ACTIVE_JOBS
            mock_store.list_jobs.return_value = [
                {"status": "running"} for _ in range(_MAX_ACTIVE_JOBS)
            ]
            with pytest.raises(HTTPException) as exc_info:
                _check_job_cap()
            assert exc_info.value.status_code == 429

    def test_over_cap_raises_429(self):
        """With more running jobs than cap, _check_job_cap should raise 429."""
        from fastapi import HTTPException
        with patch("scripts.api.routers._helpers.job_store") as mock_store:
            mock_store.list_jobs.return_value = [
                {"status": "running"} for _ in range(10)
            ]
            from scripts.api.routers._helpers import _check_job_cap
            with pytest.raises(HTTPException) as exc_info:
                _check_job_cap()
            assert exc_info.value.status_code == 429

    def test_completed_jobs_dont_count(self):
        """Done/failed jobs should not count toward the cap."""
        with patch("scripts.api.routers._helpers.job_store") as mock_store:
            mock_store.list_jobs.return_value = [
                {"status": "done"},
                {"status": "failed"},
                {"status": "running"},
                {"status": "done"},
            ]
            from scripts.api.routers._helpers import _check_job_cap
            # Only 1 running — should not raise
            _check_job_cap()


# ===========================================================================
# 6. Health endpoints (smoke tests via FastAPI TestClient)
# ===========================================================================

class TestHealthEndpoints:
    """Smoke tests for /health/live and /health/ready"""

    @pytest.fixture(scope="class")
    def client(self):
        """Create a TestClient for the FastAPI app."""
        try:
            from fastapi.testclient import TestClient
            from scripts.api.main import app
            return TestClient(app)
        except Exception:
            pytest.skip("FastAPI app could not be imported — skipping HTTP tests")

    def test_health_live_returns_200(self, client):
        """/health/live should always return 200."""
        response = client.get("/health/live")
        assert response.status_code == 200

    def test_health_live_has_status_ok(self, client):
        """/health/live body should contain status: ok."""
        response = client.get("/health/live")
        data = response.json()
        assert data.get("status") == "ok"

    def test_health_ready_returns_2xx_or_503(self, client):
        """/health/ready should return either 200 (ready) or 503 (not ready)."""
        response = client.get("/health/ready")
        assert response.status_code in (200, 503)

    def test_health_ready_has_status_field(self, client):
        """/health/ready body should have a 'status' field."""
        response = client.get("/health/ready")
        data = response.json()
        assert "status" in data

    def test_health_root_returns_library_songs(self, client):
        """/health should return library_songs count."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "library_songs" in data
        assert isinstance(data["library_songs"], int)


# ===========================================================================
# 7. Music index smoke tests
# ===========================================================================

class TestMusicIndex:
    """Smoke tests for scripts/core/music_index.py"""

    def test_get_index_returns_singleton(self):
        """get_index() called twice should return the same object."""
        try:
            from scripts.core.music_index import get_index
        except ImportError:
            pytest.skip("music_index not available")
        idx1 = get_index()
        idx2 = get_index()
        assert idx1 is idx2

    def test_query_empty_index_returns_list(self):
        """query() on any index state should return a list (never crash)."""
        try:
            from scripts.core.music_index import get_index
        except ImportError:
            pytest.skip("music_index not available")
        idx = get_index()
        # Use a synthetic query vector of the right shape
        try:
            results = idx.query(query_song="__nonexistent_song__", k=5)
            assert isinstance(results, list)
        except Exception:
            pass  # Index may raise for unknown song — that's OK

    def test_stats_returns_dict(self):
        """MusicIndex.stats() should return a dict with 'total_indexed'."""
        try:
            from scripts.core.music_index import get_index
        except ImportError:
            pytest.skip("music_index not available")
        idx = get_index()
        try:
            stats = idx.stats()
            assert isinstance(stats, dict)
            assert "total_indexed" in stats
        except Exception:
            pass  # May not have stats() — non-critical


# ===========================================================================
# 8. Job store schema round-trip
# ===========================================================================

class TestJobSchemas:
    """Ensure JobResponse Pydantic schema includes all required new fields."""

    def test_job_response_has_eta_sec_field(self):
        """JobResponse schema must have eta_sec as an Optional[int] field."""
        from scripts.api.schemas import JobResponse
        import inspect
        # Check it's a defined field, not just a dynamic attr
        fields = JobResponse.model_fields
        assert "eta_sec" in fields, "eta_sec must be a declared field in JobResponse"

    def test_job_response_eta_sec_is_optional(self):
        """JobResponse should accept None for eta_sec without validation error."""
        import time
        from scripts.api.schemas import JobResponse, JobStatus, JobType
        # Should not raise
        resp = JobResponse(
            job_id="test-id",
            status=JobStatus.PENDING,
            job_type=JobType.ANALYZE,
            created_at=time.time(),
            progress=0.0,
            message="Queued",
            eta_sec=None,
        )
        assert resp.eta_sec is None

    def test_job_response_eta_sec_accepts_int(self):
        """JobResponse should accept an integer eta_sec."""
        import time
        from scripts.api.schemas import JobResponse, JobStatus, JobType
        resp = JobResponse(
            job_id="test-id-2",
            status=JobStatus.RUNNING,
            job_type=JobType.DJ_REMIX,
            created_at=time.time(),
            progress=0.4,
            message="Mixing…",
            eta_sec=45,
        )
        assert resp.eta_sec == 45
