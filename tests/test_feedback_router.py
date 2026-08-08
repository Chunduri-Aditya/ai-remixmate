"""
tests/test_feedback_router.py — POST /feedback/event, GET /feedback/summary
(SELF_IMPROVING_DJ_RESEARCH.md §3, Stage A).

Follows tests/test_router_smoke.py's pattern: a minimal FastAPI app mounting
only the feedback router, real TestClient hitting the real route handler.
feedback_store's SQLite writes are isolated to a tmp_path DB (same technique
as test_feedback_store.py); the job lookup used to backfill `context` is
mocked directly rather than touching the real jobs.db, mirroring
test_router_smoke.py's TestJobsRouter.

Marked `integration` (TestClient/httpx) — excluded by the same
`pytest -m "not integration"` filter as every other TestClient-based router
test in this suite. No dj_analysis dependency: this router never touches
librosa.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed (required by TestClient)")

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _app_with(router) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.mark.integration
class TestFeedbackEventEndpoint:
    @pytest.fixture(autouse=True)
    def client(self, tmp_path, monkeypatch):
        from scripts.core import feedback_store as fs
        from scripts.api.routers import feedback as feedback_mod

        monkeypatch.setattr(fs, "_DB_PATH", tmp_path / "feedback.db")
        fs._events.clear()
        fs._init_db()

        app = _app_with(feedback_mod.router)
        with TestClient(app, raise_server_exceptions=True) as c:
            self._client = c
            self._fs = fs
            self._feedback_mod = feedback_mod
            yield c
        fs._events.clear()

    def test_logs_event_and_returns_id(self):
        r = self._client.post("/feedback/event", json={
            "event_type": "play_start", "song_a": "Song A", "song_b": "Song B",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["logged"] is True
        assert len(body["event_id"]) == 36
        assert self._fs.list_events()[0]["event_type"] == "play_start"

    def test_invalid_event_type_422(self):
        r = self._client.post("/feedback/event", json={
            "event_type": "not_a_real_type", "song_a": "Song A",
        })
        assert r.status_code == 422   # FastAPI enum validation, before it ever reaches log_event()

    def test_missing_song_a_422(self):
        r = self._client.post("/feedback/event", json={"event_type": "play_start"})
        assert r.status_code == 422

    def test_explicit_context_is_used_as_is(self):
        r = self._client.post("/feedback/event", json={
            "event_type": "skip", "song_a": "A", "song_b": "B",
            "context": {"transition_bars": 8},
        })
        assert r.status_code == 200
        assert self._fs.list_events()[0]["context"] == {"transition_bars": 8}

    def test_context_backfilled_from_job_when_omitted(self, monkeypatch):
        fake_job = {
            "job_id": "job-1",
            "meta": {"preset": "club", "effect": "echo_out"},
            "result": {"transition_bars": 16, "harmonic_score": 0.82},
        }
        monkeypatch.setattr(
            self._feedback_mod.job_store, "get_job",
            lambda job_id: fake_job if job_id == "job-1" else None,
        )
        r = self._client.post("/feedback/event", json={
            "event_type": "thumbs_up", "song_a": "A", "song_b": "B", "job_id": "job-1",
        })
        assert r.status_code == 200
        ctx = self._fs.list_events()[0]["context"]
        assert ctx["preset"] == "club"
        assert ctx["transition_bars"] == 16
        assert ctx["harmonic_score"] == 0.82

    def test_unknown_job_id_backfills_empty_context_not_error(self, monkeypatch):
        monkeypatch.setattr(self._feedback_mod.job_store, "get_job", lambda job_id: None)
        r = self._client.post("/feedback/event", json={
            "event_type": "thumbs_down", "song_a": "A", "job_id": "stale-job",
        })
        assert r.status_code == 200
        assert self._fs.list_events()[0]["context"] == {}


@pytest.mark.integration
class TestFeedbackSummaryEndpoint:
    @pytest.fixture(autouse=True)
    def client(self, tmp_path, monkeypatch):
        from scripts.core import feedback_store as fs
        from scripts.api.routers import feedback as feedback_mod

        monkeypatch.setattr(fs, "_DB_PATH", tmp_path / "feedback.db")
        fs._events.clear()
        fs._init_db()

        app = _app_with(feedback_mod.router)
        with TestClient(app, raise_server_exceptions=True) as c:
            self._client = c
            self._fs = fs
            yield c
        fs._events.clear()

    def test_empty_summary_shape(self):
        r = self._client.get("/feedback/summary")
        assert r.status_code == 200
        body = r.json()
        assert body["total_events"] == 0
        assert body["by_event_type"] == {}
        assert body["oldest_event_at"] is None

    def test_summary_reflects_logged_events(self):
        self._client.post("/feedback/event", json={"event_type": "play_start", "song_a": "A", "song_b": "B"})
        self._client.post("/feedback/event", json={"event_type": "skip", "song_a": "A", "song_b": "B"})
        r = self._client.get("/feedback/summary")
        body = r.json()
        assert body["total_events"] == 2
        assert body["by_event_type"] == {"play_start": 1, "skip": 1}
        assert "A :: B" in body["by_song_pair"]
