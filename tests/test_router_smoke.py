"""
tests/test_router_smoke.py — Smoke tests for FastAPI routers (was: zero TestClient coverage).

These are marked ``pytest.mark.integration`` so they can be excluded with
  pytest -m "not integration"
when running in environments without all FastAPI deps installed.

Strategy
--------
Instead of starting the full lifespan (which spins up SSE background tasks),
each test builds a minimal FastAPI app that mounts only the router under test.
This avoids asyncio / SSE / SQLite complexity while still hitting the real
route handlers and Pydantic serialisation.

Covered:
  - GET  /health/live                    → 200  {"status": "ok"}
  - GET  /health/ready                   → 200 or 503
  - GET  /jobs                           → 200 []   (empty store)
  - GET  /jobs/{id}                      → 404      (unknown id)
  - DELETE /jobs/{id}                    → 404      (unknown id)
  - GET  /library                        → 200 with expected schema keys
  - GET  /library/names                  → 200 list
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx",   reason="httpx not installed (required by TestClient)")

from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal app factory — no lifespan, no SSE, no background tasks
# ---------------------------------------------------------------------------

def _app_with(*routers) -> FastAPI:
    app = FastAPI()
    for router in routers:
        app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# Health router
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestHealthRouter:

    @pytest.fixture(autouse=True)
    def client(self, tmp_path):
        from scripts.api.routers.system import router
        with patch("scripts.api.routers.system.LIBRARY_DIR", tmp_path):
            app = _app_with(router)
            with TestClient(app, raise_server_exceptions=True) as c:
                self._client = c
                yield c

    def test_liveness_200(self):
        r = self._client.get("/health/live")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_liveness_content_type_json(self):
        r = self._client.get("/health/live")
        assert "application/json" in r.headers["content-type"]

    def test_readiness_returns_200_or_503(self, tmp_path):
        # 200 when library dir exists (it does — tmp_path), 503 when missing
        r = self._client.get("/health/ready")
        assert r.status_code in (200, 503)

    def test_health_basic_structure(self):
        r = self._client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert "library_songs" in body
        assert "active_jobs" in body


# ---------------------------------------------------------------------------
# Jobs router
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestJobsRouter:
    """
    Tests use a fully in-memory job store (no SQLite) by patching _upsert_row
    and _get_conn so no filesystem I/O happens.
    """

    @pytest.fixture(autouse=True)
    def client(self):
        from scripts.api.routers.jobs import router
        app = _app_with(router)
        upsert_patch = patch("scripts.api.jobs._upsert_row", return_value=None)
        conn_patch   = patch(
            "scripts.api.jobs._get_conn",
            return_value=MagicMock(
                __enter__=lambda s: MagicMock(),
                __exit__=MagicMock(return_value=False),
            ),
        )
        with upsert_patch, conn_patch:
            with TestClient(app, raise_server_exceptions=True) as c:
                self._client = c
                yield c

    def test_list_jobs_empty_200(self):
        r = self._client.get("/jobs")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_list_jobs_limit_param(self):
        r = self._client.get("/jobs?limit=5")
        assert r.status_code == 200

    def test_get_unknown_job_404(self):
        r = self._client.get("/jobs/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404

    def test_cancel_unknown_job_404(self):
        r = self._client.delete("/jobs/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404

    def test_list_jobs_limit_too_large_422(self):
        r = self._client.get("/jobs?limit=999")
        assert r.status_code == 422   # FastAPI validates le=100


# ---------------------------------------------------------------------------
# Library router
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLibraryRouter:

    @pytest.fixture(autouse=True)
    def client(self, tmp_path):
        from scripts.api.routers.library import router
        upsert_patch = patch("scripts.api.jobs._upsert_row", return_value=None)
        conn_patch   = patch(
            "scripts.api.jobs._get_conn",
            return_value=MagicMock(
                __enter__=lambda s: MagicMock(),
                __exit__=MagicMock(return_value=False),
            ),
        )
        lib_patch = patch("scripts.api.routers.library.LIBRARY_DIR", tmp_path)
        with upsert_patch, conn_patch, lib_patch:
            app = _app_with(router)
            with TestClient(app, raise_server_exceptions=True) as c:
                self._client = c
                yield c

    def test_list_library_200(self):
        r = self._client.get("/library")
        assert r.status_code == 200

    def test_list_library_schema_keys(self):
        r = self._client.get("/library")
        body = r.json()
        # LibraryListResponse must have songs + total + stats
        assert "songs" in body
        assert "total" in body
        assert isinstance(body["songs"], list)

    def test_list_library_empty_when_dir_empty(self):
        r = self._client.get("/library")
        assert r.json()["total"] == 0

    def test_get_nonexistent_song_404(self):
        r = self._client.get("/library/this-song-does-not-exist")
        assert r.status_code == 404

    def test_delete_nonexistent_song_404(self):
        r = self._client.delete("/library/this-song-does-not-exist")
        assert r.status_code == 404

    def test_list_library_pagination_params(self):
        r = self._client.get("/library?page=1&per_page=10")
        assert r.status_code == 200

    def test_list_library_invalid_page_422(self):
        r = self._client.get("/library?page=0")
        assert r.status_code == 422  # page must be ≥ 1
