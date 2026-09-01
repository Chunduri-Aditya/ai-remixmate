"""
tests/test_generative_gate.py — the generative-routes feature flag (deployment
plan P2).

Every /ai/* route 503s unless REMIXMATE_GENERATIVE_ENABLED is truthy; the core
/dj-* remix routes are untouched. The flag is read per request.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.api.routes import router


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_generative_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("REMIXMATE_GENERATIVE_ENABLED", raising=False)

    # every /ai/* route 503s — POST and GET alike (gate is on the APIRouter)
    assert client.post("/ai/style-transfer", json={}).status_code == 503
    assert client.post("/ai/inpaint", json={}).status_code == 503
    assert client.post("/ai/tokenize", json={}).status_code == 503
    assert client.get("/ai/models").status_code == 503

    # the core remix route is NOT gated — it validates the body instead (422),
    # i.e. it reached its own handler rather than the 503 dependency
    assert client.post("/dj-remix", json={}).status_code == 422


@pytest.mark.parametrize("truthy", ["true", "1", "yes", "TRUE"])
def test_generative_enabled_lets_requests_through(client, monkeypatch, truthy):
    monkeypatch.setenv("REMIXMATE_GENERATIVE_ENABLED", truthy)
    # no longer 503 — now it 422s on the empty body, same as any live route
    r = client.post("/ai/style-transfer", json={})
    assert r.status_code != 503
    assert r.status_code == 422


@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "off"])
def test_generative_flag_falsy_values_stay_disabled(client, monkeypatch, falsy):
    monkeypatch.setenv("REMIXMATE_GENERATIVE_ENABLED", falsy)
    assert client.post("/ai/style-transfer", json={}).status_code == 503


def test_flag_is_read_per_request_not_at_import(client, monkeypatch):
    monkeypatch.delenv("REMIXMATE_GENERATIVE_ENABLED", raising=False)
    assert client.post("/ai/style-transfer", json={}).status_code == 503
    monkeypatch.setenv("REMIXMATE_GENERATIVE_ENABLED", "true")
    assert client.post("/ai/style-transfer", json={}).status_code != 503
    monkeypatch.delenv("REMIXMATE_GENERATIVE_ENABLED", raising=False)
    assert client.post("/ai/style-transfer", json={}).status_code == 503
