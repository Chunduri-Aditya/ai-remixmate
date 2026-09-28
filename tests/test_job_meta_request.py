"""
tests/test_job_meta_request.py — B5.3: the remix routes persist the request.

The inspector's Retry replays ``job.meta.request``; that only works if the
route stored the full validated request at create time and the job store wrote
it to SQLite (meta_json). Two layers:

  * route layer: a fake create_job captures the meta the route hands over;
    ``meta["request"]`` must equal the request as the client sent it (with
    schema defaults filled), for /dj-remix, /dj-remix/preview and /dj-chain.
  * persistence layer: the real job store against a tmp SQLite file; the row
    read back from the database carries ``meta["request"]``.

Nothing here renders audio: submit_job is stubbed, the song lookup is stubbed.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.api import jobs as job_store
from scripts.api.auth import DEFAULT_USER, current_user
from scripts.api.routers import remix as remix_router
from scripts.api.schemas import JobType

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _isolate_user():
    current_user.set(DEFAULT_USER)
    yield
    current_user.set(DEFAULT_USER)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(remix_router.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def no_render(monkeypatch, tmp_path):
    """Never reach the worker or the library, whichever job store is live."""
    monkeypatch.setattr(remix_router.job_store, "submit_job", lambda job_id, fn, **kw: None)
    monkeypatch.setattr(remix_router, "_check_job_cap", lambda: None)
    monkeypatch.setattr(remix_router, "_require_song", lambda name: tmp_path / name)


@pytest.fixture
def captured(monkeypatch, no_render):
    """Fake create_job that records the meta each route passes."""
    state: dict = {"metas": []}

    def _fake_create_job(job_type, meta):
        state["metas"].append(meta)
        return "job-under-test"

    monkeypatch.setattr(remix_router.job_store, "create_job", _fake_create_job)
    monkeypatch.setattr(
        remix_router.job_store, "get_job",
        lambda jid: {"job_id": jid, "status": "pending", "job_type": JobType.DJ_REMIX,
                     "progress": 0.0, "message": "", "created_at": 0.0,
                     "updated_at": 0.0, "result": None, "error": None, "meta": {}},
    )
    return state


@pytest.fixture
def store(tmp_path, monkeypatch, no_render):
    """Real job store on a throwaway SQLite file (same pattern as test_user_scope)."""
    db = tmp_path / "jobs.db"
    monkeypatch.setattr(job_store, "_DB_PATH", str(db))
    monkeypatch.setattr(job_store, "_jobs", {})
    job_store._init_db()
    return db


REMIX_BODY = {
    "song_a": "alpha",
    "song_b": "beta",
    "transition_bars": 32,
    "preset": "techno",
    "bridge_beat_mode": "auto",
    "bridge_beat_genre": "house",
    "bridge_beat_intensity": 0.6,
    "transition_effect": "stepped",
}

CHAIN_BODY = {
    "songs": ["a", "b", "c"],
    "transition_bars": 8,
    "preset": "house",
    "transition_effect": "echo",
}

PREVIEW_BODY = {
    "song_a": "alpha",
    "song_b": "beta",
    "transition_bars": 24,
    "transition_effect": "filter",
}


# ---------------------------------------------------------------------------
# route layer: meta.request equals the request the client sent
# ---------------------------------------------------------------------------

def test_dj_remix_stores_the_request_in_meta(client, captured):
    r = client.post("/dj-remix", json=REMIX_BODY)
    assert r.status_code == 202, r.text
    assert len(captured["metas"]) == 1
    meta = captured["metas"][0]
    assert "request" in meta, f"meta has no request key: {sorted(meta)}"
    req = meta["request"]
    # every field the client sent, with the value it sent (non defaults on purpose)
    for key, value in REMIX_BODY.items():
        assert req.get(key) == value, (key, req.get(key), value)
    # a field the client omitted comes back with the schema default, so a
    # replay is byte identical to what the worker actually ran
    assert req.get("bridge_beat_path") is None
    # legacy keys the feedback panel and the old retry path read still exist
    assert meta["song_a"] == "alpha" and meta["song_b"] == "beta"


def test_dj_chain_stores_the_request_in_meta(client, captured):
    r = client.post("/dj-chain", json=CHAIN_BODY)
    assert r.status_code == 202, r.text
    meta = captured["metas"][0]
    assert "request" in meta, f"meta has no request key: {sorted(meta)}"
    req = meta["request"]
    for key, value in CHAIN_BODY.items():
        assert req.get(key) == value, (key, req.get(key), value)
    assert req.get("bridge_beat_mode") == "none"
    assert meta["songs"] == ["a", "b", "c"] and meta["n"] == 3


def test_dj_remix_preview_stores_the_request_in_meta(client, captured):
    r = client.post("/dj-remix/preview", json=PREVIEW_BODY)
    assert r.status_code == 202, r.text
    meta = captured["metas"][0]
    assert "request" in meta, f"meta has no request key: {sorted(meta)}"
    for key, value in PREVIEW_BODY.items():
        assert meta["request"].get(key) == value, (key, meta["request"].get(key), value)
    assert meta["preview"] is True


def test_replaying_meta_request_reproduces_the_same_request(client, captured):
    """The retry contract: POST meta.request back and the new job's
    meta.request is identical. Also proves the stored value is plain JSON."""
    r1 = client.post("/dj-remix", json=REMIX_BODY)
    assert r1.status_code == 202, r1.text
    first = captured["metas"][0]["request"]
    replay_body = json.loads(json.dumps(first))   # must survive JSON, as meta_json does

    r2 = client.post("/dj-remix", json=replay_body)
    assert r2.status_code == 202, r2.text
    second = captured["metas"][1]["request"]
    assert second == first


# ---------------------------------------------------------------------------
# persistence layer: the SQLite row carries meta.request
# ---------------------------------------------------------------------------

def test_meta_request_is_persisted_to_the_database(client, store):
    r = client.post("/dj-remix", json=REMIX_BODY)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    row = job_store.get_job_from_db(job_id)
    assert row is not None, "job row was not written"
    meta = row["meta"]
    assert "request" in meta, f"persisted meta has no request key: {sorted(meta)}"
    assert meta["request"]["song_a"] == "alpha"
    assert meta["request"]["song_b"] == "beta"
    assert meta["request"]["transition_bars"] == 32
    assert meta["request"]["preset"] == "techno"
    assert meta["request"]["transition_effect"] == "stepped"
    # user stamp from create_job still present alongside the request
    assert meta["user_id"] == DEFAULT_USER
