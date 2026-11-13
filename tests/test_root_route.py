"""
tests/test_root_route.py — the API root (deployment plan P2).

The API serves no UI. `/` and `/app` redirect to REMIXMATE_FRONTEND_URL when it
is set, otherwise return a small JSON pointer (keeps local dev usable).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from scripts.api.main import app


@pytest.fixture
def client() -> TestClient:
    # raise_server_exceptions default; no lifespan needed for these routes
    return TestClient(app)


@pytest.mark.parametrize("path", ["/", "/app"])
def test_root_redirects_when_frontend_url_set(client, monkeypatch, path):
    monkeypatch.setenv("REMIXMATE_FRONTEND_URL", "https://remix.example.com")
    r = client.get(path, follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "https://remix.example.com"


@pytest.mark.parametrize("path", ["/", "/app"])
def test_root_returns_pointer_when_unset(client, monkeypatch, path):
    monkeypatch.delenv("REMIXMATE_FRONTEND_URL", raising=False)
    r = client.get(path, follow_redirects=False)
    assert r.status_code == 200
    body = r.json()
    assert body["docs"] == "/docs"
    assert "health" in body
    assert body["service"] == "ai-remixmate API"
    assert body["author"] == "Aditya Chunduri"
    assert body["repository"] == "https://github.com/Chunduri-Aditya/ai-remixmate"


def test_openapi_info_names_author_and_license(client):
    info = client.get("/openapi.json").json()["info"]
    assert info["contact"] == {
        "name": "Aditya Chunduri",
        "url": "https://github.com/Chunduri-Aditya/ai-remixmate",
    }
    assert info["license"] == {
        "name": "MIT",
        "url": "https://github.com/Chunduri-Aditya/ai-remixmate/blob/main/LICENSE",
    }
    # download was removed from the app; the description must not advertise it
    assert "Download" not in info["description"]
    assert "harmonic compatibility" in info["description"]


def test_no_static_ui_dir_created_on_import():
    """main.py no longer touches scripts/ui/static."""
    import scripts.api.main as m

    assert not hasattr(m, "_UI_DIR")
