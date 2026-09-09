"""tests/test_cors_origins.py — Phase 7 prod vs dev CORS allow-list."""

from __future__ import annotations

from scripts.api.cors import resolve_cors_origins


def test_dev_defaults_are_vite_only(monkeypatch):
    monkeypatch.delenv("REMIXMATE_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("REMIXMATE_FRONTEND_URL", raising=False)
    origins = resolve_cors_origins(["*"])
    assert all("5173" in o for o in origins)
    assert not any("8501" in o for o in origins)
    assert not any("github.io" in o for o in origins)


def test_prod_frontend_url(monkeypatch):
    monkeypatch.delenv("REMIXMATE_CORS_ORIGINS", raising=False)
    monkeypatch.setenv("REMIXMATE_FRONTEND_URL", "https://remix.example/")
    assert resolve_cors_origins() == ["https://remix.example"]


def test_cors_origins_env_wins(monkeypatch):
    monkeypatch.setenv("REMIXMATE_CORS_ORIGINS", "https://a.example, https://b.example")
    monkeypatch.setenv("REMIXMATE_FRONTEND_URL", "https://ignored.example")
    assert resolve_cors_origins() == ["https://a.example", "https://b.example"]


def test_config_strips_legacy(monkeypatch):
    monkeypatch.delenv("REMIXMATE_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("REMIXMATE_FRONTEND_URL", raising=False)
    origins = resolve_cors_origins(
        [
            "http://localhost:5173",
            "http://localhost:8501",
            "https://chunduri-aditya.github.io",
        ]
    )
    assert origins == ["http://localhost:5173"]
