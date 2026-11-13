"""tests/test_cors_origins.py — Phase 7 prod vs dev CORS allow-list."""

from __future__ import annotations

import pytest

from scripts.api.cors import resolve_cors_origins


def _vite_origins(port: int) -> list[str]:
    return [
        f"http://localhost:{port}",
        f"https://localhost:{port}",
        f"http://127.0.0.1:{port}",
        f"https://127.0.0.1:{port}",
    ]


@pytest.fixture(autouse=True)
def _no_cors_env(monkeypatch):
    monkeypatch.delenv("REMIXMATE_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("REMIXMATE_FRONTEND_URL", raising=False)
    monkeypatch.delenv("RMX_UI_PORT", raising=False)


def test_dev_defaults_are_vite_only():
    origins = resolve_cors_origins(["*"])
    assert origins == _vite_origins(5173)
    assert not any("8501" in o for o in origins)
    assert not any("github.io" in o for o in origins)


def test_dev_defaults_follow_ui_port(monkeypatch):
    monkeypatch.setenv("RMX_UI_PORT", "5199")
    origins = resolve_cors_origins(["*"])
    assert origins == _vite_origins(5199)
    assert not any("5173" in o for o in origins)


@pytest.mark.parametrize("value", ["abc", "0", "70000", "5173,http://evil.example", ""])
def test_bad_ui_port_falls_back_to_5173(monkeypatch, value):
    monkeypatch.setenv("RMX_UI_PORT", value)
    origins = resolve_cors_origins()
    assert origins == _vite_origins(5173)
    assert not any("evil" in o for o in origins)


def test_cors_origins_env_wins_over_ui_port(monkeypatch):
    monkeypatch.setenv("RMX_UI_PORT", "5199")
    monkeypatch.setenv("REMIXMATE_CORS_ORIGINS", "https://a.example")
    assert resolve_cors_origins() == ["https://a.example"]


def test_frontend_url_wins_over_ui_port(monkeypatch):
    monkeypatch.setenv("RMX_UI_PORT", "5199")
    monkeypatch.setenv("REMIXMATE_FRONTEND_URL", "https://remix.example")
    assert resolve_cors_origins() == ["https://remix.example"]


def test_ui_port_env_leaves_config_unchanged(monkeypatch):
    from scripts.core.config import _apply_env_vars, load_config

    monkeypatch.delenv("REMIXMATE_UI_PORT", raising=False)
    before_cfg = load_config()
    before_raw = _apply_env_vars({})
    monkeypatch.setenv("RMX_UI_PORT", "5199")
    assert load_config() == before_cfg
    assert _apply_env_vars({}) == before_raw, "the env loader read RMX_UI_PORT"
    # Control: the same value under the REMIXMATE_ prefix does reach the loader.
    monkeypatch.setenv("REMIXMATE_UI_PORT", "5199")
    assert _apply_env_vars({}).get("ui") == {"port": "5199"}


def test_prod_frontend_url(monkeypatch):
    monkeypatch.setenv("REMIXMATE_FRONTEND_URL", "https://remix.example/")
    assert resolve_cors_origins() == ["https://remix.example"]


def test_cors_origins_env_wins(monkeypatch):
    monkeypatch.setenv("REMIXMATE_CORS_ORIGINS", "https://a.example, https://b.example")
    monkeypatch.setenv("REMIXMATE_FRONTEND_URL", "https://ignored.example")
    assert resolve_cors_origins() == ["https://a.example", "https://b.example"]


def test_config_strips_legacy():
    origins = resolve_cors_origins(
        [
            "http://localhost:5173",
            "http://localhost:8501",
            "https://chunduri-aditya.github.io",
        ]
    )
    assert origins == ["http://localhost:5173"]
