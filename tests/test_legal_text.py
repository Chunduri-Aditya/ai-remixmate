"""
tests/test_legal_text.py — B5.6 backend legal text (pytest half of the
"vault-account-legal" family; plan: audit-the-project-and-sorted-flask.md).

Contract these tests pin (the builder implements against it):

- GET /legal/dmca and GET /legal/terms print the *value* of
  REMIXMATE_DMCA_CONTACT when it is set. The value is read when the page is
  served, not at import time: the set and unset cases below run in one
  process, so a module level constant can only satisfy one of them.
- When REMIXMATE_DMCA_CONTACT is unset or blank, the DMCA page says
  "contact not configured" (case insensitive).
- GET /legal/privacy names the audit log and the retention limits. The limits
  are the configured numbers, read from ``cfg.retention`` (mix_max_age_days
  as "<n> days", mix_max_count as a bare number), so the text cannot drift
  from the sweep that enforces them.

Same public app shape as tests/test_legal_phase8.py: the legal router on a
bare FastAPI app behind auth_context_middleware, no auth dependency.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.api import auth as auth_mod
from scripts.api.auth import DEFAULT_USER, auth_context_middleware
from scripts.api.routers import legal

CONTACT = "legal-notices@example.test"


@pytest.fixture(autouse=True)
def _reset_user():
    auth_mod.current_user.set(DEFAULT_USER)
    yield
    auth_mod.current_user.set(DEFAULT_USER)


def _client() -> TestClient:
    app = FastAPI()
    app.middleware("http")(auth_context_middleware)
    app.include_router(legal.router)
    return TestClient(app)


def _page(client: TestClient, page: str) -> str:
    res = client.get(f"/legal/{page}")
    assert res.status_code == 200, (page, res.status_code)
    assert res.headers["content-type"].startswith("text/plain"), res.headers["content-type"]
    return res.text


@pytest.mark.parametrize("page", ["dmca", "terms"])
def test_legal_text_prints_contact_when_set(monkeypatch, page):
    monkeypatch.setenv("REMIXMATE_DMCA_CONTACT", CONTACT)
    text = _page(_client(), page)
    assert CONTACT in text, f"/legal/{page} does not print REMIXMATE_DMCA_CONTACT value:\n{text}"


def test_dmca_text_says_contact_not_configured_when_unset(monkeypatch):
    monkeypatch.delenv("REMIXMATE_DMCA_CONTACT", raising=False)
    text = _page(_client(), "dmca")
    assert "contact not configured" in text.lower(), text
    assert CONTACT not in text


def test_dmca_text_treats_blank_contact_as_unset(monkeypatch):
    monkeypatch.setenv("REMIXMATE_DMCA_CONTACT", "   ")
    text = _page(_client(), "dmca")
    assert "contact not configured" in text.lower(), text


def test_contact_is_read_per_request_not_at_import(monkeypatch):
    client = _client()
    monkeypatch.setenv("REMIXMATE_DMCA_CONTACT", "first@example.test")
    assert "first@example.test" in _page(client, "dmca")
    monkeypatch.setenv("REMIXMATE_DMCA_CONTACT", "second@example.test")
    text = _page(client, "dmca")
    assert "second@example.test" in text, text
    assert "first@example.test" not in text


def test_privacy_names_audit_log_and_retention_limits():
    from scripts.core.config import cfg

    days = int(cfg.retention.mix_max_age_days)
    count = int(cfg.retention.mix_max_count)
    text = _page(_client(), "privacy")
    low = text.lower()

    assert "audit log" in low, f"privacy text does not mention the audit log:\n{text}"
    assert re.search(r"reten|retain|kept for|deleted after|removed after", low), (
        f"privacy text has no retention statement:\n{text}"
    )
    assert re.search(rf"\b{days}\s*days?\b", low), (
        f"privacy text does not state the {days} day mix age limit:\n{text}"
    )
    assert re.search(rf"\b{count}\b", text), (
        f"privacy text does not state the {count} mix count cap:\n{text}"
    )
