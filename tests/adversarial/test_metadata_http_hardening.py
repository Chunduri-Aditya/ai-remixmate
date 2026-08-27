"""
Adversarial QA — outbound HTTP hardening in scripts/core/track_metadata.py.

Song titles and artist names are user-controlled and get interpolated into
provider URLs, and those URLs carry `api_key` in the query string. Two
distinct failure modes are guarded here:

  1. The assembled URL reaching a host the module was never meant to talk to.
  2. The API key landing in the logs on any failure path.
"""
from __future__ import annotations

import logging

import pytest

from scripts.core.track_metadata import _ALLOWED_HOSTS, _get_json, _redact_url

_SECRET = "sk-super-secret-key-0xdeadbeef"


class _RequestIssued(BaseException):
    """Tripwire signal. Derives from BaseException so `except Exception`
    inside the code under test cannot absorb it."""


# ---------------------------------------------------------------------------
# Host allowlist
# ---------------------------------------------------------------------------

@pytest.fixture
def forbid_urlopen(monkeypatch):
    """Make any outbound request an immediate, loud failure.

    Asserting only `_get_json(...) is None` would be a fake guard: these hosts
    do not resolve, so the call returns None whether or not the allowlist is
    there. (Verified — removing the gate left all six cases green.) Blowing up
    inside urlopen is what actually distinguishes "refused" from "attempted
    and failed", so the test fails if a request is ever issued.
    """
    import urllib.request

    def _tripwire(*_a, **_kw):
        # Must derive from BaseException, NOT Exception: _get_json wraps the
        # request in `except Exception`, which swallowed an AssertionError
        # tripwire and made every case here pass with the gate disabled.
        raise _RequestIssued("outbound request was issued despite the allowlist")

    monkeypatch.setattr(urllib.request, "urlopen", _tripwire)


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",   # cloud metadata service
    "https://169.254.169.254/latest/meta-data/",
    "https://evil.example.com/steal",
    "https://api.getsongbpm.com.evil.example/x",  # suffix-confusion
    "file:///etc/passwd",
    "http://localhost:8000/jobs",
])
def test_get_json_refuses_non_allowlisted_hosts(url, forbid_urlopen):
    """Must refuse before issuing the request, not merely fail to connect."""
    assert _get_json(url) is None


def test_get_json_refuses_plain_http_on_an_allowed_host(forbid_urlopen):
    """The host is fine but the scheme is not — credentials must not cross
    the wire in clear text."""
    assert _get_json("http://musicbrainz.org/ws/2/recording/?query=x") is None


def test_allowlisted_https_host_is_permitted_through(monkeypatch):
    """Control: the gate must not be a blanket deny. Without this, a gate that
    refuses everything would pass every other test in this file."""
    import urllib.request

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self): return b'{"ok": true}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_kw: _Resp())
    assert _get_json("https://musicbrainz.org/ws/2/recording/?query=x") == {"ok": True}


def test_allowlist_is_not_empty_and_is_exact_hosts():
    assert _ALLOWED_HOSTS
    for host in _ALLOWED_HOSTS:
        assert "/" not in host and ":" not in host, f"{host!r} is not a bare hostname"


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

def test_redact_url_strips_the_query_string_carrying_the_api_key():
    url = f"https://api.getsongbpm.com/search/?api_key={_SECRET}&type=song&lookup=abc"
    out = _redact_url(url)
    assert _SECRET not in out
    assert out == "https://api.getsongbpm.com/search/"


def test_refusal_log_does_not_leak_the_api_key(caplog):
    """The refusal path logs at ERROR. It must log the redacted URL."""
    url = f"https://evil.example.com/x?api_key={_SECRET}"
    with caplog.at_level(logging.DEBUG):
        assert _get_json(url) is None
    assert _SECRET not in caplog.text, "API key leaked into logs"


def test_failure_log_does_not_leak_the_api_key(caplog, monkeypatch):
    """An allowlisted host that raises mid-request must not log the raw URL."""
    import urllib.request

    def _boom(*_a, **_kw):
        raise OSError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    url = f"https://api.getsongbpm.com/search/?api_key={_SECRET}&lookup=abc"
    with caplog.at_level(logging.DEBUG):
        assert _get_json(url) is None
    assert _SECRET not in caplog.text, "API key leaked into logs on the failure path"
