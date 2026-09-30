"""
tests/test_auth.py — Phase 4: Supabase JWT auth gate (deployment plan P4).

`REMIXMATE_AUTH_ENABLED` off  → every request resolves to "local", no token.
on + valid Supabase RS256 token → the token's `sub` becomes the acting user.
on + missing / invalid token   → 401 (but `/health*` stays open).

Tokens here are synthetic: an RSA keypair generated per test, `auth._get_jwks_client`
stubbed to hand back the matching public key. The real Supabase JWKS handshake
and the browser cookie flow are `operator-pending` — they need a live project.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("jwt", reason="PyJWT not installed")
pytest.importorskip("cryptography", reason="cryptography not installed")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient

from scripts.api import auth as auth_mod
from scripts.api.auth import DEFAULT_USER, auth_context_middleware, get_user_id, require_user

_AUD = "authenticated"


def _make_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(priv, *, sub="user-abc-123", aud=_AUD, role="authenticated",
           exp_delta=3600, iat_delta=0, **extra):
    now = int(time.time())
    payload = {"sub": sub, "aud": aud, "role": role,
               "iat": now + iat_delta, "exp": now + exp_delta, **extra}
    if role is None:
        payload.pop("role")
    return jwt.encode(payload, priv, algorithm="RS256")


def _build_app() -> FastAPI:
    app = FastAPI()
    app.middleware("http")(auth_context_middleware)

    from scripts.api.routers import system
    app.include_router(system.router)  # open, no require_user

    gated = APIRouter()

    @gated.get("/whoami")
    def whoami():
        # sync handler on purpose: proves the middleware's contextvar.set()
        # reaches a sibling sync endpoint (a dependency's would not).
        return {"uid": get_user_id()}

    @gated.post("/mutate")
    def mutate():
        # body-less, like POST /library/storage/prune: a cross-site form can reach it
        return {"uid": get_user_id()}

    app.include_router(gated, dependencies=[Depends(require_user)])
    return app


@pytest.fixture(autouse=True)
def _reset_user():
    auth_mod.current_user.set(DEFAULT_USER)
    # force the lazy JWKS client to rebuild per test
    auth_mod._jwks_client = None
    auth_mod._jwks_client_url = None
    yield
    auth_mod.current_user.set(DEFAULT_USER)


@pytest.fixture
def signing_key(monkeypatch):
    """Auth enabled; JWKS client stubbed to return `pub`. Returns the private key."""
    priv = _make_key()
    pub = priv.public_key()
    monkeypatch.setenv("REMIXMATE_AUTH_ENABLED", "1")
    monkeypatch.setenv("SUPABASE_JWKS_URL", "https://project.test/auth/v1/.well-known/jwks.json")

    class _Signing:
        key = pub

    class _Client:
        def get_signing_key_from_jwt(self, token):  # noqa: ARG002
            return _Signing()

    monkeypatch.setattr(auth_mod, "_get_jwks_client", lambda: _Client())
    return priv


# ---------------------------------------------------------------------------
# auth disabled — the local default is untouched
# ---------------------------------------------------------------------------

def test_disabled_resolves_local_without_a_token(monkeypatch):
    monkeypatch.delenv("REMIXMATE_AUTH_ENABLED", raising=False)
    client = TestClient(_build_app())
    r = client.get("/whoami")
    assert r.status_code == 200
    assert r.json() == {"uid": "local"}


# ---------------------------------------------------------------------------
# auth enabled — allow
# ---------------------------------------------------------------------------

def test_valid_bearer_token_authenticates_and_sets_the_acting_user(signing_key):
    client = TestClient(_build_app())
    tok = _token(signing_key, sub="u-42")
    r = client.get("/whoami", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json() == {"uid": "u-42"}  # sync handler saw the middleware contextvar


def test_valid_session_cookie_authenticates(signing_key):
    """EventSource can't send Authorization — the SSE path reads the cookie."""
    client = TestClient(_build_app())
    tok = _token(signing_key, sub="u-cookie")
    client.cookies.set("sb-access-token", tok)
    r = client.get("/whoami")
    assert r.status_code == 200
    assert r.json() == {"uid": "u-cookie"}


def test_bearer_token_authenticates_a_post(signing_key):
    client = TestClient(_build_app())
    tok = _token(signing_key, sub="u-post")
    r = client.post("/mutate", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json() == {"uid": "u-post"}


def test_session_cookie_does_not_authenticate_a_post(signing_key):
    """CSRF: a cross-site form POST carries the cookie but can never set a
    Bearer header, so the cookie must not authenticate an unsafe method."""
    client = TestClient(_build_app())
    client.cookies.set("sb-access-token", _token(signing_key, sub="u-victim"))
    r = client.post("/mutate", content=b"x=y", headers={"Content-Type": "text/plain"})
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# auth enabled — deny (each paired with the allow above)
# ---------------------------------------------------------------------------

def test_no_token_is_401(signing_key):
    r = TestClient(_build_app()).get("/whoami")
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"


def test_expired_token_is_401(signing_key):
    tok = _token(signing_key, exp_delta=-10)
    r = TestClient(_build_app()).get("/whoami", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401


def test_wrong_audience_is_401(signing_key):
    tok = _token(signing_key, aud="some-other-service")
    r = TestClient(_build_app()).get("/whoami", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401


def test_token_signed_by_a_different_key_is_401(signing_key):
    other = _make_key()  # not the key the stub verifies against
    tok = _token(other, sub="attacker")
    r = TestClient(_build_app()).get("/whoami", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401


def test_malformed_token_is_401(signing_key):
    r = TestClient(_build_app()).get("/whoami", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401


def test_token_without_sub_is_401(signing_key):
    # PyJWT's require=["sub"] rejects it before we even look
    tok = jwt.encode(
        {"aud": _AUD, "role": "authenticated", "iat": int(time.time()), "exp": int(time.time()) + 60},
        signing_key,
        algorithm="RS256",
    )
    r = TestClient(_build_app()).get("/whoami", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401


def test_non_authenticated_role_is_401(signing_key):
    """The Supabase anon / service_role keys carry a different role — reject them
    even though the signature and audience check out."""
    for bad_role in ("anon", "service_role"):
        tok = _token(signing_key, role=bad_role)
        r = TestClient(_build_app()).get("/whoami", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 401, bad_role


# ---------------------------------------------------------------------------
# health probes stay open; misconfig is 500 not 401
# ---------------------------------------------------------------------------

def test_health_probes_stay_open_when_auth_is_on(signing_key):
    client = TestClient(_build_app())
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code in (200, 503)


def test_auth_on_but_no_jwks_url_is_500(monkeypatch):
    monkeypatch.setenv("REMIXMATE_AUTH_ENABLED", "1")
    monkeypatch.delenv("SUPABASE_JWKS_URL", raising=False)
    # a syntactically valid token, but the server can't verify it → misconfig
    tok = _token(_make_key())
    r = TestClient(_build_app()).get("/whoami", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 500


# ---------------------------------------------------------------------------
# the REAL app — proves main.py actually wires the gate onto every route
# ---------------------------------------------------------------------------

def test_the_real_app_gates_its_routes(signing_key):
    """test_new_features::TestHealthEndpoints is the only other safe-subset test
    touching scripts.api.main.app and it only hits the open /health*. This is
    the check that main.py's `dependencies=[Depends(require_user)]` and the
    `auth_context_middleware` registration are live — strip either line and the
    401 assert below goes red."""
    from scripts.api.main import app

    c = TestClient(app)  # no `with` — skip lifespan, like test_new_features

    assert c.get("/jobs").status_code == 401           # gated, no token
    assert c.get("/health/live").status_code == 200    # open probe, still fine

    tok = _token(signing_key, sub="u-real")
    assert c.get("/jobs", headers={"Authorization": f"Bearer {tok}"}).status_code == 200
