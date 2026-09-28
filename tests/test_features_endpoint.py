"""
tests/test_features_endpoint.py — B5.1 ``GET /features`` on the open system router.

Spec (plans/audit-the-project-and-sorted-flask.md, B5.1):

    GET /features on the open ``system`` router returns
    {auth, stem_split, generative, license_enforce, spectral_learning, operator}
    where ``operator`` is true when the caller passes ``require_operator``.

Contract pinned here (the builder implements against these):

* The route is declared on ``scripts.api.routers.system.router`` (not on the
  gated ``scripts.api.routes.router``), so on the real app it answers 200 with
  auth on and no token, exactly like ``/health/live``.
* Every value is a real ``bool`` and the key set is exactly the six above.
* Flag mapping, each read per request from the same reader the gated route uses:
    auth              -> scripts.api.auth.auth_enabled()          (REMIXMATE_AUTH_ENABLED)
    stem_split        -> REMIXMATE_STEM_SPLIT_ENABLED truthy       (routers/stems.py gate)
    generative        -> REMIXMATE_GENERATIVE_ENABLED truthy       (routers/generative.py gate)
    license_enforce   -> scripts.core.license.license_enforce_enabled()
    spectral_learning -> scripts.core.spectral_matching's reader   (default ON when unset)
    operator          -> True when auth.require_operator would admit the caller:
                         auth off -> True; auth on + verified admin -> True;
                         auth on + plain user -> False; auth on + no token -> False
                         (never a 401/403: the endpoint itself stays open).

Tokens are synthetic RS256 with ``auth._get_jwks_client`` stubbed, as in
tests/test_operator_gate.py.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from scripts.api import auth as auth_mod
from scripts.api.auth import DEFAULT_USER, auth_context_middleware
from scripts.api.routers import system

EXPECTED_KEYS = {"auth", "stem_split", "generative", "license_enforce", "spectral_learning", "operator"}

_FLAG_ENVS = (
    "REMIXMATE_AUTH_ENABLED",
    "REMIXMATE_STEM_SPLIT_ENABLED",
    "REMIXMATE_GENERATIVE_ENABLED",
    "REMIXMATE_LICENSE_ENFORCE",
    "REMIXMATE_SPECTRAL_LEARNING",
    "ADMIN_EMAILS",
    "ADMIN_ALLOW_UNVERIFIED",
    "SUPABASE_JWKS_URL",
)

_AUD = "authenticated"
_ADMIN_EMAIL = "admin@example.com"
_USER_EMAIL = "user@example.com"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_flags(monkeypatch):
    """Every flag unset before each test so a developer's shell cannot leak in."""
    for name in _FLAG_ENVS:
        monkeypatch.delenv(name, raising=False)
    auth_mod.current_user.set(DEFAULT_USER)
    auth_mod._jwks_client = None
    auth_mod._jwks_client_url = None
    yield
    auth_mod.current_user.set(DEFAULT_USER)


@pytest.fixture
def client() -> TestClient:
    """The system router mounted OPEN, same as scripts/api/main.py does."""
    app = FastAPI()
    app.middleware("http")(auth_context_middleware)
    app.include_router(system.router)
    return TestClient(app)


def _features_route() -> APIRoute | None:
    for r in system.router.routes:
        if isinstance(r, APIRoute) and r.path == "/features" and "GET" in r.methods:
            return r
    return None


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------

def test_features_route_is_declared_on_the_system_router():
    assert _features_route() is not None, (
        "GET /features is not declared on scripts.api.routers.system.router; "
        f"routes present: {[getattr(r, 'path', r) for r in system.router.routes]}"
    )


def test_features_returns_exactly_the_six_boolean_keys(client):
    r = client.get("/features")
    assert r.status_code == 200, f"{r.status_code} {r.text}"
    body = r.json()
    assert isinstance(body, dict), body
    assert set(body) == EXPECTED_KEYS, f"keys differ: got {sorted(body)}"
    non_bool = {k: v for k, v in body.items() if not isinstance(v, bool)}
    assert not non_bool, f"non boolean values: {non_bool}"


# ---------------------------------------------------------------------------
# flag mapping (defaults, then each flag flipped on its own)
# ---------------------------------------------------------------------------

def test_defaults_with_every_flag_unset(client):
    """Nothing set: auth off, split off, generative off, licence gate off,
    spectral learning ON (its reader defaults to true), operator True (auth off
    means require_operator returns "local")."""
    r = client.get("/features")
    assert r.status_code == 200, r.text
    assert r.json() == {
        "auth": False,
        "stem_split": False,
        "generative": False,
        "license_enforce": False,
        "spectral_learning": True,
        "operator": True,
    }


@pytest.mark.parametrize(
    ("env", "key"),
    [
        ("REMIXMATE_STEM_SPLIT_ENABLED", "stem_split"),
        ("REMIXMATE_GENERATIVE_ENABLED", "generative"),
        ("REMIXMATE_LICENSE_ENFORCE", "license_enforce"),
    ],
)
@pytest.mark.parametrize("truthy", ["1", "true", "yes", "TRUE"])
def test_truthy_env_turns_flag_on(client, monkeypatch, env, key, truthy):
    monkeypatch.setenv(env, truthy)
    body = client.get("/features").json()
    assert body[key] is True, f"{env}={truthy!r} -> {key}={body[key]!r}"


@pytest.mark.parametrize(
    ("env", "key"),
    [
        ("REMIXMATE_STEM_SPLIT_ENABLED", "stem_split"),
        ("REMIXMATE_GENERATIVE_ENABLED", "generative"),
        ("REMIXMATE_LICENSE_ENFORCE", "license_enforce"),
    ],
)
@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "off"])
def test_falsy_env_keeps_flag_off(client, monkeypatch, env, key, falsy):
    monkeypatch.setenv(env, falsy)
    body = client.get("/features").json()
    assert body[key] is False, f"{env}={falsy!r} -> {key}={body[key]!r}"


@pytest.mark.parametrize("falsy", ["0", "false", "no", "off", ""])
def test_spectral_learning_falsy_turns_it_off(client, monkeypatch, falsy):
    monkeypatch.setenv("REMIXMATE_SPECTRAL_LEARNING", falsy)
    body = client.get("/features").json()
    assert body["spectral_learning"] is False, f"REMIXMATE_SPECTRAL_LEARNING={falsy!r} -> {body['spectral_learning']!r}"


def test_spectral_learning_truthy_stays_on(client, monkeypatch):
    monkeypatch.setenv("REMIXMATE_SPECTRAL_LEARNING", "true")
    assert client.get("/features").json()["spectral_learning"] is True


def test_flags_are_read_per_request_not_at_import(client, monkeypatch):
    assert client.get("/features").json()["generative"] is False
    monkeypatch.setenv("REMIXMATE_GENERATIVE_ENABLED", "1")
    assert client.get("/features").json()["generative"] is True
    monkeypatch.delenv("REMIXMATE_GENERATIVE_ENABLED", raising=False)
    assert client.get("/features").json()["generative"] is False


# ---------------------------------------------------------------------------
# auth + operator mapping
# ---------------------------------------------------------------------------

def _make_key():
    rsa = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(priv, *, sub, email, **extra):
    jwt = pytest.importorskip("jwt", reason="PyJWT not installed")
    now = int(time.time())
    payload = {"sub": sub, "aud": _AUD, "role": "authenticated", "email": email,
               "iat": now, "exp": now + 3600, **extra}
    return jwt.encode(payload, priv, algorithm="RS256")


@pytest.fixture
def signing_key(monkeypatch):
    """Auth on, JWKS stubbed to the generated public key, ADMIN_EMAILS set."""
    pytest.importorskip("jwt", reason="PyJWT not installed")
    priv = _make_key()
    pub = priv.public_key()
    monkeypatch.setenv("REMIXMATE_AUTH_ENABLED", "1")
    monkeypatch.setenv("SUPABASE_JWKS_URL", "https://project.test/auth/v1/.well-known/jwks.json")
    monkeypatch.setenv("ADMIN_EMAILS", _ADMIN_EMAIL)

    class _Signing:
        key = pub

    class _Client:
        def get_signing_key_from_jwt(self, token):  # noqa: ARG002
            return _Signing()

    monkeypatch.setattr(auth_mod, "_get_jwks_client", lambda: _Client())
    return priv


def test_auth_off_reports_auth_false_and_operator_true(client):
    body = client.get("/features").json()
    assert body["auth"] is False
    assert body["operator"] is True


def test_auth_on_without_token_is_still_200_and_operator_false(client, signing_key):
    """The endpoint is open: an anonymous caller with auth on gets 200, not 401,
    and is told auth is on and they are not an operator."""
    r = client.get("/features")
    assert r.status_code == 200, f"{r.status_code} {r.text}"
    body = r.json()
    assert body["auth"] is True
    assert body["operator"] is False


def test_auth_on_plain_user_is_not_operator(client, signing_key):
    tok = _token(signing_key, sub="u-user", email=_USER_EMAIL, email_verified=True)
    r = client.get("/features", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200, f"{r.status_code} {r.text}"
    assert r.json()["operator"] is False


def test_auth_on_verified_admin_is_operator(client, signing_key):
    tok = _token(signing_key, sub="u-admin", email=_ADMIN_EMAIL, email_verified=True)
    r = client.get("/features", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200, f"{r.status_code} {r.text}"
    body = r.json()
    assert body["auth"] is True
    assert body["operator"] is True


def test_auth_on_unverified_admin_is_not_operator(client, signing_key):
    """require_admin needs email_verified (B2.8); the flag mirrors that rule."""
    tok = _token(signing_key, sub="u-admin", email=_ADMIN_EMAIL)
    r = client.get("/features", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200, f"{r.status_code} {r.text}"
    assert r.json()["operator"] is False


# ---------------------------------------------------------------------------
# the REAL app: mounted open (no require_user) like /health/live
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_real_app_serves_features_open_when_auth_is_on(monkeypatch):
    from scripts.api.main import app

    monkeypatch.setenv("REMIXMATE_AUTH_ENABLED", "1")
    monkeypatch.setenv("SUPABASE_JWKS_URL", "https://project.test/auth/v1/.well-known/jwks.json")
    c = TestClient(app)  # no `with`: skip lifespan, like tests/test_auth.py

    # control: a gated route 401s and the open health probe passes
    assert c.get("/health/live").status_code == 200
    assert c.get("/jobs").status_code == 401

    r = c.get("/features")
    assert r.status_code == 200, f"/features on the real app -> {r.status_code} {r.text}"
    body = r.json()
    assert set(body) == EXPECTED_KEYS
    assert body["auth"] is True
    assert body["operator"] is False
