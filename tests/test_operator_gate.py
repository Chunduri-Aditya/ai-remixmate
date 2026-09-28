"""
tests/test_operator_gate.py — B2.3 operator gate + the B2.8 admin email rule.

Spec (plans/audit-the-project-and-sorted-flask.md, B2.3 / B2.8):

* ``auth.require_operator`` returns ``"local"`` when auth is disabled and is
  ``require_admin`` otherwise.
* It is applied as a route dependency to the fourteen catalog mutating routes
  listed in ``ROUTES`` below.
* ``/library/initialize`` also 503s unless ``REMIXMATE_STEM_SPLIT_ENABLED`` is
  truthy (it runs Demucs over the whole library).
* ``require_admin`` additionally needs a verified email claim
  (``email_verified`` top level or under ``user_metadata``) unless
  ``ADMIN_ALLOW_UNVERIFIED=1``.

How the routes are exercised without touching the shared catalog: the real
route table (``scripts.api.routes.router``) is resolved through FastAPI's own
flattened iterator, and each target route is re-registered here with the SAME
declared dependencies (``dependencies=[...]`` on the decorator plus any
``Depends`` in the handler signature) but a stub handler that returns 200.
So the gate under test is whatever the real route declares, and a passing
request proves it reached the handler without running Demucs or ``rmtree``.

Tokens are synthetic RS256, ``auth._get_jwks_client`` stubbed, as in
tests/test_auth.py.
"""

from __future__ import annotations

import inspect
import time

import pytest

pytest.importorskip("jwt", reason="PyJWT not installed")
pytest.importorskip("cryptography", reason="cryptography not installed")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.params import Depends as DependsParam
from fastapi.routing import APIRoute, iter_route_contexts
from fastapi.testclient import TestClient

from scripts.api import auth as auth_mod
from scripts.api.auth import DEFAULT_USER, auth_context_middleware, require_admin, require_user
from scripts.api.routes import router as real_router

_AUD = "authenticated"
_ADMIN_EMAIL = "admin@example.com"
_USER_EMAIL = "user@example.com"

# (method, concrete request path, route template) — the B2.3 list verbatim.
ROUTES = [
    ("DELETE", "/library/some-song", "/library/{name}"),
    ("POST", "/library/delete-batch", "/library/delete-batch"),
    ("POST", "/library/storage/prune", "/library/storage/prune"),
    ("POST", "/library/storage/evict", "/library/storage/evict"),
    ("POST", "/library/initialize", "/library/initialize"),
    ("POST", "/library/analyze-missing", "/library/analyze-missing"),
    ("POST", "/library/scan-broken", "/library/scan-broken"),
    ("POST", "/library/calibrate-lufs", "/library/calibrate-lufs"),
    ("POST", "/library/build-clap-index", "/library/build-clap-index"),
    ("POST", "/index/rebuild", "/index/rebuild"),
    ("POST", "/stems/split", "/stems/split"),
    ("POST", "/stems/split-batch", "/stems/split-batch"),
    ("POST", "/stems/compress", "/stems/compress"),
    ("POST", "/stems/compress-batch", "/stems/compress-batch"),
]
_ROUTE_IDS = [f"{m} {t}" for m, _, t in ROUTES]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(priv, *, sub, email, exp_delta=3600, **extra):
    now = int(time.time())
    payload = {"sub": sub, "aud": _AUD, "role": "authenticated", "email": email,
               "iat": now, "exp": now + exp_delta, **extra}
    return jwt.encode(payload, priv, algorithm="RS256")


def _bearer(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}


def _real_route(method: str, concrete_path: str) -> APIRoute:
    """The APIRoute FastAPI would dispatch `method concrete_path` to."""
    for rc in iter_route_contexts(real_router.routes):
        if not isinstance(rc.original_route, APIRoute):
            continue
        if method in rc.methods and rc.path_regex.match(concrete_path):
            return rc.original_route
    raise AssertionError(f"no real route matches {method} {concrete_path}")


def _declared_dependencies(route: APIRoute) -> list:
    """Route level ``dependencies=[...]`` plus every ``Depends`` default in the
    handler signature, so the test accepts either way of declaring the gate."""
    deps = list(route.dependencies)
    for param in inspect.signature(route.endpoint).parameters.values():
        if isinstance(param.default, DependsParam):
            deps.append(param.default)
    return deps


def _operator_dependency():
    """``auth.require_operator`` if present, else a dependency that answers 501
    so the missing function surfaces as a failed assertion, not an AttributeError."""
    fn = getattr(auth_mod, "require_operator", None)
    if fn is not None:
        return fn

    def _missing() -> str:
        raise HTTPException(status_code=501, detail="auth.require_operator is not implemented")

    return _missing


def _build_app() -> FastAPI:
    app = FastAPI()
    app.middleware("http")(auth_context_middleware)
    gated = APIRouter()

    def _make_stub(name: str):
        def _stub():
            return {"ok": True, "handler": name}
        _stub.__name__ = f"stub_{name}"
        return _stub

    for method, concrete, template in ROUTES:
        real = _real_route(method, concrete)
        gated.add_api_route(
            template,
            _make_stub(real.endpoint.__name__),
            methods=[method],
            dependencies=_declared_dependencies(real),
        )

    # unit probes for the two dependencies themselves
    @gated.get("/_probe/operator")
    def _operator_probe(uid: str = Depends(_operator_dependency())):
        return {"uid": uid}

    @gated.get("/_probe/admin")
    def _admin_probe(uid: str = Depends(require_admin)):
        return {"uid": uid}

    # same shape as scripts/api/main.py: everything behind require_user
    app.include_router(gated, dependencies=[Depends(require_user)])
    return app


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    auth_mod.current_user.set(DEFAULT_USER)
    auth_mod._jwks_client = None
    auth_mod._jwks_client_url = None
    monkeypatch.delenv("ADMIN_ALLOW_UNVERIFIED", raising=False)
    # the two /stems/split routes already carry the stem split flag; keep it on
    # so a 503 cannot masquerade as "not 403" in the gate tests below.
    monkeypatch.setenv("REMIXMATE_STEM_SPLIT_ENABLED", "1")
    yield
    auth_mod.current_user.set(DEFAULT_USER)


@pytest.fixture
def signing_key(monkeypatch):
    """Auth on, JWKS stubbed to the generated public key, ADMIN_EMAILS set."""
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


@pytest.fixture
def auth_off(monkeypatch):
    monkeypatch.delenv("REMIXMATE_AUTH_ENABLED", raising=False)


@pytest.fixture
def user_headers(signing_key):
    return _bearer(_token(signing_key, sub="u-user", email=_USER_EMAIL, email_verified=True))


@pytest.fixture
def admin_headers(signing_key):
    return _bearer(_token(signing_key, sub="u-admin", email=_ADMIN_EMAIL, email_verified=True))


@pytest.fixture
def unverified_admin_headers(signing_key):
    # admin email, but no email_verified claim anywhere
    return _bearer(_token(signing_key, sub="u-admin", email=_ADMIN_EMAIL))


@pytest.fixture
def client():
    return TestClient(_build_app())


# ---------------------------------------------------------------------------
# require_operator itself
# ---------------------------------------------------------------------------

def test_require_operator_exists_on_auth_module():
    assert getattr(auth_mod, "require_operator", None) is not None, \
        "scripts.api.auth has no require_operator"


def test_require_operator_returns_local_when_auth_is_off(auth_off):
    fn = getattr(auth_mod, "require_operator", None)
    assert fn is not None, "scripts.api.auth has no require_operator"
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [],
             "query_string": b""}
    assert fn(Request(scope)) == "local"


def test_require_operator_is_require_admin_when_auth_is_on(client, user_headers, admin_headers):
    # deny: signed in, not an admin
    r = client.get("/_probe/operator", headers=user_headers)
    assert r.status_code == 403, r.text
    # allow: verified admin, and the dependency yields the admin's sub
    r = client.get("/_probe/operator", headers=admin_headers)
    assert r.status_code == 200, r.text
    assert r.json() == {"uid": "u-admin"}


def test_require_operator_without_a_token_is_401_when_auth_is_on(client, signing_key):
    r = client.get("/_probe/operator")
    assert r.status_code == 401, r.text


# ---------------------------------------------------------------------------
# the gate on every route in the B2.3 list
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("method", "path", "_template"), ROUTES, ids=_ROUTE_IDS)
def test_user_token_is_403(client, user_headers, method, path, _template):
    r = client.request(method, path, headers=user_headers)
    assert r.status_code == 403, f"{method} {path} -> {r.status_code} {r.text}"


@pytest.mark.parametrize(("method", "path", "_template"), ROUTES, ids=_ROUTE_IDS)
def test_verified_admin_token_reaches_the_handler(client, admin_headers, method, path, _template):
    r = client.request(method, path, headers=admin_headers)
    assert r.status_code != 403, f"{method} {path} -> {r.status_code} {r.text}"
    assert r.status_code == 200, f"{method} {path} -> {r.status_code} {r.text}"
    assert r.json()["ok"] is True


@pytest.mark.parametrize(("method", "path", "_template"), ROUTES, ids=_ROUTE_IDS)
def test_auth_off_reaches_the_handler(client, auth_off, method, path, _template):
    r = client.request(method, path)
    assert r.status_code != 403, f"{method} {path} -> {r.status_code} {r.text}"
    assert r.status_code == 200, f"{method} {path} -> {r.status_code} {r.text}"


@pytest.mark.parametrize(("method", "path", "_template"), ROUTES, ids=_ROUTE_IDS)
def test_unverified_admin_token_is_403(client, unverified_admin_headers, method, path, _template):
    r = client.request(method, path, headers=unverified_admin_headers)
    assert r.status_code == 403, f"{method} {path} -> {r.status_code} {r.text}"


# ---------------------------------------------------------------------------
# /library/initialize also carries the stem split flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", [None, "", "0", "false", "off"], ids=["unset", "empty", "0", "false", "off"])
def test_initialize_is_503_when_stem_split_flag_is_off(client, auth_off, monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("REMIXMATE_STEM_SPLIT_ENABLED", raising=False)
    else:
        monkeypatch.setenv("REMIXMATE_STEM_SPLIT_ENABLED", flag)
    r = client.post("/library/initialize")
    assert r.status_code == 503, r.text


def test_initialize_passes_when_stem_split_flag_is_on(client, auth_off, monkeypatch):
    monkeypatch.setenv("REMIXMATE_STEM_SPLIT_ENABLED", "1")
    r = client.post("/library/initialize")
    assert r.status_code == 200, r.text


def test_initialize_flag_off_still_503_for_a_verified_admin(client, admin_headers, monkeypatch):
    monkeypatch.delenv("REMIXMATE_STEM_SPLIT_ENABLED", raising=False)
    r = client.post("/library/initialize", headers=admin_headers)
    assert r.status_code == 503, r.text


# ---------------------------------------------------------------------------
# require_admin: the verified email rule (B2.8)
# ---------------------------------------------------------------------------

def test_admin_with_top_level_email_verified_is_allowed(client, admin_headers):
    r = client.get("/_probe/admin", headers=admin_headers)
    assert r.status_code == 200, r.text
    assert r.json() == {"uid": "u-admin"}


def test_admin_with_user_metadata_email_verified_is_allowed(client, signing_key):
    tok = _token(signing_key, sub="u-admin", email=_ADMIN_EMAIL,
                 user_metadata={"email_verified": True})
    r = client.get("/_probe/admin", headers=_bearer(tok))
    assert r.status_code == 200, r.text
    assert r.json() == {"uid": "u-admin"}


def test_admin_without_email_verified_claim_is_403(client, unverified_admin_headers):
    r = client.get("/_probe/admin", headers=unverified_admin_headers)
    assert r.status_code == 403, r.text


def test_admin_with_email_verified_false_is_403(client, signing_key):
    tok = _token(signing_key, sub="u-admin", email=_ADMIN_EMAIL, email_verified=False)
    r = client.get("/_probe/admin", headers=_bearer(tok))
    assert r.status_code == 403, r.text


def test_admin_with_user_metadata_email_verified_false_is_403(client, signing_key):
    tok = _token(signing_key, sub="u-admin", email=_ADMIN_EMAIL,
                 user_metadata={"email_verified": False})
    r = client.get("/_probe/admin", headers=_bearer(tok))
    assert r.status_code == 403, r.text


def test_admin_allow_unverified_env_skips_the_email_check(client, unverified_admin_headers, monkeypatch):
    monkeypatch.setenv("ADMIN_ALLOW_UNVERIFIED", "1")
    r = client.get("/_probe/admin", headers=unverified_admin_headers)
    assert r.status_code == 200, r.text
    assert r.json() == {"uid": "u-admin"}


def test_admin_allow_unverified_env_does_not_admit_a_non_admin(client, signing_key, monkeypatch):
    """The escape hatch relaxes the email verification only, never the allow list."""
    monkeypatch.setenv("ADMIN_ALLOW_UNVERIFIED", "1")
    tok = _token(signing_key, sub="u-user", email=_USER_EMAIL)
    r = client.get("/_probe/admin", headers=_bearer(tok))
    assert r.status_code == 403, r.text
