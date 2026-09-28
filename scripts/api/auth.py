"""
scripts/api/auth.py — Per-request / per-job user scoping + Supabase JWT auth.

Two layers, added in two phases of the hosted migration:

Phase 3 (contextvar scoping)
    Every request and every background job runs "as" a user id held in
    ``current_user`` (a contextvar, default ``"local"``). The stores in
    ``scripts/core`` read it via ``get_user_id()`` and filter by it.

    Load-bearing rule: ``ThreadPoolExecutor`` does not copy context into its
    worker threads, so ``scripts/api/jobs.py::_wrapper`` re-``set``s this var
    from the job's stamped ``meta["user_id"]`` before the task runs.

Phase 4 (authentication)
    When ``REMIXMATE_AUTH_ENABLED`` is truthy, incoming requests must carry a
    valid Supabase access token (RS256, verified against the project JWKS at
    ``SUPABASE_JWKS_URL``). ``auth_context_middleware`` (wired in main.py) sets
    ``current_user`` from the token's ``sub`` for the request; ``require_user``
    is the dependency that 401s an unauthenticated request. With the flag off
    every request resolves to ``"local"`` and no token is required, so local
    single-user dev and the test suite are unchanged.

    The contextvar MUST be set in the middleware, not the dependency: FastAPI
    runs sync dependencies in their own ``run_in_threadpool`` context, so a
    ``.set()`` there does not reach a sibling sync endpoint. The middleware
    runs in the request task, whose context every downstream handler copies.
    This mirrors ``request_id_middleware`` in main.py.
"""

from __future__ import annotations

import contextvars
import os
from typing import Optional

from fastapi import HTTPException, Request, status

DEFAULT_USER = "local"

current_user: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_user", default=DEFAULT_USER
)

# --- env / config ----------------------------------------------------------

_AUTH_ENABLED_ENV = "REMIXMATE_AUTH_ENABLED"
_JWKS_URL_ENV = "SUPABASE_JWKS_URL"
_AUDIENCE_ENV = "SUPABASE_JWT_AUD"          # Supabase default: "authenticated"
_ISSUER_ENV = "SUPABASE_JWT_ISS"           # optional: https://<ref>.supabase.co/auth/v1
_COOKIE_NAME_ENV = "SUPABASE_COOKIE_NAME"   # raw-JWT session cookie for EventSource
_ADMIN_EMAILS_ENV = "ADMIN_EMAILS"          # comma-separated; Phase 8 admin routes
_ADMIN_ALLOW_UNVERIFIED_ENV = "ADMIN_ALLOW_UNVERIFIED"  # "1" skips the email_verified check
_DEFAULT_AUDIENCE = "authenticated"
_DEFAULT_COOKIE_NAME = "sb-access-token"
_REQUIRED_ROLE = "authenticated"           # reject the anon / service_role keys
_TRUTHY = ("1", "true", "yes", "on")


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def auth_enabled() -> bool:
    """Whether Supabase JWT verification is active. Read per request so a test
    (or an operator) can flip it without a restart."""
    return _truthy(_AUTH_ENABLED_ENV)


def admin_emails() -> set[str]:
    """Lowercased admin emails from ``ADMIN_EMAILS`` (comma-separated)."""
    raw = os.environ.get(_ADMIN_EMAILS_ENV, "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


# ---------------------------------------------------------------------------
# Phase 3 — contextvar accessors
# ---------------------------------------------------------------------------

def set_user_id(user_id: str) -> None:
    """Set the acting user for the current context (request or worker thread)."""
    current_user.set(user_id)


def get_user_id() -> str:
    """Return the acting user id, or ``DEFAULT_USER`` outside any set context."""
    return current_user.get()


# ---------------------------------------------------------------------------
# Phase 4 — Supabase RS256 token verification
# ---------------------------------------------------------------------------

# Lazily built PyJWKClient, re-created if SUPABASE_JWKS_URL changes (tests).
_jwks_client = None
_jwks_client_url: Optional[str] = None


def _get_jwks_client():
    """The project's JWKS client. Raises HTTPException(500) if auth is on but
    no JWKS URL is configured — a deploy misconfiguration, not a client error."""
    global _jwks_client, _jwks_client_url

    url = os.environ.get(_JWKS_URL_ENV, "").strip()
    if not url:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="auth is enabled but SUPABASE_JWKS_URL is not set",
        )
    if _jwks_client is None or _jwks_client_url != url:
        try:
            import jwt  # PyJWT[crypto]
        except ModuleNotFoundError as exc:  # pragma: no cover - install-time only
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="auth is enabled but PyJWT is not installed",
            ) from exc
        # PyJWKClient caches keys in-process; lifespan bounds a key-rotation lag.
        _jwks_client = jwt.PyJWKClient(url, cache_keys=True, lifespan=3600)
        _jwks_client_url = url
    return _jwks_client


def _verify_token_claims(token: str) -> dict:
    """Verify a Supabase RS256 access token; return its claims dict.

    Raises HTTPException(401) for anything wrong with the token itself
    (signature, expiry, audience, issuer, role, missing claims) and
    HTTPException(500) for a server-side misconfiguration (no JWKS URL, PyJWT
    missing).
    """
    import jwt  # PyJWT — import here so a no-auth deploy never needs it

    audience = os.environ.get(_AUDIENCE_ENV, "").strip() or _DEFAULT_AUDIENCE
    issuer = os.environ.get(_ISSUER_ENV, "").strip() or None  # optional pin
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
            options={"require": ["exp", "sub"], "verify_iss": issuer is not None},
        )
    except HTTPException:
        raise
    except Exception as exc:  # jwt.*Error, PyJWKClientError, malformed input
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail=f"invalid authentication token: {exc}",
        ) from exc

    # An end-user access token carries role="authenticated"; the anon /
    # service_role keys do not. Reject anything that is not a logged-in user.
    if claims.get("role") != _REQUIRED_ROLE:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="token is not an authenticated-user token"
        )

    sub = claims.get("sub")
    if not sub or not isinstance(sub, str):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="token has no subject")
    return claims


def _verify_token(token: str) -> str:
    """Verify a Supabase RS256 access token, return its ``sub`` (the user id)."""
    return _verify_token_claims(token)["sub"]


def _extract_token(request: Request) -> Optional[str]:
    """Bearer header first (XHR / fetch), then a raw-JWT session cookie
    (EventSource cannot send an Authorization header). The Supabase SSR
    chunked-cookie format is the frontend commit's concern; here the cookie is
    expected to hold the access token verbatim."""
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        tok = header[len("Bearer "):].strip()
        return tok or None
    cookie_name = os.environ.get(_COOKIE_NAME_ENV, "").strip() or _DEFAULT_COOKIE_NAME
    return request.cookies.get(cookie_name) or None


def _authenticate(request: Request) -> Optional[str]:
    """Return the verified user id for this request, or ``None`` if no token was
    supplied. Raises HTTPException(401) if a token was supplied but is invalid,
    HTTPException(500) on misconfiguration. Caller decides what ``None`` means."""
    token = _extract_token(request)
    if token is None:
        return None
    return _verify_token(token)


def _authenticate_claims(request: Request) -> Optional[dict]:
    """Like ``_authenticate`` but returns full claims (for admin email checks)."""
    token = _extract_token(request)
    if token is None:
        return None
    return _verify_token_claims(token)


# ---------------------------------------------------------------------------
# Wiring — middleware sets the contextvar, dependency gates the route
# ---------------------------------------------------------------------------

async def auth_context_middleware(request: Request, call_next):
    """Resolve the acting user once per request and pin it to ``current_user``
    so sync handlers and the job store see it. Never 401s here — an unauth'd
    request is left as ``"local"`` and stopped later by ``require_user`` on the
    protected routes (``/health*`` and the docs stay open).

    INVARIANT: an open route (one NOT behind ``require_user``) must not read
    per-user state, because with auth on an unauthenticated request reaches it
    as ``"local"``. Today the only open routes are ``/health*`` (queries jobs
    with ``user_id="*"``) and the root pointer (touches no data). Any new open
    route that reads a user's mixes / crates / jobs is a cross-tenant leak.

    Token verification (JWKS fetch on a cold cache, signature check) is blocking
    I/O, so it runs in a worker thread; the contextvar is then set back here in
    the request task where downstream handlers can copy it.
    """
    if not auth_enabled():
        current_user.set(DEFAULT_USER)
        request.state.auth_uid = DEFAULT_USER
        return await call_next(request)

    from starlette.concurrency import run_in_threadpool

    try:
        uid = await run_in_threadpool(_authenticate, request)
    except HTTPException:
        uid = None  # invalid token — require_user will 401 the gated routes
    current_user.set(uid or DEFAULT_USER)
    request.state.auth_uid = uid
    return await call_next(request)


def _reject_if_flagged(uid: str) -> None:
    """403 when the acting user is on the flagged-users list (Phase 8)."""
    from scripts.core.legal_store import is_flagged

    if is_flagged(uid):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="account flagged — contact support",
        )


def require_user(request: Request) -> str:
    """FastAPI dependency for every route that touches per-user data.

    * ``REMIXMATE_AUTH_ENABLED`` off → ``"local"``, no token needed.
    * on + valid token              → the token's ``sub``.
    * on + missing / invalid token  → 401.
    * flagged user (any mode)       → 403.
    """
    if not auth_enabled():
        current_user.set(DEFAULT_USER)
        _reject_if_flagged(DEFAULT_USER)
        return DEFAULT_USER

    uid = _authenticate(request)  # raises 401 on a bad token, 500 on misconfig
    if not uid:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    _reject_if_flagged(uid)
    current_user.set(uid)
    return uid


def require_admin(request: Request) -> str:
    """FastAPI dependency for admin-only routes (Phase 8).

    Requires auth on, a valid token, and the token's ``email`` claim in
    ``ADMIN_EMAILS``. Returns the admin's ``sub``. When auth is off, only
    succeeds if ``local`` is somehow listed — operators should leave auth on
    in production; tests set ``ADMIN_EMAILS`` + a signed token with email.
    """
    if not auth_enabled():
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="admin routes require authentication",
        )

    claims = _authenticate_claims(request)
    if not claims:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    email = (claims.get("email") or "").strip().lower()
    if not email or email not in admin_emails():
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="admin access required")

    allow_unverified = os.environ.get(_ADMIN_ALLOW_UNVERIFIED_ENV, "").strip() == "1"
    if not allow_unverified and not _email_verified(claims):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="admin email is not verified")

    uid = claims["sub"]
    _reject_if_flagged(uid)
    current_user.set(uid)
    return uid


def _email_verified(claims: dict) -> bool:
    """True when the token carries ``email_verified: true`` at the top level or
    under ``user_metadata`` (Supabase puts it in either place)."""
    if claims.get("email_verified") is True:
        return True
    meta = claims.get("user_metadata")
    return isinstance(meta, dict) and meta.get("email_verified") is True


def require_operator(request: Request) -> str:
    """FastAPI dependency for catalog mutating routes (delete, prune, evict,
    initialize, analysis and stem jobs over the shared library).

    * ``REMIXMATE_AUTH_ENABLED`` off → ``"local"`` (single operator dev box).
    * on                             → ``require_admin``.
    """
    if not auth_enabled():
        return DEFAULT_USER
    return require_admin(request)
