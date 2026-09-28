"""
tests/test_admin_takedown.py — B2.8 admin takedown + B2.13 legal store schema.

Spec (plans/audit-the-project-and-sorted-flask.md, B2.8 / B2.13):

* ``DELETE /admin/mix/{job_id}`` resolves the session directory from
  ``result.output`` or ``result.stream_url`` (the real ``dj_<sid>`` and
  ``chain_<sid>`` layout written by task_dj_remix / task_dj_chain), refuses
  anything not under ``OUTPUTS_DIR``, loads the job through
  ``get_job_from_db`` when it is not in memory, cancels a RUNNING job first,
  persists the takedown, and emits the event to the owner.
* New ``DELETE /admin/user/{id}/flag`` (unflag).
* ``legal_store.init_schema()`` creates and commits the tables once;
  ``is_flagged`` is a plain SELECT with no DDL.

Storage: OUTPUTS_DIR, the jobs db and the legal db are all redirected to
``tmp_path`` per test. The admin router reads ``paths_mod.OUTPUTS_DIR`` by
attribute, so the monkeypatch on the module global is what the code sees.

Tokens are synthetic RS256 with ``auth._get_jwks_client`` stubbed, as in
tests/test_operator_gate.py. Admin tokens carry ``email_verified: true``.
"""

from __future__ import annotations

import re
import time
from contextlib import closing

import pytest

pytest.importorskip("jwt", reason="PyJWT not installed")
pytest.importorskip("cryptography", reason="cryptography not installed")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event, text

from scripts.api import auth as auth_mod
from scripts.api import jobs as job_store
from scripts.api.auth import DEFAULT_USER, auth_context_middleware, require_user
from scripts.api.routers import admin
from scripts.api.schemas import JobStatus
from scripts.core import db as db_mod
from scripts.core import legal_store
from scripts.core import paths as paths_mod

_AUD = "authenticated"
_ADMIN_EMAIL = "admin@example.com"
_USER_EMAIL = "user@example.com"
OWNER = "victim"


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


def _build_app() -> FastAPI:
    app = FastAPI()
    app.middleware("http")(auth_context_middleware)
    # same shape as scripts/api/main.py: everything behind require_user
    app.include_router(admin.router, dependencies=[Depends(require_user)])
    return app


def _job_dict(job_id: str, *, status, result, session_id: str, owner: str = OWNER) -> dict:
    return {
        "job_id":      job_id,
        "status":      status,
        "job_type":    "dj_remix",
        "created_at":  time.time(),
        "started_at":  None,
        "finished_at": None,
        "progress":    0.0,
        "message":     "Queued",
        "result":      result,
        "error":       None,
        "eta_sec":     None,
        "meta":        {"user_id": owner, "session_id": session_id},
    }


def _put_in_memory(job: dict) -> None:
    """Register a job the way the store does after create + persist."""
    with job_store._lock:
        job_store._jobs[job["job_id"]] = job
        with closing(job_store._get_conn()) as conn:
            job_store._upsert_row(conn, job)


def _put_in_db_only(job: dict) -> None:
    """Persist a row without touching ``_jobs`` (an evicted job)."""
    with closing(job_store._get_conn()) as conn:
        job_store._upsert_row(conn, job)
    assert job_store.get_job(job["job_id"]) is None


def _session(out_root, owner: str, prefix: str, sid: str):
    """Create the real on disk layout: outputs/<owner>/<prefix><sid>/<file>.wav"""
    out_dir = out_root / owner / f"{prefix}{sid}"
    out_dir.mkdir(parents=True)
    out_path = out_dir / f"{prefix}song_a_song_b.wav"
    out_path.write_bytes(b"RIFF")
    return out_dir, out_path


def _real_result(out_dir, out_path, sid: str) -> dict:
    """The result dict task_dj_remix / task_dj_chain persist."""
    return {
        "session_id": sid,
        "output":     str(out_path),
        "stream_url": f"/outputs/{out_dir.name}/{out_path.name}",
    }


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Fresh jobs db, legal db, OUTPUTS_DIR and auth context per test."""
    monkeypatch.setattr(job_store, "_DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(legal_store, "_DB_PATH", tmp_path / "legal.db")
    db_mod._engine_for.cache_clear()
    job_store._jobs.clear()
    job_store._cancelled.clear()
    job_store._init_db()
    monkeypatch.setattr(paths_mod, "OUTPUTS_DIR", tmp_path / "outputs")
    (tmp_path / "outputs").mkdir()
    auth_mod.current_user.set(DEFAULT_USER)
    auth_mod._jwks_client = None
    auth_mod._jwks_client_url = None
    monkeypatch.delenv("ADMIN_ALLOW_UNVERIFIED", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    yield
    job_store._jobs.clear()
    job_store._cancelled.clear()
    auth_mod.current_user.set(DEFAULT_USER)
    db_mod._engine_for.cache_clear()


@pytest.fixture
def out_root(tmp_path):
    return tmp_path / "outputs"


@pytest.fixture
def signing_key(monkeypatch):
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
def admin_headers(signing_key):
    return _bearer(_token(signing_key, sub="u-admin", email=_ADMIN_EMAIL, email_verified=True))


@pytest.fixture
def user_headers(signing_key):
    return _bearer(_token(signing_key, sub="u-user", email=_USER_EMAIL, email_verified=True))


@pytest.fixture
def client():
    return TestClient(_build_app())


@pytest.fixture
def sse_spy(monkeypatch):
    """Capture every (event_type, payload) the job store emits."""
    seen: list[tuple[str, dict]] = []

    def _hook(event_type: str, payload: dict) -> None:
        seen.append((event_type, payload))

    monkeypatch.setattr(job_store, "_sse_hook", _hook)
    return seen


# ---------------------------------------------------------------------------
# DELETE /admin/mix/{job_id}: the real layout is removed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix", ["dj_", "chain_"])
def test_real_layout_is_removed_from_result_output(client, admin_headers, out_root, prefix):
    sid = "a1b2c3"
    out_dir, out_path = _session(out_root, OWNER, prefix, sid)
    jid = f"job-{prefix}output"
    _put_in_memory(_job_dict(jid, status=JobStatus.DONE, session_id=sid,
                             result=_real_result(out_dir, out_path, sid)))

    r = client.delete(f"/admin/mix/{jid}", headers=admin_headers)
    assert r.status_code == 200, r.text
    assert not out_dir.exists(), f"{out_dir} survived the takedown"
    # sibling owner dir is untouched
    assert (out_root / OWNER).exists()


@pytest.mark.parametrize("prefix", ["dj_", "chain_"])
def test_real_layout_is_removed_from_stream_url_alone(client, admin_headers, out_root, prefix):
    """A row whose ``output`` is missing still resolves through ``stream_url``."""
    sid = "d4e5f6"
    out_dir, out_path = _session(out_root, OWNER, prefix, sid)
    jid = f"job-{prefix}stream"
    result = {"session_id": sid, "stream_url": f"/outputs/{out_dir.name}/{out_path.name}"}
    _put_in_memory(_job_dict(jid, status=JobStatus.DONE, session_id=sid, result=result))

    r = client.delete(f"/admin/mix/{jid}", headers=admin_headers)
    assert r.status_code == 200, r.text
    assert not out_dir.exists(), f"{out_dir} survived the takedown"


def test_takedown_uses_the_owner_dir_not_the_admin_dir(client, admin_headers, out_root):
    """``require_admin`` sets current_user to the admin; the session lives under the owner."""
    sid = "0wn3r"
    out_dir, out_path = _session(out_root, OWNER, "dj_", sid)
    admin_dir, _ = _session(out_root, "u-admin", "dj_", sid)  # decoy with the same sid
    jid = "job-owner-dir"
    result = {"session_id": sid, "stream_url": f"/outputs/{out_dir.name}/{out_path.name}"}
    _put_in_memory(_job_dict(jid, status=JobStatus.DONE, session_id=sid, result=result))

    r = client.delete(f"/admin/mix/{jid}", headers=admin_headers)
    assert r.status_code == 200, r.text
    assert not out_dir.exists(), "owner session survived"
    assert admin_dir.exists(), "the admin's own session was removed instead of the owner's"


# ---------------------------------------------------------------------------
# DELETE /admin/mix/{job_id}: anything outside OUTPUTS_DIR is refused
# ---------------------------------------------------------------------------

def _outside_cases(tmp_path):
    outside = tmp_path / "elsewhere" / "dj_evil"
    outside.mkdir(parents=True, exist_ok=True)
    victim_file = outside / "dj_x.wav"
    victim_file.write_bytes(b"RIFF")
    return outside, {
        "output_absolute":     {"session_id": "evil", "output": str(victim_file)},
        "stream_url_dotdot":   {"session_id": "evil",
                                "stream_url": f"/outputs/../../elsewhere/{outside.name}/{victim_file.name}"},
        "output_dir_absolute": {"session_id": "evil", "output_dir": str(outside)},
        "session_dir_dotdot":  {"session_id": f"../../elsewhere/{outside.name}"},
    }


@pytest.mark.parametrize("case", ["output_absolute", "stream_url_dotdot",
                                  "output_dir_absolute", "session_dir_dotdot"])
def test_outside_path_is_refused(client, admin_headers, tmp_path, case):
    outside, cases = _outside_cases(tmp_path)
    jid = f"job-outside-{case}"
    _put_in_memory(_job_dict(jid, status=JobStatus.DONE, session_id="evil", result=cases[case]))

    r = client.delete(f"/admin/mix/{jid}", headers=admin_headers)
    assert outside.exists(), f"{case}: a path outside OUTPUTS_DIR was removed: {outside}"
    assert (outside / "dj_x.wav").exists(), f"{case}: file outside OUTPUTS_DIR was removed"
    if r.status_code == 200:
        assert r.json().get("removed_dir") is None, \
            f"{case}: response claims a removal outside OUTPUTS_DIR: {r.text}"


# ---------------------------------------------------------------------------
# DELETE /admin/mix/{job_id}: a job only in the database is found
# ---------------------------------------------------------------------------

def test_db_only_job_is_found_and_taken_down(client, admin_headers, out_root):
    sid = "evicted1"
    out_dir, out_path = _session(out_root, OWNER, "dj_", sid)
    jid = "job-db-only"
    _put_in_db_only(_job_dict(jid, status=JobStatus.DONE, session_id=sid,
                              result=_real_result(out_dir, out_path, sid)))
    assert job_store.get_job_from_db(jid) is not None  # the row is there

    r = client.delete(f"/admin/mix/{jid}", headers=admin_headers)
    assert r.status_code == 200, r.text
    assert not out_dir.exists(), "evicted job's session survived"
    row = job_store.get_job_from_db(jid)
    assert row is not None
    assert row.get("error") == "admin_takedown", f"takedown not persisted: {row}"
    assert row.get("status") != "done"


def test_unknown_job_is_404(client, admin_headers):
    r = client.delete("/admin/mix/does-not-exist", headers=admin_headers)
    assert r.status_code == 404, r.text


# ---------------------------------------------------------------------------
# DELETE /admin/mix/{job_id}: RUNNING is cancelled first; takedown persisted; owner notified
# ---------------------------------------------------------------------------

def test_running_job_is_cancelled_before_takedown(client, admin_headers, out_root, sse_spy):
    sid = "run1"
    jid = "job-running"
    job = _job_dict(jid, status=JobStatus.RUNNING, session_id=sid, result=None)
    job["started_at"] = time.time()
    _put_in_memory(job)

    r = client.delete(f"/admin/mix/{jid}", headers=admin_headers)
    assert r.status_code == 200, r.text
    assert jid in job_store._cancelled, "running job was not cancelled before the takedown"
    types = [t for t, _ in sse_spy]
    assert "job_cancelled" in types, f"no job_cancelled event, saw {types}"
    final = job_store.get_job(jid)
    assert final is not None
    assert final.get("error") == "admin_takedown"


def test_takedown_is_persisted_to_the_database(client, admin_headers, out_root):
    sid = "persist1"
    out_dir, out_path = _session(out_root, OWNER, "dj_", sid)
    jid = "job-persist"
    _put_in_memory(_job_dict(jid, status=JobStatus.DONE, session_id=sid,
                             result=_real_result(out_dir, out_path, sid)))

    r = client.delete(f"/admin/mix/{jid}", headers=admin_headers)
    assert r.status_code == 200, r.text
    row = job_store.get_job_from_db(jid)
    assert row is not None
    assert row.get("error") == "admin_takedown", f"db row not updated: {row}"
    assert row.get("status") == "failed", f"db row not updated: {row}"


def test_takedown_emits_event_to_the_owner(client, admin_headers, out_root, sse_spy):
    sid = "emit1"
    out_dir, out_path = _session(out_root, OWNER, "dj_", sid)
    jid = "job-emit"
    _put_in_memory(_job_dict(jid, status=JobStatus.DONE, session_id=sid,
                             result=_real_result(out_dir, out_path, sid)))

    r = client.delete(f"/admin/mix/{jid}", headers=admin_headers)
    assert r.status_code == 200, r.text
    takedown_events = [
        (t, p) for t, p in sse_spy
        if p.get("job_id") == jid and p.get("error") == "admin_takedown"
    ]
    assert takedown_events, f"no takedown event emitted, saw {[t for t, _ in sse_spy]}"
    # the broadcaster routes by meta.user_id: the frame must be addressed to the owner
    for _, payload in takedown_events:
        assert (payload.get("meta") or {}).get("user_id") == OWNER, payload


def test_non_admin_cannot_take_down(client, user_headers, out_root):
    sid = "deny1"
    out_dir, out_path = _session(out_root, OWNER, "dj_", sid)
    jid = "job-deny"
    _put_in_memory(_job_dict(jid, status=JobStatus.DONE, session_id=sid,
                             result=_real_result(out_dir, out_path, sid)))
    r = client.delete(f"/admin/mix/{jid}", headers=user_headers)
    assert r.status_code == 403, r.text
    assert out_dir.exists()


# ---------------------------------------------------------------------------
# DELETE /admin/user/{id}/flag (unflag)
# ---------------------------------------------------------------------------

def test_unflag_route_clears_the_flag(client, admin_headers):
    legal_store.flag_user(OWNER, reason="abuse", flagged_by="u-admin")
    assert legal_store.is_flagged(OWNER)

    r = client.delete(f"/admin/user/{OWNER}/flag", headers=admin_headers)
    assert r.status_code == 200, r.text
    assert legal_store.is_flagged(OWNER) is False
    body = r.json()
    assert body.get("user_id") == OWNER
    assert body.get("flagged") is False


def test_unflag_route_rejects_non_admin(client, user_headers):
    legal_store.flag_user(OWNER, reason="abuse", flagged_by="u-admin")
    r = client.delete(f"/admin/user/{OWNER}/flag", headers=user_headers)
    assert r.status_code == 403, r.text
    assert legal_store.is_flagged(OWNER) is True


# ---------------------------------------------------------------------------
# legal_store.init_schema() and is_flagged with no DDL
# ---------------------------------------------------------------------------

def _tables() -> set[str]:
    engine = db_mod.get_engine(str(legal_store._DB_PATH))
    with engine.connect() as c:  # a fresh connection: only committed DDL is visible
        rows = c.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
    return {r[0] for r in rows}


def test_init_schema_creates_and_commits_both_tables():
    init = getattr(legal_store, "init_schema", None)
    assert init is not None, "scripts.core.legal_store has no init_schema()"
    assert not {"dmca_reports", "flagged_users"} & _tables()
    init()
    assert {"dmca_reports", "flagged_users"} <= _tables()
    init()  # idempotent: a second call neither raises nor drops rows
    legal_store.flag_user("u1", reason="x")
    init()
    assert legal_store.is_flagged("u1")


def _capture_statements(engine):
    """Spy on every statement the engine's connections send to the DBAPI cursor."""
    seen: list[str] = []

    def _spy(conn, cursor, statement, parameters, context, executemany):  # noqa: ARG001
        seen.append(statement)

    event.listen(engine, "before_cursor_execute", _spy)
    return seen, _spy


_DDL = re.compile(r"^\s*(CREATE|ALTER|DROP)\b", re.IGNORECASE)


def test_is_flagged_issues_no_ddl():
    init = getattr(legal_store, "init_schema", None)
    assert init is not None, "scripts.core.legal_store has no init_schema()"
    init()

    engine = db_mod.get_engine(str(legal_store._DB_PATH))
    seen, spy = _capture_statements(engine)
    try:
        assert legal_store.is_flagged("nobody") is False
        legal_store.flag_user("someone", reason="x")
        seen.clear()
        assert legal_store.is_flagged("someone") is True
    finally:
        event.remove(engine, "before_cursor_execute", spy)

    selects = [s for s in seen if re.match(r"^\s*SELECT\b", s, re.IGNORECASE)]
    assert selects, f"spy saw no SELECT from is_flagged; captured {seen}"  # the spy works
    ddl = [s for s in seen if _DDL.match(s)]
    assert not ddl, f"is_flagged issued DDL: {ddl}"


def test_is_flagged_on_a_fresh_db_does_not_create_the_schema():
    """Without init_schema, is_flagged must not silently build the tables."""
    assert not {"flagged_users"} & _tables()
    engine = db_mod.get_engine(str(legal_store._DB_PATH))
    seen, spy = _capture_statements(engine)
    try:
        try:
            legal_store.is_flagged("nobody")
        except Exception:  # noqa: BLE001  a missing table is an acceptable outcome here
            pass
    finally:
        event.remove(engine, "before_cursor_execute", spy)
    ddl = [s for s in seen if _DDL.match(s)]
    assert not ddl, f"is_flagged issued DDL on a fresh db: {ddl}"
