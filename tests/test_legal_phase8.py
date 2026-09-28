"""
tests/test_legal_phase8.py — Phase 8 legal / account / admin surface.
"""

from __future__ import annotations

import io
import zipfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("jwt")
pytest.importorskip("cryptography")

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from scripts.api import auth as auth_mod
from scripts.api.auth import DEFAULT_USER, auth_context_middleware, require_user
from scripts.api.routers import account, admin, legal
from scripts.core import crates, legal_store


@pytest.fixture(autouse=True)
def _isolate_legal_db(tmp_path, monkeypatch):
    from scripts.core import db as db_mod

    db = tmp_path / "legal.db"
    monkeypatch.setattr(legal_store, "_DB_PATH", db)
    db_mod._engine_for.cache_clear()
    auth_mod.current_user.set(DEFAULT_USER)
    auth_mod._jwks_client = None
    auth_mod._jwks_client_url = None
    legal._dmca_hits.clear()
    yield
    auth_mod.current_user.set(DEFAULT_USER)


@pytest.fixture
def crates_db(tmp_path, monkeypatch):
    from scripts.core import db as db_mod

    monkeypatch.setattr(crates, "_DB_PATH", tmp_path / "remixmate.db")
    db_mod._engine_for.cache_clear()
    crates._init()
    return tmp_path


def _app_public():
    app = FastAPI()
    app.middleware("http")(auth_context_middleware)
    app.include_router(legal.router)
    return app


def _app_gated():
    app = FastAPI()
    app.middleware("http")(auth_context_middleware)
    app.include_router(legal.router)
    app.include_router(account.router, dependencies=[Depends(require_user)])
    app.include_router(admin.router, dependencies=[Depends(require_user)])
    return app


def test_legal_pages_return_200():
    client = TestClient(_app_public())
    for page in ("terms", "privacy", "dmca", "acceptable-use"):
        res = client.get(f"/legal/{page}")
        assert res.status_code == 200, page
        assert "RemixMate" in res.text or "remix" in res.text.lower()


def test_dmca_stores_row_without_smtp(monkeypatch):
    monkeypatch.delenv("REMIXMATE_SMTP_HOST", raising=False)
    client = TestClient(_app_public())
    res = client.post(
        "/legal/dmca",
        json={
            "reporter_name": "Ada",
            "reporter_email": "ada@example.com",
            "copyright_work": "Song X",
            "infringing_url": "https://example.com/mix/1",
            "description": "please remove",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["email_sent"] is False
    row = legal_store.get_dmca_report(body["report_id"])
    assert row is not None
    assert row["reporter_email"] == "ada@example.com"
    assert row["copyright_work"] == "Song X"


def test_account_delete_clears_rows_and_files(tmp_path, monkeypatch, crates_db):
    from scripts.api import jobs as job_store
    from scripts.core import db as db_mod
    from scripts.core import feedback_store
    from scripts.core import paths as paths_mod

    monkeypatch.setattr(job_store, "_DB_PATH", tmp_path / "jobs.db")
    db_mod._engine_for.cache_clear()
    monkeypatch.setattr(feedback_store, "_DB_PATH", tmp_path / "feedback.db")
    job_store._jobs.clear()
    feedback_store._events.clear()
    job_store._init_db()
    feedback_store._init_db()

    out_root = tmp_path / "outputs"
    monkeypatch.setattr(paths_mod, "OUTPUTS_DIR", out_root)
    user_dir = out_root / "local"
    user_dir.mkdir(parents=True)
    (user_dir / "mix1").mkdir()
    (user_dir / "mix1" / "manifest.json").write_text('{"credits":[]}')

    auth_mod.set_user_id("local")
    crates.create_crate("Bangers")
    crates.set_favorite("Song A", True)
    jid = job_store.create_job("dj_remix", {"song_a": "A", "song_b": "B"})
    assert job_store.get_job(jid) is not None

    client = TestClient(_app_gated())
    res = client.delete("/account")
    assert res.status_code == 200
    assert res.json()["clean"] is True
    assert not user_dir.exists() or not any(user_dir.iterdir())
    assert crates.list_crates() == []
    assert crates.list_favorites() == []
    assert job_store.list_jobs(user_id="local") == []


def test_account_export_zip(tmp_path, monkeypatch, crates_db):
    from scripts.core import paths as paths_mod

    out_root = tmp_path / "outputs"
    monkeypatch.setattr(paths_mod, "OUTPUTS_DIR", out_root)
    (out_root / "local" / "s1").mkdir(parents=True)
    (out_root / "local" / "s1" / "manifest.json").write_text('{"ok":true}')

    auth_mod.set_user_id("local")
    crates.create_crate("ExportMe")

    client = TestClient(_app_gated())
    res = client.get("/account/export")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/zip")
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        names = set(zf.namelist())
        assert "library.json" in names
        assert "manifests.json" in names


def _make_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(priv, *, sub="user-1", email="user@example.com", **extra):
    import time

    now = int(time.time())
    payload = {
        "sub": sub,
        "aud": "authenticated",
        "role": "authenticated",
        "email": email,
        "iat": now,
        "exp": now + 3600,
        **extra,
    }
    return jwt.encode(payload, priv, algorithm="RS256")


@pytest.fixture
def signing_key(monkeypatch):
    priv = _make_key()
    pub = priv.public_key()
    monkeypatch.setenv("REMIXMATE_AUTH_ENABLED", "1")
    monkeypatch.setenv(
        "SUPABASE_JWKS_URL", "https://project.test/auth/v1/.well-known/jwks.json"
    )
    monkeypatch.setenv("ADMIN_EMAILS", "admin@example.com")

    class _Signing:
        key = pub

    class _Client:
        def get_signing_key_from_jwt(self, token):  # noqa: ARG002
            return _Signing()

    monkeypatch.setattr(auth_mod, "_get_jwks_client", lambda: _Client())
    return priv


def test_flagged_user_gets_403(signing_key):
    legal_store.flag_user("user-1", reason="abuse")
    client = TestClient(_app_gated())
    tok = _token(signing_key, sub="user-1", email="user@example.com")
    res = client.get("/account/export", headers={"Authorization": f"Bearer {tok}"})
    assert res.status_code == 403


def test_admin_flag_rejects_non_admin(signing_key):
    client = TestClient(_app_gated())
    tok = _token(signing_key, sub="user-1", email="user@example.com")
    res = client.post(
        "/admin/user/victim/flag",
        headers={"Authorization": f"Bearer {tok}"},
        json={"reason": "spam"},
    )
    assert res.status_code == 403


def test_admin_flag_and_takedown(signing_key, tmp_path, monkeypatch):
    from scripts.api import jobs as job_store
    from scripts.core import paths as paths_mod

    monkeypatch.setattr(job_store, "_DB_PATH", tmp_path / "jobs.db")
    from scripts.core import db as db_mod

    db_mod._engine_for.cache_clear()
    job_store._jobs.clear()
    job_store._init_db()
    out_root = tmp_path / "outputs"
    monkeypatch.setattr(paths_mod, "OUTPUTS_DIR", out_root)

    auth_mod.set_user_id("victim")
    jid = job_store.create_job(
        "dj_remix", {"user_id": "victim", "session_id": "sess1"}
    )
    # the real layout task_dj_remix writes: <OUTPUTS_DIR>/<owner>/dj_<sid>/<file>
    mix_dir = out_root / "victim" / "dj_sess1"
    mix_dir.mkdir(parents=True)
    (mix_dir / "mix.wav").write_bytes(b"x")
    job_store.update_job(
        jid,
        result={
            "session_id": "sess1",
            "output": str(mix_dir / "mix.wav"),
            "stream_url": "/outputs/dj_sess1/mix.wav",
        },
    )

    client = TestClient(_app_gated())
    admin_tok = _token(
        signing_key, sub="admin-1", email="admin@example.com", email_verified=True
    )
    res = client.post(
        "/admin/user/victim/flag",
        headers={"Authorization": f"Bearer {admin_tok}"},
        json={"reason": "takedown"},
    )
    assert res.status_code == 200
    assert legal_store.is_flagged("victim")

    res2 = client.delete(
        f"/admin/mix/{jid}",
        headers={"Authorization": f"Bearer {admin_tok}"},
    )
    assert res2.status_code == 200
    assert not mix_dir.exists()
    updated = job_store.get_job(jid)
    assert updated is not None
    assert updated.get("error") == "admin_takedown"
