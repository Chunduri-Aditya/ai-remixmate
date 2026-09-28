"""
tests/test_hardening.py — B2.9 to B2.12 and B2.14 (family "hardening").

Spec (plans/audit-the-project-and-sorted-flask.md):

* B2.9  Job ``error`` and ``result.output`` are relativized against
        OUTPUTS_DIR, LIBRARY_DIR and ROOT before persisting; storage and
        library responses carry no absolute paths.
* B2.10 ``*`` in REMIXMATE_CORS_ORIGINS is dropped with a warning.
* B2.11 /beat/upload rejects Content-Length above 100 MB before reading and
        reads in chunks with a running cap; /setlist/import-csv is a def route
        with a 5 MB cap; feedback song_a, song_b, job_id capped at 200 chars and
        context at 8 KB serialized; /instrument-lab caps songs at 4 and
        target_duration at 300 s.
* B2.12 DMCA limiter keys on request.client.host unless REMIXMATE_TRUST_PROXY=1
        (then the last X-Forwarded-For entry) and prunes expired entries on
        every call.
* B2.14 POST /dj-remix/feedback accepts only a session_id whose directory
        exists under the caller's user_outputs_dir().

One deny and one allow test per item. The upload limit tests drive the ASGI
app directly (a hand written ``receive``) because httpx's TestClient joins a
streaming body into one bytes object before the app sees it, which would hide
whether the route read the body at all.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import scripts.core.paths as paths
from scripts.api import jobs as job_store
from scripts.api.auth import DEFAULT_USER, current_user
from scripts.api.cors import resolve_cors_origins
from scripts.api.routers import feedback as feedback_router
from scripts.api.routers import legal as legal_router
from scripts.api.routers import library as library_router
from scripts.api.routers import remix as remix_router
from scripts.api.routers import setlist as setlist_router
from scripts.api.schemas import JobType

pytestmark = pytest.mark.integration

USER_HEADER = "X-Test-User"
MB = 1024 * 1024

_WAV_BYTES = (
    b"RIFF" + (36).to_bytes(4, "little") + b"WAVE"
    + b"fmt " + (16).to_bytes(4, "little")
    + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
    + (22050).to_bytes(4, "little") + (44100).to_bytes(4, "little")
    + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
    + b"data" + (0).to_bytes(4, "little")
)


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_user():
    current_user.set(DEFAULT_USER)
    yield
    current_user.set(DEFAULT_USER)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Throwaway job store (same shape as tests/test_tenancy_jobs.py)."""
    db = tmp_path / "jobs.db"
    monkeypatch.setattr(job_store, "_DB_PATH", str(db))
    monkeypatch.setattr(job_store, "_jobs", {})
    monkeypatch.setattr(job_store, "_cancelled", set())
    monkeypatch.setattr(job_store, "_sse_hook", None)
    job_store._init_db()
    return db


class _InlineExecutor:
    def submit(self, fn: Callable[[], None]) -> None:
        fn()


@pytest.fixture
def inline_executor(monkeypatch):
    ex = _InlineExecutor()
    monkeypatch.setattr(job_store, "_executor", ex)
    return ex


def _app_with_user_header(*routers) -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def _set_user(request: Request, call_next):
        current_user.set(request.headers.get(USER_HEADER, DEFAULT_USER))
        return await call_next(request)

    for r in routers:
        app.include_router(r)
    return app


def _client(*routers) -> TestClient:
    return TestClient(_app_with_user_header(*routers), raise_server_exceptions=False)


def _multipart(field: str, filename: str, data: bytes, ctype: str = "audio/wav"):
    """Return (prefix, suffix, content_type) for a one file multipart body."""
    b = "b0undary7deadbeef"
    prefix = (
        f"--{b}\r\nContent-Disposition: form-data; name=\"{field}\"; "
        f"filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{b}--\r\n".encode()
    return prefix, suffix, f"multipart/form-data; boundary={b}"


def _asgi_post(app, path: str, headers: Dict[str, str], chunks: List[bytes]) -> Tuple[int, bytes, int]:
    """Drive the ASGI app with a chunked body; return (status, body, chunks consumed).

    ``chunks consumed`` counts how many body messages the app pulled from
    ``receive`` before it answered. A route that rejects on Content-Length
    before reading consumes 0; a route with a running cap stops before the end.
    """
    consumed = {"n": 0}
    it = iter(chunks)

    async def receive():
        try:
            chunk = next(it)
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}
        consumed["n"] += 1
        return {"type": "http.request", "body": chunk, "more_body": True}

    sent: List[dict] = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    async def _run():
        try:
            await app(scope, receive, send)
        except Exception:  # noqa: BLE001 — a 500 is a status, not a test crash
            pass

    asyncio.run(_run())
    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), 500)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body, consumed["n"]


def _string_values(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _string_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _string_values(v)


def _absolute_paths_in(body) -> List[str]:
    return [s for s in _string_values(body) if s.startswith("/") and Path(s).is_absolute()]


# ---------------------------------------------------------------------------
# B2.9 Path and error leaks
# ---------------------------------------------------------------------------

def _raising(msg: str):
    def task(job_id: str):
        raise RuntimeError(msg)
    return task


def _returning(result: dict):
    def task(job_id: str):
        return result
    return task


def test_job_error_mentioning_root_path_reaches_client_relativized(store, inline_executor):
    """deny: an exception mentioning $ROOT/library/x is persisted and served as library/x."""
    current_user.set("u1")
    leaked = str(paths.PROJECT_ROOT / "library" / "x")
    jid = job_store.create_job(JobType.DJ_REMIX, {"song": "x"})

    job_store.submit_job(jid, _raising(f"cannot open {leaked}"))

    job = job_store.get_job(jid)
    assert job["error"] == "cannot open library/x", job["error"]
    assert str(paths.PROJECT_ROOT) not in job["error"]
    served = job_store.job_to_response(job)
    assert served.error == "cannot open library/x"
    row = job_store.get_job_from_db(jid)
    assert row is not None
    assert row["error"] == "cannot open library/x", row["error"]


def test_job_result_output_under_outputs_dir_is_relativized(store, inline_executor):
    """deny: result.output written as an absolute path under OUTPUTS_DIR is persisted relative."""
    current_user.set("u1")
    absolute = str(paths.OUTPUTS_DIR / "u1" / "dj_s1" / "mix.wav")
    jid = job_store.create_job(JobType.DJ_REMIX, {"song": "x"})

    job_store.submit_job(jid, _returning({"output": absolute, "session_id": "s1"}))

    job = job_store.get_job(jid)
    assert job["result"]["output"] == "u1/dj_s1/mix.wav", job["result"]["output"]
    row = job_store.get_job_from_db(jid)
    assert row["result"]["output"] == "u1/dj_s1/mix.wav", row["result"]


def test_job_error_and_output_without_paths_are_untouched(store, inline_executor):
    """allow: plain error text and an already relative output survive unchanged."""
    current_user.set("u1")
    j_err = job_store.create_job(JobType.DJ_REMIX, {"song": "x"})
    job_store.submit_job(j_err, _raising("boom"))
    assert job_store.get_job(j_err)["error"] == "boom"
    assert job_store.get_job_from_db(j_err)["error"] == "boom"

    j_ok = job_store.create_job(JobType.DJ_REMIX, {"song": "x"})
    job_store.submit_job(j_ok, _returning({"output": "u1/dj_s2/mix.wav", "session_id": "s2"}))
    assert job_store.get_job(j_ok)["result"]["output"] == "u1/dj_s2/mix.wav"
    assert job_store.get_job_from_db(j_ok)["result"]["output"] == "u1/dj_s2/mix.wav"


def test_storage_responses_carry_no_absolute_paths():
    """deny: /library/storage and /library/mixes/storage never return an absolute path.

    allow: both still answer 200 with their counters.
    """
    client = _client(library_router.router)
    current_user.set("u1")
    (paths.OUTPUTS_DIR / "u1" / "dj_abc").mkdir(parents=True, exist_ok=True)

    r = client.get("/library/storage")
    assert r.status_code == 200, r.text
    assert _absolute_paths_in(r.json()) == [], r.json()

    r2 = client.get("/library/mixes/storage", headers={USER_HEADER: "u1"})
    assert r2.status_code == 200, r2.text
    assert _absolute_paths_in(r2.json()) == [], r2.json()
    assert r2.json()["mix_count"] >= 1


# ---------------------------------------------------------------------------
# B2.10 CORS
# ---------------------------------------------------------------------------

def test_cors_wildcard_in_env_is_dropped_with_warning(monkeypatch, caplog):
    """deny: '*' never reaches the allow list, and a warning says so."""
    monkeypatch.setenv("REMIXMATE_CORS_ORIGINS", "*, https://a.example")
    monkeypatch.delenv("REMIXMATE_FRONTEND_URL", raising=False)

    with caplog.at_level(logging.WARNING):
        origins = resolve_cors_origins()

    assert "*" not in origins, origins
    assert origins == ["https://a.example"]
    warned = [rec for rec in caplog.records if rec.levelno >= logging.WARNING and "*" in rec.getMessage()]
    assert warned, "no warning was logged for the dropped '*'"

    # A bare '*' is dropped too and never becomes an allow-all.
    monkeypatch.setenv("REMIXMATE_CORS_ORIGINS", "*")
    assert "*" not in resolve_cors_origins()


def test_cors_env_without_wildcard_is_kept_silently(monkeypatch, caplog):
    """allow: concrete origins pass through and nothing is logged at WARNING."""
    monkeypatch.setenv("REMIXMATE_CORS_ORIGINS", "https://a.example,https://b.example")
    monkeypatch.delenv("REMIXMATE_FRONTEND_URL", raising=False)

    with caplog.at_level(logging.WARNING):
        origins = resolve_cors_origins()

    assert origins == ["https://a.example", "https://b.example"]
    assert not [rec for rec in caplog.records if rec.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# B2.11 Limits — /beat/upload
# ---------------------------------------------------------------------------

def _upload_app() -> FastAPI:
    return _app_with_user_header(remix_router.router)


def test_beat_upload_rejects_large_content_length_before_reading():
    """deny: Content-Length above 100 MB is 413 and no body chunk is consumed."""
    prefix, suffix, ctype = _multipart("file", "beat.wav", _WAV_BYTES)
    body = prefix + _WAV_BYTES + suffix
    headers = {"content-type": ctype, "content-length": str(100 * MB + 1)}
    user_dir = paths.OUTPUTS_DIR / DEFAULT_USER
    before = {p.name for p in user_dir.glob("beat_*")}  # earlier tests may have uploaded legitimately

    status, _, consumed = _asgi_post(_upload_app(), "/beat/upload", headers, [body])

    assert status == 413, status
    assert consumed == 0, f"route read {consumed} body chunk(s) before rejecting"
    new_dirs = [p for p in user_dir.glob("beat_*") if p.name not in before]
    assert not new_dirs or all(not any(p.iterdir()) for p in new_dirs), "an oversized upload landed on disk"


def test_beat_upload_running_cap_stops_reading_a_chunked_body():
    """deny: a chunked body with no Content-Length is cut off at the cap, not read to the end."""
    prefix, suffix, ctype = _multipart("file", "beat.wav", _WAV_BYTES)
    payload = b"\0" * MB
    chunks = [prefix] + [payload] * 110 + [suffix]
    headers = {"content-type": ctype, "transfer-encoding": "chunked"}

    status, _, consumed = _asgi_post(_upload_app(), "/beat/upload", headers, chunks)

    assert status == 413, status
    assert consumed < len(chunks), (
        f"route consumed all {consumed} chunks ({consumed} MB) before rejecting: no running cap"
    )


def test_beat_upload_small_file_is_accepted():
    """allow: a small WAV with an honest Content-Length is stored and named relatively."""
    prefix, suffix, ctype = _multipart("file", "beat.wav", _WAV_BYTES)
    body = prefix + _WAV_BYTES + suffix
    headers = {"content-type": ctype, "content-length": str(len(body))}

    status, raw, _ = _asgi_post(_upload_app(), "/beat/upload", headers, [body])

    assert status == 200, raw
    data = json.loads(raw)
    assert data["audio_path"].startswith("beat_")
    assert (paths.OUTPUTS_DIR / DEFAULT_USER / data["audio_path"]).is_file()


# ---------------------------------------------------------------------------
# B2.11 Limits — /setlist/import-csv
# ---------------------------------------------------------------------------

_CSV_HEADER = (
    "Spotify ID,Track Name,Artist Name(s),Album Name,Duration (ms),Popularity,"
    "Danceability,Energy,Key,Loudness,Mode,Speechiness,Acousticness,"
    "Instrumentalness,Liveness,Valence,Tempo,Genres\n"
)
_CSV_ROWS = (
    "id1,Track One,Artist A,Album A,200000,50,0.7,0.6,5,-8.0,1,0.05,0.1,0.0,0.1,0.5,124.0,house\n"
    "id2,Track Two,Artist B,Album B,210000,55,0.8,0.7,0,-7.0,1,0.04,0.1,0.0,0.1,0.6,126.0,techno\n"
)


def test_import_csv_is_a_def_route():
    """The route runs in the threadpool: not a coroutine function."""
    assert not inspect.iscoroutinefunction(setlist_router.import_exportify_csv)


def test_import_csv_rejects_body_over_5mb():
    """deny: one row carrying a 5 MB field is 413, never parsed."""
    client = _client(setlist_router.router)
    big = (_CSV_HEADER + "id1," + ("x" * (5 * MB + 1)) + ",a,b\n").encode()

    r = client.post("/setlist/import-csv", files={"file": ("big.csv", big, "text/csv")})

    assert r.status_code == 413, (r.status_code, r.text[:200])


def test_import_csv_small_valid_file_is_optimized():
    """allow: a two track Exportify CSV is accepted and ordered."""
    client = _client(setlist_router.router)
    small = (_CSV_HEADER + _CSV_ROWS).encode()

    r = client.post("/setlist/import-csv", files={"file": ("p.csv", small, "text/csv")})

    assert r.status_code == 200, r.text
    assert r.json()["track_count"] == 2


# ---------------------------------------------------------------------------
# B2.11 Limits — POST /feedback/event
# ---------------------------------------------------------------------------

@pytest.fixture
def feedback_client(tmp_path, monkeypatch):
    from scripts.core import feedback_store as fs

    monkeypatch.setattr(fs, "_DB_PATH", tmp_path / "feedback.db")
    fs._events.clear()
    fs._init_db()
    with TestClient(_app_with_user_header(feedback_router.router), raise_server_exceptions=False) as c:
        yield c
    fs._events.clear()


@pytest.mark.parametrize(
    "field,value",
    [
        ("song_a", "a" * 201),
        ("song_b", "b" * 201),
        ("job_id", "j" * 201),
        ("context", {"blob": "x" * (8 * 1024 + 1)}),
    ],
    ids=["song_a_201", "song_b_201", "job_id_201", "context_over_8kb"],
)
def test_feedback_event_over_cap_is_rejected(feedback_client, field, value):
    """deny: each capped field one unit over its limit is 422 and logs nothing."""
    from scripts.core import feedback_store as fs

    body = {"event_type": "play_start", "song_a": "Song A", "song_b": "Song B"}
    body[field] = value

    r = feedback_client.post("/feedback/event", json=body)

    assert r.status_code == 422, (r.status_code, r.text[:300])
    assert fs._events == [] or all(e.get("song_a") != value for e in fs._events)


def test_feedback_event_at_cap_is_accepted(feedback_client):
    """allow: every capped field exactly at its limit is logged."""
    body = {
        "event_type": "play_start",
        "song_a": "a" * 200,
        "song_b": "b" * 200,
        "job_id": "j" * 200,
        "context": {"blob": "x" * 4000},
    }

    r = feedback_client.post("/feedback/event", json=body)

    assert r.status_code == 200, r.text
    assert r.json()["logged"] is True


# ---------------------------------------------------------------------------
# B2.11 Limits — /instrument-lab
# ---------------------------------------------------------------------------

@pytest.fixture
def lab(monkeypatch, tmp_path):
    """Fake the job store and song lookup so the router runs alone (tests/test_beat_paths.py)."""
    state = {"created": 0, "submitted": []}

    def _fake_create_job(job_type, meta):
        state["created"] += 1
        return "job-under-test"

    def _fake_submit_job(job_id, fn, **kwargs):
        state["submitted"].append(kwargs)

    monkeypatch.setattr(remix_router.job_store, "create_job", _fake_create_job)
    monkeypatch.setattr(remix_router.job_store, "submit_job", _fake_submit_job)
    monkeypatch.setattr(
        remix_router.job_store, "get_job",
        lambda jid: {"job_id": jid, "status": "pending", "job_type": JobType.DJ_REMIX,
                     "progress": 0.0, "message": "", "created_at": 0.0,
                     "updated_at": 0.0, "result": None, "error": None, "meta": {}},
    )
    monkeypatch.setattr(remix_router, "_check_job_cap", lambda: None)
    monkeypatch.setattr(remix_router, "_require_song", lambda name: tmp_path / name)
    return state


def test_instrument_lab_five_songs_is_rejected(lab):
    """deny: 5 songs is over the cap of 4; no job is created."""
    client = _client(remix_router.router)

    r = client.post("/instrument-lab", json={"songs": ["s1", "s2", "s3", "s4", "s5"]})

    assert r.status_code in (400, 422), (r.status_code, r.text[:300])
    assert lab["created"] == 0


def test_instrument_lab_duration_over_300s_is_rejected(lab):
    """deny: target_duration 301 is over the cap; no job is created."""
    client = _client(remix_router.router)

    r = client.post("/instrument-lab", json={"songs": ["s1", "s2"], "target_duration": 301})

    assert r.status_code in (400, 422), (r.status_code, r.text[:300])
    assert lab["created"] == 0


def test_instrument_lab_at_caps_is_queued(lab):
    """allow: 4 songs and 300 s sit at the caps and queue a job."""
    client = _client(remix_router.router)

    r = client.post(
        "/instrument-lab",
        json={"songs": ["s1", "s2", "s3", "s4"], "target_duration": 300},
    )

    assert r.status_code == 202, r.text
    assert lab["created"] == 1
    assert lab["submitted"][0]["target_duration"] == 300


# ---------------------------------------------------------------------------
# B2.12 DMCA limiter
# ---------------------------------------------------------------------------

_DMCA_BODY = {
    "reporter_name": "R",
    "reporter_email": "r@example.com",
    "copyright_work": "W",
    "infringing_url": "https://x.example/mix",
    "description": "",
}


@pytest.fixture
def dmca(monkeypatch):
    monkeypatch.setattr(legal_router.legal_store, "create_dmca_report", lambda **kw: "r-1")
    monkeypatch.setattr(legal_router, "_send_dmca_email", lambda report_id, body: False)
    monkeypatch.delenv("REMIXMATE_TRUST_PROXY", raising=False)
    legal_router._dmca_hits.clear()
    app = FastAPI()
    app.include_router(legal_router.router)
    yield TestClient(app, raise_server_exceptions=False)
    legal_router._dmca_hits.clear()


def test_dmca_sixth_request_in_window_is_429(dmca):
    """deny: the 6th report from one client host within the hour is 429."""
    for i in range(5):
        assert dmca.post("/legal/dmca", json=_DMCA_BODY).status_code == 200, i
    assert dmca.post("/legal/dmca", json=_DMCA_BODY).status_code == 429


def test_dmca_spoofed_forwarded_for_does_not_bypass_without_trust_proxy(dmca):
    """deny: with REMIXMATE_TRUST_PROXY unset, X-Forwarded-For is ignored and the
    6th request from the same client host is 429 no matter what the header says."""
    for i in range(5):
        r = dmca.post("/legal/dmca", json=_DMCA_BODY, headers={"X-Forwarded-For": f"10.0.0.{i}"})
        assert r.status_code == 200, (i, r.text)

    r = dmca.post("/legal/dmca", json=_DMCA_BODY, headers={"X-Forwarded-For": "10.0.0.99"})

    assert r.status_code == 429, "a rotated X-Forwarded-For bypassed the limiter"


def test_dmca_trust_proxy_keys_on_last_forwarded_entry(dmca, monkeypatch):
    """allow: with the flag on, the last X-Forwarded-For entry (the one Caddy
    appends) is the key: a different last entry is a different client, and a
    changed first entry is not."""
    monkeypatch.setenv("REMIXMATE_TRUST_PROXY", "1")
    for i in range(5):
        r = dmca.post(
            "/legal/dmca", json=_DMCA_BODY,
            headers={"X-Forwarded-For": f"spoof-{i}, 203.0.113.1"},
        )
        assert r.status_code == 200, (i, r.text)

    other = dmca.post(
        "/legal/dmca", json=_DMCA_BODY, headers={"X-Forwarded-For": "spoof-9, 203.0.113.2"}
    )
    assert other.status_code == 200, "a different real client was rate limited"

    same = dmca.post(
        "/legal/dmca", json=_DMCA_BODY, headers={"X-Forwarded-For": "spoof-9, 203.0.113.1"}
    )
    assert same.status_code == 429, "changing the first entry bypassed the limiter"


def test_dmca_expired_entries_are_pruned_on_every_call(dmca):
    """deny (unbounded growth): a stale key from another client is dropped by any call."""
    stale = time.time() - 2 * legal_router._DMCA_WINDOW_SEC
    legal_router._dmca_hits["198.51.100.7"] = [stale]

    assert dmca.post("/legal/dmca", json=_DMCA_BODY).status_code == 200

    assert not legal_router._dmca_hits.get("198.51.100.7"), dict(legal_router._dmca_hits)


# ---------------------------------------------------------------------------
# B2.14 Spectral feedback ownership
# ---------------------------------------------------------------------------

@pytest.fixture
def spectral(tmp_path, monkeypatch):
    import scripts.core.spectral_matching as sm

    monkeypatch.delenv("REMIXMATE_SPECTRAL_LEARNING", raising=False)
    monkeypatch.setattr(sm, "model_path", lambda: tmp_path / "model.json")
    monkeypatch.setattr(sm, "history_path", lambda: tmp_path / "history.json")
    sm.record_render("s1", {"sims": [0.9] * sm.N_BANDS, "entry_bar": 8, "score": 0.9})
    # The real render layout: outputs/<uid>/dj_<session_id>/
    (paths.OUTPUTS_DIR / "u1" / "dj_s1").mkdir(parents=True, exist_ok=True)
    return sm


def test_dj_remix_feedback_for_another_users_session_is_404(spectral):
    """deny: u2 cannot vote on u1's session, and the record is not consumed."""
    client = _client(remix_router.router)

    r = client.post(
        "/dj-remix/feedback", json={"session_id": "s1", "liked": True},
        headers={USER_HEADER: "u2"},
    )
    assert r.status_code == 404, (r.status_code, r.text)

    trav = client.post(
        "/dj-remix/feedback", json={"session_id": "../u1/dj_s1", "liked": True},
        headers={USER_HEADER: "u2"},
    )
    assert trav.status_code == 404, (trav.status_code, trav.text)

    history = json.loads(spectral.history_path().read_text())
    assert "s1" in history, "the denied call consumed u1's one shot record"


def test_dj_remix_feedback_for_own_session_is_accepted(spectral):
    """allow: the owner's vote updates the model once."""
    client = _client(remix_router.router)

    r = client.post(
        "/dj-remix/feedback", json={"session_id": "s1", "liked": True},
        headers={USER_HEADER: "u1"},
    )

    assert r.status_code == 200, r.text
    assert r.json()["updated"] is True
    assert r.json()["n_updates"] == 1
