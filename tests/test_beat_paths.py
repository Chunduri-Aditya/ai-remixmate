"""
tests/test_beat_paths.py — B2.4 beat path confinement (family "beat-paths").

Spec (plans/audit-the-project-and-sorted-flask.md, B2.4):

* ``bridge_beat_path`` is validated in the remix router *before* a job exists:
  a bare file name, or a path that resolves inside ``user_outputs_dir()``,
  ending in ``.wav`` or ``.flac``; anything else is 400 and no job is created.
* ``/beat/upload`` returns the file name relative to the user dir (never an
  absolute path).
* ``/beat/synthesize`` accepts ``genre`` matching ``^[a-z0-9_-]{1,32}$`` only.

Deny rows: absolute path outside the user dir, ``..``, http URL, wrong
extension, genre with a slash. Allow rows: own upload by relative name,
absolute path inside the user dir, plain genre.

The task-level bullet ("the task resolves the name under the user dir and
never opens any other path") is NOT covered here: ``task_dj_remix`` needs two
analysed library songs before it reaches the bridge branch, so it has no cheap
black box seam. Reported as a gap, not asserted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import scripts.core.paths as paths
from scripts.api.auth import DEFAULT_USER, current_user
from scripts.api.routers import remix as remix_router
from scripts.api.schemas import JobType

pytestmark = pytest.mark.integration

# A 44 byte RIFF/WAVE header: enough for the upload route (size + extension
# checks only) and for a ``.is_file()`` check afterwards.
_WAV_BYTES = (
    b"RIFF" + (36).to_bytes(4, "little") + b"WAVE"
    + b"fmt " + (16).to_bytes(4, "little")
    + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
    + (22050).to_bytes(4, "little") + (44100).to_bytes(4, "little")
    + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
    + b"data" + (0).to_bytes(4, "little")
)


@pytest.fixture(autouse=True)
def _isolate_user():
    current_user.set(DEFAULT_USER)
    yield
    current_user.set(DEFAULT_USER)


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Fake the job store and the song lookup so the router runs alone.

    ``created`` counts create_job calls: a 400 must leave it at 0 ("validated
    before a job exists"). ``submitted`` records the kwargs handed to the task.
    """
    state = {"created": 0, "submitted": []}

    def _fake_create_job(job_type, meta):
        state["created"] += 1
        return "job-under-test"

    def _fake_submit_job(job_id, fn, **kwargs):
        state["submitted"].append(kwargs)

    monkeypatch.setattr(remix_router.job_store, "create_job", _fake_create_job)
    monkeypatch.setattr(remix_router.job_store, "submit_job", _fake_submit_job)
    # get_job returns a complete raw job dict; the real job_to_response builds
    # the JobResponse from it, so the 202 body passes response_model validation
    # (a partial fake here turned every handler exit into a 500).
    monkeypatch.setattr(
        remix_router.job_store, "get_job",
        lambda jid: {"job_id": jid, "status": "pending", "job_type": JobType.DJ_REMIX,
                     "progress": 0.0, "message": "", "created_at": 0.0,
                     "updated_at": 0.0, "result": None, "error": None, "meta": {}},
    )
    monkeypatch.setattr(remix_router, "_check_job_cap", lambda: None)
    monkeypatch.setattr(remix_router, "_require_song", lambda name: tmp_path / name)
    return state


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(remix_router.router)
    # A 500 must show up as a status code, not as an exception in the test body,
    # so a wrong status fails on the assertion the spec names.
    return TestClient(app, raise_server_exceptions=False)


def _remix_body(beat_path: str) -> dict:
    return {
        "song_a": "song_a",
        "song_b": "song_b",
        "bridge_beat_mode": "file",
        "bridge_beat_path": beat_path,
    }


def _user_dir() -> Path:
    d = paths.user_outputs_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# deny: bridge_beat_path outside the user dir is 400 and creates no job
# ---------------------------------------------------------------------------

def test_absolute_path_outside_user_dir_is_400(client, captured, tmp_path):
    outside = tmp_path / "evil.wav"
    outside.write_bytes(_WAV_BYTES)
    assert outside.is_file()
    assert not str(outside.resolve()).startswith(str(paths.OUTPUTS_DIR.resolve()))

    r = client.post("/dj-remix", json=_remix_body(str(outside)))

    assert r.status_code == 400, r.text
    assert captured["created"] == 0, "a job was created before the path was validated"
    assert captured["submitted"] == []


def test_dotdot_path_is_400(client, captured):
    # Would land in another user's output root if joined naively.
    other = paths.OUTPUTS_DIR / "u2" / "beat_deadbeef"
    other.mkdir(parents=True, exist_ok=True)
    (other / "a.wav").write_bytes(_WAV_BYTES)

    r = client.post("/dj-remix", json=_remix_body("../u2/beat_deadbeef/a.wav"))

    assert r.status_code == 400, r.text
    assert captured["created"] == 0


def test_http_url_is_400(client, captured):
    r = client.post("/dj-remix", json=_remix_body("http://evil.example/a.wav"))

    assert r.status_code == 400, r.text
    assert captured["created"] == 0


def test_wrong_extension_inside_user_dir_is_400(client, captured):
    d = _user_dir() / "beat_cafef00d"
    d.mkdir(parents=True, exist_ok=True)
    (d / "beat.mp3").write_bytes(_WAV_BYTES)

    r = client.post("/dj-remix", json=_remix_body("beat_cafef00d/beat.mp3"))

    assert r.status_code == 400, r.text
    assert captured["created"] == 0


def test_dj_chain_applies_the_same_check(client, captured, tmp_path):
    outside = tmp_path / "evil.wav"
    outside.write_bytes(_WAV_BYTES)

    r = client.post("/dj-chain", json={
        "songs": ["song_a", "song_b"],
        "bridge_beat_mode": "file",
        "bridge_beat_path": str(outside),
    })

    assert r.status_code == 400, r.text
    assert captured["created"] == 0


# ---------------------------------------------------------------------------
# allow: own upload by its relative name is 202
# ---------------------------------------------------------------------------

def test_upload_returns_relative_name_and_own_upload_is_202(client, captured):
    up = client.post(
        "/beat/upload",
        files={"file": ("my_beat.wav", _WAV_BYTES, "audio/wav")},
    )
    assert up.status_code == 200, up.text
    body = up.json()

    # The contract: the returned path is relative to the user dir, never absolute.
    rel = body.get("audio_path") or body.get("path") or body.get("filename")
    assert rel, body
    assert not Path(rel).is_absolute(), f"upload leaked an absolute path: {rel!r}"
    assert not rel.startswith(".."), rel
    target = (_user_dir() / rel).resolve()
    assert target.is_file(), f"{rel!r} does not resolve to the stored file under the user dir"
    assert str(target).startswith(str(_user_dir().resolve()))
    for v in body.values():
        if isinstance(v, str):
            assert not Path(v).is_absolute() or not v.startswith(str(paths.OUTPUTS_DIR)), (
                f"upload response carries an absolute outputs path: {v!r}"
            )

    r = client.post("/dj-remix", json=_remix_body(rel))
    assert r.status_code == 202, r.text
    assert captured["created"] == 1
    assert len(captured["submitted"]) == 1

    # Whatever the router hands the task must resolve inside the user dir.
    handed = captured["submitted"][0]["bridge_beat_path"]
    assert handed
    resolved = (_user_dir() / handed).resolve()
    assert resolved == target


def test_absolute_path_inside_user_dir_is_202(client, captured):
    d = _user_dir() / "beat_0badf00d"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "loop.flac"
    f.write_bytes(_WAV_BYTES)

    r = client.post("/dj-remix", json=_remix_body(str(f)))

    assert r.status_code == 202, r.text
    assert captured["created"] == 1


# ---------------------------------------------------------------------------
# /beat/synthesize genre allowlist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("genre", ["../x", "a/b", "TECHNO", "t" * 33, "te chno", ""])
def test_genre_outside_allowlist_is_400(client, genre):
    r = client.get("/beat/synthesize", params={"genre": genre, "bars": 1})

    assert r.status_code == 400, f"{genre!r} -> {r.status_code}: {r.text[:200]}"
    # Nothing was written under the user dir for a rejected genre.
    assert not list(_user_dir().glob("beat_*/bridge_..*"))


def test_plain_genre_is_accepted(client):
    r = client.get("/beat/synthesize", params={"genre": "techno", "bars": 1})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["genre"] == "techno"
    assert body["audio_url"].startswith("/outputs/beat_")
