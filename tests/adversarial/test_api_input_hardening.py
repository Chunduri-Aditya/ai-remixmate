"""
Adversarial QA — API input-hardening reproductions (2026-07-12).

Each test documents a real defect found by hostile-input fuzzing of the running
API. Tests that assert the *desired* (fixed) behavior are marked xfail so the
suite stays green until the bug is addressed — when a fix lands, the xfail flips
to xpass and the marker should be removed. Tests that merely characterize
current buggy behavior are left as plain passing assertions with a docstring.

Run:  pytest tests/adversarial/test_api_input_hardening.py -v
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scripts.api.main import app

# raise_server_exceptions=False so unhandled exceptions surface as 500s
# (production behavior) instead of propagating into the test.
client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# BUG 1 — Untyped `body: dict` handlers 500 on non-string field types.
# crates.py create/rename/add_to_crate/add_song_tag all do
# `(body.get(X) or "").strip()`; a list/int/dict value raises
# AttributeError -> HTTP 500 where a 422/400 is correct.
# ---------------------------------------------------------------------------

def test_create_crate_rejects_non_string_name_without_500():
    r = client.post("/crates", json={"name": ["not", "a", "string"]})
    assert r.status_code != 500, r.text
    assert r.status_code in (400, 422)


def test_rename_crate_rejects_non_string_name_without_500():
    # Crate id need not exist — the crash happens before any lookup.
    r = client.patch("/crates/does-not-matter", json={"name": [1, 2]})
    assert r.status_code != 500, r.text


def test_add_to_crate_rejects_non_string_song_without_500():
    r = client.post("/crates/does-not-matter/songs", json={"song_name": 123})
    assert r.status_code != 500, r.text


def test_add_song_tag_rejects_non_string_tag_without_500():
    r = client.post("/library/anything/tags", json={"tag": {"a": 1}})
    assert r.status_code != 500, r.text


# ---------------------------------------------------------------------------
# BUG 2 — Tag/favorite writes never validate the song name or its existence.
# add_song_tag / add_favorite use `{song_name:path}` (accepts slashes) and skip
# `_require_song`, so arbitrary/traversal-shaped names get persisted, polluting
# the crate/favorites DB with phantom rows for songs that don't exist.
# ---------------------------------------------------------------------------

def test_tag_write_rejects_nonexistent_song():
    name = "QA_PHANTOM_SONG_THAT_DOES_NOT_EXIST_0xdeadbeef"
    r = client.post(f"/library/{name}/tags", json={"tag": "qa-should-not-persist"})
    try:
        # A write against a song that isn't in the library should 404, not 201.
        assert r.status_code == 404, r.text
    finally:
        client.delete(f"/library/{name}/tags/qa-should-not-persist")


def test_favorite_write_rejects_nonexistent_song():
    name = "QA_PHANTOM_FAV_0xdeadbeef"
    r = client.post(f"/favorites/{name}")
    try:
        assert r.status_code == 404, r.text
    finally:
        client.delete(f"/favorites/{name}")


# ---------------------------------------------------------------------------
# BUG 3 — /compatibility fabricates a confident score for songs not in the
# library. Unknown names fall through to the (keyless) external metadata API,
# which returns all-default TrackMetadata; comparing two all-default objects
# yields overall≈0.575 with energy_score==1.0 (0.5 default vs 0.5 default) and
# `compatible: true`. It also skips _require_song entirely (accepts traversal
# names). A nonexistent pair should 404 or clearly signal "unknown", not a score.
# ---------------------------------------------------------------------------

def test_compatibility_does_not_fabricate_score_for_unknown_songs():
    r = client.post("/compatibility", json={
        "song_a": "Nonexistent Song ZZZ", "song_b": "Also Fake QQQ",
        "artist_a": "", "artist_b": "",
    })
    body = r.json() if r.status_code == 200 else {}
    # The tell-tale signature of the empty-object comparison:
    fabricated = (
        r.status_code == 200
        and body.get("compatible") is True
        and body.get("energy_score") == 1.0
        and body.get("bpm_a") == 0.0
        and body.get("camelot_a") is None
    )
    assert not fabricated, f"fabricated confident score: {body}"


@pytest.mark.parametrize("hostile", [
    "../../../../../../etc/passwd",
    "..",
    "foo/bar",
    "foo%00bar",
    "a" * 10000,
])
def test_compatibility_rejects_hostile_song_names(hostile):
    """/compatibility shape-validates both names against the shared allowlist.

    It deliberately does NOT require the song to exist (song_a is documented
    as "Song name or search query" and falls through to an external metadata
    lookup), so this asserts the name *shape* gate only."""
    r = client.post("/compatibility", json={
        "song_a": hostile, "song_b": hostile,
        "artist_a": "x", "artist_b": "y",
    })
    assert r.status_code == 400, r.text


# ---------------------------------------------------------------------------
# CONTROL — the properly-hardened endpoints must keep rejecting hostile names.
# These should always PASS; a regression here is a serious traversal exposure.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", ["foo%00bar", "a" * 10000])
def test_require_song_endpoints_reject_hostile_names(payload):
    r = client.get(f"/library/{payload}")
    assert r.status_code in (400, 404), r.text


def test_stem_param_is_allowlisted():
    # A non-allowlisted stem name is rejected — 400 from the handler's
    # allowlist, or 404 when the router normalizes an encoded traversal away
    # before dispatch. Either way it never serves a file outside the stem set.
    r = client.get("/library/whatever/stems/..%2f..%2fetc%2fpasswd")
    assert r.status_code in (400, 404), r.text
    r2 = client.get("/library/whatever/stems/evilstem")
    assert r2.status_code in (400, 404), r2.text
