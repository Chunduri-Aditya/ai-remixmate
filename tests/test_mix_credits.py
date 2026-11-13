"""
tests/test_mix_credits.py — CREDITS.txt + manifest.json next to a rendered mix
(deployment plan P1).

A full DJ render is outside the safe subset, so this drives
`scripts/api/task_modules/remix._write_mix_credits` directly and structurally
checks that both full-render tasks wire it in.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from scripts.api.task_modules import remix
from scripts.core.license import LicenseInfo, LicenseType, save_license

_BY = "https://creativecommons.org/licenses/by/4.0/"


@pytest.fixture
def library(tmp_path, monkeypatch):
    """Temp library root; `remix.song_dir(name)` -> `<root>/<name>`.

    The Phase 0 conftest deliberately leaves `LIBRARY_DIR` pointed at the real
    `library/` tree, so a test that reads licences must redirect the lookup
    rather than write into it.
    """
    root = tmp_path / "library"
    root.mkdir()
    monkeypatch.setattr(remix, "song_dir", lambda name: root / name)
    return root


def _make_song(root: Path, name: str, info: LicenseInfo | None) -> None:
    (root / name).mkdir(parents=True)
    if info is not None:
        save_license(root / name, info)


def test_credits_written_one_licensed_one_missing(library, tmp_path):
    _make_song(
        library,
        "Kellee Maize - Wanna Be",
        LicenseInfo(
            license_type=LicenseType.CC_BY,
            source="ccmixter",
            artist_name="Kellee Maize",
            license_url=_BY,
            source_url="https://ccmixter.org/files/kellee/123",
        ),
    )
    _make_song(library, "Unknown Local Track", None)  # no license.json

    out_dir = tmp_path / "dj_abc12345"
    out_dir.mkdir()
    songs = ["Kellee Maize - Wanna Be", "Unknown Local Track"]
    credits = remix._write_mix_credits(out_dir, songs)

    # CREDITS.txt — human-readable, both tracks present
    txt = (out_dir / "CREDITS.txt").read_text()
    assert "Kellee Maize" in txt
    assert _BY in txt
    assert "Unknown Local Track" in txt        # missing-licence track is NOT dropped
    assert "unknown" in txt.lower()
    assert txt.rstrip("\n").splitlines()[-1] == (
        "Rendered with AI RemixMate by Aditya Chunduri, "
        "https://github.com/Chunduri-Aditya/ai-remixmate (MIT)"
    )

    # manifest.json — machine-readable credits[]
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["songs"] == songs
    assert manifest["rendered_with"] == {
        "name": "AI RemixMate",
        "author": "Aditya Chunduri",
        "repository": "https://github.com/Chunduri-Aditya/ai-remixmate",
        "license": "MIT",
    }
    assert [c["song"] for c in manifest["credits"]] == songs

    licensed, missing = manifest["credits"]
    assert licensed["artist"] == "Kellee Maize"
    assert licensed["license"] == "cc_by"
    assert licensed["license_url"] == _BY
    assert licensed["source"] == "ccmixter"
    assert licensed["attribution_required"] is True

    assert missing["artist"] is None
    assert missing["license"] == "unknown"
    assert missing["license_url"] is None
    assert missing["attribution_required"] is True  # conservative when unknown

    # the returned list is exactly what was written
    assert credits == manifest["credits"]


def test_missing_licence_track_is_never_dropped(library, tmp_path):
    """The failure mode: a mix that silently ships without an attribution line
    for a track whose license.json did not load."""
    _make_song(library, "A", None)
    _make_song(library, "B", None)
    out_dir = tmp_path / "dj_zzz00000"
    out_dir.mkdir()

    credits = remix._write_mix_credits(out_dir, ["A", "B"])
    assert [c["song"] for c in credits] == ["A", "B"]
    assert all(c["attribution_required"] for c in credits)


def test_write_failure_does_not_raise(library, tmp_path):
    """A credits-write failure must not fail an otherwise-good render."""
    _make_song(
        library,
        "A",
        LicenseInfo(license_type=LicenseType.CC_BY, source="ccmixter",
                    artist_name="X", license_url=_BY),
    )
    missing_dir = tmp_path / "never_created"  # not mkdir'd -> write_text raises

    credits = remix._write_mix_credits(missing_dir, ["A"])

    assert credits and credits[0]["artist"] == "X"       # list still returned
    assert not (missing_dir / "CREDITS.txt").exists()    # the write was swallowed


def test_both_full_render_tasks_wire_credits():
    """`task_dj_remix` and `task_dj_chain` must call `_write_mix_credits` and put
    the result under a `credits` key in their response. Guards a future refactor
    from dropping it. `task_remix_preview` is intentionally excluded — it renders
    a throwaway audition, not a distributable mix."""
    tree = ast.parse(inspect.getsource(remix))
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    for fname in ("task_dj_remix", "task_dj_chain"):
        fn = funcs[fname]
        calls = {
            n.func.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "_write_mix_credits" in calls, f"{fname} no longer calls _write_mix_credits"
        keys = {
            k.value
            for n in ast.walk(fn)
            if isinstance(n, ast.Dict)
            for k in n.keys
            if isinstance(k, ast.Constant)
        }
        assert "credits" in keys, f"{fname} response has no 'credits' key"

    assert "task_remix_preview" in funcs
    preview_calls = {
        n.func.id
        for n in ast.walk(funcs["task_remix_preview"])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_write_mix_credits" not in preview_calls
