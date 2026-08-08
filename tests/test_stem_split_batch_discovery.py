"""
tests/test_stem_split_batch_discovery.py

Regression test for POST /stems/split-batch's default song-discovery list
(songs=null case). It used to filter on `(d / "full.wav").exists()` only,
which silently excludes any song that needs stems but has already had
full.wav pruned/evicted while a usable full_enhanced.wav remains — exactly
the shape of the 185-song backlog found in the live library (see
scripts/core/stems.py's resolve_source_file() fallback, added alongside
this fix). Without this, those songs could never be reached by a plain
"process everything" batch call even after separate_song_stems() itself
learned to resume from full_enhanced.wav.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

SR = 44100


@pytest.fixture
def fake_library(tmp_path, monkeypatch):
    import scripts.api.routers._helpers as helpers_mod
    import scripts.api.routers.stems as stems_router
    import scripts.core.paths as paths

    lib = tmp_path / "library"
    lib.mkdir()

    audio = 0.1 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)

    # Song 1: normal, still has full.wav.
    d1 = lib / "Has Full Wav"
    d1.mkdir()
    sf.write(str(d1 / "full.wav"), audio, SR)

    # Song 2: the stuck-backlog shape — full.wav pruned/evicted, only
    # full_enhanced.wav survives, no stems.
    d2 = lib / "Only Enhanced Left"
    d2.mkdir()
    sf.write(str(d2 / "full_enhanced.wav"), audio, SR)

    # Song 3: a bare download shell — no audio at all (metadata-only).
    d3 = lib / "No Audio At All"
    d3.mkdir()
    (d3 / "license.json").write_text("{}")

    # Song 4: already fully split — 4 stems, no single-file source at all
    # (both full.wav and full_enhanced.wav pruned away post-split).
    # has_source_audio() returns True here (it counts stems), but
    # separate_song_stems() has nothing it can actually read — the router
    # must use resolve_source_file(), not has_source_audio(), for discovery.
    d4 = lib / "Stems Only No Single File"
    d4.mkdir()
    for stem in ("vocals", "drums", "bass", "other"):
        sf.write(str(d4 / f"{stem}.wav"), audio, SR)

    monkeypatch.setattr(paths, "LIBRARY_DIR", lib)
    monkeypatch.setattr(stems_router, "LIBRARY_DIR", lib)
    # _require_song() (used to validate an explicit req.songs list) binds
    # its own independent copy of LIBRARY_DIR at import time in _helpers.py
    # — patching only paths.LIBRARY_DIR / stems_router.LIBRARY_DIR leaves it
    # pointed at the real library, so every explicit-song-list test would
    # 404 against production data instead of this fixture.
    monkeypatch.setattr(helpers_mod, "LIBRARY_DIR", lib)

    return lib


def test_default_discovery_includes_full_enhanced_only_songs(fake_library, monkeypatch):
    from scripts.api.routers import stems as stems_router
    from scripts.api.schemas import BatchStemSplitRequest

    captured = {}

    def _fake_create_job(job_type, meta):
        captured["meta"] = meta
        return "fake-job-id"

    def _fake_submit_job(job_id, fn, **kwargs):
        captured["songs"] = kwargs["songs"]

    def _fake_job_to_response(job):
        return job

    monkeypatch.setattr(stems_router.job_store, "create_job", _fake_create_job)
    monkeypatch.setattr(stems_router.job_store, "submit_job", _fake_submit_job)
    monkeypatch.setattr(stems_router.job_store, "get_job", lambda jid: {"job_id": jid})
    monkeypatch.setattr(stems_router.job_store, "job_to_response", _fake_job_to_response)

    req = BatchStemSplitRequest(songs=None, enhance=True, model="htdemucs", skip_existing=True)
    stems_router.stem_split_batch(req)

    assert "Has Full Wav" in captured["songs"]
    assert "Only Enhanced Left" in captured["songs"], (
        "a song with only full_enhanced.wav (no full.wav) must still be "
        "included in the default batch — it has usable source audio"
    )
    assert "No Audio At All" not in captured["songs"], (
        "a song with no audio source at all has nothing for Demucs to "
        "process and should stay excluded"
    )
    assert "Stems Only No Single File" not in captured["songs"], (
        "a fully-split song with no remaining single-file source has "
        "nothing separate_song_stems() can read — has_source_audio() "
        "wrongly includes it (it counts stems); resolve_source_file() "
        "correctly excludes it"
    )


def test_explicit_song_list_rejects_path_traversal(fake_library, monkeypatch):
    """req.songs is caller-supplied. Before this fix, names went straight
    to task_batch_stem_split() -> separate_song_stems() ->
    _prune_after_split() -> unlink(), with no validation at all — the same
    traversal confinement _require_song() already applies to every other
    song-name endpoint. A bad name must be dropped, not passed through."""
    from scripts.api.routers import stems as stems_router
    from scripts.api.schemas import BatchStemSplitRequest

    captured = {}
    monkeypatch.setattr(stems_router.job_store, "create_job", lambda *a, **kw: "job-id")
    monkeypatch.setattr(
        stems_router.job_store, "submit_job",
        lambda job_id, fn, **kwargs: captured.__setitem__("songs", kwargs["songs"]),
    )
    monkeypatch.setattr(stems_router.job_store, "get_job", lambda jid: {"job_id": jid})
    monkeypatch.setattr(stems_router.job_store, "job_to_response", lambda job: job)

    req = BatchStemSplitRequest(
        songs=["Has Full Wav", "../../etc/passwd", "../secrets"],
        enhance=True, model="htdemucs", skip_existing=True,
    )
    stems_router.stem_split_batch(req)

    assert captured["songs"] == ["Has Full Wav"], (
        "traversal-shaped names must be dropped; only the real, existing "
        "library song should reach the batch job"
    )


def test_explicit_song_list_all_invalid_returns_400(fake_library, monkeypatch):
    from fastapi import HTTPException

    from scripts.api.routers import stems as stems_router
    from scripts.api.schemas import BatchStemSplitRequest

    req = BatchStemSplitRequest(
        songs=["../../etc/passwd", "Nonexistent Song"],
        enhance=True, model="htdemucs", skip_existing=True,
    )
    with pytest.raises(HTTPException) as exc_info:
        stems_router.stem_split_batch(req)
    assert exc_info.value.status_code == 400
