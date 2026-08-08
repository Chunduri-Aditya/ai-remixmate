"""
tests/test_batch_skip_existing.py

Regression test: task_batch_stem_split()'s skip_existing check tested only
vocals.{wav,flac}. Combined with the new default-discovery fix (which
correctly includes songs whose only remaining source is full_enhanced.wav,
some of which — mid-batch backlog songs — carry a single stray vocals stem
from a partial prior run), a song with 1-3 stems got skipped as "already
split" and reported with a fabricated stems: [vocals, drums, bass, other]
list, permanently hiding the fact that 3 of its 4 stems were never
produced. Fixed to require all 4 stems present before skipping.
"""

from __future__ import annotations

import pytest

SR = 44100


@pytest.fixture
def fake_library(tmp_path, monkeypatch):
    import scripts.api.task_modules.stems as stems_task_mod
    import scripts.core.paths as paths

    lib = tmp_path / "library"
    lib.mkdir()
    monkeypatch.setattr(paths, "LIBRARY_DIR", lib)
    monkeypatch.setattr(stems_task_mod, "LIBRARY_DIR", lib)
    monkeypatch.setattr(stems_task_mod, "update_job", lambda *a, **kw: None)
    monkeypatch.setattr(stems_task_mod, "_prune_after_split", lambda name: None)

    return lib


def test_partial_stems_are_not_skipped(fake_library, monkeypatch):
    """A song with only vocals.wav (the mid-batch backlog shape) must NOT
    be treated as already-split — it needs the other 3 stems."""
    import scripts.api.task_modules.stems as stems_task_mod
    from scripts.core.stems import StemResult

    lib = fake_library
    song_dir = lib / "Partial Song"
    song_dir.mkdir()
    (song_dir / "vocals.wav").write_bytes(b"\x00" * 512)  # only 1/4 stems

    called_with = []

    def _fake_separate(song_name, enhance=True, model="htdemucs", progress_cb=None):
        called_with.append(song_name)
        return StemResult(song=song_name, success=True, stems={
            "vocals": song_dir / "vocals.wav",
            "drums": song_dir / "drums.wav",
            "bass": song_dir / "bass.wav",
            "other": song_dir / "other.wav",
        })

    monkeypatch.setattr("scripts.core.stems.separate_song_stems", _fake_separate)

    result = stems_task_mod.task_batch_stem_split(
        "job-1", ["Partial Song"], skip_existing=True,
    )

    assert called_with == ["Partial Song"], (
        "a song with only 1/4 stems must be processed, not skipped"
    )
    assert result["results"][0].get("skipped") is not True


def test_all_four_stems_present_is_skipped(fake_library, monkeypatch):
    import scripts.api.task_modules.stems as stems_task_mod

    lib = fake_library
    song_dir = lib / "Complete Song"
    song_dir.mkdir()
    for stem in ("vocals", "drums", "bass", "other"):
        (song_dir / f"{stem}.wav").write_bytes(b"\x00" * 512)

    called_with = []

    def _fake_separate(song_name, **kw):
        called_with.append(song_name)
        raise AssertionError("should never be called — all 4 stems already exist")

    monkeypatch.setattr("scripts.core.stems.separate_song_stems", _fake_separate)

    result = stems_task_mod.task_batch_stem_split(
        "job-1", ["Complete Song"], skip_existing=True,
    )

    assert called_with == []
    assert result["results"][0]["skipped"] is True


def test_mixed_wav_and_flac_stems_count_toward_completeness(fake_library, monkeypatch):
    """Post-compression a song can have some stems as .flac and others still
    as .wav (or all .flac) — completeness must check both extensions."""
    import scripts.api.task_modules.stems as stems_task_mod

    lib = fake_library
    song_dir = lib / "Flac Mixed Song"
    song_dir.mkdir()
    (song_dir / "vocals.flac").write_bytes(b"\x00" * 512)
    (song_dir / "drums.flac").write_bytes(b"\x00" * 512)
    (song_dir / "bass.wav").write_bytes(b"\x00" * 512)
    (song_dir / "other.wav").write_bytes(b"\x00" * 512)

    def _fake_separate(song_name, **kw):
        raise AssertionError("should never be called — all 4 stems already exist (mixed formats)")

    monkeypatch.setattr("scripts.core.stems.separate_song_stems", _fake_separate)

    result = stems_task_mod.task_batch_stem_split(
        "job-1", ["Flac Mixed Song"], skip_existing=True,
    )
    assert result["results"][0]["skipped"] is True
