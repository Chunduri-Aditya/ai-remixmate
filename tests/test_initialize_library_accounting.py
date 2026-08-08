"""
tests/test_initialize_library_accounting.py

Regression test for a real bug found while diagnosing the live library's
backlog: task_initialize_library()'s Phase 1 loop called
separate_song_stems() and assumed any failure would raise — but
separate_song_stems() reports failure via StemResult.success=False, not by
raising. That made the `except Exception` branch unreachable for the common
case (a Demucs subprocess failure), so every failed song was silently
counted into split_done with split_failed staying at 0.

This is exactly what happened on the real library: a single
"initialize_library" job reported {"split_done": 353, "split_failed": 0}
while, in reality, 185 of those "done" songs got nothing but an orphaned
full_enhanced.wav and no stems at all — confirmed by the job's own
started_at/finished_at window matching the mtimes of all 185 orphaned
full_enhanced.wav files to the second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 44100


@pytest.fixture
def fake_library(tmp_path, monkeypatch):
    """Three songs, none with stems yet, so all three enter Phase 1."""
    import scripts.core.paths as paths

    lib = tmp_path / "library"
    lib.mkdir()
    songs = ["Song A (will succeed)", "Song B (demucs fails)", "Song C (raises)"]
    for name in songs:
        d = lib / name
        d.mkdir()
        audio = 0.1 * np.sin(2 * np.pi * 440 * np.arange(int(1.0 * SR)) / SR).astype(np.float32)
        sf.write(str(d / "full.wav"), audio, SR)

    monkeypatch.setattr(paths, "LIBRARY_DIR", lib)

    import scripts.api.task_modules.analysis as analysis_mod
    monkeypatch.setattr(analysis_mod, "LIBRARY_DIR", lib)
    monkeypatch.setattr(analysis_mod, "log_audit", lambda *a, **kw: None)
    monkeypatch.setattr(analysis_mod, "update_job", lambda *a, **kw: None)

    return songs, lib


def test_demucs_failure_is_not_counted_as_split_done(fake_library, monkeypatch):
    """The core regression: a StemResult(success=False) must land in
    split_failed, not silently inflate split_done."""
    import scripts.api.task_modules.analysis as analysis_mod
    from scripts.core.stems import StemResult

    songs, _lib = fake_library
    monkeypatch.setattr(analysis_mod, "_index_upsert", lambda name: None)

    def _fake_separate(song_name, enhance=True, model="htdemucs"):
        if "will succeed" in song_name:
            return StemResult(song=song_name, success=True,
                              stems={"vocals": Path("vocals.wav")})
        if "demucs fails" in song_name:
            return StemResult(song=song_name, success=False,
                              error="Demucs exited with code 1.\nstderr: CUDA out of memory")
        raise RuntimeError("unexpected explosion")

    monkeypatch.setattr("scripts.core.stems.separate_song_stems", _fake_separate)

    result = analysis_mod.task_initialize_library(
        "test-job", enhance=True, model="htdemucs",
        run_compress=False, run_index=False,
    )

    assert result["split_done"] == 1, "only the genuinely successful song should count"
    assert result["split_failed"] == 2, "both the reported failure and the raised exception must be counted as failed"


def test_success_result_still_upserts_into_index(fake_library, monkeypatch):
    """Confirms the fix didn't break the normal success path — _index_upsert
    must still run exactly once, for the one song that actually succeeded."""
    import scripts.api.task_modules.analysis as analysis_mod
    from scripts.core.stems import StemResult

    songs, _lib = fake_library
    upserted = []
    monkeypatch.setattr(analysis_mod, "_index_upsert", lambda name: upserted.append(name))

    def _fake_separate(song_name, enhance=True, model="htdemucs"):
        if "will succeed" in song_name:
            return StemResult(song=song_name, success=True, stems={"vocals": Path("vocals.wav")})
        return StemResult(song=song_name, success=False, error="boom")

    monkeypatch.setattr("scripts.core.stems.separate_song_stems", _fake_separate)

    result = analysis_mod.task_initialize_library(
        "test-job", enhance=True, model="htdemucs",
        run_compress=False, run_index=False,
    )

    assert upserted == ["Song A (will succeed)"]
    assert result["split_done"] == 1
