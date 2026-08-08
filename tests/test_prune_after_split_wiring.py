"""
tests/test_prune_after_split_wiring.py

Regression test: scripts/api/task_modules/stems.py's task_stem_split() and
task_batch_stem_split() previously never reclaimed full.wav/full_enhanced.wav
after a successful split — only the legacy scripts/download.py CLI path
called prune_raw(). Every song split through the live API (single or batch)
was left carrying a permanently redundant full-length audio file. Confirmed
live: after triggering a batch split on the 185-song backlog, 127 completed
songs still had full_enhanced.wav sitting on disk (~4GB) even though real
stems already existed for them.
"""

from __future__ import annotations

import pytest

from scripts.core.stems import StemResult


@pytest.fixture
def patched_env(tmp_path, monkeypatch):
    import scripts.api.task_modules.stems as stems_task_mod
    import scripts.core.paths as paths

    lib = tmp_path / "library"
    lib.mkdir()
    monkeypatch.setattr(paths, "LIBRARY_DIR", lib)
    monkeypatch.setattr(stems_task_mod, "LIBRARY_DIR", lib)
    monkeypatch.setattr(stems_task_mod, "update_job", lambda *a, **kw: None)

    pruned_calls = []

    class _FakeMgr:
        def prune_raw(self, name):
            pruned_calls.append(("raw", name))
            return True

        def prune_enhanced(self, name):
            pruned_calls.append(("enhanced", name))
            return True

    import scripts.core.library as library_mod
    monkeypatch.setattr(library_mod, "get_library_manager", lambda: _FakeMgr())

    return lib, pruned_calls


def test_single_split_prunes_on_success(patched_env, monkeypatch):
    import scripts.api.task_modules.stems as stems_task_mod

    lib, pruned_calls = patched_env

    def _fake_separate(song_name, enhance=True, model="htdemucs", progress_cb=None):
        return StemResult(song=song_name, success=True, stems={"vocals": lib / "x"})

    monkeypatch.setattr("scripts.core.stems.separate_song_stems", _fake_separate)

    stems_task_mod.task_stem_split("job-1", "My Song")

    assert ("raw", "My Song") in pruned_calls
    assert ("enhanced", "My Song") in pruned_calls


def test_single_split_does_not_prune_on_failure(patched_env, monkeypatch):
    import scripts.api.task_modules.stems as stems_task_mod

    lib, pruned_calls = patched_env

    def _fake_separate(song_name, enhance=True, model="htdemucs", progress_cb=None):
        return StemResult(song=song_name, success=False, error="Demucs failed")

    monkeypatch.setattr("scripts.core.stems.separate_song_stems", _fake_separate)

    with pytest.raises(RuntimeError):
        stems_task_mod.task_stem_split("job-1", "Broken Song")

    assert pruned_calls == []


def test_batch_split_prunes_only_successful_songs(patched_env, monkeypatch):
    import scripts.api.task_modules.stems as stems_task_mod

    lib, pruned_calls = patched_env
    for name in ("Good Song", "Bad Song"):
        (lib / name).mkdir()

    def _fake_separate(song_name, enhance=True, model="htdemucs", progress_cb=None):
        if song_name == "Good Song":
            return StemResult(song=song_name, success=True, stems={"vocals": lib / "x"})
        return StemResult(song=song_name, success=False, error="Demucs failed")

    monkeypatch.setattr("scripts.core.stems.separate_song_stems", _fake_separate)

    stems_task_mod.task_batch_stem_split("job-1", ["Good Song", "Bad Song"], skip_existing=False)

    assert ("raw", "Good Song") in pruned_calls
    assert ("enhanced", "Good Song") in pruned_calls
    assert ("raw", "Bad Song") not in pruned_calls
    assert ("enhanced", "Bad Song") not in pruned_calls


def test_prune_after_split_respects_prune_on_download_flag(patched_env, monkeypatch):
    """When library.prune_on_download is false, _prune_after_split() must
    not delete anything — keep_raw_after_separation-style opt-out."""
    import scripts.api.task_modules.stems as stems_task_mod

    lib, pruned_calls = patched_env

    class _FakeLibraryCfg:
        prune_on_download = False

    class _FakeCfg:
        library = _FakeLibraryCfg()

    monkeypatch.setattr("scripts.core.config.cfg", _FakeCfg())

    stems_task_mod._prune_after_split("Any Song")
    assert pruned_calls == []
