"""
tests/test_storage_prune_safety.py

POST /library/storage/prune was extended to also sweep full_enhanced.wav
(previously nothing ever cleaned it up — see LibraryManager.prune_enhanced()
and scripts/api/task_modules/stems.py::_prune_after_split()). The critical
safety property under test here: this must NOT delete full_enhanced.wav for
a song that doesn't have all 4 stems yet. That song has no full.wav either
(the real backlog shape), so full_enhanced.wav is its ONLY remaining audio —
deleting it would recreate the exact "no source, no stems, unrecoverable
without a re-download" state this whole fix exists to undo, and would do
real damage to a stem-split batch that's still in flight.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def fake_library(tmp_path, monkeypatch):
    from scripts.core.library import LibraryManager

    lib = tmp_path / "library"
    lib.mkdir()

    # Song A: mid-batch — full_enhanced.wav written, Demucs not done yet.
    # This is the dangerous case: no full.wav, only 1/4 stems so far.
    a = lib / "Mid Batch Song"
    a.mkdir()
    (a / "full_enhanced.wav").write_bytes(b"\x00" * 4096)
    (a / "vocals.wav").write_bytes(b"\x00" * 512)

    # Song B: genuinely done — all 4 stems, full_enhanced.wav now redundant.
    # Sized at 2MB (not a few KB) so freed_mb doesn't round to 0.0 below.
    # Stems are real (if tiny) audio, not raw byte blobs — prune_enhanced()
    # health-probes each stem before trusting it as present.
    b = lib / "Finished Song"
    b.mkdir()
    (b / "full_enhanced.wav").write_bytes(b"\x00" * (2 * 1024 * 1024))
    tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(4410) / 44100)).astype(np.float32)
    for stem in ("vocals", "drums", "bass", "other"):
        sf.write(str(b / f"{stem}.wav"), tone, 44100)

    # Song C: freshly downloaded, never split at all — full.wav only, no
    # stems, no full_enhanced.wav. A prior version of this endpoint used
    # force=True on the full.wav branch, which bypasses the all-4-stems
    # check entirely and deletes full.wav here unconditionally — destroying
    # this song's only audio. Confirmed by direct reproduction; see the
    # force=False fix in routers/library.py::storage_prune().
    c = lib / "Never Split Song"
    c.mkdir()
    (c / "full.wav").write_bytes(b"\x00" * (2 * 1024 * 1024))

    mgr = LibraryManager(library_dir=lib)
    mgr.register("Mid Batch Song")
    mgr.register("Finished Song")
    mgr.register("Never Split Song")

    import scripts.core.library as library_mod
    monkeypatch.setattr(library_mod, "get_library_manager", lambda: mgr)

    return lib, a, b, c


def test_prune_endpoint_never_touches_mid_batch_songs(fake_library):
    from scripts.api.routers import library as library_router

    lib, mid_batch_dir, finished_dir, never_split_dir = fake_library

    response = library_router.storage_prune()

    assert (mid_batch_dir / "full_enhanced.wav").exists(), (
        "a song without all 4 stems must keep its only remaining source "
        "audio — pruning it here would be unrecoverable without a re-download"
    )
    assert (never_split_dir / "full.wav").exists(), (
        "a freshly-downloaded, never-split song must keep full.wav — it has "
        "zero stems, so deleting full.wav here destroys its only audio"
    )
    assert not (finished_dir / "full_enhanced.wav").exists(), (
        "a genuinely finished song's redundant full_enhanced.wav should be reclaimed"
    )
    assert "Finished Song" in response.pruned
    assert "Mid Batch Song" not in response.pruned
    assert "Never Split Song" not in response.pruned
    assert response.freed_mb > 0
