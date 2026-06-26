"""
tests/test_audio_source.py — Source-audio resolution priority + stem rebuild.

Verifies the resolver the DJ render path depends on:
  full.wav → full_enhanced.wav → reconstruct by summing the 4 Demucs stems.

Uses a temp library dir (monkeypatched song_dir) so no real library is touched.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

import scripts.core.audio_source as audio_source

SR = 22050


def _write(path, freq=440.0, secs=1.0, amp=0.3):
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    sf.write(str(path), (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32), SR)


@pytest.fixture
def song(tmp_path, monkeypatch):
    """A fake song dir; monkeypatch song_dir() to point at it."""
    d = tmp_path / "Fake Song"
    d.mkdir()
    monkeypatch.setattr(audio_source, "song_dir", lambda name: d)
    return d


class TestResolveSourceFile:
    def test_prefers_full_wav(self, song):
        _write(song / "full.wav")
        _write(song / "full_enhanced.wav")
        assert audio_source.resolve_source_file("Fake Song").name == "full.wav"

    def test_falls_back_to_enhanced(self, song):
        _write(song / "full_enhanced.wav")
        assert audio_source.resolve_source_file("Fake Song").name == "full_enhanced.wav"

    def test_none_when_no_single_file(self, song):
        _write(song / "vocals.flac")  # stem only
        assert audio_source.resolve_source_file("Fake Song") is None


class TestLoadSourceAudio:
    def test_loads_full_wav(self, song):
        _write(song / "full.wav")
        audio, sr = audio_source.load_source_audio("Fake Song", sr=SR)
        assert sr == SR
        assert audio.dtype == np.float32 and len(audio) > 0

    def test_loads_enhanced_when_no_full(self, song):
        _write(song / "full_enhanced.wav")
        audio, _ = audio_source.load_source_audio("Fake Song", sr=SR)
        assert np.isfinite(audio).all() and len(audio) > 0

    def test_reconstructs_from_stems(self, song):
        # Four stems, no full mix — must sum to a usable, non-clipping signal.
        for name, f in (("vocals", 220.0), ("drums", 440.0), ("bass", 110.0), ("other", 880.0)):
            _write(song / f"{name}.flac", freq=f)
        audio, sr = audio_source.load_source_audio("Fake Song", sr=SR)
        assert sr == SR
        assert len(audio) > 0
        assert np.isfinite(audio).all()
        assert float(np.abs(audio).max()) <= 1.0  # peak-normalised, no clipping

    def test_raises_when_no_source(self, song):
        # Empty dir — no full mix, no stems.
        with pytest.raises(FileNotFoundError, match="Fake Song"):
            audio_source.load_source_audio("Fake Song", sr=SR)

    def test_has_source_audio(self, song):
        assert audio_source.has_source_audio("Fake Song") is False
        _write(song / "drums.wav")
        assert audio_source.has_source_audio("Fake Song") is True


class TestPurge:
    def _make_lib(self, tmp_path, monkeypatch):
        lib = tmp_path / "library"
        lib.mkdir()
        monkeypatch.setattr(audio_source, "LIBRARY_DIR", lib, raising=False)
        # patch the names the functions import lazily from paths
        import scripts.core.paths as paths
        monkeypatch.setattr(paths, "LIBRARY_DIR", lib)
        monkeypatch.setattr(paths, "song_dir", lambda name: lib / name)
        monkeypatch.setattr(audio_source, "song_dir", lambda name: lib / name)
        return lib

    def test_find_and_purge(self, tmp_path, monkeypatch):
        lib = self._make_lib(tmp_path, monkeypatch)

        # usable: has a full mix
        (lib / "Good").mkdir()
        _write(lib / "Good" / "full.wav")
        # usable: has a stem
        (lib / "StemOnly").mkdir()
        _write(lib / "StemOnly" / "vocals.flac")
        # unusable: metadata shell only
        (lib / "Shell").mkdir()
        (lib / "Shell" / "full.info.json").write_text("{}")
        (lib / "Shell" / "license.json").write_text("{}")

        assert audio_source.find_unusable_songs() == ["Shell"]

        # dry run deletes nothing
        assert audio_source.purge_unusable_songs(dry_run=True) == ["Shell"]
        assert (lib / "Shell").exists()

        # apply deletes only the shell
        assert audio_source.purge_unusable_songs(dry_run=False) == ["Shell"]
        assert not (lib / "Shell").exists()
        assert (lib / "Good").exists()
        assert (lib / "StemOnly").exists()
