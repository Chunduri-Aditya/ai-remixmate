"""
tests/test_audio_sources.py — B3.4 "audio sources everywhere".

Every reader that opens a song's audio must resolve it through
scripts.core.audio_source (full.wav -> full_enhanced.wav -> FLAC/WAV stems),
not by probing full.wav directly. Under prune_on_download most songs keep only
full_enhanced.wav plus FLAC stems, so a full.wav only reader sees an empty
library.

Cases (each with the input that bites under the current code):
  recommend        a candidate with only full_enhanced.wav is recommended;
                   a source with only full_enhanced.wav gets a quick BPM
  music_index      a song with only full_enhanced.wav is indexed
  crate_digger     a song with only full_enhanced.wav is embedded
  library          .index.json counts FLAC stems as available
  analysis         a FLAC only song keeps vocal_mean_pitch_hz after re analysis
  analysis         chroma_vector is a 12 float chroma or absent, never chord
                   names and never zeros

Everything runs against a temp library; the real library/ and data/ trees
are never read or written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.dj_analysis

SR = 44100
STEMS = ("vocals", "drums", "bass", "other")


# ---------------------------------------------------------------------------
# Synthetic audio
# ---------------------------------------------------------------------------

def _tone(freq: float, secs: float, amp: float = 0.2, sr: int = SR) -> np.ndarray:
    t = np.arange(int(secs * sr), dtype=np.float64) / sr
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _clicks(bpm: float, secs: float, sr: int = SR) -> np.ndarray:
    n = int(secs * sr)
    out = np.zeros(n, dtype=np.float32)
    click_len = int(0.01 * sr)
    click = (np.hanning(click_len * 2)[click_len:] * 0.9).astype(np.float32)
    for beat_t in np.arange(0, secs, 60.0 / bpm):
        i = int(beat_t * sr)
        if i + click_len <= n:
            out[i:i + click_len] += click
    return out


def _pad(secs: float, sr: int = SR) -> np.ndarray:
    """C major pad: a tonal signal with a clearly non zero chroma."""
    return sum(_tone(f, secs, amp=0.12, sr=sr) for f in (261.63, 329.63, 392.0))


def _tonal_song(secs: float, bpm: float = 120.0, sr: int = SR) -> np.ndarray:
    return np.clip(_clicks(bpm, secs, sr) * 0.6 + _pad(secs, sr), -1.0, 1.0).astype(np.float32)


def _write(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def _meta(song_d: Path, **fields) -> None:
    song_d.mkdir(parents=True, exist_ok=True)
    (song_d / "meta.json").write_text(json.dumps(fields))


# ---------------------------------------------------------------------------
# Temp library wired into every module that binds LIBRARY_DIR / song_dir
# ---------------------------------------------------------------------------

@pytest.fixture
def lib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    lib = tmp_path / "library"
    lib.mkdir()
    data = tmp_path / "data"
    data.mkdir()

    import scripts.core.analysis_pipeline as analysis_pipeline
    import scripts.core.audio_source as audio_source
    import scripts.core.crate_digger as crate_digger
    import scripts.core.library as library
    import scripts.core.music_index as music_index
    import scripts.core.paths as paths

    def _song_dir(name: str) -> Path:
        return lib / name

    monkeypatch.setattr(paths, "LIBRARY_DIR", lib)
    monkeypatch.setattr(paths, "song_dir", _song_dir)
    monkeypatch.setattr(audio_source, "song_dir", _song_dir)
    monkeypatch.setattr(analysis_pipeline, "song_dir", _song_dir)
    monkeypatch.setattr(library, "_song_dir", _song_dir)

    monkeypatch.setattr(music_index, "LIBRARY_DIR", lib)
    monkeypatch.setattr(music_index, "DATA_DIR", data)
    monkeypatch.setattr(music_index, "INDEX_PATH", data / "music_index.json")
    monkeypatch.setattr(music_index, "_index_instance", None)

    monkeypatch.setattr(crate_digger, "LIBRARY_DIR", lib)
    monkeypatch.setattr(crate_digger, "DATA_DIR", data)
    monkeypatch.setattr(crate_digger, "_INDEX_EMB_PATH", data / "clap_index.npy")
    monkeypatch.setattr(crate_digger, "_INDEX_META_PATH", data / "clap_index_meta.json")
    return lib


# ---------------------------------------------------------------------------
# recommend.py
# ---------------------------------------------------------------------------

class TestRecommend:
    def test_candidate_with_only_full_enhanced_wav_is_recommended(self, lib):
        from scripts.core.recommend import get_recommendations

        src = lib / "Source"
        _write(src / "full.wav", _clicks(128.0, 5.0))
        _meta(src, bpm=128.0)

        cand = lib / "Enhanced Only"
        _write(cand / "full_enhanced.wav", _clicks(128.0, 5.0))
        _meta(cand, bpm=128.0)

        # Deny control: cached BPM but no audio at all must stay out.
        _meta(lib / "Shell", bpm=128.0)

        names = {r["name"] for r in get_recommendations(src, lib, limit=5)}
        assert "Enhanced Only" in names, f"full_enhanced.wav only song not recommended: {names}"
        assert "Shell" not in names, "song with no audio was recommended"

    def test_quick_bpm_resolves_full_enhanced_wav(self, lib):
        from scripts.core.recommend import get_or_cache_bpm

        d = lib / "Enhanced Uncached"
        _write(d / "full_enhanced.wav", _clicks(120.0, 12.0))

        bpm = get_or_cache_bpm(d)
        assert bpm is not None and bpm > 0, "no BPM from a full_enhanced.wav only song"


# ---------------------------------------------------------------------------
# music_index.py
# ---------------------------------------------------------------------------

class TestMusicIndex:
    def test_upsert_indexes_song_with_only_full_enhanced_wav(self, lib):
        from scripts.core.music_index import MusicIndex

        d = lib / "Enhanced Only"
        _write(d / "full_enhanced.wav", _tonal_song(8.0, bpm=128.0))
        (lib / "Shell").mkdir()

        idx = MusicIndex()
        assert idx.upsert_song("Enhanced Only") is True, "upsert_song saw no audio"
        assert "Enhanced Only" in idx._songs
        # Deny control: an empty folder is still not indexable.
        assert idx.upsert_song("Shell") is False


# ---------------------------------------------------------------------------
# crate_digger.py
# ---------------------------------------------------------------------------

class TestCrateDigger:
    def test_index_library_embeds_song_with_only_full_enhanced_wav(self, lib, monkeypatch):
        import scripts.core.crate_digger as cd

        d = lib / "Enhanced Only"
        _write(d / "full_enhanced.wav", _tonal_song(4.0, bpm=128.0))
        (lib / "Shell").mkdir()

        def _no_clap():
            raise ImportError("laion_clap not installed (test)")

        # The 35-D fallback reads MusicIndex._embeddings, an attribute
        # MusicIndex never defines; that defect is outside B3.4, so the embed
        # step is stubbed to isolate audio resolution. The stub records which
        # songs reached it, which only happens once an audio file resolved.
        embedded: list = []

        def _fake_embed(self, name, audio, sr):
            assert len(audio) > 0 and sr > 0
            embedded.append(name)
            v = np.ones(cd.CLAP_DIM, dtype=np.float32)
            return v / np.linalg.norm(v)

        monkeypatch.setattr(cd, "_load_clap_model", _no_clap)
        monkeypatch.setattr(cd.CrateDigger, "_fallback_embed", _fake_embed)

        digger = cd.CrateDigger()
        n = digger.index_library(library_dir=lib)

        assert embedded == ["Enhanced Only"], f"embedded={embedded}"
        assert n == 1
        assert "Enhanced Only" in digger._name_to_idx
        assert "Shell" not in digger._name_to_idx


# ---------------------------------------------------------------------------
# library.py
# ---------------------------------------------------------------------------

class TestLibraryIndex:
    def test_index_counts_flac_stems_as_available(self, lib):
        from scripts.core.library import LibraryManager

        flac = lib / "Stems FLAC"
        for s in STEMS:
            _write(flac / f"{s}.flac", _tone(220.0, 1.0))
        (lib / "No Stems").mkdir()
        (lib / "No Stems" / "license.json").write_text("{}")

        mgr = LibraryManager(library_dir=lib)
        mgr._rebuild_index()
        index = json.loads(mgr._index_path.read_text())

        assert sorted(index["Stems FLAC"]["stems_available"]) == sorted(STEMS), (
            f"FLAC stems not counted: {index['Stems FLAC']['stems_available']}"
        )
        assert index["No Stems"]["stems_available"] == []


# ---------------------------------------------------------------------------
# analysis_pipeline.py + music_intelligence.py
# ---------------------------------------------------------------------------

def _flac_only_song(lib: Path, name: str, secs: float = 20.0) -> Path:
    d = lib / name
    _write(d / "vocals.flac", _tone(220.0, secs, amp=0.4))
    _write(d / "drums.flac", _clicks(120.0, secs))
    _write(d / "bass.flac", _tone(110.0, secs, amp=0.2))
    _write(d / "other.flac", _pad(secs))
    return d


def _chroma_ok(cv) -> bool:
    """A chroma_vector is 12 finite floats with at least one non zero bin."""
    if not isinstance(cv, list) or len(cv) != 12:
        return False
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in cv):
        return False
    arr = np.asarray(cv, dtype=np.float64)
    return bool(np.isfinite(arr).all() and np.abs(arr).max() > 0.0)


class TestAnalysisPipeline:
    def test_flac_only_song_keeps_vocal_pitch_after_reanalysis(self, lib):
        from scripts.core.analysis_pipeline import run_song_analysis

        d = _flac_only_song(lib, "FLAC Only")
        # A previous analysis (when vocals.wav still existed) recorded a pitch.
        _meta(d, bpm=120.0, vocal_mean_pitch_hz=219.5, vocal_voiced_fraction=0.9)

        run_song_analysis("FLAC Only")

        meta = json.loads((d / "meta.json").read_text())
        assert "bpm" in meta and "key" in meta, f"analysis block did not write: {meta}"
        pitch = meta.get("vocal_mean_pitch_hz")
        assert pitch is not None, f"vocal_mean_pitch_hz dropped after re analysis: {sorted(meta)}"
        assert 100.0 < float(pitch) < 500.0, f"vocal pitch {pitch} Hz is not the 220 Hz stem"
        # Never a zero chroma: either the key is absent or it carries signal.
        cv = meta.get("chroma_vector")
        assert cv is None or _chroma_ok(cv), f"chroma_vector={cv!r}"

    def test_tonal_song_chroma_vector_is_chroma_not_chord_names(self, lib, monkeypatch):
        import librosa

        import scripts.core.dj_analysis as dj
        from scripts.core.analysis_pipeline import run_song_analysis

        d = lib / "Tonal"
        _write(d / "full.wav", _tonal_song(20.0, bpm=120.0))

        # A structure that provides chroma_vector (the mean chroma key
        # detection already computes). The pipeline must write that chroma
        # under "chroma_vector", or nothing; never the chord name list.
        real = dj.analyze_structure

        def with_chroma(audio, sr, *args, **kwargs):
            struct = real(audio, sr, *args, **kwargs)
            if not hasattr(struct, "chroma_vector"):
                chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
                struct.chroma_vector = [float(x) for x in chroma.mean(axis=1)]
            return struct

        monkeypatch.setattr(dj, "analyze_structure", with_chroma)

        run_song_analysis("Tonal")

        meta = json.loads((d / "meta.json").read_text())
        assert "bpm" in meta and "key" in meta, f"analysis block did not write: {meta}"
        cv = meta.get("chroma_vector")
        assert cv is None or _chroma_ok(cv), (
            f"chroma_vector must be 12 non zero floats or absent, got {cv!r}"
        )
        for k in ("beat_strength", "tempo_stability"):
            v = meta.get(k)
            assert v is None or (isinstance(v, float) and 0.0 <= v <= 1.0), f"{k}={v!r}"
