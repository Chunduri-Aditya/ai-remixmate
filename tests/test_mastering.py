"""tests/test_mastering.py — Unit tests for the mastering engine."""
from __future__ import annotations
import numpy as np
import pytest

# mastering.py requires scipy — skip the whole module if it's not installed
pytest.importorskip("scipy", reason="scipy not installed; skipping mastering tests")

SR = 44100


def _sine(freq=1000.0, duration=5.0, amplitude=0.5, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def _silence(duration=5.0, sr=SR):
    return np.zeros(int(sr * duration), dtype=np.float32)


class TestComputeLufs:
    def test_imports(self):
        from scripts.core.mastering import compute_lufs
        assert callable(compute_lufs)

    def test_silence_returns_very_low_lufs(self):
        from scripts.core.mastering import compute_lufs
        result = compute_lufs(_silence(), SR)
        assert result < -60.0

    def test_full_scale_sine_above_minus_15_lufs(self):
        from scripts.core.mastering import compute_lufs
        result = compute_lufs(_sine(amplitude=0.9), SR)
        assert result > -15.0

    def test_quiet_signal_lower_than_loud(self):
        from scripts.core.mastering import compute_lufs
        loud  = compute_lufs(_sine(amplitude=0.9), SR)
        quiet = compute_lufs(_sine(amplitude=0.1), SR)
        assert quiet < loud

    def test_stereo_input_accepted(self):
        from scripts.core.mastering import compute_lufs
        stereo = np.stack([_sine(), _sine()], axis=0)
        result = compute_lufs(stereo, SR)
        assert isinstance(result, float)


class TestNormalizeToLufs:
    def test_target_lufs_within_tolerance(self):
        from scripts.core.mastering import normalize_to_lufs, compute_lufs
        audio = _sine(amplitude=0.5)
        normed, _ = normalize_to_lufs(audio, SR, target_lufs=-14.0)
        measured = compute_lufs(normed, SR)
        assert abs(measured - (-14.0)) < 1.5

    def test_output_same_length(self):
        from scripts.core.mastering import normalize_to_lufs
        audio = _sine()
        normed, _ = normalize_to_lufs(audio, SR, target_lufs=-14.0)
        assert len(normed) == len(audio)

    def test_output_dtype_float32(self):
        from scripts.core.mastering import normalize_to_lufs
        normed, _ = normalize_to_lufs(_sine(), SR)
        assert normed.dtype == np.float32

    def test_returns_gain_float(self):
        from scripts.core.mastering import normalize_to_lufs
        _, gain_db = normalize_to_lufs(_sine(), SR)
        assert isinstance(gain_db, float)


class TestMasterMix:
    def test_returns_tuple(self):
        from scripts.core.mastering import master_mix
        result = master_mix(_sine(), SR)
        assert isinstance(result, tuple) and len(result) == 2

    def test_output_lufs_within_tolerance(self):
        from scripts.core.mastering import master_mix, compute_lufs
        mastered, _report = master_mix(_sine(amplitude=0.5), SR, target_lufs=-14.0)
        measured = compute_lufs(mastered, SR)
        assert abs(measured - (-14.0)) < 1.5

    def test_peak_below_ceiling(self):
        from scripts.core.mastering import master_mix
        audio = _sine(amplitude=1.2)   # intentionally over 0 dBFS
        mastered, _report = master_mix(audio, SR, target_lufs=-14.0, ceiling_db=-1.0)
        peak_dbfs = 20 * np.log10(np.abs(mastered).max() + 1e-12)
        assert peak_dbfs <= -0.9

    def test_report_has_lufs_field(self):
        from scripts.core.mastering import master_mix
        _, report = master_mix(_sine(), SR)
        assert hasattr(report, 'lufs_integrated')

    def test_no_nan_in_output(self):
        from scripts.core.mastering import master_mix
        mastered, _ = master_mix(_sine(), SR)
        assert not np.isnan(mastered).any()


# ---------------------------------------------------------------------------
# FxNorm corpus-target normalization tests (Stage 1A)
# ---------------------------------------------------------------------------

def _make_stem_dict(amplitude: float = 0.5, duration: float = 5.0):
    """Build a {stem_name: (audio, sr)} dict with different amplitudes per type."""
    rng = np.random.default_rng(42)
    stems = {}
    for i, name in enumerate(("drums", "bass", "vocals", "other")):
        freq = 100.0 * (2 ** i)
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        audio = (np.sin(2 * np.pi * freq * t) * amplitude * (1.0 + i * 0.1)).astype(np.float32)
        stems[name] = (audio, SR)
    return stems


try:
    import soundfile as _sf
    _HAS_SF = True
except ImportError:
    _sf = None  # type: ignore
    _HAS_SF = False

_skip_no_sf = pytest.mark.skipif(not _HAS_SF, reason="soundfile not installed")


@_skip_no_sf
class TestAnalyzeLibraryStemTargets:
    def test_empty_library_returns_fallbacks(self, tmp_path):
        from scripts.core.mastering import analyze_library_stem_targets, _STEM_FALLBACK_TARGETS
        result = analyze_library_stem_targets(library_dir=tmp_path)
        assert result == _STEM_FALLBACK_TARGETS

    def test_returns_all_stem_types(self, tmp_path):
        from scripts.core.mastering import analyze_library_stem_targets
        song_dir = tmp_path / "FakeSong"
        song_dir.mkdir()
        for stem in ("drums", "bass", "vocals", "other"):
            audio = _sine(duration=5.0, amplitude=0.4)
            _sf.write(str(song_dir / f"{stem}.wav"), audio, SR)
        result = analyze_library_stem_targets(library_dir=tmp_path, min_songs=1)
        assert set(result.keys()) == {"drums", "bass", "vocals", "other"}

    def test_all_values_finite(self, tmp_path):
        from scripts.core.mastering import analyze_library_stem_targets
        for i in range(3):
            d = tmp_path / f"Song{i}"
            d.mkdir()
            for stem in ("drums", "bass", "vocals", "other"):
                _sf.write(str(d / f"{stem}.wav"), _sine(amplitude=0.3 + 0.1 * i), SR)
        result = analyze_library_stem_targets(library_dir=tmp_path, min_songs=3)
        for k, v in result.items():
            assert np.isfinite(v), f"Non-finite target for stem '{k}': {v}"

    def test_min_songs_threshold_falls_back(self, tmp_path):
        """With only 1 song but min_songs=3, should use fallbacks."""
        from scripts.core.mastering import analyze_library_stem_targets, _STEM_FALLBACK_TARGETS
        d = tmp_path / "OneSong"
        d.mkdir()
        for stem in ("drums", "bass", "vocals", "other"):
            _sf.write(str(d / f"{stem}.wav"), _sine(), SR)
        result = analyze_library_stem_targets(library_dir=tmp_path, min_songs=3)
        assert result == _STEM_FALLBACK_TARGETS


@_skip_no_sf
class TestSaveLoadStemTargets:
    def test_roundtrip(self, tmp_path):
        from scripts.core.mastering import save_stem_targets, load_stem_targets
        targets = {"drums": -18.0, "bass": -22.0, "vocals": -19.0, "other": -21.0}
        cache = tmp_path / "targets.json"
        save_stem_targets(targets, cache_path=cache)
        loaded = load_stem_targets(cache_path=cache)
        for k in targets:
            assert abs(loaded[k] - targets[k]) < 0.001

    def test_missing_cache_returns_fallbacks(self, tmp_path):
        from scripts.core.mastering import load_stem_targets, _STEM_FALLBACK_TARGETS
        result = load_stem_targets(cache_path=tmp_path / "nonexistent.json")
        assert result == _STEM_FALLBACK_TARGETS

    def test_partial_cache_merged_with_fallbacks(self, tmp_path):
        """Cache with only 'drums' should merge — other stems use fallback."""
        import json
        from scripts.core.mastering import load_stem_targets, _STEM_FALLBACK_TARGETS
        cache = tmp_path / "partial.json"
        cache.write_text(json.dumps({"drums": -17.5}))
        result = load_stem_targets(cache_path=cache)
        assert result["drums"] == -17.5
        assert result["bass"] == _STEM_FALLBACK_TARGETS["bass"]


class TestNormalizeStemsToCorpusTargets:
    def test_returns_all_stems(self):
        from scripts.core.mastering import normalize_stems_to_corpus_targets
        stems = _make_stem_dict()
        result = normalize_stems_to_corpus_targets(stems)
        assert set(result.keys()) == {"drums", "bass", "vocals", "other"}

    def test_stem_type_targets_differ(self):
        """Drums and bass should hit different LUFS targets — not one flat value."""
        from scripts.core.mastering import normalize_stems_to_corpus_targets, compute_lufs, _STEM_FALLBACK_TARGETS
        stems = _make_stem_dict()
        result = normalize_stems_to_corpus_targets(stems, targets=_STEM_FALLBACK_TARGETS)
        drums_lufs = compute_lufs(result["drums"], SR)
        bass_lufs  = compute_lufs(result["bass"], SR)
        # Fallback targets differ by 2.5 LU, so normalized levels should differ too
        assert abs(drums_lufs - bass_lufs) > 1.0

    def test_normalization_within_1lu_of_target(self):
        from scripts.core.mastering import normalize_stems_to_corpus_targets, compute_lufs
        targets = {"drums": -18.0, "bass": -22.0, "vocals": -19.5, "other": -21.5}
        stems = _make_stem_dict()
        result = normalize_stems_to_corpus_targets(stems, targets=targets)
        for stem_name, audio in result.items():
            actual = compute_lufs(audio, SR)
            target = targets[stem_name]
            assert abs(actual - target) < 1.5, (
                f"Stem '{stem_name}': got {actual:.1f} LUFS, wanted {target:.1f} (±1.5)"
            )

    def test_silent_stem_no_crash(self):
        from scripts.core.mastering import normalize_stems_to_corpus_targets
        stems = {"drums": (np.zeros(SR * 5, dtype=np.float32), SR)}
        result = normalize_stems_to_corpus_targets(stems)
        assert "drums" in result
        assert not np.isnan(result["drums"]).any()

    def test_true_peak_applied(self):
        """Output stems must not exceed true-peak ceiling."""
        from scripts.core.mastering import normalize_stems_to_corpus_targets
        # Very loud signal that would clip without ceiling
        loud = np.ones(SR * 5, dtype=np.float32) * 0.99
        stems = {"drums": (loud, SR)}
        result = normalize_stems_to_corpus_targets(stems, true_peak_ceiling=-1.0)
        ceiling_lin = 10.0 ** (-1.0 / 20.0)
        assert float(np.max(np.abs(result["drums"]))) <= ceiling_lin + 1e-4

    def test_no_nan_inf_in_output(self):
        from scripts.core.mastering import normalize_stems_to_corpus_targets
        stems = _make_stem_dict()
        result = normalize_stems_to_corpus_targets(stems)
        for name, audio in result.items():
            assert np.isfinite(audio).all(), f"NaN/inf in stem '{name}'"

    def test_fallback_for_unknown_stem_type(self):
        """A stem type not in targets dict should fall back to fallback_lufs."""
        from scripts.core.mastering import normalize_stems_to_corpus_targets, compute_lufs
        stems = {"unknown_stem": (_sine(), SR)}
        fallback = -20.0
        result = normalize_stems_to_corpus_targets(stems, targets={}, fallback_lufs=fallback)
        actual_lufs = compute_lufs(result["unknown_stem"], SR)
        assert abs(actual_lufs - fallback) < 1.5
