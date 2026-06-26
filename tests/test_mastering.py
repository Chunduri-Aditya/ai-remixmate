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
