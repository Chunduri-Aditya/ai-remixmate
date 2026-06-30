import numpy as np
import pytest

from scripts.core import vocal_analyzer
from scripts.core.vocal_analyzer import (
    F0Curve,
    analyze_vocal_performance,
    estimate_f0,
)


def _sine(freq_hz: float, sr: int, duration_s: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * duration_s), dtype=np.float64) / sr
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _modulated_sine(
    base_hz: float,
    sr: int,
    duration_s: float,
    *,
    vibrato_rate_hz: float,
    vibrato_extent_cents: float,
    amp: float = 0.5,
) -> np.ndarray:
    t = np.arange(int(sr * duration_s), dtype=np.float64) / sr
    cents = vibrato_extent_cents * np.sin(2.0 * np.pi * vibrato_rate_hz * t)
    inst_freq = base_hz * (2.0 ** (cents / 1200.0))
    phase = np.cumsum(2.0 * np.pi * inst_freq / sr)
    return (amp * np.sin(phase)).astype(np.float32)


class TestVocalAnalyzer:
    def test_silence_returns_empty_report(self):
        audio = np.zeros(22050, dtype=np.float32)

        report = analyze_vocal_performance(audio, 22050, backend="autocorrelation")

        assert report.mean_pitch_hz == 0.0
        assert report.pitch_std_cents == 0.0
        assert report.vibrato_rate_hz is None
        assert report.vibrato_extent_cents is None
        assert report.phrase_count == 0
        assert report.avg_phrase_duration_s == 0.0
        assert report.energy_dynamics_db == 0.0
        assert report.duration_s == pytest.approx(1.0)

    def test_autocorrelation_tracks_stable_pitch(self):
        sr = 22050
        audio = _sine(220.0, sr, 2.0)

        report = analyze_vocal_performance(audio, sr, backend="autocorrelation")

        assert report.backend == "autocorrelation"
        assert report.mean_pitch_hz == pytest.approx(220.0, abs=2.0)
        assert report.pitch_std_cents < 15.0
        assert report.vibrato_rate_hz is None
        assert report.phrase_count == 1
        assert report.voiced_fraction > 0.85

    def test_vibrato_rate_and_extent_are_detected(self):
        sr = 16000
        audio = _modulated_sine(
            220.0,
            sr,
            4.0,
            vibrato_rate_hz=6.0,
            vibrato_extent_cents=55.0,
        )

        report = analyze_vocal_performance(audio, sr, backend="autocorrelation")

        assert report.vibrato_rate_hz == pytest.approx(6.0, abs=0.6)
        assert report.vibrato_extent_cents is not None
        assert report.vibrato_extent_cents > 15.0
        assert report.pitch_std_cents > 10.0

    def test_phrase_count_and_energy_dynamics(self):
        sr = 22050
        audio = np.concatenate(
            [
                _sine(196.0, sr, 0.60, amp=0.18),
                np.zeros(int(sr * 0.45), dtype=np.float32),
                _sine(196.0, sr, 0.70, amp=0.55),
            ]
        )

        report = analyze_vocal_performance(audio, sr, backend="autocorrelation")

        assert report.phrase_count == 2
        assert report.avg_phrase_duration_s == pytest.approx(0.65, abs=0.12)
        assert report.energy_dynamics_db > 6.0

    def test_estimate_f0_crepe_failure_falls_back(self, monkeypatch):
        def _raise_import_error(*_args, **_kwargs):
            raise ImportError("no crepe in test")

        monkeypatch.setattr(vocal_analyzer, "_estimate_f0_crepe", _raise_import_error)
        audio = _sine(330.0, 22050, 1.0)

        curve = estimate_f0(audio, 22050, backend="crepe")

        assert curve.backend == "autocorrelation"
        assert np.nanmedian(curve.frequency_hz) == pytest.approx(330.0, abs=3.0)

    def test_crepe_backend_result_is_respected(self, monkeypatch):
        def _fake_crepe(audio, sr, **_kwargs):
            times = np.arange(0.0, len(audio) / sr, 0.02, dtype=np.float32)
            return F0Curve(
                times_s=times,
                frequency_hz=np.full(times.shape, 440.0, dtype=np.float32),
                confidence=np.full(times.shape, 0.99, dtype=np.float32),
                backend="crepe",
            )

        monkeypatch.setattr(vocal_analyzer, "_estimate_f0_crepe", _fake_crepe)
        audio = _sine(220.0, 22050, 1.0)

        report = analyze_vocal_performance(audio, 22050, backend="auto")

        assert report.backend == "crepe"
        assert report.mean_pitch_hz == pytest.approx(440.0, abs=0.1)

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError, match="sample rate"):
            analyze_vocal_performance(np.ones(100, dtype=np.float32), 0)
