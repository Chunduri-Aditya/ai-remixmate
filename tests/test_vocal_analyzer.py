import numpy as np
import pytest

from scripts.core import vocal_analyzer
from scripts.core.track_metadata import MetadataClient, TrackMetadata
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


class TestVocalClashPenaltyIntegration:
    """
    DJ_ENGINE_IMPROVEMENT_PLAN.md §1.2 — vocal_analyzer.py's voiced_fraction
    + mean_pitch_hz feed compatibility_score()'s vocal_clash_penalty as a
    register-aware upgrade over the old vocal_density energy-ratio proxy,
    which scored two vocals purely on how much energy sat in the 300-3400 Hz
    band and could not distinguish "same register, high masking risk" from
    "an octave-plus apart, low masking risk".
    """

    @staticmethod
    def _vocal_report(freq_hz: float, sr: int = 22050, duration_s: float = 2.0):
        audio = _sine(freq_hz, sr, duration_s, amp=0.5)
        return analyze_vocal_performance(audio, sr, backend="autocorrelation")

    @staticmethod
    def _meta_from_report(report) -> TrackMetadata:
        return TrackMetadata(
            title="t", artist="a", bpm=120.0, camelot="8A",
            vocal_voiced_fraction=report.voiced_fraction,
            vocal_mean_pitch_hz=report.mean_pitch_hz,
        )

    def test_same_register_scores_higher_clash_than_different_register(self):
        client = MetadataClient()

        report_a = self._vocal_report(220.00)    # A3
        report_close = self._vocal_report(246.94)  # B3, ~196 cents above A3 — same register
        report_far = self._vocal_report(880.00)   # A5, 2400 cents above A3 — two octaves apart

        meta_a = self._meta_from_report(report_a)
        meta_close = self._meta_from_report(report_close)
        meta_far = self._meta_from_report(report_far)

        assert meta_a.vocal_voiced_fraction > 0.85   # sanity: strong "vocal" in both
        assert meta_far.vocal_voiced_fraction > 0.85

        score_close = client.compatibility_score(meta_a, meta_close)
        score_far = client.compatibility_score(meta_a, meta_far)

        # Same distinction the old vocal_density*vocal_density proxy could
        # never make: two equally "strong vocal" tracks score very
        # differently depending on register overlap.
        assert score_close["vocal_clash_penalty"] > score_far["vocal_clash_penalty"]
        assert score_far["vocal_clash_penalty"] == 0.0
        assert score_close["vocal_clash_penalty"] > 0.0

    def test_falls_back_to_vocal_density_when_analyzer_data_absent(self):
        """Backward compatibility: no vocal_analyzer data on either track ->
        identical to the pre-existing vocal_density*vocal_density behavior."""
        client = MetadataClient()
        meta_a = TrackMetadata(title="a", artist="", bpm=120.0, camelot="8A", vocal_density=0.6)
        meta_b = TrackMetadata(title="b", artist="", bpm=120.0, camelot="8A", vocal_density=0.6)

        score = client.compatibility_score(meta_a, meta_b)

        assert score["vocal_clash_penalty"] == pytest.approx(min(0.30, 0.6 * 0.6), abs=1e-6)
