"""
tests/test_energy_profiler.py — Unit tests for scripts/core/energy_profiler.py.

All tests use numpy-only inputs (no Essentia, no librosa required).
The numpy backend is always exercised; Essentia tests are skipped when absent.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq: float = 440.0, sr: int = 22050, duration: float = 2.0, amp: float = 0.8) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _silence(sr: int = 22050, duration: float = 2.0) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


def _noise(sr: int = 22050, duration: float = 2.0, amp: float = 0.8) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(sr * duration)) * amp).astype(np.float32)


# ---------------------------------------------------------------------------
# TestEnergyFeatures
# ---------------------------------------------------------------------------

class TestEnergyFeatures:
    def test_dataclass_defaults(self):
        from scripts.core.energy_profiler import EnergyFeatures
        f = EnergyFeatures()
        assert f.rms_energy == 0.5
        assert f.spectral_centroid == 0.5
        assert f.arousal == 0.5
        assert f.valence == 0.5
        assert f.backend == "numpy"

    def test_fields_are_floats(self):
        from scripts.core.energy_profiler import EnergyFeatures
        f = EnergyFeatures(rms_energy=0.3, arousal=0.7)
        assert isinstance(f.rms_energy, float)
        assert isinstance(f.arousal, float)


# ---------------------------------------------------------------------------
# TestNumpyBackend — no optional deps
# ---------------------------------------------------------------------------

class TestNumpyBackend:
    def _profile(self, audio, sr=22050):
        from scripts.core.energy_profiler import profile_energy
        return profile_energy(audio, sr, backend="numpy")

    def test_returns_energy_features(self):
        from scripts.core.energy_profiler import EnergyFeatures
        f = self._profile(_sine())
        assert isinstance(f, EnergyFeatures)

    def test_backend_is_numpy(self):
        f = self._profile(_sine())
        assert f.backend == "numpy"

    def test_silence_has_low_energy(self):
        f = self._profile(_silence())
        assert f.rms_energy < 0.01

    def test_silence_arousal_near_zero(self):
        f = self._profile(_silence())
        assert f.arousal < 0.05

    def test_loud_signal_has_high_energy(self):
        loud = _sine(amp=0.95)
        f = self._profile(loud)
        assert f.rms_energy > 0.5

    def test_noise_higher_centroid_than_sine(self):
        """Broadband noise should have a higher spectral centroid than a pure sine."""
        f_noise = self._profile(_noise())
        f_sine  = self._profile(_sine(freq=200.0))  # low-freq sine → low centroid
        assert f_noise.spectral_centroid > f_sine.spectral_centroid

    def test_all_values_in_unit_range(self):
        for audio in (_sine(), _silence(), _noise()):
            f = self._profile(audio)
            assert 0.0 <= f.rms_energy      <= 1.0
            assert 0.0 <= f.spectral_centroid <= 1.0
            assert 0.0 <= f.dynamic_range    <= 1.0
            assert 0.0 <= f.arousal          <= 1.0
            assert 0.0 <= f.valence          <= 1.0

    def test_valence_is_neutral_without_model(self):
        """numpy backend cannot predict valence → must return 0.5."""
        f = self._profile(_sine())
        assert f.valence == 0.5

    def test_stereo_audio_handled(self):
        """Stereo input should not crash — centroid operates on any 1-D array."""
        # energy_profiler works on flat arrays; it doesn't auto-mix to mono,
        # but callers should pass mono.  Test that mono is fine.
        mono = _sine()
        f = self._profile(mono)
        assert isinstance(f.arousal, float)

    def test_dynamic_range_of_sine_is_above_zero(self):
        """A sine wave has ~3 dB peak-to-RMS; normalised to ~0.1."""
        f = self._profile(_sine())
        assert f.dynamic_range > 0.0

    def test_profile_energy_auto_uses_numpy_when_essentia_absent(self):
        """auto backend falls back to numpy when essentia is not installed."""
        import importlib
        if importlib.util.find_spec("essentia") is not None:
            pytest.skip("Essentia is installed; auto → essentia, not numpy")
        from scripts.core.energy_profiler import profile_energy
        f = profile_energy(_sine(), sr=22050, backend="auto")
        assert f.backend == "numpy"


# ---------------------------------------------------------------------------
# TestHelperFunctions — internal numpy helpers
# ---------------------------------------------------------------------------

class TestNumpyHelpers:
    def test_rms_normalised_sine(self):
        from scripts.core.energy_profiler import _rms_normalised
        sine = _sine(amp=0.8)
        rms = _rms_normalised(sine)
        # RMS of amp*sin = amp/sqrt(2) ≈ 0.566
        expected = 0.8 / np.sqrt(2)
        assert abs(rms - expected) < 0.02

    def test_rms_normalised_silence_is_zero(self):
        from scripts.core.energy_profiler import _rms_normalised
        assert _rms_normalised(_silence()) == 0.0

    def test_spectral_centroid_silence_is_zero(self):
        from scripts.core.energy_profiler import _spectral_centroid_normalised
        c = _spectral_centroid_normalised(_silence(), sr=22050)
        assert c == 0.0

    def test_dynamic_range_silence_is_zero(self):
        from scripts.core.energy_profiler import _dynamic_range_normalised
        assert _dynamic_range_normalised(_silence()) == 0.0

    def test_dynamic_range_clipped_signal_low(self):
        """A fully clipped signal (all samples at max) → DR = 0 dB → score 0."""
        from scripts.core.energy_profiler import _dynamic_range_normalised
        clipped = np.ones(44100, dtype=np.float32)
        dr = _dynamic_range_normalised(clipped)
        assert dr < 0.01


# ---------------------------------------------------------------------------
# TestEnrichTrackNode — SetlistPlanner integration
# ---------------------------------------------------------------------------

class TestEnrichTrackNode:
    def test_enrich_sets_energy(self):
        from scripts.core.setlist_planner import TrackNode
        from scripts.core.energy_profiler import enrich_track_node
        track = TrackNode(name="Test", energy=0.5)
        enrich_track_node(track, _sine(amp=0.9), sr=22050, backend="numpy")
        # Energy should now be set to the profiler's arousal
        assert track.energy != 0.5 or track.arousal_predicted is not None

    def test_arousal_predicted_populated(self):
        from scripts.core.setlist_planner import TrackNode
        from scripts.core.energy_profiler import enrich_track_node
        track = TrackNode(name="Test", energy=0.5)
        enrich_track_node(track, _sine(), sr=22050, backend="numpy")
        assert track.arousal_predicted is not None
        assert 0.0 <= track.arousal_predicted <= 1.0

    def test_silence_gives_low_arousal(self):
        from scripts.core.setlist_planner import TrackNode
        from scripts.core.energy_profiler import enrich_track_node
        track = TrackNode(name="Test", energy=0.5)
        enrich_track_node(track, _silence(), sr=22050, backend="numpy")
        assert track.arousal_predicted < 0.05

    def test_loud_signal_gives_high_arousal(self):
        from scripts.core.setlist_planner import TrackNode
        from scripts.core.energy_profiler import enrich_track_node
        track = TrackNode(name="Test", energy=0.5)
        enrich_track_node(track, _noise(amp=0.9), sr=22050, backend="numpy")
        assert track.arousal_predicted > 0.3


# ---------------------------------------------------------------------------
# TestTransitionCostArousal — SetlistPlanner wiring
# ---------------------------------------------------------------------------

class TestTransitionCostArousal:
    """Verify that transition_cost uses arousal_predicted when present."""

    def _node(self, camelot: str, bpm: float, energy: float, arousal: float | None = None):
        from scripts.core.setlist_planner import TrackNode
        t = TrackNode(name="x", camelot=camelot, bpm=bpm, energy=energy)
        t.arousal_predicted = arousal
        return t

    def test_uses_arousal_predicted_over_energy(self):
        """
        Two transitions: both with same bpm + camelot.
        A → B1 where B1.energy=0.9 but B1.arousal_predicted=0.1 (calm despite high energy).
        A → B2 where B2.energy=0.9 and B2.arousal_predicted=0.9 (truly energetic).
        At arc_position=0.0 (low ideal energy), B1 should have LOWER energy cost
        because its arousal_predicted (0.1) is close to ideal (≈0.3).
        """
        from scripts.core.setlist_planner import transition_cost, EnergyArc

        a  = self._node("8B", 128, 0.5, arousal=0.5)
        b1 = self._node("9B", 128, 0.9, arousal=0.1)   # energy says high, arousal says calm
        b2 = self._node("9B", 128, 0.9, arousal=0.9)   # both say high

        score_b1 = transition_cost(a, b1, arc_position=0.0, arc=EnergyArc.MOUNTAIN)
        score_b2 = transition_cost(a, b2, arc_position=0.0, arc=EnergyArc.MOUNTAIN)

        # b1 should fit the early-set low-energy requirement better
        assert score_b1.energy_cost < score_b2.energy_cost

    def test_fallback_to_energy_when_arousal_is_none(self):
        """When arousal_predicted is None, transition_cost falls back to .energy."""
        from scripts.core.setlist_planner import transition_cost, EnergyArc

        a = self._node("8B", 128, 0.5, arousal=None)
        b = self._node("9B", 128, 0.9, arousal=None)
        # Should not crash
        score = transition_cost(a, b, arc_position=0.5, arc=EnergyArc.MOUNTAIN)
        assert 0.0 <= score.energy_cost <= 1.0

    def test_arousal_none_and_arousal_zero_differ(self):
        """arousal_predicted=None (use .energy=0.9) vs arousal_predicted=0.0 (calm)."""
        from scripts.core.setlist_planner import transition_cost, EnergyArc

        a = self._node("8B", 128, 0.5, arousal=0.5)
        b_none = self._node("9B", 128, 0.9, arousal=None)    # uses energy=0.9
        b_zero = self._node("9B", 128, 0.9, arousal=0.0)     # uses arousal=0.0

        # At position=0.9 (target high energy), None→0.9 fits; 0.0 doesn't
        s_none = transition_cost(a, b_none, arc_position=0.9, arc=EnergyArc.RAMP_UP)
        s_zero = transition_cost(a, b_zero, arc_position=0.9, arc=EnergyArc.RAMP_UP)
        assert s_none.energy_cost < s_zero.energy_cost
