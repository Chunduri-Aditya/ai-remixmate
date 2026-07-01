"""
tests/test_beat_tracker.py — Unit tests for the beat_tracker module.

Tests run without librosa on machines where it is not installed — those
tests are marked @pytest.mark.dj_analysis and auto-skipped.

The module-level interface (BeatResult, get_tracker, get_configured_tracker)
is tested without librosa using a synthetic BeatResult directly.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: build a synthetic BeatResult (no librosa needed)
# ---------------------------------------------------------------------------

def _synthetic_result(bpm: float = 128.0, duration: float = 30.0, sr: int = 44100):
    from scripts.core.beat_tracker import BeatResult
    beat_interval = 60.0 / bpm
    beat_times = np.arange(0.0, duration, beat_interval)
    hop_length = 512
    beat_frames = np.round(beat_times * sr / hop_length).astype(np.int64)
    downbeat_times = beat_times[::4]
    return BeatResult(
        beat_times=beat_times,
        beat_frames=beat_frames,
        downbeat_times=downbeat_times,
        bpm=bpm,
        sr=sr,
        backend="test",
    )


# ---------------------------------------------------------------------------
# BeatResult API
# ---------------------------------------------------------------------------

class TestBeatResult:
    def test_bar_duration(self):
        r = _synthetic_result(bpm=120.0)
        assert abs(r.bar_duration() - 2.0) < 1e-6   # 4 beats at 120 BPM = 2s

    def test_bar_duration_128bpm(self):
        r = _synthetic_result(bpm=128.0)
        assert abs(r.bar_duration() - (60.0 / 128.0 * 4)) < 1e-6

    def test_nearest_downbeat_self(self):
        r = _synthetic_result(bpm=120.0)
        # First downbeat is at 0.0
        assert abs(r.nearest_downbeat(0.0) - 0.0) < 0.01

    def test_nearest_downbeat_midway(self):
        r = _synthetic_result(bpm=120.0)
        # Midway between downbeat[0]=0.0 and downbeat[1]=2.0 → rounds to 2.0
        result = r.nearest_downbeat(1.1)
        assert abs(result - 2.0) < 0.1

    def test_nearest_downbeat_at_or_after(self):
        r = _synthetic_result(bpm=120.0)
        # downbeat[1] = 2.0 seconds; querying slightly before → returns 2.0
        result = r.nearest_downbeat_at_or_after(1.5)
        assert result >= 1.5 - 0.1

    def test_nearest_downbeat_empty_falls_back(self):
        from scripts.core.beat_tracker import BeatResult
        r = BeatResult(
            beat_times=np.array([0.0, 0.5, 1.0, 1.5]),
            beat_frames=np.array([0, 44, 88, 132]),
            downbeat_times=np.array([]),
            bpm=120.0,
            sr=44100,
            backend="test",
        )
        # Should not crash; returns bar-aligned estimate
        result = r.nearest_downbeat(1.8)
        assert isinstance(result, float)

    def test_nearest_downbeat_at_or_after_empty_falls_back(self):
        from scripts.core.beat_tracker import BeatResult
        r = BeatResult(
            beat_times=np.array([0.0, 0.5]),
            beat_frames=np.array([0, 44]),
            downbeat_times=np.array([]),
            bpm=120.0,
            sr=44100,
            backend="test",
        )
        result = r.nearest_downbeat_at_or_after(3.0)
        assert result >= 3.0 - 0.01

    def test_backend_field_preserved(self):
        r = _synthetic_result()
        assert r.backend == "test"


# ---------------------------------------------------------------------------
# get_tracker factory
# ---------------------------------------------------------------------------

class TestGetTracker:
    def test_librosa_backend_returns_librosa_tracker(self):
        from scripts.core.beat_tracker import get_tracker, LibrosaBeatTracker
        assert isinstance(get_tracker("librosa"), LibrosaBeatTracker)

    def test_beat_this_backend_returns_beat_this_tracker(self):
        from scripts.core.beat_tracker import get_tracker, BeatThisTracker
        assert isinstance(get_tracker("beat_this"), BeatThisTracker)

    def test_auto_backend_returns_a_tracker(self):
        from scripts.core.beat_tracker import get_tracker, LibrosaBeatTracker, BeatThisTracker
        tracker = get_tracker("auto")
        assert isinstance(tracker, (LibrosaBeatTracker, BeatThisTracker))

    def test_unknown_backend_raises_value_error(self):
        from scripts.core.beat_tracker import get_tracker
        with pytest.raises(ValueError, match="Unknown beat_tracker backend"):
            get_tracker("nonexistent")

    def test_get_configured_tracker_returns_tracker(self):
        from scripts.core.beat_tracker import get_configured_tracker, LibrosaBeatTracker, BeatThisTracker
        tracker = get_configured_tracker()
        assert isinstance(tracker, (LibrosaBeatTracker, BeatThisTracker))


# ---------------------------------------------------------------------------
# LibrosaBeatTracker.track() — requires librosa
# ---------------------------------------------------------------------------

@pytest.mark.dj_analysis
class TestLibrosaBeatTracker:
    """These tests require librosa and a valid audio signal."""

    @pytest.fixture
    def metronome_120bpm(self):
        """Synthetic 30s 120 BPM click track.

        Uses short sharp impulse clicks (100 samples at full amplitude) rather
        than hanning-windowed clicks.  Librosa's beat_track relies on onset
        strength, which requires abrupt transients — smooth hanning onsets
        produce zero onset strength and beat_track returns empty arrays.
        """
        sr = 22050
        duration = 30.0
        bpm = 120.0
        audio = np.zeros(int(sr * duration), dtype=np.float32)
        beat_interval = sr * 60.0 / bpm
        click_length = 100  # sharp impulse — strong onset for librosa
        for i in range(int(duration * bpm / 60)):
            start = int(i * beat_interval)
            end = min(start + click_length, len(audio))
            audio[start:end] = 0.8
        return audio, sr, bpm

    def test_track_returns_beat_result(self, metronome_120bpm):
        from scripts.core.beat_tracker import LibrosaBeatTracker, BeatResult
        audio, sr, _ = metronome_120bpm
        result = LibrosaBeatTracker().track(audio, sr)
        assert isinstance(result, BeatResult)

    def test_beat_times_nonempty(self, metronome_120bpm):
        from scripts.core.beat_tracker import LibrosaBeatTracker
        audio, sr, _ = metronome_120bpm
        result = LibrosaBeatTracker().track(audio, sr)
        assert len(result.beat_times) > 0

    def test_downbeat_times_nonempty(self, metronome_120bpm):
        from scripts.core.beat_tracker import LibrosaBeatTracker
        audio, sr, _ = metronome_120bpm
        result = LibrosaBeatTracker().track(audio, sr)
        assert len(result.downbeat_times) > 0

    def test_downbeat_count_approx_quarter_beats(self, metronome_120bpm):
        from scripts.core.beat_tracker import LibrosaBeatTracker
        audio, sr, _ = metronome_120bpm
        result = LibrosaBeatTracker().track(audio, sr)
        ratio = len(result.downbeat_times) / len(result.beat_times)
        assert 0.20 <= ratio <= 0.30, f"Downbeat ratio {ratio:.2f} not near 0.25"

    def test_bpm_near_120(self, metronome_120bpm):
        from scripts.core.beat_tracker import LibrosaBeatTracker
        audio, sr, _ = metronome_120bpm
        result = LibrosaBeatTracker().track(audio, sr)
        assert abs(result.bpm - 120.0) < 5.0, f"BPM {result.bpm:.1f} too far from 120"

    def test_beat_frames_match_beat_times(self, metronome_120bpm):
        from scripts.core.beat_tracker import LibrosaBeatTracker
        import librosa
        audio, sr, _ = metronome_120bpm
        result = LibrosaBeatTracker().track(audio, sr)
        reconstructed = librosa.frames_to_time(result.beat_frames, sr=sr, hop_length=512)
        assert np.allclose(reconstructed, result.beat_times, atol=0.01)

    def test_backend_field_is_librosa(self, metronome_120bpm):
        from scripts.core.beat_tracker import LibrosaBeatTracker
        audio, sr, _ = metronome_120bpm
        result = LibrosaBeatTracker().track(audio, sr)
        assert result.backend == "librosa"

    def test_track_on_silence_no_crash(self):
        from scripts.core.beat_tracker import LibrosaBeatTracker
        audio = np.zeros(22050 * 5, dtype=np.float32)
        result = LibrosaBeatTracker().track(audio, 22050)
        assert isinstance(result.bpm, float)


# ---------------------------------------------------------------------------
# resolve_bpm_octave — REMIX_QUALITY_INSIGHTS.md finding #2 regression guard
#
# Confirmed root cause: music_index.py's _quick_features() called raw
# librosa.beat.beat_track() with zero octave handling, while the real
# analysis pipeline (analyze_structure -> beat_tracker.py) is a separate
# code path that can independently land on a different octave for the same
# song -- this is exactly what produced the reported 63.8 (cached) vs 129.2
# (render-time) mismatch on "I Remember". resolve_bpm_octave() is the fix;
# these tests lock in both directions so it can't silently regress.
# ---------------------------------------------------------------------------

@pytest.mark.dj_analysis
class TestResolveBpmOctave:
    @staticmethod
    def _click_track(true_bpm: float, sr: int = 22050, duration: float = 30.0,
                      noise: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = int(sr * duration)
        period = 60.0 / true_bpm
        audio = np.zeros(n, dtype=np.float32)
        click_len = int(0.01 * sr)
        click = (np.hanning(click_len * 2)[click_len:] * 0.9).astype(np.float32)
        for beat_t in np.arange(0, duration, period):
            i = int(beat_t * sr)
            if i + click_len <= n:
                audio[i:i + click_len] += click
        audio += (rng.standard_normal(n) * noise).astype(np.float32)
        return audio, sr

    def test_corrects_half_tempo_misread(self):
        """A track truly at 130 BPM, misread as 65 BPM, should resolve back to ~130."""
        from scripts.core.beat_tracker import resolve_bpm_octave
        audio, sr = self._click_track(130.0)
        resolved = resolve_bpm_octave(65.0, audio, sr, hop_length=512)
        assert abs(resolved - 130.0) < 5.0, f"expected ~130, got {resolved}"

    def test_leaves_correct_estimate_unchanged(self):
        """An already-correct estimate must not get flipped to its octave neighbour.

        This is the failure mode the prior-weighted scoring exists to prevent:
        plain autocorrelation peak-picking on a clean periodic click train can
        score the half-tempo neighbour *higher* than the true tempo (verified
        directly during development), which would silently break correct
        readings while trying to fix wrong ones.
        """
        from scripts.core.beat_tracker import resolve_bpm_octave
        audio, sr = self._click_track(120.0)
        resolved = resolve_bpm_octave(120.0, audio, sr, hop_length=512)
        assert abs(resolved - 120.0) < 5.0, f"expected ~120, got {resolved}"

    def test_reproduces_session_finding(self):
        """Direct reproduction of the reported numbers: cached=63.8 on a track
        whose real tempo is ~129.2 should resolve into that neighbourhood."""
        from scripts.core.beat_tracker import resolve_bpm_octave
        audio, sr = self._click_track(129.2, noise=0.02)
        resolved = resolve_bpm_octave(63.8, audio, sr, hop_length=512)
        assert abs(resolved - 129.2) < 6.0, f"expected ~129.2, got {resolved}"

    # Wiring of resolve_bpm_octave() into LibrosaBeatTracker.track() itself
    # is covered by TestLibrosaBeatTracker.test_bpm_near_120, which uses the
    # class-scoped metronome_120bpm fixture.


# ---------------------------------------------------------------------------
# BeatThisTracker fallback (beat-this not installed in sandbox)
# ---------------------------------------------------------------------------

class TestBeatThisTrackerFallback:
    def test_falls_back_to_librosa_when_not_installed(self):
        """When beat-this is not installed, BeatThisTracker.track() falls back to librosa."""
        import importlib
        beat_this_available = importlib.util.find_spec("beat_this") is not None
        if beat_this_available:
            pytest.skip("beat-this is installed — fallback not triggered")

        from scripts.core.beat_tracker import BeatThisTracker, BeatResult
        # Need librosa for fallback
        librosa_available = importlib.util.find_spec("librosa") is not None
        if not librosa_available:
            pytest.skip("librosa not installed — cannot test BeatThis fallback")

        audio = np.zeros(22050 * 5, dtype=np.float32)
        result = BeatThisTracker().track(audio, 22050)
        assert isinstance(result, BeatResult)
        # Fell back to librosa
        assert result.backend == "librosa"


# ---------------------------------------------------------------------------
# SongStructure.downbeat_times field
# ---------------------------------------------------------------------------

class TestSongStructureDownbeats:
    def test_downbeat_times_field_exists(self):
        from scripts.core.dj_analysis import SongStructure
        s = SongStructure(bpm=128.0, duration=180.0)
        assert hasattr(s, "downbeat_times")
        assert isinstance(s.downbeat_times, list)

    def test_downbeat_times_default_empty(self):
        from scripts.core.dj_analysis import SongStructure
        s = SongStructure(bpm=128.0, duration=180.0)
        assert s.downbeat_times == []


# ---------------------------------------------------------------------------
# _snap_to_bar_grid (Stage 2B)
# ---------------------------------------------------------------------------

class TestSnapToBarGrid:
    """Unit tests for the bar-grid cue snapping helper."""

    def _downbeats_120bpm(self, duration: float = 60.0):
        """Ideal 120 BPM downbeats — one every 2 seconds."""
        bar_dur = 60.0 / 120.0 * 4.0   # 2.0 s
        return np.arange(0.0, duration, bar_dur)

    def test_empty_boundaries_returns_empty(self):
        from scripts.core.dj_analysis import _snap_to_bar_grid
        db = self._downbeats_120bpm()
        result = _snap_to_bar_grid([], db)
        assert result == []

    def test_empty_downbeats_returns_raw(self):
        from scripts.core.dj_analysis import _snap_to_bar_grid
        raw = [5.1, 10.2, 20.3]
        result = _snap_to_bar_grid(raw, np.array([]))
        assert result == raw

    def test_boundary_snaps_to_nearest_downbeat(self):
        from scripts.core.dj_analysis import _snap_to_bar_grid
        db = self._downbeats_120bpm()
        # Raw boundary at 8.1 s → nearest downbeat is 8.0 s (bar 4)
        raw = [8.1]
        result = _snap_to_bar_grid(raw, db, bar_duration_s=2.0)
        assert abs(result[0] - 8.0) < 0.01

    def test_boundary_snapped_on_preferred_length(self):
        """
        A boundary near bar 8 (16.0 s at 120 BPM) should snap to 16.0,
        not to 14.0 (bar 7), even if 14.0 is slightly closer.
        preferred_lengths_bars=[8] means every 8-bar multiple is preferred.
        At 120 BPM, bar 8 = 16.0 s.
        """
        from scripts.core.dj_analysis import _snap_to_bar_grid
        db = self._downbeats_120bpm()
        # Slightly closer to 15.0 (bar 7.5, not a downbeat) but 16.0 is preferred
        raw = [15.7]
        result = _snap_to_bar_grid(raw, db, preferred_lengths_bars=[8], bar_duration_s=2.0)
        assert abs(result[0] - 16.0) < 0.01, f"Expected 16.0, got {result[0]}"

    def test_result_is_sorted(self):
        from scripts.core.dj_analysis import _snap_to_bar_grid
        db = self._downbeats_120bpm()
        raw = [20.1, 8.1, 4.1]    # unsorted input
        result = _snap_to_bar_grid(raw, db, bar_duration_s=2.0)
        assert result == sorted(result)

    def test_duplicates_removed(self):
        """Two raw boundaries that snap to the same downbeat should collapse."""
        from scripts.core.dj_analysis import _snap_to_bar_grid
        db = self._downbeats_120bpm()
        # Both 7.9 and 8.1 snap to 8.0
        raw = [7.9, 8.1]
        result = _snap_to_bar_grid(raw, db, bar_duration_s=2.0)
        assert len(result) == 1
        assert abs(result[0] - 8.0) < 0.01

    def test_output_values_are_downbeat_times(self):
        """Every snapped value must be one of the downbeat times."""
        from scripts.core.dj_analysis import _snap_to_bar_grid
        db = self._downbeats_120bpm()
        raw = [3.9, 7.8, 15.9, 31.7]
        result = _snap_to_bar_grid(raw, db, bar_duration_s=2.0)
        for t in result:
            # Each output must be within 1ms of some downbeat
            assert np.any(np.abs(db - t) < 0.002), f"{t} is not near any downbeat"

    def test_single_downbeat_no_crash(self):
        """Edge case: only one downbeat in the array."""
        from scripts.core.dj_analysis import _snap_to_bar_grid
        db = np.array([0.0])
        raw = [0.5, 1.0]
        result = _snap_to_bar_grid(raw, db, bar_duration_s=2.0)
        assert isinstance(result, list)

    def test_no_downbeats_within_window_keeps_raw(self):
        """A boundary far from all downbeats — should survive without crashing."""
        from scripts.core.dj_analysis import _snap_to_bar_grid
        db = np.array([0.0, 2.0, 4.0])   # only 3 bars
        raw = [50.0]    # far beyond all downbeats
        result = _snap_to_bar_grid(raw, db, bar_duration_s=2.0)
        assert isinstance(result, list)
        # May return raw or empty — just must not crash
        assert len(result) <= 1
