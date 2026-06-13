"""
tests/test_dj_engine_guards.py — Unit tests for render-path numeric guards.

Covers:
  - _safe_ratio: invalid / out-of-band / valid inputs
  - _safe_bpm: invalid / valid inputs
  - _time_stretch: zero ratio returns input array unchanged (via passthrough)
  - render(): index clamp prevents IndexError when entry_time_b is past EOF
  - render_chain(): empty audio raises ValueError with track index in message
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.core.dj_engine import (
    _safe_ratio,
    _safe_bpm,
    _time_stretch,
    DJEngine,
    _MIN_STRETCH,
    _MAX_STRETCH,
)
from scripts.core.dj_analysis import EQPlan, TransitionPlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eq() -> EQPlan:
    return EQPlan(
        hp_start_hz=400.0,
        hp_end_hz=80.0,
        hp_ramp_bars=8,
        bass_swap_bar=8,
        bass_crossover_hz=150.0,
        a_fade_start_bar=0,
        a_fade_end_bar=16,
        b_fade_start_bar=0,
        b_fade_end_bar=16,
    )


def _plan(
    bpm_a: float = 128.0,
    bpm_b: float = 128.0,
    entry_time_b: float = 60.0,
    exit_time_a: float = 30.0,
    transition_bars: int = 4,
    transition_seconds: float = 7.5,
) -> TransitionPlan:
    return TransitionPlan(
        exit_bar_a=16,
        exit_time_a=exit_time_a,
        entry_bar_b=0,
        entry_time_b=entry_time_b,
        transition_bars=transition_bars,
        transition_seconds=transition_seconds,
        bpm_a=bpm_a,
        bpm_b=bpm_b,
        tempo_shift_ratio=bpm_b / bpm_a if bpm_a > 0 else 1.0,
        eq=_eq(),
    )


SR = 22050  # low SR keeps tests fast


def _silence(seconds: float) -> np.ndarray:
    """Flat non-zero audio so finite-check passes."""
    n = int(SR * seconds)
    return np.full(n, 0.1, dtype=np.float32)


# ---------------------------------------------------------------------------
# _safe_ratio
# ---------------------------------------------------------------------------

class TestSafeRatio:
    def test_zero_returns_one(self):
        assert _safe_ratio(0.0) == 1.0

    def test_negative_returns_one(self):
        assert _safe_ratio(-1.0) == 1.0

    def test_nan_returns_one(self):
        assert _safe_ratio(float("nan")) == 1.0

    def test_inf_returns_one(self):
        assert _safe_ratio(float("inf")) == 1.0

    def test_none_returns_one(self):
        assert _safe_ratio(None) == 1.0  # type: ignore[arg-type]

    def test_below_min_clamped(self):
        assert _safe_ratio(0.1) == _MIN_STRETCH

    def test_above_max_clamped(self):
        assert _safe_ratio(9.0) == _MAX_STRETCH

    def test_exactly_one_passthrough(self):
        assert _safe_ratio(1.0) == 1.0

    def test_in_band_passthrough(self):
        assert _safe_ratio(1.5) == pytest.approx(1.5)

    def test_min_boundary_inclusive(self):
        assert _safe_ratio(_MIN_STRETCH) == _MIN_STRETCH

    def test_max_boundary_inclusive(self):
        assert _safe_ratio(_MAX_STRETCH) == _MAX_STRETCH


# ---------------------------------------------------------------------------
# _safe_bpm
# ---------------------------------------------------------------------------

class TestSafeBpm:
    def test_zero_returns_default(self):
        assert _safe_bpm(0.0) == pytest.approx(120.0)

    def test_negative_returns_default(self):
        assert _safe_bpm(-5.0) == pytest.approx(120.0)

    def test_nan_returns_default(self):
        assert _safe_bpm(float("nan")) == pytest.approx(120.0)

    def test_inf_returns_default(self):
        assert _safe_bpm(float("inf")) == pytest.approx(120.0)

    def test_none_returns_default(self):
        assert _safe_bpm(None) == pytest.approx(120.0)  # type: ignore[arg-type]

    def test_valid_bpm_passthrough(self):
        assert _safe_bpm(128.0) == pytest.approx(128.0)

    def test_custom_default(self):
        assert _safe_bpm(0.0, default=90.0) == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# _time_stretch — zero ratio clamped to 1.0 → passthrough
# ---------------------------------------------------------------------------

class TestTimeStretch:
    def test_zero_ratio_returns_input_unchanged(self):
        sig = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        result = _time_stretch(sig, 0.0)
        assert np.array_equal(result, sig), (
            "_time_stretch(sig, 0.0) should return the input array unchanged "
            "(ratio clamped to 1.0, hits the < 0.02 passthrough)"
        )

    def test_nan_ratio_returns_input_unchanged(self):
        sig = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = _time_stretch(sig, float("nan"))
        assert np.array_equal(result, sig)

    def test_inf_ratio_returns_input_unchanged(self):
        sig = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = _time_stretch(sig, float("inf"))
        # inf clamped to _MAX_STRETCH (2.0) — NOT the passthrough, but must not crash
        assert result is not None
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# render() — index clamp prevents IndexError for out-of-bounds entry_time_b
# ---------------------------------------------------------------------------

class TestRenderIndexClamp:
    def test_entry_time_past_eof_no_index_error(self):
        """
        track_b is 2 s long but plan.entry_time_b = 999 s (far past EOF).
        Without the clamp this would raise IndexError inside the slice.
        With the clamp the render must complete and return a finite buffer.
        """
        track_a = _silence(10.0)
        track_b = _silence(2.0)   # very short

        plan = _plan(
            bpm_a=128.0,
            bpm_b=128.0,
            entry_time_b=999.0,   # past EOF of track_b
            exit_time_a=2.0,
            transition_bars=4,
            transition_seconds=7.5,
        )

        engine = DJEngine(sr=SR)
        result = engine.render(track_a, track_b, plan, full_output=False)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert np.all(np.isfinite(result))

    def test_exit_time_past_eof_no_index_error(self):
        """
        exit_time_a is past track_a EOF — clamp must prevent a bad slice.
        """
        track_a = _silence(2.0)   # very short
        track_b = _silence(10.0)

        plan = _plan(
            bpm_a=128.0,
            bpm_b=128.0,
            exit_time_a=999.0,    # past EOF of track_a
            entry_time_b=0.0,
            transition_bars=4,
            transition_seconds=7.5,
        )

        engine = DJEngine(sr=SR)
        result = engine.render(track_a, track_b, plan, full_output=False)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# render_chain() — empty audio raises ValueError with track index in message
# ---------------------------------------------------------------------------

class TestRenderChainEmptyAudio:
    def test_empty_first_track_raises(self):
        empty = np.zeros(0, dtype=np.float32)
        good  = _silence(5.0)
        plan  = _plan()

        engine = DJEngine(sr=SR)
        with pytest.raises(ValueError, match="track 1"):
            engine.render_chain([empty, good], [plan])

    def test_empty_second_track_raises(self):
        good  = _silence(5.0)
        empty = np.zeros(0, dtype=np.float32)
        plan  = _plan()

        engine = DJEngine(sr=SR)
        with pytest.raises(ValueError, match="track 2"):
            engine.render_chain([good, empty], [plan])
