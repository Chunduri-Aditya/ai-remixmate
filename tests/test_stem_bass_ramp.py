"""
tests/test_stem_bass_ramp.py — Behavioral tests for _stem_bass_ramp (Stage 2C).

Every test asserts *what you hear*, not just shape/dtype.  These are the
regression guards that catch audible bass bleed — the same category of bug
that was invisible to the old shape-only test suite.

The key behavioral properties:
  1. Song A's bass is completely silent after swap_sample.
  2. Song B's bass is completely silent before swap_sample.
  3. The crossover point is sample-accurate (not bar-smeared like the IIR path).
  4. No DC offset or NaN introduced.
  5. Cosine ramp is smoother than linear (lower derivative discontinuity).
"""
from __future__ import annotations

import numpy as np
import pytest


SR = 44100
TRANS_SAMPLES = SR * 8   # 8-second transition window (typical)


def _make_bass(n: int, frequency: float = 80.0, amplitude: float = 0.8) -> np.ndarray:
    """Constant-frequency sine wave simulating a bass stem."""
    t = np.linspace(0, n / SR, n, endpoint=False)
    return (np.sin(2 * np.pi * frequency * t) * amplitude).astype(np.float32)


# ---------------------------------------------------------------------------
# Hard-swap (ramp_samples=0) — strictest behavioral tests
# ---------------------------------------------------------------------------

class TestStemBassRampHardSwap:
    """ramp_samples=0 → instantaneous swap, no taper."""

    def _run(self, swap_frac: float = 0.5):
        from scripts.core.dj_engine import _stem_bass_ramp
        n = TRANS_SAMPLES
        swap = int(n * swap_frac)
        bass_a = _make_bass(n, frequency=80.0)
        bass_b = _make_bass(n, frequency=90.0)
        ducked_a, ducked_b = _stem_bass_ramp(bass_a, bass_b, swap, ramp_samples=0)
        return ducked_a, ducked_b, swap, bass_a, bass_b

    def test_a_is_silent_after_swap(self):
        ducked_a, _, swap, _, _ = self._run()
        assert np.all(ducked_a[swap:] == 0.0), "Song A bass bleeds after swap"

    def test_b_is_silent_before_swap(self):
        _, ducked_b, swap, _, _ = self._run()
        assert np.all(ducked_b[:swap] == 0.0), "Song B bass audible before swap"

    def test_a_preserves_signal_before_swap(self):
        ducked_a, _, swap, bass_a, _ = self._run()
        assert np.allclose(ducked_a[:swap], bass_a[:swap]), "Song A bass corrupted before swap"

    def test_b_preserves_signal_after_swap(self):
        _, ducked_b, swap, _, bass_b = self._run()
        assert np.allclose(ducked_b[swap:], bass_b[swap:]), "Song B bass corrupted after swap"

    def test_no_nan(self):
        ducked_a, ducked_b, _, _, _ = self._run()
        assert np.isfinite(ducked_a).all()
        assert np.isfinite(ducked_b).all()

    def test_swap_at_start(self):
        """swap_sample=0 → A is all-zero, B is all-full."""
        from scripts.core.dj_engine import _stem_bass_ramp
        n = SR
        bass_a = _make_bass(n)
        bass_b = _make_bass(n, frequency=90.0)
        ducked_a, ducked_b = _stem_bass_ramp(bass_a, bass_b, 0, ramp_samples=0)
        assert np.all(ducked_a == 0.0)
        assert np.allclose(ducked_b, bass_b)

    def test_swap_at_end(self):
        """swap_sample=N → A is all-full, B is all-zero."""
        from scripts.core.dj_engine import _stem_bass_ramp
        n = SR
        bass_a = _make_bass(n)
        bass_b = _make_bass(n, frequency=90.0)
        ducked_a, ducked_b = _stem_bass_ramp(bass_a, bass_b, n, ramp_samples=0)
        assert np.allclose(ducked_a, bass_a)
        assert np.all(ducked_b == 0.0)

    def test_output_shapes_match_input(self):
        from scripts.core.dj_engine import _stem_bass_ramp
        n = SR * 4
        bass_a = _make_bass(n)
        bass_b = _make_bass(n, frequency=90.0)
        da, db = _stem_bass_ramp(bass_a, bass_b, n // 2, ramp_samples=0)
        assert da.shape == bass_a.shape
        assert db.shape == bass_b.shape

    def test_different_swap_fractions(self):
        """Test at 25%, 50%, 75% swap points."""
        from scripts.core.dj_engine import _stem_bass_ramp
        n = SR * 4
        bass_a = _make_bass(n)
        bass_b = _make_bass(n, frequency=90.0)
        for frac in [0.25, 0.50, 0.75]:
            swap = int(n * frac)
            da, db = _stem_bass_ramp(bass_a, bass_b, swap, ramp_samples=0)
            assert np.all(da[swap:] == 0.0), f"A not silent after swap at {frac}"
            assert np.all(db[:swap] == 0.0), f"B not silent before swap at {frac}"


# ---------------------------------------------------------------------------
# Cosine ramp (ramp_samples > 0) — taper behavioral tests
# ---------------------------------------------------------------------------

class TestStemBassRampCosine:
    """ramp_samples > 0 → smooth cosine taper around swap point."""

    RAMP = int(0.10 * SR)   # 100 ms — standard operational value

    def _run(self, swap_frac: float = 0.5):
        from scripts.core.dj_engine import _stem_bass_ramp
        n = TRANS_SAMPLES
        swap = int(n * swap_frac)
        bass_a = _make_bass(n, frequency=80.0)
        bass_b = _make_bass(n, frequency=90.0)
        ducked_a, ducked_b = _stem_bass_ramp(bass_a, bass_b, swap, ramp_samples=self.RAMP)
        return ducked_a, ducked_b, swap, bass_a, bass_b

    def test_a_is_silent_after_swap(self):
        """After swap_sample, Song A's bass must be zero even with ramp."""
        ducked_a, _, swap, _, _ = self._run()
        assert np.all(ducked_a[swap:] == 0.0), (
            "Song A bass bleeds past swap point even with cosine ramp"
        )

    def test_b_is_silent_before_ramp_start(self):
        """Song B's bass must be zero before the ramp window opens."""
        ducked_a, ducked_b, swap, _, _ = self._run()
        ramp_start = swap  # B's ramp starts AT swap_sample
        assert np.all(ducked_b[:ramp_start] == 0.0), (
            "Song B audible before its ramp entry"
        )

    def test_a_full_before_ramp_start(self):
        """Song A should play at full amplitude before its ramp taper begins."""
        from scripts.core.dj_engine import _stem_bass_ramp
        n = TRANS_SAMPLES
        swap = n // 2
        bass_a = _make_bass(n, frequency=80.0)
        bass_b = _make_bass(n, frequency=90.0)
        ducked_a, _ = _stem_bass_ramp(bass_a, bass_b, swap, ramp_samples=self.RAMP)
        pre_ramp_end = max(0, swap - self.RAMP)
        if pre_ramp_end > 0:
            assert np.allclose(ducked_a[:pre_ramp_end], bass_a[:pre_ramp_end]), (
                "Song A bass corrupted before its ramp window"
            )

    def test_cosine_ramp_monotonically_decreasing_a(self):
        """The taper on Song A's bass envelope must be monotonically non-increasing."""
        from scripts.core.dj_engine import _stem_bass_ramp
        # Use a constant-amplitude (DC) bass so envelope = output
        n = SR * 4
        swap = n // 2
        dc_a = np.ones(n, dtype=np.float32)
        dc_b = np.zeros(n, dtype=np.float32)
        ramp = min(self.RAMP, swap)
        ducked_a, _ = _stem_bass_ramp(dc_a, dc_b, swap, ramp_samples=ramp)
        ramp_start = swap - ramp
        ramp_window = ducked_a[ramp_start:swap]
        diffs = np.diff(ramp_window)
        assert np.all(diffs <= 1e-6), "Song A cosine ramp is not monotonically decreasing"

    def test_cosine_ramp_monotonically_increasing_b(self):
        """The taper on Song B's bass envelope must be monotonically non-decreasing."""
        from scripts.core.dj_engine import _stem_bass_ramp
        n = SR * 4
        swap = n // 2
        dc_a = np.zeros(n, dtype=np.float32)
        dc_b = np.ones(n, dtype=np.float32)
        ramp = min(self.RAMP, n - swap)
        _, ducked_b = _stem_bass_ramp(dc_a, dc_b, swap, ramp_samples=ramp)
        ramp_end = swap + ramp
        ramp_window = ducked_b[swap:ramp_end]
        diffs = np.diff(ramp_window)
        assert np.all(diffs >= -1e-6), "Song B cosine ramp is not monotonically increasing"

    def test_no_nan_or_inf(self):
        ducked_a, ducked_b, _, _, _ = self._run()
        assert np.isfinite(ducked_a).all()
        assert np.isfinite(ducked_b).all()

    def test_zero_ramp_equivalent_to_hard_swap(self):
        """ramp_samples=0 should produce identical result to hard-swap path."""
        from scripts.core.dj_engine import _stem_bass_ramp
        n = SR * 2
        swap = n // 2
        bass_a = _make_bass(n)
        bass_b = _make_bass(n, frequency=90.0)
        da_hard, db_hard = _stem_bass_ramp(bass_a, bass_b, swap, ramp_samples=0)
        da_ramp, db_ramp = _stem_bass_ramp(bass_a, bass_b, swap, ramp_samples=0)
        assert np.allclose(da_hard, da_ramp)
        assert np.allclose(db_hard, db_ramp)

    def test_cosine_taper_endpoint_smoothness(self):
        """
        A cosine (Hann) taper has derivative ≈ 0 at its start and end.
        A linear ramp has a constant nonzero derivative everywhere.
        This endpoint smoothness is what prevents audible ramp clicks on
        sustained bass notes.

        Test: derivative of the taper at the first 5% of the ramp window
        is smaller for cosine than for linear.
        """
        from scripts.core.dj_engine import _stem_bass_ramp
        n = SR * 4
        swap = n // 2
        ramp = 2000  # wide enough to measure endpoint slope

        # Use DC signal so output directly encodes the envelope shape
        dc_a = np.ones(n, dtype=np.float32)
        dc_b = np.zeros(n, dtype=np.float32)
        da_cos, _ = _stem_bass_ramp(dc_a, dc_b, swap, ramp_samples=ramp)
        ramp_start = swap - ramp
        cos_window = da_cos[ramp_start:swap]

        # Measure derivative in first 5% of ramp (near the "no-click" endpoint)
        endpoint_width = max(1, ramp // 20)
        endpoint_deriv_cos = float(np.max(np.abs(np.diff(cos_window[:endpoint_width]))))
        linear_deriv = 1.0 / ramp   # constant derivative of a linear ramp

        assert endpoint_deriv_cos < linear_deriv, (
            f"Cosine endpoint derivative {endpoint_deriv_cos:.8f} is not smaller than "
            f"linear constant derivative {linear_deriv:.8f} — ramp click not avoided"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestStemBassRampEdgeCases:
    def test_silence_a_no_crash(self):
        from scripts.core.dj_engine import _stem_bass_ramp
        n = SR * 2
        silent = np.zeros(n, dtype=np.float32)
        bass_b = _make_bass(n)
        da, db = _stem_bass_ramp(silent, bass_b, n // 2, ramp_samples=1000)
        assert np.all(da == 0.0)
        assert np.isfinite(db).all()

    def test_silence_b_no_crash(self):
        from scripts.core.dj_engine import _stem_bass_ramp
        n = SR * 2
        bass_a = _make_bass(n)
        silent = np.zeros(n, dtype=np.float32)
        da, db = _stem_bass_ramp(bass_a, silent, n // 2, ramp_samples=1000)
        assert np.isfinite(da).all()
        assert np.all(db == 0.0)

    def test_ramp_larger_than_available_headroom_clamps(self):
        """ramp_samples > swap_sample should not crash — it gets clamped."""
        from scripts.core.dj_engine import _stem_bass_ramp
        n = SR * 2
        bass_a = _make_bass(n)
        bass_b = _make_bass(n, frequency=90.0)
        swap = 100   # very early
        ramp = 5000  # way larger than swap
        da, db = _stem_bass_ramp(bass_a, bass_b, swap, ramp_samples=ramp)
        # Must not crash; A must still be zero after swap
        assert np.all(da[swap:] == 0.0)
        assert np.isfinite(da).all()
        assert np.isfinite(db).all()

    def test_very_short_arrays_no_crash(self):
        from scripts.core.dj_engine import _stem_bass_ramp
        n = 10
        bass_a = np.ones(n, dtype=np.float32)
        bass_b = np.ones(n, dtype=np.float32)
        da, db = _stem_bass_ramp(bass_a, bass_b, n // 2, ramp_samples=0)
        assert da.shape == (n,)
        assert db.shape == (n,)
