"""
tests/test_stepped_crossfade.py — beat-quantized ("stepped") stem crossfade.

The `stepped` transition effect divides the transition into one step per beat and
moves each song's volume by an equal increment per step, instead of sweeping
smoothly. The increments are equal in POWER, not amplitude: two different songs
are uncorrelated so their powers add, and equal *amplitude* steps would put both
curves at 0.5 mid-transition (0.5² + 0.5² = 0.5, a −3.01 dB dip). That is the
same equal-gain mistake TestEqualPowerCrossfade in test_behavioral.py guards.

Pure numpy — no librosa, so no dj_analysis mark.
"""

import numpy as np
import pytest

from scripts.core.dj_engine import _stem_crossfade_curves, _stepped_crossfade_curves

SR = 44100


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))))) if len(x) else 0.0


class TestSteppedCurve:
    def test_produces_exactly_one_level_per_beat(self):
        """The staircase claim itself. A smooth ramp would show ~n levels."""
        steps = 64                      # 16 bars x 4 beats
        fade_out, fade_in = _stepped_crossfade_curves(SR * 4, steps)

        assert len(np.unique(fade_out)) == steps, (
            f"expected {steps} distinct levels (one per beat), got "
            f"{len(np.unique(fade_out))} — the curve was not quantized"
        )
        assert len(np.unique(fade_in)) == steps

    def test_increments_are_equal_in_power(self):
        """Each beat must move an identical slice of loudness."""
        steps = 64
        fade_out, _ = _stepped_crossfade_curves(SR * 4, steps)

        levels = np.unique(fade_out)[::-1]        # 1.0 → 0.0
        deltas = np.diff(levels.astype(np.float64) ** 2)

        assert float(deltas.max() - deltas.min()) < 1e-5, (
            f"power increments are uneven (spread {deltas.max() - deltas.min():.2e}) "
            "— steps are not equal-power"
        )

    def test_stays_constant_power_across_the_window(self):
        """No mid-transition dip: fade_out² + fade_in² == 1 at every sample."""
        fade_out, fade_in = _stepped_crossfade_curves(SR * 4, 64)
        power = fade_out.astype(np.float64) ** 2 + fade_in.astype(np.float64) ** 2

        assert float(np.max(np.abs(power - 1.0))) < 1e-3, (
            f"combined power spans [{power.min():.4f}, {power.max():.4f}] — "
            "a stepped fade must not reintroduce the equal-gain midpoint dip"
        )

    def test_endpoints_are_an_exact_handoff(self):
        """_seam_blend() splices this window into surrounding audio and assumes
        A is silent and B is full by the end. Dividing by N instead of N-1 would
        leave A at sqrt(1/64) = 0.125 here."""
        fade_out, fade_in = _stepped_crossfade_curves(SR * 4, 64)

        assert fade_out[0] == pytest.approx(1.0, abs=1e-6)
        assert fade_in[0] == pytest.approx(0.0, abs=1e-6)
        assert fade_out[-1] == pytest.approx(0.0, abs=1e-6)
        assert fade_in[-1] == pytest.approx(1.0, abs=1e-6)

    def test_length_is_exactly_the_window(self):
        """Integer division leaves a remainder; it must not shorten the array."""
        for n, steps in ((SR * 4, 64), (10_001, 64), (9_999, 32), (12_345, 128)):
            fade_out, fade_in = _stepped_crossfade_curves(n, steps)
            assert len(fade_out) == n and len(fade_in) == n, f"n={n} steps={steps}"

    def test_levels_are_monotonic(self):
        """A only ever gets quieter, B only ever louder."""
        fade_out, fade_in = _stepped_crossfade_curves(SR * 4, 64)
        assert np.all(np.diff(fade_out) <= 0)
        assert np.all(np.diff(fade_in) >= 0)

    @pytest.mark.parametrize("steps", [0, 1, -5])
    def test_degenerate_step_counts_fall_back(self, steps):
        """Fewer than two plateaus cannot form a staircase — must not divide by
        zero or return a wrong-length array."""
        fade_out, fade_in = _stepped_crossfade_curves(1000, steps)
        assert len(fade_out) == 1000 and len(fade_in) == 1000
        assert np.all(np.isfinite(fade_out)) and np.all(np.isfinite(fade_in))

    def test_more_steps_than_samples_falls_back(self):
        fade_out, fade_in = _stepped_crossfade_curves(10, 64)
        assert len(fade_out) == 10 and len(fade_in) == 10
        assert np.all(np.isfinite(fade_out))

    def test_zero_length_window(self):
        fade_out, fade_in = _stepped_crossfade_curves(0, 64)
        assert len(fade_out) == 0 and len(fade_in) == 0


class TestSteppedDoesNotDisturbTheDefault:
    """steps=None must leave every existing caller on the similarity curves."""

    @pytest.mark.parametrize("sim", [0.9, 0.5, 0.2])
    def test_default_matches_no_steps_argument(self, sim):
        a_out, a_in = _stem_crossfade_curves(10_000, similarity=sim)
        b_out, b_in = _stem_crossfade_curves(10_000, similarity=sim, steps=None)
        assert np.array_equal(a_out, b_out) and np.array_equal(a_in, b_in)

    def test_medium_band_is_still_the_cos_sin_sweep(self):
        """Golden values captured from the pre-change implementation."""
        fade_out, fade_in = _stem_crossfade_curves(10_000, similarity=0.5)
        idx = np.linspace(0, 9_999, 21).astype(int)

        assert [round(float(x), 6) for x in fade_out[idx]][::5] == [
            1.0, 0.923925, 0.707162, 0.38272, -0.0
        ]
        assert [round(float(x), 6) for x in fade_in[idx]][::5] == [
            0.0, 0.382575, 0.707051, 0.923865, 1.0
        ]

    def test_steps_bypasses_the_similarity_bands(self):
        """With steps set, all three bands return the same staircase."""
        curves = [_stem_crossfade_curves(4096, similarity=s, steps=16)[0]
                  for s in (0.9, 0.5, 0.2)]
        assert np.array_equal(curves[0], curves[1])
        assert np.array_equal(curves[1], curves[2])


class TestSteppedAudibleOnOneStem:
    """
    The audible claim, measured on a single stepped stem rather than a summed
    mix. Bass bypasses this curve entirely (_stem_bass_ramp sweeps it
    continuously), so RMS of the full mix could show plateaus for reasons
    unrelated to this code, or hide working ones.
    """

    def test_level_holds_flat_within_each_beat_and_moves_between_them(self):
        steps = 16
        n = SR * 4
        rng = np.random.default_rng(4242)
        stem = rng.standard_normal(n).astype(np.float32) * 0.3

        fade_out, _ = _stepped_crossfade_curves(n, steps)
        faded = stem * fade_out

        # Measure the middle 60% of each plateau, avoiding the boundaries.
        width = n // steps
        plateau_rms = []
        for k in range(steps):
            lo = k * width + int(0.2 * width)
            hi = k * width + int(0.8 * width)
            plateau_rms.append(_rms(faded[lo:hi]) / max(_rms(stem[lo:hi]), 1e-9))

        # Within a beat: the applied gain is one constant, so the ratio to the
        # source is flat. Between beats it must actually move.
        for k in range(steps - 1):
            assert plateau_rms[k] > plateau_rms[k + 1], (
                f"plateau {k} did not step down to {k + 1} — no staircase"
            )

        # And the gain inside a single plateau must be genuinely constant.
        k = steps // 3
        lo, hi = k * width, (k + 1) * width
        gains = fade_out[lo:hi]
        assert float(gains.max() - gains.min()) == 0.0, (
            "gain varies inside a beat — this is a ramp, not a step"
        )
