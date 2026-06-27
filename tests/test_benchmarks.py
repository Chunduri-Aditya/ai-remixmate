"""
tests/test_benchmarks.py — Smoke tests for the GiantSteps benchmark script.

Runs the benchmark in synthetic mode (no real dataset required) and validates
the structure and range of the returned metrics dict.
"""

from __future__ import annotations

import pytest


@pytest.mark.dj_analysis
class TestGiantStepsBenchmark:
    """
    Smoke tests for scripts/benchmarks/giantsteps_eval.py.

    Marked dj_analysis because run_synthetic() calls detect_key() internally,
    which uses librosa for chroma extraction.
    """

    def test_synthetic_benchmark_runs(self):
        """
        run_synthetic() must return a dict keyed by all four profiles,
        each containing a 'weighted' score in [0, 1].
        """
        from scripts.benchmarks.giantsteps_eval import run_synthetic, PROFILES

        results = run_synthetic(sr=22050, verbose=False)

        assert isinstance(results, dict), (
            f"run_synthetic() should return dict, got {type(results).__name__}"
        )
        assert set(results.keys()) == set(PROFILES), (
            f"Expected profiles {sorted(PROFILES)}, got {sorted(results.keys())}"
        )
        for profile in PROFILES:
            assert 'weighted' in results[profile], (
                f"Profile {profile!r} missing 'weighted' key in results"
            )
            w = results[profile]['weighted']
            assert isinstance(w, float), (
                f"Profile {profile!r}: 'weighted' should be float, got {type(w).__name__}"
            )
            assert 0.0 <= w <= 1.0, (
                f"Profile {profile!r}: weighted score {w:.4f} outside [0.0, 1.0]"
            )

    def test_synthetic_benchmark_all_metric_keys_present(self):
        """
        Each profile entry must contain the full set of MIREX category keys.
        """
        from scripts.benchmarks.giantsteps_eval import run_synthetic

        results = run_synthetic(sr=22050, verbose=False)

        required_keys = {'correct', 'fifth', 'relative', 'parallel', 'other',
                         'weighted', 'n_tracks'}
        for profile, data in results.items():
            missing = required_keys - set(data.keys())
            assert not missing, (
                f"Profile {profile!r} missing keys: {missing}"
            )
            assert data['n_tracks'] == 20, (
                f"Profile {profile!r}: expected n_tracks=20, got {data['n_tracks']}"
            )

    def test_mirex_score_correct_cases(self):
        """
        MIREX scoring function must return correct values for well-known cases.
        No librosa needed — pure arithmetic.
        """
        from scripts.benchmarks.giantsteps_eval import mirex_score

        # Exact match
        assert mirex_score('C', 'major', 'C', 'major') == 1.0
        assert mirex_score('F#', 'minor', 'F#', 'minor') == 1.0

        # Perfect fifth (G is dominant of C)
        assert mirex_score('G', 'major', 'C', 'major') == 0.5
        # Subdominant (F is 5 semitones above C)
        assert mirex_score('F', 'major', 'C', 'major') == 0.5

        # Relative minor of C major is A minor (9 semitones above C)
        assert mirex_score('A', 'minor', 'C', 'major') == 0.3
        # Relative major of A minor is C major (3 semitones above A)
        assert mirex_score('C', 'major', 'A', 'minor') == 0.3

        # Parallel mode (same root, different mode)
        assert mirex_score('C', 'minor', 'C', 'major') == 0.2
        assert mirex_score('D', 'major', 'D', 'minor') == 0.2

        # Unrelated keys
        assert mirex_score('F#', 'major', 'C', 'major') == 0.0
