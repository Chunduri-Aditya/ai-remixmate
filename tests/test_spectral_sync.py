"""
tests/test_spectral_sync.py — dynamic in-transition spectral syncing.

All tests are librosa-backed (STFT/mel filterbank) → marked dj_analysis,
auto-skipping via the conftest probe where librosa is broken.
"""

import numpy as np
import pytest

SR = 44100


def _tone(hz: float, secs: float, amp: float = 0.4) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _noise(secs: float, seed: int = 0, amp: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(SR * secs))).astype(np.float32)


def _make_transition_mix(pre_s=10.0, win_s=10.0, post_s=10.0):
    """A-material (bright noise) → abrupt overlap → B-material (low tone).

    The overlap window deliberately jumps between the two spectra so the
    sync stage has something to smooth.
    """
    pre = _noise(pre_s, seed=1)                      # "A": broadband
    post = _tone(330.0, post_s)                      # "B": narrowband tone
    half = int(win_s * SR // 2)
    window = np.concatenate([_noise(win_s, seed=2)[:half], _tone(330.0, win_s)[:half]])
    return np.concatenate([pre, window, post]), pre_s, win_s


@pytest.mark.dj_analysis
class TestSyncTransitionSpectrum:
    def test_output_shape_and_outside_untouched(self):
        from scripts.core.spectral_sync import sync_transition_spectrum
        mix, start, dur = _make_transition_mix()
        out, info = sync_transition_spectrum(mix, SR, start, dur)
        assert info is not None
        assert len(out) == len(mix)
        w0, w1 = int(start * SR), int((start + dur) * SR)
        assert np.array_equal(out[:w0], mix[:w0])
        assert np.array_equal(out[w1:], mix[w1:])
        assert np.all(np.isfinite(out))

    def test_sync_shrinks_the_mid_window_spectral_jump(self):
        """The synthetic window switches abruptly from A-noise to B-tone at
        its midpoint. Syncing must shrink that spectral discontinuity — the
        profiles on either side of the switch end up closer together."""
        import librosa

        from scripts.core.spectral_sync import sync_transition_spectrum
        mix, start, dur = _make_transition_mix()
        out, info = sync_transition_spectrum(mix, SR, start, dur, max_db=6.0)
        assert info is not None

        def prof(seg):
            m = librosa.feature.melspectrogram(y=seg.astype(np.float32), sr=SR, n_mels=24)
            p = m.mean(axis=1)
            return p / max(p.sum(), 1e-12)

        w0 = int(start * SR)
        half = int(dur * SR // 2)
        q = int(dur * SR // 4)
        # quarter on each side of the abrupt switch
        jump_before = np.abs(
            prof(mix[w0 + q: w0 + half]) - prof(mix[w0 + half: w0 + half + q])
        ).sum()
        jump_after = np.abs(
            prof(out[w0 + q: w0 + half]) - prof(out[w0 + half: w0 + half + q])
        ).sum()
        assert jump_after < jump_before, (
            f"sync did not shrink the mid-window spectral jump "
            f"({jump_before:.4f} → {jump_after:.4f})"
        )

    def test_gain_cap_respected(self):
        from scripts.core.spectral_sync import sync_transition_spectrum
        mix, start, dur = _make_transition_mix()
        _, info = sync_transition_spectrum(mix, SR, start, dur, max_db=4.0)
        assert info["max_gain_db"] <= 4.0 + 0.01
        assert info["min_gain_db"] >= -4.0 - 0.01

    def test_window_out_of_range_is_noop(self):
        from scripts.core.spectral_sync import sync_transition_spectrum
        mix, _, _ = _make_transition_mix()
        out, info = sync_transition_spectrum(mix, SR, len(mix) / SR - 1.0, 10.0)
        assert info is None
        assert np.array_equal(out, mix)

    def test_tiny_window_is_noop(self):
        from scripts.core.spectral_sync import sync_transition_spectrum
        mix, start, _ = _make_transition_mix()
        out, info = sync_transition_spectrum(mix, SR, start, 0.05)
        assert info is None
        assert np.array_equal(out, mix)


@pytest.mark.dj_analysis
class TestApplySpectralSync:
    class _Plan:
        exit_time_a = 10.0
        transition_seconds = 10.0

    def test_config_gate_off_returns_unchanged(self, monkeypatch):
        from scripts.core.config import cfg
        from scripts.core.spectral_sync import apply_spectral_sync
        monkeypatch.setattr(cfg.dj, "spectral_sync", False)
        mix, _, _ = _make_transition_mix()
        out, info = apply_spectral_sync(mix, SR, self._Plan())
        assert info is None
        assert np.array_equal(out, mix)

    def test_applies_on_rendered_layout(self):
        from scripts.core.spectral_sync import apply_spectral_sync
        mix, _, _ = _make_transition_mix()
        out, info = apply_spectral_sync(mix, SR, self._Plan())
        assert info is not None
        assert info["frames"] > 0
        assert len(out) == len(mix)


@pytest.mark.dj_analysis
class TestTrajectoryMatching:
    def test_trajectory_prefers_evolving_match(self):
        """Trajectory scoring must reward a window that *evolves* like A's
        outro, not one that merely averages to the same spectrum."""
        from scripts.core.spectral_matching import mel_trajectory, trajectory_similarity

        # A's outro: tone rising 300→600 Hz over 8 s (chirp)
        t = np.linspace(0, 8.0, int(SR * 8.0), endpoint=False)
        chirp = (0.4 * np.sin(2 * np.pi * (300 + 37.5 * t) * t)).astype(np.float32)
        # Candidate 1: same chirp (matching evolution)
        cand_same = chirp.copy()
        # Candidate 2: same *average* spectrum but reversed evolution
        cand_rev = chirp[::-1].copy()

        traj_a = mel_trajectory(chirp, SR)
        sim_same = trajectory_similarity(traj_a, mel_trajectory(cand_same, SR)).mean()
        sim_rev = trajectory_similarity(traj_a, mel_trajectory(cand_rev, SR)).mean()
        assert sim_same > sim_rev, (
            f"trajectory match should beat reversed evolution ({sim_same:.3f} vs {sim_rev:.3f})"
        )

    def test_sims_vector_shape_feeds_learning_loop(self):
        from scripts.core.spectral_matching import (
            N_BANDS,
            SpectralMatchModel,
            mel_trajectory,
            trajectory_similarity,
        )
        a = _noise(4.0, seed=3)
        b = _noise(4.0, seed=4)
        sims = trajectory_similarity(mel_trajectory(a, SR), mel_trajectory(b, SR))
        assert sims.shape == (N_BANDS,)
        m = SpectralMatchModel()
        m.update(sims, liked=True)
        assert m.n_updates == 1


@pytest.mark.dj_analysis
class TestTransientPreservation:
    def test_transient_frames_keep_their_energy(self):
        """A loud click inside the window is a transient — sync must leave its
        peak nearly untouched even while shaping the surrounding audio."""
        from scripts.core.spectral_sync import sync_transition_spectrum

        mix, start, dur = _make_transition_mix()
        w0 = int(start * SR)
        click_at = w0 + int(dur * SR * 0.3)
        mix2 = mix.copy()
        mix2[click_at: click_at + 256] += 0.9   # sharp broadband transient

        out, info = sync_transition_spectrum(mix2, SR, start, dur, max_db=6.0)
        assert info is not None
        assert info["transient_frames"] > 0

        peak_before = float(np.abs(mix2[click_at - 512: click_at + 768]).max())
        peak_after = float(np.abs(out[click_at - 512: click_at + 768]).max())
        # within 1.5 dB of the original click peak
        assert peak_after >= peak_before * 10 ** (-1.5 / 20.0)


class TestTransitionTimeline:
    class _EQ:
        bass_swap_bar = 8

    class _Plan:
        exit_time_a = 149.2
        transition_seconds = 30.5
        transition_bars = 16
        entry_time_b = 14.9
        tempo_shift_ratio = 1.025

    def test_timeline_fields(self):
        from scripts.api.task_modules.remix import _transition_timeline

        plan = self._Plan()
        plan.eq = self._EQ()

        class _Struct:
            drop_position = 45.0
        tl = _transition_timeline(plan, _Struct())
        assert tl["overlap_start_s"] == 149.2
        assert tl["overlap_end_s"] == pytest.approx(179.7)
        # bass swap at bar 8 of 16 → midpoint of the overlap
        assert tl["bass_swap_s"] == pytest.approx(149.2 + 30.5 / 2, abs=0.01)
        # B-local 45.0 → output: 149.2 + (45.0 − 14.9)/1.025
        assert tl["drop_b_s"] == pytest.approx(149.2 + (45.0 - 14.9) / 1.025, abs=0.01)

    def test_timeline_without_drop(self):
        from scripts.api.task_modules.remix import _transition_timeline

        plan = self._Plan()
        plan.eq = self._EQ()

        class _Struct:
            drop_position = None
        tl = _transition_timeline(plan, _Struct())
        assert tl["drop_b_s"] is None


class TestConfidenceGatedStemBlend:
    """
    DJ_ENGINE_IMPROVEMENT_PLAN.md §2.1 — _should_use_stem_blend() gates the
    full stem-aware render path on compatibility, not just on whether stem
    files happen to exist on disk.
    """

    def test_missing_stems_never_blends_regardless_of_confidence(self):
        from scripts.api.task_modules.remix import _should_use_stem_blend

        use_blend, confidence = _should_use_stem_blend(
            has_stems_a=False, has_stems_b=True,
            harmonic_score=0.95, spectral_score=0.95, threshold=0.30,
        )
        assert use_blend is False
        assert confidence is None

    def test_high_confidence_pair_uses_stem_blend(self):
        from scripts.api.task_modules.remix import _should_use_stem_blend

        use_blend, confidence = _should_use_stem_blend(
            has_stems_a=True, has_stems_b=True,
            harmonic_score=0.8, spectral_score=0.7, threshold=0.30,
        )
        assert use_blend is True
        assert confidence == pytest.approx(0.5 * 0.8 + 0.5 * 0.7)

    def test_low_confidence_pair_falls_back_to_plain_crossfade(self):
        from scripts.api.task_modules.remix import _should_use_stem_blend

        use_blend, confidence = _should_use_stem_blend(
            has_stems_a=True, has_stems_b=True,
            harmonic_score=0.1, spectral_score=0.1, threshold=0.30,
        )
        assert use_blend is False
        assert confidence == pytest.approx(0.1)

    def test_missing_spectral_score_falls_back_to_harmonic_alone(self):
        from scripts.api.task_modules.remix import _should_use_stem_blend

        use_blend, confidence = _should_use_stem_blend(
            has_stems_a=True, has_stems_b=True,
            harmonic_score=0.5, spectral_score=None, threshold=0.30,
        )
        assert use_blend is True
        assert confidence == pytest.approx(0.5)

    def test_unset_harmonic_score_sentinel_does_not_block_blend(self):
        """harmonic_score=-1.0 ("not computed", e.g. some classic-engine
        paths) is missing data, not evidence of a bad pairing — must not
        block the blend, same convention as compatibility_score()'s
        missing-metadata handling elsewhere in this codebase."""
        from scripts.api.task_modules.remix import _should_use_stem_blend

        use_blend, confidence = _should_use_stem_blend(
            has_stems_a=True, has_stems_b=True,
            harmonic_score=-1.0, spectral_score=0.9, threshold=0.30,
        )
        assert use_blend is True
        assert confidence is None

    def test_confidence_exactly_at_threshold_still_blends(self):
        from scripts.api.task_modules.remix import _should_use_stem_blend

        use_blend, confidence = _should_use_stem_blend(
            has_stems_a=True, has_stems_b=True,
            harmonic_score=0.30, spectral_score=None, threshold=0.30,
        )
        assert use_blend is True
        assert confidence == pytest.approx(0.30)
