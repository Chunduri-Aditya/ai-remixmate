"""tests/test_mastering.py — Unit tests for the mastering engine."""
from __future__ import annotations

import numpy as np
import pytest

# mastering.py requires scipy — skip the whole module if it's not installed
pytest.importorskip("scipy", reason="scipy not installed; skipping mastering tests")

from tests.synth_audio import synth_click_track  # noqa: E402

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
        from scripts.core.mastering import compute_lufs, normalize_to_lufs
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
        from scripts.core.mastering import compute_lufs, master_mix
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


# ---------------------------------------------------------------------------
# FxNorm corpus-target normalization tests (Stage 1A)
# ---------------------------------------------------------------------------

def _make_stem_dict(amplitude: float = 0.5, duration: float = 5.0):
    """Build a {stem_name: (audio, sr)} dict with different amplitudes per type."""
    stems = {}
    for i, name in enumerate(("drums", "bass", "vocals", "other")):
        freq = 100.0 * (2 ** i)
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        audio = (np.sin(2 * np.pi * freq * t) * amplitude * (1.0 + i * 0.1)).astype(np.float32)
        stems[name] = (audio, SR)
    return stems


try:
    import soundfile as _sf
    _HAS_SF = True
except ImportError:
    _sf = None  # type: ignore
    _HAS_SF = False

_skip_no_sf = pytest.mark.skipif(not _HAS_SF, reason="soundfile not installed")


@_skip_no_sf
class TestAnalyzeLibraryStemTargets:
    def test_empty_library_returns_fallbacks(self, tmp_path):
        from scripts.core.mastering import _STEM_FALLBACK_TARGETS, analyze_library_stem_targets
        result = analyze_library_stem_targets(library_dir=tmp_path)
        assert result == _STEM_FALLBACK_TARGETS

    def test_returns_all_stem_types(self, tmp_path):
        from scripts.core.mastering import analyze_library_stem_targets
        song_dir = tmp_path / "FakeSong"
        song_dir.mkdir()
        for stem in ("drums", "bass", "vocals", "other"):
            audio = _sine(duration=5.0, amplitude=0.4)
            _sf.write(str(song_dir / f"{stem}.wav"), audio, SR)
        result = analyze_library_stem_targets(library_dir=tmp_path, min_songs=1)
        assert set(result.keys()) == {"drums", "bass", "vocals", "other"}

    def test_all_values_finite(self, tmp_path):
        from scripts.core.mastering import analyze_library_stem_targets
        for i in range(3):
            d = tmp_path / f"Song{i}"
            d.mkdir()
            for stem in ("drums", "bass", "vocals", "other"):
                _sf.write(str(d / f"{stem}.wav"), _sine(amplitude=0.3 + 0.1 * i), SR)
        result = analyze_library_stem_targets(library_dir=tmp_path, min_songs=3)
        for k, v in result.items():
            assert np.isfinite(v), f"Non-finite target for stem '{k}': {v}"

    def test_min_songs_threshold_falls_back(self, tmp_path):
        """With only 1 song but min_songs=3, should use fallbacks."""
        from scripts.core.mastering import _STEM_FALLBACK_TARGETS, analyze_library_stem_targets
        d = tmp_path / "OneSong"
        d.mkdir()
        for stem in ("drums", "bass", "vocals", "other"):
            _sf.write(str(d / f"{stem}.wav"), _sine(), SR)
        result = analyze_library_stem_targets(library_dir=tmp_path, min_songs=3)
        assert result == _STEM_FALLBACK_TARGETS


@_skip_no_sf
class TestSaveLoadStemTargets:
    def test_roundtrip(self, tmp_path):
        from scripts.core.mastering import load_stem_targets, save_stem_targets
        targets = {"drums": -18.0, "bass": -22.0, "vocals": -19.0, "other": -21.0}
        cache = tmp_path / "targets.json"
        save_stem_targets(targets, cache_path=cache)
        loaded = load_stem_targets(cache_path=cache)
        for k in targets:
            assert abs(loaded[k] - targets[k]) < 0.001

    def test_missing_cache_returns_fallbacks(self, tmp_path):
        from scripts.core.mastering import _STEM_FALLBACK_TARGETS, load_stem_targets
        result = load_stem_targets(cache_path=tmp_path / "nonexistent.json")
        assert result == _STEM_FALLBACK_TARGETS

    def test_partial_cache_merged_with_fallbacks(self, tmp_path):
        """Cache with only 'drums' should merge — other stems use fallback."""
        import json

        from scripts.core.mastering import _STEM_FALLBACK_TARGETS, load_stem_targets
        cache = tmp_path / "partial.json"
        cache.write_text(json.dumps({"drums": -17.5}))
        result = load_stem_targets(cache_path=cache)
        assert result["drums"] == -17.5
        assert result["bass"] == _STEM_FALLBACK_TARGETS["bass"]


class TestNormalizeStemsToCorpusTargets:
    def test_returns_all_stems(self):
        from scripts.core.mastering import normalize_stems_to_corpus_targets
        stems = _make_stem_dict()
        result = normalize_stems_to_corpus_targets(stems)
        assert set(result.keys()) == {"drums", "bass", "vocals", "other"}

    def test_stem_type_targets_differ(self):
        """Drums and bass should hit different LUFS targets — not one flat value."""
        from scripts.core.mastering import (
            _STEM_FALLBACK_TARGETS,
            compute_lufs,
            normalize_stems_to_corpus_targets,
        )
        stems = _make_stem_dict()
        result = normalize_stems_to_corpus_targets(stems, targets=_STEM_FALLBACK_TARGETS)
        drums_lufs = compute_lufs(result["drums"], SR)
        bass_lufs  = compute_lufs(result["bass"], SR)
        # Fallback targets differ by 2.5 LU, so normalized levels should differ too
        assert abs(drums_lufs - bass_lufs) > 1.0

    def test_normalization_within_1lu_of_target(self):
        from scripts.core.mastering import compute_lufs, normalize_stems_to_corpus_targets
        targets = {"drums": -18.0, "bass": -22.0, "vocals": -19.5, "other": -21.5}
        stems = _make_stem_dict()
        result = normalize_stems_to_corpus_targets(stems, targets=targets)
        for stem_name, audio in result.items():
            actual = compute_lufs(audio, SR)
            target = targets[stem_name]
            assert abs(actual - target) < 1.5, (
                f"Stem '{stem_name}': got {actual:.1f} LUFS, wanted {target:.1f} (±1.5)"
            )

    def test_silent_stem_no_crash(self):
        from scripts.core.mastering import normalize_stems_to_corpus_targets
        stems = {"drums": (np.zeros(SR * 5, dtype=np.float32), SR)}
        result = normalize_stems_to_corpus_targets(stems)
        assert "drums" in result
        assert not np.isnan(result["drums"]).any()

    def test_true_peak_applied(self):
        """Output stems must not exceed true-peak ceiling."""
        from scripts.core.mastering import normalize_stems_to_corpus_targets
        # Very loud signal that would clip without ceiling
        loud = np.ones(SR * 5, dtype=np.float32) * 0.99
        stems = {"drums": (loud, SR)}
        result = normalize_stems_to_corpus_targets(stems, true_peak_ceiling=-1.0)
        ceiling_lin = 10.0 ** (-1.0 / 20.0)
        assert float(np.max(np.abs(result["drums"]))) <= ceiling_lin + 1e-4

    def test_no_nan_inf_in_output(self):
        from scripts.core.mastering import normalize_stems_to_corpus_targets
        stems = _make_stem_dict()
        result = normalize_stems_to_corpus_targets(stems)
        for name, audio in result.items():
            assert np.isfinite(audio).all(), f"NaN/inf in stem '{name}'"

    def test_fallback_for_unknown_stem_type(self):
        """A stem type not in targets dict should fall back to fallback_lufs."""
        from scripts.core.mastering import compute_lufs, normalize_stems_to_corpus_targets
        stems = {"unknown_stem": (_sine(), SR)}
        fallback = -20.0
        result = normalize_stems_to_corpus_targets(stems, targets={}, fallback_lufs=fallback)
        actual_lufs = compute_lufs(result["unknown_stem"], SR)
        assert abs(actual_lufs - fallback) < 1.5


# ---------------------------------------------------------------------------
# apply_limiter — REMIX_QUALITY_INSIGHTS.md finding #4 regression guard
#
# Confirmed root cause: the old implementation smoothed gain reduction with
# a single release_ms time constant in BOTH directions (one scipy.lfilter
# call). A transient needing full reduction took the same ~50ms to engage as
# a fully-limited section took to recover, so the unattenuated peak passed
# through during that attack lag and got caught by the hard-clip safety net
# instead of being smoothly limited — audible distortion, not loudness
# control. Reproduced live this session as a 227,029-sample clipping defect
# on an extreme-tempo-stretch render. Fix: independent fast-attack /
# slow-release coefficients via _smooth_gain_envelope().
# ---------------------------------------------------------------------------

class TestLimiterAttackRelease:
    def test_attack_faster_than_release(self):
        """Gain must drop toward a sudden reduction target faster than it
        recovers back toward unity afterward — that asymmetry is the entire
        point of a look-ahead limiter and was missing before this fix."""
        from scripts.core.mastering import _smooth_gain_envelope
        reduction = np.ones(2000)
        reduction[500:520] = 0.1  # sudden heavy reduction for 20 samples
        alpha_attack = float(np.exp(-1.0 / (SR * 0.002)))   # 2ms
        alpha_release = float(np.exp(-1.0 / (SR * 0.05)))   # 50ms
        gain = _smooth_gain_envelope(reduction, alpha_attack, alpha_release)

        # Samples to fall within 10% of the reduction target after the step down
        attack_idx = int(np.argmax(gain[500:] <= 0.11))
        # Samples to climb back within 10% of unity after the step back up
        release_idx = int(np.argmax(gain[520:] >= 0.9))

        assert attack_idx < release_idx, (
            f"attack ({attack_idx} samples) should be faster than "
            f"release ({release_idx} samples)"
        )
        assert attack_idx < 200, "attack should engage in well under 5ms"

    def test_dense_transients_no_clipping(self):
        """Dense, sharp transients (the exact material the original symmetric
        smoothing failed on) must not exceed the ceiling after limiting."""
        from scripts.core.mastering import apply_limiter
        rng = np.random.default_rng(0)
        n = SR * 2
        audio = (rng.standard_normal(n) * 0.05).astype(np.float32)
        for i in range(0, n, int(SR * 0.05)):
            audio[i:i + 5] = 1.5  # sharp transient well above ceiling

        ceiling_db = -1.0
        ceiling = 10.0 ** (ceiling_db / 20.0)
        out = apply_limiter(audio, ceiling_db=ceiling_db, sr=SR)

        assert np.max(np.abs(out)) <= ceiling + 1e-6
        assert np.all(np.isfinite(out))

    def test_attack_ms_parameter_respected(self):
        """A slower attack_ms should reach the reduction target later than a
        faster one, confirming the parameter actually controls attack speed."""
        from scripts.core.mastering import _smooth_gain_envelope
        reduction = np.ones(2000)
        reduction[500:] = 0.1
        alpha_release = float(np.exp(-1.0 / (SR * 0.05)))

        fast = _smooth_gain_envelope(
            reduction, float(np.exp(-1.0 / (SR * 0.001))), alpha_release)
        slow = _smooth_gain_envelope(
            reduction, float(np.exp(-1.0 / (SR * 0.02))), alpha_release)

        # 5ms after the step, the fast-attack envelope must have dropped
        # further toward the target than the slow-attack one.
        check_at = 500 + int(SR * 0.005)
        assert fast[check_at] < slow[check_at]


# ---------------------------------------------------------------------------
# apply_limiter — true-peak (not just sample-peak) regression guard.
#
# Confirmed root cause (external behavioral-correctness review): the limiter
# detected and clamped only np.abs(audio) sample values, despite being
# documented as an ITU-R BS.1770-4 "true-peak" limiter. A signal whose every
# discrete sample sat under the ceiling could still reconstruct to an
# inter-sample (true) peak above it — measured up to +2 dB over ceiling on
# hot transient material via 4x-oversampled peak detection, the same
# measurement technique used here. Fix: detect/limit on a 4x-oversampled
# copy of the signal, decimate the resulting (already-smoothed, so
# alias-safe) gain envelope back to native rate, apply it to the original
# audio. Sample-peak compliance (the pre-existing guarantee) is preserved by
# the unchanged final native-rate hard-clip.
# ---------------------------------------------------------------------------

def _true_peak_dbfs(audio: np.ndarray, oversample: int = 4) -> float:
    """Measure true (inter-sample) peak via oversampled reconstruction —
    independent of apply_limiter's own internal oversampling, so this
    actually exercises the fix rather than assuming it worked."""
    from scipy import signal as _signal
    os_audio = _signal.resample_poly(audio.astype(np.float64), oversample, 1)
    peak = float(np.max(np.abs(os_audio))) + 1e-12
    return 20 * np.log10(peak)


class TestLimiterTruePeak:
    def test_true_peak_stays_under_ceiling_on_hot_transients(self):
        """The exact failure mode the review measured: dense sharp
        transients that pass sample-peak detection but overshoot ceiling
        between samples once reconstructed."""
        from scripts.core.mastering import apply_limiter

        rng = np.random.default_rng(0)
        n = SR * 2
        audio = (rng.standard_normal(n) * 0.05).astype(np.float32)
        for i in range(0, n, int(SR * 0.05)):
            audio[i:i + 5] = 1.5  # sharp transient well above ceiling

        ceiling_db = -1.0
        out = apply_limiter(audio, ceiling_db=ceiling_db, sr=SR)

        true_peak = _true_peak_dbfs(out)
        assert true_peak <= ceiling_db + 0.3, (
            f"true peak {true_peak:.2f} dBFS exceeds ceiling {ceiling_db} dBFS "
            f"by more than the 0.3 dB tolerance — inter-sample overshoot"
        )

    def test_true_peak_stays_under_ceiling_on_gentle_bed(self):
        """The review's 'gentle bed' case still measured 0.38 dB of true-peak
        overshoot pre-fix even with no extreme transients at all — the
        overshoot was systematic, not just a hot-signal edge case."""
        from scripts.core.mastering import apply_limiter

        audio = _sine(freq=1200.0, duration=2.0, amplitude=0.85, sr=SR)

        ceiling_db = -1.0
        out = apply_limiter(audio, ceiling_db=ceiling_db, sr=SR)

        true_peak = _true_peak_dbfs(out)
        assert true_peak <= ceiling_db + 0.3

    def test_oversample_1_restores_old_sample_peak_only_behavior(self):
        """oversample=1 is an explicit escape hatch back to the original
        sample-peak-only detection — must still guarantee sample-peak
        compliance even though true-peak is no longer checked."""
        from scripts.core.mastering import apply_limiter

        rng = np.random.default_rng(1)
        audio = (rng.standard_normal(SR) * 1.2).astype(np.float32)
        ceiling_db = -1.0
        ceiling = 10.0 ** (ceiling_db / 20.0)

        out = apply_limiter(audio, ceiling_db=ceiling_db, sr=SR, oversample=1)
        assert np.max(np.abs(out)) <= ceiling + 1e-6

    def test_output_same_length_as_input(self):
        """The oversample/decimate round trip on the gain envelope must not
        change the output length — resample_poly can be off by a few
        samples, which the implementation must pad/trim to fix."""
        from scripts.core.mastering import apply_limiter

        for n in (SR - 7, SR, SR + 13, SR * 3 + 1):
            audio = (np.random.default_rng(n).standard_normal(n) * 0.3).astype(np.float32)
            out = apply_limiter(audio, sr=SR)
            assert len(out) == n

    def test_no_nan_inf_after_true_peak_fix(self):
        from scripts.core.mastering import apply_limiter

        rng = np.random.default_rng(2)
        audio = (rng.standard_normal(SR * 2) * 0.9).astype(np.float32)
        out = apply_limiter(audio, sr=SR)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Make up gain after the limiter (fix4b defect L).
#
# master_mix normalised to target, then limited, and nothing restored the
# loudness the limiter removed. A click plus pad track (high crest) mastered
# to -14 LUFS landed near -17.7 LUFS: about 1.9 dB of the loss came from the
# limiter's whole file true peak correction, the rest from gain reduction on
# the clicks. These tests pin the fix: the output lands within 1 LU of the
# target, the true peak stays at the ceiling, and the report describes the
# audio that is returned.
# ---------------------------------------------------------------------------

def _dense_mix(bpm=124.0, seconds=30.0, seed=1, sr=SR):
    """Music like bed: kick on every beat, sustained bass and chord tones,
    noise hats on the off beats. Crest factor about 12 dB."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    y = 0.3 * np.sin(2 * np.pi * 55.0 * t)
    for f in (220.0, 277.18, 329.63):
        y += 0.08 * np.sin(2 * np.pi * f * t)
    beat = 60.0 / bpm
    kick_len = int(0.15 * sr)
    kt = np.arange(kick_len) / sr
    kick = np.sin(2 * np.pi * (50.0 + 80.0 * np.exp(-kt / 0.03)) * kt) * np.exp(-kt / 0.08)
    hat_len = int(0.03 * sr)
    hat = rng.standard_normal(hat_len) * np.exp(-np.arange(hat_len) / (0.005 * sr))
    k = 0
    while k * beat < seconds:
        s = int(k * beat * sr)
        e = min(n, s + kick_len)
        y[s:e] += 0.6 * kick[: e - s]
        hs = int((k + 0.5) * beat * sr)
        if hs < n:
            he = min(n, hs + hat_len)
            y[hs:he] += 0.15 * hat[: he - hs]
        k += 1
    y = y / np.max(np.abs(y)) * 0.9
    return y.astype(np.float32)


def _crest_db(audio: np.ndarray) -> float:
    a = audio.astype(np.float64)
    return float(20 * np.log10(np.max(np.abs(a)) / np.sqrt(np.mean(a ** 2))))


def _pyln_lufs(audio: np.ndarray, sr: int = SR) -> float:
    """Reference meter, independent of mastering.compute_lufs."""
    import pyloudnorm as pyln
    return float(pyln.Meter(sr).integrated_loudness(audio.astype(np.float64)))


class TestMasterMixMakeup:
    TARGET = -14.0
    CEILING = -1.0

    @pytest.fixture(scope="class")
    def click_master(self):
        from scripts.core.mastering import master_mix
        return master_mix(synth_click_track(124.0, 60.0, 220.0, 0), SR, target_lufs=self.TARGET, ceiling_db=self.CEILING)

    @pytest.fixture(scope="class")
    def dense_master(self):
        from scripts.core.mastering import master_mix
        audio = _dense_mix()
        crest = _crest_db(audio)
        print(f"dense mix crest factor {crest:.2f} dB")
        assert 10.0 <= crest <= 14.0, f"dense fixture crest {crest:.2f} dB outside 10 to 14"
        return master_mix(audio, SR, target_lufs=self.TARGET, ceiling_db=self.CEILING)

    def test_click_pad_within_1lu_of_target(self, click_master):
        out, _ = click_master
        lufs = _pyln_lufs(out)
        assert abs(lufs - self.TARGET) <= 1.0, (
            f"click pad mastered to {lufs:.3f} LUFS, target {self.TARGET}"
        )

    def test_click_pad_true_peak_at_or_under_ceiling(self, click_master):
        out, _ = click_master
        tp = _true_peak_dbfs(out)
        assert tp <= self.CEILING + 0.1, f"true peak {tp:.3f} dBFS over ceiling {self.CEILING}"

    def test_dense_mix_within_1lu_and_under_ceiling(self, dense_master):
        out, _ = dense_master
        lufs = _pyln_lufs(out)
        tp = _true_peak_dbfs(out)
        assert abs(lufs - self.TARGET) <= 1.0, (
            f"dense mix mastered to {lufs:.3f} LUFS, target {self.TARGET}"
        )
        assert tp <= self.CEILING + 0.1, f"true peak {tp:.3f} dBFS over ceiling {self.CEILING}"

    def test_report_lufs_matches_returned_audio(self, click_master):
        from scripts.core.mastering import compute_lufs
        out, report = click_master
        assert abs(report.lufs_integrated - compute_lufs(out, SR)) <= 0.01

    def test_makeup_respects_20db_gain_clamp(self):
        from scripts.core.mastering import compute_lufs, master_mix
        audio = _sine(freq=1000.0, duration=5.0, amplitude=0.003)
        in_lufs = compute_lufs(audio, SR)
        out, report = master_mix(audio, SR, target_lufs=self.TARGET, ceiling_db=self.CEILING)
        assert report.lufs_gain_applied <= 20.0 + 1e-6
        assert abs(compute_lufs(out, SR) - (in_lufs + 20.0)) <= 0.5

    @pytest.mark.parametrize("kind", ["single_impulse", "sparse_impulse_train"])
    def test_makeup_respects_20db_clamp_when_limiter_engages(self, kind):
        # A full scale impulse is very quiet in LUFS, so normalisation hits
        # its 20 dB clamp and the limiter then pulls the peaks back down.
        # Make up must not push the total gain past that clamp (read 26.0).
        from scripts.core.mastering import master_mix
        audio = np.zeros(int(SR * 5.0), dtype=np.float32)
        if kind == "single_impulse":
            audio[SR] = 1.0
        else:
            audio[:: SR // 2] = 1.0
        out, report = master_mix(audio, SR, target_lufs=self.TARGET, ceiling_db=self.CEILING)
        assert report.lufs_gain_applied <= 20.0 + 1e-6, (
            f"{kind}: lufs_gain_applied {report.lufs_gain_applied:.3f} dB over the 20 dB clamp"
        )
        tp = _true_peak_dbfs(out)
        assert tp <= self.CEILING + 0.1, f"{kind}: true peak {tp:.3f} dBFS over ceiling {self.CEILING}"

    def test_silence_gets_no_makeup(self):
        from scripts.core.mastering import master_mix
        out, report = master_mix(_silence(), SR, target_lufs=self.TARGET, ceiling_db=self.CEILING)
        assert np.all(np.isfinite(out))
        assert np.all(out == 0.0)
        assert report.lufs_gain_applied == 0.0

    def test_unlimited_signal_gets_no_makeup(self):
        from scripts.core.mastering import compute_lufs, master_mix
        audio = _sine(freq=1000.0, duration=5.0, amplitude=0.5)
        in_lufs = compute_lufs(audio, SR)
        _, report = master_mix(audio, SR, target_lufs=self.TARGET, ceiling_db=self.CEILING)
        assert abs(report.lufs_gain_applied - (self.TARGET - in_lufs)) <= 0.05
