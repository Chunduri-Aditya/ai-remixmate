"""
tests/test_behavioral.py — Behavioral tests for bugs found in the June 2026 audit.

These tests assert *what you hear*, not just shape/dtype.  Every test here
corresponds to a bug that was invisible to the pre-audit green suite because
the existing tests only checked dtype, no-NaN, and length.

Coverage:
  - render() B-silence: Song B must be ≈ 0 in the first half of the window
  - key_detection Camelot table: pitch_shift_for_camelot agrees with CAMELOT/NOTE_NAMES
  - jobs.py lifecycle: create / update / complete / cancel / ETA guard
  - dj_analysis clash-path: bass_swap_bar lands at midpoint even when shortened
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

SR = 22050  # low SR keeps tests fast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * seconds), dtype=np.float32)


def _tone(seconds: float, freq: float = 440.0, amplitude: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _make_plan(transition_seconds: float = 4.0, bpm: float = 128.0):
    from scripts.core.dj_analysis import EQPlan, TransitionPlan
    bars = 4
    eq = EQPlan(
        hp_start_hz=400.0,
        hp_end_hz=80.0,
        hp_ramp_bars=bars // 2,
        bass_swap_bar=bars // 2,
        bass_crossover_hz=150.0,
        a_fade_start_bar=0,
        a_fade_end_bar=bars,
        b_fade_start_bar=0,
        b_fade_end_bar=bars,
    )
    return TransitionPlan(
        exit_bar_a=0,
        exit_time_a=0.0,
        entry_bar_b=0,
        entry_time_b=0.0,
        transition_bars=bars,
        transition_seconds=transition_seconds,
        bpm_a=bpm,
        bpm_b=bpm,
        tempo_shift_ratio=1.0,
        eq=eq,
    )


# ---------------------------------------------------------------------------
# 1.  render() — Song B must be silent in the first half
# ---------------------------------------------------------------------------

class TestRenderBSilence:
    """
    D1-critical: _apply_dynamic_eq_fade(direction='in') was copying the full
    audio into result before zeroing only the tail, leaving B at full amplitude
    in the first half (the "solo A" section).

    Verification strategy: feed A = silence, B = constant tone.
    In the first half of the output, B's contribution must be ≈ 0.
    """

    def test_b_is_silent_in_first_half(self):
        from scripts.core.dj_engine import DJEngine

        engine = DJEngine(sr=SR)
        plan   = _make_plan(transition_seconds=4.0)

        trans_samples = int(plan.transition_seconds * SR)
        mid           = trans_samples // 2

        # A is silence → any non-zero in output[:mid] must come from B
        track_a = _silence(10.0)
        # B is a loud constant tone
        track_b = _tone(10.0, amplitude=0.8)

        result = engine.render(track_a, track_b, plan, full_output=False)

        assert len(result) >= trans_samples, "output shorter than transition window"

        first_half_rms  = _rms(result[:mid])
        second_half_rms = _rms(result[mid:trans_samples])

        # First half: B must be (nearly) silent
        # Threshold is generous to allow HP ramp residuals; real silence is < 1e-5
        assert first_half_rms < 0.05, (
            f"Song B is audible in the 'solo A' section: rms={first_half_rms:.4f}. "
            "Expected < 0.05 (near silence)."
        )

        # Second half: B should rise in noticeably
        assert second_half_rms > first_half_rms * 5, (
            f"Song B does not rise in the second half: "
            f"first_half_rms={first_half_rms:.4f}, second_half_rms={second_half_rms:.4f}"
        )

    def test_b_silence_with_scipy_fallback(self):
        """Same assertion for the scipy-absent fallback path."""
        import sys
        with patch.dict(sys.modules, {"scipy": None, "scipy.signal": None}):
            # Re-import DJEngine with scipy blocked so it hits the fallback path
            import importlib

            import scripts.core.dj_engine as _mod
            importlib.reload(_mod)
            engine = _mod.DJEngine(sr=SR)
            plan   = _make_plan(transition_seconds=4.0)

            trans_samples = int(plan.transition_seconds * SR)
            mid           = trans_samples // 2

            track_a = _silence(10.0)
            track_b = _tone(10.0, amplitude=0.8)

            result = engine.render(track_a, track_b, plan, full_output=False)
            first_half_rms = _rms(result[:mid])

            assert first_half_rms < 0.05, (
                f"Scipy-fallback: B audible in first half rms={first_half_rms:.4f}"
            )


# ---------------------------------------------------------------------------
# 2.  key_detection — Camelot semitone table
# ---------------------------------------------------------------------------

class TestCamelotSemitoneTable:
    """
    D1-major: pitch_shift_for_camelot had an inline table with A-ring off by
    +1 and B-ring 1B–7B off by −4 semitones.

    Fix: derive from CAMELOT / NOTE_NAMES (single source of truth).

    Assertions:
      - Same code → 0 semitones (identity)
      - Known intervals from the circle of fifths
      - Normalised to ±6 semitones
    """

    SAME_CODE_CASES = [
        '1A', '2A', '3A', '4A', '5A', '6A', '7A', '8A', '9A', '10A', '11A', '12A',
        '1B', '2B', '3B', '4B', '5B', '6B', '7B', '8B', '9B', '10B', '11B', '12B',
    ]

    @pytest.mark.parametrize("code", SAME_CODE_CASES)
    def test_same_code_returns_zero(self, code):
        from scripts.core.key_detection import pitch_shift_for_camelot
        assert pitch_shift_for_camelot(code, code) == 0, (
            f"pitch_shift_for_camelot({code!r}, {code!r}) should be 0"
        )

    def test_c_to_g_major_is_seven_or_minus_five(self):
        """C major (8B) → G major (9B): G is 7 semitones above C.
        After ±6 normalisation, 7 → -5 (shorter path downward)."""
        from scripts.core.key_detection import pitch_shift_for_camelot
        result = pitch_shift_for_camelot('8B', '9B')
        # Normalised shift: 7 semitones normalised to −5 (prefer the shorter path)
        assert result == -5, (
            f"C major → G major expected -5 semitones (normalised), got {result}"
        )

    def test_a_minor_to_e_minor_is_seven_or_minus_five(self):
        """A minor (8A) → E minor (9A): same circle-of-fifths step."""
        from scripts.core.key_detection import pitch_shift_for_camelot
        result = pitch_shift_for_camelot('8A', '9A')
        assert result == -5, (
            f"A minor → E minor expected -5 semitones (normalised), got {result}"
        )

    def test_shift_normalised_to_pm6(self):
        """All computed shifts must be in the range [−6, 6]."""
        from scripts.core.key_detection import pitch_shift_for_camelot
        codes = self.SAME_CODE_CASES
        for src in codes:
            for tgt in codes:
                result = pitch_shift_for_camelot(src, tgt)
                assert -6 <= result <= 6, (
                    f"pitch_shift_for_camelot({src!r}, {tgt!r}) = {result} "
                    "is outside ±6 semitone range"
                )

    def test_agrees_with_camelot_and_note_names(self):
        """Cross-check every (src, tgt) pair against manual derivation from CAMELOT."""
        from scripts.core.key_detection import (
            CAMELOT,
            NOTE_NAMES,
            pitch_shift_for_camelot,
        )
        # Build ground-truth reverse map
        camelot_to_semitone: dict[str, int] = {}
        for key_name, code in CAMELOT.items():
            root = key_name.split()[0]
            camelot_to_semitone[code] = NOTE_NAMES.index(root)

        for src, s_tone in camelot_to_semitone.items():
            for tgt, t_tone in camelot_to_semitone.items():
                expected = t_tone - s_tone
                if expected > 6:
                    expected -= 12
                elif expected < -6:
                    expected += 12
                actual = pitch_shift_for_camelot(src, tgt)
                assert actual == expected, (
                    f"pitch_shift_for_camelot({src!r}, {tgt!r}): "
                    f"expected {expected}, got {actual}"
                )


# ---------------------------------------------------------------------------
# 3.  jobs.py — lifecycle, ETA guard, cancel, status normalisation
# ---------------------------------------------------------------------------

class TestJobsLifecycle:
    """
    D4/D2: jobs.py had no direct tests. Covers create/update/complete, ETA
    guard (elapsed=0 crash), cancel, and _norm_status → 'COMPLETED'.
    """

    def setup_method(self):
        """Each test gets a fresh patched module state."""
        # We import directly; tests are independent of SQLite by patching _upsert_row
        self._upsert_patch = patch(
            "scripts.api.jobs._upsert_row",
            return_value=None,
        )
        self._conn_patch = patch(
            "scripts.api.jobs._get_conn",
            return_value=MagicMock(__enter__=lambda s: MagicMock(), __exit__=MagicMock()),
        )

    def test_create_job_returns_uuid(self):
        from scripts.api.jobs import create_job
        from scripts.api.schemas import JobType
        with self._upsert_patch, self._conn_patch:
            job_id = create_job(JobType.DOWNLOAD)
        assert isinstance(job_id, str)
        uuid.UUID(job_id)  # must be a valid UUID

    def test_create_and_get_job(self):
        from scripts.api.jobs import create_job, get_job
        from scripts.api.schemas import JobType
        with self._upsert_patch, self._conn_patch:
            job_id = create_job(JobType.DOWNLOAD)
            job = get_job(job_id)
        assert job is not None
        assert job["job_id"] == job_id

    def test_update_job_progress(self):
        from scripts.api.jobs import create_job, get_job, update_job
        from scripts.api.schemas import JobType
        with self._upsert_patch, self._conn_patch:
            job_id = create_job(JobType.SEPARATE)
            update_job(job_id, progress=0.5, message="half done")
            job = get_job(job_id)
        assert job["progress"] == pytest.approx(0.5, abs=0.001)
        assert job["message"] == "half done"

    def test_eta_guard_no_division_by_zero(self):
        """Elapsed = 0 must not crash — was: rate = progress / 0."""
        from scripts.api.jobs import create_job, get_job, update_job
        from scripts.api.schemas import JobType
        with self._upsert_patch, self._conn_patch:
            job_id = create_job(JobType.DJ_REMIX)
            import scripts.api.jobs as _jobs_mod
            _jobs_mod._jobs[job_id]["started_at"] = time.time()
            # This must not raise ZeroDivisionError
            update_job(job_id, progress=0.01)
            job = get_job(job_id)
        assert job["eta_sec"] is not None
        assert job["eta_sec"] >= 0

    def test_cancel_marks_cancelled(self):
        from scripts.api.jobs import cancel_job, create_job, get_job
        from scripts.api.schemas import JobType
        with self._upsert_patch, self._conn_patch:
            job_id = create_job(JobType.DOWNLOAD)
            result = cancel_job(job_id)
            job = get_job(job_id)
        assert result is True
        status = job["status"]
        status_val = status.value if hasattr(status, "value") else str(status)
        assert "cancel" in status_val.lower()

    def test_job_to_response_status_is_completed_not_done(self):
        """REST serializer must return COMPLETED (uppercase) not 'done'."""
        from scripts.api.jobs import create_job, get_job, job_to_response
        from scripts.api.schemas import JobStatus, JobType
        with self._upsert_patch, self._conn_patch:
            job_id = create_job(JobType.DJ_REMIX)
            import scripts.api.jobs as _jobs_mod
            _jobs_mod._jobs[job_id]["status"] = JobStatus.DONE
            job = get_job(job_id)
            response = job_to_response(job)
        assert response.status == "COMPLETED", (
            f"REST serializer returned {response.status!r}; expected 'COMPLETED'. "
            "This causes status mismatch between SSE and REST consumers."
        )


# ---------------------------------------------------------------------------
# 4.  dj_analysis — clash-path bass_swap_bar at midpoint
# ---------------------------------------------------------------------------

class TestClashPathBassSwap:
    """
    D1-major: when consonance < 0.35 the transition was shortened AFTER
    EQPlan was built, leaving bass_swap_bar at the end of the new window.

    Fix: consonance is now computed before EQPlan.
    Assert: after shortening, eq.bass_swap_bar == new_transition_bars // 2.
    """

    @pytest.mark.dj_analysis
    def test_bass_swap_bar_at_midpoint_after_clash_shortening(self):
        from scripts.core.dj_analysis import SongStructure, plan_transition

        def _song(key="C", mode="major", bpm=128.0, camelot="8B"):
            # Real constructor (not __new__) so every defaulted field exists —
            # the fake broke once when a refactor renamed fields it had set by
            # hand and made total_bars a property derived from len(bars).
            s = SongStructure(bpm=bpm, duration=240.0)
            s.key_name  = key
            s.mode      = mode
            s.camelot   = camelot
            s.bars      = [(i * (60.0 / bpm) * 4, (i + 1) * (60.0 / bpm) * 4) for i in range(128)]
            s.phrase_boundaries = [s.bars[32][0], s.bars[64][0], s.bars[96][0]]
            return s

        # Force psychoacoustic_consonance to return a clashing score
        with patch(
            "scripts.core.dj_analysis.psychoacoustic_consonance" if False else
            "scripts.core.key_detection.psychoacoustic_consonance",
            return_value=0.10,   # well below the 0.35 threshold
        ):
            # Mock at the point of import inside plan_transition
            with patch("scripts.core.key_detection.psychoacoustic_consonance", return_value=0.10):
                song_a = _song("C", "major", camelot="8B")
                song_b = _song("F#", "minor", camelot="11A")  # distant on wheel
                plan = plan_transition(song_a, song_b, transition_bars=16)

        # After shortening (16→8 bars), midpoint should be 4
        expected_swap_bar = plan.transition_bars // 2
        assert plan.eq.bass_swap_bar == expected_swap_bar, (
            f"bass_swap_bar={plan.eq.bass_swap_bar} but expected midpoint "
            f"{expected_swap_bar} (transition_bars={plan.transition_bars})"
        )


# ---------------------------------------------------------------------------
# 5.  spectral_mixing — masking-aware multiband EQ
# ---------------------------------------------------------------------------

@pytest.mark.dj_analysis
class TestMaskingAwareMultibandEQ:
    """
    New feature: scripts/core/spectral_mixing.py generalizes the existing
    bass-only dynamic EQ to the rest of the spectrum (Hafezi & Reiss 2015,
    JAES 63(5):312-323, DOI 10.17743/jaes.2015.0021 — masking reduction via
    per-band gain reduction on the "masker" track).

    These assert AUDIBLE MIXING BEHAVIOR (measured band-energy collision,
    loudness tolerance, protected-band integrity), not array shape/dtype,
    matching the rest of this file's philosophy.

    Track A is built to be "busy" — strong energy in the bass/low-mid/mid
    bands (indices 0-2) plus a real-but-lower-ranked tone in the high-mid
    band (index 3, 2-6 kHz). Track B is a single sine concentrated ONLY in
    the high-mid band. This makes band 3 non-essential to A (A has three
    louder bands elsewhere) but essential to B (B's only real content) —
    exactly the Hafezi & Reiss masking condition, so A should get ducked in
    band 3 while B is left alone there.
    """

    @staticmethod
    def _busy_track_a(seconds: float = 6.0, sr: int = SR) -> np.ndarray:
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        bass    = np.sin(2 * np.pi * 100.0  * t) * 0.9    # band 0: 20-200 Hz
        low_mid = np.sin(2 * np.pi * 350.0  * t) * 0.9    # band 1: 200-500 Hz
        mid     = np.sin(2 * np.pi * 1000.0 * t) * 0.9    # band 2: 500-2000 Hz
        # band 3: real, and louder than B's tone there (so A actually masks
        # B) but still A's own quietest band (so it's non-essential to A).
        high_mid = np.sin(2 * np.pi * 3500.0 * t) * 0.6
        return (bass + low_mid + mid + high_mid).astype(np.float32)

    @staticmethod
    def _sparse_track_b(seconds: float = 6.0, sr: int = SR) -> np.ndarray:
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        # band 3 only, quieter than A's collocated tone but B's sole content
        # (so it's B's most essential band by rank even though A is louder there).
        high_mid = np.sin(2 * np.pi * 3500.0 * t) * 0.45
        return high_mid.astype(np.float32)

    def test_collision_reduced_in_high_mid_band(self):
        """Masker (A)'s energy in the collision band (index 3) should drop
        measurably after unmask_stem_pair; the maskee (B) should not."""
        from scripts.core.spectral_mixing import compute_band_energies, unmask_stem_pair

        a = self._busy_track_a()
        b = self._sparse_track_b()

        a_bands_before = compute_band_energies(a, SR)
        proc_a, proc_b = unmask_stem_pair(a, b, SR, strength=1.0, max_atten_db=6.0)
        a_bands_after = compute_band_energies(proc_a, SR)
        b_bands_after = compute_band_energies(proc_b, SR)
        b_bands_before = compute_band_energies(b, SR)

        collision_band = 3
        before_db = float(np.mean(a_bands_before[collision_band]))
        after_db = float(np.mean(a_bands_after[collision_band]))
        reduction = before_db - after_db

        assert reduction > 1.0, (
            f"Masker A's high-mid band energy barely changed "
            f"({before_db:.2f} dB -> {after_db:.2f} dB, reduction={reduction:.2f} dB). "
            "Expected a measurable (>1 dB) masking-reduction attenuation."
        )

        b_before_db = float(np.mean(b_bands_before[collision_band]))
        b_after_db = float(np.mean(b_bands_after[collision_band]))
        assert abs(b_before_db - b_after_db) < 1.0, (
            f"Maskee B's high-mid band changed by {abs(b_before_db - b_after_db):.2f} dB "
            "— B should be left alone since the band is essential to it, not to A."
        )

    def test_bass_band_protected(self):
        """protect_bands=[0] must leave the bass band untouched even though
        it's A's loudest band and would otherwise be a candidate for gain
        changes in a naive implementation."""
        from scripts.core.spectral_mixing import compute_band_energies, unmask_stem_pair

        a = self._busy_track_a()
        b = self._sparse_track_b()

        a_bands_before = compute_band_energies(a, SR)
        proc_a, _ = unmask_stem_pair(a, b, SR, strength=1.0, max_atten_db=6.0, protect_bands=[0])
        a_bands_after = compute_band_energies(proc_a, SR)

        bass_before = float(np.mean(a_bands_before[0]))
        bass_after = float(np.mean(a_bands_after[0]))
        assert abs(bass_before - bass_after) < 0.5, (
            f"Bass band (protected) changed by {abs(bass_before - bass_after):.2f} dB — "
            "this band must remain the sole authority of _stem_bass_ramp()/the "
            "existing dynamic low-shelf, never touched by the masking EQ."
        )

    def test_loudness_stays_within_tolerance(self):
        """No gross level change (no pumping) — overall RMS should stay
        within a small tolerance of the unprocessed signal."""
        from scripts.core.spectral_mixing import unmask_stem_pair

        a = self._busy_track_a()
        b = self._sparse_track_b()

        proc_a, proc_b = unmask_stem_pair(a, b, SR, strength=1.0, max_atten_db=6.0, protect_bands=[0])

        rms_before_a, rms_after_a = _rms(a), _rms(proc_a)
        rms_before_b, rms_after_b = _rms(b), _rms(proc_b)

        db_change_a = 20 * np.log10(max(rms_after_a, 1e-8) / max(rms_before_a, 1e-8))
        db_change_b = 20 * np.log10(max(rms_after_b, 1e-8) / max(rms_before_b, 1e-8))

        assert -3.0 <= db_change_a <= 0.5, (
            f"Track A's overall RMS changed by {db_change_a:.2f} dB — "
            "expected a gentle reduction (masking EQ only), not a gross level change."
        )
        assert -1.0 <= db_change_b <= 0.5, (
            f"Track B's overall RMS changed by {db_change_b:.2f} dB — "
            "B should be essentially unaffected (it's the maskee, not the masker)."
        )

    def test_strength_monotonicity(self):
        """Higher `strength` should never produce less attenuation than a
        lower strength, in the collision band, up to max_atten_db."""
        from scripts.core.spectral_mixing import compute_band_energies, unmask_stem_pair

        a = self._busy_track_a()
        b = self._sparse_track_b()
        collision_band = 3

        results = {}
        for strength in (0.0, 1.0, 2.0):
            proc_a, _ = unmask_stem_pair(a, b, SR, strength=strength, max_atten_db=6.0)
            bands = compute_band_energies(proc_a, SR)
            results[strength] = float(np.mean(bands[collision_band]))

        assert results[1.0] <= results[0.0] + 0.25, (
            f"strength=1.0 produced less attenuation than strength=0.0 "
            f"({results[1.0]:.2f} dB vs {results[0.0]:.2f} dB) — expected monotonic "
            "non-decreasing attenuation as strength rises."
        )
        assert results[2.0] <= results[1.0] + 0.25, (
            f"strength=2.0 produced less attenuation than strength=1.0 "
            f"({results[2.0]:.2f} dB vs {results[1.0]:.2f} dB)."
        )

    def test_window_edges_splice_transparent(self):
        """The gain curve is tapered to 0 dB at the window edges, so the
        processed window must be near-identical to the unprocessed stem at
        the very start and end — dj_engine splices it back into the full
        stem, and a level step at the splice point would click."""
        from scripts.core.spectral_mixing import unmask_stem_pair

        a = self._busy_track_a()
        b = self._sparse_track_b()

        proc_a, _ = unmask_stem_pair(a, b, SR, strength=2.0, max_atten_db=12.0)

        edge = 512  # comfortably inside the ~50 ms unity-gain taper
        for region, orig_seg, proc_seg in (
            ("start", a[:edge], proc_a[:edge]),
            ("end", a[-edge:], proc_a[-edge:]),
        ):
            err_rms = _rms(orig_seg - proc_seg)
            sig_rms = _rms(orig_seg)
            assert err_rms < 0.1 * sig_rms, (
                f"Processed window diverges from the original at the {region} "
                f"(error RMS {err_rms:.5f} vs signal RMS {sig_rms:.5f}) — the "
                "edge taper must keep splice boundaries transparent."
            )


# ---------------------------------------------------------------------------
# 6.  dj_engine crossfade curves — equal power, seam continuity
# ---------------------------------------------------------------------------

class TestEqualPowerCrossfade:
    """
    The standard (medium-similarity) stem crossfade must be EQUAL-POWER:
    fade_out² + fade_in² = 1 at every sample. Two different songs are
    uncorrelated signals, so their powers add during the overlap — the old
    raised-cosine curves (summing to 1.0 in amplitude) dipped the mix ~3 dB
    at the midpoint of every standard transition, the classic equal-gain
    mistake DJ mixers exist to avoid.

    Pure numpy — no librosa, so no dj_analysis mark.
    """

    def test_constant_power_across_window(self):
        from scripts.core.dj_engine import _stem_crossfade_curves

        fade_out, fade_in = _stem_crossfade_curves(10_000, similarity=0.5)
        power = fade_out.astype(np.float64) ** 2 + fade_in.astype(np.float64) ** 2

        assert float(np.max(np.abs(power - 1.0))) < 1e-3, (
            f"Standard crossfade is not equal-power: combined power spans "
            f"[{power.min():.4f}, {power.max():.4f}] — a mid-transition "
            "loudness dip (or bump) will be audible on uncorrelated songs."
        )

    def test_endpoints_are_exact_handoff(self):
        from scripts.core.dj_engine import _stem_crossfade_curves

        fade_out, fade_in = _stem_crossfade_curves(10_000, similarity=0.5)
        assert abs(float(fade_out[0]) - 1.0) < 1e-6 and abs(float(fade_in[0])) < 1e-6
        assert abs(float(fade_out[-1])) < 1e-3 and abs(float(fade_in[-1]) - 1.0) < 1e-3

    def test_uncorrelated_noise_holds_loudness_at_midpoint(self):
        """The audible claim itself: crossfading two uncorrelated noise beds
        must not dip at the transition midpoint by more than ~1 dB relative
        to the window edges (the equal-gain curves dipped ~3 dB)."""
        from scripts.core.dj_engine import _stem_crossfade_curves

        rng = np.random.default_rng(1337)
        n = SR * 4
        a = rng.standard_normal(n).astype(np.float32) * 0.3
        b = rng.standard_normal(n).astype(np.float32) * 0.3

        fade_out, fade_in = _stem_crossfade_curves(n, similarity=0.5)
        mixed = a * fade_out + b * fade_in

        k = SR // 4  # 250 ms analysis windows
        rms_edge = _rms(mixed[:k])
        rms_mid = _rms(mixed[n // 2 - k // 2 : n // 2 + k // 2])
        dip_db = 20 * np.log10(max(rms_mid, 1e-9) / max(rms_edge, 1e-9))

        assert dip_db > -1.0, (
            f"Mid-transition loudness dips {dip_db:.2f} dB below the window "
            "edge — crossfade is not constant-power for uncorrelated material."
        )


class TestSeamBlend:
    """
    render_stem_blend()'s full-output assembly splices a stem-summed,
    renormalized transition window against the ORIGINAL full mixes on both
    sides. _seam_blend() smooths those splices; these pin its contract.
    Pure numpy — no dj_analysis mark.
    """

    def test_endpoints_continuous(self):
        from scripts.core.dj_engine import _seam_blend

        rng = np.random.default_rng(7)
        frm = rng.standard_normal(1024).astype(np.float32)
        to = rng.standard_normal(1024).astype(np.float32)
        out = _seam_blend(frm, to)

        assert abs(float(out[0] - frm[0])) < 1e-6, (
            "Seam start must equal the outgoing signal exactly — anything "
            "else reintroduces the click the seam exists to remove."
        )
        assert abs(float(out[-1] - to[-1])) < 1e-6, (
            "Seam end must equal the incoming signal exactly."
        )

    def test_identical_signals_pass_through_unchanged(self):
        """Seam material is two renderings of the same audio moment
        (original mix vs Demucs stem sum) — i.e. highly correlated. An
        equal-gain blend of identical signals must be transparent; an
        equal-power blend would bulge +3 dB mid-seam."""
        from scripts.core.dj_engine import _seam_blend

        x = np.sin(2 * np.pi * 220.0 * np.linspace(0, 0.05, 2048)).astype(np.float32)
        out = _seam_blend(x, x)
        assert float(np.max(np.abs(out - x))) < 1e-6, (
            "Blending a signal with itself must be an identity — the seam "
            "blend must be equal-gain (linear), not equal-power."
        )
