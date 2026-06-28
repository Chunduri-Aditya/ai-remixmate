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

import threading
import time
import uuid
from unittest.mock import patch, MagicMock

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
            pitch_shift_for_camelot, CAMELOT, NOTE_NAMES,
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
        from scripts.api.jobs import create_job, update_job, get_job
        from scripts.api.schemas import JobType
        with self._upsert_patch, self._conn_patch:
            job_id = create_job(JobType.SEPARATE)
            update_job(job_id, progress=0.5, message="half done")
            job = get_job(job_id)
        assert job["progress"] == pytest.approx(0.5, abs=0.001)
        assert job["message"] == "half done"

    def test_eta_guard_no_division_by_zero(self):
        """Elapsed = 0 must not crash — was: rate = progress / 0."""
        from scripts.api.jobs import create_job, update_job, get_job
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
        from scripts.api.jobs import create_job, cancel_job, get_job
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
        from scripts.api.jobs import create_job, job_to_response, get_job
        from scripts.api.schemas import JobType, JobStatus
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
        from scripts.core.dj_analysis import plan_transition, SongStructure

        def _song(key="C", mode="major", bpm=128.0, camelot="8B"):
            s = SongStructure.__new__(SongStructure)
            s.bpm       = bpm
            s.key_name  = key
            s.mode      = mode
            s.camelot   = camelot
            s.duration  = 240.0
            s.total_bars = 128
            s.bars      = [(i * (60.0 / bpm) * 4, (i + 1) * (60.0 / bpm) * 4) for i in range(128)]
            s.downbeats = [b[0] for b in s.bars]
            s.phrase_boundaries = [s.bars[32][0], s.bars[64][0], s.bars[96][0]]
            s.segments  = []
            s.section_labels = []
            return s

        # Force psychoacoustic_consonance to return a clashing score
        with patch(
            "scripts.core.dj_analysis.psychoacoustic_consonance" if False else
            "scripts.core.key_detection.psychoacoustic_consonance",
            return_value=0.10,   # well below the 0.35 threshold
        ):
            # Import the module and monkey-patch its reference
            import scripts.core.dj_analysis as _da
            orig = getattr(_da, "_plan_transition_impl", None)

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
