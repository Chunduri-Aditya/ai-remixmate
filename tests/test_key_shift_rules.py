"""
tests/test_key_shift_rules.py — B3.3 key correction rules for the Camelot pitch shift.

Spec (plans/audit-the-project-and-sorted-flask.md, B3.3):
  pitch_shift_for_camelot(source, target) is the shift applied to the source
  song so it sits harmonically with the target song (the planner calls it as
  pitch_shift_for_camelot(song_b.camelot, song_a.camelot)).

  * No shift when the pair is already compatible: the same code, an adjacent
    number with the same letter, or the relative A/B pair (same number).
  * Otherwise the smallest shift of at most 3 semitones that lands the source
    on a code compatible with the target, preferring 0.  A pitch shift keeps
    the mode, so a minor (A) source can only land on A codes and a major (B)
    source only on B codes.
  * When no compatible code is within 3 semitones the shift is 0.

Expected values below are derived by hand from the Camelot wheel
(8B = C major, 8A = A minor, each step around the wheel is a fifth, +7 or -5
semitones; the relative minor of a B code sits 3 semitones below its major).
"""

from __future__ import annotations

import pytest

from scripts.core.key_detection import (
    CAMELOT,
    NOTE_NAMES,
    camelot_compatible,
    pitch_shift_for_camelot,
)

# ── Compatible pairs: never shift ────────────────────────────────────────────

class TestCompatiblePairsDoNotShift:

    def test_same_code_is_zero(self):
        assert pitch_shift_for_camelot("8A", "8A") == 0

    @pytest.mark.parametrize("source,target", [
        ("8A", "9A"),    # spec case: adjacent number, same letter
        ("9A", "8A"),
        ("8A", "7A"),
        ("12A", "1A"),   # wrap around the wheel
        ("1B", "12B"),
        ("8B", "9B"),
    ])
    def test_adjacent_number_same_letter_is_zero(self, source, target):
        assert pitch_shift_for_camelot(source, target) == 0, (
            f"{source} and {target} are adjacent on the wheel; no shift expected"
        )

    @pytest.mark.parametrize("source,target", [
        ("8A", "8B"),    # spec case: relative A/B pair
        ("8B", "8A"),
        ("3A", "3B"),
    ])
    def test_relative_major_minor_is_zero(self, source, target):
        assert pitch_shift_for_camelot(source, target) == 0, (
            f"{source} and {target} are the relative pair; no shift expected"
        )


# ── Incompatible pairs: smallest shift (at most 3) onto a compatible code ────

class TestIncompatiblePairsShiftToNearestCompatible:

    def test_11A_source_against_8A_target_shifts_down_two(self):
        """F# minor (11A) against A minor (8A).

        Compatible A codes for 8A: 7A (D minor), 8A (A minor), 9A (E minor).
        From F# (6): to E (4) is -2, to A (9) is +3, to D (2) is -4.
        Smallest within 3 is -2, landing on 9A.
        """
        assert pitch_shift_for_camelot("11A", "8A") == -2

    def test_8A_source_against_11A_target_shifts_up_two(self):
        """A minor (8A) against F# minor (11A).

        Compatible A codes for 11A: 10A (B minor), 11A (F# minor), 12A (C# minor).
        From A (9): to B (11) is +2, to F# (6) is -3, to C# (1) is +4.
        Smallest within 3 is +2, landing on 10A.
        """
        assert pitch_shift_for_camelot("8A", "11A") == 2

    def test_minor_source_against_major_target_lands_on_relative_minor(self):
        """A minor (8A) against A major (11B).

        A pitch shift keeps the mode, so the only compatible landing code is
        the relative minor 11A (F# minor): A (9) to F# (6) is -3.
        """
        assert pitch_shift_for_camelot("8A", "11B") == -3

    def test_minor_source_against_major_target_two_up(self):
        """A minor (8A) against D major (10B): relative minor is 10A (B minor),
        A (9) to B (11) is +2."""
        assert pitch_shift_for_camelot("8A", "10B") == 2

    @pytest.mark.parametrize("source,target", [
        ("8A", "9B"),    # only compatible A code is 9A (E minor): A to E is -5
        ("8A", "2B"),    # only compatible A code is 2A (D# minor): A to D# is 6
        ("8B", "2A"),    # only compatible B code is 2B (F# major): C to F# is 6
    ])
    def test_no_compatible_code_within_three_semitones_is_zero(self, source, target):
        assert pitch_shift_for_camelot(source, target) == 0, (
            f"{source} against {target}: nothing compatible within 3 semitones, "
            "expected 0"
        )

    def test_invalid_code_is_zero(self):
        assert pitch_shift_for_camelot("XX", "8A") == 0


# ── Wheel wide invariants derived from the spec ──────────────────────────────

def _root_semitone(code: str) -> int:
    for key_name, camelot in CAMELOT.items():
        if camelot == code:
            return NOTE_NAMES.index(key_name.split()[0])
    raise KeyError(code)


def _landing_code(source: str, shift: int) -> str:
    """Code the source lands on after `shift` semitones (mode preserved)."""
    mode = "minor" if source.endswith("A") else "major"
    root = NOTE_NAMES[(_root_semitone(source) + shift) % 12]
    return CAMELOT[f"{root} {mode}"]


ALL_CODES = sorted(CAMELOT.values(), key=lambda c: (c[-1], int(c[:-1])))
ALL_PAIRS = [(s, t) for s in ALL_CODES for t in ALL_CODES]


@pytest.mark.parametrize("source,target", ALL_PAIRS)
def test_every_pair_obeys_the_rule(source: str, target: str):
    shift = pitch_shift_for_camelot(source, target)
    assert -3 <= shift <= 3, f"{source}->{target}: {shift} exceeds 3 semitones"

    if camelot_compatible(source, target):
        assert shift == 0, f"{source}->{target} already compatible, got {shift}"
        return

    candidates = sorted(
        (s for s in range(-3, 4) if camelot_compatible(_landing_code(source, s), target)),
        key=abs,
    )
    if not candidates:
        assert shift == 0, f"{source}->{target}: nothing within 3, got {shift}"
        return

    assert shift != 0, f"{source}->{target}: compatible code reachable, got 0"
    assert camelot_compatible(_landing_code(source, shift), target), (
        f"{source}->{target}: shift {shift} lands on {_landing_code(source, shift)}, "
        "which is not compatible with the target"
    )
    assert abs(shift) == abs(candidates[0]), (
        f"{source}->{target}: shift {shift} is not the smallest; "
        f"{candidates[0]} also reaches a compatible code"
    )
