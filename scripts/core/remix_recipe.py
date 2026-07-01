"""
scripts/core/remix_recipe.py — Human-readable remix recipe generator.

Turns an already-computed TransitionPlan (scripts.core.dj_analysis.plan_transition)
into beginner-facing, step-by-step mixing instructions. This is a pure
presentation layer over existing analysis — it does not recompute
compatibility, timing, or harmonic scoring, and does not duplicate
track_metadata.compatibility_score() or dj_analysis.plan_transition().

Background: added 2026-06-30 after auditing ai_remixmate_feature_lab/, an
isolated Codex-built sandbox. Everything else in that lab (compatibility
scoring, transition planning, automix ordering, track ranking) duplicated
already-better-calibrated production logic (the SetFlow formula in
track_metadata.py, psychoacoustic consonance + TIV scoring in
dj_analysis.py/tiv_scoring.py, and the Markov/energy-arc optimizer in
setlist_planner.py) and was intentionally not ported. A plain-language
recipe generator was the one piece with no existing production equivalent,
so only that concept was rebuilt here — from scratch, on top of the real
TransitionPlan output, not copied from the lab's weaker version.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from scripts.core.dj_analysis import SongStructure, TransitionPlan

log = logging.getLogger(__name__)

# Harmonic quality (TIV if available, else psychoacoustic consonance) at or
# above this value is considered safe for a full phrase-aligned blend.
# Below it, recommend a shorter, more controlled blend instead.
_PHRASE_BLEND_QUALITY_THRESHOLD = 0.55

# vocal_density_a * vocal_density_b above this is treated as meaningful
# vocal-clash risk (mirrors the qualitative threshold used elsewhere in the
# project for "both tracks carry strong simultaneous vocal presence").
_VOCAL_CLASH_THRESHOLD = 0.35

# Tempo shift beyond this percentage is flagged as large enough that
# time-stretch artifacts may become audible.
_LARGE_TEMPO_SHIFT_PERCENT = 8.0


@dataclass
class RecipeStep:
    """One step in a beginner-readable remix recipe."""
    order: int
    title: str
    instruction: str
    at_time_sec: Optional[float] = None


@dataclass
class RemixRecipe:
    """Beginner-readable, step-by-step instructions for one transition."""
    song_a: str
    song_b: str
    method: str  # "phrase_aligned_blend" | "short_controlled_blend"
    stem_suggestions: List[str] = field(default_factory=list)
    timing_suggestion: str = ""
    risk_warnings: List[str] = field(default_factory=list)
    steps: List[RecipeStep] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"RemixRecipe: {self.song_a} → {self.song_b} "
            f"({self.method}, {len(self.steps)} steps, {len(self.risk_warnings)} warnings)"
        )


def generate_remix_recipe(
    song_a_name: str,
    song_b_name: str,
    song_a: SongStructure,
    song_b: SongStructure,
    plan: TransitionPlan,
) -> RemixRecipe:
    """
    Build a beginner-readable remix recipe from an already-computed TransitionPlan.

    Reads the harmonic score, tempo ratio, EQ plan, and vocal density already
    produced by plan_transition() / analyze_structure() and turns them into
    plain-language steps and warnings. Does no new analysis of its own.

    Parameters
    ----------
    song_a_name, song_b_name : str
        Display names for the outgoing/incoming tracks (SongStructure itself
        carries no track identity — see dj_analysis.py).
    song_a, song_b : SongStructure
        Analysed structures for the outgoing/incoming tracks.
    plan : TransitionPlan
        Output of scripts.core.dj_analysis.plan_transition(song_a, song_b).
    """
    # Prefer the continuous TIV harmonic score when chroma was available;
    # fall back to psychoacoustic consonance. harmonic_score defaults to
    # -1.0 (sentinel for "not computed") rather than a real score.
    raw_quality = plan.tiv_compatibility if plan.tiv_compatibility is not None else plan.harmonic_score
    quality_known = raw_quality is not None and raw_quality >= 0.0
    quality = raw_quality if quality_known else 0.5  # neutral default, cautious method choice

    method = (
        "phrase_aligned_blend"
        if quality >= _PHRASE_BLEND_QUALITY_THRESHOLD
        else "short_controlled_blend"
    )

    tempo_shift_percent = abs(plan.tempo_shift_ratio - 1.0) * 100.0
    vocal_clash_risk = float(song_a.vocal_density) * float(song_b.vocal_density)

    warnings: List[str] = []
    if quality_known and quality < _PHRASE_BLEND_QUALITY_THRESHOLD:
        warnings.append(
            f"Harmonic compatibility is low ({quality:.2f}); consider a shorter blend or a key change."
        )
    if tempo_shift_percent > _LARGE_TEMPO_SHIFT_PERCENT:
        warnings.append(
            f"Tempo shift is {tempo_shift_percent:.1f}% — time-stretching artifacts may be audible."
        )
    if vocal_clash_risk > _VOCAL_CLASH_THRESHOLD:
        warnings.append(
            "Both tracks have significant vocal presence — mute or delay one vocal stem during the overlap."
        )
    if plan.suggested_pitch_shift:
        warnings.append(
            f"Consider pitch-shifting {song_b_name} by {plan.suggested_pitch_shift:+.1f} semitones "
            f"to match {song_a_name}'s key."
        )

    bar_duration_a = (60.0 / plan.bpm_a) * 4 if plan.bpm_a > 0 else 0.0
    bass_swap_time = plan.exit_time_a + plan.eq.bass_swap_bar * bar_duration_a
    transition_end_time = plan.exit_time_a + plan.transition_seconds

    vocals_clash = vocal_clash_risk > _VOCAL_CLASH_THRESHOLD

    stem_suggestions = [
        f"Bring {song_b_name}'s drums and bass in early — they carry the transition.",
        (
            f"Delay {song_b_name}'s vocals until after the bass swap."
            if vocals_clash
            else f"{song_b_name}'s vocals can enter as soon as the rhythm is locked in."
        ),
        f"Swap bass ownership to {song_b_name} around {bass_swap_time:.1f}s.",
    ]

    timing_suggestion = (
        f"Start {song_a_name}'s fade at {plan.exit_time_a:.2f}s and land {song_b_name}'s cue at "
        f"{plan.entry_time_b:.2f}s, over a {plan.transition_bars}-bar "
        f"({plan.transition_seconds:.1f}s) window."
    )

    steps = [
        RecipeStep(
            1, "Prepare cues",
            f"Verify {song_a_name}'s exit and {song_b_name}'s entry both land on downbeats.",
            plan.exit_time_a,
        ),
        RecipeStep(
            2, "Open rhythm",
            f"Bring {song_b_name}'s drums up while its bass stays controlled.",
            plan.exit_time_a,
        ),
        RecipeStep(
            3, "Swap bass",
            f"Lower {song_a_name}'s low EQ as {song_b_name}'s bass becomes dominant.",
            bass_swap_time,
        ),
        RecipeStep(
            4, "Protect vocals",
            (
                "Mute or delay one vocal stem — both tracks carry strong vocals in this window."
                if vocals_clash
                else "Vocals are unlikely to clash here — no action needed."
            ),
            None,
        ),
        RecipeStep(
            5, "Resolve",
            f"Finish {song_a_name}'s fade and reset EQ to neutral by {transition_end_time:.2f}s.",
            transition_end_time,
        ),
    ]

    log.info(
        "Generated remix recipe: %s → %s (%s, %d warning(s))",
        song_a_name, song_b_name, method, len(warnings),
    )

    return RemixRecipe(
        song_a=song_a_name,
        song_b=song_b_name,
        method=method,
        stem_suggestions=stem_suggestions,
        timing_suggestion=timing_suggestion,
        risk_warnings=warnings,
        steps=steps,
    )
