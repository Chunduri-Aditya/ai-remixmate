"""
tests/test_remix_recipe.py — Tests for scripts.core.remix_recipe.

remix_recipe.generate_remix_recipe() is a pure presentation layer over an
already-computed TransitionPlan: it does no audio analysis and no scoring of
its own, so these tests construct SongStructure/TransitionPlan/EQPlan
directly rather than running the full analysis pipeline.
"""

from __future__ import annotations

from scripts.core.dj_analysis import EQPlan, SongStructure, TransitionPlan
from scripts.core.remix_recipe import generate_remix_recipe


def _song(bpm=128.0, vocal_density=0.2) -> SongStructure:
    return SongStructure(bpm=bpm, duration=240.0, vocal_density=vocal_density)


def _plan(
    harmonic_score=0.8,
    tiv_compatibility=None,
    tempo_shift_ratio=1.0,
    suggested_pitch_shift=0.0,
    bpm_a=128.0,
    bpm_b=128.0,
    transition_bars=16,
    exit_time_a=100.0,
) -> TransitionPlan:
    bar_duration = (60.0 / bpm_a) * 4
    transition_seconds = transition_bars * bar_duration
    eq = EQPlan(
        hp_start_hz=400.0,
        hp_end_hz=80.0,
        hp_ramp_bars=transition_bars // 2,
        bass_swap_bar=transition_bars // 2,
        bass_crossover_hz=150.0,
        a_fade_start_bar=0,
        a_fade_end_bar=transition_bars,
        b_fade_start_bar=0,
        b_fade_end_bar=transition_bars,
    )
    return TransitionPlan(
        exit_bar_a=64,
        exit_time_a=exit_time_a,
        entry_bar_b=0,
        entry_time_b=0.0,
        transition_bars=transition_bars,
        transition_seconds=transition_seconds,
        bpm_a=bpm_a,
        bpm_b=bpm_b,
        tempo_shift_ratio=tempo_shift_ratio,
        eq=eq,
        harmonic_score=harmonic_score,
        tiv_compatibility=tiv_compatibility,
        suggested_pitch_shift=suggested_pitch_shift,
    )


class TestMethodSelection:
    def test_high_quality_gets_phrase_aligned_blend(self):
        plan = _plan(harmonic_score=0.9)
        recipe = generate_remix_recipe("Song A", "Song B", _song(), _song(), plan)
        assert recipe.method == "phrase_aligned_blend"

    def test_low_quality_gets_short_controlled_blend(self):
        plan = _plan(harmonic_score=0.2)
        recipe = generate_remix_recipe("Song A", "Song B", _song(), _song(), plan)
        assert recipe.method == "short_controlled_blend"

    def test_tiv_compatibility_takes_priority_over_harmonic_score(self):
        # harmonic_score alone would select phrase_aligned_blend; tiv says no.
        plan = _plan(harmonic_score=0.9, tiv_compatibility=0.1)
        recipe = generate_remix_recipe("Song A", "Song B", _song(), _song(), plan)
        assert recipe.method == "short_controlled_blend"

    def test_unknown_harmonic_score_defaults_to_cautious_method(self):
        # harmonic_score sentinel (-1.0 = "not computed") must not be treated
        # as a real low score, but the method choice still defaults cautious.
        plan = _plan(harmonic_score=-1.0)
        recipe = generate_remix_recipe("Song A", "Song B", _song(), _song(), plan)
        assert recipe.method == "short_controlled_blend"
        assert not any("Harmonic compatibility is low" in w for w in recipe.risk_warnings)


class TestWarnings:
    def test_low_quality_produces_harmonic_warning(self):
        plan = _plan(harmonic_score=0.2)
        recipe = generate_remix_recipe("Song A", "Song B", _song(), _song(), plan)
        assert any("Harmonic compatibility is low" in w for w in recipe.risk_warnings)

    def test_high_quality_has_no_harmonic_warning(self):
        plan = _plan(harmonic_score=0.9)
        recipe = generate_remix_recipe("Song A", "Song B", _song(), _song(), plan)
        assert not any("Harmonic compatibility is low" in w for w in recipe.risk_warnings)

    def test_large_tempo_shift_warns(self):
        plan = _plan(tempo_shift_ratio=1.15)  # 15% shift
        recipe = generate_remix_recipe("Song A", "Song B", _song(), _song(), plan)
        assert any("Tempo shift" in w for w in recipe.risk_warnings)

    def test_small_tempo_shift_does_not_warn(self):
        plan = _plan(tempo_shift_ratio=1.02)  # 2% shift
        recipe = generate_remix_recipe("Song A", "Song B", _song(), _song(), plan)
        assert not any("Tempo shift" in w for w in recipe.risk_warnings)

    def test_vocal_clash_warns(self):
        plan = _plan()
        recipe = generate_remix_recipe(
            "Song A", "Song B", _song(vocal_density=0.8), _song(vocal_density=0.8), plan
        )
        assert any("significant vocal presence" in w for w in recipe.risk_warnings)

    def test_no_vocal_clash_does_not_warn(self):
        plan = _plan()
        recipe = generate_remix_recipe(
            "Song A", "Song B", _song(vocal_density=0.1), _song(vocal_density=0.1), plan
        )
        assert not any("vocal presence" in w for w in recipe.risk_warnings)

    def test_pitch_shift_suggestion_warns(self):
        plan = _plan(suggested_pitch_shift=2.0)
        recipe = generate_remix_recipe("Song A", "Song B", _song(), _song(), plan)
        assert any("pitch-shifting" in w for w in recipe.risk_warnings)

    def test_zero_pitch_shift_does_not_warn(self):
        plan = _plan(suggested_pitch_shift=0.0)
        recipe = generate_remix_recipe("Song A", "Song B", _song(), _song(), plan)
        assert not any("pitch-shifting" in w for w in recipe.risk_warnings)


class TestSteps:
    def test_generates_five_ordered_steps(self):
        plan = _plan()
        recipe = generate_remix_recipe("Song A", "Song B", _song(), _song(), plan)
        assert [step.order for step in recipe.steps] == [1, 2, 3, 4, 5]

    def test_bass_swap_time_matches_eq_plan(self):
        plan = _plan(bpm_a=120.0, exit_time_a=50.0, transition_bars=16)
        recipe = generate_remix_recipe("Song A", "Song B", _song(bpm=120.0), _song(), plan)
        bar_duration = (60.0 / 120.0) * 4
        expected = 50.0 + (16 // 2) * bar_duration
        swap_step = recipe.steps[2]  # "Swap bass"
        assert swap_step.title == "Swap bass"
        assert swap_step.at_time_sec == expected

    def test_vocal_clash_reflected_in_protect_vocals_step(self):
        plan = _plan()
        recipe = generate_remix_recipe(
            "Song A", "Song B", _song(vocal_density=0.9), _song(vocal_density=0.9), plan
        )
        protect_step = recipe.steps[3]
        assert protect_step.title == "Protect vocals"
        assert "Mute or delay" in protect_step.instruction

    def test_no_vocal_clash_reflected_in_protect_vocals_step(self):
        plan = _plan()
        recipe = generate_remix_recipe(
            "Song A", "Song B", _song(vocal_density=0.05), _song(vocal_density=0.05), plan
        )
        protect_step = recipe.steps[3]
        assert "no action needed" in protect_step.instruction


class TestTimingSuggestion:
    def test_timing_suggestion_mentions_both_track_names_and_window(self):
        plan = _plan(exit_time_a=42.0, transition_bars=16)
        recipe = generate_remix_recipe("Alpha", "Beta", _song(), _song(), plan)
        assert "Alpha" in recipe.timing_suggestion
        assert "Beta" in recipe.timing_suggestion
        assert "42.00s" in recipe.timing_suggestion
        assert "16-bar" in recipe.timing_suggestion


class TestStemSuggestions:
    def test_delays_vocals_when_clash_risk_high(self):
        plan = _plan()
        recipe = generate_remix_recipe(
            "Song A", "Song B", _song(vocal_density=0.9), _song(vocal_density=0.9), plan
        )
        assert any("Delay Song B's vocals" in s for s in recipe.stem_suggestions)

    def test_allows_early_vocals_when_clash_risk_low(self):
        plan = _plan()
        recipe = generate_remix_recipe(
            "Song A", "Song B", _song(vocal_density=0.1), _song(vocal_density=0.1), plan
        )
        assert any("can enter as soon as" in s for s in recipe.stem_suggestions)


def test_str_repr_includes_method_and_counts():
    plan = _plan(harmonic_score=0.9)
    recipe = generate_remix_recipe("Song A", "Song B", _song(), _song(), plan)
    text = str(recipe)
    assert "Song A" in text and "Song B" in text
    assert recipe.method in text
