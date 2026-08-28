"""
tests/test_classic_engine.py — the vendored "classic" mixing engine + selection.

Covers three things:
  1. scripts/core/classic_engine/ exposes the full render-path API the remix
     task module consumes (structures, plan fields, QualityReport fields).
  2. task_modules/remix._resolve_engine() honours dj.mixing_engine — classic
     by default, current on request, classic again on garbage values.
  3. The classic path actually renders audio end-to-end on synthetic tracks
     (marked dj_analysis — needs librosa, auto-skips via conftest probe).
"""

import numpy as np
import pytest

SR = 44100


def _make_audio(duration: float = 5.0, bpm: float = 128.0,
                freq: float = 440.0, sr: int = SR) -> np.ndarray:
    """Sine wave with BPM-aligned transients — same recipe as test_core_modules."""
    rng = np.random.default_rng(seed=42)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    envelope = np.zeros_like(t)
    beat_samples = int(sr / (bpm / 60.0))
    for onset in range(0, len(t), beat_samples):
        decay_len = min(beat_samples, len(t) - onset)
        envelope[onset:onset + decay_len] += np.exp(-np.linspace(0, 6, decay_len))
    audio = (np.sin(2 * np.pi * freq * t) * 0.7 + rng.normal(0, 0.05, len(t)))
    audio = (audio * envelope).astype(np.float32)
    audio /= max(np.abs(audio).max(), 1e-6)
    return audio


# ---------------------------------------------------------------------------
# 1. Package surface
# ---------------------------------------------------------------------------

class TestClassicPackageSurface:
    def test_exports(self):
        from scripts.core.classic_engine import (  # noqa: F401
            CLASSIC_DJ_TARGET_LUFS,
            DJEngine,
            QualityReport,
            SongStructure,
            TransitionPlan,
            _analyze_impl,
            analyze_structure,
            compute_lufs,
            master_mix,
            plan_transition,
        )
        assert CLASSIC_DJ_TARGET_LUFS == -8.0
        assert _analyze_impl is analyze_structure

    def test_classic_modules_are_the_vendored_copies(self):
        import inspect

        from scripts.core.classic_engine import DJEngine, master_mix, plan_transition
        assert "classic_engine" in inspect.getfile(DJEngine)
        assert "classic_engine" in inspect.getfile(master_mix)
        assert "classic_engine" in inspect.getfile(plan_transition)

    def test_transition_plan_has_fields_the_api_reads(self):
        """task_dj_remix/chain/preview result payloads read these plan fields."""
        import dataclasses

        from scripts.core.classic_engine import TransitionPlan
        names = {f.name for f in dataclasses.fields(TransitionPlan)}
        assert {"exit_bar_a", "entry_bar_b", "transition_bars", "transition_seconds",
                "bpm_a", "bpm_b", "tempo_shift_ratio", "harmonic_score"} <= names

    def test_quality_report_has_fields_the_api_reads(self):
        import dataclasses

        from scripts.core.classic_engine import QualityReport
        names = {f.name for f in dataclasses.fields(QualityReport)}
        assert {"lufs_integrated", "peak_dbfs", "dynamic_range_db",
                "passed", "notes"} <= names

    def test_song_structure_has_fields_the_api_reads(self):
        """Result payloads read key_name/mode/camelot off the structures."""
        from scripts.core.classic_engine import SongStructure
        s = SongStructure(bpm=120.0, duration=30.0)
        assert s.key_name == "" and s.mode == "" and s.camelot == ""

    def test_render_signatures_match_task_module_calls(self):
        """The task module passes these kwargs — a rename here breaks renders."""
        import inspect

        from scripts.core.classic_engine import DJEngine
        render_params = set(inspect.signature(DJEngine.render).parameters)
        assert {"full_output", "bridge_beat", "bridge_gain",
                "transition_effect"} <= render_params
        blend_params = set(inspect.signature(DJEngine.render_stem_blend).parameters)
        assert {"stems_dir_a", "stems_dir_b", "full_output", "bridge_beat",
                "bridge_gain", "transition_effect"} <= blend_params
        chain_params = set(inspect.signature(DJEngine.render_chain).parameters)
        assert {"tracks", "plans", "bridge_beats", "bridge_gain",
                "transition_effect"} <= chain_params


# ---------------------------------------------------------------------------
# 2. Engine selection (dj.mixing_engine)
# ---------------------------------------------------------------------------

class TestEngineSelection:
    # These pin dj.mixing_engine rather than reading the ambient config, the
    # same way test_current_selects_live_engine below does. They guard how the
    # value RESOLVES, which is the invariant; which value config.yaml happens
    # to ship is a product decision that has changed once already.
    def test_classic_selects_the_frozen_engine(self, monkeypatch):
        from scripts.api.task_modules.remix import _resolve_engine
        from scripts.core.config import cfg
        monkeypatch.setattr(cfg.dj, "mixing_engine", "classic")
        eng = _resolve_engine()
        assert eng.name == "classic"
        assert eng.dj_target_lufs == -8.0

    def test_classic_components_come_from_classic_package(self, monkeypatch):
        import inspect

        from scripts.api.task_modules.remix import _resolve_engine
        from scripts.core.config import cfg
        monkeypatch.setattr(cfg.dj, "mixing_engine", "classic")
        eng = _resolve_engine()
        assert "classic_engine" in inspect.getfile(eng.DJEngine)
        assert "classic_engine" in inspect.getmodule(eng.master_mix).__file__
        assert "classic_engine" in inspect.getmodule(eng.analyze).__file__

    def test_current_selects_live_engine(self, monkeypatch):
        import inspect

        from scripts.api.task_modules.remix import _resolve_engine
        from scripts.core.config import cfg
        monkeypatch.setattr(cfg.dj, "mixing_engine", "current")
        eng = _resolve_engine()
        assert eng.name == "current"
        assert "classic_engine" not in inspect.getfile(eng.DJEngine)
        assert "classic_engine" not in inspect.getmodule(eng.master_mix).__file__

    def test_garbage_value_falls_back_to_classic(self, monkeypatch):
        from scripts.api.task_modules.remix import _resolve_engine
        from scripts.core.config import cfg
        monkeypatch.setattr(cfg.dj, "mixing_engine", "quantum")
        assert _resolve_engine().name == "classic"


# ---------------------------------------------------------------------------
# 3. End-to-end classic render on synthetic audio
# ---------------------------------------------------------------------------

@pytest.mark.dj_analysis
class TestClassicRender:
    def test_analyze_plan_render_roundtrip(self):
        from scripts.core.classic_engine import DJEngine, analyze_structure, plan_transition
        audio_a = _make_audio(duration=30.0, bpm=126.0, freq=440.0)
        audio_b = _make_audio(duration=30.0, bpm=130.0, freq=330.0)

        struct_a = analyze_structure(audio_a, SR)
        struct_b = analyze_structure(audio_b, SR)
        assert struct_a.bpm > 0 and struct_b.bpm > 0

        plan = plan_transition(struct_a, struct_b, transition_bars=8)
        assert plan.transition_bars > 0
        assert plan.tempo_shift_ratio > 0

        mix = DJEngine(sr=SR).render(audio_a, audio_b, plan, full_output=False)
        assert len(mix) > 0
        assert np.all(np.isfinite(mix))
        assert float(np.abs(mix).max()) > 1e-4  # audibly non-silent

    def test_classic_master_mix_hits_club_target(self):
        from scripts.core.classic_engine import compute_lufs, master_mix
        # Plain sine — low crest factor, so the −8 LUFS gain stage lands
        # without the true-peak limiter clamping it back down (a click train
        # would be limiter-crushed ~7 dB below target, by design).
        t = np.linspace(0, 10.0, int(SR * 10.0), endpoint=False)
        audio = (0.05 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        mastered, report = master_mix(audio, sr=SR, target_lufs=-8.0)
        assert report.lufs_target == -8.0
        lufs = compute_lufs(mastered, SR)
        assert abs(lufs - (-8.0)) < 1.5
        assert float(np.abs(mastered).max()) <= 1.0
