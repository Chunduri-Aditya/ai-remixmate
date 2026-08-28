"""
scripts/core/classic_engine — the vendored AI_REMIXMATE_BEST_MIXING_ALGO mix path.

This package is a frozen copy of the April 2026 snapshot's mixing logic
(analysis → transition planning → rendering → mastering), kept byte-for-byte
faithful so renders sound exactly like the originals. It is selected at render
time via config `dj.mixing_engine: classic` and consumed only by
scripts/api/task_modules/remix.py; everything else in the app (analysis
endpoints, setlist planner, Library Atlas metadata) still uses the current
engine modules.

Its only imports from the live codebase are scripts.core.{config,gpu,
music_intelligence} — the two music_intelligence functions it calls
(compute_track_vector, camelot_harmonic_score) are byte-identical to the
snapshot's, verified at vendoring time.
"""

from scripts.core.classic_engine.dj_analysis import (
    Beat,
    EQPlan,
    Section,
    SongStructure,
    TransitionPlan,
    analyze_structure,
    plan_transition,
)
from scripts.core.classic_engine.dj_engine import DJEngine, _analyze_impl
from scripts.core.classic_engine.mastering import QualityReport, compute_lufs, master_mix

# The classic pipeline mastered DJ mixes to −8 LUFS ("club loud") — part of
# the sound the snapshot's reference renders were judged on.
CLASSIC_DJ_TARGET_LUFS = -8.0

__all__ = [
    "Beat",
    "EQPlan",
    "Section",
    "SongStructure",
    "TransitionPlan",
    "analyze_structure",
    "plan_transition",
    "DJEngine",
    "_analyze_impl",
    "QualityReport",
    "compute_lufs",
    "master_mix",
    "CLASSIC_DJ_TARGET_LUFS",
]
