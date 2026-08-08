from __future__ import annotations

from .core import NeuralFeatureSpec, TinyOnlineMLP
from .vectorizers import PAIR_VECTOR_SIZE, TRACK_VECTOR_SIZE

# Matches scripts/core/spectral_matching.py's N_BANDS constant (24 mel bands
# in a transition profile) — redefined independently here rather than
# imported, per this lab's isolation rule (README.md: "Nothing here imports
# from scripts/"). See ai_remixmate_features/research/spectral_entry_match.py
# (DJ_ENGINE_IMPROVEMENT_PLAN.md §5.1) for the prototype that uses this spec.
SPECTRAL_SIMS_SIZE = 24


FEATURE_MODEL_SPECS: dict[str, NeuralFeatureSpec] = {
    "bpm_compatibility": NeuralFeatureSpec(
        name="bpm_compatibility",
        input_size=PAIR_VECTOR_SIZE,
        description="Learns tempo blend quality from user feedback and manual corrections.",
    ),
    "key_compatibility": NeuralFeatureSpec(
        name="key_compatibility",
        input_size=PAIR_VECTOR_SIZE,
        description="Learns harmonic acceptability beyond simplified Camelot rules.",
    ),
    "energy_compatibility": NeuralFeatureSpec(
        name="energy_compatibility",
        input_size=PAIR_VECTOR_SIZE,
        description="Learns user preference for energy-flow similarity or contrast.",
    ),
    "timbre_compatibility": NeuralFeatureSpec(
        name="timbre_compatibility",
        input_size=PAIR_VECTOR_SIZE,
        description="Learns perceived sonic texture fit.",
    ),
    "vocal_clash_risk": NeuralFeatureSpec(
        name="vocal_clash_risk",
        input_size=PAIR_VECTOR_SIZE,
        description="Learns vocal conflict risk from render reviews.",
    ),
    "compatibility_score": NeuralFeatureSpec(
        name="compatibility_score",
        input_size=PAIR_VECTOR_SIZE,
        hidden_size=16,
        description="Learns overall transition approval as a neural adjustment layer.",
    ),
    "transition_planning": NeuralFeatureSpec(
        name="transition_planning",
        input_size=PAIR_VECTOR_SIZE,
        hidden_size=16,
        description="Learns whether generated entry and exit timing was accepted.",
    ),
    "remix_recipe_quality": NeuralFeatureSpec(
        name="remix_recipe_quality",
        input_size=PAIR_VECTOR_SIZE,
        hidden_size=16,
        description="Learns whether generated beginner-readable recipe steps were useful.",
    ),
    "automix_next_track": NeuralFeatureSpec(
        name="automix_next_track",
        input_size=PAIR_VECTOR_SIZE,
        hidden_size=16,
        description="Learns next-track ordering confidence from set-builder feedback.",
    ),
    "track_match": NeuralFeatureSpec(
        name="track_match",
        input_size=PAIR_VECTOR_SIZE,
        hidden_size=16,
        description="Learns recommendation relevance from click, skip, and queue actions.",
    ),
    "beatgrid_confidence": NeuralFeatureSpec(
        name="beatgrid_confidence",
        input_size=TRACK_VECTOR_SIZE,
        description="Learns whether an analyzed beatgrid needs manual correction.",
    ),
    "stem_quality": NeuralFeatureSpec(
        name="stem_quality",
        input_size=TRACK_VECTOR_SIZE,
        description="Learns whether available stems are clean enough for stem-aware remixing.",
    ),
    "waveform_interest": NeuralFeatureSpec(
        name="waveform_interest",
        input_size=TRACK_VECTOR_SIZE,
        description="Learns which track regions tend to become cues, loops, or transition points.",
    ),
    "spectral_entry_match": NeuralFeatureSpec(
        name="spectral_entry_match",
        input_size=SPECTRAL_SIMS_SIZE,
        hidden_size=12,
        description=(
            "Research prototype (DJ_ENGINE_IMPROVEMENT_PLAN.md §5.1): learns to score a "
            "B-entry candidate's per-mel-band trajectory similarity to A's outro, as an "
            "alternative to scripts/core/spectral_matching.py's hand-tuned linear weights. "
            "Not wired to any render path — see reports/spectral_entry_match_prototype.md."
        ),
    ),
}


def feature_names() -> list[str]:
    return sorted(FEATURE_MODEL_SPECS)


def create_model_registry() -> dict[str, TinyOnlineMLP]:
    return {name: TinyOnlineMLP(spec) for name, spec in FEATURE_MODEL_SPECS.items()}
