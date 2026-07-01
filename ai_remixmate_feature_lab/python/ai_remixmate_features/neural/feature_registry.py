from __future__ import annotations

from .core import NeuralFeatureSpec, TinyOnlineMLP
from .vectorizers import PAIR_VECTOR_SIZE, TRACK_VECTOR_SIZE


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
}


def feature_names() -> list[str]:
    return sorted(FEATURE_MODEL_SPECS)


def create_model_registry() -> dict[str, TinyOnlineMLP]:
    return {name: TinyOnlineMLP(spec) for name, spec in FEATURE_MODEL_SPECS.items()}
