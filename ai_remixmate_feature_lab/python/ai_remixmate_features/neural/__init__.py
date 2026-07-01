"""Dependency-free online neural learning for the feature lab."""

from .core import NeuralFeatureSpec, TinyOnlineMLP, TrainingEvent
from .feature_registry import FEATURE_MODEL_SPECS, create_model_registry, feature_names
from .online_learning import LearningController
from .vectorizers import build_feature_input, pair_feature_vector, track_feature_vector

__all__ = [
    "FEATURE_MODEL_SPECS",
    "LearningController",
    "NeuralFeatureSpec",
    "TinyOnlineMLP",
    "TrainingEvent",
    "build_feature_input",
    "create_model_registry",
    "feature_names",
    "pair_feature_vector",
    "track_feature_vector",
]
