from __future__ import annotations

from ai_remixmate_features.neural import (
    FEATURE_MODEL_SPECS,
    LearningController,
    TrainingEvent,
    build_feature_input,
    feature_names,
    pair_feature_vector,
    track_feature_vector,
)
from ai_remixmate_features.neural.vectorizers import PAIR_VECTOR_SIZE, TRACK_VECTOR_SIZE


def make_track(track_id: str = "a", bpm: float = 128.0, camelot: str = "8A") -> dict:
    return {
        "id": track_id,
        "title": track_id.upper(),
        "duration_sec": 240.0,
        "bpm": bpm,
        "camelot": camelot,
        "beatgrid": {
            "bpm": bpm,
            "beatTimes": [0, 0.5, 1.0, 1.5, 2.0],
            "downbeats": [0, 2.0],
            "beatsPerBar": 4,
            "confidence": 0.9,
        },
        "energy_curve": [0.2, 0.4, 0.7],
        "timbre_vector": [1.0, 0.0, 0.0],
        "vocal_activity": [0.0, 0.1, 0.2],
        "stemManifest": {
            "trackId": track_id,
            "stems": [
                {"type": "vocals", "path": "vocals.wav", "available": True},
                {"type": "drums", "path": "drums.wav", "available": True},
                {"type": "bass", "path": "bass.wav", "available": True},
                {"type": "other", "path": "other.wav", "available": False},
            ],
        },
    }


def test_neural_registry_covers_feature_models():
    names = feature_names()
    assert "compatibility_score" in names
    assert "transition_planning" in names
    assert "beatgrid_confidence" in names
    assert len(names) >= 13
    assert set(names) == set(FEATURE_MODEL_SPECS)


def test_neural_vectorizers_match_model_dimensions():
    a = make_track("a", 128, "8A")
    b = make_track("b", 64, "8B")
    assert len(track_feature_vector(a)) == TRACK_VECTOR_SIZE
    assert len(pair_feature_vector(a, b)) == PAIR_VECTOR_SIZE
    assert len(build_feature_input("beatgrid_confidence", a)) == TRACK_VECTOR_SIZE
    assert len(build_feature_input("compatibility_score", a, b)) == PAIR_VECTOR_SIZE


def test_online_learning_moves_prediction_toward_target():
    controller = LearningController()
    vector = build_feature_input("compatibility_score", make_track("a"), make_track("b", 64, "8B"))
    before = controller.predict("compatibility_score", vector)[0]
    for index in range(40):
        controller.learn(
            TrainingEvent(
                id=f"evt-{index}",
                feature_name="compatibility_score",
                input_vector=vector,
                target_vector=[1.0],
                source="test_fixture",
            )
        )
    after = controller.predict("compatibility_score", vector)[0]
    assert 0 <= before <= 1
    assert 0 <= after <= 1
    assert after > before


def test_learning_controller_persists_json_state(tmp_path):
    controller = LearningController()
    vector = build_feature_input("track_match", make_track("a"), make_track("b", 130, "9A"))
    controller.learn_score("track_match", vector, 0.8, source="test_fixture", event_id="persist-1")
    path = tmp_path / "registry.json"
    controller.save(path)
    loaded = LearningController.load(path)
    assert loaded.event_count == controller.event_count
    assert loaded.predict("track_match", vector) == controller.predict("track_match", vector)
