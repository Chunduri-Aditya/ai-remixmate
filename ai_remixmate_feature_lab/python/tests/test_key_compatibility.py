from ai_remixmate_features.intelligence.compatibility import key_compatibility


def test_same_and_relative_keys_score_high():
    assert key_compatibility("8A", "8A")["score"] == 1.0
    assert key_compatibility("8A", "8B")["score"] > 0.9


def test_unknown_key_is_neutral():
    result = key_compatibility(None, "8A")
    assert result["score"] == 0.5
    assert result["warnings"]
