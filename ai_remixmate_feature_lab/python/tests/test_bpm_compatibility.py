from ai_remixmate_features.intelligence.compatibility import bpm_compatibility, compatibility_score

def make_track(track_id="a", bpm=128.0, camelot="8A", energy=None, vocal=None):
    return {
        "id": track_id,
        "title": track_id.upper(),
        "duration_sec": 240.0,
        "bpm": bpm,
        "camelot": camelot,
        "beatgrid": {"bpm": bpm, "beatTimes": [0, 0.5, 1.0, 1.5, 2.0], "downbeats": [0, 2.0], "beatsPerBar": 4, "confidence": 0.9},
        "energy_curve": energy or [0.2, 0.4, 0.7],
        "timbre_vector": [1.0, 0.0, 0.0],
        "vocal_activity": vocal or [0.0, 0.1, 0.2],
    }



def test_bpm_half_double_scores_high():
    result = bpm_compatibility(128, 64)
    assert result["score"] > 0.95
    assert result["safe"] is True


def test_compatibility_score_range():
    result = compatibility_score(make_track("a"), make_track("b", 130, "9A"))
    assert 0 <= result["overall"] <= 1
    assert 0 <= result["bpmScore"] <= 1
