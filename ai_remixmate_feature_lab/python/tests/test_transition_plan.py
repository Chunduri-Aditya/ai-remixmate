from ai_remixmate_features.intelligence.transition_plan import generate_transition_plan

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



def test_transition_plan_has_valid_times():
    plan = generate_transition_plan(make_track("a"), make_track("b", 64, "8B"), transition_length_bars=1)
    assert plan["exitTimeSec"] >= 0
    assert plan["entryTimeSec"] >= 0
    assert plan["transitionLengthBars"] == 1
    assert len(plan["cueSuggestions"]) == 2
