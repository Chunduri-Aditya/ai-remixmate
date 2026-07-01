from ai_remixmate_features.intelligence.remix_recipe import generate_remix_recipe

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



def test_remix_recipe_has_ordered_steps():
    recipe = generate_remix_recipe(make_track("a"), make_track("b", 64, "8B"))
    assert len(recipe["steps"]) >= 5
    assert [step["order"] for step in recipe["steps"]] == [1, 2, 3, 4, 5]
