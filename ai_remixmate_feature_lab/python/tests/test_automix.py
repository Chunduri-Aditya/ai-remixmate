from ai_remixmate_features.intelligence.automix import generate_automix_plan
from ai_remixmate_features.intelligence.track_match import rank_tracks

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



def test_automix_returns_all_tracks_once():
    tracks = [make_track("a"), make_track("b", 64, "8B"), make_track("c", 130, "9A")]
    plan = generate_automix_plan(tracks)
    assert sorted(plan["orderedTrackIds"]) == ["a", "b", "c"]
    assert len(set(plan["orderedTrackIds"])) == 3


def test_track_match_ranks_candidates():
    source = make_track("a")
    results = rank_tracks(source, [source, make_track("b", 64, "8B"), make_track("c", 150, "2B")])
    assert [row["trackId"] for row in results] == ["b", "c"]
    assert results[0]["score"] >= results[1]["score"]
