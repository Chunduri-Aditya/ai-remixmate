from __future__ import annotations

from typing import Any

from .compatibility import compatibility_score


def _times(track: dict[str, Any], key: str) -> list[float]:
    beatgrid = track.get("beatgrid", {})
    return [float(t) for t in beatgrid.get(key, []) if float(t) >= 0]


def _nearest_at_or_before(times: list[float], target: float) -> float:
    valid = sorted(t for t in times if t <= target)
    return valid[-1] if valid else 0.0


def generate_transition_plan(from_track: dict[str, Any], to_track: dict[str, Any], transition_length_bars: int = 16) -> dict[str, Any]:
    """Generate a deterministic phrase/downbeat-oriented transition plan."""
    if transition_length_bars <= 0:
        raise ValueError("transition_length_bars must be positive")
    score = compatibility_score(from_track, to_track)
    beats_per_bar = int(from_track.get("beatgrid", {}).get("beatsPerBar") or from_track.get("beatgrid", {}).get("beats_per_bar") or 4)
    beat_dur = 60.0 / float(from_track["bpm"])
    transition_sec = transition_length_bars * beats_per_bar * beat_dur
    target_exit = max(0.0, float(from_track.get("duration_sec", from_track.get("durationSec", 0.0))) - transition_sec)
    exit_candidates = _times(from_track, "downbeats") or _times(from_track, "beatTimes") or _times(from_track, "beat_times")
    entry_candidates = _times(to_track, "downbeats") or _times(to_track, "beatTimes") or _times(to_track, "beat_times")
    exit_time = _nearest_at_or_before(exit_candidates, target_exit)
    entry_time = entry_candidates[0] if entry_candidates else 0.0
    warnings = list(score["warnings"])
    return {
        "id": f"{from_track['id']}-to-{to_track['id']}",
        "fromTrackId": from_track["id"],
        "toTrackId": to_track["id"],
        "entryTimeSec": entry_time,
        "exitTimeSec": exit_time,
        "transitionLengthBars": transition_length_bars,
        "compatibility": score,
        "eqAutomationNotes": ["Reduce outgoing low EQ before bass swap.", "Bring incoming drums and bass up before vocals."],
        "filterAutomationNotes": ["Use a gentle high-pass on the incoming track until the downbeat lands."],
        "cueSuggestions": [
            {"id": f"{from_track['id']}-mix-out", "trackId": from_track["id"], "timestampSec": exit_time, "label": "Mix out", "type": "mix_out"},
            {"id": f"{to_track['id']}-mix-in", "trackId": to_track["id"], "timestampSec": entry_time, "label": "Mix in", "type": "mix_in"},
        ],
        "loopSuggestions": [{"active": False, "startSec": exit_time, "endSec": exit_time + max(1.0, beat_dur * 4), "lengthBeats": 4, "quantized": True}],
        "warnings": warnings,
    }
