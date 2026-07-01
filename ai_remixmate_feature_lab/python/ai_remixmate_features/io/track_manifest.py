from __future__ import annotations

from typing import Any


def build_track_manifest(track_analysis: dict[str, Any]) -> dict[str, Any]:
    """Build a lightweight manifest from a TrackAnalysis-like dictionary."""
    required = ["id", "title", "bpm"]
    missing = [key for key in required if key not in track_analysis]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    return {
        "trackId": track_analysis["id"],
        "title": track_analysis["title"],
        "bpm": float(track_analysis["bpm"]),
        "camelot": track_analysis.get("camelot"),
        "hasBeatgrid": bool(track_analysis.get("beatgrid")),
        "hasEnergy": bool(track_analysis.get("energy_curve") or track_analysis.get("energyCurve")),
    }
