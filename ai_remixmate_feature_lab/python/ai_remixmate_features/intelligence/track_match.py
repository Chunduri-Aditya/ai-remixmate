from __future__ import annotations

from typing import Any

from .compatibility import compatibility_score


def rank_tracks(source: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank candidate tracks by deterministic compatibility score."""
    results = []
    for candidate in candidates:
        if candidate.get("id") == source.get("id"):
            continue
        breakdown = compatibility_score(source, candidate)
        results.append({"trackId": candidate["id"], "score": breakdown["overall"], "breakdown": breakdown})
    return sorted(results, key=lambda item: item["score"], reverse=True)
