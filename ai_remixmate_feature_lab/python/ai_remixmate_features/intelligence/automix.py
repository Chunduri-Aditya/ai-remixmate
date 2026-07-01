from __future__ import annotations

from typing import Any

from .compatibility import compatibility_score
from .transition_plan import generate_transition_plan


def generate_automix_plan(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """Greedily order tracks by next-track compatibility."""
    if not tracks:
        return {"orderedTrackIds": [], "transitionPlans": [], "totalConfidence": 0.0, "warnings": ["No tracks supplied."]}
    remaining = list(tracks)
    ordered = [remaining.pop(0)]
    plans = []
    warnings: list[str] = []
    while remaining:
        current = ordered[-1]
        best_index = max(range(len(remaining)), key=lambda idx: compatibility_score(current, remaining[idx])["overall"])
        next_track = remaining.pop(best_index)
        plan = generate_transition_plan(current, next_track)
        plans.append(plan)
        warnings.extend(plan.get("warnings", []))
        ordered.append(next_track)
    confidence = sum(plan["compatibility"]["overall"] for plan in plans) / len(plans) if plans else 1.0
    return {"orderedTrackIds": [track["id"] for track in ordered], "transitionPlans": plans, "totalConfidence": confidence, "warnings": sorted(set(warnings))}
