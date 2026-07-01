from __future__ import annotations

from typing import Any

from .transition_plan import generate_transition_plan


def generate_remix_recipe(from_track: dict[str, Any], to_track: dict[str, Any], plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create beginner-readable deterministic remix recipe steps."""
    plan = plan or generate_transition_plan(from_track, to_track)
    method = "phrase_aligned_blend" if plan["compatibility"]["overall"] >= 0.75 else "short_controlled_blend"
    return {
        "id": f"recipe-{plan['id']}",
        "transitionPlanId": plan["id"],
        "method": method,
        "stemSuggestions": ["Bring incoming drums in first if available.", "Keep only one lead vocal active.", "Swap bass ownership near the incoming downbeat."],
        "timingSuggestion": f"Start outgoing fade at {plan['exitTimeSec']:.2f}s and launch incoming cue at {plan['entryTimeSec']:.2f}s.",
        "riskWarnings": plan.get("warnings", []),
        "steps": [
            {"order": 1, "title": "Prepare cues", "instruction": "Verify both mix cues land on downbeats.", "startSec": plan["exitTimeSec"]},
            {"order": 2, "title": "Open rhythm", "instruction": "Fade incoming drums while bass remains controlled."},
            {"order": 3, "title": "Swap bass", "instruction": "Lower outgoing low EQ as incoming bass becomes dominant."},
            {"order": 4, "title": "Protect vocals", "instruction": "Delay or mute one vocal stem if vocals overlap."},
            {"order": 5, "title": "Resolve", "instruction": "Finish outgoing fade and reset controls to neutral."},
        ],
    }
