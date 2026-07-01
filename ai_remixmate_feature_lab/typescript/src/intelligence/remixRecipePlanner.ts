import type { RemixRecipe } from "../models/RemixRecipe";
import type { TrackAnalysis } from "../models/TrackAnalysis";
import type { TransitionPlan } from "../models/TransitionPlan";
import { transitionPlanner } from "./transitionPlanner";

export function remixRecipePlanner(from: TrackAnalysis, to: TrackAnalysis, plan: TransitionPlan = transitionPlanner(from, to)): RemixRecipe {
  const stemSuggestions = [
    "Bring incoming drums in first if the drum stem is available.",
    "Keep only one lead vocal active during the overlap.",
    "Swap bass ownership near the first strong downbeat of the incoming phrase."
  ];
  return {
    id: `recipe-${plan.id}`,
    transitionPlanId: plan.id,
    method: plan.compatibility.overall >= 0.75 ? "phrase_aligned_blend" : "short_controlled_blend",
    stemSuggestions,
    timingSuggestion: `Start the outgoing fade at ${plan.exitTimeSec.toFixed(2)}s and launch the incoming cue at ${plan.entryTimeSec.toFixed(2)}s.`,
    riskWarnings: plan.warnings,
    steps: [
      { order: 1, title: "Prepare cues", instruction: "Load both mix cues and verify they land on downbeats.", startSec: plan.exitTimeSec },
      { order: 2, title: "Open the incoming rhythm", instruction: "Fade in incoming drums while keeping bass controlled.", startSec: plan.exitTimeSec },
      { order: 3, title: "Swap bass", instruction: "Lower outgoing low EQ as the incoming bass becomes dominant." },
      { order: 4, title: "Protect vocals", instruction: "Delay or mute one vocal stem if both vocals overlap." },
      { order: 5, title: "Resolve", instruction: "Finish the outgoing fade and reset EQ/filter controls to neutral." }
    ]
  };
}
