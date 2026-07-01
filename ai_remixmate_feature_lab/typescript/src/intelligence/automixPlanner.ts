import type { TrackAnalysis } from "../models/TrackAnalysis";
import type { TransitionPlan } from "../models/TransitionPlan";
import { compatibilityScore } from "./compatibilityScore";
import { transitionPlanner } from "./transitionPlanner";

export interface AutomixPlan {
  orderedTrackIds: string[];
  transitionPlans: TransitionPlan[];
  totalConfidence: number;
  warnings: string[];
}

export function automixPlanner(tracks: TrackAnalysis[]): AutomixPlan {
  if (tracks.length === 0) return { orderedTrackIds: [], transitionPlans: [], totalConfidence: 0, warnings: ["No tracks supplied."] };
  const remaining = [...tracks];
  const ordered = [remaining.shift() as TrackAnalysis];
  const transitionPlans: TransitionPlan[] = [];
  const warnings: string[] = [];
  while (remaining.length) {
    const current = ordered[ordered.length - 1];
    let bestIndex = 0;
    let bestScore = -1;
    remaining.forEach((candidate, index) => {
      const score = compatibilityScore(current, candidate).overall;
      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    });
    const [next] = remaining.splice(bestIndex, 1);
    const plan = transitionPlanner(current, next);
    transitionPlans.push(plan);
    warnings.push(...plan.warnings);
    ordered.push(next);
  }
  const totalConfidence = transitionPlans.length
    ? transitionPlans.reduce((sum, plan) => sum + plan.compatibility.overall, 0) / transitionPlans.length
    : 1;
  return { orderedTrackIds: ordered.map((track) => track.id), transitionPlans, totalConfidence, warnings: Array.from(new Set(warnings)) };
}
