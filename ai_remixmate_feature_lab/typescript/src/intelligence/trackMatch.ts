import type { TrackAnalysis } from "../models/TrackAnalysis";
import type { CompatibilityScore } from "../models/CompatibilityScore";
import { compatibilityScore } from "./compatibilityScore";

export interface TrackMatchResult {
  trackId: string;
  score: number;
  breakdown: CompatibilityScore;
}

export function trackMatch(source: TrackAnalysis, candidates: TrackAnalysis[]): TrackMatchResult[] {
  return candidates
    .filter((candidate) => candidate.id !== source.id)
    .map((candidate) => {
      const breakdown = compatibilityScore(source, candidate);
      return { trackId: candidate.id, score: breakdown.overall, breakdown };
    })
    .sort((a, b) => b.score - a.score);
}
