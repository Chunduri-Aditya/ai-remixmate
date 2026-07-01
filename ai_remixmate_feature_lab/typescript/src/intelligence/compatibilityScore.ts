import type { TrackAnalysis } from "../models/TrackAnalysis";
import type { CompatibilityScore } from "../models/CompatibilityScore";
import { bpmCompatibility } from "./bpmCompatibility";
import { keyCompatibility } from "./keyCompatibility";
import { energyCompatibility } from "./energyCompatibility";
import { timbreCompatibility } from "./timbreCompatibility";
import { vocalClashRisk } from "./vocalClashRisk";

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function compatibilityScore(a: TrackAnalysis, b: TrackAnalysis): CompatibilityScore {
  const bpm = bpmCompatibility(a.bpm, b.bpm);
  const key = keyCompatibility(a.camelot, b.camelot);
  const energy = energyCompatibility(a.energyCurve, b.energyCurve);
  const timbre = timbreCompatibility(a.timbreVector, b.timbreVector);
  const vocal = vocalClashRisk(a.vocalActivity, b.vocalActivity);
  const overall = clamp01(
    bpm.score * 0.30 +
    key.score * 0.25 +
    energy.score * 0.20 +
    timbre.score * 0.15 +
    (1 - vocal.risk) * 0.10
  );
  return {
    overall,
    bpmScore: bpm.score,
    keyScore: key.score,
    energyScore: energy.score,
    timbreScore: timbre.score,
    vocalClashPenalty: vocal.risk,
    explanation: [...bpm.explanation, ...key.explanation, ...energy.explanation, ...timbre.explanation],
    warnings: [...bpm.warnings, ...key.warnings, ...energy.warnings, ...timbre.warnings, ...vocal.warnings]
  };
}
