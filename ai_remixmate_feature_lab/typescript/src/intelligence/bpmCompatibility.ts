import { isTempoChangeSafe, normalizeTargetBpm, tempoPercent } from "../audio/tempoMath";

export interface BpmCompatibilityResult {
  score: number;
  sourceBpm: number;
  targetBpm: number;
  normalizedTargetBpm: number;
  percentChange: number;
  safe: boolean;
  explanation: string[];
  warnings: string[];
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function bpmCompatibility(sourceBpm: number, targetBpm: number, maxSafePercent = 8): BpmCompatibilityResult {
  const normalizedTargetBpm = normalizeTargetBpm(sourceBpm, targetBpm);
  const percentChange = tempoPercent(sourceBpm, targetBpm);
  const distance = Math.abs(percentChange);
  const score = clamp01(1 - distance / 16);
  const safe = isTempoChangeSafe(sourceBpm, targetBpm, maxSafePercent);
  const explanation = [distance < 0.1 ? "BPMs are effectively identical." : `Tempo shift is ${percentChange.toFixed(2)}%.`];
  if (Math.abs(normalizedTargetBpm - targetBpm) > 0.01) explanation.push("Half/double BPM relation was normalized before scoring.");
  const warnings = safe ? [] : [`Tempo shift exceeds ${maxSafePercent}% and may create stretching artifacts.`];
  return { score, sourceBpm, targetBpm, normalizedTargetBpm, percentChange, safe, explanation, warnings };
}
