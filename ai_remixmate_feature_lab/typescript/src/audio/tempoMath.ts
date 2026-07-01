function assertPositiveBpm(bpm: number, name: string): void {
  if (!Number.isFinite(bpm) || bpm <= 0) {
    throw new RangeError(`${name} must be a positive BPM`);
  }
}

export function normalizeTargetBpm(sourceBpm: number, targetBpm: number): number {
  assertPositiveBpm(sourceBpm, "sourceBpm");
  assertPositiveBpm(targetBpm, "targetBpm");
  let adjusted = targetBpm;
  while (adjusted / sourceBpm > 1.5) adjusted /= 2;
  while (adjusted / sourceBpm < 0.67) adjusted *= 2;
  return adjusted;
}

export function bpmRatio(sourceBpm: number, targetBpm: number): number {
  return normalizeTargetBpm(sourceBpm, targetBpm) / sourceBpm;
}

export function tempoPercent(sourceBpm: number, targetBpm: number): number {
  return (bpmRatio(sourceBpm, targetBpm) - 1) * 100;
}

export function adjustBpmByPercent(bpm: number, percent: number): number {
  assertPositiveBpm(bpm, "bpm");
  if (!Number.isFinite(percent)) throw new RangeError("percent must be finite");
  return bpm * (1 + percent / 100);
}

export function isTempoChangeSafe(sourceBpm: number, targetBpm: number, maxPercent = 8): boolean {
  return Math.abs(tempoPercent(sourceBpm, targetBpm)) <= maxPercent;
}
