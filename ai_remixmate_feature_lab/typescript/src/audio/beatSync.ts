function finiteSorted(times: number[]): number[] {
  return times.filter((t) => Number.isFinite(t) && t >= 0).sort((a, b) => a - b);
}

export function nearestBeat(time: number, beatTimes: number[]): number | null {
  if (!Number.isFinite(time)) return null;
  const beats = finiteSorted(beatTimes);
  if (beats.length === 0) return null;
  return beats.reduce((best, beat) => Math.abs(beat - time) < Math.abs(best - time) ? beat : best, beats[0]);
}

export function nextBeat(time: number, beatTimes: number[]): number | null {
  if (!Number.isFinite(time)) return null;
  return finiteSorted(beatTimes).find((beat) => beat >= time) ?? null;
}

export function previousBeat(time: number, beatTimes: number[]): number | null {
  if (!Number.isFinite(time)) return null;
  const beats = finiteSorted(beatTimes).filter((beat) => beat <= time);
  return beats.length ? beats[beats.length - 1] : null;
}

export function phaseOffsetSeconds(deckABeat: number, deckBBeat: number): number {
  if (!Number.isFinite(deckABeat) || !Number.isFinite(deckBBeat)) return 0;
  return deckBBeat - deckABeat;
}
