export interface BeatLoop {
  startBeatIndex: number;
  endBeatIndex: number;
  lengthBeats: number;
  startSec: number;
  endSec: number;
}

const MIN_LOOP_BEATS = 1 / 16;
const MAX_LOOP_BEATS = 32;

function clampLoopLength(lengthBeats: number): number {
  if (!Number.isFinite(lengthBeats) || lengthBeats <= 0) return 1;
  return Math.max(MIN_LOOP_BEATS, Math.min(MAX_LOOP_BEATS, lengthBeats));
}

function timeAtBeat(index: number, beatTimes: number[]): number | null {
  if (beatTimes.length === 0 || index < 0) return null;
  if (Number.isInteger(index) && index < beatTimes.length) return beatTimes[index];
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower >= beatTimes.length) return null;
  const interval = beatTimes[Math.min(lower + 1, beatTimes.length - 1)] - beatTimes[Math.max(0, lower)];
  if (upper < beatTimes.length) {
    const frac = index - lower;
    return beatTimes[lower] + (beatTimes[upper] - beatTimes[lower]) * frac;
  }
  return beatTimes[lower] + Math.max(interval, 0.5) * (index - lower);
}

export function createBeatLoop(startBeatIndex: number, lengthBeats: number, beatTimes: number[]): BeatLoop | null {
  if (!Number.isInteger(startBeatIndex) || startBeatIndex < 0) return null;
  const sorted = beatTimes.filter((t) => Number.isFinite(t) && t >= 0).sort((a, b) => a - b);
  const length = clampLoopLength(lengthBeats);
  const startSec = timeAtBeat(startBeatIndex, sorted);
  const endSec = timeAtBeat(startBeatIndex + length, sorted);
  if (startSec === null || endSec === null || !isValidLoop(startSec, endSec)) return null;
  return {
    startBeatIndex,
    endBeatIndex: startBeatIndex + Math.ceil(length),
    lengthBeats: length,
    startSec,
    endSec
  };
}

export function halveLoop(lengthBeats: number): number {
  return clampLoopLength(lengthBeats / 2);
}

export function doubleLoop(lengthBeats: number): number {
  return clampLoopLength(lengthBeats * 2);
}

export function isValidLoop(startSec: number, endSec: number): boolean {
  return Number.isFinite(startSec) && Number.isFinite(endSec) && startSec >= 0 && endSec > startSec;
}
