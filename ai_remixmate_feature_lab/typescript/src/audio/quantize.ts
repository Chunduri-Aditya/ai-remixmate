import { nearestBeat, nextBeat } from "./beatSync";

export function quantizeToNearestBeat(time: number, beatTimes: number[]): number | null {
  return nearestBeat(time, beatTimes);
}

export function quantizeToNextBeat(time: number, beatTimes: number[]): number | null {
  return nextBeat(time, beatTimes);
}

export function quantizeToBar(time: number, downbeats: number[]): number | null {
  return nextBeat(time, downbeats) ?? nearestBeat(time, downbeats);
}
