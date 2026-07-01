import type { TrackAnalysis } from "../models/TrackAnalysis";
import { PAIR_VECTOR_SIZE, TRACK_VECTOR_SIZE } from "./featureRegistry";

function clamp(value: number, low: number, high: number): number {
  if (!Number.isFinite(value)) return low;
  return Math.max(low, Math.min(high, value));
}

function mean(values: number[], fallback = 0): number {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : fallback;
}

function std(values: number[]): number {
  if (!values.length) return 0;
  const avg = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length);
}

function camelot(value?: string): [number, number, number] {
  const match = (value ?? "").trim().toUpperCase().match(/^(1[0-2]|[1-9])([AB])$/);
  if (!match) return [0, 0, 1];
  return [Number(match[1]) / 12, match[2] === "B" ? 1 : -1, 0];
}

function cosine(a: number[] = [], b: number[] = []): number {
  const length = Math.min(a.length, b.length);
  if (!length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let index = 0; index < length; index += 1) {
    dot += a[index] * b[index];
    normA += a[index] ** 2;
    normB += b[index] ** 2;
  }
  if (normA <= 0 || normB <= 0) return 0;
  return clamp(dot / (Math.sqrt(normA) * Math.sqrt(normB)), -1, 1);
}

export function trackFeatureVector(track: TrackAnalysis): number[] {
  const [keyNumber, keyMode, keyUnknown] = camelot(track.camelot);
  const energy = track.energyCurve.map((value) => clamp(value, 0, 1));
  const timbre = track.timbreVector ?? [];
  const vocal = (track.vocalActivity ?? []).map((value) => clamp(value, 0, 1));
  const stemAvailability = track.stemManifest?.stems.length
    ? track.stemManifest.stems.filter((stem) => stem.available).length / track.stemManifest.stems.length
    : 0;
  const vector = [
    clamp(track.bpm / 200, 0, 1),
    clamp(track.durationSec / 600, 0, 1),
    keyNumber,
    keyMode,
    keyUnknown,
    mean(energy, 0.5),
    clamp(std(energy), 0, 1),
    energy[0] ?? 0.5,
    energy[energy.length - 1] ?? 0.5,
    clamp(mean(timbre, 0) / 50, -1, 1),
    clamp(mean(timbre.map(Math.abs), 0) / 50, 0, 1),
    mean(vocal, 0),
    vocal.length ? Math.max(...vocal) : 0,
    clamp(track.beatgrid.confidence, 0, 1),
    clamp(track.beatgrid.beatTimes.length / 1024, 0, 1),
    clamp(track.beatgrid.downbeats.length / 256 + stemAvailability * 0.25, 0, 1)
  ];
  if (vector.length !== TRACK_VECTOR_SIZE) throw new Error("track vector size mismatch");
  return vector.map((value) => clamp(value, -1, 1));
}

export function pairFeatureVector(a: TrackAnalysis, b: TrackAnalysis): number[] {
  const va = trackFeatureVector(a);
  const vb = trackFeatureVector(b);
  let normalizedBpmB = b.bpm;
  while (normalizedBpmB / a.bpm > 1.5) normalizedBpmB /= 2;
  while (normalizedBpmB / a.bpm < 0.67) normalizedBpmB *= 2;
  const percent = ((normalizedBpmB / a.bpm) - 1) * 100;
  const [aNumber, aMode, aUnknown] = camelot(a.camelot);
  const [bNumber, bMode] = camelot(b.camelot);
  const vocalA = a.vocalActivity ?? [];
  const vocalB = b.vocalActivity ?? [];
  const vocalOverlap = mean(vocalA.slice(0, vocalB.length).map((value, index) => value * vocalB[index]), 0);
  const extra = [
    clamp(percent / 50, -1, 1),
    Math.abs(percent) <= 2 ? 1 : 0,
    aUnknown === 0 && aNumber === bNumber && aMode === bMode ? 1 : 0,
    aUnknown === 0 && aNumber === bNumber && aMode !== bMode ? 1 : 0,
    clamp(mean(b.energyCurve, 0.5) - mean(a.energyCurve, 0.5), -1, 1),
    cosine(a.timbreVector, b.timbreVector),
    clamp(vocalOverlap, 0, 1),
    clamp((b.durationSec / Math.max(1, a.durationSec)) - 1, -1, 1)
  ];
  const vector = [...va, ...vb, ...extra];
  if (vector.length !== PAIR_VECTOR_SIZE) throw new Error("pair vector size mismatch");
  return vector.map((value) => clamp(value, -1, 1));
}
