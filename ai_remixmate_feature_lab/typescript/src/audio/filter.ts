export type FilterKind = "lowpass" | "highpass";

export interface FilterState {
  type: FilterKind;
  enabled: boolean;
  cutoffHz: number;
  q: number;
}

export function clampCutoff(cutoffHz: number, min = 20, max = 20000): number {
  if (!Number.isFinite(cutoffHz)) return 1000;
  return Math.max(min, Math.min(max, cutoffHz));
}

export function clampResonance(q: number, min = 0.1, max = 24): number {
  if (!Number.isFinite(q)) return 0.707;
  return Math.max(min, Math.min(max, q));
}

export function validateFilterState(state: Partial<FilterState> & { type?: FilterKind }): FilterState {
  return {
    type: state.type ?? "lowpass",
    enabled: state.enabled ?? false,
    cutoffHz: clampCutoff(state.cutoffHz ?? 1000),
    q: clampResonance(state.q ?? 0.707)
  };
}
