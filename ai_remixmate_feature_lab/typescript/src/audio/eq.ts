export interface EqState {
  lowDb: number;
  midDb: number;
  highDb: number;
}

export const EQ_MIN_DB = -24;
export const EQ_MAX_DB = 6;

export function clampEqBand(db: number): number {
  if (!Number.isFinite(db)) return 0;
  return Math.max(EQ_MIN_DB, Math.min(EQ_MAX_DB, db));
}

export function validateEqState(state: Partial<EqState>): EqState {
  return {
    lowDb: clampEqBand(state.lowDb ?? 0),
    midDb: clampEqBand(state.midDb ?? 0),
    highDb: clampEqBand(state.highDb ?? 0)
  };
}
