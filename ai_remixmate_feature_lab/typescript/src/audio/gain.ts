export function dbToLinear(db: number): number {
  if (!Number.isFinite(db)) return 1;
  return Math.pow(10, db / 20);
}

export function linearToDb(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return -120;
  return 20 * Math.log10(value);
}

export function clampDb(db: number, min = -60, max = 6): number {
  if (!Number.isFinite(db)) return 0;
  return Math.max(min, Math.min(max, db));
}
