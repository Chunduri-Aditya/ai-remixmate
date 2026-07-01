export interface AutomationCurvePoint {
  timeSec: number;
  value: number;
}

export function validateAutomationPoints(points: AutomationCurvePoint[]): AutomationCurvePoint[] {
  const valid = points.filter((p) => Number.isFinite(p.timeSec) && p.timeSec >= 0 && Number.isFinite(p.value));
  return valid.sort((a, b) => a.timeSec - b.timeSec);
}

export function interpolateLinear(points: AutomationCurvePoint[], timeSec: number): number | null {
  const sorted = validateAutomationPoints(points);
  if (sorted.length === 0 || !Number.isFinite(timeSec)) return null;
  if (timeSec <= sorted[0].timeSec) return sorted[0].value;
  if (timeSec >= sorted[sorted.length - 1].timeSec) return sorted[sorted.length - 1].value;
  for (let i = 0; i < sorted.length - 1; i += 1) {
    const a = sorted[i];
    const b = sorted[i + 1];
    if (timeSec >= a.timeSec && timeSec <= b.timeSec) {
      const span = b.timeSec - a.timeSec;
      if (span <= 0) return b.value;
      const t = (timeSec - a.timeSec) / span;
      return a.value + (b.value - a.value) * t;
    }
  }
  return null;
}
