export interface VocalClashResult {
  risk: number;
  warnings: string[];
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function vocalClashRisk(a: number[] = [], b: number[] = []): VocalClashResult {
  const length = Math.min(a.length, b.length);
  if (length === 0) return { risk: 0, warnings: [] };
  let overlap = 0;
  for (let i = 0; i < length; i += 1) overlap += clamp01(a[i]) * clamp01(b[i]);
  const risk = clamp01(overlap / length);
  const warnings = risk > 0.35 ? ["Both tracks have active vocals in the same region; delay or isolate one vocal stem."] : [];
  return { risk, warnings };
}
