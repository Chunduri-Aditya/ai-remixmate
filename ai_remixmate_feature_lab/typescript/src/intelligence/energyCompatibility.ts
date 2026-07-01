export interface EnergyCompatibilityResult {
  score: number;
  explanation: string[];
  warnings: string[];
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function sampleAt(values: number[], index: number, length: number): number {
  if (values.length === 0) return 0.5;
  if (values.length === 1 || length <= 1) return clamp01(values[0]);
  const position = (index / (length - 1)) * (values.length - 1);
  const left = Math.floor(position);
  const right = Math.min(values.length - 1, Math.ceil(position));
  const t = position - left;
  return clamp01(values[left] + (values[right] - values[left]) * t);
}

export function energyCompatibility(a: number[], b: number[]): EnergyCompatibilityResult {
  const length = Math.max(4, Math.min(64, Math.max(a.length, b.length)));
  let total = 0;
  for (let i = 0; i < length; i += 1) total += Math.abs(sampleAt(a, i, length) - sampleAt(b, i, length));
  const averageDistance = total / length;
  const score = clamp01(1 - averageDistance);
  const warnings = score < 0.55 ? ["Energy curves differ strongly; plan a shorter or more obvious transition."] : [];
  return { score, explanation: [`Average normalized energy distance is ${averageDistance.toFixed(3)}.`], warnings };
}
