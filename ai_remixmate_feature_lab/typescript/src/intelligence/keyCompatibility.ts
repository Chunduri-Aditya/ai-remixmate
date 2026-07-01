export interface KeyCompatibilityResult {
  score: number;
  relation: string;
  explanation: string[];
  warnings: string[];
}

interface ParsedCamelot { number: number; letter: "A" | "B"; }

function parseCamelot(key?: string): ParsedCamelot | null {
  const match = (key ?? "").trim().toUpperCase().match(/^(1[0-2]|[1-9])([AB])$/);
  if (!match) return null;
  return { number: Number(match[1]), letter: match[2] as "A" | "B" };
}

function wrappedDelta(a: number, b: number): number {
  const raw = Math.abs(a - b);
  return Math.min(raw, 12 - raw);
}

function forwardDelta(a: number, b: number): number {
  return ((b - a + 12) % 12);
}

export function keyCompatibility(fromCamelot?: string, toCamelot?: string): KeyCompatibilityResult {
  const a = parseCamelot(fromCamelot);
  const b = parseCamelot(toCamelot);
  if (!a || !b) {
    return { score: 0.5, relation: "unknown", explanation: ["Unknown key; using neutral harmonic score."], warnings: ["Missing or invalid Camelot key."] };
  }
  if (a.number === b.number && a.letter === b.letter) {
    return { score: 1, relation: "same_key", explanation: ["Same Camelot key; safest harmonic blend."], warnings: [] };
  }
  if (a.number === b.number && a.letter !== b.letter) {
    return { score: 0.92, relation: "relative_major_minor", explanation: ["Relative major/minor movement on the same Camelot number."], warnings: [] };
  }
  if (a.letter === b.letter && wrappedDelta(a.number, b.number) === 1) {
    return { score: 0.85, relation: "adjacent", explanation: ["Adjacent Camelot number; common harmonic mix."], warnings: [] };
  }
  if (a.letter === b.letter && forwardDelta(a.number, b.number) === 2) {
    return { score: 0.72, relation: "energy_boost", explanation: ["Forward +2 Camelot movement can raise energy but needs attention."], warnings: ["Energy boost harmonic move is less neutral than same or adjacent key."] };
  }
  return { score: 0.35, relation: "distant", explanation: ["Distant Camelot movement; prefer a short blend, percussion-only section, or key shift."], warnings: ["Potential harmonic clash."] };
}
