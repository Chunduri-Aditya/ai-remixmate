export function timbreCompatibility(a: number[] = [], b: number[] = []): { score: number; explanation: string[]; warnings: string[] } {
  const length = Math.min(a.length, b.length);
  if (length === 0) return { score: 0.5, explanation: ["No timbre vectors; using neutral score."], warnings: ["Missing timbre data."] };
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA <= 0 || normB <= 0) return { score: 0.5, explanation: ["Zero timbre vector; using neutral score."], warnings: ["Invalid timbre vector."] };
  const cosine = dot / (Math.sqrt(normA) * Math.sqrt(normB));
  const score = Math.max(0, Math.min(1, (cosine + 1) / 2));
  return { score, explanation: [`Timbre cosine similarity mapped to ${score.toFixed(3)}.`], warnings: [] };
}
