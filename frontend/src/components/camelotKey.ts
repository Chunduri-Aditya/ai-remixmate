/* ============================================================
   Musical key normalization for the Camelot wheel.

   Split out of CamelotWheel.tsx so that file exports only the
   CamelotWheel component — react-refresh/only-export-components
   needs component files to export components and nothing else.
   ============================================================ */

// Backend returns "C major", "A minor", "F# major" — strip qualifier.
export function normalizeKey(key?: string): string | undefined {
  if (!key) return undefined
  const k = key.trim()
  if (k.endsWith(' major')) return k.slice(0, -6).trim()
  if (k.endsWith(' minor')) return k.slice(0, -6).trim() + 'm'
  return k
}
