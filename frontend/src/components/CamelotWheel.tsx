/* ============================================================
   CamelotWheel — SVG Camelot mixing-key compatibility wheel.
   Self-contained, no CSS file — all styles inline.
   ============================================================ */

const CAMELOT: Record<string, { pos: number; type: 'A' | 'B' }> = {
  'Am':  { pos: 1,  type: 'A' }, 'C':   { pos: 1,  type: 'B' },
  'Em':  { pos: 2,  type: 'A' }, 'G':   { pos: 2,  type: 'B' },
  'Bm':  { pos: 3,  type: 'A' }, 'D':   { pos: 3,  type: 'B' },
  'F#m': { pos: 4,  type: 'A' }, 'A':   { pos: 4,  type: 'B' },
  'C#m': { pos: 5,  type: 'A' }, 'E':   { pos: 5,  type: 'B' },
  'G#m': { pos: 6,  type: 'A' }, 'B':   { pos: 6,  type: 'B' },
  'Ebm': { pos: 7,  type: 'A' }, 'F#':  { pos: 7,  type: 'B' },
  'Bbm': { pos: 8,  type: 'A' }, 'Db':  { pos: 8,  type: 'B' },
  'Fm':  { pos: 9,  type: 'A' }, 'Ab':  { pos: 9,  type: 'B' },
  'Cm':  { pos: 10, type: 'A' }, 'Eb':  { pos: 10, type: 'B' },
  'Gm':  { pos: 11, type: 'A' }, 'Bb':  { pos: 11, type: 'B' },
  'Dm':  { pos: 12, type: 'A' }, 'F':   { pos: 12, type: 'B' },
}

// Backend returns "C major", "A minor", "F# major" — strip qualifier.
export function normalizeKey(key?: string): string | undefined {
  if (!key) return undefined
  const k = key.trim()
  if (k.endsWith(' major')) return k.slice(0, -6).trim()
  if (k.endsWith(' minor')) return k.slice(0, -6).trim() + 'm'
  return k
}

function toRad(deg: number) { return (deg * Math.PI) / 180 }

function polarToCart(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = toRad(angleDeg)
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

// Builds an SVG path for a donut arc segment (r1 = inner, r2 = outer).
function arcPath(
  cx: number, cy: number,
  r1: number, r2: number,
  startDeg: number, endDeg: number,
): string {
  const f = (n: number) => n.toFixed(2)
  const p1 = polarToCart(cx, cy, r2, startDeg)
  const p2 = polarToCart(cx, cy, r2, endDeg)
  const p3 = polarToCart(cx, cy, r1, endDeg)
  const p4 = polarToCart(cx, cy, r1, startDeg)
  return [
    `M ${f(p1.x)} ${f(p1.y)}`,
    `A ${f(r2)} ${f(r2)} 0 0 1 ${f(p2.x)} ${f(p2.y)}`,
    `L ${f(p3.x)} ${f(p3.y)}`,
    `A ${f(r1)} ${f(r1)} 0 0 0 ${f(p4.x)} ${f(p4.y)}`,
    'Z',
  ].join(' ')
}

export interface CamelotWheelProps {
  keyA?: string  // e.g. "C major"
  keyB?: string  // e.g. "A minor"
  size?: number  // default 260
}

export function CamelotWheel({ keyA, keyB, size = 260 }: CamelotWheelProps) {
  // Fixed 260×260 coordinate space — SVG viewBox handles scaling.
  const cx = 130, cy = 130
  const RMajorOuter = 120  // outer ring (B / major) — outer edge
  const RMajorInner = 90   // outer ring (B / major) — inner edge
  const RMinorOuter = 85   // inner ring (A / minor) — outer edge
  const RMinorInner = 55   // inner ring (A / minor) — inner edge

  const infoA = CAMELOT[normalizeKey(keyA) ?? '']
  const infoB = CAMELOT[normalizeKey(keyB) ?? '']

  function isActive(pos: number, type: 'A' | 'B'): boolean {
    return (infoA?.pos === pos && infoA?.type === type) ||
           (infoB?.pos === pos && infoB?.type === type)
  }

  function isCompatible(pos: number, type: 'A' | 'B'): boolean {
    const sources = ([infoA, infoB]).filter(
      (x): x is { pos: number; type: 'A' | 'B' } => !!x,
    )
    if (sources.length === 0) return false
    return sources.some((ref) => {
      if (pos === ref.pos) return true   // same position (incl. relative major/minor)
      if (type === ref.type) {
        const diff = Math.abs(pos - ref.pos)
        return diff === 1 || diff === 11  // adjacent, wrap-around handles 1↔12
      }
      return false
    })
  }

  function segOpacity(pos: number, type: 'A' | 'B'): number {
    if (isActive(pos, type)) return 1
    if (isCompatible(pos, type)) return 0.7
    return 0.25
  }

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 260 260"
      style={{ display: 'block' }}
      aria-label="Camelot key compatibility wheel"
    >
      {Array.from({ length: 12 }, (_, i) => {
        const pos = i + 1
        const hue = i * 30
        const startDeg = i * 30 - 90 + 0.5   // 0.5° gap on each side
        const endDeg   = (i + 1) * 30 - 90 - 0.5
        const midDeg   = (startDeg + endDeg) / 2

        const activeA = isActive(pos, 'A')
        const activeB = isActive(pos, 'B')

        const colorA = activeA ? `hsl(${hue},80%,55%)` : `hsl(${hue},55%,28%)`
        const colorB = activeB ? `hsl(${hue},80%,55%)` : `hsl(${hue},55%,28%)`

        const midA = polarToCart(cx, cy, (RMinorInner + RMinorOuter) / 2, midDeg)
        const midB = polarToCart(cx, cy, (RMajorInner + RMajorOuter) / 2, midDeg)

        return (
          <g key={pos}>
            {/* Inner ring — A / minor */}
            <path
              d={arcPath(cx, cy, RMinorInner, RMinorOuter, startDeg, endDeg)}
              fill={colorA}
              opacity={segOpacity(pos, 'A')}
            />
            <text
              x={midA.x.toFixed(1)} y={midA.y.toFixed(1)}
              textAnchor="middle" dominantBaseline="central"
              fill={activeA ? '#fff' : 'rgba(255,255,255,0.75)'}
              fontSize="8"
              fontFamily="monospace"
              style={{ userSelect: 'none', pointerEvents: 'none' }}
            >
              {pos}A
            </text>

            {/* Outer ring — B / major */}
            <path
              d={arcPath(cx, cy, RMajorInner, RMajorOuter, startDeg, endDeg)}
              fill={colorB}
              opacity={segOpacity(pos, 'B')}
            />
            <text
              x={midB.x.toFixed(1)} y={midB.y.toFixed(1)}
              textAnchor="middle" dominantBaseline="central"
              fill={activeB ? '#fff' : 'rgba(255,255,255,0.75)'}
              fontSize="8"
              fontFamily="monospace"
              style={{ userSelect: 'none', pointerEvents: 'none' }}
            >
              {pos}B
            </text>
          </g>
        )
      })}

      {/* Center hint */}
      <text
        x={cx} y={cy}
        textAnchor="middle" dominantBaseline="central"
        fill="rgba(255,255,255,0.3)"
        fontSize="9"
        fontFamily="monospace"
        style={{ userSelect: 'none' }}
      >
        Camelot
      </text>
    </svg>
  )
}
