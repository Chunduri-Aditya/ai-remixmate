/* ============================================================
   TransitionTimeline — horizontal strip showing the planned transition:
   [Song A body] [crossfade zone] [Song B body]
   ============================================================ */
import type { CompatibilityResult } from '@/types'
import './TransitionTimeline.css'

interface TransitionTimelineProps {
  result: CompatibilityResult
  durationA?: number    // total duration of Song A in seconds
  durationB?: number    // total duration of Song B in seconds
}

export function TransitionTimeline({ result, durationA, durationB }: TransitionTimelineProps) {
  const plan = result.transition_plan

  const tempoRatio = result.bpm_a && result.bpm_b
    ? result.bpm_b / result.bpm_a
    : 1.0

  const transBars = plan?.transition_bars ?? 16

  // If we have timing data, compute proportional widths; otherwise use 2:1:2.
  let flexA = 2
  let flexB = 2
  if (plan && durationA && durationB && plan.exit_time_a !== undefined) {
    flexA = Math.max(0.5, plan.exit_time_a / (durationA + durationB) * 10)
    flexB = Math.max(0.5, (durationB - (plan.entry_time_b ?? 0)) / (durationA + durationB) * 10)
  }

  const isKeyCompatible = (result.key_score ?? 0) >= 0.7

  return (
    <div className="tt-wrap">
      <div className="tt-strip">
        <div className="tt-zone tt-zone--a" style={{ flex: flexA }}>
          <span className="tt-zone-label">Song A</span>
          {result.bpm_a ? (
            <span className="tt-bpm font-mono">{result.bpm_a.toFixed(0)} bpm</span>
          ) : null}
        </div>

        <div className="tt-crossfade">
          <span className="tt-xfade-label">{transBars} bars</span>
        </div>

        <div className="tt-zone tt-zone--b" style={{ flex: flexB }}>
          <span className="tt-zone-label">Song B</span>
          {result.bpm_b ? (
            <span className="tt-bpm font-mono">{result.bpm_b.toFixed(0)} bpm</span>
          ) : null}
        </div>
      </div>

      <div className="tt-meta">
        <span className="tt-meta-item">
          stretch <span className="font-mono">{tempoRatio.toFixed(3)}×</span>
        </span>

        {result.camelot_a && result.camelot_b && (
          <span className="tt-badge tt-badge--camelot">
            {result.camelot_a} → {result.camelot_b}
          </span>
        )}

        {isKeyCompatible && (
          <span className="tt-badge tt-badge--key">key match</span>
        )}

        {result.genre_a && result.genre_b && result.genre_a !== result.genre_b && (
          <span className="tt-meta-item">
            {result.genre_a} / {result.genre_b}
          </span>
        )}
      </div>
    </div>
  )
}
