/* ============================================================
   StemsPlayerRow — playback strip for the 4 Demucs stems.
   Mute toggle per stem; clicking the progress bar seeks.
   ============================================================ */
import { useRef, useState, useEffect, useCallback } from 'react'
import { stemsApi } from '@/lib/api'
import './StemsPlayerRow.css'

const STEM_COLORS: Record<string, string> = {
  vocals: 'var(--color-violet-400)',
  drums:  'var(--color-crimson-500)',
  bass:   'var(--color-amber-500)',
  other:  'var(--color-ice-400)',
}

interface StemTrackProps {
  name: string
  stem: string
  muted: boolean
  onToggleMute: () => void
}

function StemTrack({ name, stem, muted, onToggleMute }: StemTrackProps) {
  const audioRef    = useRef<HTMLAudioElement>(null)
  const [progress, setProgress] = useState(0)
  const [duration, setDuration] = useState(0)

  useEffect(() => {
    const el = audioRef.current
    if (!el) return
    el.muted = muted
  }, [muted])

  function onTimeUpdate() {
    const el = audioRef.current
    if (!el || !el.duration) return
    setProgress(el.currentTime / el.duration)
    setDuration(el.duration)
  }

  function seek(e: React.MouseEvent<HTMLDivElement>) {
    const el = audioRef.current
    if (!el || !el.duration) return
    const rect = e.currentTarget.getBoundingClientRect()
    el.currentTime = ((e.clientX - rect.left) / rect.width) * el.duration
  }

  const elapsed = Math.round(progress * duration)
  const color   = STEM_COLORS[stem] ?? 'var(--color-text-muted)'

  return (
    <div className="spr-stem">
      <div className="spr-stem__dot" style={{ background: color }} />
      <span className="spr-stem__label">{stem}</span>

      <div className="spr-stem__audio-wrap" onClick={seek} title="Click to seek">
        <div
          className="spr-stem__progress"
          style={{ width: `${progress * 100}%`, background: color + '4D' }}
        />
        <span className="spr-stem__time font-mono">
          {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, '0')}
        </span>
      </div>

      <button
        className={`spr-mute-btn ${muted ? 'spr-mute-btn--muted' : ''}`}
        onClick={onToggleMute}
        title={muted ? 'Unmute' : 'Mute'}
      >
        {muted ? '🔇' : '🔊'}
      </button>

      <audio
        ref={audioRef}
        src={stemsApi.stemUrl(name, stem)}
        onTimeUpdate={onTimeUpdate}
        onLoadedMetadata={onTimeUpdate}
        controls
        style={{ display: 'none' }}
      />
    </div>
  )
}

interface StemsPlayerRowProps {
  name: string
  stems: string[]
}

export function StemsPlayerRow({ name, stems }: StemsPlayerRowProps) {
  const [muted, setMuted] = useState<Record<string, boolean>>(
    Object.fromEntries(stems.map((s) => [s, false])),
  )

  const toggleMute = useCallback((stem: string) => {
    setMuted((prev) => ({ ...prev, [stem]: !prev[stem] }))
  }, [])

  return (
    <div className="spr-wrap">
      {stems.map((stem) => (
        <StemTrack
          key={stem}
          name={name}
          stem={stem}
          muted={muted[stem] ?? false}
          onToggleMute={() => toggleMute(stem)}
        />
      ))}
    </div>
  )
}
