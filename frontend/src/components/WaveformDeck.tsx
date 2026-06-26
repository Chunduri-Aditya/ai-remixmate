/* ============================================================
   WaveformDeck — interactive waveform strip powered by WaveSurfer.js v7.
   Shows the track waveform and an optional cue-region overlay marking
   the transition zone (exit/entry bars from the DJ engine).
   ============================================================ */
import { useRef, useEffect, useState, useCallback } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { Play, Pause, Loader2 } from 'lucide-react'

interface WaveformDeckProps {
  /** Full URL to the audio file (served by GET /library/{name}/audio). */
  src: string
  /** Deck accent color as a hex string, e.g. '#f59e0b'. */
  color: string
  /** Transition cue start in seconds (optional — from transition_plan). */
  cueStart?: number
  /** Transition cue end in seconds (optional — from transition_plan). */
  cueEnd?: number
}

export function WaveformDeck({ src, color, cueStart, cueEnd }: WaveformDeckProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wsRef        = useRef<WaveSurfer | null>(null)
  const [playing,  setPlaying]  = useState(false)
  const [duration, setDuration] = useState(0)
  const [loading,  setLoading]  = useState(true)
  const [errored,  setErrored]  = useState(false)

  useEffect(() => {
    if (!containerRef.current || !src) return

    wsRef.current?.destroy()
    setPlaying(false)
    setDuration(0)
    setLoading(true)
    setErrored(false)

    const ws = WaveSurfer.create({
      container:     containerRef.current,
      waveColor:     color + '88',   // 53 % opacity for unplayed portion
      progressColor: color,
      cursorColor:   color + 'cc',
      cursorWidth:   1,
      height:        80,
      barWidth:      2,
      barGap:        1,
      barRadius:     2,
      interact:      true,
      normalize:     true,
    })

    wsRef.current = ws

    ws.on('ready', () => {
      setDuration(ws.getDuration())
      setLoading(false)
    })
    ws.on('play',   () => setPlaying(true))
    ws.on('pause',  () => setPlaying(false))
    ws.on('finish', () => setPlaying(false))
    ws.on('error',  () => { setLoading(false); setErrored(true) })

    ws.load(src)

    return () => {
      ws.destroy()
      wsRef.current = null
    }
  }, [src, color])

  const toggle = useCallback(() => { wsRef.current?.playPause() }, [])

  const showRegion =
    !loading &&
    duration > 0 &&
    cueStart !== undefined &&
    cueEnd   !== undefined &&
    cueEnd > cueStart

  return (
    <div className="md-waveform">
      <div className="md-waveform__canvas-wrap">
        {loading && !errored && (
          <div className="md-waveform__loading">
            <Loader2 size={14} className="md-spin" style={{ color }} />
            <span>Loading waveform…</span>
          </div>
        )}
        <div
          ref={containerRef}
          style={{ opacity: loading ? 0 : 1, transition: 'opacity 0.3s' }}
        />
        {showRegion && (
          <div
            className="md-waveform__region"
            style={{
              left:        `${(cueStart! / duration) * 100}%`,
              width:       `${((cueEnd! - cueStart!) / duration) * 100}%`,
              background:  color + '40',
              borderLeft:  `2px solid ${color}`,
              borderRight: `2px solid ${color}`,
            }}
          />
        )}
      </div>

      {!errored && (
        <button
          className="md-waveform-btn"
          onClick={toggle}
          disabled={loading}
          title={playing ? 'Pause' : 'Play preview'}
        >
          {playing ? <Pause size={11} /> : <Play size={11} />}
        </button>
      )}
    </div>
  )
}
