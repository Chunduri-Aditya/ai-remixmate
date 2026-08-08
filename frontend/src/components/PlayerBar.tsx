/* ============================================================
   PlayerBar — persistent bottom player with 3-band EQ.

   Ported from the legacy static UI's global player: transport,
   seek, queue shuffle/repeat, BASS/MID/TREBLE rotary knobs
   (±12 dB, drag vertically, double-click resets), volume, and
   a compact analyser visualizer. Renders across every page via
   AppShell; audio flows through lib/playerEngine's Web Audio
   chain so the EQ affects everything the bar plays.
   ============================================================ */
import { useEffect, useRef } from 'react'
import {
  Music2,
  Pause,
  Play,
  Repeat,
  Shuffle,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
} from 'lucide-react'
import { usePlayerStore } from '@/stores/playerStore'
import { playerEngine, EqBand } from '@/lib/playerEngine'
import './PlayerBar.css'

function fmt(s: number): string {
  if (!Number.isFinite(s) || s <= 0) return '0:00'
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}

/* Rotary EQ knob — drag up/down to adjust, double-click to reset */
function EqKnob({ band }: { band: EqBand }) {
  const value = usePlayerStore((s) => s.eq[band])
  const setEq = usePlayerStore((s) => s.setEq)
  const drag = useRef<{ startY: number; startVal: number } | null>(null)

  function onPointerDown(e: React.PointerEvent) {
    drag.current = { startY: e.clientY, startVal: value }
    e.currentTarget.setPointerCapture(e.pointerId)
  }
  function onPointerMove(e: React.PointerEvent) {
    if (!drag.current) return
    const dy = drag.current.startY - e.clientY
    setEq(band, Math.round(drag.current.startVal + dy / 4))
  }
  function onPointerUp() {
    drag.current = null
  }

  const angle = (value / 12) * 135 // ±12 dB → ±135°
  return (
    <div className="pb-knob-group">
      <span className="pb-knob-label">{band}</span>
      <div
        className="pb-knob"
        role="slider"
        aria-label={`${band} EQ`}
        aria-valuemin={-12}
        aria-valuemax={12}
        aria-valuenow={value}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onDoubleClick={() => setEq(band, 0)}
        title={`${band} ${value > 0 ? '+' : ''}${value} dB — drag to adjust, double-click to reset`}
      >
        <div className="pb-knob__pointer" style={{ transform: `rotate(${angle}deg)` }} />
      </div>
      <span className="pb-knob-value font-mono">{value}</span>
    </div>
  )
}

/* Compact analyser visualizer (idle breathing wave when silent) */
function MiniViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const playing = usePlayerStore((s) => s.playing)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    let raf = 0
    const draw = () => {
      raf = requestAnimationFrame(draw)
      const { width: W, height: H } = canvas
      ctx.clearRect(0, 0, W, H)
      const analyser = playerEngine.analyser
      if (!analyser || !playing) {
        const now = Date.now() / 1000
        ctx.fillStyle = 'rgba(167, 139, 250, 0.25)'
        for (let x = 0; x < W; x += 3) {
          const amp = 4 * Math.sin((x / W) * Math.PI * 3 + now)
          ctx.fillRect(x, H / 2 + amp, 2, 2)
        }
        return
      }
      const data = new Uint8Array(analyser.frequencyBinCount)
      analyser.getByteFrequencyData(data)
      const bars = 32
      const step = Math.floor((data.length * 0.6) / bars)
      for (let i = 0; i < bars; i++) {
        const v = data[i * step] / 255
        const h = Math.max(2, v * H)
        ctx.fillStyle = `rgba(245, 158, 11, ${0.35 + v * 0.65})`
        ctx.fillRect(i * (W / bars), H - h, (W / bars) * 0.7, h)
      }
    }
    draw()
    return () => cancelAnimationFrame(raf)
  }, [playing])

  return <canvas ref={canvasRef} className="pb-viz" width={120} height={28} />
}

export function PlayerBar() {
  const track = usePlayerStore((s) => s.queue[s.index] ?? null)
  const playing = usePlayerStore((s) => s.playing)
  const currentTime = usePlayerStore((s) => s.currentTime)
  const duration = usePlayerStore((s) => s.duration)
  const volume = usePlayerStore((s) => s.volume)
  const muted = usePlayerStore((s) => s.muted)
  const shuffle = usePlayerStore((s) => s.shuffle)
  const repeat = usePlayerStore((s) => s.repeat)
  const queueLen = usePlayerStore((s) => s.queue.length)
  const { toggle, next, prev, seek, setVolume, toggleMute, toggleShuffle, toggleRepeat } =
    usePlayerStore.getState()

  function onSeek(e: React.MouseEvent<HTMLDivElement>) {
    if (!duration) return
    const rect = e.currentTarget.getBoundingClientRect()
    seek(((e.clientX - rect.left) / rect.width) * duration)
  }

  return (
    <footer className="player-bar" data-testid="player-bar">
      <div className="player-bar__track">
        <Music2 size={14} className="player-bar__track-icon" />
        <div className="player-bar__track-text">
          <span className="player-bar__title" title={track?.title}>
            {track?.title ?? 'No track playing'}
          </span>
          {queueLen > 1 && (
            <span className="player-bar__queue text-muted font-mono">
              {usePlayerStore.getState().index + 1}/{queueLen}
            </span>
          )}
        </div>
        <MiniViz />
      </div>

      <div className="player-bar__transport">
        <button
          className={`pb-icon-btn ${shuffle ? 'pb-icon-btn--active' : ''}`}
          onClick={toggleShuffle}
          title="Shuffle"
        >
          <Shuffle size={13} />
        </button>
        <button className="pb-icon-btn" onClick={prev} title="Previous / restart" disabled={!track}>
          <SkipBack size={15} />
        </button>
        <button className="pb-play-btn" onClick={toggle} title={playing ? 'Pause' : 'Play'} disabled={!track}>
          {playing ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <button className="pb-icon-btn" onClick={next} title="Next" disabled={queueLen < 2 && !shuffle}>
          <SkipForward size={15} />
        </button>
        <button
          className={`pb-icon-btn ${repeat ? 'pb-icon-btn--active' : ''}`}
          onClick={toggleRepeat}
          title="Repeat track"
        >
          <Repeat size={13} />
        </button>
      </div>

      <div className="player-bar__timeline">
        <span className="font-mono pb-time">{fmt(currentTime)}</span>
        <div className="pb-seek" onClick={onSeek} title="Seek">
          <div
            className="pb-seek__fill"
            style={{ width: duration ? `${(currentTime / duration) * 100}%` : '0%' }}
          />
        </div>
        <span className="font-mono pb-time">{fmt(duration)}</span>
      </div>

      <div className="player-bar__eq">
        <EqKnob band="bass" />
        <EqKnob band="mid" />
        <EqKnob band="treble" />
      </div>

      <div className="player-bar__volume">
        <button className="pb-icon-btn" onClick={toggleMute} title={muted ? 'Unmute' : 'Mute'}>
          {muted ? <VolumeX size={14} /> : <Volume2 size={14} />}
        </button>
        <div className="pb-vol-group">
          <span className="pb-knob-label">vol</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={volume}
            onChange={(e) => setVolume(parseFloat(e.target.value))}
            className="pb-vol-slider"
            title="Volume"
          />
          <span className="pb-knob-value font-mono">{Math.round(volume * 100)}%</span>
        </div>
      </div>
    </footer>
  )
}
