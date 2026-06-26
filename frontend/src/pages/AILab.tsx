/* ============================================================
   AI RemixMate — AI Lab
   Generative features: style transfer, inpainting, tokenization.
   Model selector · parameter controls · job launcher · output player.
   ============================================================ */
import { useState, useMemo, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  FlaskConical,
  Cpu,
  Paintbrush,
  AudioLines,
  Play,
  Pause,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import { libraryApi, aiApi } from '@/lib/api'
import { useAppStore } from '@/stores/appStore'
import type { SongInfo } from '@/types'
import './PageBase.css'
import './AILab.css'

type Mode = 'style-transfer' | 'inpaint' | 'tokenize'

const MODES: { id: Mode; label: string; icon: React.ElementType; desc: string }[] = [
  {
    id: 'style-transfer',
    label: 'Style Transfer',
    icon: Paintbrush,
    desc: 'Apply the sonic style of one track to another — same melody, new texture.',
  },
  {
    id: 'inpaint',
    label: 'Inpainting',
    icon: AudioLines,
    desc: 'Reconstruct or modify a time region using the generative model.',
  },
  {
    id: 'tokenize',
    label: 'Tokenize',
    icon: Cpu,
    desc: 'Encode a track into discrete audio tokens for downstream generation.',
  },
]

function MiniPlayer({ src }: { src: string }) {
  const ref   = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [prog, setProg]       = useState(0)

  function toggle() {
    const el = ref.current
    if (!el) return
    if (playing) { el.pause(); setPlaying(false) }
    else         { el.play();  setPlaying(true)  }
  }

  return (
    <div className="al-mini-player">
      <audio ref={ref} src={src}
        onTimeUpdate={() => {
          const el = ref.current
          if (el && el.duration) setProg(el.currentTime / el.duration)
        }}
        onEnded={() => setPlaying(false)}
      />
      <button className="al-mini-play" onClick={toggle}>
        {playing ? <Pause size={14} /> : <Play size={14} />}
      </button>
      <div className="al-mini-seek">
        <div className="al-mini-fill" style={{ width: `${prog * 100}%` }} />
      </div>
    </div>
  )
}

interface JobResultBannerProps {
  jobId: string
}

function JobResultBanner({ jobId }: JobResultBannerProps) {
  const job = useAppStore((s) => s.jobs[jobId])
  if (!job) return null
  const BASE = import.meta.env.VITE_API_BASE || '/api'

  const result = job.result as { stream_url?: string; output_url?: string } | undefined
  const url    = result?.stream_url
    ? `${BASE}${result.stream_url}`
    : result?.output_url
    ? `${BASE}${result.output_url}`
    : null

  if (job.status === 'RUNNING' || job.status === 'PENDING') {
    return (
      <div className="al-status al-status--running">
        <Loader2 size={14} className="al-spin" />
        <span>{job.progress}% — {job.message}</span>
      </div>
    )
  }
  if (job.status === 'FAILED') {
    return (
      <div className="al-status al-status--failed">
        <AlertTriangle size={14} />
        <span>{job.error ?? 'Job failed'}</span>
      </div>
    )
  }
  if (job.status === 'COMPLETED') {
    return (
      <div className="al-status al-status--done">
        <CheckCircle2 size={14} />
        <span>Done!</span>
        {url && <MiniPlayer src={url} />}
      </div>
    )
  }
  return null
}

export default function AILab() {
  const [mode, setMode]         = useState<Mode>('style-transfer')
  const [song, setSong]         = useState('')
  const [styleSong, setStyleSong] = useState('')
  const [model, setModel]       = useState('')
  const [inpaintStart, setInpaintStart] = useState('0')
  const [inpaintEnd, setInpaintEnd]     = useState('10')
  const [temperature, setTemperature]   = useState(0.8)
  const [jobId, setJobId]       = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError]       = useState<string | null>(null)
  const [paramsOpen, setParamsOpen] = useState(true)

  const upsertJob = useAppStore((s) => s.upsertJob)

  const { data: songs = [] } = useQuery<SongInfo[]>({
    queryKey: ['library-atlas'],
    queryFn: libraryApi.list,
    staleTime: 60_000,
  })

  const { data: models = [] } = useQuery<string[]>({
    queryKey: ['ai-models'],
    queryFn:  aiApi.models,
    staleTime: 300_000,
  })

  const songNames = useMemo(() => songs.map((s) => s.name).sort(), [songs])

  async function launch() {
    if (!song) return
    setError(null)
    setSubmitting(true)
    try {
      let res: { job_id: string }
      const opts: Record<string, unknown> = {}
      if (model) opts.model = model
      if (mode === 'style-transfer') {
        if (!styleSong) throw new Error('Pick a style reference track.')
        res = await aiApi.styleTransfer(song, styleSong)
      } else if (mode === 'inpaint') {
        res = await aiApi.inpaint(song, {
          ...opts,
          start: parseFloat(inpaintStart),
          end: parseFloat(inpaintEnd),
          temperature,
        })
      } else {
        // tokenize — use inpaint endpoint with tokenize flag
        res = await aiApi.inpaint(song, { ...opts, tokenize: true })
      }
      setJobId(res.job_id)
      upsertJob({
        job_id: res.job_id, status: 'PENDING', type: mode,
        progress: 0, message: `Running ${mode}…`,
        created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Launch failed')
    } finally {
      setSubmitting(false)
    }
  }

  const currentMode = MODES.find((m) => m.id === mode)!
  const ModeIcon    = currentMode.icon

  return (
    <div className="page-base">
      <header className="page-base__header">
        <FlaskConical size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div>
          <h1 className="page-base__title font-display">AI Lab</h1>
          <p className="page-base__sub text-muted">
            Generative audio — style transfer · inpainting · tokenization
          </p>
        </div>
      </header>

      <div className="page-base__body al-body">
        {/* Mode tabs */}
        <div className="al-mode-tabs">
          {MODES.map((m) => {
            const Icon = m.icon
            return (
              <button
                key={m.id}
                className={`al-mode-tab ${mode === m.id ? 'al-mode-tab--active' : ''}`}
                onClick={() => { setMode(m.id); setJobId(null); setError(null) }}
              >
                <Icon size={16} strokeWidth={1.5} />
                <span>{m.label}</span>
              </button>
            )
          })}
        </div>

        <div className="al-layout">
          {/* Left: config */}
          <div className="al-config">
            {/* Mode description */}
            <div className="al-desc">
              <ModeIcon size={14} style={{ color: 'var(--color-amber-500)', flexShrink: 0 }} />
              <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>{currentMode.desc}</p>
            </div>

            {/* Song select */}
            <div className="al-field">
              <label className="al-label">Source track</label>
              <select
                className="al-select"
                value={song}
                onChange={(e) => { setSong(e.target.value); setJobId(null) }}
              >
                <option value="">— select a song —</option>
                {songNames.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>

            {/* Style reference (style-transfer only) */}
            {mode === 'style-transfer' && (
              <div className="al-field">
                <label className="al-label">Style reference</label>
                <select
                  className="al-select"
                  value={styleSong}
                  onChange={(e) => setStyleSong(e.target.value)}
                >
                  <option value="">— select style source —</option>
                  {songNames.filter((n) => n !== song).map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            )}

            {/* Model selector */}
            {models.length > 0 && (
              <div className="al-field">
                <label className="al-label">Model</label>
                <select
                  className="al-select"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                >
                  <option value="">— default —</option>
                  {models.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
            )}

            {/* Advanced params */}
            <div className="al-params">
              <button
                className="al-params__toggle"
                onClick={() => setParamsOpen((v) => !v)}
              >
                <span className="al-label">Parameters</span>
                {paramsOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </button>

              {paramsOpen && (
                <div className="al-params__body">
                  {mode === 'inpaint' && (
                    <>
                      <div className="al-field">
                        <label className="al-label">Start (s)</label>
                        <input
                          className="al-input"
                          type="number"
                          min={0}
                          step={0.5}
                          value={inpaintStart}
                          onChange={(e) => setInpaintStart(e.target.value)}
                        />
                      </div>
                      <div className="al-field">
                        <label className="al-label">End (s)</label>
                        <input
                          className="al-input"
                          type="number"
                          min={0}
                          step={0.5}
                          value={inpaintEnd}
                          onChange={(e) => setInpaintEnd(e.target.value)}
                        />
                      </div>
                    </>
                  )}
                  <div className="al-field">
                    <label className="al-label">Temperature</label>
                    <div className="al-slider-row">
                      <input
                        type="range" min={0} max={2} step={0.05}
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        className="al-slider"
                      />
                      <span className="al-slider-val font-mono">{temperature.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {error && <div className="al-error">{error}</div>}

            <button
              className="al-run-btn"
              disabled={!song || submitting || (mode === 'style-transfer' && !styleSong)}
              onClick={launch}
            >
              {submitting
                ? <Loader2 size={14} className="al-spin" />
                : <FlaskConical size={14} />}
              Run {currentMode.label}
            </button>
          </div>

          {/* Right: output */}
          <div className="al-output">
            <div className="al-output__header">
              <span className="al-label">Output</span>
            </div>
            {!jobId ? (
              <div className="al-output__empty">
                <FlaskConical size={32} strokeWidth={0.75} />
                <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>
                  Configure the parameters and press Run
                </p>
              </div>
            ) : (
              <JobResultBanner jobId={jobId} />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
