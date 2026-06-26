/* ============================================================
   AI RemixMate — Mix Vault
   Completed mix browser: in-page audio player, download,
   metadata (BPM, Camelot, harmonic score, LUFS).
   ============================================================ */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Archive,
  Play,
  Pause,
  Download,
  GitMerge,
  GitBranch,
  Volume2,
  Eye,
  EyeOff,
  ArrowRight,
} from 'lucide-react'
import WaveSurfer from 'wavesurfer.js'
import { jobsApi } from '@/lib/api'
import { useAppStore } from '@/stores/appStore'
import type { Job } from '@/types'
import './PageBase.css'
import './MixVault.css'

const BASE = import.meta.env.VITE_API_BASE || '/api'

function formatDuration(s: number | undefined): string {
  if (!s) return '—'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

function mixTitle(job: Job): string {
  const meta = job.meta as { song_a?: string; song_b?: string; songs?: string[] } | undefined
  if (meta?.song_a && meta?.song_b) return `${meta.song_a} → ${meta.song_b}`
  if (meta?.songs && meta.songs.length > 0) return `Chain: ${meta.songs.slice(0, 3).join(' → ')}${meta.songs.length > 3 ? '…' : ''}`
  return `Mix — ${new Date(job.created_at).toLocaleDateString()}`
}

function WaveformPlayer({ src }: { src: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wsRef        = useRef<WaveSurfer | null>(null)
  const [playing, setPlaying]   = useState(false)
  const [progress, setProgress] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume]     = useState(1)

  useEffect(() => {
    if (!containerRef.current) return
    const ws = WaveSurfer.create({
      container:     containerRef.current,
      waveColor:     '#a78bfa',  // violet-400
      progressColor: '#f59e0b',  // amber-500
      height:        64,
      barWidth:      2,
      barGap:        1,
      barRadius:     2,
      url:           src,
    })
    ws.on('ready', () => setDuration(ws.getDuration()))
    ws.on('timeupdate', (t: number) => {
      const dur = ws.getDuration()
      if (dur) setProgress(t / dur)
    })
    ws.on('finish', () => setPlaying(false))
    wsRef.current = ws
    return () => ws.destroy()
  }, [src])

  function togglePlay() {
    wsRef.current?.playPause()
    setPlaying((p) => !p)
  }

  function onVolumeChange(e: React.ChangeEvent<HTMLInputElement>) {
    const v = parseFloat(e.target.value)
    setVolume(v)
    wsRef.current?.setVolume(v)
  }

  return (
    <div className="mv-waveform-player">
      <div ref={containerRef} className="mv-waveform-canvas" />
      <div className="mv-waveform-controls">
        <button className="mv-icon-btn" onClick={togglePlay}>
          {playing ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <span className="text-muted font-mono" style={{ fontSize: 'var(--text-xs)' }}>
          {formatDuration(progress * duration)} / {formatDuration(duration)}
        </span>
        <Volume2 size={12} style={{ marginLeft: 'auto', color: 'var(--color-text-secondary)' }} />
        <input
          type="range" min={0} max={1} step={0.05}
          value={volume} onChange={onVolumeChange}
          className="mv-volume"
          title="Volume"
        />
      </div>
    </div>
  )
}

interface MixCardProps {
  job: Job
}

function MixCard({ job }: MixCardProps) {
  const [open, setOpen]    = useState(false)
  const result = job.result as {
    stream_url?: string
    output_url?: string
    bpm_a?: number
    bpm_b?: number
    camelot_a?: string
    camelot_b?: string
    harmonic_score?: number
    tempo_ratio?: number
    exit_bar_a?: number
    entry_bar_b?: number
    duration?: number
    lufs?: number
  } | undefined

  const streamUrl = result?.stream_url
    ? `${BASE}${result.stream_url}`
    : result?.output_url
    ? `${BASE}${result.output_url}`
    : null

  const isChain   = job.type === 'dj_chain'
  const TypeIcon  = isChain ? GitBranch : GitMerge

  const harPct = result?.harmonic_score !== undefined
    ? Math.round(result.harmonic_score * 100)
    : null
  const harColor =
    harPct === null ? 'var(--color-text-muted)' :
    harPct >= 80   ? 'var(--color-green-500)'  :
    harPct >= 60   ? 'var(--color-amber-500)'  :
                     'var(--color-crimson-500)'

  return (
    <div className="mv-card">
      <div className="mv-card__top">
        <div className="mv-card__icon">
          <TypeIcon size={16} strokeWidth={1.5} style={{ color: 'var(--color-amber-500)' }} />
        </div>
        <div className="mv-card__info">
          <span className="mv-card__title">{mixTitle(job)}</span>
          <span className="mv-card__date text-muted">
            {new Date(job.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
          </span>
        </div>
        <div className="mv-card__badges">
          {result?.bpm_a && (
            <span className="mv-badge mv-badge--amber">
              {result.bpm_a.toFixed(0)} → {result.bpm_b?.toFixed(0) ?? '?'} bpm
            </span>
          )}
          {result?.camelot_a && (
            <span className="mv-badge mv-badge--violet">
              {result.camelot_a} → {result.camelot_b ?? '?'}
            </span>
          )}
          {harPct !== null && (
            <span className="mv-badge" style={{ background: 'rgba(0,0,0,0.2)', color: harColor }}>
              {harPct}% harmonic
            </span>
          )}
        </div>
        <div className="mv-card__actions">
          {streamUrl && (
            <a className="mv-icon-btn" href={streamUrl} download title="Download mix">
              <Download size={14} />
            </a>
          )}
          <button
            className="mv-icon-btn"
            onClick={() => setOpen((v) => !v)}
            title={open ? 'Collapse' : 'Expand details'}
          >
            {open ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>
      </div>

      {streamUrl && (
        <WaveformPlayer src={streamUrl} />
      )}

      {open && result && (
        <div className="mv-card__detail">
          <div className="mv-detail-grid">
            {result.exit_bar_a   !== undefined && <><span className="text-muted">Exit bar A</span><span className="font-mono">{result.exit_bar_a}</span></>}
            {result.entry_bar_b  !== undefined && <><span className="text-muted">Entry bar B</span><span className="font-mono">{result.entry_bar_b}</span></>}
            {result.tempo_ratio  !== undefined && <><span className="text-muted">Tempo ratio</span><span className="font-mono">{result.tempo_ratio.toFixed(3)}</span></>}
            {result.lufs         !== undefined && <><span className="text-muted">LUFS</span><span className="font-mono">{result.lufs.toFixed(1)}</span></>}
            {result.duration     !== undefined && <><span className="text-muted">Duration</span><span className="font-mono">{formatDuration(result.duration)}</span></>}
          </div>
        </div>
      )}
    </div>
  )
}

export default function MixVault() {
  const navigate = useNavigate()
  const [showPreviews, setShowPreviews] = useState(false)
  const jobs     = useAppStore((s) => s.jobs)
  const upsertJob = useAppStore((s) => s.upsertJob)

  useEffect(() => {
    jobsApi
      .list()
      .then((list) =>
        list
          .filter(
            (j) =>
              (j.type === 'dj_remix' || j.type === 'dj_chain') &&
              j.status === 'COMPLETED',
          )
          .forEach(upsertJob),
      )
      .catch(() => { /* silent — API may not be reachable */ })
  }, [upsertJob])

  const mixes = useMemo(
    () =>
      Object.values(jobs)
        .filter(
          (j) =>
            (j.type === 'dj_remix' || j.type === 'dj_chain') &&
            j.status === 'COMPLETED' &&
            (!j.meta?.preview || showPreviews),
        )
        .sort((a, b) => (b.created_at > a.created_at ? 1 : -1)),
    [jobs, showPreviews],
  )

  const previewCount = useMemo(
    () =>
      Object.values(jobs).filter(
        (j) => (j.type === 'dj_remix') && j.status === 'COMPLETED' && j.meta?.preview,
      ).length,
    [jobs],
  )

  return (
    <div className="page-base">
      <header className="page-base__header">
        <Archive size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div style={{ flex: 1 }}>
          <h1 className="page-base__title font-display">Mix Vault</h1>
          <p className="page-base__sub text-muted">
            {mixes.length} completed mix{mixes.length !== 1 ? 'es' : ''}
          </p>
        </div>
        {previewCount > 0 && (
          <button
            className="mv-toggle-btn"
            onClick={() => setShowPreviews((v) => !v)}
          >
            {showPreviews ? <EyeOff size={13} /> : <Eye size={13} />}
            {showPreviews ? 'Hide' : 'Show'} previews ({previewCount})
          </button>
        )}
      </header>

      <div className="page-base__body mv-body">
        {mixes.length === 0 ? (
          <div className="mv-empty">
            <Archive size={40} strokeWidth={0.75} />
            <p className="font-display" style={{ fontSize: 'var(--text-xl)' }}>Vault is empty</p>
            <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>Completed mixes appear here after rendering.</p>
            <button className="mv-empty__action" onClick={() => navigate('/mix-deck')}>
              Create a mix in Mix Deck <ArrowRight size={13} />
            </button>
          </div>
        ) : (
          <div className="mv-list">
            {mixes.map((job) => (
              <MixCard key={job.job_id} job={job} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
