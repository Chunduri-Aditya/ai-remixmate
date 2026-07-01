/* ============================================================
   AI RemixMate — Mix Deck
   Dual-deck DJ UI: BPM/key display, compatibility score,
   transition preview → full remix job.
   ============================================================ */
import { useState, useMemo, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Sliders,
  Music2,
  Zap,
  Play,
  GitMerge,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Loader2,
} from 'lucide-react'
import { libraryApi, analysisApi, remixApi, audioApi } from '@/lib/api'
import { useAppStore } from '@/stores/appStore'
import { WaveformDeck } from '@/components/WaveformDeck'
import { TransitionTimeline } from '@/components/TransitionTimeline'
import { RemixControls, RemixOptions, REMIX_DEFAULTS } from '@/components/RemixControls'
import { CamelotWheel } from '@/components/CamelotWheel'
import type { SongInfo, CompatibilityResult, SimilarTrack } from '@/types'
import './PageBase.css'
import './MixDeck.css'

const DECK_HEX: Record<'A' | 'B', string> = {
  A: '#f59e0b',  // amber-500
  B: '#38bdf8',  // ice-400
}

function DeckCard({
  label,
  song,
  songInfo,
  songs,
  suggestedSongs,
  onChange,
  audioUrl,
  cueStart,
  cueEnd,
  shortcutTarget,
  loadingSongInfo,
}: {
  label: 'A' | 'B'
  song: string
  songInfo?: SongInfo
  songs: string[]
  suggestedSongs?: string[]
  onChange: (v: string) => void
  audioUrl?: string
  cueStart?: number
  cueEnd?: number
  shortcutTarget?: boolean
  loadingSongInfo?: boolean
}) {
  const accentVar = label === 'A' ? 'var(--color-amber-500)' : 'var(--color-ice-400)'
  const glowVar   = label === 'A' ? 'var(--color-amber-glow)' : 'var(--color-ice-glow)'
  const hexColor  = DECK_HEX[label]

  return (
    <div
      className="md-deck"
      style={{ '--deck-color': accentVar, '--deck-glow': glowVar } as React.CSSProperties}
    >
      <div className="md-deck__label font-display">
        Deck {label}
      </div>

      <div className="md-select-wrap">
        <Music2 size={13} className="md-select-icon" />
        <select
          className="md-select"
          value={song}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="">— select a track —</option>
          {suggestedSongs && suggestedSongs.length > 0 ? (
            <>
              <optgroup label="✦ Similar to Deck A">
                {suggestedSongs.map((n) => (
                  <option key={`sug-${n}`} value={n}>{n}</option>
                ))}
              </optgroup>
              <optgroup label="All tracks">
                {songs.filter((n) => !suggestedSongs.includes(n)).map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </optgroup>
            </>
          ) : (
            songs.map((n) => <option key={n} value={n}>{n}</option>)
          )}
        </select>
      </div>

      {song ? (
        <div className="md-deck__meta">
          <div className="md-meta-row">
            <span className="md-meta-key text-muted">BPM</span>
            <span className="md-meta-val font-mono">
              {loadingSongInfo && !songInfo?.bpm
                ? <Loader2 size={11} className="md-spin" />
                : songInfo?.bpm?.toFixed(1) ?? '—'}
            </span>
          </div>
          <div className="md-meta-row">
            <span className="md-meta-key text-muted">Key</span>
            <span className="md-meta-val font-mono">
              {loadingSongInfo && !songInfo?.key
                ? <Loader2 size={11} className="md-spin" />
                : (
                  <>
                    {songInfo?.key ?? '—'}
                    {songInfo?.camelot && (
                      <span className="md-camelot">{songInfo.camelot}</span>
                    )}
                  </>
                )}
            </span>
          </div>
          <div className="md-meta-row">
            <span className="md-meta-key text-muted">Energy</span>
            <span className="md-meta-val font-mono">
              {loadingSongInfo && songInfo?.energy == null
                ? <Loader2 size={11} className="md-spin" />
                : songInfo?.energy != null
                  ? (
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      {Math.round(songInfo.energy * 100)}
                      <span className="md-energy-bar">
                        <span
                          className="md-energy-fill"
                          style={{ width: `${Math.round(songInfo.energy * 100)}%` }}
                        />
                      </span>
                    </span>
                  )
                  : '—'}
            </span>
          </div>
          <div className="md-meta-row">
            <span className="md-meta-key text-muted">Stems</span>
            <span className={`md-meta-val font-mono ${songInfo?.has_stems ? 'md-green' : 'md-muted'}`}>
              {songInfo?.has_stems ? 'yes' : 'no'}
            </span>
          </div>
        </div>
      ) : (
        <div className="md-deck__empty text-muted">
          No track loaded
        </div>
      )}

      {audioUrl && (
        <WaveformDeck
          src={audioUrl}
          color={hexColor}
          cueStart={cueStart}
          cueEnd={cueEnd}
          shortcutTarget={shortcutTarget}
        />
      )}
    </div>
  )
}

function CompatPanel({ result }: { result: CompatibilityResult }) {
  const pct        = Math.round(result.overall * 100)
  const keyPct     = Math.round(result.key_score * 100)
  const tempoRatio = result.bpm_a && result.bpm_b ? result.bpm_b / result.bpm_a : 1.0
  const tempoOk    = tempoRatio >= 0.9 && tempoRatio <= 1.1

  const color =
    pct >= 80 ? 'var(--color-green-500)' :
    pct >= 60 ? 'var(--color-amber-500)' :
                'var(--color-crimson-500)'

  const verdict = result.compatible
    ? 'These tracks blend well — good harmonic and tempo match.'
    : 'Significant mismatch — mix with caution or adjust tempo.'

  return (
    <div className="md-compat">
      <div className="md-compat__score" style={{ '--score-color': color } as React.CSSProperties}>
        <span className="md-compat__pct font-display">{pct}</span>
        <span className="md-compat__unit text-muted">/ 100</span>
      </div>

      <p className="md-compat__verdict">{verdict}</p>

      <div className="md-compat__rows">
        <div className="md-compat__row">
          <span className="text-muted">Harmonic (key)</span>
          <div className="md-compat__bar">
            <div
              className="md-compat__fill"
              style={{ width: `${keyPct}%`, background: 'var(--color-ice-400)' }}
            />
          </div>
          <span className="font-mono" style={{ fontSize: 'var(--text-xs)' }}>{keyPct}</span>
        </div>
        <div className="md-compat__row">
          <span className="text-muted">Tempo ratio</span>
          <span
            className="font-mono"
            style={{ fontSize: 'var(--text-xs)', color: tempoOk ? 'var(--color-green-500)' : 'var(--color-amber-500)' }}
          >
            {tempoRatio.toFixed(3)}
          </span>
        </div>
        <div className="md-compat__row">
          <span className="text-muted">Camelot</span>
          <span className="font-mono" style={{ fontSize: 'var(--text-xs)' }}>
            {result.camelot_a ?? '?'} → {result.camelot_b ?? '?'}
          </span>
        </div>
        {result.bpm_a !== undefined && result.bpm_b !== undefined && (
          <div className="md-compat__row">
            <span className="text-muted">BPM</span>
            <span className="font-mono" style={{ fontSize: 'var(--text-xs)' }}>
              {result.bpm_a.toFixed(1)} → {result.bpm_b.toFixed(1)}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

function JobStatusBadge({ jobId }: { jobId: string }) {
  const job = useAppStore((s) => s.jobs[jobId])
  if (!job) return null

  const { color, icon: Icon, spin } = (() => {
    switch (job.status) {
      case 'RUNNING':   return { color: 'var(--color-amber-500)', icon: Loader2, spin: true }
      case 'COMPLETED': return { color: 'var(--color-green-500)', icon: CheckCircle2, spin: false }
      case 'FAILED':    return { color: 'var(--color-crimson-500)', icon: AlertTriangle, spin: false }
      default:          return { color: 'var(--color-text-muted)', icon: Clock, spin: false }
    }
  })()

  return (
    <div className="md-job-badge">
      <Icon size={13} style={{ color }} className={spin ? 'md-spin' : ''} />
      <span style={{ color, fontSize: 'var(--text-xs)', fontFamily: 'var(--font-mono)' }}>
        {job.status === 'RUNNING' ? `${job.progress}% — ${job.message}` : job.status.toLowerCase()}
      </span>
      {job.status === 'COMPLETED' && job.result && (job.result as { stream_url?: string }).stream_url && (
        <a
          className="md-job-link"
          href={`${import.meta.env.VITE_API_BASE || '/api'}${(job.result as { stream_url?: string }).stream_url}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Play size={11} /> Listen
        </a>
      )}
    </div>
  )
}

function RemixResultCard({ job }: { job: import('@/types').Job }) {
  const r = job.result as {
    stream_url?: string
    lufs?: number
    duration?: number
    bpm_a?: number
    bpm_b?: number
    harmonic_score?: number
    tempo_ratio?: number
  } | undefined
  if (!r) return null
  const BASE = import.meta.env.VITE_API_BASE || '/api'
  const audioUrl = r.stream_url ? `${BASE}${r.stream_url}` : null

  return (
    <div className="md-result-card">
      <div className="md-result-card__header">
        <CheckCircle2 size={14} className="md-result-card__icon" />
        <span className="font-display">Mix Ready</span>
        {audioUrl && (
          <a className="md-result-card__vault-link" href="/mix-vault">
            Open in Vault →
          </a>
        )}
      </div>
      <div className="md-result-card__stats">
        {r.lufs !== undefined && (
          <><span className="text-muted">LUFS</span><span className="font-mono">{r.lufs.toFixed(1)}</span></>
        )}
        {r.duration !== undefined && (
          <><span className="text-muted">Duration</span><span className="font-mono">{Math.floor(r.duration / 60)}:{String(Math.floor(r.duration % 60)).padStart(2, '0')}</span></>
        )}
        {r.tempo_ratio !== undefined && (
          <><span className="text-muted">Tempo ratio</span><span className="font-mono">{r.tempo_ratio.toFixed(3)}</span></>
        )}
        {r.harmonic_score !== undefined && (
          <><span className="text-muted">Key match</span><span className="font-mono">{Math.round(r.harmonic_score * 100)}%</span></>
        )}
      </div>
      {audioUrl && (
        <audio controls src={audioUrl} className="md-result-card__player" />
      )}
    </div>
  )
}

export default function MixDeck() {
  const [searchParams, setSearchParams] = useSearchParams()

  const [songA, setSongA]           = useState(searchParams.get('song_a') ?? '')
  const [songB, setSongB]           = useState(searchParams.get('song_b') ?? '')
  const [compat, setCompat]         = useState<CompatibilityResult | null>(null)
  const [compatLoading, setCompatLoading] = useState(false)
  const [compatError, setCompatError]     = useState<string | null>(null)
  const [transitionDuration, setTransitionDuration] = useState(32)
  const [targetBpm, setTargetBpm]   = useState('')
  const [remixOpts, setRemixOpts]       = useState<RemixOptions>(REMIX_DEFAULTS)
  const [previewJobId, setPreviewJobId] = useState<string | null>(null)
  const [remixJobId, setRemixJobId]     = useState<string | null>(null)
  const [submitting, setSubmitting]     = useState(false)

  const upsertJob  = useAppStore((s) => s.upsertJob)
  const remixJob   = useAppStore((s) => remixJobId ? s.jobs[remixJobId] : null)

  // Clear URL params after consuming them (clean up the address bar).
  useEffect(() => {
    if (searchParams.get('song_a') || searchParams.get('song_b')) {
      setSearchParams({}, { replace: true })
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const { data: songs = [] } = useQuery<SongInfo[]>({
    queryKey: ['library-atlas'],
    queryFn: libraryApi.list,
    staleTime: 60_000,
  })

  // Fetch full detail for each selected song (includes BPM / key / energy from analysis.json)
  const { data: detailA, isLoading: loadingA } = useQuery<SongInfo>({
    queryKey: ['song-detail', songA],
    queryFn: () => libraryApi.get(songA),
    enabled: !!songA,
    staleTime: 120_000,
  })
  const { data: detailB, isLoading: loadingB } = useQuery<SongInfo>({
    queryKey: ['song-detail', songB],
    queryFn: () => libraryApi.get(songB),
    enabled: !!songB,
    staleTime: 120_000,
  })

  // Fetch similar songs for Deck B when Deck A is loaded
  const { data: similarTracksA = [] } = useQuery<SimilarTrack[]>({
    queryKey: ['similar', songA],
    queryFn: () => analysisApi.similar(songA, 8),
    enabled: !!songA,
    staleTime: 300_000,
  })

  const songNames  = useMemo(() => songs.map((s) => s.name).sort(), [songs])
  const infoA      = useMemo(() => songs.find((s) => s.name === songA), [songs, songA])
  const infoB      = useMemo(() => songs.find((s) => s.name === songB), [songs, songB])

  // Use the detailed fetch (has analysis) if available; fall back to list entry
  const songInfoA = detailA ?? infoA
  const songInfoB = detailB ?? infoB

  // Names of similar songs for Deck B, excluding Deck A's song itself
  const similarNamesForB = useMemo(
    () => similarTracksA.map((t) => t.name).filter((n) => n !== songA),
    [similarTracksA, songA],
  )
  const canCompare = songA && songB && songA !== songB
  const shortcutDeck: 'A' | 'B' | null = songA ? 'A' : songB ? 'B' : null

  // Audio URLs for waveform display.
  const audioUrlA  = songA ? audioApi.streamUrl(songA) : undefined
  const audioUrlB  = songB ? audioApi.streamUrl(songB) : undefined

  // Cue times from transition plan (undefined if plan not available).
  const cueStartA  = compat?.transition_plan?.exit_time_a
  const cueEndA    = compat?.transition_plan?.entry_time_b
  const cueStartB  = compat?.transition_plan?.entry_time_b
  const cueEndB    = cueStartB !== undefined ? cueStartB + (songInfoB?.duration ?? 0) * 0.1 : undefined

  async function checkCompat() {
    if (!canCompare) return
    setCompatError(null)
    setCompatLoading(true)
    try {
      const res = await analysisApi.compatibility(songA, songB)
      if ('overall' in res) setCompat(res as CompatibilityResult)
    } catch (e) {
      setCompatError(e instanceof Error ? e.message : 'Compatibility check failed')
    } finally {
      setCompatLoading(false)
    }
  }

  async function launchPreview() {
    if (!canCompare || submitting) return
    setSubmitting(true)
    try {
      const res = await remixApi.preview({
        song_a: songA,
        song_b: songB,
        transition_duration: transitionDuration,
        transition_bars: remixOpts.transition_bars,
        preset: remixOpts.preset,
      })
      setPreviewJobId(res.job_id)
      upsertJob({ job_id: res.job_id, status: 'PENDING', type: 'dj_remix', progress: 0, message: 'Rendering preview…', created_at: new Date().toISOString(), updated_at: new Date().toISOString(), meta: { preview: true } })
    } catch (e) {
      setCompatError(e instanceof Error ? e.message : 'Preview failed')
    } finally {
      setSubmitting(false)
    }
  }

  async function launchRemix() {
    if (!canCompare || submitting) return
    setSubmitting(true)
    try {
      const res = await remixApi.create({
        song_a: songA,
        song_b: songB,
        transition_duration: transitionDuration,
        target_bpm: targetBpm ? parseFloat(targetBpm) : undefined,
        transition_bars: remixOpts.transition_bars,
        preset: remixOpts.preset,
        transition_effect: remixOpts.transition_effect,
        bridge_beat_mode: remixOpts.bridge_beat_mode,
        bridge_beat_genre: remixOpts.bridge_beat_genre,
        bridge_beat_intensity: remixOpts.bridge_beat_intensity,
      })
      setRemixJobId(res.job_id)
      upsertJob({ job_id: res.job_id, status: 'PENDING', type: 'dj_remix', progress: 0, message: 'Rendering full mix…', created_at: new Date().toISOString(), updated_at: new Date().toISOString() })
    } catch (e) {
      setCompatError(e instanceof Error ? e.message : 'Remix failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="page-base">
      <header className="page-base__header">
        <Sliders size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div>
          <h1 className="page-base__title font-display">Mix Deck</h1>
          <p className="page-base__sub text-muted">
            Harmonic matching · stem-aware crossfade · broadcast-mastered output
          </p>
        </div>
      </header>

      <div className="page-base__body md-body">
        {/* Decks row */}
        <div className="md-decks-row">
          <DeckCard
            label="A"
            song={songA}
            songInfo={songInfoA}
            songs={songNames}
            onChange={(v) => { setSongA(v); setCompat(null) }}
            audioUrl={audioUrlA}
            cueStart={cueStartA}
            cueEnd={cueEndA}
            shortcutTarget={shortcutDeck === 'A'}
            loadingSongInfo={loadingA}
          />

          <div className="md-center">
            <GitMerge size={20} strokeWidth={1.5} style={{ color: 'var(--color-text-faint)' }} />
            <button
              className="md-compat-btn"
              disabled={!canCompare || compatLoading}
              onClick={checkCompat}
            >
              {compatLoading ? <RefreshCw size={12} className="md-spin" /> : <Zap size={12} />}
              Check compatibility
            </button>
          </div>

          <DeckCard
            label="B"
            song={songB}
            songInfo={songInfoB}
            songs={songNames}
            suggestedSongs={similarNamesForB}
            onChange={(v) => { setSongB(v); setCompat(null) }}
            audioUrl={audioUrlB}
            cueStart={cueStartB}
            cueEnd={cueEndB}
            shortcutTarget={shortcutDeck === 'B'}
            loadingSongInfo={loadingB}
          />
        </div>

        {/* Compat result + transition timeline */}
        {compatError && (
          <div className="md-error">{compatError}</div>
        )}
        {compat && <CompatPanel result={compat} />}
        {songA && songB && (
          <div style={{ display: 'flex', justifyContent: 'center', paddingTop: 'var(--space-2)' }}>
            <CamelotWheel keyA={songInfoA?.key} keyB={songInfoB?.key} size={220} />
          </div>
        )}
        {compat && (
          <TransitionTimeline
            result={compat}
            durationA={songInfoA?.duration}
            durationB={songInfoB?.duration}
          />
        )}

        {/* Remix options */}
        {canCompare && (
          <RemixControls value={remixOpts} onChange={setRemixOpts} />
        )}

        {/* Transition controls */}
        {canCompare && (
          <div className="md-controls">
            <h3 className="md-controls__title">Transition Controls</h3>
            <div className="md-controls__row">
              <div className="md-control">
                <label className="md-control__label">Crossfade bars</label>
                <div className="md-slider-row">
                  <input
                    type="range"
                    min={8}
                    max={128}
                    step={8}
                    value={transitionDuration}
                    onChange={(e) => setTransitionDuration(parseInt(e.target.value))}
                    className="md-slider"
                  />
                  <span className="md-slider-val font-mono">{transitionDuration}</span>
                </div>
              </div>
              <div className="md-control">
                <label className="md-control__label">Target BPM (optional)</label>
                <input
                  type="number"
                  className="md-input"
                  placeholder={songInfoA?.bpm?.toFixed(1) ?? 'auto'}
                  value={targetBpm}
                  onChange={(e) => setTargetBpm(e.target.value)}
                  min={60}
                  max={200}
                />
              </div>
            </div>

            <div className="md-action-row">
              <button
                className="md-btn md-btn--secondary"
                disabled={submitting}
                onClick={launchPreview}
              >
                {submitting ? <Loader2 size={14} className="md-spin" /> : <Play size={14} />}
                Preview transition
              </button>
              <button
                className="md-btn md-btn--primary"
                disabled={submitting}
                onClick={launchRemix}
              >
                {submitting ? <Loader2 size={14} className="md-spin" /> : <GitMerge size={14} />}
                Full remix (−14 LUFS)
              </button>
            </div>

            {previewJobId && <JobStatusBadge jobId={previewJobId} />}
            {remixJobId   && <JobStatusBadge jobId={remixJobId} />}
            {remixJob?.status === 'COMPLETED' && <RemixResultCard job={remixJob} />}
          </div>
        )}

        {!songA && !songB && (
          <div className="md-splash">
            <Sliders size={36} strokeWidth={0.75} />
            <p className="font-display" style={{ fontSize: 'var(--text-xl)' }}>
              Load two tracks to start mixing
            </p>
            <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>
              Pick a song in each deck — the engine will lock their downbeats, match tempo, and fade stems independently.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
