/* ============================================================
   AI RemixMate — Operations / Downloads
   Dynamic download pipeline: single track, batch, or playlist.
   Live queue driven by SSE job events (polling fallback).
   ============================================================ */
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Download,
  ListMusic,
  Layers,
  Music2,
  XCircle,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Clock,
  ArrowRight,
  Sparkles,
  Scissors,
  CircleDashed,
  HardDrive,
  Trash2,
  Eye,
} from 'lucide-react'
import { downloadApi, jobsApi, libraryApi, storageApi } from '@/lib/api'
import { useAppStore } from '@/stores/appStore'
import { useJobTimer } from '@/hooks/useJobTimer'
import type { Job, ProcessingStatus, StorageStatus } from '@/types'
import './PageBase.css'
import './Operations.css'

type Mode = 'single' | 'batch' | 'playlist'

// --- Job card ---

function statusMeta(status: Job['status']) {
  switch (status) {
    case 'RUNNING':
      return { color: 'var(--color-amber-500)', icon: Loader2, label: 'running', spin: true }
    case 'COMPLETED':
      return { color: 'var(--color-green-500)', icon: CheckCircle2, label: 'done', spin: false }
    case 'FAILED':
      return { color: 'var(--color-crimson-500)', icon: AlertTriangle, label: 'failed', spin: false }
    case 'CANCELLED':
      return { color: 'var(--color-text-muted)', icon: XCircle, label: 'cancelled', spin: false }
    default:
      return { color: 'var(--color-text-muted)', icon: Clock, label: 'queued', spin: false }
  }
}

function jobTitle(job: Job): string {
  const result = job.result as { name?: string } | undefined
  if (result?.name) return result.name
  const meta = job.meta as { query?: string; url?: string } | undefined
  return meta?.query ?? meta?.url ?? 'download'
}

function DownloadTimer({ job }: { job: Job }) {
  const { elapsed, eta } = useJobTimer(job)
  return (
    <span className="ops-job__timer text-muted font-mono">
      ⏱ {elapsed}s{eta !== null ? ` · ~${eta}s left` : ''}
    </span>
  )
}

function DownloadJobCard({
  job,
  onRetry,
  onCancel,
}: {
  job: Job
  onRetry: (job: Job) => void
  onCancel: (job: Job) => void
}) {
  const { color, icon: Icon, label, spin } = statusMeta(job.status)
  const active = job.status === 'RUNNING' || job.status === 'PENDING'
  const result = job.result as
    | { name?: string; stems?: Record<string, string>; playlist_title?: string; downloaded?: number; total?: number }
    | undefined
  const stems = result?.stems ? Object.keys(result.stems) : []

  return (
    <div className={`ops-job ${active ? 'ops-job--active' : ''}`}>
      <div className="ops-job__head">
        <Icon
          size={14}
          strokeWidth={1.75}
          style={{ color }}
          className={spin ? 'ops-spin' : undefined}
        />
        <span className="ops-job__title" title={jobTitle(job)}>{jobTitle(job)}</span>
        <span className="ops-job__status font-mono" style={{ color }}>
          {active ? `${job.progress}%` : label}
        </span>
      </div>

      {active && (
        <div className="ops-job__bar">
          <div className="ops-job__bar-fill" style={{ width: `${job.progress}%` }} />
        </div>
      )}

      {job.status === 'RUNNING' && <DownloadTimer job={job} />}

      <div className="ops-job__foot">
        <span className="ops-job__msg text-muted">
          {job.status === 'FAILED' ? (job.error ?? job.message ?? 'Download failed') : job.message || '—'}
        </span>
        <div className="ops-job__actions">
          {job.status === 'FAILED' && (
            <button className="ops-job__btn" onClick={() => onRetry(job)} title="Retry download">
              <RotateCcw size={12} /> retry
            </button>
          )}
          {active && (
            <button className="ops-job__btn ops-job__btn--danger" onClick={() => onCancel(job)} title="Cancel job">
              <XCircle size={12} /> cancel
            </button>
          )}
        </div>
      </div>

      {job.status === 'COMPLETED' && (stems.length > 0 || result?.playlist_title) && (
        <div className="ops-job__result">
          {result?.playlist_title && (
            <span className="ops-chip ops-chip--ice">
              {result.downloaded}/{result.total} tracks · {result.playlist_title}
            </span>
          )}
          {stems.map((s) => (
            <span key={s} className="ops-chip">{s}</span>
          ))}
        </div>
      )}
    </div>
  )
}

// --- Processing Queue (live, polls every 1s) ---

function ProcessingBucket({
  label,
  icon: Icon,
  color,
  songs,
}: {
  label: string
  icon: React.ElementType
  color: string
  songs: string[]
}) {
  const [open, setOpen] = useState(false)
  return (
    <div className="ops-bucket">
      <button
        className="ops-bucket__head"
        onClick={() => setOpen((v) => !v)}
        disabled={songs.length === 0}
      >
        <Icon size={14} style={{ color }} />
        <span className="ops-bucket__label">{label}</span>
        <span className="ops-bucket__count font-mono" style={{ color }}>{songs.length}</span>
      </button>
      {open && songs.length > 0 && (
        <ul className="ops-bucket__list">
          {songs.slice(0, 50).map((s) => (
            <li key={s} className="ops-bucket__item text-muted" title={s}>{s}</li>
          ))}
          {songs.length > 50 && (
            <li className="ops-bucket__item text-muted">…and {songs.length - 50} more</li>
          )}
        </ul>
      )}
    </div>
  )
}

function ProcessingQueuePanel() {
  const { data, isError } = useQuery<ProcessingStatus>({
    queryKey: ['processing-status'],
    queryFn: libraryApi.processingStatus,
    refetchInterval: 1000,   // live — updates every second as downloads/analysis land
    refetchIntervalInBackground: true,
  })

  if (isError) return null
  const d = data ?? { fully_processed: [], stems_only: [], analysis_only: [], unprocessed: [], total: 0, generated_at: 0 }

  if (d.total === 0) return null

  return (
    <section className="ops-processing">
      <div className="ops-queue__header">
        <h2 className="ops-queue__label">Processing Status</h2>
        <span className="ops-queue__count font-mono">{d.total} song{d.total === 1 ? '' : 's'} · live</span>
      </div>
      <div className="ops-bucket-grid">
        <ProcessingBucket label="Remix-ready (stems + analyzed)" icon={CheckCircle2} color="var(--color-green-500)" songs={d.fully_processed} />
        <ProcessingBucket label="Stems only — not analyzed" icon={Scissors} color="var(--color-amber-500)" songs={d.stems_only} />
        <ProcessingBucket label="Analyzed only — missing stems" icon={Sparkles} color="var(--color-ice-400)" songs={d.analysis_only} />
        <ProcessingBucket label="Unprocessed" icon={CircleDashed} color="var(--color-text-muted)" songs={d.unprocessed} />
      </div>
    </section>
  )
}

// --- Storage panel (size cap, prune, eviction) ---

function StoragePanel() {
  const [busy, setBusy] = useState<'prune' | 'evict-preview' | 'evict-run' | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [previewNames, setPreviewNames] = useState<string[] | null>(null)

  const { data, refetch, isError } = useQuery<StorageStatus>({
    queryKey: ['storage-status'],
    queryFn: storageApi.status,
    refetchInterval: 5000,
  })

  if (isError || !data) return null
  const storage = data   // narrow once — closures below can't see the guard above

  const pct = storage.cap_gb > 0 ? Math.min(100, (storage.total_size_gb / storage.cap_gb) * 100) : 0

  async function runPrune() {
    setBusy('prune')
    setNotice(null)
    try {
      const r = await storageApi.prune()
      setNotice(
        r.pruned.length > 0
          ? `Pruned full.wav for ${r.pruned.length} song${r.pruned.length === 1 ? '' : 's'} — freed ${r.freed_mb.toFixed(0)} MB.`
          : 'Nothing to prune — every song with complete stems already has its source WAV removed.',
      )
      refetch()
    } catch (e) {
      setNotice(e instanceof Error ? e.message : 'Prune failed.')
    } finally {
      setBusy(null)
    }
  }

  async function previewEvict() {
    setBusy('evict-preview')
    setNotice(null)
    try {
      const r = await storageApi.evict(undefined, true)
      setPreviewNames(r.evicted)
      setNotice(
        r.evicted.length > 0
          ? `Would evict ${r.evicted.length} song${r.evicted.length === 1 ? '' : 's'} (oldest-accessed first) to get back under the ${storage.cap_gb} GB cap.`
          : 'Library is within cap — nothing would be evicted.',
      )
    } catch (e) {
      setNotice(e instanceof Error ? e.message : 'Preview failed.')
    } finally {
      setBusy(null)
    }
  }

  async function runEvict() {
    if (!previewNames || previewNames.length === 0) return
    if (!window.confirm(
      `This will permanently remove ${previewNames.length} song${previewNames.length === 1 ? '' : 's'} ` +
      `(oldest-accessed first) to get back under the ${storage.cap_gb} GB cap. This cannot be undone. Continue?`,
    )) return
    setBusy('evict-run')
    setNotice(null)
    try {
      const r = await storageApi.evict(undefined, false)
      setNotice(`Evicted ${r.evicted.length} song${r.evicted.length === 1 ? '' : 's'} — ${r.size_before_gb.toFixed(1)} GB → ${r.size_after_gb.toFixed(1)} GB.`)
      setPreviewNames(null)
      refetch()
    } catch (e) {
      setNotice(e instanceof Error ? e.message : 'Eviction failed.')
    } finally {
      setBusy(null)
    }
  }

  return (
    <section className="ops-processing">
      <div className="ops-queue__header">
        <h2 className="ops-queue__label">
          <HardDrive size={11} style={{ marginRight: 4, verticalAlign: '-1px' }} />
          Storage
        </h2>
        <span className="ops-queue__count font-mono">
          {data.total_size_gb.toFixed(1)} / {data.cap_gb.toFixed(0)} GB
        </span>
      </div>

      <div className="ops-storage">
        <div className="ops-storage__bar">
          <div
            className={`ops-storage__bar-fill ${!data.within_cap ? 'ops-storage__bar-fill--over' : ''}`}
            style={{ width: `${pct}%` }}
          />
        </div>

        <div className="ops-storage__stats">
          <span className="text-muted">{data.total_songs} songs</span>
          <span className="text-muted">{data.songs_with_full_wav} with source WAV still on disk</span>
          <span className="text-muted">{data.songs_stems_only} stems-only (already pruned)</span>
        </div>

        <p className="ops-note text-muted">
          Library lives at <code>{data.library_dir}</code>. To move it onto an external drive or NAS,
          set <code>library.library_dir</code> in <code>config.local.yaml</code> to an absolute path and restart the API.
          {' '}Auto-eviction on download is currently <strong>{data.auto_evict_on_download ? 'ON' : 'off'}</strong>
          {data.auto_evict_on_download && ' — downloading while over cap can silently remove other old songs.'}
        </p>

        {notice && <div className="ops-storage__notice">{notice}</div>}

        {previewNames && previewNames.length > 0 && (
          <ul className="ops-bucket__list" style={{ padding: 0 }}>
            {previewNames.slice(0, 20).map((n) => (
              <li key={n} className="ops-bucket__item text-muted">{n}</li>
            ))}
            {previewNames.length > 20 && (
              <li className="ops-bucket__item text-muted">…and {previewNames.length - 20} more</li>
            )}
          </ul>
        )}

        <div className="ops-storage__actions">
          <button className="ops-job__btn" disabled={busy !== null} onClick={runPrune}>
            <Scissors size={12} />
            {busy === 'prune' ? 'Pruning…' : 'Prune source WAVs (free space, keeps stems)'}
          </button>
          <button className="ops-job__btn" disabled={busy !== null} onClick={previewEvict}>
            <Eye size={12} />
            {busy === 'evict-preview' ? 'Checking…' : 'Preview eviction'}
          </button>
          {previewNames && previewNames.length > 0 && (
            <button className="ops-job__btn ops-job__btn--danger" disabled={busy !== null} onClick={runEvict}>
              <Trash2 size={12} />
              {busy === 'evict-run' ? 'Evicting…' : `Evict ${previewNames.length} song${previewNames.length === 1 ? '' : 's'} now`}
            </button>
          )}
        </div>
      </div>
    </section>
  )
}

// --- Main page ---

export default function Operations() {
  const navigate = useNavigate()
  const [mode, setMode] = useState<Mode>('single')
  const [query, setQuery] = useState('')
  const [name, setName] = useState('')
  const [batchText, setBatchText] = useState('')
  const [playlistUrl, setPlaylistUrl] = useState('')
  const [playlistLimit, setPlaylistLimit] = useState('')
  const [autoAnalyze, setAutoAnalyze] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const jobs = useAppStore((s) => s.jobs)
  const upsertJob = useAppStore((s) => s.upsertJob)

  // Hydrate job history on first visit — SSE only delivers new events
  useEffect(() => {
    jobsApi.list().then((js) => js.forEach(upsertJob)).catch(() => {})
  }, [upsertJob])

  const downloadJobs = useMemo(
    () =>
      Object.values(jobs)
        .filter((j) => j.type === 'download')
        .sort((a, b) => (b.created_at > a.created_at ? 1 : -1)),
    [jobs],
  )
  const activeCount = downloadJobs.filter(
    (j) => j.status === 'RUNNING' || j.status === 'PENDING',
  ).length

  const batchQueries = batchText
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)

  async function submit() {
    setError(null)
    setSubmitting(true)
    try {
      if (mode === 'single') {
        if (!query.trim()) throw new Error('Enter a song name or URL.')
        const job = await downloadApi.single(query.trim(), name.trim() || undefined, autoAnalyze)
        upsertJob(job)
        setQuery('')
        setName('')
      } else if (mode === 'batch') {
        if (batchQueries.length === 0) throw new Error('Enter at least one song (one per line).')
        const newJobs = await downloadApi.batch(batchQueries, autoAnalyze)
        newJobs.forEach(upsertJob)
        setBatchText('')
      } else {
        if (!playlistUrl.trim()) throw new Error('Enter a playlist URL.')
        const limit = playlistLimit ? parseInt(playlistLimit, 10) : undefined
        const job = await downloadApi.playlist(playlistUrl.trim(), limit, autoAnalyze)
        upsertJob(job)
        setPlaylistUrl('')
        setPlaylistLimit('')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSubmitting(false)
    }
  }

  async function retry(job: Job) {
    const meta = job.meta as { query?: string; url?: string } | undefined
    const q = meta?.query ?? meta?.url
    if (!q) return
    try {
      const newJob = await downloadApi.single(q)
      upsertJob(newJob)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  async function cancel(job: Job) {
    try {
      await jobsApi.cancel(job.job_id)
    } catch {
      // job may have already finished — SSE will reconcile
    }
  }

  const canSubmit =
    !submitting &&
    (mode === 'single'
      ? query.trim().length > 0
      : mode === 'batch'
        ? batchQueries.length > 0
        : playlistUrl.trim().length > 0)

  return (
    <div className="page-base">
      <header className="page-base__header">
        <Download size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div>
          <h1 className="page-base__title font-display">Downloads</h1>
          <p className="page-base__sub text-muted">
            Search, batch and playlist downloads · auto stem separation · live progress
          </p>
        </div>
      </header>

      <div className="page-base__body ops-body">
        {/* --- Input panel --- */}
        <section className="ops-panel">
          <div className="ops-tabs" role="tablist">
            <button
              role="tab"
              aria-selected={mode === 'single'}
              className={`ops-tab ${mode === 'single' ? 'ops-tab--active' : ''}`}
              onClick={() => setMode('single')}
            >
              <Music2 size={13} /> Single
            </button>
            <button
              role="tab"
              aria-selected={mode === 'batch'}
              className={`ops-tab ${mode === 'batch' ? 'ops-tab--active' : ''}`}
              onClick={() => setMode('batch')}
            >
              <Layers size={13} /> Batch
            </button>
            <button
              role="tab"
              aria-selected={mode === 'playlist'}
              className={`ops-tab ${mode === 'playlist' ? 'ops-tab--active' : ''}`}
              onClick={() => setMode('playlist')}
            >
              <ListMusic size={13} /> Playlist
            </button>
          </div>

          {mode === 'single' && (
            <div className="ops-form">
              <input
                className="ops-input"
                placeholder="Song name, artist, or YouTube URL…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && canSubmit && submit()}
              />
              <input
                className="ops-input"
                placeholder="Custom name (optional)"
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && canSubmit && submit()}
              />
            </div>
          )}

          {mode === 'batch' && (
            <div className="ops-form">
              <textarea
                className="ops-input ops-textarea"
                placeholder={'One song per line, e.g.\nEric Prydz - Opus\nDaft Punk - One More Time\nhttps://youtube.com/watch?v=…'}
                rows={6}
                value={batchText}
                onChange={(e) => setBatchText(e.target.value)}
              />
              {batchQueries.length > 0 && (
                <span className="ops-hint text-muted">{batchQueries.length} track{batchQueries.length === 1 ? '' : 's'} queued for download</span>
              )}
            </div>
          )}

          {mode === 'playlist' && (
            <div className="ops-form">
              <input
                className="ops-input"
                placeholder="YouTube / YouTube Music playlist URL…"
                value={playlistUrl}
                onChange={(e) => setPlaylistUrl(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && canSubmit && submit()}
              />
              <input
                className="ops-input"
                type="number"
                min={1}
                max={200}
                placeholder="Track limit (optional, max 200)"
                value={playlistLimit}
                onChange={(e) => setPlaylistLimit(e.target.value)}
              />
            </div>
          )}

          {error && <div className="ops-error">{error}</div>}

          <label className="ops-checkbox">
            <input
              type="checkbox"
              checked={autoAnalyze}
              onChange={(e) => setAutoAnalyze(e.target.checked)}
            />
            <span>Auto-process for quicker remix (stems + BPM/key analysis)</span>
          </label>

          <button className="ops-submit" disabled={!canSubmit} onClick={submit}>
            {submitting ? <Loader2 size={14} className="ops-spin" /> : <Download size={14} />}
            {mode === 'single' ? 'Download Track' : mode === 'batch' ? `Download ${batchQueries.length || ''} Tracks` : 'Download Playlist'}
          </button>

          <p className="ops-note text-muted">
            Every download lands in the library with Demucs stems (vocals · drums · bass · other) split automatically.
            {autoAnalyze
              ? ' BPM/key/structure analysis also runs automatically, so it’s remix-ready immediately.'
              : ' Analysis is off — run it later from Library Atlas or the AI Lab before remixing.'}
          </p>
        </section>

        {/* --- Live queue --- */}
        <section className="ops-queue">
          <div className="ops-queue__header">
            <h2 className="ops-queue__label">Download Queue</h2>
            <span className="ops-queue__count font-mono">
              {activeCount > 0 ? `${activeCount} active` : 'idle'}
            </span>
          </div>

          <div className="ops-queue__list">
            {downloadJobs.length === 0 ? (
              <div className="ops-empty">
                <Download size={28} strokeWidth={1} />
                <span className="text-muted">No download jobs yet — queue a track to get started</span>
                <button className="ops-empty__action" onClick={() => navigate('/library-atlas')}>
                  View Library Atlas <ArrowRight size={13} />
                </button>
              </div>
            ) : (
              downloadJobs.map((job) => (
                <DownloadJobCard key={job.job_id} job={job} onRetry={retry} onCancel={cancel} />
              ))
            )}
          </div>
        </section>

        <ProcessingQueuePanel />
        <StoragePanel />
      </div>
    </div>
  )
}
