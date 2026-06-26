/* ============================================================
   AI RemixMate — Operations / Downloads
   Dynamic download pipeline: single track, batch, or playlist.
   Live queue driven by SSE job events (polling fallback).
   ============================================================ */
import { useEffect, useMemo, useState } from 'react'
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
} from 'lucide-react'
import { downloadApi, jobsApi } from '@/lib/api'
import { useAppStore } from '@/stores/appStore'
import { useJobTimer } from '@/hooks/useJobTimer'
import type { Job } from '@/types'
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

// --- Main page ---

export default function Operations() {
  const [mode, setMode] = useState<Mode>('single')
  const [query, setQuery] = useState('')
  const [name, setName] = useState('')
  const [batchText, setBatchText] = useState('')
  const [playlistUrl, setPlaylistUrl] = useState('')
  const [playlistLimit, setPlaylistLimit] = useState('')
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
        const job = await downloadApi.single(query.trim(), name.trim() || undefined)
        upsertJob(job)
        setQuery('')
        setName('')
      } else if (mode === 'batch') {
        if (batchQueries.length === 0) throw new Error('Enter at least one song (one per line).')
        const newJobs = await downloadApi.batch(batchQueries)
        newJobs.forEach(upsertJob)
        setBatchText('')
      } else {
        if (!playlistUrl.trim()) throw new Error('Enter a playlist URL.')
        const limit = playlistLimit ? parseInt(playlistLimit, 10) : undefined
        const job = await downloadApi.playlist(playlistUrl.trim(), limit)
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

          <button className="ops-submit" disabled={!canSubmit} onClick={submit}>
            {submitting ? <Loader2 size={14} className="ops-spin" /> : <Download size={14} />}
            {mode === 'single' ? 'Download Track' : mode === 'batch' ? `Download ${batchQueries.length || ''} Tracks` : 'Download Playlist'}
          </button>

          <p className="ops-note text-muted">
            Every download lands in the library with Demucs stems (vocals · drums · bass · other) split automatically.
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
                <span className="text-muted">No downloads yet — queue a track to get started</span>
              </div>
            ) : (
              downloadJobs.map((job) => (
                <DownloadJobCard key={job.job_id} job={job} onRetry={retry} onCancel={cancel} />
              ))
            )}
          </div>
        </section>
      </div>
    </div>
  )
}
