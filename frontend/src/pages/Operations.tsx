/* ============================================================
   AI RemixMate — Operations
   Library operations: processing status, storage, maintenance.
   The catalogue is operator-curated (see scripts/download.py); there is
   no in-app track acquisition. Add tracks with the operator ingest tool.
   ============================================================ */
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  RotateCcw,
  CheckCircle2,
  Sparkles,
  Scissors,
  CircleDashed,
  HardDrive,
  Trash2,
  Eye,
  BarChart3,
} from 'lucide-react'
import { analysisApi, libraryApi, storageApi } from '@/lib/api'
import { useFeatures } from '@/hooks/useFeatures'
import type { ProcessingStatus, StorageStatus } from '@/types'
import './PageBase.css'
import './Operations.css'

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

// --- Maintenance panel (index builds, calibration, re-analysis) ---

function MaintenancePanel() {
  const [busy, setBusy] = useState<'clap' | 'lufs' | 'reanalyze' | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  async function run(kind: 'clap' | 'lufs' | 'reanalyze') {
    if (kind === 'reanalyze' && !window.confirm(
      'Re-analyze the ENTIRE library? This refreshes stale BPM/key values ' +
      '(e.g. from before the octave-error fix) but can take a long time on ' +
      'large libraries. Continue?',
    )) return
    setBusy(kind)
    setNotice(null)
    try {
      const r =
        kind === 'clap' ? await analysisApi.buildClapIndex()
        : kind === 'lufs' ? await analysisApi.calibrateLufs()
        : await analysisApi.analyzeMissing(true)
      setNotice(`Job started (${r.job_id.slice(0, 8)}…) — track progress in the Inspector panel.`)
    } catch (e) {
      setNotice(e instanceof Error ? e.message : 'Failed to start job.')
    } finally {
      setBusy(null)
    }
  }

  return (
    <section className="ops-processing">
      <div className="ops-queue__header">
        <h2 className="ops-queue__label">
          <Sparkles size={11} style={{ marginRight: 4, verticalAlign: '-1px' }} />
          Library Maintenance
        </h2>
      </div>

      <div className="ops-storage">
        <p className="ops-note text-muted">
          One-off jobs that improve remix and search quality. Build the CLAP index to enable
          text search on Signal Search; calibrate stem LUFS so remixes use loudness targets
          measured from your own library; re-analyze to refresh stale cached BPM/key data.
        </p>

        {notice && <div className="ops-storage__notice">{notice}</div>}

        <div className="ops-storage__actions">
          <button className="ops-job__btn" disabled={busy !== null} onClick={() => run('clap')}>
            <Sparkles size={12} />
            {busy === 'clap' ? 'Starting…' : 'Build CLAP search index'}
          </button>
          <button className="ops-job__btn" disabled={busy !== null} onClick={() => run('lufs')}>
            <BarChart3 size={12} />
            {busy === 'lufs' ? 'Starting…' : 'Calibrate stem loudness (LUFS)'}
          </button>
          <button className="ops-job__btn" disabled={busy !== null} onClick={() => run('reanalyze')}>
            <RotateCcw size={12} />
            {busy === 'reanalyze' ? 'Starting…' : 'Re-analyze entire library'}
          </button>
        </div>
      </div>
    </section>
  )
}

// --- Main page ---

export default function Operations() {
  const { data: features } = useFeatures()
  const operator = features?.operator === true

  return (
    <div className="page-base">
      <header className="page-base__header">
        <HardDrive size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div>
          <h1 className="page-base__title font-display">Operations</h1>
          <p className="page-base__sub text-muted">
            Processing status · storage · library maintenance
          </p>
        </div>
      </header>

      <div className="page-base__body ops-body">
        <section className="ops-panel">
          <p className="ops-note text-muted">
            There is no in-app track acquisition. Import a folder of your music
            with <code>./start.sh scan &lt;folder&gt;</code>. This page covers processing
            status, storage, and library maintenance for what is already loaded.
          </p>
        </section>

        <ProcessingQueuePanel />
        {operator && <StoragePanel />}
        {operator && <MaintenancePanel />}
      </div>
    </div>
  )
}
