/* ============================================================
   AI RemixMate — Mission Control
   Live command-room dashboard: system health, active jobs,
   library stats, recent activity. Updates via Zustand + SSE.
   ============================================================ */

import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useShallow } from 'zustand/react/shallow'
import {
  Activity,
  Cpu,
  Music2,
  Zap,
  GitBranch,
  Clock,
  RefreshCw,
  ChevronRight,
  type LucideIcon,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useAppStore, selectActiveJobs, selectRecentJobs } from '@/stores/appStore'
import { healthApi, jobsApi, libraryApi } from '@/lib/api'
import type { Job } from '@/types'
import './MissionControl.css'

// --- Stat card ---

interface StatCardProps {
  label: string
  value: string | number
  sub?: string
  accent?: 'amber' | 'ice' | 'green' | 'crimson' | 'default'
  icon: LucideIcon
  loading?: boolean
  onClick?: () => void
}

function StatCard({ label, value, sub, accent = 'default', icon: Icon, loading, onClick }: StatCardProps) {
  const accentClass = {
    amber:   'mc-stat-card--amber',
    ice:     'mc-stat-card--ice',
    green:   'mc-stat-card--green',
    crimson: 'mc-stat-card--crimson',
    default: '',
  }[accent]

  return (
    <div
      className={`mc-stat-card ${accentClass} ${onClick ? 'mc-stat-card--clickable' : ''}`}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => e.key === 'Enter' && onClick() : undefined}
    >
      <div className="mc-stat-card__icon">
        <Icon size={14} strokeWidth={1.5} />
      </div>
      <div className="mc-stat-card__body">
        <span className="mc-stat-card__label">{label}</span>
        {loading ? (
          <div className="skeleton" style={{ width: 60, height: 24, marginTop: 2 }} />
        ) : (
          <span className="mc-stat-card__value font-display">{value}</span>
        )}
        {sub && !loading && (
          <span className="mc-stat-card__sub">{sub}</span>
        )}
      </div>
      {onClick && <ChevronRight size={12} className="mc-stat-card__arrow" />}
    </div>
  )
}

// --- Job row ---

function jobStatusColor(status: Job['status']): string {
  switch (status) {
    case 'RUNNING':   return 'var(--color-amber-500)'
    case 'COMPLETED': return 'var(--color-green-500)'
    case 'FAILED':    return 'var(--color-crimson-500)'
    default:          return 'var(--color-text-muted)'
  }
}

function JobRow({ job }: { job: Job }) {
  return (
    <div className="mc-job-row">
      <span
        className="mc-job-row__dot"
        style={{ backgroundColor: jobStatusColor(job.status) }}
      />
      <div className="mc-job-row__info">
        <span className="mc-job-row__type font-mono">{job.type}</span>
        <span className="mc-job-row__msg text-muted">{job.message || '—'}</span>
      </div>
      {job.status === 'RUNNING' && (
        <div className="mc-job-row__progress">
          <div className="mc-job-row__progress-fill" style={{ width: `${job.progress}%` }} />
        </div>
      )}
      <span
        className="mc-job-row__status font-mono"
        style={{ color: jobStatusColor(job.status) }}
      >
        {job.status === 'RUNNING' ? `${job.progress}%` : job.status.toLowerCase()}
      </span>
    </div>
  )
}

// --- Quick-action button ---

interface QuickActionProps {
  label: string
  description: string
  icon: LucideIcon
  accent?: 'amber' | 'ice' | 'green'
  onClick: () => void
}

function QuickAction({ label, description, icon: Icon, accent = 'amber', onClick }: QuickActionProps) {
  const colorMap = {
    amber: 'var(--color-amber-500)',
    ice:   'var(--color-ice-400)',
    green: 'var(--color-green-500)',
  }
  const glowMap = {
    amber: 'var(--color-amber-glow)',
    ice:   'var(--color-ice-glow)',
    green: 'var(--color-green-glow)',
  }
  return (
    <button
      className="mc-quick-action"
      onClick={onClick}
      style={{
        '--qa-color': colorMap[accent],
        '--qa-glow': glowMap[accent],
      } as React.CSSProperties}
    >
      <div className="mc-quick-action__icon">
        <Icon size={18} strokeWidth={1.5} />
      </div>
      <div className="mc-quick-action__text">
        <span className="mc-quick-action__label">{label}</span>
        <span className="mc-quick-action__desc">{description}</span>
      </div>
      <ChevronRight size={12} className="mc-quick-action__arrow" />
    </button>
  )
}

// --- Completion sparkline ---

const SPARKLINE_BUCKETS = 12
const BUCKET_MS = 5 * 60 * 1000   // 5-minute buckets → 60-min window

function Sparkline({ timestamps }: { timestamps: number[] }) {
  const now = Date.now()
  const buckets = Array.from({ length: SPARKLINE_BUCKETS }, (_, i) => {
    const bucketStart = now - (SPARKLINE_BUCKETS - i) * BUCKET_MS
    const bucketEnd   = bucketStart + BUCKET_MS
    return timestamps.filter((t) => t >= bucketStart && t < bucketEnd).length
  })
  const max = Math.max(...buckets, 1)
  const W = 120
  const H = 32
  const barW = Math.floor(W / SPARKLINE_BUCKETS) - 1
  return (
    <svg width={W} height={H} style={{ display: 'block', overflow: 'visible' }}>
      {buckets.map((count, i) => {
        const barH = Math.max(2, Math.round((count / max) * (H - 4)))
        return (
          <rect
            key={i}
            x={i * (barW + 1)}
            y={H - barH}
            width={barW}
            height={barH}
            fill={count > 0 ? 'var(--color-green-500)' : 'var(--color-bg-overlay)'}
            rx={1}
            opacity={count > 0 ? 1 : 0.5}
          />
        )
      })}
    </svg>
  )
}

// --- Heartbeat widget ---

function HeartbeatWidget() {
  const { apiHealth, sseConnected, uptimeSeconds } = useAppStore(
    useShallow((s) => ({
      apiHealth: s.apiHealth,
      sseConnected: s.sseConnected,
      uptimeSeconds: s.uptimeSeconds,
    })),
  )

  function formatUptime(s: number): string {
    if (s === 0) return 'unknown'
    const h = Math.floor(s / 3600)
    const m = Math.floor((s % 3600) / 60)
    return h > 0 ? `${h}h ${m}m` : `${m}m`
  }

  const healthy = apiHealth === 'ok'

  return (
    <div className={`mc-heartbeat ${healthy ? 'mc-heartbeat--ok' : 'mc-heartbeat--warn'}`}>
      <div className="mc-heartbeat__pulse">
        <Activity size={14} />
        <span className={`pulse-dot ${healthy ? 'pulse-dot--green' : 'pulse-dot--amber'}`} />
      </div>
      <div className="mc-heartbeat__info">
        <span className="mc-heartbeat__status font-display">
          {healthy ? 'System Online' : apiHealth === 'degraded' ? 'Degraded' : 'Offline'}
        </span>
        <span className="mc-heartbeat__sub text-muted">
          {sseConnected ? 'Live stream · ' : 'Polling · '}
          uptime {formatUptime(uptimeSeconds)}
        </span>
      </div>
    </div>
  )
}

// --- Main page ---

export default function MissionControl() {
  const navigate = useNavigate()
  const activeJobs      = useAppStore(useShallow(selectActiveJobs))
  const recentJobs      = useAppStore(useShallow(selectRecentJobs))
  const completionLog   = useAppStore((s) => s.completionLog)

  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: healthApi.live,
    refetchInterval: 10_000,
  })

  const { data: librarySongs, isLoading: statsLoading } = useQuery({
    queryKey: ['library-stats'],
    queryFn: libraryApi.list,
    refetchInterval: 30_000,
  })

  const { data: jobs } = useQuery({
    queryKey: ['jobs-init'],
    queryFn: jobsApi.list,
    refetchInterval: false,   // SSE handles updates; this is just initial hydration
    staleTime: 0,
  })

  // Hydrate jobs into store on first load (effect — never set state during render)
  const setJobs = useAppStore((s) => s.setJobs)
  useEffect(() => {
    if (jobs) setJobs(jobs)
  }, [jobs, setJobs])

  const songCount = librarySongs?.length ?? 0

  return (
    <div className="mc-page">
      {/* Page header */}
      <header className="mc-header">
        <div>
          <h1 className="mc-header__title font-display">Mission Control</h1>
          <p className="mc-header__sub text-muted">
            System health · Active pipeline · Quick actions
          </p>
        </div>
        <HeartbeatWidget />
      </header>

      <div className="mc-body">
        {/* Stats row */}
        <section className="mc-section">
          <h2 className="mc-section__label">Overview</h2>
          <div className="mc-stats-grid">
            <StatCard
              label="Library"
              value={statsLoading ? '—' : songCount}
              sub="songs indexed"
              accent="ice"
              icon={Music2}
              loading={statsLoading}
              onClick={() => navigate('/library-atlas')}
            />
            <StatCard
              label="Active Jobs"
              value={activeJobs.length}
              sub={activeJobs.length > 0 ? 'processing now' : 'all clear'}
              accent={activeJobs.length > 0 ? 'amber' : 'green'}
              icon={Zap}
            />
            <StatCard
              label="API"
              value={health?.status ?? (healthLoading ? '…' : 'unknown')}
              sub="backend status"
              accent={health?.status === 'ok' ? 'green' : 'crimson'}
              icon={Cpu}
              loading={healthLoading}
            />
            {/* Sparkline card — replaces plain "Completed" StatCard */}
            <div className="mc-stat-card">
              <div className="mc-stat-card__icon">
                <GitBranch size={14} strokeWidth={1.5} />
              </div>
              <div className="mc-stat-card__body">
                <span className="mc-stat-card__label">Completions</span>
                <span className="mc-stat-card__value font-display" style={{ fontSize: 'var(--text-lg)' }}>
                  {completionLog.length}
                </span>
                <span className="mc-stat-card__sub">last 60 min</span>
                <div style={{ marginTop: 'var(--space-1)' }}>
                  <Sparkline timestamps={completionLog} />
                </div>
              </div>
            </div>
          </div>
        </section>

        <div className="mc-columns">
          {/* Active / recent jobs */}
          <section className="mc-section mc-section--jobs">
            <div className="mc-section__header">
              <h2 className="mc-section__label">
                <Clock size={12} style={{ display: 'inline', marginRight: 6 }} />
                Recent Jobs
              </h2>
              <button
                className="mc-section__action text-muted"
                onClick={() => navigate('/operations')}
              >
                View all
              </button>
            </div>
            <div className="mc-jobs-list">
              {recentJobs.length === 0 ? (
                <div className="mc-empty">
                  <span className="text-muted">No jobs yet — start a mix or download</span>
                </div>
              ) : (
                recentJobs.slice(0, 8).map((job) => <JobRow key={job.job_id} job={job} />)
              )}
            </div>
          </section>

          {/* Quick actions */}
          <section className="mc-section mc-section--actions">
            <h2 className="mc-section__label">Quick Actions</h2>
            <div className="mc-quick-actions">
              <QuickAction
                label="Mix two tracks"
                description="DJ-style crossfade with harmonic matching"
                icon={Zap}
                accent="amber"
                onClick={() => navigate('/mix-deck')}
              />
              <QuickAction
                label="Browse Library"
                description="Search, filter and analyse your songs"
                icon={Music2}
                accent="ice"
                onClick={() => navigate('/library-atlas')}
              />
              <QuickAction
                label="AI Lab"
                description="Style transfer, inpainting, generation"
                icon={RefreshCw}
                accent="green"
                onClick={() => navigate('/ai-lab')}
              />
              <QuickAction
                label="Build a Set"
                description="Sequence songs for a DJ set or playlist"
                icon={GitBranch}
                accent="ice"
                onClick={() => navigate('/set-builder')}
              />
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
