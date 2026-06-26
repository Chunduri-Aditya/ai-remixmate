/* ============================================================
   AI RemixMate — Right Inspector Panel
   Live job queue, activity log, and system health.
   Updates in real-time from Zustand (fed by SSE).
   ============================================================ */

import { X, Activity, ListOrdered, Cpu } from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'
import { useAppStore, selectRecentJobs, type ActivityEntry } from '@/stores/appStore'
import { useJobTimer } from '@/hooks/useJobTimer'
import { remixApi } from '@/lib/api'
import type { Job } from '@/types'
import './RightInspector.css'

// --- Job card ---

function jobStatusClass(status: Job['status']): string {
  switch (status) {
    case 'RUNNING':   return 'badge--running'
    case 'COMPLETED': return 'badge--success'
    case 'FAILED':    return 'badge--error'
    case 'CANCELLED': return 'badge--pending'
    case 'PENDING':   return 'badge--pending'
    default:          return 'badge--pending'
  }
}

function formatDuration(createdAt: string, updatedAt: string): string {
  const ms = new Date(updatedAt).getTime() - new Date(createdAt).getTime()
  if (ms < 0) return '—'
  const s = Math.round(ms / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  return `${m}m ${s % 60}s`
}

function RunningTimer({ job }: { job: Job }) {
  const { elapsed, eta } = useJobTimer(job)
  return (
    <span className="text-muted font-mono inspector-job-card__timer" style={{ fontSize: 'var(--text-xs)' }}>
      ⏱ {elapsed}s elapsed{eta !== null ? ` · ~${eta}s left` : ''}
    </span>
  )
}

function JobCard({ job }: { job: Job }) {
  const cancelJob = useAppStore((s) => s.removeJob)

  async function handleRetry() {
    const meta = (job.meta as Record<string, unknown>) ?? {}
    const upsertJob = useAppStore.getState().upsertJob
    try {
      let res: { job_id: string }
      if (job.type === 'dj_remix' && meta.song_a && meta.song_b) {
        res = await remixApi.create({ song_a: meta.song_a as string, song_b: meta.song_b as string })
      } else if (job.type === 'dj_chain' && Array.isArray(meta.songs)) {
        res = await remixApi.chain(meta.songs as string[])
      } else {
        return
      }
      upsertJob({
        job_id: res.job_id, status: 'PENDING', type: job.type,
        progress: 0, message: 'Retrying…',
        created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
      })
    } catch {
      // silently ignore — user sees the existing error card
    }
  }

  return (
    <div className={`inspector-job-card ${job.status === 'RUNNING' ? 'inspector-job-card--running' : ''}`}>
      <div className="inspector-job-card__header">
        <span className="inspector-job-card__type font-mono">{job.type}</span>
        <span className={`badge ${jobStatusClass(job.status)}`}>
          {job.status === 'RUNNING' && (
            <span className="pulse-dot pulse-dot--amber" style={{ marginRight: 4 }} />
          )}
          {job.status}
        </span>
      </div>

      {job.status === 'RUNNING' && (
        <div className="inspector-job-card__progress-bar">
          <div
            className="inspector-job-card__progress-fill"
            style={{ width: `${job.progress}%` }}
          />
        </div>
      )}

      <div className="inspector-job-card__meta">
        <span className="text-muted font-mono" style={{ fontSize: 'var(--text-xs)' }}>
          {job.message || '—'}
        </span>
        {job.status === 'RUNNING' && <RunningTimer job={job} />}
        {job.status !== 'PENDING' && job.status !== 'RUNNING' && (
          <span className="text-muted font-mono" style={{ fontSize: 'var(--text-xs)' }}>
            {formatDuration(job.created_at, job.updated_at)}
          </span>
        )}
      </div>

      {(job.status === 'PENDING' || job.status === 'RUNNING') && (
        <button
          className="inspector-job-card__cancel"
          onClick={() => cancelJob(job.job_id)}
          title="Cancel job"
          aria-label="Cancel job"
        >
          <X size={10} />
        </button>
      )}

      {job.status === 'FAILED' && (job.type === 'dj_remix' || job.type === 'dj_chain') && (
        <button className="inspector-job-card__retry" onClick={handleRetry}>
          ↺ Retry
        </button>
      )}
    </div>
  )
}

// --- Activity entry row ---

function ActivityRow({ entry }: { entry: ActivityEntry }) {
  const levelColor = {
    info:    'var(--color-text-secondary)',
    success: 'var(--color-green-500)',
    warn:    'var(--color-amber-500)',
    error:   'var(--color-crimson-500)',
  }[entry.level]

  const time = new Date(entry.ts).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })

  return (
    <div className="activity-row">
      <span className="activity-row__dot" style={{ backgroundColor: levelColor }} />
      <div className="activity-row__body">
        <span className="activity-row__message">{entry.message}</span>
        <span className="activity-row__time font-mono">{time}</span>
      </div>
    </div>
  )
}

// --- System tab ---

function SystemTab() {
  const { apiHealth, sseConnected, uptimeSeconds, machineProfile } = useAppStore(
    useShallow((s) => ({
      apiHealth:      s.apiHealth,
      sseConnected:   s.sseConnected,
      uptimeSeconds:  s.uptimeSeconds,
      machineProfile: s.machineProfile,
    })),
  )

  function formatUptime(s: number): string {
    const h = Math.floor(s / 3600)
    const m = Math.floor((s % 3600) / 60)
    const sec = s % 60
    return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${sec}s` : `${sec}s`
  }

  return (
    <div className="inspector-system">
      <div className="inspector-system__row">
        <span className="text-muted">API</span>
        <span className={`badge ${apiHealth === 'ok' ? 'badge--success' : apiHealth === 'degraded' ? 'badge--running' : 'badge--error'}`}>
          {apiHealth}
        </span>
      </div>
      <div className="inspector-system__row">
        <span className="text-muted">Live stream</span>
        <span className={`badge ${sseConnected ? 'badge--success' : 'badge--error'}`}>
          {sseConnected ? 'connected' : 'off'}
        </span>
      </div>
      {uptimeSeconds > 0 && (
        <div className="inspector-system__row">
          <span className="text-muted">Uptime</span>
          <span className="font-mono text-secondary" style={{ fontSize: 'var(--text-sm)' }}>
            {formatUptime(uptimeSeconds)}
          </span>
        </div>
      )}
      {machineProfile && (
        <>
          <div className="inspector-system__divider" />
          <div className="inspector-system__row">
            <span className="text-muted">Host</span>
            <span className="font-mono text-secondary" style={{ fontSize: 'var(--text-xs)', maxWidth: '160px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {machineProfile.hostname}
            </span>
          </div>
          <div className="inspector-system__row">
            <span className="text-muted">GPU</span>
            <span className={`badge ${machineProfile.gpu_backend === 'cpu' ? 'badge--pending' : 'badge--ice'}`}>
              {machineProfile.gpu_backend.toUpperCase()}
              {machineProfile.gpu_name ? ` · ${machineProfile.gpu_name}` : ''}
            </span>
          </div>
          <div className="inspector-system__row">
            <span className="text-muted">Tier</span>
            <span className={`badge ${machineProfile.tier === 'pro' ? 'badge--ice' : machineProfile.tier === 'high' ? 'badge--success' : 'badge--pending'}`}>
              {machineProfile.tier.toUpperCase()}
            </span>
          </div>
        </>
      )}
    </div>
  )
}

// --- Main component ---

export function RightInspector() {
  const { inspectorTab, setInspectorTab, toggleInspector, activityLog, clearActivity } =
    useAppStore(
      useShallow((s) => ({
        inspectorTab:    s.inspectorTab,
        setInspectorTab: s.setInspectorTab,
        toggleInspector: s.toggleInspector,
        activityLog:     s.activityLog,
        clearActivity:   s.clearActivity,
      })),
    )

  const recentJobs = useAppStore(useShallow(selectRecentJobs))
  const activeJobCount = recentJobs.filter(
    (j) => j.status === 'RUNNING' || j.status === 'PENDING',
  ).length

  return (
    <aside className="right-inspector">
      {/* Header */}
      <div className="right-inspector__header">
        <span className="right-inspector__title font-display">Inspector</span>
        <button
          className="right-inspector__close"
          onClick={toggleInspector}
          title="Close inspector"
          aria-label="Close inspector"
        >
          <X size={14} />
        </button>
      </div>

      {/* Tabs */}
      <div className="right-inspector__tabs" role="tablist">
        <button
          className={`inspector-tab ${inspectorTab === 'jobs' ? 'inspector-tab--active' : ''}`}
          onClick={() => setInspectorTab('jobs')}
          role="tab"
          aria-selected={inspectorTab === 'jobs'}
        >
          <ListOrdered size={12} />
          Jobs
          {activeJobCount > 0 && (
            <span className="inspector-tab__badge">{activeJobCount}</span>
          )}
        </button>
        <button
          className={`inspector-tab ${inspectorTab === 'activity' ? 'inspector-tab--active' : ''}`}
          onClick={() => setInspectorTab('activity')}
          role="tab"
          aria-selected={inspectorTab === 'activity'}
        >
          <Activity size={12} />
          Activity
        </button>
        <button
          className={`inspector-tab ${inspectorTab === 'system' ? 'inspector-tab--active' : ''}`}
          onClick={() => setInspectorTab('system')}
          role="tab"
          aria-selected={inspectorTab === 'system'}
        >
          <Cpu size={12} />
          System
        </button>
      </div>

      {/* Tab content */}
      <div className="right-inspector__content scroll-y">
        {inspectorTab === 'jobs' && (
          <div className="inspector-jobs">
            {recentJobs.length === 0 ? (
              <div className="inspector-empty">
                <span className="text-muted">No recent jobs</span>
              </div>
            ) : (
              recentJobs.map((job) => <JobCard key={job.job_id} job={job} />)
            )}
          </div>
        )}

        {inspectorTab === 'activity' && (
          <div className="inspector-activity">
            {activityLog.length > 0 && (
              <button
                className="inspector-activity__clear text-muted"
                onClick={clearActivity}
              >
                Clear
              </button>
            )}
            {activityLog.length === 0 ? (
              <div className="inspector-empty">
                <span className="text-muted">No activity yet</span>
              </div>
            ) : (
              activityLog.map((entry) => <ActivityRow key={entry.id} entry={entry} />)
            )}
          </div>
        )}

        {inspectorTab === 'system' && <SystemTab />}
      </div>
    </aside>
  )
}
