/* ============================================================
   AI RemixMate — Global Zustand store
   Lightweight client-side state: navigation, jobs, health, SSE.
   Server state (library data, etc.) lives in React Query cache.
   ============================================================ */

import { create } from 'zustand'
import type { Job, NavDestination, HealthStatus, MachineProfile } from '@/types'

interface AppState {
  // --- Navigation ---
  activeNav: NavDestination
  setActiveNav: (dest: NavDestination) => void

  // --- Health / connection ---
  apiHealth: 'unknown' | 'ok' | 'degraded' | 'down'
  sseConnected: boolean
  uptimeSeconds: number
  machineProfile: MachineProfile | null
  setApiHealth: (h: HealthStatus['status'] | 'unknown') => void
  setSseConnected: (v: boolean) => void
  setUptimeSeconds: (v: number) => void
  setMachineProfile: (p: MachineProfile) => void

  // --- Live jobs (from SSE or polling) ---
  jobs: Record<string, Job>    // keyed by job_id
  upsertJob: (job: Job) => void
  removeJob: (id: string) => void
  setJobs: (jobs: Job[]) => void

  // --- Activity log (recent SSE events, human-readable) ---
  activityLog: ActivityEntry[]
  pushActivity: (entry: Omit<ActivityEntry, 'id' | 'ts'>) => void
  clearActivity: () => void

  // --- Inspector panel ---
  inspectorTab: 'jobs' | 'activity' | 'system'
  setInspectorTab: (tab: AppState['inspectorTab']) => void

  // --- Right inspector collapse state ---
  inspectorOpen: boolean
  toggleInspector: () => void
}

export interface ActivityEntry {
  id: string
  ts: string
  level: 'info' | 'success' | 'warn' | 'error'
  message: string
  job_id?: string
}

let _activitySeq = 0

export const useAppStore = create<AppState>((set, get) => ({
  // Navigation
  activeNav: 'mission-control',
  setActiveNav: (dest) => set({ activeNav: dest }),

  // Health
  apiHealth: 'unknown',
  sseConnected: false,
  uptimeSeconds: 0,
  machineProfile: null,
  setApiHealth: (h) => set({ apiHealth: h }),
  setSseConnected: (v) => set({ sseConnected: v }),
  setUptimeSeconds: (v) => set({ uptimeSeconds: v }),
  setMachineProfile: (p) => set({ machineProfile: p }),

  // Jobs
  jobs: {},
  upsertJob: (job) =>
    set((s) => ({ jobs: { ...s.jobs, [job.job_id]: job } })),
  removeJob: (id) =>
    set((s) => {
      const next = { ...s.jobs }
      delete next[id]
      return { jobs: next }
    }),
  setJobs: (jobs) =>
    set({ jobs: Object.fromEntries(jobs.map((j) => [j.job_id, j])) }),

  // Activity
  activityLog: [],
  pushActivity: (entry) => {
    const id = `act_${++_activitySeq}`
    const ts = new Date().toISOString()
    const full: ActivityEntry = { id, ts, ...entry }
    set((s) => ({
      activityLog: [full, ...s.activityLog].slice(0, 200),   // keep latest 200
    }))
  },
  clearActivity: () => set({ activityLog: [] }),

  // Inspector
  inspectorTab: 'jobs',
  setInspectorTab: (tab) => set({ inspectorTab: tab }),
  inspectorOpen: true,
  toggleInspector: () => set((s) => ({ inspectorOpen: !s.inspectorOpen })),
}))

// --- Selectors (convenience) ---

export const selectActiveJobs = (s: AppState) =>
  Object.values(s.jobs).filter((j) => j.status === 'RUNNING' || j.status === 'PENDING')

export const selectRecentJobs = (s: AppState) =>
  Object.values(s.jobs)
    .sort((a, b) => (b.updated_at > a.updated_at ? 1 : -1))
    .slice(0, 20)
