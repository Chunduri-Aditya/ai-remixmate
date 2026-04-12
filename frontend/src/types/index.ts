/* ============================================================
   AI RemixMate — Shared TypeScript types
   Mirror of FastAPI schemas; kept in sync manually.
   ============================================================ */

// --- Job types ---

export type JobStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED'

export interface Job {
  job_id: string
  status: JobStatus
  type: string
  progress: number          // 0–100
  message: string
  created_at: string        // ISO
  updated_at: string        // ISO
  result?: Record<string, unknown>
  error?: string
  meta?: Record<string, unknown>
}

// --- Library ---

export interface SongInfo {
  name: string
  path: string
  has_stems: boolean
  has_analysis: boolean
  bpm?: number
  key?: string
  camelot?: string
  duration?: number         // seconds
  genre?: string
  energy?: number           // 0–1
  embedding?: number[]
}

export interface LibraryStats {
  total_songs: number
  indexed_songs: number
  stems_split: number
  total_size_mb: number
}

// --- Analysis ---

export interface CompatibilityResult {
  song_a: string
  song_b: string
  score: number             // 0–1
  harmonic_score: number
  tempo_ratio: number
  camelot_a?: string
  camelot_b?: string
  bpm_a?: number
  bpm_b?: number
  verdict: string
}

export interface Recommendation {
  name: string
  score: number
  reason?: string
}

// --- Remix ---

export interface DJRemixRequest {
  song_a: string
  song_b: string
  transition_duration?: number
  effects?: string[]
  target_bpm?: number
}

export interface DJPreviewRequest {
  song_a: string
  song_b: string
  transition_duration?: number
}

// --- Crates ---

export interface Crate {
  id: number
  name: string
  song_count: number
  created_at: string
}

// --- SSE events ---

export type SSEEventType =
  | 'heartbeat'
  | 'job_created'
  | 'job_updated'
  | 'job_completed'
  | 'job_failed'
  | 'job_cancelled'
  | 'library_changed'
  | 'system_status'

export interface SSEEvent<T = unknown> {
  type: SSEEventType
  data: T
  ts: string                // ISO timestamp
}

export interface HeartbeatData {
  uptime_seconds: number
  active_jobs: number
  api_version: string
  machine_profile?: MachineProfile
}

// --- Machine profile ---

export interface MachineProfile {
  hostname: string
  platform: string
  cpu_model: string
  cpu_cores_physical: number
  cpu_cores_logical: number
  ram_gb: number
  gpu_backend: 'cuda' | 'mps' | 'cpu'
  gpu_name?: string
  gpu_vram_gb?: number
  demucs_device: string
  recommended_batch_size: number
  tier: 'low' | 'mid' | 'high' | 'pro'
}

// --- Health ---

export interface HealthStatus {
  status: 'ok' | 'degraded' | 'down'
  version?: string
  uptime_seconds?: number
}

// --- Navigation ---

export type NavDestination =
  | 'mission-control'
  | 'library-atlas'
  | 'mix-deck'
  | 'set-builder'
  | 'signal-search'
  | 'ai-lab'
  | 'mix-vault'
  | 'operations'
