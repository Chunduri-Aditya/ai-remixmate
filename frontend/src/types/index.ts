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
  stems?: string[]          // e.g. ['vocals', 'drums', 'bass', 'other']
}

export interface LibraryStats {
  total_songs: number
  indexed_songs: number
  stems_split: number
  total_size_mb: number
}

// --- Analysis ---

export interface TransitionPlan {
  exit_bar_a: number
  entry_bar_b: number
  transition_bars: number
  stretch_ratio: number
  exit_time_a?: number      // seconds in Song A's timeline
  entry_time_b?: number     // seconds in Song B's timeline
  key_compatible?: boolean
}

export interface CompatibilityResult {
  song_a: string
  song_b: string
  compatible: boolean
  overall: number           // 0–1 composite score
  bpm_score: number         // 0–1
  key_score: number         // 0–1
  energy_score: number      // 0–1
  bpm_a: number
  bpm_b: number
  camelot_a?: string
  camelot_b?: string
  genre_a?: string
  genre_b?: string
  transition_plan?: TransitionPlan
}

export interface Recommendation {
  name: string
  bpm?: number
  bpm_score?: number
  overall?: number
  score?: number
  reason?: string
}

/** Result row from /library/similar — 35-dim RAG vector match. */
export interface SimilarTrack {
  name: string
  score: number
  bpm?: number
  key?: string
  mode?: string
  camelot?: string
  genre?: string
  breakdown?: Record<string, number>
}

// --- Remix ---

export interface DJRemixRequest {
  song_a: string
  song_b: string
  transition_duration?: number
  transition_bars?: number
  effects?: string[]
  target_bpm?: number
  preset?: string
  transition_effect?: string
  bridge_beat_mode?: string
  bridge_beat_genre?: string
  bridge_beat_intensity?: number
}

export interface DJPreviewRequest {
  song_a: string
  song_b: string
  transition_duration?: number
  transition_bars?: number
  preset?: string
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
  | 'widget'
