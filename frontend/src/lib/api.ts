/* ============================================================
   AI RemixMate — API client
   Thin wrapper around fetch; base URL resolved via Vite proxy.
   All requests go to /api/* which Vite forwards to FastAPI.
   ============================================================ */

import type {
  Job,
  SongInfo,
  ProcessingStatus,
  StorageStatus,
  StoragePruneResult,
  StorageEvictResult,
  CompatibilityResult,
  CompatMatchesResponse,
  FeedbackEventType,
  Recommendation,
  SimilarTrack,
  CrateSearchResult,
  DJRemixRequest,
  DJPreviewRequest,
  Crate,
  HealthStatus,
  Features,
} from '@/types'

// In dev the Vite proxy rewrites /api/* → http://localhost:8000/*.
// In static builds (GitHub Pages) VITE_API_BASE points straight at the
// locally running backend, e.g. http://localhost:8000.
import { getAccessToken } from './supabase'

const BASE = import.meta.env.VITE_API_BASE || '/api'

export const EVENTS_URL = import.meta.env.VITE_EVENTS_URL || `${BASE}/events/stream`

const DEFAULT_TIMEOUT_MS = 30_000

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly path: string,
    message: string,
  ) {
    super(`[${status}] ${path}: ${message}`)
    this.name = 'ApiError'
  }
}

// Called on any 401 from a gated route — the session lapsed server-side.
// AuthProvider registers a handler that drops the session and shows the login
// screen. No-op when auth is off.
let onUnauthorized: (() => void) | null = null
export function setUnauthorizedHandler(fn: (() => void) | null): void {
  onUnauthorized = fn
}

// Generic fetcher — throws ApiError on non-2xx or timeout
async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const headers: Record<string, string> = body ? { 'Content-Type': 'application/json' } : {}
    const token = await getAccessToken()
    if (token) headers.Authorization = `Bearer ${token}`

    const res = await fetch(`${BASE}${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
      credentials: 'include', // send the SSE session cookie / any auth cookie
    })

    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText)
      if (res.status === 401) onUnauthorized?.()
      throw new ApiError(res.status, path, text)
    }

    return (await res.json()) as T
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError(0, path, `request timed out after ${timeoutMs}ms`)
    }
    throw err
  } finally {
    clearTimeout(timer)
  }
}

const get  = <T>(path: string) => request<T>('GET', path)
const post = <T>(path: string, body?: unknown) => request<T>('POST', path, body)
const del  = <T>(path: string) => request<T>('DELETE', path)
const patch = <T>(path: string, body?: unknown) => request<T>('PATCH', path, body)

// --- Job normalization ---
// The REST endpoints (/jobs, /download, …) return the raw Pydantic shape
// (lowercase status, `job_type`, progress 0–1, epoch timestamps) while SSE
// frames use the frontend shape (uppercase status, `type`, progress 0–100).
// Normalize everything to the canonical frontend Job here.

type RawJob = Record<string, unknown>

export function normalizeJob(raw: RawJob): Job {
  let status = String(raw.status ?? 'pending')
  if (status.includes('.')) status = status.split('.').pop() as string
  status = status.toUpperCase()
  if (status === 'DONE') status = 'COMPLETED'

  let type = String(raw.type ?? raw.job_type ?? '')
  if (type.includes('.')) type = type.split('.').pop() as string
  type = type.toLowerCase()

  // REST shape (has job_type) reports progress as a 0–1 fraction
  let progress = Number(raw.progress ?? 0)
  if ('job_type' in raw && progress <= 1) progress = progress * 100
  progress = Math.max(0, Math.min(100, Math.round(progress)))

  const toIso = (v: unknown): string =>
    typeof v === 'number'
      ? new Date(v * 1000).toISOString()
      : (v as string) ?? new Date().toISOString()

  return {
    job_id: String(raw.job_id),
    status: status as Job['status'],
    type,
    progress,
    message: (raw.message as string) ?? '',
    created_at: toIso(raw.created_at),
    updated_at: toIso(raw.finished_at ?? raw.started_at ?? raw.updated_at ?? raw.created_at),
    result: (raw.result as Job['result']) ?? undefined,
    error: (raw.error as string) ?? undefined,
    meta: (raw.meta as Job['meta']) ?? undefined,
  }
}

// --- Health ---

export const healthApi = {
  live:  () => get<HealthStatus>('/health/live'),
}

// --- Features ---

export const featuresApi = {
  get: () => get<Features>('/features'),
}

// --- Library ---

export const libraryApi = {
  // GET /library returns { stats, songs: [...] } — paginated, default 50/page
  list:    ()           =>
    get<{ songs: SongInfo[] }>('/library?per_page=5000').then((r) => r.songs ?? []),
  get:     (name: string) => get<SongInfo>(`/library/${encodeURIComponent(name)}`),
  delete:  (name: string) => del<void>(`/library/${encodeURIComponent(name)}`),
  // Segregates the library into fully-processed / stems-only / analysis-only /
  // unprocessed buckets. Cheap file-existence scan — safe to poll every ~1s.
  processingStatus: () => get<ProcessingStatus>('/library/processing-status'),
  // Health-scans every song for damage (no playable audio, undecodable
  // audio, or corrupt meta.json/analysis.json) — read-only, deletes
  // nothing. Poll the returned job_id and read result.broken once COMPLETED.
  scanBroken: () => post<{ job_id: string }>('/library/scan-broken', {}),
  // Bulk-delete counterpart to libraryApi.delete — one request instead of N.
  deleteBatch: (names: string[]) =>
    post<{ deleted: string[]; failed: { name: string; error: string }[] }>(
      '/library/delete-batch',
      { names },
    ),
}

// --- Storage (size cap, pruning, eviction, library location) ---

export const storageApi = {
  status: () => get<StorageStatus>('/library/storage'),
  // Deletes full.wav for every song that already has all 4 stems. Safe —
  // stems are preserved, only the redundant pre-split source is removed.
  prune:  () => post<StoragePruneResult>('/library/storage/prune', {}),
  // dry_run defaults true server-side; pass false explicitly to actually delete.
  evict:  (targetGb?: number, dryRun = true) =>
    post<StorageEvictResult>(
      `/library/storage/evict?dry_run=${dryRun}${targetGb ? `&target_gb=${targetGb}` : ''}`,
      {},
    ),
}

// --- Catalogue ---
// There is no in-app track acquisition. The catalogue is operator-curated CC
// audio (scripts/download.py). Browse it with libraryApi.list / libraryApi.get.

// --- Audio streaming URLs (no fetch — just URL builders) ---

export const audioApi = {
  streamUrl: (name: string) =>
    `${BASE}/library/${encodeURIComponent(name)}/audio`,
  // Download URL for DJ-software cue export (rekordbox XML or Serato MP3).
  exportCuesUrl: (name: string, fmt: 'rekordbox' | 'serato' = 'rekordbox') =>
    `${BASE}/library/${encodeURIComponent(name)}/export-cues?fmt=${fmt}`,
}

// --- Stems ---

export const stemsApi = {
  split:         (song: string, opts?: Record<string, unknown>) =>
    post<{ job_id: string }>('/stems/split', { song, ...opts }),
  stemUrl: (name: string, stem: string) =>
    `${BASE}/library/${encodeURIComponent(name)}/stems/${encodeURIComponent(stem)}`,
}

// --- Analysis ---

export const analysisApi = {
  analyze:       (song: string) =>
    post<{ job_id: string }>('/analyze', { song }),
  compatibility: (song_a: string, song_b: string) =>
    post<CompatibilityResult | { job_id: string }>('/compatibility', { song_a, song_b }),
  recommend:     (name: string, limit = 8) =>
    get<{ song: string; recommendations: Recommendation[] }>(
      `/recommend/${encodeURIComponent(name)}?limit=${limit}`,
    ).then((r) => r.recommendations),
  similar:       (name: string, k = 8) =>
    get<{ source: string; similar: SimilarTrack[] }>(
      `/library/similar/${encodeURIComponent(name)}?k=${k}`,
    ).then((r) => r.similar),
  // Strict ranked matches — full compatibility scorer, min_score on the
  // 0–100 scale, sorted best-first. 409 when the source isn't analyzed yet.
  compatibilityMatches: (name: string, minScore = 85, limit = 50) =>
    get<CompatMatchesResponse>(
      `/compatibility/matches/${encodeURIComponent(name)}?min_score=${minScore}&limit=${limit}`,
    ),
  // CLAP 512-D semantic search (falls back to 35-D index server-side when
  // laion_clap isn't installed). Provide q (free text) and/or name (song).
  crateSearch: (opts: {
    q?: string
    name?: string
    k?: number
    camelot?: string
    bpmMin?: number
    bpmMax?: number
  }) => {
    const p = new URLSearchParams()
    if (opts.q) p.set('q', opts.q)
    if (opts.name) p.set('name', opts.name)
    if (opts.k) p.set('k', String(opts.k))
    if (opts.camelot) p.set('camelot', opts.camelot)
    if (opts.bpmMin != null) p.set('bpm_min', String(opts.bpmMin))
    if (opts.bpmMax != null) p.set('bpm_max', String(opts.bpmMax))
    return get<{ results: CrateSearchResult[]; backend: string }>(
      `/library/crate-search?${p.toString()}`,
    )
  },
  // Build / update the CLAP embedding index (queued job).
  buildClapIndex: (force = false) =>
    post<{ job_id: string }>(`/library/build-clap-index?force=${force}`, {}),
  // Compute per-stem-type LUFS corpus targets (queued job). Run once after
  // library setup so remixes use corpus-derived loudness targets.
  calibrateLufs: () =>
    post<{ job_id: string }>('/library/calibrate-lufs', {}),
  // Batch-analyzes every library song that's missing BPM/key/energy data —
  // backs the "Analyze missing" button on Library Atlas. No payload needed;
  // the backend scans has_analysis() across the whole library itself.
  // force=true re-analyzes the whole library (refreshes stale cached
  // BPM/key values from before analysis-algorithm fixes).
  analyzeMissing: (force = false) =>
    post<{ job_id: string }>(`/library/analyze-missing${force ? '?force=true' : ''}`, {}),
}

// --- Remix ---

export const remixApi = {
  create:  (req: DJRemixRequest) =>
    post<{ job_id: string }>('/dj-remix', req),
  preview: (req: DJPreviewRequest) =>
    post<{ job_id: string }>('/dj-remix/preview', req),
  chain:   (songs: string[], opts?: Record<string, unknown>) =>
    post<{ job_id: string }>('/dj-chain', { songs, ...opts }),
  // 👍/👎 on a rendered mix/preview — trains the spectral cue-matching weights
  feedback: (sessionId: string, liked: boolean) =>
    post<{ updated: boolean; n_updates?: number }>('/dj-remix/feedback', {
      session_id: sessionId,
      liked,
    }),
}

// --- Jobs ---

export interface JobListFilters {
  limit?: number
  /** Backend enum value, e.g. 'dj_remix'. Filtered server-side, before the limit. */
  jobType?: string
  /** Backend enum value, e.g. 'done' (not the uppercase frontend 'COMPLETED'). */
  status?: string
}

export const jobsApi = {
  // Filters are applied server-side on purpose: /jobs returns the newest `limit`
  // jobs, so filtering client-side hides every match that falls outside that
  // window — which is what made Mix Vault look empty against a full job store.
  list: (filters: JobListFilters = {}) => {
    const qs = new URLSearchParams()
    if (filters.limit   !== undefined) qs.set('limit', String(filters.limit))
    if (filters.jobType !== undefined) qs.set('job_type', filters.jobType)
    if (filters.status  !== undefined) qs.set('status', filters.status)
    const suffix = qs.toString() ? `?${qs}` : ''
    return get<RawJob[]>(`/jobs${suffix}`).then((js) => js.map(normalizeJob))
  },
  cancel: (id: string) => del<void>(`/jobs/${id}`),
}

// --- Crates ---

export const cratesApi = {
  // Backend GET /crates returns { crates: [...] } — an object, not a bare
  // array (same shape mismatch class as the favoritesApi.list bug noted in
  // DEVELOPING.md). Unwrapping here is what actually crashed Library Atlas:
  // allCrates.map() inside CratesSection on an object with no .map().
  list:    ()                           => get<{ crates: Crate[] }>('/crates').then((r) => r.crates ?? []),
  create:  (name: string)               => post<Crate>('/crates', { name }),
  delete:  (id: string)                 => del<void>(`/crates/${id}`),
  songs:   (id: string)                 => get<string[]>(`/crates/${id}/songs`),
  // Backend POST /crates/{id}/songs reads `song_name`, not `name` — sending
  // `name` made every add-to-crate from Library Atlas 400 with
  // "'song_name' is required". Same shape-mismatch class as favoritesApi.list.
  addSong: (id: string, name: string)   => post<void>(`/crates/${id}/songs`, { song_name: name }),
  removeSong: (id: string, name: string) =>
    del<void>(`/crates/${id}/songs/${encodeURIComponent(name)}`),
  // PATCH /crates/{id} exists on the backend (routers/crates.py rename_crate) and
  // src/__tests__/api-contract.test.ts pins its wire shape, so it stays.
  rename:  (id: string, name: string)   => patch<Crate>(`/crates/${id}`, { name }),
}

// --- Tags ---

export const tagsApi = {}

// --- Favorites ---

export const favoritesApi = {
  list:   () => get<{ songs: string[]; count: number }>('/favorites').then((r) => r.songs ?? []),
  add:    (name: string) => post<void>(`/favorites/${encodeURIComponent(name)}`),
  remove: (name: string) => del<void>(`/favorites/${encodeURIComponent(name)}`),
}

// --- Setlist ---

export const setlistApi = {
  optimize: (
    songs: Array<{ name: string; bpm?: number; energy?: number; camelot?: string }>,
  ) =>
    post<{ setlist: Array<{ name: string }> }>('/setlist/optimize', {
      tracks: songs.map((s) => ({
        title: s.name,
        artist: '',
        bpm: s.bpm ?? null,
        energy: s.energy ?? 0.5,
        camelot: s.camelot ?? null,
      })),
    }).then((r) => r.setlist.map((t) => t.name)),
}

// --- Feedback telemetry (SELF_IMPROVING_DJ_RESEARCH.md §3, Stage A) ---
// Instrumentation only — this collects the event stream Stage B will later
// analyze to decide whether a learned preference correction (Stage C) is
// even worth building. Nothing here scores or ranks anything.

export const feedbackApi = {
  // Fire-and-forget: callers should NOT await this on the playback/UI hot
  // path — chain .catch(() => {}) at the call site (mirrors the backend's
  // own "log and swallow" posture on a broken feedback pipe) so a network
  // hiccup can never stall or interrupt what the user is actually doing.
  logEvent: (opts: {
    eventType: FeedbackEventType
    songA: string
    songB?: string
    jobId?: string
    positionSec?: number
    context?: Record<string, unknown>
  }) =>
    post<{ event_id: string; logged: boolean }>('/feedback/event', {
      event_type: opts.eventType,
      song_a: opts.songA,
      song_b: opts.songB,
      job_id: opts.jobId,
      position_sec: opts.positionSec,
      context: opts.context,
    }),
}

// --- AI / Generative ---

/** GET /ai/models body (scripts/api/schemas.py ModelStatusResponse). */
export interface AiModelStatus {
  device: string
  mps_allocated_gb: number
  max_vram_gb: number
  models: Array<{
    name: string
    loaded: boolean
    vram_estimate_gb: number
    description: string
    last_used: number
  }>
}

export const aiApi = {
  models: () => get<AiModelStatus>('/ai/models'),
  // Backend StyleTransferRequest requires song_name + description (free-text
  // MusicGen style prompt) — NOT a second track name. See schemas.py.
  styleTransfer: (songName: string, description: string, opts?: Record<string, unknown>) =>
    post<{ job_id: string }>('/ai/style-transfer', { song_name: songName, description, ...opts }),
  // Backend InpaintRequest requires song_a (outgoing) + song_b (incoming).
  inpaint: (songA: string, songB: string, opts?: Record<string, unknown>) =>
    post<{ job_id: string }>('/ai/inpaint', { song_a: songA, song_b: songB, ...opts }),
  // Separate endpoint — was previously (incorrectly) faked via inpaint(tokenize:true).
  tokenize: (songName: string, opts?: Record<string, unknown>) =>
    post<{ job_id: string }>('/ai/tokenize', { song_name: songName, ...opts }),
}
