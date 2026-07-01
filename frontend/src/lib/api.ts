/* ============================================================
   AI RemixMate — API client
   Thin wrapper around fetch; base URL resolved via Vite proxy.
   All requests go to /api/* which Vite forwards to FastAPI.
   ============================================================ */

import type {
  Job,
  SongInfo,
  LibraryStats,
  ProcessingStatus,
  StorageStatus,
  StoragePruneResult,
  StorageEvictResult,
  CompatibilityResult,
  Recommendation,
  SimilarTrack,
  DJRemixRequest,
  DJPreviewRequest,
  Crate,
  HealthStatus,
} from '@/types'

// In dev the Vite proxy rewrites /api/* → http://localhost:8000/*.
// In static builds (GitHub Pages) VITE_API_BASE points straight at the
// locally running backend, e.g. http://localhost:8000.
const BASE = import.meta.env.VITE_API_BASE || '/api'

export const EVENTS_URL = import.meta.env.VITE_EVENTS_URL || '/events/stream'

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
    const res = await fetch(`${BASE}${path}`, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    })

    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText)
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
  ready: () => get<HealthStatus>('/health/ready'),
}

// --- Library ---

export const libraryApi = {
  // GET /library returns { stats, songs: [...] } — paginated, default 50/page
  list:    ()           =>
    get<{ songs: SongInfo[] }>('/library?per_page=5000').then((r) => r.songs ?? []),
  get:     (name: string) => get<SongInfo>(`/library/${encodeURIComponent(name)}`),
  delete:  (name: string) => del<void>(`/library/${encodeURIComponent(name)}`),
  stats:   ()           => get<LibraryStats>('/library/init'),   // reuses init endpoint for stats
  initRun: (opts: Record<string, unknown>) => post<{ job_id: string }>('/library/init', opts),
  // Segregates the library into fully-processed / stems-only / analysis-only /
  // unprocessed buckets. Cheap file-existence scan — safe to poll every ~1s.
  processingStatus: () => get<ProcessingStatus>('/library/processing-status'),
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

// --- Downloads ---

export const downloadApi = {
  // API expects { query, name, auto_analyze } — query is a search string or URL.
  // auto_analyze defaults true: stems + BPM/key/structure analysis both run
  // automatically so the song is fully remix-ready the moment it lands.
  single:     (query: string, name?: string, autoAnalyze = true) =>
    post<RawJob>('/download', { query, name: name || undefined, auto_analyze: autoAnalyze }).then(normalizeJob),
  batch:      (queries: string[], autoAnalyze = true) =>
    post<RawJob[]>('/download-batch', { queries, auto_analyze: autoAnalyze }).then((js) => js.map(normalizeJob)),
  fromSpotify: (url: string) =>
    post<{ job_id: string }>('/spotify/import', { url }),
  playlist:   (url: string, limit?: number, autoAnalyze = true) =>
    post<RawJob>('/download-playlist', { url, limit: limit || undefined, auto_analyze: autoAnalyze }).then(normalizeJob),
}

// --- Audio streaming URLs (no fetch — just URL builders) ---

export const audioApi = {
  streamUrl: (name: string) =>
    `${BASE}/library/${encodeURIComponent(name)}/audio`,
}

// --- Stems ---

export const stemsApi = {
  split:         (song: string, opts?: Record<string, unknown>) =>
    post<{ job_id: string }>('/stems/split', { song, ...opts }),
  splitBatch:    (songs: string[]) =>
    post<{ job_id: string }>('/stems/split-batch', { songs }),
  compress:      (song: string) =>
    post<{ job_id: string }>('/stems/compress', { song }),
  compressBatch: (songs: string[]) =>
    post<{ job_id: string }>('/stems/compress-batch', { songs }),
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
  rebuildIndex:  () =>
    post<{ job_id: string }>('/index/rebuild', {}),
  // Batch-analyzes every library song that's missing BPM/key/energy data —
  // backs the "Analyze missing" button on Library Atlas. No payload needed;
  // the backend scans has_analysis() across the whole library itself.
  analyzeMissing: () =>
    post<{ job_id: string }>('/library/analyze-missing', {}),
}

// --- Remix ---

export const remixApi = {
  create:  (req: DJRemixRequest) =>
    post<{ job_id: string }>('/dj-remix', req),
  preview: (req: DJPreviewRequest) =>
    post<{ job_id: string }>('/dj-remix/preview', req),
  chain:   (songs: string[], opts?: Record<string, unknown>) =>
    post<{ job_id: string }>('/dj-chain', { songs, ...opts }),
}

// --- Jobs ---

export const jobsApi = {
  list:   () => get<RawJob[]>('/jobs').then((js) => js.map(normalizeJob)),
  get:    (id: string) => get<RawJob>(`/jobs/${id}`).then(normalizeJob),
  cancel: (id: string) => del<void>(`/jobs/${id}`),
}

// --- Crates ---

export const cratesApi = {
  // Backend GET /crates returns { crates: [...] } — an object, not a bare
  // array (same shape mismatch class as the favoritesApi.list bug noted in
  // CLAUDE.md). Unwrapping here is what actually crashed Library Atlas:
  // allCrates.map() inside CratesSection on an object with no .map().
  list:    ()                           => get<{ crates: Crate[] }>('/crates').then((r) => r.crates ?? []),
  create:  (name: string)               => post<Crate>('/crates', { name }),
  rename:  (id: number, name: string)   => patch<Crate>(`/crates/${id}`, { name }),
  delete:  (id: number)                 => del<void>(`/crates/${id}`),
  songs:   (id: number)                 => get<string[]>(`/crates/${id}/songs`),
  addSong: (id: number, name: string)   => post<void>(`/crates/${id}/songs`, { name }),
  removeSong: (id: number, name: string) =>
    del<void>(`/crates/${id}/songs/${encodeURIComponent(name)}`),
}

// --- Tags ---

export const tagsApi = {
  list:      ()                             => get<string[]>('/tags'),
  songTags:  (name: string)                 => get<string[]>(`/library/${encodeURIComponent(name)}/tags`),
  addTag:    (name: string, tag: string)    =>
    post<void>(`/library/${encodeURIComponent(name)}/tags`, { tag }),
  removeTag: (name: string, tag: string)    =>
    del<void>(`/library/${encodeURIComponent(name)}/tags/${encodeURIComponent(tag)}`),
}

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

// --- AI / Generative ---

export const aiApi = {
  models: () => get<string[]>('/ai/models'),
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
