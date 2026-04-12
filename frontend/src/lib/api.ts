/* ============================================================
   AI RemixMate — API client
   Thin wrapper around fetch; base URL resolved via Vite proxy.
   All requests go to /api/* which Vite forwards to FastAPI.
   ============================================================ */

import type {
  Job,
  SongInfo,
  LibraryStats,
  CompatibilityResult,
  Recommendation,
  DJRemixRequest,
  DJPreviewRequest,
  Crate,
  HealthStatus,
} from '@/types'

const BASE = '/api'

// Generic fetcher — throws on non-2xx
async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`[${res.status}] ${path}: ${text}`)
  }

  return res.json() as Promise<T>
}

const get  = <T>(path: string) => request<T>('GET', path)
const post = <T>(path: string, body?: unknown) => request<T>('POST', path, body)
const del  = <T>(path: string) => request<T>('DELETE', path)
const patch = <T>(path: string, body?: unknown) => request<T>('PATCH', path, body)

// --- Health ---

export const healthApi = {
  live:  () => get<HealthStatus>('/health/live'),
  ready: () => get<HealthStatus>('/health/ready'),
}

// --- Library ---

export const libraryApi = {
  list:    ()           => get<SongInfo[]>('/library'),
  get:     (name: string) => get<SongInfo>(`/library/${encodeURIComponent(name)}`),
  delete:  (name: string) => del<void>(`/library/${encodeURIComponent(name)}`),
  stats:   ()           => get<LibraryStats>('/library/init'),   // reuses init endpoint for stats
  initRun: (opts: Record<string, unknown>) => post<{ job_id: string }>('/library/init', opts),
}

// --- Downloads ---

export const downloadApi = {
  fromUrl:    (url: string, name?: string) =>
    post<{ job_id: string }>('/download', { url, name }),
  fromSpotify: (url: string) =>
    post<{ job_id: string }>('/spotify/import', { url }),
  playlist:   (url: string) =>
    post<{ job_id: string }>('/download-playlist', { url }),
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
}

// --- Analysis ---

export const analysisApi = {
  analyze:       (song: string) =>
    post<{ job_id: string }>('/analyze', { song }),
  compatibility: (song_a: string, song_b: string) =>
    post<CompatibilityResult | { job_id: string }>('/compatibility', { song_a, song_b }),
  recommend:     (name: string) =>
    get<Recommendation[]>(`/recommend/${encodeURIComponent(name)}`),
  similar:       (name: string) =>
    get<Recommendation[]>(`/library/similar/${encodeURIComponent(name)}`),
  rebuildIndex:  () =>
    post<{ job_id: string }>('/index/rebuild', {}),
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
  list:   () => get<Job[]>('/jobs'),
  get:    (id: string) => get<Job>(`/jobs/${id}`),
  cancel: (id: string) => del<void>(`/jobs/${id}`),
}

// --- Crates ---

export const cratesApi = {
  list:    ()                           => get<Crate[]>('/crates'),
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
  list:   ()           => get<string[]>('/favorites'),
  add:    (name: string) => post<void>(`/favorites/${encodeURIComponent(name)}`),
  remove: (name: string) => del<void>(`/favorites/${encodeURIComponent(name)}`),
}

// --- AI / Generative ---

export const aiApi = {
  models:       ()                          => get<string[]>('/ai/models'),
  styleTransfer: (song: string, style: string) =>
    post<{ job_id: string }>('/ai/style-transfer', { song, style }),
  inpaint:      (song: string, opts?: Record<string, unknown>) =>
    post<{ job_id: string }>('/ai/inpaint', { song, ...opts }),
}
