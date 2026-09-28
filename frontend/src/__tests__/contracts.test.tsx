/**
 * B5.7 copy, contracts, dead code (plans/audit-the-project-and-sorted-flask.md).
 *
 * Contract these tests pin (the builder implements against it):
 *
 *  - Download era copy is gone: LibraryAtlas' empty state names the operator
 *    ingest command (`python -m scripts.download`) and has no button;
 *    MissionControl's empty jobs line no longer says "or download"; Operations
 *    no longer talks about auto eviction "on download"; the Widget's
 *    unreachable API message no longer hard codes port 8000; the LeftRail
 *    header comment says nine destinations; smoke.test.ts has no
 *    `type: 'download'` fixture.
 *  - api.ts: EVENTS_URL defaults to `${BASE}/events/stream`; the methods with
 *    no caller are gone while the live ones stay; GET /ai/models is the
 *    backend object and AILab renders its `.models[].name` in the Model
 *    select; a COMPLETED generative job is played from `result.stream_url`.
 *  - MixDeck: the "Crossfade bars" slider is the source of `transition_bars`
 *    on POST /dj-remix, clamped to 8..64; no `target_bpm` control or key.
 *
 * Seam: `fetch` is stubbed globally and routed by path, so the real
 * `@/lib/api` client runs end to end and the bytes on the wire are asserted.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import leftRailSource from '../shell/LeftRail.tsx?raw'
import smokeSource from './smoke.test.ts?raw'
import type { ReactElement } from 'react'

vi.mock('wavesurfer.js', () => ({
  default: {
    create: () => ({
      destroy: () => undefined,
      getDuration: () => 0,
      getCurrentTime: () => 0,
      // wavesurfer's load() returns a Promise; WaveformDeck attaches .catch to it.
      load: () => Promise.resolve(),
      on: () => undefined,
      un: () => undefined,
      play: () => undefined,
      pause: () => undefined,
      playPause: () => undefined,
      seekTo: () => undefined,
      setVolume: () => undefined,
    }),
  },
}))

// jsdom has no layout: a real virtualizer renders zero rows.
vi.mock('@tanstack/react-virtual', () => ({
  useVirtualizer: ({ count }: { count: number }) => ({
    getVirtualItems: () =>
      Array.from({ length: count }, (_, i) => ({ index: i, key: i, start: i * 48, end: (i + 1) * 48, size: 48 })),
    getTotalSize: () => count * 48,
    measureElement: () => {},
  }),
}))

import * as api from '@/lib/api'
import LibraryAtlas from '@/pages/LibraryAtlas'
import MissionControl from '@/pages/MissionControl'
import Operations from '@/pages/Operations'
import Widget from '@/pages/Widget'
import AILab from '@/pages/AILab'
import MixDeck from '@/pages/MixDeck'
import { useAppStore } from '@/stores/appStore'
import type { SongInfo } from '@/types'

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

const FEATURES = {
  auth: false, stem_split: true, generative: true, license_enforce: false, spectral_learning: true, operator: true,
}

const SONG_A: SongInfo = { name: 'Song A', path: '/lib/a', has_stems: true, has_analysis: true, bpm: 124, key: 'A minor', duration: 200 } as SongInfo
const SONG_B: SongInfo = { name: 'Song B', path: '/lib/b', has_stems: true, has_analysis: true, bpm: 126, key: 'E minor', duration: 210 } as SongInfo
const SONGS = [SONG_A, SONG_B]

const PROCESSING = { fully_processed: ['Song A', 'Song B'], stems_only: [], analysis_only: [], unprocessed: [], total: 2, generated_at: 0 }
const STORAGE = {
  library_dir: '/lib/root', outputs_dir: '/out', total_songs: 2, total_size_gb: 1.2, cap_gb: 100, within_cap: true,
  songs_with_full_wav: 1, songs_stems_only: 1, prune_on_download: false, keep_raw_after_separation: true, auto_evict_on_download: true,
}

// The backend shape of GET /ai/models (scripts/api/schemas.py ModelStatusResponse).
const MODELS = {
  device: 'mps',
  mps_allocated_gb: 0.0,
  max_vram_gb: 12.0,
  models: [
    { name: 'musicgen-melody', loaded: false, vram_estimate_gb: 4.0, description: 'MusicGen Melody', last_used: 0 },
    { name: 'vampnet', loaded: false, vram_estimate_gb: 2.0, description: 'VampNet', last_used: 0 },
  ],
}

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: String(status),
    headers: new Headers({ 'content-type': 'application/json' }),
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

interface Seen { method: string; path: string; body?: unknown }
let requests: Seen[] = []

function installFetch(opts: { songs?: SongInfo[]; libraryFails?: boolean } = {}) {
  requests = []
  const songs = opts.songs ?? SONGS
  vi.stubGlobal('fetch', async (input: string | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString()
    const path = url.replace(/^https?:\/\/[^/]+/, '').replace(/^\/api/, '')
    const method = init?.method ?? 'GET'
    const body = init?.body ? JSON.parse(init.body as string) : undefined
    requests.push({ method, path, body })

    if (/^\/features(\?|$)/.test(path)) return jsonResponse(FEATURES)
    if (/^\/library\?/.test(path)) {
      if (opts.libraryFails) return jsonResponse({ detail: 'boom' }, 500)
      return jsonResponse({ songs, stats: { total_songs: songs.length, indexed_songs: songs.length, stems_split: 0, total_size_mb: 0 } })
    }
    if (/^\/library\/processing-status/.test(path)) return jsonResponse(PROCESSING)
    if (/^\/library\/storage(\?|$)/.test(path)) return jsonResponse(STORAGE)
    if (/^\/library\/similar\//.test(path)) return jsonResponse({ source: 'Song A', similar: [] })
    if (/^\/compatibility\/matches\//.test(path)) return jsonResponse({ matches: [] })
    if (/^\/library\/[^/?]+$/.test(path)) {
      const name = decodeURIComponent(path.slice('/library/'.length))
      const found = songs.find((s) => s.name === name)
      return found ? jsonResponse(found) : jsonResponse({ detail: 'not found' }, 404)
    }
    if (/^\/favorites(\?|$)/.test(path)) return jsonResponse({ songs: [], count: 0 })
    if (/^\/crates(\?|$)/.test(path)) return jsonResponse({ crates: [] })
    if (/^\/jobs(\?|$)/.test(path)) return jsonResponse([])
    if (/^\/health\//.test(path)) return jsonResponse({ status: 'ok' })
    if (/^\/ai\/models(\?|$)/.test(path)) return jsonResponse(MODELS)
    if (/^\/ai\/style-transfer/.test(path)) return jsonResponse({ job_id: 'gen-1', status: 'pending', job_type: 'analyze', progress: 0, created_at: 0 }, 202)
    if (/^\/dj-remix(\?|$)/.test(path)) return jsonResponse({ job_id: 'job-remix', status: 'pending', job_type: 'dj_remix', progress: 0, created_at: 0 }, 202)
    return jsonResponse([])
  })
}

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
}

function mount(ui: ReactElement, route = '/', qc = newClient()) {
  const view = render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  )
  return { ...view, qc }
}

async function flush() {
  await act(async () => { await new Promise((r) => setTimeout(r, 0)) })
}

function readSource(relToTests: string): string {
  // Vite ?raw imports keep tsc free of node:* types (no @types/node in this project).
  const sources: Record<string, string> = { '../shell/LeftRail.tsx': leftRailSource, './smoke.test.ts': smokeSource }
  return sources[relToTests]
}

beforeEach(() => {
  vi.stubGlobal('confirm', () => true)
  useAppStore.setState({ jobs: {} })
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

// ---------------------------------------------------------------------------
// Download era copy
// ---------------------------------------------------------------------------

describe('download era copy', () => {
  it('LibraryAtlas empty state names the operator ingest command and has no button', async () => {
    installFetch({ songs: [] })
    const { container } = mount(<LibraryAtlas />, '/library-atlas')
    await waitFor(() => expect(requests.some((r) => /^\/library\?/.test(r.path))).toBe(true))
    await waitFor(() => expect(container.textContent).not.toMatch(/Loading library/i))

    expect(container.textContent, 'empty state must name the ingest command').toMatch(/python -m scripts\.download/)
    expect(container.textContent).not.toMatch(/add songs in Operations/i)
    expect(screen.queryByRole('button', { name: /add songs/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /operations/i })).toBeNull()
  })

  it('MissionControl empty jobs line does not mention download', async () => {
    installFetch()
    const { container } = mount(<MissionControl />, '/mission-control')
    await waitFor(() => expect(container.textContent).toMatch(/no jobs/i))
    expect(container.textContent).not.toMatch(/start a mix or download/i)
    expect(container.textContent).not.toMatch(/\bdownload\b/i)
  })

  it('Operations storage panel no longer describes auto eviction on download', async () => {
    installFetch()
    const { container } = mount(<Operations />, '/operations')
    await waitFor(() => expect(container.textContent).toContain(STORAGE.library_dir))
    expect(container.textContent).not.toMatch(/auto-?eviction on download/i)
    expect(container.textContent).not.toMatch(/downloading while over cap/i)
  })

  it('Widget unreachable API message does not hard code port 8000', async () => {
    installFetch({ libraryFails: true })
    const { container } = mount(<Widget />, '/widget')
    await waitFor(() => expect(container.querySelector('.wgt-error')).not.toBeNull())
    const msg = container.querySelector('.wgt-error')!.textContent ?? ''
    expect(msg.length).toBeGreaterThan(0)
    expect(msg).not.toMatch(/8000/)
  })

  it('LeftRail header comment says nine destinations', () => {
    const header = readSource('../shell/LeftRail.tsx').split('\n').slice(0, 6).join('\n')
    expect(header).not.toMatch(/\b8 destinations\b/)
    expect(header).toMatch(/nine destinations/i)
  })

  it("smoke.test.ts fixture has no type: 'download' job", () => {
    const src = readSource('./smoke.test.ts')
    expect(src).not.toMatch(/type:\s*['"]download['"]/)
  })
})

// ---------------------------------------------------------------------------
// api.ts contracts
// ---------------------------------------------------------------------------

describe('api.ts', () => {
  it('EVENTS_URL defaults to BASE + /events/stream', () => {
    expect(import.meta.env.VITE_EVENTS_URL, 'test env must not pin VITE_EVENTS_URL').toBeUndefined()
    const base = import.meta.env.VITE_API_BASE || '/api'
    expect(api.EVENTS_URL).toBe(`${base}/events/stream`)
  })

  // Every method here has zero callers under frontend/src outside lib/api.ts
  // and __tests__ (grep "Api.<name>" on 2026-09-28).
  const DEAD: Array<[keyof typeof api, string]> = [
    ['healthApi', 'ready'],
    ['libraryApi', 'stats'],
    ['libraryApi', 'initRun'],
    ['stemsApi', 'splitBatch'],
    ['stemsApi', 'compress'],
    ['stemsApi', 'compressBatch'],
    ['analysisApi', 'structure'],
    ['analysisApi', 'rebuildIndex'],
    // cratesApi.rename stays: the backend serves PATCH /crates/{id} and api-contract.test.ts pins it.
    ['tagsApi', 'songTags'],
    ['tagsApi', 'addTag'],
    ['tagsApi', 'removeTag'],
    ['feedbackApi', 'summary'],
  ]
  it.each(DEAD)('%s.%s (no caller) is removed', (group, name) => {
    const obj = api[group] as Record<string, unknown> | undefined
    expect(obj, `${String(group)} itself is gone`).toBeDefined()
    expect(obj).not.toHaveProperty(name)
  })

  // Allow control: the live methods (including the ones B5.3 wired up) stay.
  const LIVE: Array<[keyof typeof api, string]> = [
    ['healthApi', 'live'],
    ['libraryApi', 'list'],
    ['libraryApi', 'get'],
    ['jobsApi', 'list'],
    ['jobsApi', 'cancel'],
    ['remixApi', 'create'],
    ['remixApi', 'preview'],
    ['aiApi', 'models'],
    ['favoritesApi', 'list'],
    ['cratesApi', 'addSong'],
    ['feedbackApi', 'logEvent'],
  ]
  it.each(LIVE)('%s.%s (has callers) is kept', (group, name) => {
    const obj = api[group] as Record<string, unknown>
    expect(typeof obj?.[name]).toBe('function')
  })

  it('aiApi.models resolves to the backend object with a models array', async () => {
    installFetch()
    const res = (await api.aiApi.models()) as unknown as typeof MODELS
    expect(Array.isArray(res)).toBe(false)
    expect(res.device).toBe('mps')
    expect(res.models.map((m) => m.name)).toEqual(['musicgen-melody', 'vampnet'])
  })
})

// ---------------------------------------------------------------------------
// AILab reads .models and stream_url
// ---------------------------------------------------------------------------

describe('AILab', () => {
  it('renders the Model select from the backend object (.models[].name)', async () => {
    installFetch()
    mount(<AILab />, '/ai-lab')
    await waitFor(() => expect(requests.some((r) => /^\/ai\/models/.test(r.path))).toBe(true))
    expect(await screen.findByRole('option', { name: 'musicgen-melody' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'vampnet' })).toBeInTheDocument()
  })

  it('plays a COMPLETED job from result.stream_url under BASE', async () => {
    installFetch()
    const { container } = mount(<AILab />, '/ai-lab')
    await screen.findByRole('option', { name: 'Song A' })

    const selects = container.querySelectorAll('select')
    fireEvent.change(selects[0], { target: { value: 'Song A' } })
    fireEvent.change(container.querySelector('textarea')!, { target: { value: 'dark techno' } })
    fireEvent.click(screen.getByRole('button', { name: /run style transfer/i }))
    await waitFor(() => expect(requests.some((r) => r.method === 'POST' && /^\/ai\/style-transfer/.test(r.path))).toBe(true))
    await flush()

    act(() => {
      useAppStore.getState().upsertJob({
        job_id: 'gen-1', status: 'COMPLETED', type: 'style-transfer', progress: 100, message: 'Done',
        created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
        result: { stream_url: '/outputs/gen_gen-1/style_transfer_1.wav' },
      })
    })
    await waitFor(() => expect(container.querySelector('audio')).not.toBeNull())
    const base = import.meta.env.VITE_API_BASE || '/api'
    expect(container.querySelector('audio')!.getAttribute('src')).toBe(`${base}/outputs/gen_gen-1/style_transfer_1.wav`)
  })
})

// ---------------------------------------------------------------------------
// MixDeck: Crossfade bars -> transition_bars (8..64); no target_bpm
// ---------------------------------------------------------------------------

function crossfadeSlider(): HTMLInputElement {
  const label = screen.getByText(/crossfade bars/i)
  let node: HTMLElement | null = label
  for (let i = 0; i < 4 && node; i++) {
    const found = node.querySelector<HTMLInputElement>('input[type="range"]')
    if (found) return found
    node = node.parentElement
  }
  throw new Error('no range input near the "Crossfade bars" label')
}

async function renderMixDeckAndSubmit(sliderValue: string): Promise<Record<string, unknown>> {
  installFetch()
  mount(<MixDeck />, `/mix-deck?song_a=${encodeURIComponent('Song A')}&song_b=${encodeURIComponent('Song B')}`)
  await screen.findByText(/crossfade bars/i)
  const slider = crossfadeSlider()
  fireEvent.change(slider, { target: { value: sliderValue } })
  fireEvent.click(screen.getByRole('button', { name: /full remix/i }))
  await waitFor(() => expect(requests.some((r) => r.method === 'POST' && /^\/dj-remix(\?|$)/.test(r.path))).toBe(true))
  const post = requests.find((r) => r.method === 'POST' && /^\/dj-remix(\?|$)/.test(r.path))!
  return post.body as Record<string, unknown>
}

describe('MixDeck', () => {
  it('renders no Target BPM control once two songs are loaded', async () => {
    installFetch()
    mount(<MixDeck />, `/mix-deck?song_a=${encodeURIComponent('Song A')}&song_b=${encodeURIComponent('Song B')}`)
    await screen.findByText(/crossfade bars/i)
    expect(screen.queryByText(/target bpm/i)).toBeNull()
    expect(screen.queryByPlaceholderText(/auto/i)).toBeNull()
  })

  it('slider at 128 sends transition_bars clamped to 64 and no target_bpm key', async () => {
    const body = await renderMixDeckAndSubmit('128')
    expect(body.transition_bars, JSON.stringify(body)).toBe(64)
    expect(body, 'dead target_bpm key still sent').not.toHaveProperty('target_bpm')
  })

  it('slider at 48 sends transition_bars 48 (the slider, not the 8/16/32 segment, is the source)', async () => {
    const body = await renderMixDeckAndSubmit('48')
    expect(body.transition_bars, JSON.stringify(body)).toBe(48)
  })

  it('slider below 8 sends transition_bars 8', async () => {
    const body = await renderMixDeckAndSubmit('0')
    expect(body.transition_bars, JSON.stringify(body)).toBe(8)
  })
})
