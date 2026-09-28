/**
 * Library Atlas crate membership against the real GET /crates wire shape.
 *
 * The backend identifies a crate by `crate_id`, a string, e.g.
 * {"crate_id":"8e8104e9-...","name":"...","created_at":1790588467.05,"song_count":0}.
 * The client once typed it as `id: number`, so CratesSection read `c.id`,
 * rendered every row with key undefined (React key warning) and posted to
 * `/crates/undefined/songs` (404 "Crate not found: undefined") on every
 * add-to-crate. smoke.test.ts mocks '@/lib/api' wholesale and cannot see
 * that; this file runs the real client over a path routed `fetch` stub.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// jsdom has no layout: a real virtualizer sees a 0px scroll element and renders
// no rows, so the song row could never appear.
vi.mock('@tanstack/react-virtual', () => ({
  useVirtualizer: ({ count }: { count: number }) => ({
    getVirtualItems: () =>
      Array.from({ length: count }, (_, i) => ({ index: i, key: i, start: i * 48, end: (i + 1) * 48, size: 48 })),
    getTotalSize: () => count * 48,
    measureElement: () => {},
  }),
}))

import LibraryAtlas from '@/pages/LibraryAtlas'
import type { SongInfo } from '@/types'

const FEATURES = {
  auth: false, stem_split: true, generative: true, license_enforce: false, spectral_learning: true, operator: true,
}

const SONGS: SongInfo[] = [
  { name: 'Song Without Stems', path: '/lib/b', has_stems: false, has_analysis: true, bpm: 124 },
]

// Two rows so a missing key is a duplicate key (undefined), as in the live pass.
const CRATES = [
  { crate_id: '8e8104e9-dffe-4bb5-81bf-a2df5ca7bfe5', name: 'Browser check crate', created_at: 1790588467.0582242, song_count: 0 },
  { crate_id: '5c1f0a2b-1111-4222-8333-444455556666', name: 'Second crate', created_at: 1790588468.0, song_count: 1 },
]

const PROCESSING = { fully_processed: [], stems_only: [], analysis_only: ['Song Without Stems'], unprocessed: [], total: 1, generated_at: 0 }
const STORAGE = {
  library_dir: '/lib', outputs_dir: '/out', total_songs: 1, total_size_gb: 0.2, cap_gb: 100, within_cap: true,
  songs_with_full_wav: 1, songs_stems_only: 0, prune_on_download: false, keep_raw_after_separation: true, auto_evict_on_download: false,
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

function installFetch() {
  requests = []
  vi.stubGlobal('fetch', async (input: string | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString()
    const path = url.replace(/^https?:\/\/[^/]+/, '').replace(/^\/api/, '')
    const method = init?.method ?? 'GET'
    const body = init?.body ? JSON.parse(init.body as string) : undefined
    requests.push({ method, path, body })

    if (/^\/features(\?|$)/.test(path)) return jsonResponse(FEATURES)
    if (/^\/library\?/.test(path)) return jsonResponse({ songs: SONGS, stats: {} })
    if (/^\/library\/processing-status/.test(path)) return jsonResponse(PROCESSING)
    if (/^\/library\/storage(\?|$)/.test(path)) return jsonResponse(STORAGE)
    if (/^\/favorites(\?|$)/.test(path)) return jsonResponse({ songs: [], count: 0 })
    if (/^\/crates(\?|$)/.test(path)) return jsonResponse({ crates: CRATES })
    if (/^\/crates\/[^/]+\/songs$/.test(path)) return jsonResponse(method === 'GET' ? [] : {})
    return jsonResponse([])
  })
}

function mount() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={['/library-atlas']}><LibraryAtlas /></MemoryRouter>
    </QueryClientProvider>,
  )
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('LibraryAtlas crates', () => {
  it('adds a song to a crate by crate_id, the string id GET /crates returns, with no key warning', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    installFetch()
    mount()

    await screen.findByText('Song Without Stems')
    fireEvent.click(screen.getByText('Song Without Stems'))

    const toggle = await screen.findByRole('checkbox', { name: /Browser check crate/ })
    fireEvent.click(toggle)

    await waitFor(() => expect(requests.some((r) => r.method === 'POST')).toBe(true))
    const post = requests.find((r) => r.method === 'POST')!
    expect(post.path).toBe(`/crates/${CRATES[0].crate_id}/songs`)
    expect(post.body).toEqual({ song_name: 'Song Without Stems' })

    // The regression: every membership GET and the add POST went to /crates/undefined/...
    expect(requests.filter((r) => r.path.includes('/crates/undefined')).map((r) => `${r.method} ${r.path}`)).toEqual([])

    const keyWarnings = errorSpy.mock.calls.filter((args) => String(args[0]).includes('unique "key" prop'))
    expect(keyWarnings, 'React warned about a missing list key in CratesSection').toEqual([])
  })
})
