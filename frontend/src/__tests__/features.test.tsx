/**
 * B5.2 feature aware UI (plans/audit-the-project-and-sorted-flask.md).
 *
 * Contract these tests pin (the builder implements against it):
 *
 *  - A `useFeatures()` hook backed by react-query fetches `GET /features`
 *    through the real `@/lib/api` client with `staleTime: Infinity`. Every
 *    component below consults it itself (LeftRail, useKeyboardShortcuts,
 *    MissionControl, LibraryAtlas, Operations), so each renders standalone
 *    under a QueryClientProvider + MemoryRouter and hides its own controls.
 *  - `generative: false` -> LeftRail has no "AI Lab" button, neither the `g a`
 *    chord nor Cmd/Ctrl+6 navigates, MissionControl has no "AI Lab" quick
 *    action (the /ai-lab route itself is covered in shell/__tests__/AppShell.test.tsx).
 *  - `stem_split: false` -> LibraryAtlas renders no per row "Split stems"
 *    button and no bulk "Split Stems" button, and the song drawer shows one
 *    note matching /stem split\w*\b.*\b(off|disabled|unavailable|not available)/i.
 *  - `operator: false` -> Operations hides the prune, evict, CLAP, LUFS and
 *    re-analyze buttons; the processing status panel stays.
 *
 * Seams: `fetch` is stubbed globally and routed by path so the real API client
 * runs end to end; `@tanstack/react-virtual` is stubbed to lay out every row
 * because jsdom has no layout (a real virtualizer renders zero rows here).
 * "Features loaded" is detected by scanning the QueryClient cache for a payload
 * carrying a `generative` key, so the hook's query key is not dictated.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'

// jsdom has no layout: a real virtualizer sees a 0px scroll element and renders
// no rows, so the Split button could never appear even when enabled.
vi.mock('@tanstack/react-virtual', () => ({
  useVirtualizer: ({ count }: { count: number }) => ({
    getVirtualItems: () =>
      Array.from({ length: count }, (_, i) => ({ index: i, key: i, start: i * 48, end: (i + 1) * 48, size: 48 })),
    getTotalSize: () => count * 48,
    measureElement: () => {},
  }),
}))

import { LeftRail } from '@/shell/LeftRail'
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'
import MissionControl from '@/pages/MissionControl'
import LibraryAtlas from '@/pages/LibraryAtlas'
import Operations from '@/pages/Operations'
import type { SongInfo } from '@/types'

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

interface Features {
  auth: boolean
  stem_split: boolean
  generative: boolean
  license_enforce: boolean
  spectral_learning: boolean
  operator: boolean
}

const ALL_ON: Features = {
  auth: false, stem_split: true, generative: true, license_enforce: false, spectral_learning: true, operator: true,
}
const ALL_OFF: Features = { ...ALL_ON, stem_split: false, generative: false, operator: false }

const SONGS: SongInfo[] = [
  { name: 'Song With Stems', path: '/lib/a', has_stems: true, has_analysis: true, stems: ['vocals', 'drums', 'bass', 'other'], bpm: 120 },
  { name: 'Song Without Stems', path: '/lib/b', has_stems: false, has_analysis: true, bpm: 124 },
]

const PROCESSING = { fully_processed: ['Song With Stems'], stems_only: [], analysis_only: ['Song Without Stems'], unprocessed: [], total: 2, generated_at: 0 }
const STORAGE = {
  library_dir: '/lib', outputs_dir: '/out', total_songs: 2, total_size_gb: 1.2, cap_gb: 100, within_cap: true,
  songs_with_full_wav: 1, songs_stems_only: 1, prune_on_download: false, keep_raw_after_separation: true, auto_evict_on_download: false,
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

let requests: string[] = []

function installFetch(features: Features) {
  requests = []
  vi.stubGlobal('fetch', async (input: string | URL, _init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString()
    const path = url.replace(/^https?:\/\/[^/]+/, '').replace(/^\/api/, '')
    requests.push(path)
    if (/^\/features(\?|$)/.test(path)) return jsonResponse(features)
    if (/^\/library\?/.test(path)) return jsonResponse({ songs: SONGS, stats: {} })
    if (/^\/library\/processing-status/.test(path)) return jsonResponse(PROCESSING)
    if (/^\/library\/storage(\?|$)/.test(path)) return jsonResponse(STORAGE)
    if (/^\/favorites(\?|$)/.test(path)) return jsonResponse({ songs: [], count: 0 })
    if (/^\/health\//.test(path)) return jsonResponse({ status: 'ok' })
    if (/^\/jobs(\?|$)/.test(path)) return jsonResponse([])
    if (/^\/crates(\?|$)/.test(path)) return jsonResponse([])
    return jsonResponse([])
  })
}

function featuresLoaded(qc: QueryClient): boolean {
  return qc.getQueryCache().getAll().some((q) => {
    const d = q.state.data as Record<string, unknown> | undefined
    return !!d && typeof d === 'object' && 'generative' in d && 'stem_split' in d
  })
}

async function waitForFeatures(qc: QueryClient) {
  await waitFor(() =>
    expect(
      featuresLoaded(qc),
      'no query in the cache holds a /features payload: nothing under test called useFeatures()',
    ).toBe(true),
  )
  // let the components re-render on the resolved data
  await act(async () => { await Promise.resolve() })
}

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
}

function mount(ui: ReactElement, qc = newClient(), route = '/mission-control') {
  const view = render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  )
  return { ...view, qc }
}

beforeEach(() => {
  vi.stubGlobal('confirm', () => true)
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

// ---------------------------------------------------------------------------
// LeftRail + useFeatures caching
// ---------------------------------------------------------------------------

describe('LeftRail', () => {
  it('hides the AI Lab link when generative is off (other links stay)', async () => {
    installFetch(ALL_OFF)
    const { qc } = mount(<LeftRail />)
    await waitForFeatures(qc)
    expect(screen.getByRole('button', { name: 'Library Atlas' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'AI Lab' })).not.toBeInTheDocument()
  })

  it('shows the AI Lab link when generative is on', async () => {
    installFetch(ALL_ON)
    const { qc } = mount(<LeftRail />)
    await waitForFeatures(qc)
    expect(screen.getByRole('button', { name: 'AI Lab' })).toBeInTheDocument()
  })

  it('fetches /features once per QueryClient (staleTime Infinity: a remount does not refetch)', async () => {
    installFetch(ALL_ON)
    const qc = newClient()
    const first = mount(<LeftRail />, qc)
    await waitForFeatures(qc)
    first.unmount()
    mount(<LeftRail />, qc)
    await screen.findByRole('button', { name: 'AI Lab' })
    await act(async () => { await new Promise((r) => setTimeout(r, 20)) })
    const featureCalls = requests.filter((p) => /^\/features(\?|$)/.test(p))
    expect(featureCalls, `requests seen: ${JSON.stringify(requests)}`).toHaveLength(1)
  })
})

// ---------------------------------------------------------------------------
// keyboard shortcut `g a`
// ---------------------------------------------------------------------------

function ShortcutHarness() {
  useKeyboardShortcuts(() => {}, () => {})
  const loc = useLocation()
  return <div data-testid="path">{loc.pathname}</div>
}

function pressChord(...keys: string[]) {
  for (const key of keys) fireEvent.keyDown(window, { key })
}

describe('useKeyboardShortcuts', () => {
  it('ignores `g a` when generative is off, while `g l` still navigates', async () => {
    installFetch(ALL_OFF)
    const { qc } = mount(<ShortcutHarness />)
    await waitForFeatures(qc)
    pressChord('g', 'a')
    expect(screen.getByTestId('path').textContent).toBe('/mission-control')
    pressChord('g', 'l')
    expect(screen.getByTestId('path').textContent).toBe('/library-atlas')
  })

  it('navigates to /ai-lab on `g a` when generative is on', async () => {
    installFetch(ALL_ON)
    const { qc } = mount(<ShortcutHarness />)
    await waitForFeatures(qc)
    pressChord('g', 'a')
    expect(screen.getByTestId('path').textContent).toBe('/ai-lab')
  })

  it('ignores Cmd/Ctrl+6 when generative is off, while Cmd/Ctrl+2 still navigates', async () => {
    installFetch(ALL_OFF)
    const { qc } = mount(<ShortcutHarness />)
    await waitForFeatures(qc)
    fireEvent.keyDown(window, { key: '6', metaKey: true })
    expect(screen.getByTestId('path').textContent).toBe('/mission-control')
    fireEvent.keyDown(window, { key: '6', ctrlKey: true })
    expect(screen.getByTestId('path').textContent).toBe('/mission-control')
    fireEvent.keyDown(window, { key: '2', metaKey: true })
    expect(screen.getByTestId('path').textContent).toBe('/library-atlas')
  })

  it('navigates to /ai-lab on Cmd/Ctrl+6 when generative is on', async () => {
    installFetch(ALL_ON)
    const { qc } = mount(<ShortcutHarness />)
    await waitForFeatures(qc)
    fireEvent.keyDown(window, { key: '6', metaKey: true })
    expect(screen.getByTestId('path').textContent).toBe('/ai-lab')
  })
})

// ---------------------------------------------------------------------------
// MissionControl quick action
// ---------------------------------------------------------------------------

describe('MissionControl', () => {
  it('hides the AI Lab quick action when generative is off (others stay)', async () => {
    installFetch(ALL_OFF)
    const { qc } = mount(<MissionControl />)
    await waitForFeatures(qc)
    expect(screen.getByRole('button', { name: /Mix two tracks/ })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /AI Lab/ })).not.toBeInTheDocument()
  })

  it('shows the AI Lab quick action when generative is on', async () => {
    installFetch(ALL_ON)
    const { qc } = mount(<MissionControl />)
    await waitForFeatures(qc)
    expect(screen.getByRole('button', { name: /AI Lab/ })).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// LibraryAtlas Split buttons + drawer note
// ---------------------------------------------------------------------------

const NOTE_RE = /stem split\w*\b.*\b(off|disabled|unavailable|not available)/i

describe('LibraryAtlas', () => {
  it('renders no Split button (row or bulk) and a one line drawer note when stem_split is off', async () => {
    installFetch(ALL_OFF)
    const { qc } = mount(<LibraryAtlas />, undefined, '/library-atlas')
    await waitForFeatures(qc)
    await screen.findByText('Song Without Stems')

    // per row: the Analyze action stays, the Split action is gone
    expect(screen.getAllByTitle(/Already analyzed|Analyze track/).length).toBeGreaterThan(0)
    expect(screen.queryByTitle('Split stems')).not.toBeInTheDocument()
    expect(screen.queryByTitle('Stems already split')).not.toBeInTheDocument()

    // bulk bar: select one row, Analyze stays, Split Stems is gone
    const rowCheckboxes = screen.getAllByRole('checkbox').slice(1)
    fireEvent.click(rowCheckboxes[0])
    expect(screen.getByRole('button', { name: /^Analyze$/ })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Split Stems/ })).not.toBeInTheDocument()

    // drawer: one note explains why
    fireEvent.click(screen.getByText('Song Without Stems'))
    expect(screen.getAllByText(NOTE_RE)).toHaveLength(1)
  })

  it('renders the Split buttons and no note when stem_split is on', async () => {
    installFetch(ALL_ON)
    const { qc } = mount(<LibraryAtlas />, undefined, '/library-atlas')
    await waitForFeatures(qc)
    await screen.findByText('Song Without Stems')

    expect(screen.getByTitle('Split stems')).toBeInTheDocument()
    const rowCheckboxes = screen.getAllByRole('checkbox').slice(1)
    fireEvent.click(rowCheckboxes[0])
    expect(screen.getByRole('button', { name: /Split Stems/ })).toBeInTheDocument()

    fireEvent.click(screen.getByText('Song Without Stems'))
    expect(screen.queryByText(NOTE_RE)).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Operations operator panels
// ---------------------------------------------------------------------------

const OPERATOR_BUTTONS = [
  /Prune source WAVs/i,
  /Preview eviction/i,
  /Build CLAP search index/i,
  /Calibrate stem loudness/i,
  /Re-analyze entire library/i,
]

describe('Operations', () => {
  it('hides prune, evict, CLAP, LUFS and re-analyze when operator is false (status panel stays)', async () => {
    installFetch(ALL_OFF)
    const { qc } = mount(<Operations />, undefined, '/operations')
    await waitForFeatures(qc)
    await screen.findByText('Processing Status')
    for (const name of OPERATOR_BUTTONS) {
      expect(screen.queryByRole('button', { name }), `button ${name} should be hidden`).not.toBeInTheDocument()
    }
  })

  it('shows every operator button when operator is true', async () => {
    installFetch(ALL_ON)
    const { qc } = mount(<Operations />, undefined, '/operations')
    await waitForFeatures(qc)
    for (const name of OPERATOR_BUTTONS) {
      expect(await screen.findByRole('button', { name })).toBeInTheDocument()
    }
  })
})
