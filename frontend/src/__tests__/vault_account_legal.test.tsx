/**
 * B5.4 Mix Vault, B5.5 Account page, B5.6 Legal surface
 * (plans/audit-the-project-and-sorted-flask.md, family "vault-account-legal").
 *
 * Contract these tests pin (the builder implements against it):
 *
 * B5.4 MixVault
 *  - Each credit row renders `artist`, `source`, and `license` where the
 *    licence text is an <a> whose href is `license_url`.
 *  - A "Credits (txt)" link (text or title) in the same card as Download,
 *    href `${BASE}/outputs/<dir>/CREDITS.txt` where <dir> is the directory
 *    segment of `result.stream_url` (`/outputs/dj_<sid>/mix.wav` -> `dj_<sid>`,
 *    see scripts/api/task_modules/remix.py:441,489). The fixture keeps
 *    `session_id` and the directory name different so a link built from the
 *    bare session id fails here as it would 404 live.
 *  - A job whose `result.expired` is true is not listed and not counted.
 *  - `result.output_url` is never read: a result carrying only output_url
 *    yields no link to it and no WaveSurfer instance loading it.
 *
 * B5.5 Account
 *  - LeftRail footer has an element with text "Account" that navigates to
 *    `/account`; the route renders a page with a heading matching /account/i.
 *  - The page shows the signed in email, or "local" when auth is off.
 *  - "Export" fetches `GET /account/export` through the API client (so the
 *    Bearer token rides along; asserted) and hands the bytes to a file via
 *    `URL.createObjectURL(Blob)`. A bare <a href download> cannot carry the
 *    token and is not accepted.
 *  - "Delete" is a button, disabled until the page's single textbox holds
 *    exactly `DELETE`; a click then sends `DELETE /account` and calls
 *    `signOut()` from the auth context. `delete` (lowercase) keeps it gated.
 *  - When auth is not configured, Delete stays disabled with DELETE typed,
 *    sends nothing, and a note matching NOTE_RE explains why.
 *
 * B5.6 Legal
 *  - `/legal/dmca` renders a <form> whose controls carry `name` attributes
 *    equal to the DmcaRequest fields (reporter_name, reporter_email,
 *    copyright_work, infringing_url, description); submitting posts JSON with
 *    those keys to `POST /legal/dmca` and the response `report_id` appears.
 *  - LoginPage renders four links with hrefs ending in /legal/terms,
 *    /legal/privacy, /legal/dmca and /legal/acceptable-use.
 *
 * Seams: `fetch` is stubbed and routed by path so the real `@/lib/api` runs;
 * `@/lib/supabase` is mocked with a switchable `authConfigured` (getter on a
 * hoisted state object) so one file covers local mode and signed in mode;
 * `wavesurfer.js` is stubbed and records `create` calls; the shell's SSE,
 * poller and toast hooks are stubbed like smoke.test.ts.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement, ReactNode } from 'react'
import type { Session } from '@supabase/supabase-js'

const { authState, wsCreateCalls } = vi.hoisted(() => ({
  authState: { configured: false, token: null as string | null },
  wsCreateCalls: [] as Array<Record<string, unknown>>,
}))

vi.mock('@/lib/supabase', () => ({
  get authConfigured() {
    return authState.configured
  },
  supabase: null,
  getAccessToken: async () => authState.token,
  syncSseCookie: vi.fn(),
}))

vi.mock('wavesurfer.js', () => ({
  default: {
    create: (opts: Record<string, unknown>) => {
      wsCreateCalls.push(opts)
      return {
        destroy: () => undefined,
        getDuration: () => 0,
        getCurrentTime: () => 0,
        // wavesurfer's load() returns a Promise; WaveformDeck attaches .catch to it.
        load: () => Promise.resolve(),
        on: () => undefined,
        playPause: () => undefined,
        setVolume: () => undefined,
      }
    },
  },
}))

vi.mock('@/hooks/useSSE', () => ({ useSSE: () => undefined }))
vi.mock('@/hooks/useJobPoller', () => ({ useJobPoller: () => undefined }))
vi.mock('@/hooks/useJobToasts', () => ({ useJobToasts: () => undefined }))

import * as supabaseModule from '@/lib/supabase'
import App from '@/App'
import MixVault from '@/pages/MixVault'
import { LeftRail } from '@/shell/LeftRail'
import { LoginPage } from '@/shell/LoginPage'
import { AuthContext, type AuthState } from '@/shell/authContext'
import { useAppStore } from '@/stores/appStore'
import type { Job } from '@/types'

// ---------------------------------------------------------------------------
// fetch stub
// ---------------------------------------------------------------------------

type Call = { path: string; method: string; body: unknown; headers: Record<string, string> }
let calls: Call[] = []

const FEATURES = {
  auth: false, stem_split: true, generative: true, license_enforce: false, spectral_learning: true, operator: true,
}

const REPORT_ID = 'dmca-7f3a9c'

function makeResponse(status: number, body: unknown, kind: 'json' | 'text' | 'blob' = 'json'): Response {
  const text = kind === 'text' ? String(body) : JSON.stringify(body)
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: String(status),
    headers: new Headers({
      'content-type':
        kind === 'json' ? 'application/json' : kind === 'text' ? 'text/plain' : 'application/zip',
    }),
    json: async () => (kind === 'json' ? body : JSON.parse(text)),
    text: async () => text,
    blob: async () => new Blob([kind === 'blob' ? (body as BlobPart) : text]),
    arrayBuffer: async () => new Uint8Array(kind === 'blob' ? (body as Uint8Array) : []).buffer,
  } as unknown as Response
}

function installFetch() {
  calls = []
  vi.stubGlobal('fetch', async (input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    const path = url.replace(/^https?:\/\/[^/]+/, '').replace(/^\/api/, '')
    const method = (init?.method ?? 'GET').toUpperCase()
    let body: unknown = undefined
    if (typeof init?.body === 'string') {
      try { body = JSON.parse(init.body) } catch { body = init.body }
    }
    const headers: Record<string, string> = {}
    const h = init?.headers
    if (h instanceof Headers) h.forEach((v, k) => { headers[k] = v })
    else if (h) Object.assign(headers, h as Record<string, string>)
    calls.push({ path, method, body, headers })

    if (/^\/features(\?|$)/.test(path)) return makeResponse(200, FEATURES)
    if (/^\/health\//.test(path)) return makeResponse(200, { status: 'ok' })
    if (/^\/jobs(\?|$)/.test(path)) return makeResponse(200, [])
    if (/^\/crates(\?|$)/.test(path)) return makeResponse(200, { crates: [] })
    if (/^\/favorites(\?|$)/.test(path)) return makeResponse(200, { songs: [], count: 0 })
    if (/^\/library/.test(path)) return makeResponse(200, { songs: [], stats: {} })
    if (/^\/account\/export(\?|$)/.test(path) && method === 'GET') {
      return makeResponse(200, new Uint8Array([0x50, 0x4b, 0x03, 0x04]), 'blob')
    }
    if (/^\/account\/?(\?|$)/.test(path) && method === 'DELETE') {
      return makeResponse(200, { user_id: 'u1', clean: true, status: 'complete', errors: [] })
    }
    if (/^\/legal\/dmca(\?|$)/.test(path) && method === 'POST') {
      return makeResponse(200, { report_id: REPORT_ID, email_sent: false })
    }
    if (/^\/legal\//.test(path) && method === 'GET') {
      return makeResponse(200, `AI RemixMate — legal text for ${path}`, 'text')
    }
    return makeResponse(200, {})
  })
}

const byMethod = (m: string) => calls.filter((c) => c.method === m)

// ---------------------------------------------------------------------------
// harness
// ---------------------------------------------------------------------------

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
}

function authValue(partial: Partial<AuthState> = {}): AuthState {
  return {
    session: null,
    user: null,
    loading: false,
    signIn: async () => ({ error: null }),
    signUp: async () => ({ error: null, needsConfirm: false }),
    signOut: async () => {},
    ...partial,
  }
}

const SESSION = {
  access_token: 'tok-abc',
  user: { id: 'u1', email: 'ada@example.com' },
} as unknown as Session

function mount(ui: ReactElement, route: string, auth?: AuthState) {
  const qc = newClient()
  const tree: ReactNode = auth ? <AuthContext.Provider value={auth}>{ui}</AuthContext.Provider> : ui
  const view = render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[route]}>{tree}</MemoryRouter>
    </QueryClientProvider>,
  )
  return { ...view, qc }
}

function LocationProbe() {
  const loc = useLocation()
  return <div data-testid="path">{loc.pathname}</div>
}

const NOW = new Date().toISOString()

function completedJob(partial: Partial<Job> & { job_id: string }): Job {
  return {
    status: 'COMPLETED', type: 'dj_remix', progress: 100, message: '',
    created_at: NOW, updated_at: NOW, ...partial,
  }
}

const CREDIT = {
  song: 'Song A',
  artist: 'Artist A',
  license: 'CC BY 4.0',
  license_url: 'https://creativecommons.org/licenses/by/4.0/',
  source: 'ccMixter',
  source_url: 'https://ccmixter.org/files/artist-a/1',
  attribution_required: true,
}

const LIVE_MIX = completedJob({
  job_id: 'j-live',
  meta: { song_a: 'Song A', song_b: 'Song B' },
  result: {
    session_id: 'sessa',
    stream_url: '/outputs/dj_sessa/mix.wav',
    credits: [CREDIT, { ...CREDIT, song: 'Song B', artist: 'Artist B', source: 'Free Music Archive' }],
  },
})

const EXPIRED_MIX = completedJob({
  job_id: 'j-expired',
  meta: { song_a: 'Gone A', song_b: 'Gone B' },
  result: { session_id: 'sessx', stream_url: '/outputs/dj_sessx/mix.wav', expired: true, credits: [CREDIT] },
})

const LEGACY_MIX = completedJob({
  job_id: 'j-legacy',
  meta: { song_a: 'Legacy A', song_b: 'Legacy B' },
  result: { session_id: 'legacy', output_url: '/outputs/legacy/mix.wav' },
})

function seedJobs(jobs: Job[]) {
  useAppStore.setState({
    jobs: Object.fromEntries(jobs.map((j) => [j.job_id, j])),
    sseConnected: false,
    sessionStartedAt: 0,
  })
}

beforeEach(() => {
  authState.configured = false
  authState.token = null
  wsCreateCalls.length = 0
  installFetch()
  seedJobs([])
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

// ---------------------------------------------------------------------------
// B5.4 Mix Vault
// ---------------------------------------------------------------------------

describe('B5.4 MixVault credits and links', () => {
  function openDetails() {
    const toggle = screen.queryByTitle(/expand/i)
    if (toggle) fireEvent.click(toggle)
  }

  it('renders artist, licence as a link to license_url, and source for each credit', async () => {
    seedJobs([LIVE_MIX])
    mount(<MixVault />, '/mix-vault')
    await screen.findByText('Song A → Song B')
    openDetails()

    expect(screen.getByText('Artist A')).toBeInTheDocument()
    expect(screen.getByText('Artist B')).toBeInTheDocument()
    expect(screen.getByText('ccMixter')).toBeInTheDocument()
    expect(screen.getByText('Free Music Archive')).toBeInTheDocument()

    const licenceLinks = screen.getAllByRole('link', { name: /CC BY 4\.0/i })
    expect(licenceLinks, 'licence text is not rendered as a link').toHaveLength(2)
    for (const a of licenceLinks) expect(a).toHaveAttribute('href', CREDIT.license_url)
  })

  it('links "Credits (txt)" to ${BASE}/outputs/<dir>/CREDITS.txt in the same card as Download', async () => {
    seedJobs([LIVE_MIX])
    mount(<MixVault />, '/mix-vault')
    await screen.findByText('Song A → Song B')

    const download = screen.getByTitle('Download mix')
    expect(download).toHaveAttribute('href', '/api/outputs/dj_sessa/mix.wav')

    const credits = screen.getByRole('link', { name: /credits \(txt\)/i })
    expect(credits).toHaveAttribute('href', '/api/outputs/dj_sessa/CREDITS.txt')

    const card = download.closest('.mv-card')
    expect(card, 'Download link is not inside a card').not.toBeNull()
    expect(card!.contains(credits), 'Credits (txt) is not in the same card as Download').toBe(true)
  })

  it('hides and does not count rows whose result.expired is true', async () => {
    seedJobs([LIVE_MIX, EXPIRED_MIX])
    mount(<MixVault />, '/mix-vault')
    await screen.findByText('Song A → Song B')

    expect(screen.queryByText('Gone A → Gone B')).not.toBeInTheDocument()
    expect(screen.getByText(/^1 completed mix$/)).toBeInTheDocument()
  })

  it('never reads result.output_url: a legacy result yields no link and no waveform for it', async () => {
    seedJobs([LEGACY_MIX])
    mount(<MixVault />, '/mix-vault')
    await screen.findByText('Legacy A → Legacy B')

    expect(document.querySelector('a[href*="/outputs/legacy/"]')).toBeNull()
    const legacyLoads = wsCreateCalls.filter((o) => String(o.url ?? '').includes('/outputs/legacy/'))
    expect(legacyLoads, 'WaveSurfer was created with the output_url').toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// B5.5 Account page
// ---------------------------------------------------------------------------

const NOTE_RE = /(local mode|without auth|no auth|auth\w*\s+(is\s+)?(off|not configured|disabled|unavailable))/i

describe('B5.5 Account page', () => {
  it('rail footer has an Account link that navigates to /account', async () => {
    mount(<><LeftRail /><LocationProbe /></>, '/mission-control')
    const link = await screen.findByText(/^account$/i)
    fireEvent.click(link)
    expect(screen.getByTestId('path').textContent).toBe('/account')
  })

  it('local mode: shows "local", Delete stays disabled with a note and sends nothing', async () => {
    expect(supabaseModule.authConfigured, 'supabase mock did not switch to local mode').toBe(false)
    mount(<App />, '/account')
    await screen.findByRole('heading', { name: /account/i })

    expect(screen.getByText(/\blocal\b/)).toBeInTheDocument()
    expect(screen.getByText(NOTE_RE)).toBeInTheDocument()

    const del = screen.getByRole('button', { name: /delete/i })
    expect(del).toBeDisabled()

    const box = screen.queryByRole('textbox')
    if (box) {
      fireEvent.change(box, { target: { value: 'DELETE' } })
      expect(del).toBeDisabled()
    }
    fireEvent.click(del)
    await act(async () => { await new Promise((r) => setTimeout(r, 20)) })
    expect(byMethod('DELETE')).toHaveLength(0)
  })

  it('signed in: shows the email; Export fetches GET /account/export with the token and creates a file', async () => {
    authState.configured = true
    authState.token = 'tok-abc'
    expect(supabaseModule.authConfigured, 'supabase mock did not switch to auth mode').toBe(true)
    const createObjectURL = vi.fn((_blob: Blob) => 'blob:mock-export')
    const revokeObjectURL = vi.fn()
    // Subclass so the real global URL is left untouched once unstubbed.
    vi.stubGlobal('URL', Object.assign(class extends URL {}, { createObjectURL, revokeObjectURL }))

    mount(<App />, '/account', authValue({ session: SESSION, user: SESSION.user }))
    await screen.findByRole('heading', { name: /account/i })
    expect(screen.getByText(/ada@example\.com/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /export/i }))

    await waitFor(() => {
      const exportCalls = calls.filter((c) => /^\/account\/export/.test(c.path) && c.method === 'GET')
      expect(exportCalls, `requests seen: ${JSON.stringify(calls.map((c) => `${c.method} ${c.path}`))}`).toHaveLength(1)
      expect(exportCalls[0].headers.Authorization ?? exportCalls[0].headers.authorization).toBe('Bearer tok-abc')
    })
    await waitFor(() => expect(createObjectURL).toHaveBeenCalledTimes(1))
    expect(createObjectURL.mock.calls[0][0]).toBeInstanceOf(Blob)
  })

  it('signed in: Delete is gated by typing DELETE, then calls DELETE /account and signs out', async () => {
    authState.configured = true
    authState.token = 'tok-abc'
    const signOut = vi.fn(async () => {})

    mount(<App />, '/account', authValue({ session: SESSION, user: SESSION.user, signOut }))
    await screen.findByRole('heading', { name: /account/i })

    const del = screen.getByRole('button', { name: /delete/i })
    expect(del, 'Delete must start disabled').toBeDisabled()

    const box = screen.getByRole('textbox')
    fireEvent.change(box, { target: { value: 'delete' } })
    expect(del, 'lowercase "delete" must not unlock Delete').toBeDisabled()
    fireEvent.click(del)
    await act(async () => { await new Promise((r) => setTimeout(r, 20)) })
    expect(byMethod('DELETE')).toHaveLength(0)
    expect(signOut).not.toHaveBeenCalled()

    fireEvent.change(box, { target: { value: 'DELETE' } })
    expect(del).toBeEnabled()
    fireEvent.click(del)

    await waitFor(() => {
      const dels = byMethod('DELETE')
      expect(dels, `requests seen: ${JSON.stringify(calls.map((c) => `${c.method} ${c.path}`))}`).toHaveLength(1)
      expect(dels[0].path).toMatch(/^\/account\/?$/)
    })
    await waitFor(() => expect(signOut).toHaveBeenCalledTimes(1))
  })
})

// ---------------------------------------------------------------------------
// B5.6 Legal surface
// ---------------------------------------------------------------------------

const DMCA_FIELDS = ['reporter_name', 'reporter_email', 'copyright_work', 'infringing_url', 'description'] as const

describe('B5.6 Legal surface', () => {
  it('/legal/dmca renders a form for the DmcaRequest fields, posts to POST /legal/dmca and shows the report id', async () => {
    const { container } = mount(<App />, '/legal/dmca')

    await waitFor(() => {
      expect(container.querySelector('form [name="reporter_name"]'), 'no form control named reporter_name').not.toBeNull()
    })
    const fields: Record<string, Element> = {}
    for (const f of DMCA_FIELDS) {
      const el = container.querySelector(`form [name="${f}"]`)
      expect(el, `no form control named ${f}`).not.toBeNull()
      fields[f] = el!
    }

    const values = {
      reporter_name: 'Ada Lovelace',
      reporter_email: 'ada@example.com',
      copyright_work: 'Song X (album Y)',
      infringing_url: 'https://example.com/mix/1',
      description: 'please remove',
    }
    for (const f of DMCA_FIELDS) fireEvent.change(fields[f], { target: { value: values[f] } })

    const form = fields.reporter_name.closest('form')!
    fireEvent.submit(form)

    await waitFor(() => {
      const posts = calls.filter((c) => c.method === 'POST' && /^\/legal\/dmca(\?|$)/.test(c.path))
      expect(posts, `requests seen: ${JSON.stringify(calls.map((c) => `${c.method} ${c.path}`))}`).toHaveLength(1)
      expect(posts[0].body).toMatchObject(values)
    })
    expect(await screen.findByText(new RegExp(REPORT_ID))).toBeInTheDocument()
  })

  it('LoginPage carries the four legal links', () => {
    mount(<LoginPage />, '/', authValue())
    const hrefs = screen.getAllByRole('link').map((a) => a.getAttribute('href') ?? '')
    for (const slug of ['terms', 'privacy', 'dmca', 'acceptable-use']) {
      expect(hrefs.some((h) => h.endsWith(`/legal/${slug}`)), `no link to /legal/${slug}; links: ${JSON.stringify(hrefs)}`).toBe(true)
    }
  })
})
