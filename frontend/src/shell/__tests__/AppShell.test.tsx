/**
 * B5.9: the /ai-lab route is gated on the `generative` feature flag, not only
 * the rail link and the shortcuts. Plan text: "/ai-lab redirects" when
 * generative is off. The flag counts as off until it is known on
 * (useFeatures docstring), so the page never renders before the answer.
 *
 * The shell's SSE, poller and job toast hooks are stubbed as smoke.test.ts
 * does; `fetch` is routed by path so the real API client and useFeatures run.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('@/hooks/useSSE', () => ({ useSSE: () => undefined }))
vi.mock('@/hooks/useJobPoller', () => ({ useJobPoller: () => undefined }))
vi.mock('@/hooks/useJobToasts', () => ({ useJobToasts: () => undefined }))

import { AppShell } from '@/shell/AppShell'

const ALL_ON = {
  auth: false, stem_split: true, generative: true, license_enforce: false, spectral_learning: true, operator: true,
}
const GENERATIVE_OFF = { ...ALL_ON, generative: false }

const PROCESSING = { fully_processed: [], stems_only: [], analysis_only: [], unprocessed: [], total: 0, generated_at: 0 }
const STORAGE = {
  library_dir: '/lib', outputs_dir: '/out', total_songs: 0, total_size_gb: 0, cap_gb: 100, within_cap: true,
  songs_with_full_wav: 0, songs_stems_only: 0, prune_on_download: false, keep_raw_after_separation: true, auto_evict_on_download: false,
}
const MODELS = { device: 'cpu', mps_allocated_gb: 0, max_vram_gb: 0, models: [] }

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

function installFetch(features: typeof ALL_ON) {
  vi.stubGlobal('fetch', async (input: string | URL) => {
    const url = typeof input === 'string' ? input : input.toString()
    const path = url.replace(/^https?:\/\/[^/]+/, '').replace(/^\/api/, '')
    if (/^\/features(\?|$)/.test(path)) return jsonResponse(features)
    if (/^\/library\?/.test(path)) return jsonResponse({ songs: [], stats: {} })
    if (/^\/library\/processing-status/.test(path)) return jsonResponse(PROCESSING)
    if (/^\/library\/storage(\?|$)/.test(path)) return jsonResponse(STORAGE)
    if (/^\/favorites(\?|$)/.test(path)) return jsonResponse({ songs: [], count: 0 })
    if (/^\/crates(\?|$)/.test(path)) return jsonResponse({ crates: [] })
    if (/^\/jobs(\?|$)/.test(path)) return jsonResponse([])
    if (/^\/health\//.test(path)) return jsonResponse({ status: 'ok' })
    if (/^\/ai\/models(\?|$)/.test(path)) return jsonResponse(MODELS)
    return jsonResponse([])
  })
}

function LocationProbe() {
  const loc = useLocation()
  return <div data-testid="path">{loc.pathname}</div>
}

function mountAt(route: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[route]}>
        <AppShell />
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('AppShell /ai-lab gate', () => {
  it('redirects /ai-lab to /mission-control when generative is off and never renders AI Lab', async () => {
    installFetch(GENERATIVE_OFF)
    mountAt('/ai-lab')

    await waitFor(() => expect(screen.getByTestId('path').textContent).toBe('/mission-control'))
    expect(await screen.findByRole('heading', { name: 'Mission Control' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'AI Lab' })).not.toBeInTheDocument()
  })

  it('serves /ai-lab when generative is on', async () => {
    installFetch(ALL_ON)
    mountAt('/ai-lab')

    expect(await screen.findByRole('heading', { name: 'AI Lab' })).toBeInTheDocument()
    expect(screen.getByTestId('path').textContent).toBe('/ai-lab')
  })
})

describe('LeftRail footer credit', () => {
  it('shows the author line with a link to the public repo', async () => {
    installFetch(ALL_ON)
    mountAt('/mission-control')

    const credit = await screen.findByTestId('left-rail-credit')
    expect(credit.textContent).toBe('AI RemixMate by Aditya Chunduri · GitHub')
    const link = screen.getByRole('link', { name: 'GitHub' })
    expect(link.getAttribute('href')).toBe('https://github.com/Chunduri-Aditya/ai-remixmate')
    expect(link.getAttribute('target')).toBe('_blank')
    expect(link.getAttribute('rel')).toBe('noopener noreferrer')
  })
})
