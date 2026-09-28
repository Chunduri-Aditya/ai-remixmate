/**
 * B5.3 Jobs UX: the inspector talks to the backend, the poller merges.
 *
 * Real `@/lib/api` (unconfigured supabase, so no token) over a stubbed
 * `fetch`, so every assertion is on the request that would leave the browser:
 *   - Cancel sends DELETE /api/jobs/<id> and removes the card only on 2xx.
 *   - useJobPoller merges the polled page by id; an entry the page does not
 *     carry (the 21st) survives the poll.
 *   - Retry replays job.meta.request byte for byte (not a rebuilt subset).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, renderHook, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { useAppStore } from '@/stores/appStore'
import { useJobPoller } from '@/hooks/useJobPoller'
import { RightInspector } from '@/shell/RightInspector'
import type { Job } from '@/types'

type Call = { url: string; method: string; body: unknown }
let calls: Call[]
let respond: (url: string, method: string) => { status: number; json: unknown }

const NOW = new Date().toISOString()

function job(partial: Partial<Job> & { job_id: string }): Job {
  return {
    status: 'RUNNING', type: 'dj_remix', progress: 10, message: 'working',
    created_at: NOW, updated_at: NOW, ...partial,
  }
}

function seedStore(jobs: Job[]) {
  useAppStore.setState({
    jobs: Object.fromEntries(jobs.map((j) => [j.job_id, j])),
    sseConnected: false,
    inspectorTab: 'jobs',
    sessionStartedAt: 0,
  })
}

beforeEach(() => {
  calls = []
  respond = () => ({ status: 200, json: {} })
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    const method = init?.method ?? 'GET'
    calls.push({ url, method, body: init?.body ? JSON.parse(init.body as string) : undefined })
    const r = respond(url, method)
    return {
      ok: r.status >= 200 && r.status < 300,
      status: r.status,
      statusText: String(r.status),
      json: async () => r.json,
      text: async () => JSON.stringify(r.json),
    } as unknown as Response
  })
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

const deletes = () => calls.filter((c) => c.method === 'DELETE')

describe('Inspector Cancel', () => {
  it('sends DELETE /jobs/<id> and drops the card once the backend says 2xx', async () => {
    seedStore([job({ job_id: 'j-cancel-1' })])
    render(<MemoryRouter><RightInspector /></MemoryRouter>)

    await userEvent.click(screen.getByRole('button', { name: 'Cancel job' }))

    await waitFor(() => expect(deletes()).toHaveLength(1))
    expect(deletes()[0].url).toBe('/api/jobs/j-cancel-1')
    await waitFor(() => expect(useAppStore.getState().jobs['j-cancel-1']).toBeUndefined())
  })

  it('keeps the card in the store when the DELETE fails', async () => {
    respond = (url, method) =>
      method === 'DELETE' && url.endsWith('/jobs/j-cancel-2')
        ? { status: 500, json: { detail: 'boom' } }
        : { status: 200, json: {} }
    seedStore([job({ job_id: 'j-cancel-2' })])
    render(<MemoryRouter><RightInspector /></MemoryRouter>)

    await userEvent.click(screen.getByRole('button', { name: 'Cancel job' }))

    await waitFor(() => expect(deletes()).toHaveLength(1))
    // Give any removal a tick to land; the entry must still be there.
    await act(async () => { await new Promise((r) => setTimeout(r, 20)) })
    expect(useAppStore.getState().jobs['j-cancel-2']).toBeDefined()
  })
})

describe('useJobPoller', () => {
  // REST /jobs shape: lowercase status, job_type, epoch seconds, 0..1 progress
  const rawPage = Array.from({ length: 20 }, (_, i) => ({
    job_id: `polled-${String(i).padStart(2, '0')}`,
    status: 'running', job_type: 'dj_remix', progress: 0.5, message: 'm',
    created_at: 1_700_000_000 + i, updated_at: 1_700_000_100 + i,
    result: null, error: null, meta: {},
  }))

  it('a poll of 20 jobs leaves an existing 21st entry in the store', async () => {
    respond = (url, method) =>
      method === 'GET' && url.startsWith('/api/jobs')
        ? { status: 200, json: rawPage }
        : { status: 200, json: {} }
    seedStore([job({ job_id: 'existing-21', status: 'PENDING' })])

    const hook = renderHook(() => useJobPoller(true))

    await waitFor(() => {
      const ids = Object.keys(useAppStore.getState().jobs)
      expect(ids).toHaveLength(21)
    })
    const jobs = useAppStore.getState().jobs
    expect(jobs['existing-21']).toBeDefined()
    expect(jobs['existing-21'].status).toBe('PENDING')
    for (const r of rawPage) expect(jobs[r.job_id]).toBeDefined()
    // Polled entries are normalized, not stored raw.
    expect(jobs['polled-00'].status).toBe('RUNNING')
    expect(jobs['polled-00'].progress).toBe(50)
    hook.unmount()
  })

  it('an entry present in the poll is updated in place by id', async () => {
    respond = (url, method) =>
      method === 'GET' && url.startsWith('/api/jobs')
        ? { status: 200, json: [{ ...rawPage[0], job_id: 'shared', status: 'done', progress: 1.0 }] }
        : { status: 200, json: {} }
    seedStore([job({ job_id: 'shared', status: 'RUNNING', progress: 10 }), job({ job_id: 'other' })])

    const hook = renderHook(() => useJobPoller(true))

    await waitFor(() => expect(useAppStore.getState().jobs['shared'].status).toBe('COMPLETED'))
    expect(useAppStore.getState().jobs['shared'].progress).toBe(100)
    expect(useAppStore.getState().jobs['other']).toBeDefined()
    hook.unmount()
  })
})

describe('Inspector Retry', () => {
  const request = {
    song_a: 'alpha', song_b: 'beta',
    transition_bars: 32, preset: 'techno',
    bridge_beat_mode: 'auto', bridge_beat_genre: 'house', bridge_beat_intensity: 0.6,
    bridge_beat_path: null, transition_effect: 'stepped',
  }

  it('replays job.meta.request verbatim to POST /dj-remix', async () => {
    respond = (url, method) =>
      method === 'POST' && url === '/api/dj-remix'
        ? { status: 202, json: { job_id: 'retried-1' } }
        : { status: 200, json: {} }
    // Backend shape after B5.3: the legacy song keys stay, request is added.
    seedStore([job({
      job_id: 'failed-1', status: 'FAILED', type: 'dj_remix', progress: 0,
      meta: { song_a: 'alpha', song_b: 'beta', request },
    })])
    render(<MemoryRouter><RightInspector /></MemoryRouter>)

    await userEvent.click(screen.getByRole('button', { name: '↺ Retry' }))

    await waitFor(() => expect(calls.filter((c) => c.method === 'POST')).toHaveLength(1))
    const post = calls.find((c) => c.method === 'POST')!
    expect(post.url).toBe('/api/dj-remix')
    expect(post.body).toEqual(request)
    await waitFor(() => expect(useAppStore.getState().jobs['retried-1']).toBeDefined())
  })

  it('replays a dj_chain request including its non default fields', async () => {
    const chainRequest = {
      songs: ['a', 'b', 'c'], transition_bars: 8, preset: 'house',
      bridge_beat_mode: 'none', bridge_beat_genre: 'auto', bridge_beat_intensity: 0.38,
      bridge_beat_path: null, transition_effect: 'echo',
    }
    respond = (url, method) =>
      method === 'POST' && url === '/api/dj-chain'
        ? { status: 202, json: { job_id: 'retried-chain' } }
        : { status: 200, json: {} }
    seedStore([job({
      job_id: 'failed-chain', status: 'FAILED', type: 'dj_chain', progress: 0,
      meta: { songs: ['a', 'b', 'c'], n: 3, request: chainRequest },
    })])
    render(<MemoryRouter><RightInspector /></MemoryRouter>)

    await userEvent.click(screen.getByRole('button', { name: '↺ Retry' }))

    await waitFor(() => expect(calls.filter((c) => c.method === 'POST')).toHaveLength(1))
    const post = calls.find((c) => c.method === 'POST')!
    expect(post.url).toBe('/api/dj-chain')
    expect(post.body).toEqual(chainRequest)
  })
})
