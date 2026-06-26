import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from '../App'

vi.mock('@/hooks/useSSE', () => ({ useSSE: () => undefined }))
vi.mock('@/hooks/useJobPoller', () => ({ useJobPoller: () => undefined }))
vi.mock('@/hooks/useJobToasts', () => ({ useJobToasts: () => undefined }))

vi.mock('wavesurfer.js', () => ({
  default: {
    create: () => ({
      destroy: () => undefined,
      getDuration: () => 0,
      load: () => undefined,
      on: () => undefined,
      playPause: () => undefined,
      setVolume: () => undefined,
    }),
  },
}))

vi.mock('@/lib/api', () => ({
  EVENTS_URL: '/events/stream',
  normalizeJob: (raw: Record<string, unknown>) => raw,
  healthApi: {
    live: async () => ({ status: 'ok' }),
    ready: async () => ({ status: 'ok' }),
  },
  libraryApi: {
    list: async () => [],
    get: async () => ({
      name: 'Mock Track',
      path: '/mock.wav',
      has_stems: false,
      has_analysis: false,
    }),
    delete: async () => undefined,
    stats: async () => ({ total_songs: 0, indexed_songs: 0, stems_split: 0, total_size_mb: 0 }),
    initRun: async () => ({ job_id: 'job-init' }),
  },
  jobsApi: {
    list: async () => [],
    get: async () => ({
      job_id: 'job-1',
      status: 'COMPLETED',
      type: 'download',
      progress: 100,
      message: '',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }),
    cancel: async () => undefined,
  },
  downloadApi: {
    single: async () => ({
      job_id: 'job-download',
      status: 'PENDING',
      type: 'download',
      progress: 0,
      message: '',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }),
    batch: async () => [],
    fromSpotify: async () => ({ job_id: 'job-spotify' }),
    playlist: async () => ({
      job_id: 'job-playlist',
      status: 'PENDING',
      type: 'download',
      progress: 0,
      message: '',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }),
  },
  analysisApi: {
    analyze: async () => ({ job_id: 'job-analysis' }),
    compatibility: async () => ({
      song_a: '',
      song_b: '',
      compatible: true,
      overall: 1,
      bpm_score: 1,
      key_score: 1,
      energy_score: 1,
      bpm_a: 120,
      bpm_b: 120,
    }),
    recommend: async () => [],
    similar: async () => [],
    rebuildIndex: async () => ({ job_id: 'job-index' }),
  },
  remixApi: {
    create: async () => ({ job_id: 'job-remix' }),
    preview: async () => ({ job_id: 'job-preview' }),
    chain: async () => ({ job_id: 'job-chain' }),
  },
  audioApi: {
    streamUrl: (name: string) => `/audio/${encodeURIComponent(name)}`,
  },
  stemsApi: {
    split: async () => ({ job_id: 'job-stems' }),
    splitBatch: async () => ({ job_id: 'job-stems-batch' }),
    compress: async () => ({ job_id: 'job-compress' }),
    compressBatch: async () => ({ job_id: 'job-compress-batch' }),
    stemUrl: (name: string, stem: string) => `/stems/${encodeURIComponent(name)}/${encodeURIComponent(stem)}`,
  },
  cratesApi: {
    list: async () => [],
    create: async () => ({ id: 1, name: 'Mock Crate', song_count: 0, created_at: new Date().toISOString() }),
    rename: async () => ({ id: 1, name: 'Mock Crate', song_count: 0, created_at: new Date().toISOString() }),
    delete: async () => undefined,
    songs: async () => [],
    addSong: async () => undefined,
    removeSong: async () => undefined,
  },
  tagsApi: {
    list: async () => [],
    songTags: async () => [],
    addTag: async () => undefined,
    removeTag: async () => undefined,
  },
  favoritesApi: {
    list: async () => [],
    add: async () => undefined,
    remove: async () => undefined,
  },
  setlistApi: {
    optimize: async () => [],
  },
  aiApi: {
    models: async () => [],
    styleTransfer: async () => ({ job_id: 'job-style' }),
    inpaint: async () => ({ job_id: 'job-inpaint' }),
  },
}))

const ROUTES = [
  { path: '/mission-control', heading: 'Mission Control' },
  { path: '/library-atlas', heading: 'Library Atlas' },
  { path: '/mix-deck', heading: 'Mix Deck' },
  { path: '/set-builder', heading: 'Set Builder' },
  { path: '/signal-search', heading: 'Signal Search' },
  { path: '/ai-lab', heading: 'AI Lab' },
  { path: '/mix-vault', heading: 'Mix Vault' },
  { path: '/operations', heading: 'Downloads' },
]

function renderRoute(path: string) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  })

  const view = render(
    React.createElement(
      MemoryRouter,
      { initialEntries: [path] },
      React.createElement(
        QueryClientProvider,
        { client: queryClient },
        React.createElement(App),
      ),
    ),
  )

  return { ...view, queryClient }
}

afterEach(() => {
  cleanup()
})

describe('route smoke rendering', () => {
  it.each(ROUTES)('renders $path without throwing', async ({ path, heading }) => {
    const { queryClient } = renderRoute(path)

    expect(await screen.findByRole('heading', { name: heading })).toBeInTheDocument()
    queryClient.clear()
  })
})
