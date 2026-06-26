import { renderHook } from '@testing-library/react'
import { useJobTimer } from '../useJobTimer'
import type { Job } from '@/types'

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    job_id: 'test-1',
    status: 'RUNNING',
    type: 'dj_remix',
    progress: 50,
    message: '',
    meta: {},
    created_at: new Date(Date.now() - 30_000).toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  }
}

describe('useJobTimer', () => {
  it('returns elapsed seconds >= 29 for a job started 30s ago', () => {
    const { result } = renderHook(() => useJobTimer(makeJob()))
    expect(result.current.elapsed).toBeGreaterThanOrEqual(29)
  })

  it('returns null eta when progress is 0', () => {
    const { result } = renderHook(() => useJobTimer(makeJob({ progress: 0 })))
    expect(result.current.eta).toBeNull()
  })

  it('returns positive eta when progress > 2', () => {
    const { result } = renderHook(() => useJobTimer(makeJob({ progress: 50 })))
    expect(result.current.eta).not.toBeNull()
    expect(result.current.eta!).toBeGreaterThan(0)
  })
})
