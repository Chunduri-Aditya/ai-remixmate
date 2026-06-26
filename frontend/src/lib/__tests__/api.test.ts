/**
 * Tests for the api.ts request wrapper:
 *   - happy path: 200 resolves to parsed JSON
 *   - non-2xx: rejects with ApiError carrying the correct status
 *   - timeout: fetch that never resolves rejects with "timed out" ApiError
 */

import { describe, it, expect, vi, afterEach } from 'vitest'
import { ApiError, healthApi } from '../api'

// ── helpers ────────────────────────────────────────────────────────────────

function mockFetch(impl: (url: string, opts: RequestInit) => Promise<Response>) {
  vi.stubGlobal('fetch', impl)
}

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  } as unknown as Response
}

// ── setup / teardown ────────────────────────────────────────────────────────

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

// ── tests ───────────────────────────────────────────────────────────────────

describe('api request wrapper', () => {
  describe('happy path', () => {
    it('resolves with parsed JSON on 200', async () => {
      const data = { status: 'ok', version: '1.0' }
      mockFetch(() => Promise.resolve(jsonResponse(data)))

      const result = await healthApi.live()
      expect(result).toEqual(data)
    })
  })

  describe('error handling', () => {
    it('throws ApiError on 404', async () => {
      mockFetch(() =>
        Promise.resolve({
          ok: false,
          status: 404,
          statusText: 'Not Found',
          text: () => Promise.resolve('Not Found'),
        } as unknown as Response),
      )

      const err = await healthApi.live().catch((e) => e)
      expect(err).toBeInstanceOf(ApiError)
      expect((err as ApiError).status).toBe(404)
      expect((err as ApiError).message).toContain('404')
    })

    it('throws ApiError on 500', async () => {
      mockFetch(() =>
        Promise.resolve({
          ok: false,
          status: 500,
          statusText: 'Internal Server Error',
          text: () => Promise.resolve('Internal Server Error'),
        } as unknown as Response),
      )

      const err = await healthApi.live().catch((e) => e)
      expect(err).toBeInstanceOf(ApiError)
      expect((err as ApiError).status).toBe(500)
    })

    it('ApiError carries the request path', async () => {
      mockFetch(() =>
        Promise.resolve({
          ok: false,
          status: 403,
          statusText: 'Forbidden',
          text: () => Promise.resolve('Forbidden'),
        } as unknown as Response),
      )

      const err = await healthApi.live().catch((e) => e)
      expect(err).toBeInstanceOf(ApiError)
      // health.live() calls GET /health/live
      expect((err as ApiError).path).toBe('/health/live')
    })
  })

  describe('timeout', () => {
    it('rejects with ApiError containing "timed out" when fetch hangs', async () => {
      vi.useFakeTimers()

      // fetch that never resolves, but honours the abort signal
      mockFetch((_url: string, opts: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          opts.signal?.addEventListener('abort', () => {
            reject(new DOMException('The operation was aborted', 'AbortError'))
          })
        }),
      )

      // Attach .catch() synchronously before advancing time so the rejection
      // is never momentarily unhandled between the timer firing and our await.
      const errPromise = healthApi.live().catch((e: unknown) => e)

      // Advance past DEFAULT_TIMEOUT_MS (30 000 ms)
      await vi.advanceTimersByTimeAsync(30_001)

      const err = await errPromise
      expect(err).toBeInstanceOf(ApiError)
      expect((err as ApiError).status).toBe(0)
      expect((err as ApiError).message).toContain('timed out')
    })
  })
})
