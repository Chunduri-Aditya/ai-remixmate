/**
 * Unit tests for src/lib/supabase.ts with NO env configured — the local /
 * test posture. `authConfigured` is false, `supabase` is null, nothing throws,
 * and the SSE cookie helper still round-trips.
 */
import { afterEach, describe, expect, it } from 'vitest'
import { authConfigured, getAccessToken, supabase, syncSseCookie } from '@/lib/supabase'

describe('supabase.ts (unconfigured)', () => {
  it('is inert without VITE_SUPABASE_* env', () => {
    expect(authConfigured).toBe(false)
    expect(supabase).toBeNull()
  })

  it('getAccessToken resolves to null (no network)', async () => {
    await expect(getAccessToken()).resolves.toBeNull()
  })
})

describe('syncSseCookie', () => {
  afterEach(() => {
    document.cookie = 'sb-access-token=; Path=/; Max-Age=0'
  })

  it('writes the token cookie the SSE stream reads', () => {
    syncSseCookie('jwt-abc-123')
    expect(document.cookie).toContain('sb-access-token=jwt-abc-123')
  })

  it('clears the cookie on sign-out', () => {
    syncSseCookie('jwt-abc-123')
    syncSseCookie(null)
    expect(document.cookie).not.toContain('sb-access-token=jwt-abc-123')
  })
})
