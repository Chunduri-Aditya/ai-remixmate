/**
 * Phase 4 frontend auth shell.
 *
 * `@/lib/supabase` is mocked here with `authConfigured = true` so <RequireAuth>
 * exercises the gated paths; the no-auth pass-through is covered by
 * supabase.test.ts + smoke.test.ts (real, unconfigured module).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import type { Session } from '@supabase/supabase-js'

const { getAccessToken } = vi.hoisted(() => ({
  getAccessToken: vi.fn<() => Promise<string | null>>(),
}))

vi.mock('@/lib/supabase', () => ({
  authConfigured: true,
  supabase: {},
  getAccessToken,
  syncSseCookie: vi.fn(),
}))

import { setUnauthorizedHandler, healthApi, ApiError } from '@/lib/api'
import { AuthContext, type AuthState } from '@/shell/authContext'
import { RequireAuth } from '@/shell/RequireAuth'

function withAuth(partial: Partial<AuthState>) {
  const value: AuthState = {
    session: null,
    user: null,
    loading: false,
    signIn: async () => ({ error: null }),
    signUp: async () => ({ error: null, needsConfirm: false }),
    signOut: async () => {},
    ...partial,
  }
  return (
    <AuthContext.Provider value={value}>
      <RequireAuth>
        <div>protected content</div>
      </RequireAuth>
    </AuthContext.Provider>
  )
}

const FAKE_SESSION = { access_token: 'tok', user: { id: 'u1', email: 'a@b.c' } } as unknown as Session

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  setUnauthorizedHandler(null)
})

describe('<RequireAuth> (auth configured)', () => {
  it('shows the login screen when there is no session', () => {
    render(withAuth({ session: null, loading: false }))
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeInTheDocument()
    expect(screen.queryByText('protected content')).not.toBeInTheDocument()
  })

  it('shows a splash while the session is loading', () => {
    render(withAuth({ loading: true }))
    expect(screen.getByRole('status')).toBeInTheDocument()
    expect(screen.queryByText('protected content')).not.toBeInTheDocument()
  })

  it('renders children once a session exists', () => {
    render(withAuth({ session: FAKE_SESSION, user: FAKE_SESSION.user, loading: false }))
    expect(screen.getByText('protected content')).toBeInTheDocument()
  })

  it('can toggle to the sign-up form', async () => {
    const { default: userEvent } = await import('@testing-library/user-event')
    render(withAuth({ session: null }))
    await userEvent.click(screen.getByRole('button', { name: /create one/i }))
    expect(screen.getByRole('button', { name: 'Create account' })).toBeInTheDocument()
  })
})

describe('api.ts auth wiring', () => {
  let calls: Array<{ url: string; init: RequestInit }>

  beforeEach(() => {
    calls = []
    vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
      calls.push({ url, init })
      return { ok: true, status: 200, json: async () => ({}), text: async () => '' } as unknown as Response
    })
  })
  afterEach(() => vi.unstubAllGlobals())

  it('attaches Authorization: Bearer <token> and credentials: include', async () => {
    getAccessToken.mockResolvedValue('tok-xyz')
    await healthApi.live()
    expect(calls).toHaveLength(1)
    expect((calls[0].init.headers as Record<string, string>).Authorization).toBe('Bearer tok-xyz')
    expect(calls[0].init.credentials).toBe('include')
  })

  it('sends no Authorization header when there is no token', async () => {
    getAccessToken.mockResolvedValue(null)
    await healthApi.live()
    expect((calls[0].init.headers as Record<string, string>).Authorization).toBeUndefined()
    expect(calls[0].init.credentials).toBe('include')
  })

  it('fires the unauthorized handler on a 401', async () => {
    getAccessToken.mockResolvedValue('stale')
    vi.stubGlobal('fetch', async () =>
      ({ ok: false, status: 401, json: async () => ({}), text: async () => 'unauthorized' }) as unknown as Response,
    )
    const onUnauth = vi.fn()
    setUnauthorizedHandler(onUnauth)

    await expect(healthApi.live()).rejects.toBeInstanceOf(ApiError)
    expect(onUnauth).toHaveBeenCalledOnce()
  })
})
