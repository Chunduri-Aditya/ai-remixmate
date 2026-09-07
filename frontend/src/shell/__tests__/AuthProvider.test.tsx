/**
 * AuthProvider session lifecycle — the piece the auth.test.tsx cases skip by
 * injecting AuthContext directly.
 *
 * `@/lib/supabase` is mocked with a fake auth client whose onAuthStateChange
 * callback we drive by hand, so we can assert the cookie mirror, the query
 * cache clear on SIGNED_OUT, and the subscription / handler cleanup.
 */
import { cleanup, render, waitFor } from '@testing-library/react'
import { act } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'

const h = vi.hoisted(() => {
  const unsubscribe = vi.fn()
  const state = { cb: null as null | ((e: string, s: unknown) => void) }
  return {
    state,
    unsubscribe,
    syncSseCookie: vi.fn(),
    setUnauthorizedHandler: vi.fn(),
    signOut: vi.fn(),
    getSession: vi.fn<
      () => Promise<{ data: { session: { access_token: string } | null } }>
    >(async () => ({ data: { session: null } })),
    onAuthStateChange: vi.fn((fn: (e: string, s: unknown) => void) => {
      state.cb = fn
      return { data: { subscription: { unsubscribe } } }
    }),
    // capture the 401 handler AuthProvider registers so we can invoke it
    handler: { fn: null as null | (() => void) },
  }
})
h.setUnauthorizedHandler.mockImplementation((fn: (() => void) | null) => {
  h.handler.fn = fn
})

vi.mock('@/lib/supabase', () => ({
  authConfigured: true,
  supabase: {
    auth: {
      getSession: h.getSession,
      onAuthStateChange: h.onAuthStateChange,
      signOut: h.signOut,
    },
  },
  syncSseCookie: h.syncSseCookie,
}))
vi.mock('@/lib/api', () => ({ setUnauthorizedHandler: h.setUnauthorizedHandler }))

import { AuthProvider } from '../AuthProvider'

function mount() {
  const qc = new QueryClient()
  const clearSpy = vi.spyOn(qc, 'clear')
  const view = render(
    <QueryClientProvider client={qc}>
      <AuthProvider>
        <div>app</div>
      </AuthProvider>
    </QueryClientProvider>,
  )
  return { ...view, clearSpy }
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  h.state.cb = null
  h.handler.fn = null
})

describe('AuthProvider', () => {
  it('mirrors the initial (empty) session to the SSE cookie and registers a 401 handler', async () => {
    mount()
    await waitFor(() => expect(h.syncSseCookie).toHaveBeenCalledWith(null))
    expect(h.setUnauthorizedHandler).toHaveBeenCalledWith(expect.any(Function))
  })

  it('re-mirrors the cookie on every auth-state change (token refresh included)', async () => {
    mount()
    await waitFor(() => expect(h.onAuthStateChange).toHaveBeenCalled())
    h.syncSseCookie.mockClear()

    act(() => h.state.cb!('TOKEN_REFRESHED', { access_token: 'fresh-jwt' }))
    expect(h.syncSseCookie).toHaveBeenCalledWith('fresh-jwt')
  })

  it('clears the cookie and the query cache on SIGNED_OUT', async () => {
    const { clearSpy } = mount()
    await waitFor(() => expect(h.onAuthStateChange).toHaveBeenCalled())
    h.syncSseCookie.mockClear()

    act(() => h.state.cb!('SIGNED_OUT', null))
    expect(h.syncSseCookie).toHaveBeenCalledWith(null)
    expect(clearSpy).toHaveBeenCalledOnce()
  })

  it('does NOT clear the query cache on a non-signout event', async () => {
    const { clearSpy } = mount()
    await waitFor(() => expect(h.onAuthStateChange).toHaveBeenCalled())

    act(() => h.state.cb!('TOKEN_REFRESHED', { access_token: 'x' }))
    expect(clearSpy).not.toHaveBeenCalled()
  })

  it('tears down the subscription and the 401 handler on unmount', async () => {
    const { unmount } = mount()
    await waitFor(() => expect(h.onAuthStateChange).toHaveBeenCalled())

    unmount()
    expect(h.unsubscribe).toHaveBeenCalledOnce()
    expect(h.setUnauthorizedHandler).toHaveBeenLastCalledWith(null)
  })

  it('mirrors a non-empty initial session from getSession()', async () => {
    h.getSession.mockResolvedValueOnce({ data: { session: { access_token: 'initial-jwt' } } })
    mount()
    await waitFor(() => expect(h.syncSseCookie).toHaveBeenCalledWith('initial-jwt'))
  })

  it('the registered 401 handler signs the user out', async () => {
    mount()
    await waitFor(() => expect(h.handler.fn).toBeInstanceOf(Function))

    h.handler.fn!()
    expect(h.signOut).toHaveBeenCalledOnce()
  })
})
