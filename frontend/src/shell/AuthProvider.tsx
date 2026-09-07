/* ============================================================
   AI RemixMate — AuthProvider (Phase 4)

   Owns the Supabase session lifecycle:
     - loads the initial session, subscribes to auth-state changes
     - mirrors the access token into the `sb-access-token` cookie the SSE
       stream authenticates with (syncSseCookie)
     - registers the 401 handler api.ts calls when a gated request lapses

   When auth is not configured (no VITE_SUPABASE_* env), it renders children
   immediately with an inert context — the app behaves exactly as pre-Phase-4.
   ============================================================ */

import { useEffect, useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { Session } from '@supabase/supabase-js'
import { authConfigured, supabase, syncSseCookie } from '@/lib/supabase'
import { setUnauthorizedHandler } from '@/lib/api'
import { AuthContext, type AuthState } from './authContext'

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(authConfigured)
  const queryClient = useQueryClient()

  useEffect(() => {
    if (!supabase) return
    const sb = supabase // non-null within this effect

    sb.auth.getSession().then(({ data }) => {
      setSession(data.session)
      syncSseCookie(data.session?.access_token ?? null)
      setLoading(false)
    })

    const { data: sub } = sb.auth.onAuthStateChange((event, next) => {
      setSession(next)
      syncSseCookie(next?.access_token ?? null)
      // Drop the previous user's cached crates/jobs/etc so a re-login on the
      // same tab doesn't briefly show them (staleTime is 15s).
      if (event === 'SIGNED_OUT') queryClient.clear()
    })

    // A 401 from a gated route means the server rejected our token — drop the
    // session so <RequireAuth> falls back to the login screen.
    setUnauthorizedHandler(() => {
      void sb.auth.signOut()
    })

    return () => {
      sub.subscription.unsubscribe()
      setUnauthorizedHandler(null)
    }
  }, [queryClient]) // queryClient is stable — effect runs once

  const value = useMemo<AuthState>(
    () => ({
      session,
      user: session?.user ?? null,
      loading,
      signIn: async (email, password) => {
        if (!supabase) return { error: 'auth is not configured' }
        const { error } = await supabase.auth.signInWithPassword({ email, password })
        return { error: error?.message ?? null }
      },
      signUp: async (email, password) => {
        if (!supabase) return { error: 'auth is not configured', needsConfirm: false }
        const { data, error } = await supabase.auth.signUp({ email, password })
        // No session back = email confirmation is required for this project.
        return { error: error?.message ?? null, needsConfirm: !error && !data.session }
      },
      signOut: async () => {
        await supabase?.auth.signOut()
      },
    }),
    [session, loading],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
