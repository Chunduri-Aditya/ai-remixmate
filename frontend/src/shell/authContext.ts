/* ============================================================
   Auth context + consumer hook (Phase 4).

   Split out of AuthProvider.tsx so that file exports only a component —
   react-refresh/only-export-components needs component files to export
   components and nothing else (same pattern as toastContext.ts).
   ============================================================ */
import { createContext, useContext } from 'react'
import type { Session, User } from '@supabase/supabase-js'

export interface AuthState {
  /** Current Supabase session, or null when signed out / auth is off. */
  session: Session | null
  user: User | null
  /** True until the initial getSession() settles. */
  loading: boolean
  signIn: (email: string, password: string) => Promise<{ error: string | null }>
  signUp: (email: string, password: string) => Promise<{ error: string | null; needsConfirm: boolean }>
  signOut: () => Promise<void>
}

const NOOP_AUTH: AuthState = {
  session: null,
  user: null,
  loading: false,
  signIn: async () => ({ error: 'auth is not configured' }),
  signUp: async () => ({ error: 'auth is not configured', needsConfirm: false }),
  signOut: async () => {},
}

export const AuthContext = createContext<AuthState>(NOOP_AUTH)

export function useAuth(): AuthState {
  return useContext(AuthContext)
}
