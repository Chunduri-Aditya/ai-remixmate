/* ============================================================
   AI RemixMate — RequireAuth (Phase 4)

   Gate around the whole app:
     - auth not configured  → render children (pre-Phase-4 behaviour)
     - session loading      → a splash
     - no session           → the login screen
     - session              → render children
   ============================================================ */

import { authConfigured } from '@/lib/supabase'
import { useAuth } from './authContext'
import { LoginPage } from './LoginPage'
import './LoginPage.css'

export function RequireAuth({ children }: { children: React.ReactNode }) {
  const { session, loading } = useAuth()

  if (!authConfigured) return <>{children}</>

  if (loading) {
    return (
      <div className="auth-screen">
        <div className="auth-splash" role="status" aria-label="Loading session" />
      </div>
    )
  }

  if (!session) return <LoginPage />

  return <>{children}</>
}
