/* ============================================================
   AI RemixMate — LoginPage (Phase 4)

   Email + password sign-in / sign-up against Supabase. Shown by
   <RequireAuth> when auth is configured and there is no session.
   ============================================================ */

import { useState } from 'react'
import { useAuth } from './authContext'

type Mode = 'signin' | 'signup'

export function LoginPage() {
  const { signIn, signUp } = useAuth()
  const [mode, setMode] = useState<Mode>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      if (mode === 'signin') {
        const { error } = await signIn(email, password)
        if (error) setError(error)
      } else {
        const { error, needsConfirm } = await signUp(email, password)
        if (error) setError(error)
        else if (needsConfirm) setNotice('Check your email to confirm the account, then sign in.')
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-screen">
      <form className="auth-card" onSubmit={onSubmit}>
        <div className="auth-card__brand">
          <span className="auth-card__mark">◇</span>
          <h1 className="auth-card__title">AI RemixMate</h1>
        </div>
        <p className="auth-card__subtitle">
          {mode === 'signin' ? 'Sign in to your workspace' : 'Create your workspace'}
        </p>

        <label className="auth-field">
          <span>Email</span>
          <input
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </label>

        <label className="auth-field">
          <span>Password</span>
          <input
            type="password"
            autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>

        {error && <p className="auth-msg auth-msg--error">{error}</p>}
        {notice && <p className="auth-msg auth-msg--notice">{notice}</p>}

        <button className="auth-submit" type="submit" disabled={busy}>
          {busy ? '…' : mode === 'signin' ? 'Sign in' : 'Create account'}
        </button>

        <button
          type="button"
          className="auth-toggle"
          onClick={() => {
            setMode((m) => (m === 'signin' ? 'signup' : 'signin'))
            setError(null)
            setNotice(null)
          }}
        >
          {mode === 'signin'
            ? "Don't have an account? Create one"
            : 'Already have an account? Sign in'}
        </button>
      </form>
    </div>
  )
}
