/* ============================================================
   AI RemixMate — Account
   Who is signed in, export of the user's data, and account
   deletion gated by typing DELETE. Delete is off in local mode.
   ============================================================ */
import { useState } from 'react'
import { UserRound } from 'lucide-react'
import { authConfigured, getAccessToken } from '@/lib/supabase'
import { useAuth } from '@/shell/authContext'
import './PageBase.css'
import './AccountPage.css'

const BASE = import.meta.env.VITE_API_BASE || '/api'
const CONFIRM_WORD = 'DELETE'

/** Fetch against the API with the Bearer token, the same way lib/api does. */
async function authedFetch(method: string, path: string): Promise<Response> {
  const headers: Record<string, string> = {}
  const token = await getAccessToken()
  if (token) headers.Authorization = `Bearer ${token}`
  const res = await fetch(`${BASE}${path}`, { method, headers, credentials: 'include' })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`[${res.status}] ${path}: ${text}`)
  }
  return res
}

/** Account page: signed in email (or "local"), Export, and gated Delete. */
export default function AccountPage() {
  const { user, session, signOut } = useAuth()
  const authOn = authConfigured
  const email = user?.email ?? session?.user.email ?? null

  const [confirm, setConfirm] = useState('')
  const [busy, setBusy] = useState<'export' | 'delete' | null>(null)
  const [error, setError] = useState<string | null>(null)

  const canDelete = authOn && confirm === CONFIRM_WORD && busy === null

  async function onExport() {
    setBusy('export')
    setError(null)
    try {
      const res = await authedFetch('GET', '/account/export')
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'remixmate-account-export.zip'
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  async function onDelete() {
    if (!canDelete) return
    setBusy('delete')
    setError(null)
    try {
      await authedFetch('DELETE', '/account')
      await signOut()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="page-base">
      <header className="page-base__header">
        <UserRound size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div style={{ flex: 1 }}>
          <h1 className="page-base__title font-display">Account</h1>
          <p className="page-base__sub text-muted">
            Signed in as <span className="font-mono">{authOn ? email ?? 'unknown' : 'local'}</span>
          </p>
        </div>
      </header>

      <div className="page-base__body acct-body">
        <section className="acct-section">
          <h2 className="acct-section__title">Export your data</h2>
          <p className="text-muted acct-section__text">
            Download a zip of your mix manifests, crates, and favorites.
          </p>
          <button type="button" className="acct-btn" onClick={() => void onExport()} disabled={busy !== null}>
            {busy === 'export' ? 'Exporting…' : 'Export'}
          </button>
        </section>

        <section className="acct-section acct-section--danger">
          <h2 className="acct-section__title">Delete everything</h2>
          <p className="text-muted acct-section__text">
            Removes your rows, files, audit lines, and the sign in user. This cannot be undone.
            Type {CONFIRM_WORD} to confirm.
          </p>
          {!authOn && (
            <p className="acct-note">Auth is not configured, so there is no hosted account to delete.</p>
          )}
          <div className="acct-confirm">
            <input
              type="text"
              className="acct-input"
              aria-label={`Type ${CONFIRM_WORD} to confirm`}
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              disabled={!authOn}
              autoComplete="off"
            />
            <button
              type="button"
              className="acct-btn acct-btn--danger"
              onClick={() => void onDelete()}
              disabled={!canDelete}
            >
              {busy === 'delete' ? 'Deleting…' : 'Delete'}
            </button>
          </div>
        </section>

        {error && <p className="acct-error" role="alert">{error}</p>}
      </div>
    </div>
  )
}
