/* ============================================================
   AI RemixMate — Public legal pages (Phase 8)
   Reachable without auth so visitors can read terms / file DMCA.
   ============================================================ */
import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import './LegalPage.css'

const BASE = import.meta.env.VITE_API_BASE || '/api'

const PAGES = [
  { slug: 'terms', label: 'Terms' },
  { slug: 'privacy', label: 'Privacy' },
  { slug: 'dmca', label: 'DMCA' },
  { slug: 'acceptable-use', label: 'Acceptable Use' },
] as const

export default function LegalPage() {
  const { page = 'terms' } = useParams<{ page: string }>()
  const [text, setText] = useState<string>('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setError(null)
    setText('')
    fetch(`${BASE}/legal/${encodeURIComponent(page)}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`${res.status}`)
        return res.text()
      })
      .then((body) => {
        if (!cancelled) setText(body)
      })
      .catch((err) => {
        if (!cancelled) setError(String(err?.message || err))
      })
    return () => {
      cancelled = true
    }
  }, [page])

  return (
    <div className="legal-page">
      <header className="legal-page__header">
        <Link to="/mission-control" className="legal-page__brand">
          AI RemixMate
        </Link>
        <nav className="legal-page__nav" aria-label="Legal">
          {PAGES.map((p) => (
            <Link
              key={p.slug}
              to={`/legal/${p.slug}`}
              className={page === p.slug ? 'legal-page__link legal-page__link--active' : 'legal-page__link'}
            >
              {p.label}
            </Link>
          ))}
        </nav>
      </header>
      <main className="legal-page__body">
        {error && <p className="legal-page__error">Could not load legal text ({error}).</p>}
        {!error && !text && <p className="legal-page__muted">Loading…</p>}
        {text && <pre className="legal-page__text">{text}</pre>}
      </main>
    </div>
  )
}
