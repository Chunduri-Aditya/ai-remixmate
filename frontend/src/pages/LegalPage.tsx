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

// Mirrors DmcaRequest in scripts/api/routers/legal.py.
const DMCA_FIELDS = [
  { name: 'reporter_name', label: 'Your name', type: 'text', required: true },
  { name: 'reporter_email', label: 'Your email', type: 'email', required: true },
  { name: 'copyright_work', label: 'Copyrighted work', type: 'text', required: true },
  { name: 'infringing_url', label: 'Infringing URL', type: 'url', required: true },
  { name: 'description', label: 'Description', type: 'textarea', required: false },
] as const

type DmcaField = (typeof DMCA_FIELDS)[number]['name']
type DmcaValues = Record<DmcaField, string>

const EMPTY_DMCA: DmcaValues = {
  reporter_name: '',
  reporter_email: '',
  copyright_work: '',
  infringing_url: '',
  description: '',
}

/** DMCA intake form: posts the DmcaRequest fields and shows the stored report id. */
function DmcaForm() {
  const [values, setValues] = useState<DmcaValues>(EMPTY_DMCA)
  const [busy, setBusy] = useState(false)
  const [reportId, setReportId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const res = await fetch(`${BASE}/legal/dmca`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
      })
      if (!res.ok) {
        const text = await res.text().catch(() => res.statusText)
        throw new Error(`${res.status} ${text}`)
      }
      const body = (await res.json()) as { report_id: string }
      setReportId(body.report_id)
      setValues(EMPTY_DMCA)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="legal-form" onSubmit={onSubmit}>
      <h2 className="legal-form__title">File a takedown report</h2>
      {DMCA_FIELDS.map((f) => (
        <label key={f.name} className="legal-form__field">
          <span>{f.label}</span>
          {f.type === 'textarea' ? (
            <textarea
              name={f.name}
              rows={5}
              maxLength={5000}
              value={values[f.name]}
              onChange={(e) => setValues((v) => ({ ...v, [f.name]: e.target.value }))}
            />
          ) : (
            <input
              name={f.name}
              type={f.type}
              required={f.required}
              value={values[f.name]}
              onChange={(e) => setValues((v) => ({ ...v, [f.name]: e.target.value }))}
            />
          )}
        </label>
      ))}
      <button type="submit" className="legal-form__submit" disabled={busy}>
        {busy ? 'Sending…' : 'Submit report'}
      </button>
      {reportId && (
        <p className="legal-form__ok" role="status">
          Report received. Report id: <code>{reportId}</code>
        </p>
      )}
      {error && <p className="legal-page__error">Could not submit the report ({error}).</p>}
    </form>
  )
}

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
        {page === 'dmca' && <DmcaForm />}
      </main>
    </div>
  )
}
