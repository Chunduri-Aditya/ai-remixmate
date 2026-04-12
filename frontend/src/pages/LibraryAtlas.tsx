/* ============================================================
   AI RemixMate — Library Atlas
   Full song library: search, filter, sort, per-song actions.
   ============================================================ */
import { Music2 } from 'lucide-react'
import './PageBase.css'

export default function LibraryAtlas() {
  return (
    <div className="page-base">
      <header className="page-base__header">
        <Music2 size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div>
          <h1 className="page-base__title font-display">Library Atlas</h1>
          <p className="page-base__sub text-muted">Browse, search, and manage your song collection</p>
        </div>
      </header>
      <div className="page-base__body">
        <div className="page-stub">
          <div className="page-stub__icon"><Music2 size={32} strokeWidth={1} /></div>
          <p className="page-stub__label font-display">Library Atlas</p>
          <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>Coming in Phase 5 — song table, search, filtering, per-song analysis</p>
        </div>
      </div>
    </div>
  )
}
