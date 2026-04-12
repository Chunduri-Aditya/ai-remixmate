/* ============================================================
   AI RemixMate — Signal Search
   Semantic vector search across the indexed library.
   ============================================================ */
import { Search } from 'lucide-react'
import './PageBase.css'

export default function SignalSearch() {
  return (
    <div className="page-base">
      <header className="page-base__header">
        <Search size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div>
          <h1 className="page-base__title font-display">Signal Search</h1>
          <p className="page-base__sub text-muted">Semantic similarity search powered by the 35-dim embedding index</p>
        </div>
      </header>
      <div className="page-base__body">
        <div className="page-stub">
          <div className="page-stub__icon"><Search size={32} strokeWidth={1} /></div>
          <p className="page-stub__label font-display">Signal Search</p>
          <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>Coming in Phase 5 — query bar, similarity results, BPM/key/energy filters</p>
        </div>
      </div>
    </div>
  )
}
