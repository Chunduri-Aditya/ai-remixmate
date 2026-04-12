/* ============================================================
   AI RemixMate — Set Builder
   Sequence songs into DJ sets; drag-and-drop ordering.
   ============================================================ */
import { ListMusic } from 'lucide-react'
import './PageBase.css'

export default function SetBuilder() {
  return (
    <div className="page-base">
      <header className="page-base__header">
        <ListMusic size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div>
          <h1 className="page-base__title font-display">Set Builder</h1>
          <p className="page-base__sub text-muted">Drag-and-drop set sequencing with energy flow visualisation</p>
        </div>
      </header>
      <div className="page-base__body">
        <div className="page-stub">
          <div className="page-stub__icon"><ListMusic size={32} strokeWidth={1} /></div>
          <p className="page-stub__label font-display">Set Builder</p>
          <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>Coming in Phase 6 — crate integration, set export, energy arc view</p>
        </div>
      </div>
    </div>
  )
}
