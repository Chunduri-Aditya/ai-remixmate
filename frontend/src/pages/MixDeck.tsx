/* ============================================================
   AI RemixMate — Mix Deck
   DJ-style two-deck remixer: compatibility, preview, full mix.
   ============================================================ */
import { Sliders } from 'lucide-react'
import './PageBase.css'

export default function MixDeck() {
  return (
    <div className="page-base">
      <header className="page-base__header">
        <Sliders size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div>
          <h1 className="page-base__title font-display">Mix Deck</h1>
          <p className="page-base__sub text-muted">Harmonic matching, transition preview, DJ remixer</p>
        </div>
      </header>
      <div className="page-base__body">
        <div className="page-stub">
          <div className="page-stub__icon"><Sliders size={32} strokeWidth={1} /></div>
          <p className="page-stub__label font-display">Mix Deck</p>
          <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>Coming in Phase 6 — dual deck UI, BPM/key display, transition controls</p>
        </div>
      </div>
    </div>
  )
}
