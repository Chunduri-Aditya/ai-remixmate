/* ============================================================
   AI RemixMate — Mix Vault
   Archive of completed mixes; playback, re-export, metadata.
   ============================================================ */
import { Archive } from 'lucide-react'
import './PageBase.css'

export default function MixVault() {
  return (
    <div className="page-base">
      <header className="page-base__header">
        <Archive size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div>
          <h1 className="page-base__title font-display">Mix Vault</h1>
          <p className="page-base__sub text-muted">Every mix you've made — playback, metadata, re-export</p>
        </div>
      </header>
      <div className="page-base__body">
        <div className="page-stub">
          <div className="page-stub__icon"><Archive size={32} strokeWidth={1} /></div>
          <p className="page-stub__label font-display">Mix Vault</p>
          <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>Coming in Phase 7 — output browser, waveform preview, mix history</p>
        </div>
      </div>
    </div>
  )
}
