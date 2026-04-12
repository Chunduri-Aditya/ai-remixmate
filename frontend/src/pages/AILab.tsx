/* ============================================================
   AI RemixMate — AI Lab
   Generative features: style transfer, inpainting, tokenization.
   ============================================================ */
import { FlaskConical } from 'lucide-react'
import './PageBase.css'

export default function AILab() {
  return (
    <div className="page-base">
      <header className="page-base__header">
        <FlaskConical size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div>
          <h1 className="page-base__title font-display">AI Lab</h1>
          <p className="page-base__sub text-muted">Style transfer, inpainting, generative remixing</p>
        </div>
      </header>
      <div className="page-base__body">
        <div className="page-stub">
          <div className="page-stub__icon"><FlaskConical size={32} strokeWidth={1} /></div>
          <p className="page-stub__label font-display">AI Lab</p>
          <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>Coming in Phase 7 — model selector, parameter controls, output player</p>
        </div>
      </div>
    </div>
  )
}
