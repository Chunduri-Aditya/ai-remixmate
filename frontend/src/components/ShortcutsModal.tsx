import './ShortcutsModal.css'

const SHORTCUTS = [
  { keys: 'Space', desc: 'Play or pause the active Mix Deck track' },
  { keys: '⌘/Ctrl K', desc: 'Focus Library Atlas search' },
  { keys: '⌘/Ctrl 1–8', desc: 'Navigate primary pages' },
  { keys: 'g → m', desc: 'Mission Control' },
  { keys: 'g → l', desc: 'Library Atlas' },
  { keys: 'g → x', desc: 'Mix Deck' },
  { keys: 'g → s', desc: 'Set Builder' },
  { keys: 'g → q', desc: 'Signal Search' },
  { keys: 'g → a', desc: 'AI Lab' },
  { keys: 'g → v', desc: 'Mix Vault' },
  { keys: 'g → o', desc: 'Operations' },
  { keys: '?',     desc: 'Show this help' },
  { keys: 'Esc',   desc: 'Close modal or inspector' },
]

export function ShortcutsModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="shortcuts-backdrop" onClick={onClose}>
      <div className="shortcuts-modal" onClick={(e) => e.stopPropagation()}>
        <h3 className="shortcuts-modal__title font-display">Keyboard shortcuts</h3>
        <div className="shortcuts-modal__list">
          {SHORTCUTS.map((s) => (
            <div key={s.keys} className="shortcuts-modal__row">
              <kbd className="shortcuts-modal__kbd font-mono">{s.keys}</kbd>
              <span className="text-secondary">{s.desc}</span>
            </div>
          ))}
        </div>
        <button className="shortcuts-modal__close text-muted" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  )
}
