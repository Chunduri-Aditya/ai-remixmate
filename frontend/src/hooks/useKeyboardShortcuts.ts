import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

const NAV_MAP: Record<string, string> = {
  m: '/mission-control',
  l: '/library-atlas',
  x: '/mix-deck',
  s: '/set-builder',
  q: '/signal-search',
  a: '/ai-lab',
  v: '/mix-vault',
  o: '/operations',
}

export function useKeyboardShortcuts(
  onShowHelp: () => void,
  onHideHelp: () => void,
) {
  const navigate = useNavigate()
  const gPressed = useRef(false)
  const timer    = useRef<ReturnType<typeof setTimeout>>()

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
      if (e.metaKey || e.ctrlKey) return

      if (e.key === '?') { onShowHelp(); return }
      if (e.key === 'Escape') { onHideHelp(); return }

      if (e.key === 'g') {
        gPressed.current = true
        clearTimeout(timer.current)
        timer.current = setTimeout(() => { gPressed.current = false }, 1000)
        return
      }

      if (gPressed.current && NAV_MAP[e.key]) {
        gPressed.current = false
        navigate(NAV_MAP[e.key])
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [navigate, onShowHelp, onHideHelp])
}
