import { useEffect, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useAppStore } from '@/stores/appStore'
import type { NavDestination } from '@/types'

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

const NUMBER_NAV: Array<{ id: NavDestination; path: string }> = [
  { id: 'mission-control', path: '/mission-control' },
  { id: 'library-atlas', path: '/library-atlas' },
  { id: 'mix-deck', path: '/mix-deck' },
  { id: 'set-builder', path: '/set-builder' },
  { id: 'signal-search', path: '/signal-search' },
  { id: 'ai-lab', path: '/ai-lab' },
  { id: 'mix-vault', path: '/mix-vault' },
  { id: 'operations', path: '/operations' },
]

const MIX_DECK_TOGGLE_EVENT = 'remixmate:mixdeck-toggle-playback'

function isTypingTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false
  return Boolean(target.closest('input, textarea, select, [contenteditable="true"]'))
}

export function useKeyboardShortcuts(
  onShowHelp: () => void,
  onHideHelp: () => void,
) {
  const navigate = useNavigate()
  const location = useLocation()
  const setActiveNav = useAppStore((s) => s.setActiveNav)
  const inspectorOpen = useAppStore((s) => s.inspectorOpen)
  const toggleInspector = useAppStore((s) => s.toggleInspector)
  const gPressed = useRef(false)
  const timer    = useRef<ReturnType<typeof setTimeout>>()

  useEffect(() => {
    function focusLibrarySearch(attempt = 0) {
      const input = document.querySelector<HTMLInputElement>('[data-library-search-input]')
      if (input) {
        input.focus()
        input.select()
        return
      }
      if (attempt < 12) {
        window.setTimeout(() => focusLibrarySearch(attempt + 1), 50)
      }
    }

    function handler(e: KeyboardEvent) {
      if (isTypingTarget(e.target)) return

      const hasCommandModifier = e.metaKey || e.ctrlKey
      const key = e.key.toLowerCase()

      if (hasCommandModifier && key === 'k') {
        e.preventDefault()
        setActiveNav('library-atlas')
        navigate('/library-atlas')
        window.setTimeout(() => focusLibrarySearch(), 0)
        return
      }

      if (hasCommandModifier && /^[1-8]$/.test(e.key)) {
        e.preventDefault()
        const destination = NUMBER_NAV[Number(e.key) - 1]
        if (destination) {
          setActiveNav(destination.id)
          navigate(destination.path)
        }
        return
      }

      if (hasCommandModifier) return

      if (e.key === ' ' && location.pathname.startsWith('/mix-deck')) {
        e.preventDefault()
        window.dispatchEvent(new Event(MIX_DECK_TOGGLE_EVENT))
        return
      }

      if (e.key === '?') { onShowHelp(); return }
      if (e.key === 'Escape') {
        onHideHelp()
        if (inspectorOpen) toggleInspector()
        return
      }

      if (e.key === 'g') {
        gPressed.current = true
        clearTimeout(timer.current)
        timer.current = setTimeout(() => { gPressed.current = false }, 1000)
        return
      }

      if (gPressed.current && NAV_MAP[e.key]) {
        gPressed.current = false
        const destination = NUMBER_NAV.find((item) => item.path === NAV_MAP[e.key])
        if (destination) setActiveNav(destination.id)
        navigate(NAV_MAP[e.key])
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [navigate, location.pathname, onShowHelp, onHideHelp, inspectorOpen, toggleInspector, setActiveNav])
}
