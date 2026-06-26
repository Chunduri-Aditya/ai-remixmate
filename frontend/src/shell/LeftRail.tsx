/* ============================================================
   AI RemixMate — Left Navigation Rail
   64px fixed-width column; icon + label; 8 destinations.
   ============================================================ */

import { useNavigate, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Library,
  Sliders,
  ListMusic,
  Search,
  FlaskConical,
  Archive,
  Settings,
  Radio,
  PictureInPicture2,
  type LucideIcon,
} from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'
import { useAppStore } from '@/stores/appStore'
import type { NavDestination } from '@/types'
import './LeftRail.css'

interface NavItem {
  id: NavDestination
  path: string
  icon: LucideIcon
  label: string
  shortLabel: string
}

const NAV_ITEMS: NavItem[] = [
  {
    id: 'mission-control',
    path: '/mission-control',
    icon: LayoutDashboard,
    label: 'Mission Control',
    shortLabel: 'Control',
  },
  {
    id: 'library-atlas',
    path: '/library-atlas',
    icon: Library,
    label: 'Library Atlas',
    shortLabel: 'Library',
  },
  {
    id: 'mix-deck',
    path: '/mix-deck',
    icon: Sliders,
    label: 'Mix Deck',
    shortLabel: 'Mix',
  },
  {
    id: 'set-builder',
    path: '/set-builder',
    icon: ListMusic,
    label: 'Set Builder',
    shortLabel: 'Sets',
  },
  {
    id: 'signal-search',
    path: '/signal-search',
    icon: Search,
    label: 'Signal Search',
    shortLabel: 'Search',
  },
  {
    id: 'ai-lab',
    path: '/ai-lab',
    icon: FlaskConical,
    label: 'AI Lab',
    shortLabel: 'AI Lab',
  },
  {
    id: 'mix-vault',
    path: '/mix-vault',
    icon: Archive,
    label: 'Mix Vault',
    shortLabel: 'Vault',
  },
  {
    id: 'operations',
    path: '/operations',
    icon: Settings,
    label: 'Operations',
    shortLabel: 'Ops',
  },
  {
    id: 'widget',
    path: '/widget',
    icon: PictureInPicture2,
    label: 'DJ Widget (floating)',
    shortLabel: 'Widget',
  },
]

function ConnectionDot() {
  const { apiHealth, sseConnected } = useAppStore(
    useShallow((s) => ({
      apiHealth: s.apiHealth,
      sseConnected: s.sseConnected,
    })),
  )

  const cls =
    apiHealth === 'ok' && sseConnected
      ? 'pulse-dot pulse-dot--green'
      : apiHealth === 'degraded'
        ? 'pulse-dot pulse-dot--amber'
        : 'pulse-dot pulse-dot--crimson'

  const title =
    apiHealth === 'ok' && sseConnected
      ? 'Connected — live stream active'
      : apiHealth === 'degraded'
        ? 'Degraded — API reachable, no live stream'
        : 'Disconnected'

  return <span className={cls} title={title} />
}

export function LeftRail() {
  const navigate = useNavigate()
  const location = useLocation()
  const setActiveNav = useAppStore((s) => s.setActiveNav)

  function handleNav(item: NavItem) {
    setActiveNav(item.id)
    navigate(item.path)
  }

  const currentPath = location.pathname.replace(/\/$/, '') || '/mission-control'

  return (
    <nav className="left-rail" aria-label="Primary navigation">
      {/* Wordmark / logo */}
      <div className="left-rail__logo" title="AI RemixMate">
        <Radio size={22} strokeWidth={1.5} className="left-rail__logo-icon" />
      </div>

      <div className="left-rail__divider" />

      {/* Nav items */}
      <ul className="left-rail__nav" role="list">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon
          const isActive = currentPath.startsWith(item.path)
          return (
            <li key={item.id}>
              <button
                className={`left-rail__item ${isActive ? 'left-rail__item--active' : ''}`}
                onClick={() => handleNav(item)}
                title={item.label}
                aria-label={item.label}
                aria-current={isActive ? 'page' : undefined}
              >
                <Icon size={18} strokeWidth={isActive ? 2 : 1.5} />
                <span className="left-rail__item-label">{item.shortLabel}</span>
                {isActive && <span className="left-rail__item-indicator" aria-hidden="true" />}
              </button>
            </li>
          )
        })}
      </ul>

      {/* Bottom: connection status dot */}
      <div className="left-rail__footer">
        <div className="left-rail__status-dot">
          <ConnectionDot />
        </div>
      </div>
    </nav>
  )
}
