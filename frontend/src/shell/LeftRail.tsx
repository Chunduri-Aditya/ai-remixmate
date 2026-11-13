/* ============================================================
   AI RemixMate — Left Navigation Rail
   64px fixed-width column; icon + label; nine destinations.
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
  LogOut,
  UserRound,
  type LucideIcon,
} from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'
import { useAppStore } from '@/stores/appStore'
import { authConfigured } from '@/lib/supabase'
import { useFeatures } from '@/hooks/useFeatures'
import { useAuth } from './authContext'
import type { NavDestination } from '@/types'
import './LeftRail.css'

const REPO_URL = 'https://github.com/Chunduri-Aditya/ai-remixmate'

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
  const { session, signOut } = useAuth()
  const { data: features } = useFeatures()
  const navItems = NAV_ITEMS.filter((item) => item.id !== 'ai-lab' || features?.generative === true)

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
        {navItems.map((item) => {
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

      {/* Bottom: legal links + sign out (when auth is on) + connection status */}
      <div className="left-rail__footer">
        <div className="left-rail__legal" aria-label="Legal">
          <button type="button" className="left-rail__legal-link" onClick={() => navigate('/legal/terms')}>
            Terms
          </button>
          <button type="button" className="left-rail__legal-link" onClick={() => navigate('/legal/privacy')}>
            Privacy
          </button>
          <button type="button" className="left-rail__legal-link" onClick={() => navigate('/legal/dmca')}>
            DMCA
          </button>
          <button type="button" className="left-rail__legal-link" onClick={() => navigate('/legal/acceptable-use')}>
            AUP
          </button>
        </div>
        <button
          className={`left-rail__item ${currentPath.startsWith('/account') ? 'left-rail__item--active' : ''}`}
          onClick={() => navigate('/account')}
          title="Account"
          aria-label="Account"
          aria-current={currentPath.startsWith('/account') ? 'page' : undefined}
        >
          <UserRound size={18} strokeWidth={1.5} />
          <span className="left-rail__item-label">Account</span>
        </button>
        {authConfigured && session && (
          <button
            className="left-rail__item"
            onClick={() => void signOut()}
            title={`Sign out — ${session.user.email ?? 'account'}`}
            aria-label="Sign out"
          >
            <LogOut size={18} strokeWidth={1.5} />
            <span className="left-rail__item-label">Sign out</span>
          </button>
        )}
        <div className="left-rail__status-dot">
          <ConnectionDot />
        </div>
        <p className="left-rail__credit" data-testid="left-rail-credit">
          AI RemixMate by Aditya Chunduri ·{' '}
          <a
            className="left-rail__credit-link"
            href={REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
        </p>
      </div>
    </nav>
  )
}
