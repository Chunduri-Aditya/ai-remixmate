/* ============================================================
   AI RemixMate — AppShell
   3-zone CSS Grid: [LeftRail | Canvas | RightInspector]
   The canvas is router-controlled; the two side zones are persistent.
   ============================================================ */

import { Suspense, lazy } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { LeftRail } from './LeftRail'
import { RightInspector } from './RightInspector'
import { useAppStore } from '@/stores/appStore'
import { useSSE } from '@/hooks/useSSE'
import { useJobPoller } from '@/hooks/useJobPoller'
import './AppShell.css'

// Lazy-load pages — each is a separate code chunk
const MissionControl = lazy(() => import('@/pages/MissionControl'))
const LibraryAtlas   = lazy(() => import('@/pages/LibraryAtlas'))
const MixDeck        = lazy(() => import('@/pages/MixDeck'))
const SetBuilder     = lazy(() => import('@/pages/SetBuilder'))
const SignalSearch   = lazy(() => import('@/pages/SignalSearch'))
const AILab          = lazy(() => import('@/pages/AILab'))
const MixVault       = lazy(() => import('@/pages/MixVault'))
const Operations     = lazy(() => import('@/pages/Operations'))

function PageFallback() {
  return (
    <div className="page-fallback">
      <div className="page-fallback__spinner" />
    </div>
  )
}

export function AppShell() {
  // Mount SSE connection at shell level — persists across page navigation
  useSSE()
  // Polling fallback when SSE is unavailable
  const sseConnected = useAppStore((s) => s.sseConnected)
  useJobPoller(!sseConnected)

  const inspectorOpen = useAppStore((s) => s.inspectorOpen)

  return (
    <div className={`app-shell ${inspectorOpen ? 'app-shell--inspector-open' : ''}`}>
      <LeftRail />

      <main className="app-shell__canvas">
        <Suspense fallback={<PageFallback />}>
          <Routes>
            <Route index element={<Navigate to="/mission-control" replace />} />
            <Route path="mission-control" element={<MissionControl />} />
            <Route path="library-atlas"   element={<LibraryAtlas />} />
            <Route path="mix-deck"        element={<MixDeck />} />
            <Route path="set-builder"     element={<SetBuilder />} />
            <Route path="signal-search"   element={<SignalSearch />} />
            <Route path="ai-lab"          element={<AILab />} />
            <Route path="mix-vault"       element={<MixVault />} />
            <Route path="operations"      element={<Operations />} />
            <Route path="*"              element={<Navigate to="/mission-control" replace />} />
          </Routes>
        </Suspense>
      </main>

      {inspectorOpen && <RightInspector />}
    </div>
  )
}
