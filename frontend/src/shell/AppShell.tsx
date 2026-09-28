/* ============================================================
   AI RemixMate — AppShell
   3-zone CSS Grid: [LeftRail | Canvas | RightInspector]
   The canvas is router-controlled; the two side zones are persistent.
   ============================================================ */

import { Suspense, lazy, useState, type ReactNode } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { LeftRail } from './LeftRail'
import { RightInspector } from './RightInspector'
import { useAppStore } from '@/stores/appStore'
import { useSSE } from '@/hooks/useSSE'
import { useJobPoller } from '@/hooks/useJobPoller'
import { ToastProvider } from '@/components/Toast'
import { useJobToasts } from '@/hooks/useJobToasts'
import { ShortcutsModal } from '@/components/ShortcutsModal'
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'
import { useFeatures } from '@/hooks/useFeatures'
import { PageErrorBoundary } from '@/components/PageErrorBoundary'
import { PlayerBar } from '@/components/PlayerBar'
import './AppShell.css'

function JobToastWatcher() {
  useJobToasts()
  return null
}

// Lazy-load pages — each is a separate code chunk
const MissionControl = lazy(() => import('@/pages/MissionControl'))
const LibraryAtlas   = lazy(() => import('@/pages/LibraryAtlas'))
const MixDeck        = lazy(() => import('@/pages/MixDeck'))
const SetBuilder     = lazy(() => import('@/pages/SetBuilder'))
const SignalSearch   = lazy(() => import('@/pages/SignalSearch'))
const AILab          = lazy(() => import('@/pages/AILab'))
const MixVault       = lazy(() => import('@/pages/MixVault'))
const Operations     = lazy(() => import('@/pages/Operations'))
const AccountPage    = lazy(() => import('@/pages/AccountPage'))

function PageFallback() {
  return (
    <div className="page-fallback">
      <div className="page-fallback__spinner" />
    </div>
  )
}

// B5.9: /ai-lab is not a page while `generative` is off. The flag counts as
// off until it is known on (useFeatures), so the page never renders before
// the answer; a loaded false redirects, matching the rail and the shortcuts.
function RequireGenerative({ children }: { children: ReactNode }) {
  const { data: features, isPending } = useFeatures()
  if (features?.generative === true) return <>{children}</>
  if (isPending) return <PageFallback />
  return <Navigate to="/mission-control" replace />
}

export function AppShell() {
  // Mount SSE connection at shell level — persists across page navigation
  useSSE()
  // Polling fallback when SSE is unavailable
  const sseConnected = useAppStore((s) => s.sseConnected)
  useJobPoller(!sseConnected)

  const inspectorOpen = useAppStore((s) => s.inspectorOpen)
  const [showShortcuts, setShowShortcuts] = useState(false)
  useKeyboardShortcuts(
    () => setShowShortcuts(true),
    () => setShowShortcuts(false),
  )

  return (
    <ToastProvider>
      <JobToastWatcher />
      <div className={`app-shell ${inspectorOpen ? 'app-shell--inspector-open' : ''}`}>
        <LeftRail />

        <main className="app-shell__canvas">
          <Routes>
            <Route index element={<Navigate to="/mission-control" replace />} />
              <Route path="mission-control" element={
                <PageErrorBoundary pageName="Mission Control">
                  <Suspense fallback={<PageFallback />}><MissionControl /></Suspense>
                </PageErrorBoundary>
              } />
              <Route path="library-atlas" element={
                <PageErrorBoundary pageName="Library Atlas">
                  <Suspense fallback={<PageFallback />}><LibraryAtlas /></Suspense>
                </PageErrorBoundary>
              } />
              <Route path="mix-deck" element={
                <PageErrorBoundary pageName="Mix Deck">
                  <Suspense fallback={<PageFallback />}><MixDeck /></Suspense>
                </PageErrorBoundary>
              } />
              <Route path="set-builder" element={
                <PageErrorBoundary pageName="Set Builder">
                  <Suspense fallback={<PageFallback />}><SetBuilder /></Suspense>
                </PageErrorBoundary>
              } />
              <Route path="signal-search" element={
                <PageErrorBoundary pageName="Signal Search">
                  <Suspense fallback={<PageFallback />}><SignalSearch /></Suspense>
                </PageErrorBoundary>
              } />
              <Route path="ai-lab" element={
                <RequireGenerative>
                  <PageErrorBoundary pageName="AI Lab">
                    <Suspense fallback={<PageFallback />}><AILab /></Suspense>
                  </PageErrorBoundary>
                </RequireGenerative>
              } />
              <Route path="mix-vault" element={
                <PageErrorBoundary pageName="Mix Vault">
                  <Suspense fallback={<PageFallback />}><MixVault /></Suspense>
                </PageErrorBoundary>
              } />
              <Route path="operations" element={
                <PageErrorBoundary pageName="Operations">
                  <Suspense fallback={<PageFallback />}><Operations /></Suspense>
                </PageErrorBoundary>
              } />
              <Route path="account" element={
                <PageErrorBoundary pageName="Account">
                  <Suspense fallback={<PageFallback />}><AccountPage /></Suspense>
                </PageErrorBoundary>
              } />
              <Route path="*" element={<Navigate to="/mission-control" replace />} />
          </Routes>
        </main>

        {inspectorOpen && <RightInspector />}
      </div>
      <PlayerBar />
      {showShortcuts && <ShortcutsModal onClose={() => setShowShortcuts(false)} />}
    </ToastProvider>
  )
}
