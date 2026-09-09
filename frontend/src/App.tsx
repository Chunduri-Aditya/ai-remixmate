import { Suspense, lazy } from 'react'
import { Routes, Route } from 'react-router-dom'
import { AppShell } from './shell/AppShell'
import { RequireAuth } from './shell/RequireAuth'

// The widget renders without the shell — it is meant to float beside a DAW
const Widget = lazy(() => import('@/pages/Widget'))
const LegalPage = lazy(() => import('@/pages/LegalPage'))

export default function App() {
  return (
    <Routes>
      <Route
        path="/legal/:page"
        element={
          <Suspense fallback={null}>
            <LegalPage />
          </Suspense>
        }
      />
      <Route
        path="/legal"
        element={
          <Suspense fallback={null}>
            <LegalPage />
          </Suspense>
        }
      />
      <Route
        path="*"
        element={
          <RequireAuth>
            <Routes>
              <Route
                path="/widget"
                element={
                  <Suspense fallback={null}>
                    <Widget />
                  </Suspense>
                }
              />
              <Route path="*" element={<AppShell />} />
            </Routes>
          </RequireAuth>
        }
      />
    </Routes>
  )
}
