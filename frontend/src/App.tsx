import { Suspense, lazy } from 'react'
import { Routes, Route } from 'react-router-dom'
import { AppShell } from './shell/AppShell'

// The widget renders without the shell — it is meant to float beside a DAW
const Widget = lazy(() => import('@/pages/Widget'))

export default function App() {
  return (
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
  )
}
