import { Suspense, lazy, useEffect, useRef, useState, type ReactNode } from 'react'
import { Routes, Route } from 'react-router-dom'
import { MutationCache, QueryCache, QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { AppShell } from './shell/AppShell'
import { RequireAuth } from './shell/RequireAuth'
import { AuthProvider } from './shell/AuthProvider'
import { RootErrorBoundary } from './components/RootErrorBoundary'
import { ToastProvider } from './components/Toast'
import { useToast } from './components/toastContext'
import { ApiError } from './lib/api'
import { supabase } from './lib/supabase'

// The widget renders without the shell — it is meant to float beside a DAW
const Widget = lazy(() => import('@/pages/Widget'))
const LegalPage = lazy(() => import('@/pages/LegalPage'))

// 401 is AuthProvider's to handle; every backend 503 is a disabled feature
// (stem split, generative, wordplay) the pages already render as unavailable.
function isSilentError(error: unknown): boolean {
  return error instanceof ApiError && (error.status === 401 || error.status === 503)
}

function createQueryClient(onError: (error: unknown) => void): QueryClient {
  return new QueryClient({
    queryCache: new QueryCache({ onError }),
    mutationCache: new MutationCache({ onError }),
    defaultOptions: {
      queries: {
        staleTime: 15_000,           // 15 seconds
        // Only retry network failures / 5xx — never 4xx. A 400/404 means the
        // request itself is invalid (e.g. a song name the backend rejects),
        // and blindly retrying it twice per query, on every component that
        // queries it, on every poll interval, turns one bad request into a
        // permanent flood — which is exactly what surfaced as repeated
        // identical 400s for one song in the API logs. 4xx won't succeed on
        // retry; only 5xx/network errors are worth one retry.
        retry: (failureCount, error) => {
          if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
            return false
          }
          return failureCount < 2
        },
        refetchOnWindowFocus: false,
      },
    },
  })
}

// Owns the QueryClient so its cache level onError can push into the toast area above it.
function QueryLayer({ children }: { children: ReactNode }) {
  const { push } = useToast()
  const pushRef = useRef(push)
  useEffect(() => { pushRef.current = push }, [push])
  const [queryClient] = useState(() =>
    createQueryClient((error) => {
      if (isSilentError(error)) return
      const message = error instanceof Error && error.message ? error.message : 'Request failed'
      pushRef.current({ level: 'error', message })
    }),
  )
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
}

// The session owner mounts only when there is a client to own. Without one
// AuthProvider renders the same inert values AuthContext defaults to, and an
// AuthContext supplied above App (a test harness) stays visible to the routes.
function AuthLayer({ children }: { children: ReactNode }) {
  if (!supabase) return <>{children}</>
  return <AuthProvider>{children}</AuthProvider>
}

export default function App() {
  return (
    <RootErrorBoundary>
      <ToastProvider>
        <QueryLayer>
          <AuthLayer>
            <AppRoutes />
            {import.meta.env.DEV && <ReactQueryDevtools initialIsOpen={false} />}
          </AuthLayer>
        </QueryLayer>
      </ToastProvider>
    </RootErrorBoundary>
  )
}

function AppRoutes() {
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
