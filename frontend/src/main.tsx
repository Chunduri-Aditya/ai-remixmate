import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import App from './App'
import { ApiError } from './lib/api'
import './styles/global.css'

const queryClient = new QueryClient({
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

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <QueryClientProvider client={queryClient}>
        <App />
        {import.meta.env.DEV && <ReactQueryDevtools initialIsOpen={false} />}
      </QueryClientProvider>
    </BrowserRouter>
  </StrictMode>,
)
