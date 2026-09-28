/**
 * B5.8 error surface (plans/audit-the-project-and-sorted-flask.md).
 *
 * Contract these tests pin (the builder implements against it, allowlist:
 * App.tsx, main.tsx, components/RootErrorBoundary.tsx):
 *
 *  - `@/components/RootErrorBoundary` exports `RootErrorBoundary`, a boundary
 *    that renders its children and, when a child throws during render, swaps
 *    them for a fallback carrying `role="alert"`.
 *  - `App` is the composition root of the error surface, so that mounting
 *    `<MemoryRouter><App /></MemoryRouter>` alone is enough:
 *      (a) the RootErrorBoundary wraps everything App renders, so a route
 *          element that throws yields the alert fallback;
 *      (b) App owns the QueryClientProvider (moved out of main.tsx together
 *          with AuthProvider, which needs the client) and a ToastProvider above
 *          the routes; the client's QueryCache and MutationCache `onError`
 *          push an error toast into it, except for ApiError 401 (AuthProvider
 *          handles it) and the feature 503s (a disabled feature such as
 *          /ai/models or /stems/split answering 503).
 *    main.tsx keeps BrowserRouter + createRoot only. A client built in main.tsx
 *    is unreachable from a test (main.tsx mounts into #root on import).
 *
 * Seams: `@/shell/AppShell` is mocked with a per test page so the tests
 * control what the routed element does (throw, run a failing query, run a
 * failing mutation). Errors are rejected straight from queryFn/mutationFn as
 * the real `ApiError` class, so no fetch stub is needed and `retry: false`
 * keeps the run under the default 5xx retry delays. Toasts are located by the
 * existing ToastProvider markup (`.toast-area .toast`). The boundary module is
 * loaded with a runtime dynamic import so its absence is an assertion failure
 * rather than a Vite import resolution error for the whole file.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, type ComponentType, type ReactElement, type ReactNode } from 'react'

const shell = vi.hoisted(() => ({
  Page: (() => null) as () => ReactElement | null,
}))

vi.mock('@/shell/AppShell', () => ({
  AppShell: () => {
    const Page = shell.Page
    return <Page />
  },
}))

import App from '@/App'
import { ApiError } from '@/lib/api'

type BoundaryProps = { children: ReactNode }
type BoundaryModule = { RootErrorBoundary?: ComponentType<BoundaryProps> }

async function loadRootErrorBoundary(): Promise<ComponentType<BoundaryProps> | null> {
  // Relative and built at runtime on purpose: a static specifier for a file
  // that does not exist yet fails Vite's import analysis before any test runs.
  const specifier = '../components/' + 'RootErrorBoundary'
  const mod = (await import(/* @vite-ignore */ specifier).catch(() => null)) as BoundaryModule | null
  return mod?.RootErrorBoundary ?? null
}

function Bomb(): ReactElement {
  throw new Error('render exploded')
}

function mountApp(): { error: unknown } {
  try {
    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>,
    )
    return { error: null }
  } catch (error) {
    return { error }
  }
}

function toasts(): Element[] {
  return Array.from(document.querySelectorAll('.toast-area .toast'))
}

async function settle(): Promise<void> {
  // One macrotask so a toast push queued by onError has rendered.
  await act(async () => {
    await new Promise((r) => setTimeout(r, 50))
  })
}

function FailingQuery({ error }: { error: unknown }): ReactElement {
  const q = useQuery({
    queryKey: ['errors-test', String(error)],
    queryFn: () => Promise.reject(error),
    retry: false,
  })
  return <span data-testid="status">{q.status}</span>
}

function FailingMutation({ error }: { error: unknown }): ReactElement {
  const m = useMutation({
    mutationFn: () => Promise.reject(error),
    retry: false,
  })
  const { mutate } = m
  useEffect(() => { mutate() }, [mutate])
  return <span data-testid="status">{m.status}</span>
}

function OkQuery(): ReactElement {
  const q = useQuery({ queryKey: ['errors-test', 'ok'], queryFn: async () => ({ ok: true }) })
  return <span data-testid="status">{q.status}</span>
}

let consoleError: ReturnType<typeof vi.spyOn>

beforeEach(() => {
  shell.Page = () => null
  // React logs every boundary caught error; keep the run readable.
  consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  consoleError.mockRestore()
})

// ---------------------------------------------------------------------------
// RootErrorBoundary on its own
// ---------------------------------------------------------------------------

describe('<RootErrorBoundary>', () => {
  it('renders its children when nothing throws', async () => {
    const RootErrorBoundary = await loadRootErrorBoundary()
    expect(RootErrorBoundary, 'RootErrorBoundary is exported from @/components/RootErrorBoundary').toBeTypeOf('function')
    const Boundary = RootErrorBoundary as ComponentType<BoundaryProps>
    render(
      <Boundary>
        <div>healthy child</div>
      </Boundary>,
    )
    expect(screen.getByText('healthy child')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('a thrown render error shows the fallback and drops the children', async () => {
    const RootErrorBoundary = await loadRootErrorBoundary()
    expect(RootErrorBoundary, 'RootErrorBoundary is exported from @/components/RootErrorBoundary').toBeTypeOf('function')
    const Boundary = RootErrorBoundary as ComponentType<BoundaryProps>
    render(
      <Boundary>
        <div>sibling</div>
        <Bomb />
      </Boundary>,
    )
    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(screen.queryByText('sibling')).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// App as the composition root of the error surface
// ---------------------------------------------------------------------------

describe('<App> error surface', () => {
  it('wraps the whole app: a route element that throws shows the root fallback', () => {
    shell.Page = () => <Bomb />
    const { error } = mountApp()
    expect(error, 'render must not throw past App; the root boundary catches it').toBeNull()
    expect(screen.getByRole('alert')).toBeInTheDocument()
  })

  it('a failed query (ApiError 500) shows a toast', async () => {
    shell.Page = () => <FailingQuery error={new ApiError(500, '/library', 'server exploded')} />
    const { error } = mountApp()
    expect(error, 'App must provide its own QueryClient').toBeNull()
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('error'))
    await waitFor(() => expect(toasts()).toHaveLength(1))
    expect(toasts()[0].textContent?.trim().length).toBeGreaterThan(1)
  })

  it('a failed query with a non API error (network) shows a toast', async () => {
    shell.Page = () => <FailingQuery error={new Error('network down')} />
    const { error } = mountApp()
    expect(error, 'App must provide its own QueryClient').toBeNull()
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('error'))
    await waitFor(() => expect(toasts()).toHaveLength(1))
  })

  it('a failed mutation (ApiError 500) shows a toast', async () => {
    shell.Page = () => <FailingMutation error={new ApiError(500, '/dj-remix', 'render failed')} />
    const { error } = mountApp()
    expect(error, 'App must provide its own QueryClient').toBeNull()
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('error'))
    await waitFor(() => expect(toasts()).toHaveLength(1))
  })

  it('a 401 shows no toast (AuthProvider owns that path)', async () => {
    shell.Page = () => <FailingQuery error={new ApiError(401, '/crates', 'unauthorized')} />
    const { error } = mountApp()
    expect(error, 'App must provide its own QueryClient').toBeNull()
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('error'))
    await settle()
    expect(toasts()).toHaveLength(0)
  })

  it('a feature 503 (query) shows no toast', async () => {
    shell.Page = () => <FailingQuery error={new ApiError(503, '/ai/models', 'generative is disabled')} />
    const { error } = mountApp()
    expect(error, 'App must provide its own QueryClient').toBeNull()
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('error'))
    await settle()
    expect(toasts()).toHaveLength(0)
  })

  it('a feature 503 (mutation) shows no toast', async () => {
    shell.Page = () => <FailingMutation error={new ApiError(503, '/stems/split', 'stem split is disabled')} />
    const { error } = mountApp()
    expect(error, 'App must provide its own QueryClient').toBeNull()
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('error'))
    await settle()
    expect(toasts()).toHaveLength(0)
  })

  it('a successful query shows no toast', async () => {
    shell.Page = () => <OkQuery />
    const { error } = mountApp()
    expect(error, 'App must provide its own QueryClient').toBeNull()
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('success'))
    await settle()
    expect(toasts()).toHaveLength(0)
  })
})
