/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute API base URL (e.g. http://localhost:8000). Empty = use Vite dev proxy at /api. */
  readonly VITE_API_BASE?: string
  /** Absolute SSE stream URL. Empty = /events/stream via Vite dev proxy. */
  readonly VITE_EVENTS_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
