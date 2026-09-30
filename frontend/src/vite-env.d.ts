/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute API base URL (e.g. http://localhost:8000). Empty = use Vite dev proxy at /api. */
  readonly VITE_API_BASE?: string
  /** Absolute SSE stream URL. Empty = /events/stream via Vite dev proxy. */
  readonly VITE_EVENTS_URL?: string
  /** Supabase project URL. Set with VITE_SUPABASE_ANON_KEY to turn on auth; unset = no auth. */
  readonly VITE_SUPABASE_URL?: string
  /** Supabase anon/publishable key. Required alongside VITE_SUPABASE_URL for auth. */
  readonly VITE_SUPABASE_ANON_KEY?: string
  /** Parent domain for the SSE auth cookie (e.g. example.com) so api.<domain> receives it. Unset = host-only cookie. */
  readonly VITE_COOKIE_DOMAIN?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
