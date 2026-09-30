/* ============================================================
   AI RemixMate — Supabase client (Phase 4 auth)

   Auth is opt-in. When VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY are unset
   (local dev against a no-auth backend, and the test suite), `supabase` is
   null and `authConfigured` is false — the app renders exactly as before and
   every request goes out unauthenticated (the backend resolves it to "local").

   When both are set, we use supabase-js with its default localStorage session
   store, attach `Authorization: Bearer <access_token>` to every fetch, and
   mirror the token into a readable `sb-access-token` cookie so the SSE
   EventSource (which cannot send headers) authenticates too. The cookie is
   NOT httpOnly — a true httpOnly session cookie needs an SSR/edge proxy to
   set it; deferred. The token also lives in localStorage (supabase-js
   default), so httpOnly would add little here.
   ============================================================ */

import { createClient, type SupabaseClient } from '@supabase/supabase-js'

const URL = import.meta.env.VITE_SUPABASE_URL?.trim()
const ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY?.trim()

export const authConfigured: boolean = Boolean(URL && ANON_KEY)

export const supabase: SupabaseClient | null = authConfigured
  ? createClient(URL as string, ANON_KEY as string, {
      auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
    })
  : null

/** Access token for the current session, or null when unauthenticated / auth off. */
export async function getAccessToken(): Promise<string | null> {
  if (!supabase) return null
  const { data } = await supabase.auth.getSession()
  return data.session?.access_token ?? null
}

// --- SSE cookie mirror ----------------------------------------------------

const COOKIE = 'sb-access-token' // must match scripts/api/auth.py _DEFAULT_COOKIE_NAME
const COOKIE_DOMAIN = import.meta.env.VITE_COOKIE_DOMAIN?.trim()

/** Write (or clear) the readable cookie the SSE stream authenticates with.
 *
 * Prod topology is a Cloudflare-Pages frontend calling an API on another
 * domain, so the cookie must be `SameSite=None; Secure` to ride a cross-site
 * `EventSource(withCredentials)` request. On http localhost (same-origin via
 * the Vite proxy) `SameSite=None` is invalid without Secure, so fall back to
 * `Lax` there. When VITE_COOKIE_DOMAIN is set, `Domain=<value>` is added to
 * both the write and the clear so api.<domain> receives (and loses) it. */
export function syncSseCookie(accessToken: string | null): void {
  if (typeof document === 'undefined') return
  const domain = COOKIE_DOMAIN ? `; Domain=${COOKIE_DOMAIN}` : ''
  const attrs =
    domain +
    (window.location.protocol === 'https:' ? '; SameSite=None; Secure' : '; SameSite=Lax')
  if (accessToken) {
    // Session cookie (no Max-Age) — gone when the tab closes, like the SSE stream.
    document.cookie = `${COOKIE}=${accessToken}; Path=/${attrs}`
  } else {
    document.cookie = `${COOKIE}=; Path=/; Max-Age=0${attrs}`
  }
}
