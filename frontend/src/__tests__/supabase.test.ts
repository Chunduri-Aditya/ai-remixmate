/**
 * Unit tests for src/lib/supabase.ts with NO env configured — the local /
 * test posture. `authConfigured` is false, `supabase` is null, nothing throws,
 * and the SSE cookie helper still round-trips.
 *
 * The B8.5 block below covers VITE_COOKIE_DOMAIN: when set, the cookie string
 * carries `Domain=<value>` so api.<domain> receives it; unset leaves the
 * string as before. jsdom's `document.cookie` getter drops attributes and its
 * jar rejects a Domain that does not match the page host, so those tests read
 * the raw string handed to the setter instead of the jar.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { authConfigured, getAccessToken, supabase, syncSseCookie } from '@/lib/supabase'

describe('supabase.ts (unconfigured)', () => {
  it('is inert without VITE_SUPABASE_* env', () => {
    expect(authConfigured).toBe(false)
    expect(supabase).toBeNull()
  })

  it('getAccessToken resolves to null (no network)', async () => {
    await expect(getAccessToken()).resolves.toBeNull()
  })
})

describe('syncSseCookie', () => {
  afterEach(() => {
    document.cookie = 'sb-access-token=; Path=/; Max-Age=0'
  })

  it('writes the token cookie the SSE stream reads', () => {
    syncSseCookie('jwt-abc-123')
    expect(document.cookie).toContain('sb-access-token=jwt-abc-123')
  })

  it('clears the cookie on sign-out', () => {
    syncSseCookie('jwt-abc-123')
    syncSseCookie(null)
    expect(document.cookie).not.toContain('sb-access-token=jwt-abc-123')
  })
})

// --- B8.5 cookie domain ----------------------------------------------------

const COOKIE_DOMAIN = 'remixmate.example'

/** Capture every raw string assigned to document.cookie; returns a restore fn. */
function captureCookieWrites(writes: string[]): () => void {
  const desc = Object.getOwnPropertyDescriptor(Document.prototype, 'cookie')
  if (!desc?.get || !desc.set) throw new Error('jsdom Document.prototype.cookie descriptor missing')
  const get = desc.get
  const set = desc.set
  Object.defineProperty(document, 'cookie', {
    configurable: true,
    get: () => get.call(document),
    set: (v: string) => {
      writes.push(v)
      set.call(document, v)
    },
  })
  return () => {
    delete (document as unknown as Record<string, unknown>).cookie
  }
}

/** Fresh module instance so a module-scope read of import.meta.env sees the stub. */
async function loadSync(): Promise<typeof syncSseCookie> {
  vi.resetModules()
  const mod = await import('@/lib/supabase')
  return mod.syncSseCookie
}

describe('syncSseCookie with VITE_COOKIE_DOMAIN (B8.5)', () => {
  const writes: string[] = []
  let restore: () => void = () => {}

  afterEach(() => {
    restore()
    vi.unstubAllEnvs()
    vi.resetModules()
    document.cookie = 'sb-access-token=; Path=/; Max-Age=0'
    writes.length = 0
  })

  it('adds Domain=<VITE_COOKIE_DOMAIN> to the token cookie when set', async () => {
    vi.stubEnv('VITE_COOKIE_DOMAIN', COOKIE_DOMAIN)
    const sync = await loadSync()
    restore = captureCookieWrites(writes)
    sync('jwt-abc-123')
    expect(writes).toHaveLength(1)
    const raw = writes[0]
    expect(raw).toContain('sb-access-token=jwt-abc-123')
    expect(raw).toContain(`Domain=${COOKIE_DOMAIN}`)
    expect(raw).toContain('Path=/')
  })

  it('adds the same Domain to the clearing cookie so the domain cookie is actually removed', async () => {
    vi.stubEnv('VITE_COOKIE_DOMAIN', COOKIE_DOMAIN)
    const sync = await loadSync()
    restore = captureCookieWrites(writes)
    sync(null)
    expect(writes).toHaveLength(1)
    const raw = writes[0]
    expect(raw).toContain('sb-access-token=;')
    expect(raw).toContain('Max-Age=0')
    expect(raw).toContain(`Domain=${COOKIE_DOMAIN}`)
  })

  it('leaves the cookie string unchanged (no Domain attribute) when unset', async () => {
    vi.stubEnv('VITE_COOKIE_DOMAIN', '')
    const sync = await loadSync()
    restore = captureCookieWrites(writes)
    sync('jwt-abc-123')
    sync(null)
    expect(writes).toHaveLength(2)
    expect(writes[0]).toBe('sb-access-token=jwt-abc-123; Path=/; SameSite=Lax')
    expect(writes[1]).toBe('sb-access-token=; Path=/; Max-Age=0; SameSite=Lax')
    for (const w of writes) expect(w).not.toMatch(/Domain=/i)
  })
})

// --- security: the https cookie must not ride cross-site requests ----------
// api.<domain> under Domain=<domain> is same-site with the frontend, so Lax
// still reaches the SSE stream and media GETs; None let a cross-site form POST
// carry the JWT to body-less mutating routes (CSRF).

describe('syncSseCookie on https', () => {
  const writes: string[] = []
  let restore: () => void = () => {}
  const dom = (globalThis as unknown as { jsdom: { reconfigure(o: { url: string }): void } }).jsdom
  const originalUrl = window.location.href

  afterEach(() => {
    restore()
    dom.reconfigure({ url: originalUrl })
    vi.unstubAllEnvs()
    vi.resetModules()
    writes.length = 0
  })

  it('writes and clears the token cookie as SameSite=Lax; Secure, never SameSite=None', async () => {
    vi.stubEnv('VITE_COOKIE_DOMAIN', COOKIE_DOMAIN)
    dom.reconfigure({ url: `https://app.${COOKIE_DOMAIN}/` })
    expect(window.location.protocol).toBe('https:')
    const sync = await loadSync()
    restore = captureCookieWrites(writes)
    sync('jwt-abc-123')
    sync(null)
    expect(writes).toEqual([
      `sb-access-token=jwt-abc-123; Path=/; Domain=${COOKIE_DOMAIN}; SameSite=Lax; Secure`,
      `sb-access-token=; Path=/; Max-Age=0; Domain=${COOKIE_DOMAIN}; SameSite=Lax; Secure`,
    ])
  })
})
