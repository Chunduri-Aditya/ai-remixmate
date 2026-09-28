/**
 * Wire-contract guards for lib/api.ts.
 *
 * smoke.test.ts mocks '@/lib/api' wholesale, so it cannot catch a mismatch
 * between the key the client sends and the key FastAPI reads. That gap let
 * cratesApi.addSong ship sending `{ name }` to POST /crates/{id}/songs, which
 * reads `song_name` — every add-to-crate from Library Atlas 400'd.
 *
 * These tests import the real client and assert the bytes on the wire.
 * Keep them in sync with scripts/api/schemas.py request models.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cratesApi } from '@/lib/api'

let calls: Array<{ url: string; method: string; body: unknown }>

beforeEach(() => {
  calls = []
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    calls.push({
      url,
      method: init?.method ?? 'GET',
      body: init?.body ? JSON.parse(init.body as string) : undefined,
    })
    return {
      ok: true,
      status: 200,
      json: async () => ({}),
      text: async () => '',
    } as unknown as Response
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('cratesApi wire contract', () => {
  it('addSong sends song_name, the key add_to_crate actually reads', async () => {
    await cratesApi.addSong('7', 'Some Track')
    expect(calls).toHaveLength(1)
    expect(calls[0].url).toContain('/crates/7/songs')
    expect(calls[0].method).toBe('POST')
    // The regression: `{ name: ... }` here is a silent 400 in production.
    expect(calls[0].body).toEqual({ song_name: 'Some Track' })
  })

  it('create sends name, the key create_crate reads', async () => {
    await cratesApi.create('My Crate')
    expect(calls[0].body).toEqual({ name: 'My Crate' })
  })

  it('rename sends name, the key rename_crate reads', async () => {
    await cratesApi.rename('3', 'Renamed')
    expect(calls[0].method).toBe('PATCH')
    expect(calls[0].body).toEqual({ name: 'Renamed' })
  })
})
