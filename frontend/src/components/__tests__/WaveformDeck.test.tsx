/**
 * WaveformDeck teardown while wavesurfer's fetch is still in flight.
 *
 * wavesurfer v7 `destroy()` aborts its fetch AbortController; the pending
 * `fetch` rejects with the signal's AbortError, `load()` re-emits it as an
 * 'error' event and rethrows, so the promise `ws.load(src)` returned rejects.
 * Nothing awaited it, so Mix Deck and Signal Search logged an uncaught
 * "AbortError: signal is aborted without reason" on every unmount or source
 * change mid load, and the stale 'error' listener flipped the next source's
 * deck into the errored state.
 *
 * The real wavesurfer runs here over a `fetch` stub that never resolves and
 * rejects with the signal's reason on abort, as a browser does.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { WaveformDeck } from '@/components/WaveformDeck'

function pendingFetch(): typeof fetch {
  return ((_input: RequestInfo | URL, init?: RequestInit) =>
    new Promise<Response>((_resolve, reject) => {
      const signal = init?.signal
      if (!signal) return
      const fail = () =>
        reject(signal.reason ?? new DOMException('signal is aborted without reason', 'AbortError'))
      if (signal.aborted) fail()
      else signal.addEventListener('abort', fail, { once: true })
    })) as typeof fetch
}

// Let the aborted fetch's rejection travel through wavesurfer's async load()
// and reach Node's unhandledRejection check (emitted after the microtask queue).
async function settle() {
  for (let i = 0; i < 3; i++) await new Promise((r) => setTimeout(r, 0))
}

// The tsconfig carries no node types; only the two listener methods are needed.
interface NodeProcessLike {
  on(event: 'unhandledRejection', listener: (reason: unknown) => void): unknown
  off(event: 'unhandledRejection', listener: (reason: unknown) => void): unknown
}
const proc = (globalThis as unknown as { process: NodeProcessLike }).process

let unhandled: unknown[] = []
const onUnhandled = (reason: unknown) => { unhandled.push(reason) }

beforeEach(() => {
  unhandled = []
  proc.on('unhandledRejection', onUnhandled)
  vi.stubGlobal('fetch', pendingFetch())
})

afterEach(() => {
  proc.off('unhandledRejection', onUnhandled)
  cleanup()
  vi.unstubAllGlobals()
})

describe('WaveformDeck', () => {
  it('unmounting while the waveform fetch is in flight throws nothing and leaves no unhandled rejection', async () => {
    const { unmount } = render(<WaveformDeck src="/api/library/Song%20A/audio" color="#f59e0b" />)
    await settle()

    expect(() => unmount()).not.toThrow()
    await settle()

    expect(unhandled.map((e) => String(e))).toEqual([])
  })

  it('changing src while the previous fetch is in flight keeps the new deck out of the errored state', async () => {
    const { rerender } = render(<WaveformDeck src="/api/library/Song%20A/audio" color="#f59e0b" />)
    await settle()

    rerender(<WaveformDeck src="/api/library/Song%20B/audio" color="#f59e0b" />)
    await settle()

    // The play button is hidden only when the deck is errored; it stays (disabled) while loading.
    expect(screen.getByTitle('Play preview')).toBeInTheDocument()
    expect(screen.getByText('Loading waveform…')).toBeInTheDocument()
    expect(unhandled.map((e) => String(e))).toEqual([])
  })
})
