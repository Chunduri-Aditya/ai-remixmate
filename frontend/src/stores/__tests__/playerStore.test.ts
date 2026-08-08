/* playerStore — queue/transport semantics (engine mocked). */
import { beforeEach, describe, expect, it, vi } from 'vitest'

const engineMock = vi.hoisted(() => ({
  play: vi.fn().mockResolvedValue(undefined),
  pause: vi.fn(),
  seek: vi.fn(),
  setEq: vi.fn(),
  setVolume: vi.fn(),
  setMuted: vi.fn(),
  onTime: vi.fn(() => () => {}),
  onEnded: vi.fn(() => () => {}),
  currentTime: 0,
  duration: 0,
  analyser: null,
}))

vi.mock('@/lib/playerEngine', () => ({ playerEngine: engineMock }))

import { usePlayerStore } from '@/stores/playerStore'

const tracks = [
  { title: 'One', url: 'u1' },
  { title: 'Two', url: 'u2' },
  { title: 'Three', url: 'u3' },
]

describe('playerStore', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    usePlayerStore.setState({
      queue: [], index: 0, playing: false, shuffle: false, repeat: false,
      eq: { bass: 0, mid: 0, treble: 0 }, volume: 0.8, muted: false,
    })
  })

  it('playQueue starts at the requested index', () => {
    usePlayerStore.getState().playQueue(tracks, 1)
    const s = usePlayerStore.getState()
    expect(s.index).toBe(1)
    expect(s.playing).toBe(true)
    expect(engineMock.play).toHaveBeenCalledWith('u2')
  })

  it('next advances and stops at queue end', () => {
    usePlayerStore.getState().playQueue(tracks, 1)
    usePlayerStore.getState().next()
    expect(usePlayerStore.getState().index).toBe(2)
    usePlayerStore.getState().next() // at end — no wrap
    expect(usePlayerStore.getState().index).toBe(2)
  })

  it('shuffle next never repeats the current index', () => {
    usePlayerStore.getState().playQueue(tracks, 0)
    usePlayerStore.getState().toggleShuffle()
    for (let i = 0; i < 10; i++) {
      const before = usePlayerStore.getState().index
      usePlayerStore.getState().next()
      expect(usePlayerStore.getState().index).not.toBe(before)
    }
  })

  it('prev restarts the track when >3s in, else goes back', () => {
    usePlayerStore.getState().playQueue(tracks, 1)
    engineMock.currentTime = 10
    usePlayerStore.getState().prev()
    expect(engineMock.seek).toHaveBeenCalledWith(0)
    expect(usePlayerStore.getState().index).toBe(1)

    engineMock.currentTime = 1
    usePlayerStore.getState().prev()
    expect(usePlayerStore.getState().index).toBe(0)
  })

  it('toggle pauses and resumes', () => {
    usePlayerStore.getState().playQueue(tracks, 0)
    usePlayerStore.getState().toggle()
    expect(engineMock.pause).toHaveBeenCalled()
    expect(usePlayerStore.getState().playing).toBe(false)
    usePlayerStore.getState().toggle()
    expect(usePlayerStore.getState().playing).toBe(true)
  })

  it('setEq clamps to ±12 dB and reaches the engine', () => {
    usePlayerStore.getState().setEq('bass', 40)
    expect(usePlayerStore.getState().eq.bass).toBe(12)
    expect(engineMock.setEq).toHaveBeenCalledWith('bass', 12)
    usePlayerStore.getState().setEq('treble', -40)
    expect(usePlayerStore.getState().eq.treble).toBe(-12)
  })

  it('toggle with empty queue is a no-op', () => {
    usePlayerStore.getState().toggle()
    expect(usePlayerStore.getState().playing).toBe(false)
    expect(engineMock.play).not.toHaveBeenCalled()
  })
})
