/* ============================================================
   playerStore — global player state (queue, transport, EQ).

   Thin Zustand layer over lib/playerEngine's singleton audio
   graph. Pages enqueue tracks; the PlayerBar renders/controls.
   Queue semantics ported from the legacy static UI: play a song
   from a list → the rest of the list becomes the queue; shuffle
   picks randomly; repeat loops the current track.
   ============================================================ */
import { create } from 'zustand'
import { playerEngine, EqBand } from '@/lib/playerEngine'
import { audioApi, feedbackApi } from '@/lib/api'
import type { FeedbackEventType } from '@/types'

export interface PlayerTrack {
  title: string
  url: string
  /** Library song name when the track is a library song (enables per-song UI) */
  songName?: string
  // Optional DJ-transition context (populated by Mix Vault when queuing a
  // rendered dj-remix/dj-chain output) so playback events can attribute to
  // a song pair + render job, not just a single song name — SELF_IMPROVING_
  // DJ_RESEARCH.md §3, Stage A.
  songA?: string
  songB?: string
  jobId?: string
}

// Fire-and-forget playback telemetry (SELF_IMPROVING_DJ_RESEARCH.md §3).
// Never awaited by callers and always .catch()'d here so a feedback-pipe
// hiccup can never stall or interrupt actual playback.
function logPlaybackEvent(
  eventType: FeedbackEventType,
  track: PlayerTrack | undefined,
  positionSec?: number,
) {
  const songA = track?.songA ?? track?.songName
  if (!songA) return   // nothing meaningful to attribute this event to (e.g. an ad-hoc URL play)
  feedbackApi
    .logEvent({ eventType, songA, songB: track?.songB, jobId: track?.jobId, positionSec })
    .catch(() => {})
}

interface PlayerState {
  queue: PlayerTrack[]
  index: number
  playing: boolean
  currentTime: number
  duration: number
  volume: number
  muted: boolean
  shuffle: boolean
  repeat: boolean
  eq: Record<EqBand, number>

  current: () => PlayerTrack | null
  playQueue: (tracks: PlayerTrack[], startIndex?: number) => void
  playTrack: (songName: string) => void
  playUrl: (url: string, title: string, meta?: { songA?: string; songB?: string; jobId?: string }) => void
  toggle: () => void
  next: () => void
  prev: () => void
  seek: (seconds: number) => void
  setVolume: (v: number) => void
  toggleMute: () => void
  toggleShuffle: () => void
  toggleRepeat: () => void
  setEq: (band: EqBand, db: number) => void
}

function pickNextIndex(state: { queue: PlayerTrack[]; index: number; shuffle: boolean }): number | null {
  const n = state.queue.length
  if (n === 0) return null
  if (state.shuffle && n > 1) {
    let i = state.index
    while (i === state.index) i = Math.floor(Math.random() * n)
    return i
  }
  return state.index + 1 < n ? state.index + 1 : null
}

export const usePlayerStore = create<PlayerState>((set, get) => {
  // Engine → store sync (module singleton: subscribe once)
  playerEngine.onTime(() => {
    set({ currentTime: playerEngine.currentTime, duration: playerEngine.duration })
  })
  playerEngine.onEnded(() => {
    const s = get()
    if (s.repeat) {
      logPlaybackEvent('replay', s.queue[s.index])
      void playerEngine.play(s.queue[s.index]?.url)
      return
    }
    // The track played all the way through — that's a "complete", the
    // strongest available proxy for "this transition/track worked" (see
    // SELF_IMPROVING_DJ_RESEARCH.md §2's signal inventory), logged BEFORE
    // advancing so it's attributed to the track that just finished, not
    // whatever plays next.
    logPlaybackEvent('complete', s.queue[s.index])
    const nextIdx = pickNextIndex(s)
    if (nextIdx === null) {
      set({ playing: false })
      return
    }
    set({ index: nextIdx, playing: true })
    logPlaybackEvent('play_start', s.queue[nextIdx])
    void playerEngine.play(s.queue[nextIdx].url)
  })

  const startAt = (queue: PlayerTrack[], index: number) => {
    set({ queue, index, playing: true })
    logPlaybackEvent('play_start', queue[index])
    void playerEngine.play(queue[index].url)
  }

  return {
    queue: [],
    index: 0,
    playing: false,
    currentTime: 0,
    duration: 0,
    volume: 0.8,
    muted: false,
    shuffle: false,
    repeat: false,
    eq: { bass: 0, mid: 0, treble: 0 },

    current: () => {
      const s = get()
      return s.queue[s.index] ?? null
    },

    playQueue: (tracks, startIndex = 0) => {
      if (tracks.length === 0) return
      startAt(tracks, Math.min(startIndex, tracks.length - 1))
    },

    playTrack: (songName) => {
      startAt([{ title: songName, url: audioApi.streamUrl(songName), songName }], 0)
    },

    playUrl: (url, title, meta) => {
      startAt([{ title, url, songA: meta?.songA, songB: meta?.songB, jobId: meta?.jobId }], 0)
    },

    toggle: () => {
      const s = get()
      if (!s.queue.length) return
      if (s.playing) {
        playerEngine.pause()
        set({ playing: false })
      } else {
        void playerEngine.play()
        set({ playing: true })
      }
    },

    next: () => {
      const s = get()
      const nextIdx = pickNextIndex(s)
      if (nextIdx === null) return
      // Manual skip-forward — the single most valuable implicit signal per
      // SELF_IMPROVING_DJ_RESEARCH.md §2 ("Skip during playback"). onEnded()
      // above never calls next(), so every call here is genuinely
      // user-initiated, not an auto-advance being mislabeled as a rejection.
      logPlaybackEvent('skip', s.queue[s.index], playerEngine.currentTime)
      set({ index: nextIdx, playing: true })
      logPlaybackEvent('play_start', s.queue[nextIdx])
      void playerEngine.play(s.queue[nextIdx].url)
    },

    prev: () => {
      const s = get()
      // DJ convention: early in the track go to previous, else restart
      if (playerEngine.currentTime > 3 || s.index === 0) {
        // Restart-in-place isn't a rejection of this track (arguably the
        // opposite) — deliberately NOT logged as 'skip' to avoid feeding
        // Stage B's skip-rate analysis an ambiguous signal.
        playerEngine.seek(0)
        return
      }
      logPlaybackEvent('skip', s.queue[s.index], playerEngine.currentTime)
      const prevIdx = s.index - 1
      set({ index: prevIdx, playing: true })
      logPlaybackEvent('play_start', s.queue[prevIdx])
      void playerEngine.play(s.queue[prevIdx].url)
    },

    seek: (seconds) => playerEngine.seek(seconds),

    setVolume: (v) => {
      playerEngine.setVolume(v)
      set({ volume: v, muted: false })
    },

    toggleMute: () => {
      const muted = !get().muted
      playerEngine.setMuted(muted)
      set({ muted })
    },

    toggleShuffle: () => set((s) => ({ shuffle: !s.shuffle })),
    toggleRepeat: () => set((s) => ({ repeat: !s.repeat })),

    setEq: (band, db) => {
      const clamped = Math.min(12, Math.max(-12, db))
      playerEngine.setEq(band, clamped)
      set((s) => ({ eq: { ...s.eq, [band]: clamped } }))
    },
  }
})
