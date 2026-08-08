/* ============================================================
   playerEngine — singleton Web Audio chain for the global player.

   media element → bass (lowshelf 100 Hz) → mid (peaking 1 kHz, Q1)
   → treble (highshelf 8 kHz) → gain → analyser → destination

   Same chain the legacy static UI used, ported as a module. The
   AudioContext is created lazily on the first play() (must happen
   inside a user gesture) and the MediaElementSource is created
   exactly once per page load — Web Audio forbids re-wrapping.
   ============================================================ */

export type EqBand = 'bass' | 'mid' | 'treble'

type Listener = () => void

class PlayerEngine {
  readonly audio: HTMLAudioElement
  private ctx: AudioContext | null = null
  private filters: Record<EqBand, BiquadFilterNode> | null = null
  private gainNode: GainNode | null = null
  analyser: AnalyserNode | null = null

  private timeListeners = new Set<Listener>()
  private endedListeners = new Set<Listener>()

  constructor() {
    this.audio = new Audio()
    this.audio.crossOrigin = 'anonymous'
    this.audio.preload = 'auto'
    this.audio.addEventListener('timeupdate', () => this.timeListeners.forEach((l) => l()))
    this.audio.addEventListener('durationchange', () => this.timeListeners.forEach((l) => l()))
    this.audio.addEventListener('ended', () => this.endedListeners.forEach((l) => l()))
  }

  /** Build the Web Audio graph. Call from a user-gesture context. */
  private ensureGraph() {
    if (this.ctx) {
      if (this.ctx.state === 'suspended') void this.ctx.resume()
      return
    }
    const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
    if (!Ctx) return // jsdom / very old browsers: plain <audio> playback still works
    this.ctx = new Ctx()
    const src = this.ctx.createMediaElementSource(this.audio)
    const bass = this.ctx.createBiquadFilter()
    bass.type = 'lowshelf'
    bass.frequency.value = 100
    const mid = this.ctx.createBiquadFilter()
    mid.type = 'peaking'
    mid.frequency.value = 1000
    mid.Q.value = 1
    const treble = this.ctx.createBiquadFilter()
    treble.type = 'highshelf'
    treble.frequency.value = 8000
    this.gainNode = this.ctx.createGain()
    this.gainNode.gain.value = 0.8
    this.analyser = this.ctx.createAnalyser()
    this.analyser.fftSize = 2048
    this.analyser.smoothingTimeConstant = 0.82
    src.connect(bass).connect(mid).connect(treble).connect(this.gainNode).connect(this.analyser).connect(this.ctx.destination)
    this.filters = { bass, mid, treble }
  }

  async play(url?: string) {
    this.ensureGraph()
    if (url && this.audio.src !== url) this.audio.src = url
    await this.audio.play()
  }

  pause() {
    this.audio.pause()
  }

  seek(seconds: number) {
    if (Number.isFinite(seconds)) this.audio.currentTime = Math.max(0, seconds)
  }

  setEq(band: EqBand, db: number) {
    if (this.filters) this.filters[band].gain.value = db
  }

  setVolume(v: number) {
    const clamped = Math.min(1, Math.max(0, v))
    if (this.gainNode) this.gainNode.gain.value = clamped
    else this.audio.volume = clamped
  }

  setMuted(muted: boolean) {
    this.audio.muted = muted
  }

  get currentTime() {
    return this.audio.currentTime
  }

  get duration() {
    return Number.isFinite(this.audio.duration) ? this.audio.duration : 0
  }

  onTime(l: Listener) {
    this.timeListeners.add(l)
    return () => this.timeListeners.delete(l)
  }

  onEnded(l: Listener) {
    this.endedListeners.add(l)
    return () => this.endedListeners.delete(l)
  }
}

export const playerEngine = new PlayerEngine()
