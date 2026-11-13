/* ============================================================
   Remix parameter shape + defaults.

   Split out of RemixControls.tsx so that file exports only the
   RemixControls component — react-refresh/only-export-components
   needs component files to export components and nothing else.
   ============================================================ */

export interface RemixOptions {
  // 8..64 bars (the API range); the picker offers 8/16/32, MixDeck's slider the rest.
  transition_bars: number
  preset: string
  transition_effect: string
  bridge_beat_mode: 'none' | 'auto'
  bridge_beat_genre: string
  bridge_beat_intensity: number
}

export const REMIX_DEFAULTS: RemixOptions = {
  transition_bars: 16,
  preset: 'auto',
  transition_effect: 'auto',
  bridge_beat_mode: 'none',
  bridge_beat_genre: 'auto',
  bridge_beat_intensity: 0.38,
}
