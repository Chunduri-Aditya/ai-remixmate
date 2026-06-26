# AI RemixMate — Batch 2 Improvements for Claude Code

## Context

Read `CLAUDE.md` first. Batch 1 already added: WaveSurfer waveforms in MixDeck,
TransitionTimeline strip, useJobTimer hook, and stems mini-player in LibraryAtlas.
WaveSurfer.js is already installed.

This batch exposes hidden backend controls, upgrades the Mix Vault player, and adds
retry logic for failed jobs.

---

## Improvement 1: MixDeck — Remix Controls Panel

**Goal:** The backend `DJRemixRequest` supports `transition_bars` (8/16/32),
`preset` (auto/techno/house/hiphop/trap/dnb/ambient), `transition_effect`
(auto/echo/filter/reverb/none), and bridge beat controls — but the UI submits
all defaults. Expose these before the user hits "Remix".

### What to do

1. **Create `frontend/src/components/RemixControls.tsx`:**
   ```typescript
   export interface RemixOptions {
     transition_bars: 8 | 16 | 32
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
   ```
   Props: `value: RemixOptions`, `onChange: (v: RemixOptions) => void`

   Layout — a compact 2-column grid of controls (all on one card, no accordion needed):
   - **Transition bars** — segmented button row: `8` | `16` | `32` (amber highlight on active)
   - **Preset** — `<select>`: Auto / Techno / House / Hip-Hop / Trap / DnB / Ambient
   - **Transition effect** — `<select>`: Auto / Echo / Filter / Reverb / None
   - **Bridge beat** — toggle switch ("off" / "on"); when on, show a genre select
     (same options as Preset) + an `<input type="range" min="0" max="1" step="0.05">` for intensity
   
   Style it as a `remix-controls` card using existing token variables — no new CSS file needed,
   inline the `<style>` block inside the component or add to `MixDeck.css`.

2. **In `MixDeck.tsx`:**
   - Add `const [remixOpts, setRemixOpts] = useState<RemixOptions>(REMIX_DEFAULTS)`
   - Render `<RemixControls value={remixOpts} onChange={setRemixOpts} />` between the
     TransitionTimeline and the action buttons
   - In `handleRemix` / `handlePreview`, spread `remixOpts` into the API call:
     ```typescript
     await remixApi.remix({
       song_a: songA,
       song_b: songB,
       ...remixOpts,
     })
     ```
   - Check `frontend/src/lib/api.ts` — `remixApi.remix()` already accepts a
     `DJRemixRequest`-shaped object; just make sure all fields pass through.
   - Do the same for `remixApi.preview()` — it takes `DJPreviewRequest` which has
     `transition_bars` and `preset` at minimum.

---

## Improvement 2: Set Builder — Chain Settings Accordion

**Goal:** SetBuilder calls `remixApi.chain(songs)` with no options. Expose the same
`RemixOptions` (reuse the component from Improvement 1) in a collapsible settings panel.

### What to do

1. **In `SetBuilder.tsx`:**
   - Import `RemixControls`, `RemixOptions`, `REMIX_DEFAULTS` from the component
     created in Improvement 1
   - Add state: `const [chainOpts, setChainOpts] = useState<RemixOptions>(REMIX_DEFAULTS)`
   - Add state: `const [showSettings, setShowSettings] = useState(false)`
   - In the header actions row (where Export / Shuffle buttons live), add a
     `⚙ Settings` ghost button that toggles `showSettings`
   - Below the header, conditionally render:
     ```tsx
     {showSettings && (
       <div className="sb-settings">
         <RemixControls value={chainOpts} onChange={setChainOpts} />
       </div>
     )}
     ```
   - In `handleChain`, pass opts:
     ```typescript
     await remixApi.chain(set.map((e) => e.name), {
       transition_bars: chainOpts.transition_bars,
       preset: chainOpts.preset,
       transition_effect: chainOpts.transition_effect,
       bridge_beat_mode: chainOpts.bridge_beat_mode,
       bridge_beat_genre: chainOpts.bridge_beat_genre,
       bridge_beat_intensity: chainOpts.bridge_beat_intensity,
     })
     ```

2. **In `SetBuilder.css`**, add:
   ```css
   .sb-settings {
     padding: 0 var(--space-4) var(--space-3);
     border-bottom: 1px solid var(--color-border);
   }
   ```

---

## Improvement 3: Mix Vault — WaveSurfer Waveform Player

**Goal:** Replace the custom `AudioPlayer` in `MixVault.tsx` with a proper
WaveSurfer waveform (WaveSurfer is already installed from Batch 1).

### What to do

1. **In `MixVault.tsx`**, replace the `AudioPlayer` component entirely with a
   new inline `WaveformPlayer` component (keep it local to this file, ~60 lines):

   ```typescript
   import WaveSurfer from 'wavesurfer.js'

   function WaveformPlayer({ src }: { src: string }) {
     const containerRef = useRef<HTMLDivElement>(null)
     const wsRef        = useRef<WaveSurfer | null>(null)
     const [playing, setPlaying]   = useState(false)
     const [progress, setProgress] = useState(0)
     const [duration, setDuration] = useState(0)
     const [volume, setVolume]     = useState(1)

     useEffect(() => {
       if (!containerRef.current) return
       const ws = WaveSurfer.create({
         container:   containerRef.current,
         waveColor:   'var(--color-violet-400)',
         progressColor: 'var(--color-amber-500)',
         height:      64,
         barWidth:    2,
         barGap:      1,
         barRadius:   2,
         backend:     'WebAudio',
         url:         src,
       })
       ws.on('ready',      () => setDuration(ws.getDuration()))
       ws.on('audioprocess', () => setProgress(ws.getCurrentTime() / ws.getDuration()))
       ws.on('finish',     () => setPlaying(false))
       wsRef.current = ws
       return () => ws.destroy()
     }, [src])

     function togglePlay() {
       wsRef.current?.playPause()
       setPlaying((p) => !p)
     }
     function onVolumeChange(e: React.ChangeEvent<HTMLInputElement>) {
       const v = parseFloat(e.target.value)
       setVolume(v)
       wsRef.current?.setVolume(v)
     }

     return (
       <div className="mv-waveform-player">
         <div ref={containerRef} className="mv-waveform-canvas" />
         <div className="mv-waveform-controls">
           <button className="mv-icon-btn" onClick={togglePlay}>
             {playing ? <Pause size={16} /> : <Play size={16} />}
           </button>
           <span className="text-muted font-mono" style={{ fontSize: 'var(--text-xs)' }}>
             {formatDuration(progress * duration)} / {formatDuration(duration)}
           </span>
           <Volume2 size={12} style={{ marginLeft: 'auto', color: 'var(--color-text-secondary)' }} />
           <input
             type="range" min="0" max="1" step="0.05"
             value={volume} onChange={onVolumeChange}
             className="mv-volume"
           />
         </div>
       </div>
     )
   }
   ```

2. Where `AudioPlayer` was rendered, render `WaveformPlayer` instead.
   The `src` is already computed as `streamUrl` in the existing `MixCard` component
   (`${BASE}${result.stream_url ?? result.output_url}`).

3. **In `MixVault.css`**, add:
   ```css
   .mv-waveform-player  { display: flex; flex-direction: column; gap: var(--space-2); }
   .mv-waveform-canvas  { border-radius: var(--radius-sm); overflow: hidden; }
   .mv-waveform-controls {
     display: flex; align-items: center; gap: var(--space-2);
   }
   .mv-volume { width: 64px; accent-color: var(--color-amber-500); }
   ```

---

## Improvement 4: Inspector — Retry Button for Failed Jobs

**Goal:** When a job has status `FAILED`, show a Retry button in its card that
re-submits the same job type with the same parameters stored in `job.meta`.

### What to do

1. **In `RightInspector.tsx`**, update `JobCard` to handle retries.

   The `job.meta` shape (set at job creation time) is:
   - `dj_remix`:  `{ song_a: string, song_b: string }`
   - `dj_chain`:  `{ songs: string[] }`
   - `download`:  `{ url: string }` or `{ name: string }`
   - `stems`:     `{ song: string }`
   - `analysis`:  `{ song: string }`

   Add this function inside `JobCard`:
   ```typescript
   async function handleRetry() {
     const meta = job.meta as Record<string, unknown> ?? {}
     const upsertJob = useAppStore.getState().upsertJob
     try {
       let res: { job_id: string }
       if (job.type === 'dj_remix' && meta.song_a && meta.song_b) {
         res = await remixApi.remix({ song_a: meta.song_a as string, song_b: meta.song_b as string })
       } else if (job.type === 'dj_chain' && Array.isArray(meta.songs)) {
         res = await remixApi.chain(meta.songs as string[])
       } else {
         return  // can't auto-retry other job types
       }
       upsertJob({
         job_id: res.job_id, status: 'PENDING', type: job.type,
         progress: 0, message: 'Retrying…',
         created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
       })
     } catch {
       // silently ignore — user sees the existing error card
     }
   }
   ```

   **Note:** `useAppStore.getState()` is the Zustand escape hatch for calling
   store actions inside a non-hook callback. This is correct here.

2. Add a Retry button to the card footer for FAILED jobs:
   ```tsx
   {job.status === 'FAILED' && (job.type === 'dj_remix' || job.type === 'dj_chain') && (
     <button className="inspector-job-card__retry" onClick={handleRetry}>
       ↺ Retry
     </button>
   )}
   ```

3. Import `remixApi` at the top of `RightInspector.tsx`.

4. **In `RightInspector.css`**, add:
   ```css
   .inspector-job-card__retry {
     margin-top: var(--space-1);
     padding: 2px 8px;
     border: 1px solid var(--color-amber-500);
     border-radius: var(--radius-sm);
     background: transparent;
     color: var(--color-amber-500);
     font-size: var(--text-xs);
     cursor: pointer;
   }
   .inspector-job-card__retry:hover {
     background: color-mix(in srgb, var(--color-amber-500) 15%, transparent);
   }
   ```

---

## Notes for Claude Code

- `RemixControls` from Improvement 1 is reused as-is in Improvement 2 — build 1 first
- MixVault's `formatDuration` already exists in the file — don't duplicate it
- `remixApi.chain` signature is `(songs: string[], opts?: Record<string, unknown>)` —
  confirmed in `api.ts`; the opts spread directly into the POST body
- No new routes needed — all backend endpoints already exist
- Run `tsc --noEmit` after each improvement to catch any type errors before moving on
- **Priority order:** 1 → 2 (shares component) → 3 → 4
