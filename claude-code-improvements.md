# AI RemixMate — Improvement Prompts for Claude Code

## Project Overview (read CLAUDE.md first — it has full architecture context)

You are working on **AI RemixMate** — a real-time DJ engine with a FastAPI backend
(`scripts/api/`) and React/TypeScript frontend (`frontend/src/`). All 8 pages are
"done" in the baseline sense. These improvements add high-impact features to existing pages.

---

## Improvement 1: Waveform Visualization in Mix Deck

**Goal:** Add interactive waveform display to the Mix Deck using WaveSurfer.js.
Each DeckCard should show the track waveform, and after the compatibility check
runs, an overlay should mark the transition region (entry/exit bars).

### What to do

1. **Install WaveSurfer.js:**
   ```
   cd frontend && npm install wavesurfer.js
   ```

2. **Create `frontend/src/components/WaveformDeck.tsx`:**
   - Props: `src: string` (URL to the audio file), `color: string` (deck accent),
     `cueStart?: number`, `cueEnd?: number` (both in seconds, optional)
   - Use `useRef` + `useEffect` to init WaveSurfer on mount, destroy on unmount
   - Show a play/pause toggle button below the waveform
   - If `cueStart`/`cueEnd` are provided, add WaveSurfer `addRegion` markers
     colored with 30% opacity of the deck accent color
   - Match the dark theme: `waveColor: 'var(--color-ice-400)'` for deck B,
     `'var(--color-amber-500)'` for deck A; `backgroundColor: 'transparent'`;
     `height: 80`

3. **Wire it into `frontend/src/pages/MixDeck.tsx`:**
   - After a song is selected and `songInfo` loads, derive the audio URL via
     `/library/{name}/audio` (add this endpoint if it doesn't exist — see below)
   - Render `<WaveformDeck>` inside `DeckCard`, below the BPM/key meta rows
   - When compatibility result loads (`compatResult`), extract
     `cueStart = compatResult.transition_plan.exit_time_a` and
     `cueEnd = compatResult.transition_plan.entry_time_b` and pass to both waveforms

4. **Backend — add audio serve endpoint** (only if missing):
   Check `scripts/api/routers/library.py`. If there's no route like
   `GET /library/{name}/audio` that streams the actual audio file, add one:
   ```python
   @router.get("/{name}/audio")
   async def stream_audio(name: str):
       from fastapi.responses import FileResponse
       from scripts.core.library import get_song_path
       path = get_song_path(name)
       return FileResponse(path, media_type="audio/mpeg")
   ```
   Also add `audioApi.streamUrl(name: string)` to `frontend/src/lib/api.ts`.

5. **Style in `MixDeck.css`:** add `.md-waveform { margin-top: 12px; border-radius: 6px; overflow: hidden; }` and a `.md-waveform-btn` for the play/pause toggle.

---

## Improvement 2: Transition Timeline Strip

**Goal:** After compatibility check, show a horizontal visual timeline strip on
the Mix Deck page showing the structure of the planned transition:
`[Song A body] [crossfade zone] [Song B body]`

### What to do

1. **Create `frontend/src/components/TransitionTimeline.tsx`:**
   - Props: `plan: TransitionPlan` (use the existing TypeScript type from `types/index.ts`),
     `durationA: number`, `durationB: number` (total track durations in seconds)
   - Render a single flex row divided proportionally:
     - Left zone (amber, Song A pre-mix portion)
     - Center overlap zone (gradient amber→ice, labeled with `{plan.transition_bars} bars`)
     - Right zone (ice, Song B post-entry portion)
   - Show BPM of each song and the stretch ratio below the strip
   - Show a small "🔑 key match" badge if `plan.key_compatible === true`

2. **Add it to `MixDeck.tsx`** between the two DeckCards and the action buttons,
   rendered only when `compatResult` exists.

3. **Check `types/index.ts`** to see if `TransitionPlan` is already typed.
   If not, add:
   ```typescript
   export interface TransitionPlan {
     exit_bar_a: number
     entry_bar_b: number
     transition_bars: number
     stretch_ratio: number
     exit_time_a?: number
     entry_time_b?: number
     key_compatible?: boolean
   }
   ```
   And ensure `CompatibilityResult` includes `transition_plan?: TransitionPlan`.

4. **Style:** Self-contained CSS in the component file. Use CSS variables from
   `tokens.css`. The strip should be ~48px tall, full-width, with rounded corners.

---

## Improvement 3: Job ETA + Elapsed Timer in Inspector Panel

**Goal:** Each running job in the right inspector panel should show
"⏱ 23s elapsed · ~45s remaining" instead of just a progress bar.

### What to do

1. **In `frontend/src/stores/appStore.ts`**, the `Job` type already has
   `created_at` (ISO string). Add a helper hook:
   `frontend/src/hooks/useJobTimer.ts`:
   ```typescript
   import { useEffect, useState } from 'react'
   import type { Job } from '@/types'

   export function useJobTimer(job: Job) {
     const [elapsed, setElapsed] = useState(0)
     useEffect(() => {
       const start = new Date(job.created_at).getTime()
       const tick = () => setElapsed(Math.floor((Date.now() - start) / 1000))
       tick()
       const id = setInterval(tick, 1000)
       return () => clearInterval(id)
     }, [job.created_at])

     const eta = job.progress > 2
       ? Math.round((elapsed / job.progress) * (100 - job.progress))
       : null

     return { elapsed, eta }
   }
   ```

2. **Find the job card component** in `frontend/src/shell/` (likely `RightInspector.tsx`
   or inline in `AppShell.tsx`). In each running job row, call `useJobTimer(job)` and
   render:
   ```
   ⏱ {elapsed}s elapsed{eta ? ` · ~${eta}s left` : ''}
   ```
   below the progress bar. Only show for jobs with status `RUNNING`.

---

## Improvement 4: Stems Mini-Player in Library Atlas

**Goal:** In the Library Atlas track table, add an expandable stems row.
When a user clicks "Stems" on a track that has stems processed, show
4 small playback bars (drums / bass / vocals / other) each with a mute toggle.

### What to do

1. **Backend check:** Look at `scripts/api/routers/stems.py` and `library.py` to
   see if there's a `GET /stems/{name}` or `GET /library/{name}/stems` endpoint
   that returns stem file paths. If not, add one:
   ```python
   @router.get("/{name}/stems")
   async def list_stems(name: str):
       from scripts.core.paths import STEMS_DIR
       import os
       stem_dir = STEMS_DIR / name
       if not stem_dir.exists():
           return {"stems": []}
       stems = [f for f in os.listdir(stem_dir) if f.endswith(".mp3") or f.endswith(".wav")]
       return {"stems": stems}
   ```

2. **In `LibraryAtlas.tsx`**, add an expandable row per track:
   - A small "♪ Stems" chip button (only shown if `songInfo.has_stems` is true or
     if stems loaded non-empty from the above endpoint)
   - When clicked, expand inline below that row to show `<StemsPlayerRow name={name} />`

3. **Create `frontend/src/components/StemsPlayerRow.tsx`:**
   - Fetches `GET /stems/{name}` (add to `api.ts` as `stemApi.list(name)`)
   - Renders one row per stem: icon + label + HTML `<audio>` element styled as a
     minimal progress bar, plus a mute toggle (🔊 / 🔇)
   - Implements a simple "solo" concept: clicking one stem's mute toggles others

4. **Add to `LibraryAtlas.css`** the expansion animation: `max-height` transition
   from 0 → auto via a CSS custom property trick.

---

## Notes for Claude Code

- **No Tailwind** — this project uses vanilla CSS with design tokens from `tokens.css`
- **No new pages** — all improvements are additions to existing pages/components
- **Respect existing patterns** — new components go in `frontend/src/components/`,
  new hooks in `frontend/src/hooks/`, new API calls added to `frontend/src/lib/api.ts`
- **Python style** — async FastAPI routes with `job_store` for anything long-running;
  simple `GET` endpoints for data reads don't need jobs
- **Test the audio endpoint** before wiring up the waveform — verify the file path
  resolves correctly from `scripts/core/library.py` or `paths.py`
- **Priority order:** Improvement 1 (waveform) has the highest visual impact for
  portfolio purposes. Do it first, then 2, then 3, then 4.
