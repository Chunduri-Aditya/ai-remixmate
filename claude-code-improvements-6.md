# AI RemixMate — Batch 6 Improvements for Claude Code

## Context

Read `CLAUDE.md` first. Batches 1–5 complete. Existing components to reuse:
`CamelotWheel` (from Batch 3), `useJobTimer` (Batch 1), `cratesApi` (already in `api.ts`).
AppShell uses `<Suspense fallback={<PageFallback />}>` but has no error boundary.

---

## Improvement 1: Crates Panel in LibraryAtlas Drawer

**Goal:** The backend has full crates CRUD and `cratesApi` covers all methods.
The LibraryAtlas track drawer already shows metadata, deck buttons, and recommendations.
Add a fourth "Crates" section to that drawer so users can manage crate membership inline.

### What to do

1. **Read `LibraryAtlas.tsx`** — find the `TrackDrawer` component and its existing
   `la-drawer__section` blocks (metadata, deck load, recommendations).

2. **Add a `CratesSection` component** inside `LibraryAtlas.tsx`:

   ```typescript
   function CratesSection({ song }: { song: SongInfo }) {
     const queryClient = useQueryClient()
     const { data: allCrates = [] } = useQuery({
       queryKey: ['crates'],
       queryFn: cratesApi.list,
       staleTime: 30_000,
     })
     const { data: songCrates = [] } = useQuery({
       queryKey: ['crate-songs', song.name],
       queryFn: async () => {
         // fetch all crates and check which contain this song
         const results = await Promise.all(
           allCrates.map((c) => cratesApi.songs(c.id).then((songs) => ({ id: c.id, has: songs.includes(song.name) })))
         )
         return results.filter((r) => r.has).map((r) => r.id)
       },
       enabled: allCrates.length > 0,
     })

     const [newName, setNewName] = useState('')
     const [creating, setCreating] = useState(false)

     async function toggleCrate(crateId: number, currently: boolean) {
       if (currently) {
         await cratesApi.removeSong(crateId, song.name)
       } else {
         await cratesApi.addSong(crateId, song.name)
       }
       queryClient.invalidateQueries({ queryKey: ['crate-songs', song.name] })
     }

     async function createCrate() {
       if (!newName.trim()) return
       setCreating(true)
       try {
         await cratesApi.create(newName.trim())
         queryClient.invalidateQueries({ queryKey: ['crates'] })
         setNewName('')
       } finally {
         setCreating(false)
       }
     }

     return (
       <div className="la-drawer__section">
         <span className="la-drawer__section-title">Crates</span>
         <div className="la-crates-list">
           {allCrates.map((c) => (
             <label key={c.id} className="la-crate-item">
               <input
                 type="checkbox"
                 className="la-checkbox"
                 checked={songCrates.includes(c.id)}
                 onChange={() => toggleCrate(c.id, songCrates.includes(c.id))}
               />
               <span>{c.name}</span>
               <span className="text-muted font-mono la-crate-count">{c.song_count ?? ''}</span>
             </label>
           ))}
         </div>
         <div className="la-crate-create">
           <input
             className="la-crate-input"
             placeholder="New crate…"
             value={newName}
             onChange={(e) => setNewName(e.target.value)}
             onKeyDown={(e) => e.key === 'Enter' && createCrate()}
           />
           <button className="la-crate-add-btn" onClick={createCrate} disabled={creating || !newName.trim()}>
             +
           </button>
         </div>
       </div>
     )
   }
   ```

3. **Add `CratesSection`** inside `TrackDrawer` after the recommendations section:
   ```tsx
   {song && <CratesSection song={song} />}
   ```

4. **Import `cratesApi`** and `useQueryClient` at the top of `LibraryAtlas.tsx`.
   Check if `cratesApi` is already imported — if not, add it from `@/lib/api`.

5. **CSS in `LibraryAtlas.css`**:
   ```css
   .la-crates-list  { display: flex; flex-direction: column; gap: var(--space-1); margin-bottom: var(--space-2); }
   .la-crate-item   { display: flex; align-items: center; gap: var(--space-2); cursor: pointer; font-size: var(--text-sm); }
   .la-crate-count  { margin-left: auto; font-size: var(--text-xs); }
   .la-crate-create { display: flex; gap: var(--space-1); }
   .la-crate-input  {
     flex: 1; background: var(--color-bg-base); border: 1px solid var(--color-border-default);
     border-radius: var(--radius-sm); padding: 3px 8px; font-size: var(--text-xs);
     color: var(--color-text-primary);
   }
   .la-crate-add-btn {
     padding: 3px 10px; background: var(--color-amber-500); border: none;
     border-radius: var(--radius-sm); color: #000; cursor: pointer; font-size: var(--text-sm);
   }
   .la-crate-add-btn:disabled { opacity: 0.4; cursor: not-allowed; }
   ```

---

## Improvement 2: Widget — CamelotWheel + BPM Display

**Goal:** The Widget panel has a `camelotHue()` helper that does its own HSL coloring.
Replace it with the real `CamelotWheel` component (already built in Batch 3) to show
the current track's position on the wheel. Also add a BPM badge to `RecRow`.

### What to do

1. **In `Widget.tsx`**, import `CamelotWheel`:
   ```typescript
   import { CamelotWheel } from '@/components/CamelotWheel'
   ```

2. **In `WidgetPanel`**, add the wheel below the "now playing" track info.
   The component already has `current` (selected track name) and `recs` (similar tracks).
   Fetch the current track's `SongInfo` to get its key:
   ```typescript
   const { data: currentInfo } = useQuery<SongInfo>({
     queryKey: ['song-info-widget', current],
     queryFn: () => libraryApi.get(current!),
     enabled: !!current,
     staleTime: 5 * 60_000,
   })
   ```
   Then render:
   ```tsx
   {current && (
     <div className="wgt-wheel-wrap">
       <CamelotWheel keyA={currentInfo?.key} size={160} />
     </div>
   )}
   ```

3. **Update `RecRow`** to show BPM alongside the camelot code:
   ```tsx
   <span className="wgt-rec__bpm font-mono text-muted">
     {rec.bpm ? `${rec.bpm.toFixed(0)} bpm` : ''}
   </span>
   ```
   Place this after the camelot span inside the rec row.

4. **CSS in `Widget.css`**:
   ```css
   .wgt-wheel-wrap { display: flex; justify-content: center; padding: var(--space-2) 0; }
   .wgt-rec__bpm   { font-size: var(--text-xs); margin-left: auto; }
   ```

5. **Remove** the `camelotHue` function if it's now only used in `RecRow` and you've
   replaced its usage — or keep it if `RecRow` still needs it for the camelot badge color.
   Check before deleting.

---

## Improvement 3: Operations — ETA Timer on Download Jobs

**Goal:** Download job cards in Operations show `${job.progress}%` for running jobs.
Add elapsed time + ETA using the existing `useJobTimer` hook — identical pattern to
the Inspector panel from Batch 1.

### What to do

1. **In `Operations.tsx`**, import:
   ```typescript
   import { useJobTimer } from '@/hooks/useJobTimer'
   ```

2. **Create a small `DownloadTimer` sub-component** (same pattern as `RunningTimer` in
   `RightInspector.tsx`):
   ```tsx
   function DownloadTimer({ job }: { job: Job }) {
     const { elapsed, eta } = useJobTimer(job)
     return (
       <span className="ops-job__timer text-muted font-mono">
         ⏱ {elapsed}s{eta !== null ? ` · ~${eta}s left` : ''}
       </span>
     )
   }
   ```

3. **In `DownloadJobCard`**, render it for RUNNING jobs. Find the existing progress bar
   or status display area and add `<DownloadTimer job={job} />` below it when
   `job.status === 'RUNNING'`.

4. **CSS in `Operations.css`**:
   ```css
   .ops-job__timer { font-size: var(--text-xs); display: block; margin-top: 2px; }
   ```

---

## Improvement 4: Error Boundaries in AppShell

**Goal:** The Suspense fallback handles loading, but if a page component throws
(runtime error, bad API shape, etc.) React will crash the whole app. Wrap each
lazy-loaded route with a per-page error boundary.

### What to do

1. **Create `frontend/src/components/PageErrorBoundary.tsx`**:
   ```typescript
   import { Component, type ErrorInfo, type ReactNode } from 'react'

   interface Props  { children: ReactNode; pageName: string }
   interface State  { hasError: boolean; error: Error | null }

   export class PageErrorBoundary extends Component<Props, State> {
     state: State = { hasError: false, error: null }

     static getDerivedStateFromError(error: Error): State {
       return { hasError: true, error }
     }

     componentDidCatch(error: Error, info: ErrorInfo) {
       console.error(`[PageErrorBoundary:${this.props.pageName}]`, error, info)
     }

     render() {
       if (!this.state.hasError) return this.props.children
       return (
         <div className="peb-container">
           <div className="peb-card">
             <span className="peb-icon">⚠</span>
             <h2 className="peb-title font-display">{this.props.pageName} failed to load</h2>
             <p className="peb-message text-muted">
               {this.state.error?.message ?? 'An unexpected error occurred.'}
             </p>
             <button
               className="peb-retry"
               onClick={() => this.setState({ hasError: false, error: null })}
             >
               Try again
             </button>
           </div>
         </div>
       )
     }
   }
   ```

2. **In `AppShell.tsx`**, wrap each page route:
   ```tsx
   // Before (example):
   <Route path="mix-deck" element={
     <Suspense fallback={<PageFallback />}>
       <MixDeck />
     </Suspense>
   } />

   // After:
   <Route path="mix-deck" element={
     <PageErrorBoundary pageName="Mix Deck">
       <Suspense fallback={<PageFallback />}>
         <MixDeck />
       </Suspense>
     </PageErrorBoundary>
   } />
   ```
   Apply this pattern to all 8 lazy-loaded routes.

3. **CSS — add to a new `PageErrorBoundary.css`** (import in the component):
   ```css
   .peb-container {
     display: flex; align-items: center; justify-content: center;
     height: 100%; min-height: 300px;
   }
   .peb-card {
     display: flex; flex-direction: column; align-items: center; gap: var(--space-3);
     padding: var(--space-6); background: var(--color-bg-elevated);
     border: 1px solid var(--color-crimson-500); border-radius: var(--radius-lg);
     max-width: 400px; text-align: center;
   }
   .peb-icon    { font-size: 2rem; }
   .peb-title   { font-size: var(--text-base); margin: 0; }
   .peb-message { font-size: var(--text-sm); margin: 0; }
   .peb-retry   {
     padding: var(--space-2) var(--space-4); background: var(--color-crimson-500);
     border: none; border-radius: var(--radius-sm); color: #fff;
     cursor: pointer; font-size: var(--text-sm);
   }
   .peb-retry:hover { opacity: 0.85; }
   ```

---

## Notes for Claude Code

- **Build order:** 3 (trivial, one hook import) → 4 (self-contained class component) →
  1 (needs useQueryClient check) → 2 (needs CamelotWheel import check)
- For Improvement 1: verify `cratesApi.removeSong` is the correct method name in `api.ts`
  (it may be `removeSong` or `deleteSong`). Read `api.ts` before calling it.
- For Improvement 1: the `CratesSection`'s inner query pattern (fetching all songs per
  crate to check membership) is O(n crates) API calls — acceptable for small libraries.
  If `cratesApi.list` returns crates with embedded song arrays, simplify accordingly.
- For Improvement 2: `CamelotWheel` takes `keyA?: string` and `keyB?: string` and `size?: number`.
  Only pass `keyA` here (current track only) and `size={160}` to fit the widget panel.
- For Improvement 2: check if `SimilarTrack` type includes `bpm` — read `types/index.ts`.
  If not, add it as `bpm?: number`.
- For Improvement 4: `getDerivedStateFromError` is a static method — TypeScript may warn
  about the return type; annotate it as `Partial<State>` if needed.
- Run `tsc --noEmit` after each improvement.
