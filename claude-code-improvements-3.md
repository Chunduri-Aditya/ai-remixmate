# AI RemixMate — Batch 3 Improvements for Claude Code

## Context

Read `CLAUDE.md` first. Batches 1 and 2 added: WaveSurfer waveforms, TransitionTimeline,
RemixControls, WaveformPlayer in Vault, stems mini-player, job ETA, retry button.
`@dnd-kit` is NOT yet installed.

---

## Improvement 1: Drag-and-Drop Reorder in Set Builder

**Goal:** Replace the up/down arrow buttons in SetBuilder with @dnd-kit drag-and-drop
sortable rows. This is the most visible UX upgrade for the page.

### What to do

1. **Install:**
   ```
   cd frontend && npm install @dnd-kit/core @dnd-kit/sortable @dnd-kit/utilities
   ```

2. **Rewrite `SetRow` in `SetBuilder.tsx`** to be a sortable item:
   ```typescript
   import {
     DndContext, closestCenter, KeyboardSensor, PointerSensor,
     useSensor, useSensors, type DragEndEvent,
   } from '@dnd-kit/core'
   import {
     SortableContext, verticalListSortingStrategy,
     useSortable, arrayMove,
   } from '@dnd-kit/sortable'
   import { CSS } from '@dnd-kit/utilities'
   ```

   The `useSortable` hook gives you `attributes`, `listeners`, `setNodeRef`,
   `transform`, `transition`, `isDragging`. Apply them to the row element:
   ```tsx
   const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
     useSortable({ id: entry.id })
   const style = {
     transform: CSS.Transform.toString(transform),
     transition,
     opacity: isDragging ? 0.4 : 1,
   }
   ```

   Replace the `<ArrowUp>` / `<ArrowDown>` buttons with a drag handle icon
   (`GripVertical` from lucide-react, size 14) that receives `{...attributes} {...listeners}`.
   Keep the `<Trash2>` remove button.

3. **Wrap the set list** in `DndContext` + `SortableContext`:
   ```tsx
   const sensors = useSensors(
     useSensor(PointerSensor),
     useSensor(KeyboardSensor),
   )

   function handleDragEnd(event: DragEndEvent) {
     const { active, over } = event
     if (over && active.id !== over.id) {
       setSet((prev) => {
         const oldIndex = prev.findIndex((e) => e.id === active.id)
         const newIndex = prev.findIndex((e) => e.id === over.id)
         return arrayMove(prev, oldIndex, newIndex)
       })
     }
   }
   ```
   Wrap the list:
   ```tsx
   <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
     <SortableContext items={set.map((e) => e.id)} strategy={verticalListSortingStrategy}>
       {set.map((entry, i) => (
         <SetRow key={entry.id} entry={entry} index={i} total={set.length} onRemove={removeEntry} />
       ))}
     </SortableContext>
   </DndContext>
   ```
   Remove `onMove` prop from `SetRow` and delete `moveEntry` function entirely.

4. **In `SetBuilder.css`**, add:
   ```css
   .sb-drag-handle {
     cursor: grab;
     color: var(--color-text-muted);
     display: flex;
     align-items: center;
     padding: 0 var(--space-1);
   }
   .sb-drag-handle:active { cursor: grabbing; }
   ```

---

## Improvement 2: Camelot Wheel Visualization

**Goal:** Create a reusable `CamelotWheel` SVG component. Render it in `MixDeck.tsx`
next to the CompatPanel to show the two tracks' keys on the wheel. Compatible positions
glow green; incompatible positions are dim.

### Camelot map (hardcode this in the component)

```typescript
// camelotMap[key_string] = { pos: 1–12, type: 'A' | 'B' }
// A = minor, B = major
const CAMELOT: Record<string, { pos: number; type: 'A' | 'B' }> = {
  'Am': {pos:1,'A'}, 'C':  {pos:1,'B'},
  'Em': {pos:2,'A'}, 'G':  {pos:2,'B'},
  'Bm': {pos:3,'A'}, 'D':  {pos:3,'B'},
  'F#m':{pos:4,'A'}, 'A':  {pos:4,'B'},
  'C#m':{pos:5,'A'}, 'E':  {pos:5,'B'},
  'G#m':{pos:6,'A'}, 'B':  {pos:6,'B'},
  'Ebm':{pos:7,'A'}, 'F#': {pos:7,'B'},
  'Bbm':{pos:8,'A'}, 'Db': {pos:8,'B'},
  'Fm': {pos:9,'A'}, 'Ab': {pos:9,'B'},
  'Cm': {pos:10,'A'},'Eb': {pos:10,'B'},
  'Gm': {pos:11,'A'},'Bb': {pos:11,'B'},
  'Dm': {pos:12,'A'},'F':  {pos:12,'B'},
}
```

The backend returns keys like `"C major"`, `"A minor"`, `"F# major"` — normalize by
stripping `" major"` / `" minor"` before lookup (e.g. `"C major"` → `"C"`,
`"A minor"` → `"Am"`).

### SVG layout

The wheel is two concentric rings of 12 segments each, drawn with SVG `path` arcs:
- **Outer ring** (B = major): radius 90–120px, lighter color
- **Inner ring** (A = minor): radius 55–85px, deeper color

Each of the 12 positions is a 30° arc slice. Color them with a 12-step hue rotation
(use `hsl(${(pos - 1) * 30}, 55%, 28%)` as base, lighter when active).

Active segment (matches a selected song): bright `hsl(${(pos-1)*30}, 80%, 55%)` + white label.
Compatible-but-not-active (adjacent position ±1 or same pos opposite ring): `opacity: 0.7`.
All others: `opacity: 0.25`.

Label each segment with its Camelot code (e.g. "1A", "1B") in the center of the arc.
Draw these with SVG `text` elements rotated to follow the arc center angle.

### Props

```typescript
interface CamelotWheelProps {
  keyA?: string   // e.g. "C major"
  keyB?: string   // e.g. "A minor"
  size?: number   // default 260
}
```

### Where to render

In `MixDeck.tsx`, render `<CamelotWheel keyA={songInfoA?.key} keyB={songInfoB?.key} />`
inside the CompatPanel section, below the score and Camelot text labels — only when
both songs are selected. The wheel should be 220px and centered.

### File location

`frontend/src/components/CamelotWheel.tsx` — self-contained, no CSS file needed
(inline styles only, all derived from the `size` prop and color calculations).

---

## Improvement 3: Library Atlas — Bulk Operations

**Goal:** Add checkbox selection to Library Atlas with a sticky bulk action bar
that appears when tracks are selected.

### What to do

1. **In `LibraryAtlas.tsx`**, add state:
   ```typescript
   const [selected, setSelected] = useState<Set<string>>(new Set())
   ```

2. **Add a checkbox column** to the table header and each `SongRow`:
   - Header checkbox: checked when `selected.size === songs.length`, indeterminate
     when `selected.size > 0 && selected.size < songs.length` (set via `ref` and
     `inputRef.current.indeterminate = ...`), clicking it selects/deselects all
   - Row checkbox: in the first cell, `stopPropagation()` so it doesn't trigger
     the row-click inspector open

3. **Bulk action bar** — a fixed bar that slides up from the bottom when
   `selected.size > 0`:
   ```tsx
   {selected.size > 0 && (
     <div className="la-bulk-bar">
       <span className="text-muted font-mono">{selected.size} selected</span>
       <button className="la-bulk-btn" onClick={handleBulkAnalyze}>
         <BarChart2 size={12} /> Analyze
       </button>
       <button className="la-bulk-btn" onClick={handleBulkStems}>
         <Layers size={12} /> Split Stems
       </button>
       <button className="la-bulk-btn la-bulk-btn--ghost" onClick={() => setSelected(new Set())}>
         Clear
       </button>
     </div>
   )}
   ```

4. **`handleBulkAnalyze`** — iterate over `selected`, call `analysisApi.analyze(name)`
   for each, collect job IDs, upsert all into the job store. Fire-and-forget with
   `Promise.allSettled`.

5. **`handleBulkStems`** — same pattern with `stemsApi.split(name)`.
   Check `frontend/src/lib/api.ts` for the exact method names — they likely exist
   as `analysisApi.analyze` and `stemsApi.split`.

6. **CSS in `LibraryAtlas.css`**:
   ```css
   .la-bulk-bar {
     position: fixed;
     bottom: var(--space-4);
     left: 50%;
     transform: translateX(-50%);
     display: flex;
     align-items: center;
     gap: var(--space-2);
     padding: var(--space-2) var(--space-4);
     background: var(--color-surface-elevated);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-lg);
     box-shadow: 0 8px 32px rgba(0,0,0,0.4);
     z-index: 100;
   }
   .la-bulk-btn {
     display: flex; align-items: center; gap: 4px;
     padding: 4px 10px;
     background: var(--color-amber-500);
     color: #000;
     border: none;
     border-radius: var(--radius-sm);
     font-size: var(--text-xs);
     cursor: pointer;
   }
   .la-bulk-btn--ghost {
     background: transparent;
     color: var(--color-text-secondary);
     border: 1px solid var(--color-border);
   }
   .la-checkbox { accent-color: var(--color-amber-500); cursor: pointer; }
   ```

---

## Improvement 4: Mission Control — Job Activity Sparkline

**Goal:** Replace the static "Active jobs" StatCard with a small inline sparkline
bar chart showing job completions over the last 60 minutes, built purely from
`useAppStore` data (no extra backend calls).

### What to do

1. **Track job completion timestamps** in `appStore.ts`. Add to state:
   ```typescript
   completionLog: number[]   // array of Unix timestamps (ms) for COMPLETED jobs
   logCompletion: () => void
   ```
   In `upsertJob`, when a job transitions to `COMPLETED`, call `logCompletion()`.
   Keep only the last 60 entries.

2. **Create a `Sparkline` component** inline in `MissionControl.tsx` (~40 lines):
   ```typescript
   function Sparkline({ timestamps }: { timestamps: number[] }) {
     const W = 120, H = 32, BUCKETS = 12
     const now = Date.now()
     const bucketMs = 5 * 60 * 1000  // 5-min buckets → 60 min total
     const counts = Array.from({ length: BUCKETS }, (_, i) => {
       const bucketEnd   = now - i * bucketMs
       const bucketStart = bucketEnd - bucketMs
       return timestamps.filter((t) => t >= bucketStart && t < bucketEnd).length
     }).reverse()
     const max = Math.max(...counts, 1)
     const barW = W / BUCKETS - 1
     return (
       <svg width={W} height={H} style={{ display: 'block' }}>
         {counts.map((c, i) => {
           const barH = Math.max(2, (c / max) * H)
           return (
             <rect
               key={i}
               x={i * (barW + 1)}
               y={H - barH}
               width={barW}
               height={barH}
               rx={1}
               fill={c > 0 ? 'var(--color-green-500)' : 'var(--color-border)'}
               opacity={0.6 + 0.4 * (i / BUCKETS)}
             />
           )
         })}
       </svg>
     )
   }
   ```

3. **Replace** the "Active jobs" `StatCard` in `MissionControl.tsx` with a custom card
   that shows the sparkline below the count:
   ```tsx
   <div className="mc-stat-card mc-stat-card--green mc-stat-card--clickable"
        onClick={() => navigate('/operations')}>
     <div className="mc-stat-card__icon"><Activity size={16} /></div>
     <div className="mc-stat-card__body">
       <span className="mc-stat-card__label">Completions</span>
       <span className="mc-stat-card__value font-display">{completionLog.length}</span>
       <Sparkline timestamps={completionLog} />
       <span className="mc-stat-card__sub">last 60 min</span>
     </div>
   </div>
   ```

4. **Export `completionLog` and `logCompletion`** from the store and call
   `logCompletion()` in `upsertJob` when `status === 'COMPLETED'`.

---

## Notes for Claude Code

- Build order: 1 (no deps) → 2 (no deps) → 3 (needs `analysisApi`/`stemsApi` check) → 4
- The `CAMELOT` map object literal in Improvement 2 has a syntax error as written above —
  fix it when implementing: values should be `{ pos: 1, type: 'A' }` not shorthand
- For Improvement 3, check `api.ts` for the exact stem-split and analyze method names
  before calling them; they may be `stemsApi.split(name)` and `analysisApi.run(name)`
- `arrayMove` from `@dnd-kit/sortable` is a pure array utility — no DOM side effects,
  safe to call inside `setSet`
- Run `tsc --noEmit` after each improvement
