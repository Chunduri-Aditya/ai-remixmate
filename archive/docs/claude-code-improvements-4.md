# AI RemixMate — Batch 4 Improvements for Claude Code

## Context

Read `CLAUDE.md` first. Batches 1–3 added waveforms, remix controls, DnD set builder,
Camelot wheel, bulk ops, sparkline, and more. `--z-toast: 300` is already defined in
`tokens.css` but no toast system exists.

---

## Improvement 1: Toast Notification System

**Goal:** Show a non-blocking toast when any job reaches COMPLETED or FAILED.
Zero dependencies — build it with a React context + a store watcher.

### What to do

1. **Create `frontend/src/components/Toast.tsx`** — context + hook + renderer:

   ```typescript
   // Types
   export interface ToastItem {
     id: string
     message: string
     level: 'success' | 'error' | 'info'
     jobId?: string
   }

   // Context
   const ToastContext = createContext<{ push: (t: Omit<ToastItem,'id'>) => void }>
     ({ push: () => {} })

   export function useToast() { return useContext(ToastContext) }

   // Provider — manages the toast queue and auto-dismiss
   export function ToastProvider({ children }: { children: React.ReactNode }) {
     const [toasts, setToasts] = useState<ToastItem[]>([])

     function push(t: Omit<ToastItem, 'id'>) {
       const id = crypto.randomUUID()
       setToasts((prev) => [...prev.slice(-4), { ...t, id }])  // cap at 5
       setTimeout(() => setToasts((prev) => prev.filter((x) => x.id !== id)), 4000)
     }

     return (
       <ToastContext.Provider value={{ push }}>
         {children}
         <div className="toast-area" aria-live="polite">
           {toasts.map((t) => (
             <div key={t.id} className={`toast toast--${t.level}`}>
               <span className="toast__message">{t.message}</span>
               <button className="toast__close"
                 onClick={() => setToasts((prev) => prev.filter((x) => x.id !== t.id))}>
                 ×
               </button>
             </div>
           ))}
         </div>
       </ToastContext.Provider>
     )
   }
   ```

2. **Create `frontend/src/hooks/useJobToasts.ts`** — watches the job store and fires
   toasts on status transitions:
   ```typescript
   export function useJobToasts() {
     const { push } = useToast()
     const jobs = useAppStore((s) => s.jobs)
     const prevRef = useRef<Record<string, Job['status']>>({})

     useEffect(() => {
       Object.values(jobs).forEach((job) => {
         const prev = prevRef.current[job.job_id]
         if (prev === job.status) return
         prevRef.current[job.job_id] = job.status
         if (job.status === 'COMPLETED') {
           push({ level: 'success', message: `✓ ${job.type} complete`, jobId: job.job_id })
         } else if (job.status === 'FAILED') {
           push({ level: 'error', message: `✗ ${job.type} failed`, jobId: job.job_id })
         }
       })
     }, [jobs, push])
   }
   ```

3. **Wire up in `AppShell.tsx`:**
   - Wrap the return with `<ToastProvider>`
   - Add `<JobToastWatcher />` (a component that calls `useJobToasts()` and returns null)
   inside the provider

4. **CSS — add to a new `frontend/src/components/Toast.css`** (import it in Toast.tsx):
   ```css
   .toast-area {
     position: fixed;
     bottom: var(--space-4);
     right: var(--space-4);
     z-index: var(--z-toast);
     display: flex;
     flex-direction: column;
     gap: var(--space-2);
     pointer-events: none;
   }
   .toast {
     display: flex;
     align-items: center;
     justify-content: space-between;
     gap: var(--space-3);
     padding: var(--space-2) var(--space-3);
     border-radius: var(--radius-md);
     border: 1px solid var(--color-border);
     background: var(--color-surface-elevated);
     min-width: 220px;
     max-width: 340px;
     pointer-events: all;
     animation: toast-in 0.2s ease;
     font-size: var(--text-sm);
   }
   @keyframes toast-in {
     from { opacity: 0; transform: translateX(16px); }
     to   { opacity: 1; transform: translateX(0); }
   }
   .toast--success { border-color: var(--color-green-500); }
   .toast--error   { border-color: var(--color-crimson-500); }
   .toast--info    { border-color: var(--color-ice-400); }
   .toast__message { color: var(--color-text-primary); flex: 1; }
   .toast__close   {
     background: none; border: none; cursor: pointer;
     color: var(--color-text-muted); font-size: 1rem; line-height: 1;
     pointer-events: all;
   }
   ```

---

## Improvement 2: MixDeck Post-Remix Result Card

**Goal:** When the remix job in MixDeck reaches COMPLETED, show a result card
below the JobStatusBadge with audio output stats and a direct Vault link.

### What to do

1. **Read `MixDeck.tsx`** — `remixJobId` state exists; `useAppStore` gives the full
   `Job` object with `result`. The result shape (from completed dj_remix jobs) includes:
   `stream_url`, `lufs`, `duration`, `bpm_a`, `bpm_b`, `camelot_a`, `camelot_b`,
   `harmonic_score`, `tempo_ratio`.

2. **Add a `RemixResultCard` component** inside `MixDeck.tsx`:
   ```tsx
   function RemixResultCard({ job }: { job: Job }) {
     const r = job.result as {
       stream_url?: string; lufs?: number; duration?: number;
       bpm_a?: number; bpm_b?: number; harmonic_score?: number; tempo_ratio?: number
     } | undefined
     if (!r) return null
     const BASE = import.meta.env.VITE_API_BASE || '/api'
     const audioUrl = r.stream_url ? `${BASE}${r.stream_url}` : null

     return (
       <div className="md-result-card">
         <div className="md-result-card__header">
           <CheckCircle2 size={14} className="md-result-card__icon" />
           <span className="font-display">Mix Ready</span>
           {audioUrl && (
             <a className="md-result-card__vault-link" href="/mix-vault">
               Open in Vault →
             </a>
           )}
         </div>
         <div className="md-result-card__stats">
           {r.lufs    !== undefined && <><span className="text-muted">LUFS</span><span className="font-mono">{r.lufs.toFixed(1)}</span></>}
           {r.duration !== undefined && <><span className="text-muted">Duration</span><span className="font-mono">{Math.floor(r.duration/60)}:{String(Math.floor(r.duration%60)).padStart(2,'0')}</span></>}
           {r.tempo_ratio !== undefined && <><span className="text-muted">Tempo ratio</span><span className="font-mono">{r.tempo_ratio.toFixed(3)}</span></>}
           {r.harmonic_score !== undefined && <><span className="text-muted">Key match</span><span className="font-mono">{Math.round(r.harmonic_score*100)}%</span></>}
         </div>
         {audioUrl && (
           <audio controls src={audioUrl} className="md-result-card__player" />
         )}
       </div>
     )
   }
   ```

3. **Render it** in the MixDeck JSX, right after `{remixJobId && <JobStatusBadge jobId={remixJobId} />}`:
   ```tsx
   {remixJob?.status === 'COMPLETED' && <RemixResultCard job={remixJob} />}
   ```
   Where `remixJob = useAppStore((s) => remixJobId ? s.jobs[remixJobId] : null)`.
   Check if this selector already exists in the component — add it if not.

4. **CSS in `MixDeck.css`**:
   ```css
   .md-result-card {
     background: var(--color-surface-elevated);
     border: 1px solid var(--color-green-500);
     border-radius: var(--radius-md);
     padding: var(--space-3);
     display: flex;
     flex-direction: column;
     gap: var(--space-2);
   }
   .md-result-card__header {
     display: flex;
     align-items: center;
     gap: var(--space-2);
     color: var(--color-green-500);
   }
   .md-result-card__vault-link {
     margin-left: auto;
     color: var(--color-ice-400);
     font-size: var(--text-xs);
     text-decoration: none;
   }
   .md-result-card__vault-link:hover { text-decoration: underline; }
   .md-result-card__stats {
     display: grid;
     grid-template-columns: 1fr 1fr;
     gap: var(--space-1) var(--space-4);
     font-size: var(--text-xs);
   }
   .md-result-card__player {
     width: 100%;
     height: 32px;
     accent-color: var(--color-green-500);
   }
   ```

---

## Improvement 3: SignalSearch — Quick Deck Actions

**Goal:** Add "→ A", "→ B", "⚡ Mix" hover action buttons to each `ResultCard`
in SignalSearch. "⚡ Mix" navigates to MixDeck with both the reference song and
this result pre-loaded via URL params.

### What to do

1. **Read `SignalSearch.tsx`** — `refSong` is the selected reference track name
   (string state). `ResultCard` renders each `SimilarTrack`. The component renders
   `filteredResults.map((t, i) => <ResultCard key={t.name} track={t} rank={i+1} />)`.

2. **Update `ResultCard` props:**
   ```typescript
   interface ResultCardProps {
     track: SimilarTrack
     rank: number
     refSong: string   // add this
   }
   ```

3. **Add action buttons** inside `ResultCard`:
   ```tsx
   import { useNavigate } from 'react-router-dom'

   function ResultCard({ track, rank, refSong }: ResultCardProps) {
     const navigate = useNavigate()
     // existing card JSX...
     // Add at the bottom of the card:
     return (
       <div className="ss-card">
         {/* existing content */}
         <div className="ss-card__actions">
           <button className="ss-card__action-btn"
             onClick={() => navigate(`/mix-deck?song_a=${encodeURIComponent(refSong)}`)}>
             → A
           </button>
           <button className="ss-card__action-btn"
             onClick={() => navigate(`/mix-deck?song_b=${encodeURIComponent(track.name)}`)}>
             → B
           </button>
           <button className="ss-card__action-btn ss-card__action-btn--mix"
             onClick={() => navigate(`/mix-deck?song_a=${encodeURIComponent(refSong)}&song_b=${encodeURIComponent(track.name)}`)}>
             ⚡ Mix
           </button>
         </div>
       </div>
     )
   }
   ```

4. **Pass `refSong`** down where `ResultCard` is rendered:
   ```tsx
   filteredResults.map((t, i) => (
     <ResultCard key={t.name} track={t} rank={i + 1} refSong={refSong} />
   ))
   ```

5. **CSS in `SignalSearch.css`**:
   ```css
   .ss-card__actions {
     display: flex;
     gap: var(--space-1);
     padding-top: var(--space-2);
     opacity: 0;
     transition: opacity 0.15s;
   }
   .ss-card:hover .ss-card__actions { opacity: 1; }
   .ss-card__action-btn {
     padding: 2px 8px;
     font-size: var(--text-xs);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-sm);
     background: transparent;
     color: var(--color-text-secondary);
     cursor: pointer;
   }
   .ss-card__action-btn:hover { border-color: var(--color-ice-400); color: var(--color-ice-400); }
   .ss-card__action-btn--mix  { border-color: var(--color-amber-500); color: var(--color-amber-500); }
   .ss-card__action-btn--mix:hover { background: color-mix(in srgb, var(--color-amber-500) 15%, transparent); }
   ```

---

## Improvement 4: Keyboard Shortcuts

**Goal:** Gmail-style navigation shortcuts (`g` + letter) plus a `?` help overlay.
Register globally in AppShell.

### What to do

1. **Create `frontend/src/hooks/useKeyboardShortcuts.ts`**:
   ```typescript
   import { useEffect, useRef } from 'react'
   import { useNavigate } from 'react-router-dom'

   const NAV_MAP: Record<string, string> = {
     m: '/mission-control',
     l: '/library-atlas',
     x: '/mix-deck',
     s: '/set-builder',
     q: '/signal-search',
     a: '/ai-lab',
     v: '/mix-vault',
     o: '/operations',
   }

   export function useKeyboardShortcuts(
     onShowHelp: () => void,
     onHideHelp: () => void,
   ) {
     const navigate = useNavigate()
     const gPressed = useRef(false)
     const timer    = useRef<ReturnType<typeof setTimeout>>()

     useEffect(() => {
       function handler(e: KeyboardEvent) {
         const tag = (e.target as HTMLElement).tagName
         if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
         if (e.metaKey || e.ctrlKey) return

         if (e.key === '?') { onShowHelp(); return }
         if (e.key === 'Escape') { onHideHelp(); return }

         if (e.key === 'g') {
           gPressed.current = true
           clearTimeout(timer.current)
           timer.current = setTimeout(() => { gPressed.current = false }, 1000)
           return
         }

         if (gPressed.current && NAV_MAP[e.key]) {
           gPressed.current = false
           navigate(NAV_MAP[e.key])
         }
       }
       window.addEventListener('keydown', handler)
       return () => window.removeEventListener('keydown', handler)
     }, [navigate, onShowHelp, onHideHelp])
   }
   ```

2. **Create `frontend/src/components/ShortcutsModal.tsx`** — a centered overlay:
   ```tsx
   const SHORTCUTS = [
     { keys: 'g → m', desc: 'Mission Control' },
     { keys: 'g → l', desc: 'Library Atlas' },
     { keys: 'g → x', desc: 'Mix Deck' },
     { keys: 'g → s', desc: 'Set Builder' },
     { keys: 'g → q', desc: 'Signal Search' },
     { keys: 'g → a', desc: 'AI Lab' },
     { keys: 'g → v', desc: 'Mix Vault' },
     { keys: 'g → o', desc: 'Operations' },
     { keys: '?',     desc: 'Show this help' },
     { keys: 'Esc',   desc: 'Close' },
   ]

   export function ShortcutsModal({ onClose }: { onClose: () => void }) {
     return (
       <div className="shortcuts-backdrop" onClick={onClose}>
         <div className="shortcuts-modal" onClick={(e) => e.stopPropagation()}>
           <h3 className="shortcuts-modal__title font-display">Keyboard shortcuts</h3>
           <div className="shortcuts-modal__list">
             {SHORTCUTS.map((s) => (
               <div key={s.keys} className="shortcuts-modal__row">
                 <kbd className="shortcuts-modal__kbd font-mono">{s.keys}</kbd>
                 <span className="text-secondary">{s.desc}</span>
               </div>
             ))}
           </div>
           <button className="shortcuts-modal__close text-muted" onClick={onClose}>
             Close
           </button>
         </div>
       </div>
     )
   }
   ```

3. **Wire into `AppShell.tsx`**:
   ```tsx
   const [showShortcuts, setShowShortcuts] = useState(false)
   useKeyboardShortcuts(
     () => setShowShortcuts(true),
     () => setShowShortcuts(false),
   )
   // At the end of the return, inside ToastProvider:
   {showShortcuts && <ShortcutsModal onClose={() => setShowShortcuts(false)} />}
   ```

4. **CSS for the modal** (add to a new `ShortcutsModal.css`):
   ```css
   .shortcuts-backdrop {
     position: fixed; inset: 0;
     background: rgba(0,0,0,0.6);
     z-index: var(--z-modal, 200);
     display: flex; align-items: center; justify-content: center;
   }
   .shortcuts-modal {
     background: var(--color-surface-elevated);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-lg);
     padding: var(--space-5);
     width: 320px;
     display: flex; flex-direction: column; gap: var(--space-3);
   }
   .shortcuts-modal__title { font-size: var(--text-base); margin: 0; }
   .shortcuts-modal__list  { display: flex; flex-direction: column; gap: var(--space-2); }
   .shortcuts-modal__row   { display: flex; align-items: center; gap: var(--space-3); }
   .shortcuts-modal__kbd   {
     background: var(--color-surface);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-sm);
     padding: 2px 6px;
     font-size: var(--text-xs);
     min-width: 72px;
     text-align: center;
   }
   .shortcuts-modal__close {
     align-self: flex-end;
     background: none; border: none;
     cursor: pointer; font-size: var(--text-sm);
   }
   ```

---

## Notes for Claude Code

- **Build order:** 1 (Toast) → 2 (result card, depends on nothing) → 3 (SignalSearch)
  → 4 (shortcuts, wraps AppShell last since Toast provider changes are already there)
- For Improvement 1: `JobToastWatcher` is a component that renders `null` and calls
  `useJobToasts()`. Mount it inside `ToastProvider` in AppShell.
- For Improvement 2: check if `remixJob` is already derived in the component or if
  you need to add `const remixJob = useAppStore((s) => remixJobId ? s.jobs[remixJobId] : null)`
- For Improvement 3: `refSong` is already a string state in the component — just pass
  it as a prop to `ResultCard`. Check the exact state variable name first.
- `--z-modal` may not be defined in tokens.css — define it as `200` there, and confirm
  `--z-toast` is `300` (higher than modal).
- Run `tsc --noEmit` after each improvement.
