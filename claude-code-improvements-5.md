# AI RemixMate — Batch 5 Improvements for Claude Code

## Context

Read `CLAUDE.md` first. Batches 1–4 are complete. Python tests already cover config,
Camelot, library, genre, DJ engine, audit, logging, job store, and health endpoints.
Frontend has zero test infrastructure. Gaps to fill: mastering module, setlist planner,
API routes (library/jobs/remix), and frontend component/hook tests. Also: Library Atlas
has no virtual scrolling — adds up when the library grows past ~200 tracks.

---

## Improvement 1: Frontend Test Infrastructure + Key Tests

**Goal:** Set up Vitest + React Testing Library and write meaningful tests for the
hooks and components added across Batches 1–4.

### Setup

1. **Install:**
   ```
   cd frontend && npm install -D vitest @vitest/ui @testing-library/react @testing-library/user-event @testing-library/jest-dom jsdom
   ```

2. **Add to `vite.config.ts`** (inside `defineConfig`):
   ```typescript
   test: {
     globals: true,
     environment: 'jsdom',
     setupFiles: ['./src/test-setup.ts'],
   },
   ```

3. **Create `frontend/src/test-setup.ts`:**
   ```typescript
   import '@testing-library/jest-dom'
   ```

4. **Add to `package.json` scripts:**
   ```json
   "test": "vitest",
   "test:ui": "vitest --ui"
   ```

### Tests to write

**`frontend/src/hooks/__tests__/useJobTimer.test.ts`:**
```typescript
import { renderHook, act } from '@testing-library/react'
import { useJobTimer } from '../useJobTimer'
import type { Job } from '@/types'

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    job_id: 'test-1', status: 'RUNNING', type: 'dj_remix',
    progress: 50, message: '', meta: {},
    created_at: new Date(Date.now() - 30_000).toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  }
}

describe('useJobTimer', () => {
  it('returns elapsed seconds >= 30 for a job started 30s ago', () => {
    const { result } = renderHook(() => useJobTimer(makeJob()))
    expect(result.current.elapsed).toBeGreaterThanOrEqual(29)
  })

  it('returns null eta when progress is 0', () => {
    const { result } = renderHook(() => useJobTimer(makeJob({ progress: 0 })))
    expect(result.current.eta).toBeNull()
  })

  it('returns positive eta when progress > 2', () => {
    const { result } = renderHook(() => useJobTimer(makeJob({ progress: 50 })))
    expect(result.current.eta).not.toBeNull()
    expect(result.current.eta!).toBeGreaterThan(0)
  })
})
```

**`frontend/src/components/__tests__/CamelotWheel.test.tsx`:**

The `CamelotWheel` component has an internal key normalizer (`"C major"` → `"C"`,
`"A minor"` → `"Am"`). Export a `normalizeKey` function from `CamelotWheel.tsx` so it
can be tested in isolation, then:
```typescript
import { normalizeKey } from '../CamelotWheel'

describe('normalizeKey', () => {
  it('strips " major"',  () => expect(normalizeKey('C major')).toBe('C'))
  it('appends m for minor', () => expect(normalizeKey('A minor')).toBe('Am'))
  it('handles already-normalized', () => expect(normalizeKey('Am')).toBe('Am'))
  it('handles F# major', () => expect(normalizeKey('F# major')).toBe('F#'))
  it('returns input unchanged when unknown', () => expect(normalizeKey('Xyz')).toBe('Xyz'))
})
```

**`frontend/src/components/__tests__/RemixControls.test.tsx`:**
```typescript
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { RemixControls, REMIX_DEFAULTS } from '../RemixControls'

describe('RemixControls', () => {
  it('renders all three transition bar options', () => {
    render(<RemixControls value={REMIX_DEFAULTS} onChange={() => {}} />)
    expect(screen.getByText('8')).toBeInTheDocument()
    expect(screen.getByText('16')).toBeInTheDocument()
    expect(screen.getByText('32')).toBeInTheDocument()
  })

  it('calls onChange with transition_bars=8 when 8 is clicked', async () => {
    const onChange = vi.fn()
    render(<RemixControls value={REMIX_DEFAULTS} onChange={onChange} />)
    await userEvent.click(screen.getByText('8'))
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ transition_bars: 8 }))
  })

  it('default bridge_beat_mode is none', () => {
    expect(REMIX_DEFAULTS.bridge_beat_mode).toBe('none')
  })
})
```

Run with: `cd frontend && npm test`

---

## Improvement 2: mastering.py Unit Tests

**Goal:** Test the ITU-R BS.1770-4 LUFS implementation and true-peak limiter.
Add to a new file `tests/test_mastering.py`.

```python
"""tests/test_mastering.py — Unit tests for the mastering engine."""
from __future__ import annotations
import numpy as np
import pytest

SR = 44100

def _sine(freq=1000.0, duration=5.0, amplitude=0.5, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)

def _silence(duration=5.0, sr=SR):
    return np.zeros(int(sr * duration), dtype=np.float32)


class TestComputeLufs:
    def test_imports(self):
        from scripts.core.mastering import compute_lufs
        assert callable(compute_lufs)

    def test_silence_returns_very_low_lufs(self):
        from scripts.core.mastering import compute_lufs
        result = compute_lufs(_silence(), SR)
        assert result < -60.0

    def test_full_scale_sine_above_minus_10_lufs(self):
        from scripts.core.mastering import compute_lufs
        result = compute_lufs(_sine(amplitude=0.9), SR)
        assert result > -15.0

    def test_quiet_signal_lower_than_loud(self):
        from scripts.core.mastering import compute_lufs
        loud  = compute_lufs(_sine(amplitude=0.9), SR)
        quiet = compute_lufs(_sine(amplitude=0.1), SR)
        assert quiet < loud

    def test_stereo_input_accepted(self):
        from scripts.core.mastering import compute_lufs
        stereo = np.stack([_sine(), _sine()], axis=0)
        result = compute_lufs(stereo, SR)
        assert isinstance(result, float)


class TestNormalizeToLufs:
    def test_target_lufs_within_tolerance(self):
        from scripts.core.mastering import normalize_to_lufs, compute_lufs
        audio = _sine(amplitude=0.5)
        normed = normalize_to_lufs(audio, SR, target_lufs=-14.0)
        measured = compute_lufs(normed, SR)
        assert abs(measured - (-14.0)) < 1.5  # ±1.5 LUFS tolerance

    def test_output_same_length(self):
        from scripts.core.mastering import normalize_to_lufs
        audio = _sine()
        normed = normalize_to_lufs(audio, SR, target_lufs=-14.0)
        assert len(normed) == len(audio)

    def test_output_dtype_float32(self):
        from scripts.core.mastering import normalize_to_lufs
        normed = normalize_to_lufs(_sine(), SR)
        assert normed.dtype == np.float32


class TestMasterMix:
    def test_returns_tuple(self):
        from scripts.core.mastering import master_mix
        result = master_mix(_sine(), SR)
        assert isinstance(result, tuple) and len(result) == 2

    def test_output_lufs_within_1_of_target(self):
        from scripts.core.mastering import master_mix, compute_lufs
        mastered, report = master_mix(_sine(amplitude=0.5), SR, target_lufs=-14.0)
        measured = compute_lufs(mastered, SR)
        assert abs(measured - (-14.0)) < 1.5

    def test_peak_below_ceiling(self):
        from scripts.core.mastering import master_mix
        audio = _sine(amplitude=1.2)  # intentionally over 0 dBFS
        mastered, report = master_mix(audio, SR, target_lufs=-14.0, ceiling_dbfs=-1.0)
        peak_dbfs = 20 * np.log10(np.abs(mastered).max() + 1e-12)
        assert peak_dbfs <= -0.9  # at or below ceiling

    def test_report_has_lufs_field(self):
        from scripts.core.mastering import master_mix
        _, report = master_mix(_sine(), SR)
        assert hasattr(report, 'lufs_integrated')

    def test_no_nan_in_output(self):
        from scripts.core.mastering import master_mix
        mastered, _ = master_mix(_sine(), SR)
        assert not np.isnan(mastered).any()
```

---

## Improvement 3: setlist_planner.py Unit Tests

**Goal:** Test the greedy optimizer, energy arc, and transition cost function.
Add `tests/test_setlist_planner.py`.

```python
"""tests/test_setlist_planner.py — Unit tests for SetlistPlanner."""
from __future__ import annotations
import pytest


def _make_track(name, bpm=128.0, camelot="8A", energy=0.5):
    from scripts.core.setlist_planner import TrackNode
    return TrackNode(name=name, bpm=bpm, camelot=camelot, energy=energy)


class TestTrackNode:
    def test_energy_level_low(self):
        t = _make_track("a", energy=0.1)
        assert t.energy_level == 1

    def test_energy_level_high(self):
        t = _make_track("a", energy=0.95)
        assert t.energy_level == 10

    def test_energy_level_midrange(self):
        t = _make_track("a", energy=0.5)
        assert 4 <= t.energy_level <= 6

    def test_to_dict_has_required_keys(self):
        d = _make_track("a").to_dict()
        for key in ("name", "bpm", "energy", "camelot"):
            assert key in d


class TestTransitionCost:
    def test_same_track_zero_bpm_cost(self):
        from scripts.core.setlist_planner import transition_cost
        t = _make_track("a", bpm=128.0, camelot="8A")
        cost = transition_cost(t, t)
        assert cost.bpm_cost == pytest.approx(0.0)

    def test_large_bpm_delta_high_cost(self):
        from scripts.core.setlist_planner import transition_cost
        a = _make_track("a", bpm=80.0)
        b = _make_track("b", bpm=160.0)
        cost = transition_cost(a, b)
        assert cost.bpm_cost > 0.5

    def test_adjacent_camelot_low_harmonic_cost(self):
        from scripts.core.setlist_planner import transition_cost
        a = _make_track("a", camelot="8A")
        b = _make_track("b", camelot="9A")  # one step
        cost_adj = transition_cost(a, b)
        c = _make_track("c", camelot="2B")  # far away
        cost_far = transition_cost(a, c)
        assert cost_adj.harmonic_cost < cost_far.harmonic_cost


class TestSetlistPlanner:
    def _make_pool(self):
        return [
            _make_track("a", bpm=128, camelot="8A",  energy=0.3),
            _make_track("b", bpm=130, camelot="8B",  energy=0.5),
            _make_track("c", bpm=132, camelot="9A",  energy=0.7),
            _make_track("d", bpm=140, camelot="9B",  energy=0.9),
            _make_track("e", bpm=128, camelot="10A", energy=0.6),
        ]

    def test_optimize_returns_all_tracks(self):
        from scripts.core.setlist_planner import SetlistPlanner, EnergyArc
        result = SetlistPlanner().optimize(self._make_pool(), arc=EnergyArc.RISE)
        assert len(result) == 5

    def test_optimize_returns_unique_tracks(self):
        from scripts.core.setlist_planner import SetlistPlanner, EnergyArc
        result = SetlistPlanner().optimize(self._make_pool(), arc=EnergyArc.RISE)
        names = [t.name for t in result]
        assert len(names) == len(set(names))

    def test_rise_arc_energy_trend(self):
        """For RISE arc, average energy of second half > first half."""
        from scripts.core.setlist_planner import SetlistPlanner, EnergyArc
        result = SetlistPlanner().optimize(self._make_pool(), arc=EnergyArc.RISE)
        energies = [t.energy for t in result]
        mid = len(energies) // 2
        assert sum(energies[mid:]) / len(energies[mid:]) >= sum(energies[:mid]) / len(energies[:mid]) - 0.15

    def test_single_track_returns_one(self):
        from scripts.core.setlist_planner import SetlistPlanner, EnergyArc
        result = SetlistPlanner().optimize([_make_track("solo")], arc=EnergyArc.FLAT)
        assert len(result) == 1

    def test_empty_pool_returns_empty(self):
        from scripts.core.setlist_planner import SetlistPlanner, EnergyArc
        result = SetlistPlanner().optimize([], arc=EnergyArc.FLAT)
        assert result == []
```

---

## Improvement 4: Virtual Scrolling in Library Atlas

**Goal:** When the library contains many tracks, the Atlas table renders all rows
at once. Add `@tanstack/react-virtual` to windowed-render only the visible rows.

### What to do

1. **Install:**
   ```
   cd frontend && npm install @tanstack/react-virtual
   ```

2. **In `LibraryAtlas.tsx`**, find the `<tbody>` rendering loop. Replace it with a
   virtualizer:
   ```typescript
   import { useVirtualizer } from '@tanstack/react-virtual'

   // Inside the component, get a ref for the scroll container:
   const scrollRef = useRef<HTMLDivElement>(null)

   // The sorted/filtered song list is already computed — call it `displaySongs`.
   // Pass its length to the virtualizer:
   const virtualizer = useVirtualizer({
     count: displaySongs.length,
     getScrollElement: () => scrollRef.current,
     estimateSize: () => 48,       // px per row (match .la-row height in CSS)
     overscan: 8,                  // render 8 rows above/below viewport
   })
   ```

3. **Wrap the table in a scrollable div** with a fixed height:
   ```tsx
   <div
     ref={scrollRef}
     className="la-scroll-container"
     style={{ overflowY: 'auto', height: 'calc(100vh - 220px)' }}
   >
     <table className="la-table">
       <thead>...</thead>
       <tbody
         style={{
           height: `${virtualizer.getTotalSize()}px`,
           position: 'relative',
         }}
       >
         {virtualizer.getVirtualItems().map((vRow) => {
           const song = displaySongs[vRow.index]
           return (
             <Fragment key={song.name}>
               <SongRow
                 song={song}
                 style={{
                   position: 'absolute',
                   top: vRow.start,
                   left: 0,
                   width: '100%',
                   height: vRow.size,
                 }}
                 // ...all existing props
               />
               {expandedStems === song.name && (
                 <StemExpansionRow
                   song={song}
                   style={{ position: 'absolute', top: vRow.start + vRow.size, left: 0, width: '100%' }}
                 />
               )}
             </Fragment>
           )
         })}
       </tbody>
     </table>
   </div>
   ```

4. **Update `SongRow`** to accept and forward a `style` prop to the `<tr>` element.
   Same for `StemExpansionRow`.

5. **In `LibraryAtlas.css`**, ensure `.la-row` has a fixed height:
   ```css
   .la-row { height: 48px; }
   ```

   **Note on stems expansion:** Virtual scrolling and absolute-positioned expansion rows
   interact awkwardly. If the implementation becomes too complex, use a simpler approach:
   disable virtual scrolling for rows that have expanded stems (i.e., keep a non-virtual
   fallback path when `expandedStems !== null`). Comment this clearly in the code.

---

## Notes for Claude Code

- **Build order:** 1 (frontend, self-contained) → 2 (Python, self-contained) → 3 (Python,
  self-contained) → 4 (frontend, touches existing components)
- For Improvement 1: after installing, run `npm test` to verify the setup; then add tests
  one file at a time
- For `normalizeKey` export in `CamelotWheel.tsx`: just change `function normalizeKey`
  to `export function normalizeKey` — it's already pure and side-effect-free
- For Improvement 2+3: run `python -m pytest tests/test_mastering.py tests/test_setlist_planner.py -v`
- For Improvement 4: the `displaySongs` variable name may differ — read the component
  first to find the actual filtered/sorted array name before using it
- Virtual scrolling and the stems expansion row are the trickiest interaction — if the
  absolute-position approach doesn't work cleanly, the fallback note above is acceptable
- Run `tsc --noEmit` after Improvement 4
