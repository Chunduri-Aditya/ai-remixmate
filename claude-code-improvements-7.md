# AI RemixMate — Batch 7: Render-Path & Client Hardening (single task)

## Context

Read `CLAUDE.md` first. Batches 1–6 are complete (waveforms, remix controls, DnD set
builder, Camelot wheel, toasts, crates panel, frontend test infra via Vitest, Python
tests for config/library/genre/DJ engine/job store/health).

This is **not a feature batch** — it is one focused hardening task. The goal is to make
the audio render path and the API client *fail safe and loud* instead of *silent and
wrong*. This is the part of the app a live demo actually runs through, so a worker that
hangs or a mix that silently renders at the wrong tempo is the worst possible failure.

**Theme:** every numeric input that feeds the render math (BPM, stretch ratio, sample
indices, audio length) must be validated at a single chokepoint, and every network call
must be cancellable. No new features. No UI redesign. Behavior for valid input must be
byte-for-byte identical — this is pure defensive hardening.

---

## Why this matters (don't skip)

The render path already has *scattered, inconsistent* guards. Examples that exist today:

- `dj_engine.py:421` → `if bpm <= 0: bpm = 120.0` (good)
- `dj_engine.py:885` → `bar_samples = max(1, int(round(sr * 60.0 / bpm1 * 4)))` (good)
- `dj_engine.py:654` → `safe_ratio = max(plan.tempo_shift_ratio, 0.001)` — but this only
  guards the **index math**, and only the lower bound.

Meanwhile the actual time-stretch is unguarded:

- `dj_engine.py:142` `_time_stretch(audio, ratio)` clamps **nothing**. A ratio of `0`,
  `NaN`, `inf`, or something absurd like `12.0` (from a mis-detected BPM) flows straight
  into librosa. On a hard crash it's caught and returns audio *unchanged* — i.e. the mix
  silently renders at the wrong tempo with no error surfaced to the user.

The pattern is: the codebase *knows* these inputs are dangerous (it guards them in some
places) but the guarding is copy-pasted and incomplete. Consolidate it.

---

## Scope

**In scope**
1. Backend: centralize numeric guards in `dj_engine.py` and apply them at every call site.
2. Frontend: add request timeout + cancellation to the `api.ts` fetch wrapper.
3. Tests proving each guard triggers (this is the deliverable's proof of work).

**Out of scope** — do NOT touch: transition musicality/voicing, stem crossfade envelopes,
mastering DSP, any router business logic, any page layout. If a change alters output for
*valid* input, it's wrong — revert it.

---

## Part A — Backend: centralize render-path guards

### A1. Add two helpers near the top of `scripts/core/dj_engine.py`

Place them above `_time_stretch` (around line 140). Keep them tiny and pure:

```python
import math

# Stretch ratios outside this band are either mis-detected BPM or numeric
# garbage. librosa will happily waste minutes of CPU (or crash) on them.
_MIN_STRETCH = 0.5   # B is at most 2× faster than A
_MAX_STRETCH = 2.0   # B is at most 2× slower than A

def _safe_ratio(ratio: float) -> float:
    """Clamp a time-stretch ratio to a musically sane, finite band.

    Guards against 0, negative, NaN, and inf — any of which corrupts the
    render. Out-of-band values are clamped (not raised) so a single bad BPM
    detection degrades gracefully to a 'close enough' mix instead of failing
    the whole job.
    """
    if ratio is None or not math.isfinite(ratio) or ratio <= 0.0:
        return 1.0
    return float(min(max(ratio, _MIN_STRETCH), _MAX_STRETCH))

def _safe_bpm(bpm: float, default: float = 120.0) -> float:
    """Return a finite, positive BPM, falling back to `default`."""
    if bpm is None or not math.isfinite(bpm) or bpm <= 0.0:
        return default
    return float(bpm)
```

### A2. Make `_time_stretch` self-defending

It is the single chokepoint every stretch flows through — clamp *inside* it so no caller
can bypass the guard:

```python
def _time_stretch(audio: np.ndarray, ratio: float) -> np.ndarray:
    ratio = _safe_ratio(ratio)          # <-- add this as the first line
    if abs(ratio - 1.0) < 0.02:
        return audio
    ...  # rest unchanged
```

When `_safe_ratio` changes the value, log it once at INFO so it's visible in job logs:

```python
    raw = ratio
    ratio = _safe_ratio(ratio)
    if raw != ratio and raw is not None and math.isfinite(raw):
        log.info("Clamped stretch ratio %.3f → %.3f", raw, ratio)
```

### A3. Replace the ad-hoc guards with the helpers (consistency, not new behavior)

Audit every BPM division and ratio use and route it through the helpers. Known sites
(verify line numbers — they may drift):

- `:421` `if bpm <= 0: bpm = 120.0` → `bpm = _safe_bpm(bpm)`
- `:654` `safe_ratio = max(plan.tempo_shift_ratio, 0.001)` → `safe_ratio = _safe_ratio(plan.tempo_shift_ratio)`
- `:885` and `:886` `60.0 / bpm1` / `max(bpm1, 1.0)` → take `bpm1 = _safe_bpm(bpm1)` once at the top of that function and drop the inline `max(..., 1.0)` band-aids.
- The bare `60.0 / bpm` in `_apply_echo_out` (`:189`) is already guarded by the early
  `if bpm <= 0` return — leave it, but you may swap to `_safe_bpm` for uniformity.

Run `grep -n "/ bpm\|/ bpm1\|/ max(bpm\|tempo_shift_ratio" scripts/core/dj_engine.py`
and make sure **every** hit either goes through a helper or has a pre-existing guard you
can point to.

### A4. Clamp derived sample indices to valid array bounds

`entry_sample_b = int(plan.entry_time_b * sr / safe_ratio)` (~`:655`) and the analogous
`exit_sample_a` can land outside the buffer if a plan carries a bogus time. After each is
computed, clamp before it's used to slice:

```python
entry_sample_b = max(0, min(entry_sample_b, len(track_b_stretched) - 1))
exit_sample_a  = max(0, min(exit_sample_a,  len(track_a) - 1))
```

Do the equivalent in the chain renderer (`render_dj_chain`, ~`:860–910`) for each
per-segment index. Guard empty arrays too: if any input track is `len == 0`, raise a
clear `ValueError(f"Empty audio for {name}")` *early* so the job fails with a readable
message instead of an opaque numpy index error deep in the mix loop.

---

## Part B — Frontend: cancellable, timed-out requests

`frontend/src/lib/api.ts` → the generic `request<T>()` (~`:28`) uses bare `fetch` with no
timeout. If the backend hangs (a stuck worker, a paused container), every call hangs the
UI forever and the existing Toast/error surfaces never fire.

### B1. Add an `AbortController` + timeout to `request<T>`

```typescript
const DEFAULT_TIMEOUT_MS = 30_000

export class ApiError extends Error {
  constructor(public status: number, public path: string, message: string) {
    super(`[${status}] ${path}: ${message}`)
    this.name = 'ApiError'
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const res = await fetch(`${BASE}${path}`, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    })
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText)
      throw new ApiError(res.status, path, text)
    }
    return (await res.json()) as T
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError(0, path, `request timed out after ${timeoutMs}ms`)
    }
    throw err
  } finally {
    clearTimeout(timer)
  }
}
```

Keep the `get/post/del/patch` signatures identical so no call site changes. Throwing
`ApiError` (a subclass of `Error`) is backward-compatible with anything catching `Error`.

### B2. Don't time out long renders

Endpoints that *return fast* (they enqueue a job and return a `job_id`) are fine on the
default 30s. SSE is a separate `EventSource` and is untouched. If any direct call is
expected to run long, pass an explicit larger `timeoutMs` rather than raising the global
default. Audit `api.ts` and note in your summary whether any call needed an override.

---

## Part C — Tests (this is the proof, not optional)

### Python (`tests/test_dj_engine_guards.py`, new file)

Cover the helpers and the chokepoint directly — these are pure functions, no librosa
needed, so mark them so they run in the default suite:

- `_safe_ratio`: `0`, `-1`, `float('nan')`, `float('inf')`, `None` → `1.0`;
  `0.1 → 0.5`; `9.0 → 2.0`; `1.0 → 1.0` (passthrough); `1.5 → 1.5`.
- `_safe_bpm`: `0`, `-5`, `nan`, `inf`, `None` → `120.0`; `128 → 128.0`.
- `_time_stretch(sig, 0.0)` returns the input array **unchanged** (ratio clamped to 1.0,
  hits the `< 0.02` passthrough) — assert `np.array_equal`.
- Index-clamp: feed a tiny synthetic `track_b` and a plan with an `entry_time_b` past the
  end; assert the render does not raise `IndexError` and returns a finite-length buffer.
- Empty-audio: a zero-length track raises `ValueError` with the song name in the message.

### Frontend (`frontend/src/lib/__tests__/api.test.ts`, new file)

Mock `globalThis.fetch`:

- non-2xx → rejects with `ApiError`, and `err.status` equals the mocked status.
- A `fetch` that never resolves → with a short `timeoutMs` the call rejects with an
  `ApiError` whose message contains `timed out`. (Use fake timers / `vi.useFakeTimers()`.)
- a 200 with JSON body → resolves to the parsed object (happy path unchanged).

Run `pytest tests/test_dj_engine_guards.py` and `cd frontend && npm test` and paste the
passing output into your final summary.

---

## Acceptance criteria

- [ ] `_safe_ratio` / `_safe_bpm` added; `_time_stretch` clamps as its first action.
- [ ] Every BPM division and stretch call in `dj_engine.py` routes through a helper or a
      pre-existing guard you can cite by line.
- [ ] Derived sample indices clamped to array bounds; empty audio raises early with a
      readable message.
- [ ] `api.ts` requests time out and abort; `ApiError` carries `status` + `path`; no call
      site signatures changed.
- [ ] New Python and frontend tests pass; existing `pytest` and `npm test` still green.
- [ ] Output for **valid** input is unchanged (spot-check one real `/dj-remix` render
      before/after if a library track is available; otherwise state you reasoned it by
      inspection because clamps are no-ops in the valid band).

---

## Execution contract — DO NOT STOP UNTIL DONE

Work through Parts A → B → C to completion in one continuous pass. Do not stop after one
part to ask whether to continue — the scope above is the full approval. Only pause if you
hit a genuine blocker (a guard would change valid-input behavior, or a test can't be made
to pass without a real fix); in that case fix it, don't punt.

**If you approach a context/output limit before finishing:** before you run out, write
`HARDENING-PROGRESS.md` at the repo root capturing, in this exact structure, everything
needed to resume cold without re-reading this whole prompt:

```markdown
# Batch 7 Hardening — Resume State

## Done
- [x] A1 helpers added (dj_engine.py:~140)
- [x] ...

## In progress
- [ ] <exact file + line you were editing, and what the next edit is>

## Not started
- [ ] <remaining checklist items, copied from Acceptance criteria>

## Notes / decisions
- <any line numbers that drifted, any guard you left intentionally untouched and why>

## How to verify when resumed
- pytest tests/test_dj_engine_guards.py
- cd frontend && npm test
```

Then stop cleanly. The next session reads `HARDENING-PROGRESS.md` first and continues from
"In progress" — no re-derivation. Delete `HARDENING-PROGRESS.md` once every acceptance box
is checked.

End your run with: a one-paragraph summary, the list of files changed, and the passing
test output.
