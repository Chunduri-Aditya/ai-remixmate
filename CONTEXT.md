# AUDIT CONTEXT — handoff for the fix pass

Companion to `AUDIT.md`. Read this first if you're picking up the fixes. It explains *why* the bugs
exist, how they interrelate, and how to verify a fix actually fixes the audible/observable behavior
(not just the test shape).

## State at audit time

- Branch: `main`. Tests: 190/190 pass. Frontend builds clean.
- Python 3.12 confirmed (per `scripts/api/**/__pycache__`). librosa-dependent tests are marked
  `@pytest.mark.dj_analysis` and auto-skip when numba/librosa fail to init.
- Files audited in full: `dj_engine.py`, `dj_analysis.py`, `key_detection.py`, `mastering.py`,
  `setlist_planner.py`, `style_transfer.py`, `api/main.py`, `api/jobs.py`, `api/routes.py`,
  `api/schemas.py`, `api/task_modules/remix.py`, `frontend/src/types/index.ts`,
  `tests/test_core_modules.py`, `tests/test_benchmarks.py`.

## The core problem behind the findings

**Tests assert shape, not behavior.** Every `render()` test checks dtype, no-NaN, in-range, and
length. None checks that the right thing is audible at the right time. So the two worst render bugs
are invisible to a green suite. Any fix here must come with a *behavioral* assertion, or the next
regression hides the same way.

## How the bugs interrelate (don't fix in isolation)

- **The two render paths have diverged.** `render()` (single transition) and `render_chain()`
  (N-song) implement the same crossfade differently. `render_chain` is *correct* (explicit
  `a_env`/`b_env` arrays, `b_env[:mid]=0`); `render()` is *wrong* (relies on
  `_apply_dynamic_eq_fade` which never silences B's head). Fix `render()` to match `render_chain`'s
  semantics; consider extracting one shared crossfade helper so they can't drift again.
- **Three Camelot/key tables exist and disagree.** `key_detection.CAMELOT` (correct),
  `key_detection.pitch_shift_for_camelot`'s inline `camelot_to_semitone` (WRONG),
  `setlist_planner._SPOTIFY_KEY_CAMELOT` (correct), and `track_metadata.key_to_camelot` (tested,
  correct). The pitch-shift bug is fed into audio by `plan.suggested_pitch_shift`. Fix by deriving
  semitones from the one correct source (`CAMELOT`/`NOTE_NAMES`) and deleting the inline table.
- **The clash path is doubly degraded.** When consonance `< 0.35`, `plan_transition` shortens the
  window (good intent) but leaves `eq.bass_swap_bar` stale → bass swap moves to the end → bass-clash
  protection turns OFF exactly when keys clash. So "harmonic clash handling" currently makes clashes
  *worse*, not better.
- **SSE + jobs are coupled.** The worker-thread `get_event_loop()` failure means background job
  progress never broadcasts; the UI's polling fallback masks it. If you "fix" polling you'll hide the
  real SSE bug. Fix the loop capture, then confirm SSE frames actually arrive.

## Verification approach (per fix)

- **render B-silence (D1 critical):** render with `full_output=False`; assert
  `rms(first_half) ≈ rms(A_alone_first_half)` and that B's isolated contribution in `[:mid]` is ~0.
  Easiest: feed A=silence, B=tone → first half must be silent.
- **clash bass swap (D1):** craft two structures with clashing keys (consonance `< 0.35`) and assert
  `eq.bass_swap_bar` maps to ~midpoint of the *final* `transition_bars`, not the end.
- **Camelot table (D1):** parametrize all 24 codes; assert `pitch_shift_for_camelot(x, x) == 0` and a
  handful of known intervals (C→G = +7→normalized -5? use the ±6 normalization rule) against
  hand-computed tonics. Cross-check against `_SPOTIFY_KEY_CAMELOT`.
- **limiter perf (D1):** micro-benchmark `apply_limiter` on 3 min @ 44.1 kHz before/after; assert
  output is sample-for-sample close to the loop version (correctness preserved).
- **SSE loop (D4):** start a job that calls `update_job` from the worker; assert a `job_updated`
  frame reaches a connected SSE client (or that the broadcaster coroutine was scheduled).
- **SQLite leak (D4):** run N mutations; assert open-fd count (or connection count) stays flat.

## Gotchas carried over from CLAUDE.md (still true)

- `job_store.submit_job()` runs tasks in a ThreadPoolExecutor — no async I/O inside task functions.
- Progress is 0–1 in REST `/jobs`, 0–100 in SSE frames; `normalizeJob()` in `api.ts` bridges. Don't
  "unify" without checking both consumers.
- `config.local.yaml` is gitignored; never commit keys.
- Demucs is CPU by default; `separation.device: mps` for Apple Silicon.

## Recommended order

See `AUDIT.md` → "Suggested fix sequence". Start: (1) render B-silence, (2) SSE loop capture,
(3) Camelot table + test. These three give the biggest behavioral wins for the least surface area.

## Scope note

Audit was read-only. `task_modules/{download,stems,generative,analysis,lab}.py` and most routers were
sampled (via `remix.py` + grep), not read line-by-line — treat their findings as representative, not
exhaustive. A second pass on `routers/spotify.py` (17 HTTPException sites) and `generative.py`
(0 HTTPException, all-async) is worthwhile before a release.
