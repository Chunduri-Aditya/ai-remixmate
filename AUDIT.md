# AI RemixMate — Deep Audit (June 27, 2026)

Audit across 4 dimensions: correctness, schema drift, test coverage, production readiness.
Findings are ordered by severity within each dimension. Every item links to `file:line`.

**Headline:** the suite (190/190 green) validates *shape* (dtype, no-NaN, clipping, length), not
*behavior*. The bugs that matter most — `render()` mixing Song B during its "silent" half, the
clash-path bass swap, the Camelot semitone table, worker-thread SSE — are precisely the paths with
no behavioral test. A green checkmark here does not mean the mix is correct.

Severity legend: **critical** = wrong output / broken core path · **major** = significant defect or
risk · **minor** = correctness/quality/cleanliness.

---

## Dimension 1 — Correctness (core DSP)

- [ ] **[critical] dj_engine.py:597-605** — `_apply_dynamic_eq_fade(direction="in")` does not silence
  Song B in the first half. It copies the whole array and only fades the tail (`fade_samples`); the
  head (`result[:tail_start]`, first `mid` samples) stays at full amplitude. `render()`'s docstring
  (803-817) promises "First half → Song B is completely silent." It isn't — only B's bass is
  attenuated by the HP ramp; mids/highs ring at full level under A's "solo" half. `render_chain()`
  does this correctly via explicit `b_env[:mid]=0` (1052), so the two render paths disagree.
  **Fix:** for `"in"`, zero the head, then `sin` ramp the tail (mirror `render_chain`'s `b_env`).
  Add a test asserting `rms(B_contribution[:mid]) ≈ 0`.

- [ ] **[major] dj_analysis.py:653-685** — stale `eq.bass_swap_bar` on the harmonic-clash path.
  `EQPlan` is built with `bass_swap_bar = transition_bars//2` (=8 for default 16), then when
  consonance `< 0.35` and bars ≥ 16 the code shortens `transition_bars`→8 and recomputes
  `transition_seconds` **without** rebuilding `eq`. In `render()` (782) `swap_sample` then evaluates
  to `trans_samples` → swap lands at the end → no bass swap. So bass-clash protection is disabled
  exactly when keys clash. **Fix:** move the shortening above `EQPlan` construction, or recompute
  `eq.bass_swap_bar`/`hp_ramp_bars` after shortening.

- [ ] **[major] key_detection.py:664-677** — `pitch_shift_for_camelot` semitone table is inconsistent.
  A-ring entries are uniformly +1 semitone sharp; B-ring `1B`–`7B` are 4 semitones flat (8B–12B are
  correct). E.g. `1B` (B major, tonic 11) maps to `7`; `pitch_shift_for_camelot('8B','1B')` returns
  `-5` when C→B major is `-1`. Same-ring shifts often cancel the offset, which is why it slipped
  through. Disagrees with the correct `setlist_planner._SPOTIFY_KEY_CAMELOT` (41-66). Feeds real
  audio via `plan.suggested_pitch_shift` → `pitch_shift_audio()` (dj_engine.py:1201-1207).
  **Fix:** derive semitones from the existing `CAMELOT`/`NOTE_NAMES` data (single source of truth);
  add a parametrized test over all 24 codes.

- [ ] **[major] mastering.py:219-224** — `apply_limiter` runs a per-sample Python loop over the whole
  signal for release smoothing. Tens of millions of iterations per `master_mix` call (every
  remix/chain/style-transfer). **Fix:** vectorize the one-pole release (`scipy.signal.lfilter` on the
  reduction envelope). Benchmark before/after.

- [ ] **[minor] mastering.py:184-226** — limiter labeled "true-peak" but measures sample peak only
  (`maximum_filter1d` on `abs`, no oversampling); inter-sample peaks can exceed ceiling. Docstring
  (line 7) claims BS.1770 true-peak. **Fix:** 4× oversample before peak detection, or drop the claim.

- [ ] **[minor] dj_engine.py:783** — `swap_sample` divides by `plan.transition_bars` with no zero
  guard (`render_chain` guards with `max(...,1)` at 1011; `render()` doesn't). **Fix:** add the guard.

- [ ] **[minor] dj_engine.py:786-787** — dead bass-swap low-pass coefficients (`b_lp`, `a_lp_coeffs`,
  `b_lp_b`, `b_lp_a`) computed but never applied; bass swap only high-passes A. **Fix:** remove or
  actually apply the LP.

- [ ] **[minor] style_transfer.py:326** — `_extract_melody_chroma()` result computed then never used;
  generation passes raw `audio=source_32k` (356-362). Contradicts the "chromagram conditioning"
  docstring and wastes an HPSS+CQT per call. **Fix:** pass the chroma or delete the dead path + fix docs.

- [ ] **[minor] dj_analysis.py:259, 522** — redundant heavy librosa work: `beat_track` runs in
  `analyze_structure` and again in `_detect_phrase_boundaries`; `chroma_cqt` runs in `detect_key`,
  `_detect_phrase_boundaries`, and `compute_track_vector`. "2–5 s per track" claim unlikely.
  **Fix:** compute beats/chroma once and thread through.

---

## Dimension 2 — Schema drift (Pydantic ↔ TypeScript)

Root cause: `types/index.ts:3` "Mirror of FastAPI schemas; kept in sync manually."

- [ ] **[critical] JobStatus values disagree.** schemas.py:19-24 = `pending|running|done|failed|cancelled`
  (lowercase, `done`); index.ts:8 = `PENDING|RUNNING|COMPLETED|FAILED|CANCELLED` (uppercase,
  `COMPLETED`). The two backend serializers also disagree: SSE `_job_to_response_dict`
  (jobs.py:202-211) normalizes to `COMPLETED`, REST `job_to_response` (jobs.py:490-504) returns raw
  `done`. **Fix:** make REST `job_to_response` use the same `_norm_status` mapping as SSE.

- [ ] **[major] `SongInfo` near-total divergence.** Pydantic (schemas.py:38-45):
  `size_mb, has_full_wav, stems, license_type, source, last_accessed`. TS (index.ts:25-38):
  `path, has_stems, has_analysis, bpm, key, camelot, duration, genre, energy, embedding, stems`.
  Only `name` + `stems` overlap; every UI analysis field (`bpm/key/camelot/genre/energy`) is absent
  from the Pydantic model.

- [ ] **[major] `LibraryStats` divergence.** Pydantic (267-272):
  `total_songs, total_size_gb, cap_gb, within_cap, songs_with_stems`. TS (40-45):
  `total_songs, indexed_songs, stems_split, total_size_mb`. Shared: `total_songs`; units differ (gb vs mb).

- [ ] **[major] `Job` field names.** Pydantic: `job_type`, `created_at` (float), `started_at`,
  `finished_at`, `eta_sec`. TS: `type`, `created_at` (ISO string), `updated_at`, `meta`.
  `started_at/finished_at/eta_sec` have no TS home; `meta/updated_at` no Pydantic home.

- [ ] **[major] `TransitionPlan` ratio name.** dj_analysis.py:206 = `tempo_shift_ratio`; index.ts:53
  expects `stretch_ratio`. TS also adds `key_compatible`, emitted by no backend model.

- [ ] **[minor] request-body drift.** TS `DJRemixRequest` (99-111) has `transition_duration, effects,
  target_bpm` the backend ignores and omits `bridge_beat_path` (file-mode bridge can't be sent).
  TS `DJPreviewRequest` omits `transition_effect`, adds `preset`/`transition_duration`.

- [ ] **[minor] response models with no TS type.** `MusicVectorResponse, QualityReportResponse,
  TransitionScoreResponse, StyleTransferResponse, InpaintResponse, TokenizeResponse,
  ModelStatusResponse`, and `SimilarSongResult.danceability/vocal_density`. AI Lab + Mix Deck consume
  these untyped.

- [ ] **[fix-all] Generate TS from OpenAPI** (`openapi-typescript`) in CI and fail on drift. This is
  the type-safety item already in CLAUDE.md.

---

## Dimension 3 — Untested code paths

Broad coverage on pure/analysis code; absent on rendering correctness, mastering, the job store,
setlist, and the entire API/task layer.

- [ ] **[major] mastering.py — zero direct tests.** `compute_lufs` (gating, silence path),
  `normalize_to_lufs`, `apply_limiter`, `detect_clipping`, `master_mix`, `analyze_stem_lufs`. Only
  `normalize_stems_to_target` is hit indirectly. Add known-signal tests (e.g. −20 dBFS sine → expected LUFS).

- [ ] **[major] setlist_planner.py — zero tests.** `transition_cost`, `_arc_target_energy`,
  `MarkovTransitionModel` (train/score/save/load), `parse_exportify_csv`, `SetlistPlanner.optimize`,
  `export_csv`, `summary`, `compute_spectral_flux_energy/rms_energy`. Flagship feature, no net.

- [ ] **[major] key_detection harmonic utils — zero tests.** `camelot_distance`, `camelot_compatible`,
  `camelot_modulation`, `pitch_shift_for_camelot` (which is wrong — D1). Existing `camelot_distance`
  tests target a *different* function in `track_metadata.py`. This is why the table bug survived.

- [ ] **[major] jobs.py — zero tests.** create/update/cancel/retry/submit, SQLite write-through,
  RUNNING→FAILED restart rollback, concurrent updates. Core infra.

- [ ] **[major] task_modules/ + routers/ — zero tests.** No `TestClient` endpoint tests;
  `smoke_e2e.sh` needs a live server and isn't in the 190. `task_instrument_lab`, generative,
  download, stems tasks all uncovered.

- [ ] **[minor] dj_engine render paths.** `render_chain`, `render_from_stems`, effect helpers
  (`_apply_echo_out/_filter_sweep/_reverb_tail`, `_apply_dynamic_eq_fade`, `_stem_similarity`) have no
  direct tests; the `render` tests assert only dtype/NaN/clip/length — which is why D1-critical is invisible.

**Priority order:** mastering known-signal tests → key_detection harmonic table tests (catches the
pitch bug) → jobs.py store tests → setlist_planner → router smoke tests. Add a `render()` test
asserting B is silent in the first half (catches the critical bug).

---

## Dimension 4 — Production readiness (API)

- [ ] **[major] main.py:56-66** — SSE broadcasts from worker threads are silently dropped.
  `_sync_emit` calls `asyncio.get_event_loop()` from inside a ThreadPoolExecutor worker (every
  `update_job` from a running task). On Python 3.12 (per `__pycache__`) that raises with no running
  loop in the thread, swallowed by bare `except: pass`. Result: `job_updated/completed/failed` during
  background jobs never reach clients; UI quietly falls back to polling. Same pattern at
  events.py:151. **Fix:** capture the loop once in `lifespan` (`asyncio.get_running_loop()`), store
  it, use `loop.call_soon_threadsafe(...)`. Verify with a test that a backgrounded job emits completion.

- [ ] **[major] jobs.py:88, 222, 238, 330…** — SQLite connections never closed.
  `with _get_conn() as conn:` manages the transaction, not the connection; a new connection is opened
  per mutation and leaked (every progress tick). **Fix:** `contextlib.closing(_get_conn())` or a
  single shared connection guarded by `_lock`.

- [ ] **[major] jobs.py:443-444** — non-cooperative cancellation ties up workers. Worker checks
  `if job_id in _cancelled` only *after* `fn()` returns; with `max_workers=2`, two "cancelled" long
  jobs run to completion and block the queue. task_modules don't poll the flag. **Fix:** cooperative
  checkpoints in long task loops; consider a larger/separate pool for renders.

- [ ] **[minor] jobs.py:324-327** — ETA `rate = progress / elapsed` can divide by zero if a progress
  update lands in the same tick as start. **Fix:** `elapsed = max(elapsed, 1e-6)`.

- [ ] **[minor] jobs.py:143** — `_cancelled` set grows unbounded (IDs never removed).

- [ ] **[minor] main.py** — no global exception handler (zero `exception_handler` registrations).
  Mutating endpoints are safe (errors land in job status), but synchronous endpoints (analysis/library/
  spotify do inline work) return unstructured 500s on non-`HTTPException`; `serve_ui` (171-175) 500s if
  `index.html` is missing. **Fix:** `@app.exception_handler(Exception)` returning a JSON envelope with
  the request ID already generated in middleware.

- [ ] **[minor] main.py:78** — shutdown `_executor.shutdown(wait=True, cancel_futures=False)` waits
  indefinitely for multi-minute jobs; SIGTERM hangs (bad for orchestration). **Fix:** bound the wait
  or `cancel_futures=True` for pending work.

---

## Suggested fix sequence

1. **dj_engine.py:597-605** (critical render bug) — affects every mix.
2. **main.py:56-66** (SSE loop capture) — restores real-time UX.
3. **key_detection.py:664-677** (Camelot table) + test — wrong pitch shifts.
4. **dj_analysis.py:653-685** (clash-path bass swap).
5. **jobs.py** connection leak + ETA guard + jobs tests.
6. Schema drift: JobStatus first, then OpenAPI→TS generation in CI.
7. Backfill mastering / setlist / key_detection / router tests.
