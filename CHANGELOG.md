# AI RemixMate — Changelog

All notable changes to this project are documented here in reverse-chronological order.

---

## [Unreleased] — Beat-quantized crossfade, job filtering, frontend health pass

### Added

- **Stepped crossfade** (`transition_effect: "stepped"`). Divides the stem crossfade into one step
  per beat (`transition_bars × 4`) instead of a continuous sweep: Song A steps down and Song B steps
  up in equal increments locked to the bar grid. Increments are equal in **power**
  (`fade_out[k] = sqrt(1 - k/(N-1))`), not amplitude — equal-amplitude steps put both curves at 0.5
  mid-transition, a −3.01 dB dip that `TestEqualPowerCrossfade` already guards against. Bypasses
  loop extension so the staircase shapes Song A's own audio rather than a tiled loop; leaves bass to
  `_stem_bass_ramp()`; `render()`/`render_chain()` treat it as `"auto"` for the exit effect so
  stem-less songs keep their echo/filter. Selectable in Mix Deck.
  New: `scripts/core/dj_engine.py::_stepped_crossfade_curves()`, `tests/test_stepped_crossfade.py`.
- **Server-side job filtering.** `GET /jobs` accepts `job_type` and `status`, applied before the
  limit. Filtered queries read through to SQLite because the in-memory dict only holds the newest
  500 rows. Both enum serializations (`dj_remix` and `JobType.DJ_REMIX`) match.
  New: `tests/test_job_list_filters.py`.
- **ESLint flat config** for the frontend. `npm run lint` had never executed — ESLint 9 was
  installed with no config file at all, so the documented `--max-warnings 0` guarantee was
  unchecked. New: `frontend/eslint.config.js`.
- `prefers-reduced-motion` support, with the nine status spinners exempted so they read as working
  rather than hung.

### Changed

- **`dj.mixing_engine` default is now `current`, was `classic`.** This moves mastering from −8 to
  −14 LUFS, so mixes render noticeably quieter than before. `classic` stays available and frozen.
- Mix Deck's effect dropdown gains `stepped`.
- Operations page header renamed from "Downloads" to "Operations" — it covers downloads, processing,
  storage and maintenance, and the rail already called it OPS.
- Input focus styles moved from `:focus` to `:focus-visible`, each with a visible ring; buttons and
  links gained a global focus ring where they previously showed none.
- Page layouts now fill the canvas. Measured trailing slack at one desktop width: signal-search
  588 → 20px, mix-deck 425 → 20px, ai-lab 415 → 20px.

### Fixed

- **Mix Vault showed "Vault is empty" over 35 completed mixes.** It fetched the unfiltered job list
  and filtered client-side, but `GET /jobs` returns only the newest 20; on a 725-job store every
  mix fell outside the window. Now requests exactly what it needs. Verified 0 → 35.
- `MixVault` carried an `eslint-disable` positioned above a comment continuation rather than the
  dependency array, so it had never applied. `logEvent` is memoized instead.
- Three component files exported non-components alongside their component, breaking fast refresh.
  Split into `toastContext.ts`, `remixOptions.ts`, `camelotKey.ts`.
- `MissionControl` passed `jobsApi.list` by reference to react-query, which hands `queryFn` a
  context object.
- README claimed drums and bass enter ahead of vocals and that "vocals are delayed". No such
  stem-type-specific timing exists; an adversarial audit disproved it in 2026-07 and `CLAUDE.md`
  already recorded the correction. The README now describes the similarity-shaped curves.

---

## [Unreleased] — Initialize Library: dynamic live dashboard

**What changed** (`scripts/ui/app.py`):

- Replaced static `st.metric` widgets with a custom-rendered `st_components.html()` block featuring:
  - **Animated stat cards** — gradient radial glow, `countUp` entrance animation, colour-coded numbers (purple → amber → green as library fills up)
  - **SVG progress rings** — circular arcs for "stems split %" and "indexed %" that animate on update; ring glows when a job is running
  - **Pipeline stage stepper** — three cards (Stem Separation → FLAC Compression → RAG Index Rebuild) with per-stage status badges (Pending / Running / Complete), pulsing border animation on the active stage, and a blinking dot indicator
  - **Live message banner** — shows the current job message string when the pipeline is running
- Wrapped the entire stats block in `@st.fragment(run_every=3)` (`_init_library_live_stats`) so numbers and stage states update **automatically every 3 seconds** without a full page reload
- Moved the pipeline options into a collapsed `st.expander` to reduce visual noise
- Updated the hero header to a gradient title with an inline "Live stats · Auto-updates every 3s" indicator
- Updated results display after completion to use the same animated card style (green-tinted finish cards)

**No API or backend changes.** Pure frontend improvement.

---

## [Unreleased] — Hotfix: `_analyze_impl` import error on startup

**Problem**: Server failed to start with `ImportError: cannot import name '_analyze_impl' from 'scripts.core.dj_engine'`.

**Root cause**: During Phase 4 module splitting, `dj_engine.py` was refactored and its internal analysis function was renamed `analyze_structure` (now living in `dj_analysis.py`). The backward-compat re-export block in `dj_engine.py` correctly re-exported `analyze_structure` and `plan_transition`, but never added an alias for `_analyze_impl`. Four callers still imported the old name:
- `scripts/api/task_modules/remix.py`
- `scripts/api/task_modules/analysis.py`
- `scripts/api/routers/analysis.py`
- `scripts/core/python_bridge.py`

**Fix** (`scripts/core/dj_engine.py`):
- Added `_analyze_impl = analyze_structure` directly after the re-export block — one line, no callers changed.

**Affected files**: `scripts/core/dj_engine.py` only.

---

## [Unreleased] — Reliability-First Modernization Pass

### Phase 1 — Deployment Model Clarity

**Problem**: The project shipped two frontends (Streamlit and static HTML) with no documentation on which was supported, causing confusion about which to use and how to deploy.

**Changes**:
- `README.md` — Declared Streamlit (port 8501) as the **primary, fully-supported** interface. Static HTML served at `/` is internal/experimental only.
- `GUIDE.md` — Added runtime directory table and frontend model section to the architecture overview.
- `start.sh`, `Dockerfile`, `docker-compose.yml` — Aligned all references to use the Streamlit-first model; added `REMIXMATE_API_URL` env var for inter-service URL resolution.

---

### Phase 2 — Runtime Inconsistency Fixes

**Problem**: Several concrete contract breaks caused silent failures and mismatched client expectations.

**Changes**:

- **`/health/live` response body** (`scripts/api/routers/system.py`):
  - Was: `{"status": "alive"}`. Now: `{"status": "ok"}` — aligns with the documented health contract.

- **Hardcoded `localhost` in Streamlit app** (`scripts/ui/app.py`):
  - `_api_host()` ignored `REMIXMATE_API_URL` entirely. Replaced with `_api_base_url()` that reads the env var first, falls back to `http://localhost:8000`.
  - Two inline `f"http://localhost:8000/outputs/..."` strings at lines ~2655 and ~4068 replaced with `f"{API_PUBLIC}/outputs/..."`.

- **Spotify import bypassing job store** (`scripts/api/routers/spotify.py`):
  - Was using raw `threading.Thread` and direct `task_download()` calls, producing no job record.
  - Rewritten to use `job_store.create_job()` + `job_store.submit_job()`, returning a proper `JobResponse`.
  - Post-download metadata writing preserved via a separate fire-and-forget background thread.

- **`task_download` missing default for `name`** (`scripts/api/task_modules/download.py`):
  - `name: Optional[str]` had no default value, causing `TypeError` when Spotify import omitted it.
  - Fixed: `name: Optional[str] = None`.

- **Stale Docker volume directories** (`Dockerfile`, `docker-compose.yml`):
  - `Dockerfile` was creating `/app/output` and `/app/mixes` (old paths). Updated to canonical paths: `/app/outputs` and `/app/models` (per `scripts/core/paths.py`).
  - `docker-compose.yml` volumes updated from `./output` + `./mixes` → `./outputs`. Health check updated from `/health` → `/health/live`.

---

### Phase 3 — Test Suite Stabilization & Deprecation Fixes

**Problem**: Tests crashed unpredictably due to librosa/numba environment differences. `@app.on_event` was deprecated in FastAPI.

**Changes**:

- **`tests/conftest.py`** (new file):
  - Added a once-at-collection-time librosa probe (`_probe_librosa()`). If librosa fails to initialize (numba cache issues, missing native libs), all tests marked `@pytest.mark.dj_analysis` are automatically skipped with a clear message instead of crashing.
  - Added `clean_env` fixture that clears all `REMIXMATE_*` environment variables before each test.

- **`tests/test_core_modules.py`**:
  - `TestDJStructureAnalysis` and `TestDJRenderer` classes marked `@pytest.mark.dj_analysis`.
  - `TestDJTransitionPlanner` left unmarked (pure dataclass logic, no librosa dependency).

- **`scripts/api/main.py`** — Replaced deprecated lifecycle hooks:
  - Removed `@app.on_event("startup")` and `@app.on_event("shutdown")`.
  - Replaced with `@asynccontextmanager async def lifespan(app)` and `FastAPI(lifespan=lifespan)`.
  - Graceful shutdown now calls `_executor.shutdown(wait=True, cancel_futures=False)` so in-flight jobs complete cleanly.

- **`README.md`** — Corrected false FAISS claim:
  - Was: "FAISS-backed embedding index". Now: "35-dimensional numpy embedding index (JSON-persisted, no FAISS dependency)".

---

### Phase 4 — Module Splitting

**Problem**: Three files had grown to 1200–1900 lines, making navigation and testing impractical.

**Changes**:

- **`scripts/api/routes.py`** (1265 lines → thin aggregator):
  - All domain routes extracted into `scripts/api/routers/` package.
  - `routes.py` kept as a backward-compatible aggregator (`router.include_router(...)` for each domain).

- **`scripts/api/routers/`** (new package — 11 files):
  - `_helpers.py` — Shared utilities: `_require_song`, `_stem_file`, `_song_info`, `_check_job_cap`.
  - `system.py` — `/health`, `/health/live`, `/health/ready`.
  - `library.py` — All `/library/*` and `/outputs/*` routes.
  - `downloads.py` — `/download`, `/download-playlist`.
  - `stems.py` — All `/stems/*` routes.
  - `analysis.py` — `/analyze`, `/compatibility`, `/recommend`, `/library/similar`, `/index/*`.
  - `remix.py` — `/dj-remix`, `/dj-remix/preview`, `/dj-chain`, `/beat/*`, `/instrument-lab`.
  - `generative.py` — All `/ai/*` routes.
  - `spotify.py` — All `/spotify/*` routes.
  - `jobs.py` — `/jobs`, `/jobs/{job_id}`, `DELETE /jobs/{job_id}` (cancel).
  - `crates.py` — All crate, tag, and favorites endpoints (see Phase 6).

- **`scripts/api/tasks.py`** (1336 lines → re-export shim):
  - All task functions extracted into `scripts/api/task_modules/` package.
  - `tasks.py` kept as a re-export shim so all existing imports continue to work.

- **`scripts/api/task_modules/`** (new package — 7 files):
  - `download.py`, `stems.py`, `remix.py`, `analysis.py`, `generative.py`, `lab.py`, `__init__.py`.

- **`scripts/core/dj_engine.py`** (1914 lines → 1057 lines):
  - Analysis and planning logic extracted to `scripts/core/dj_analysis.py`.
  - `dj_engine.py` re-exports all public names from `dj_analysis` for backward compatibility.

- **`scripts/core/dj_analysis.py`** (new file — 551 lines):
  - Dataclasses: `Beat`, `Section`, `SongStructure`, `EQPlan`, `TransitionPlan`.
  - Functions: `analyze_structure()`, `_minimal_structure()`, `_label_sections()`, `plan_transition()`.

---

### Phase 5 — SQLite-Backed Job Persistence

**Problem**: Job state lived entirely in memory. Any process restart (deploy, crash) lost all job history. There was no way to cancel or retry jobs.

**Changes**:

- **`scripts/api/jobs.py`** — Complete rewrite with write-through SQLite cache:
  - Database: `data/jobs.db` (single `jobs` table with full state schema).
  - In-memory `_jobs` dict preserved for O(1) reads. Every `update_job()` call immediately flushes to SQLite.
  - `_load_from_db()` runs at module import: restores up to 500 most recent jobs; any job recorded as `RUNNING` is rolled back to `FAILED` (process-restart semantics).
  - Thread safety: all mutations protected by `threading.Lock`.
  - **New**: `cancel_job(job_id) → bool` — marks PENDING/RUNNING jobs as cancelled. `submit_job()` checks cancellation flag at start and mid-flight.
  - **New**: `retry_job(job_id, fn, **kwargs) → Optional[str]` — clones a FAILED/CANCELLED job into a fresh job record and re-submits.

- **`scripts/api/routers/jobs.py`**:
  - Added `DELETE /jobs/{job_id}` endpoint — cancels a pending or running job, returns 409 if already complete.

---

### Phase 6 — UX Improvements

#### Remix Preview Endpoint

**New**: Fast transition-only render for audition before committing to a full mix job.

- **`scripts/api/task_modules/remix.py`** — Added `task_remix_preview()`:
  - Loads and analyses both songs, plans the transition, renders **only the crossfade window** using `DJEngine.render(..., full_output=False)`.
  - Normalizes output to −1 dBFS. Saves to `outputs/{session_id}/preview.wav`.
  - Returns full explainability data: `exit_bar_a`, `entry_bar_b`, `bpm_a/b`, `harmonic_score`, `camelot_a/b`, `tempo_ratio`, `stream_url`.

- **`scripts/api/schemas.py`** — Added `DJPreviewRequest` model (lighter than `DJRemixRequest`).

- **`scripts/api/routers/remix.py`** — Added `POST /dj-remix/preview` (202, JobResponse):
  - Validates both songs exist, creates a DJ_REMIX job with `"preview": True` in meta, submits `task_remix_preview`.

#### Crate / Tag / Favorites Library Management

**New**: Named song groups (crates), per-song tags, and favorites — all persisted to SQLite.

- **`scripts/core/crates.py`** (new file):
  - SQLite database: `data/remixmate.db`.
  - Tables: `crates`, `crate_songs`, `song_tags`, `song_favorites`.
  - Full CRUD: create/delete/rename/list crates; add/remove songs from crates; add/remove/list tags; set/unset/list favorites.

- **`scripts/api/routers/crates.py`** (new router, registered in `routes.py`):
  - **Crates**: `POST /crates`, `GET /crates`, `PATCH /crates/{id}`, `DELETE /crates/{id}`, `GET/POST /crates/{id}/songs`, `DELETE /crates/{id}/songs/{name}`.
  - **Tags**: `GET /tags`, `GET /tags/{tag}/songs`, `GET/POST /library/{name}/tags`, `DELETE /library/{name}/tags/{tag}`.
  - **Favorites**: `GET /favorites`, `POST /favorites/{name}`, `DELETE /favorites/{name}`.

---

## API Surface Summary (after this pass)

| Category | Endpoints |
|---|---|
| Health | `GET /health`, `GET /health/live`, `GET /health/ready` |
| Library | `GET /library`, `GET /library/{name}`, `DELETE /library/{name}`, `POST /library/init`, `GET /outputs/{session}/{file}` |
| Downloads | `POST /download`, `POST /download-playlist` |
| Stems | `POST /stems/split`, `POST /stems/split-batch`, `POST /stems/compress`, `POST /stems/compress-batch` |
| Analysis | `POST /analyze`, `POST /compatibility`, `GET /recommend/{name}`, `GET /library/similar/{name}`, `GET/POST /index/*` |
| Remix | `POST /dj-remix`, **`POST /dj-remix/preview`** *(new)*, `POST /dj-chain`, `GET /beat/synthesize`, `POST /beat/upload`, `POST /instrument-lab`, `GET /instrument-lab/songs` |
| Generative | `POST /ai/style-transfer`, `POST /ai/inpaint`, `POST /ai/tokenize`, `GET /ai/models` |
| Spotify | `POST /spotify/import`, `POST /spotify/import-playlist`, `GET /spotify/playlists`, `GET /spotify/auth-url`, `GET /spotify/callback` |
| Jobs | `GET /jobs`, `GET /jobs/{id}`, **`DELETE /jobs/{id}`** *(new — cancel)* |
| Crates | **All new** — see Phase 6 above |
| Tags | **All new** — see Phase 6 above |
| Favorites | **All new** — see Phase 6 above |

---

*Last updated: 2026-04-05*
