# AI RemixMate Changelog

All notable changes to this project, newest first.

---

## [Unreleased] Hosted app branch (`hosted-app`)

At `2cc961d` the `hosted-app` branch carries 56 commits beyond `main` (`git rev-list --count main..2cc961d`): 29 that build the hosted product in phases (`git rev-list --count 304da7e^..fb1a5f9`) and 27 from the hardening batches that followed (`git rev-list --count fb1a5f9..2cc961d`). Each line below is `git log --oneline main..2cc961d` output, grouped by area and listed oldest first within each group.

Entries further down this file describe earlier states of the project. Commit `8c4f204` removed the user download routes and the Spotify import routes (the Spotify router now serves read only metadata), and commit `f3aa1f3` dropped the download and Streamlit dependencies from the core install, so the download endpoints and the Streamlit interface named in those older entries are not current features.

### Hardening batches

#### Test isolation

- `52cd9ad test: isolate storage in every pytest run unless opted out`

#### Toolchain and CI

- `13a6d0f ci: install dev requirements, run integration and engine tests, node 22`
- `f3aa1f3 build: drop download and Streamlit deps from the core install, ruff lint config`
- `3c5d3cb build: Python 3.12 image, complete dockerignore, loopback compose`
- `e30495f chore: start.sh uses .venv, binds loopback, cleans up; archive bin/`
- `530ba73 style: ruff safe autofix`
- `df66881 chore: install pre-commit with pins that match the venv`
- `e7a053b style: resolve the remaining ruff findings by hand`

#### Dependencies

- `62c7e58 deps: security bumps for pyjwt, multipart, requests, click, anyio, msgpack, react-router-dom`
- `638779c deps: fastapi 0.141.1 with starlette 1.3.1; route table test reads fastapi's flattened routes`

#### Backend security and tenancy

- `a0d3a30 security: tenancy and hardening batch (B2)`
- `9c3fcae fix: uvicorn ignores proxy headers so the DMCA limiter keys on the socket peer`

#### Render engine and intelligence routes

- `e251f22 fix: stretch song B toward song A's tempo; end to end render test`
- `185bfc7 fix: pitch shift only when the Camelot pair is not already compatible`
- `23402fd fix: every reader resolves audio through audio_source`
- `42dd8c2 fix: surface failures that were logged at DEBUG or swallowed`

#### Frontend

- `0f4b1f3 feat: features endpoint, feature aware UI, jobs UX, vault credits, account and legal pages, contract fixes (B5)`
- `842fc16 feat: root error boundary and global request error toast`
- `d53592d fix: crate ids match the API, deck teardown swallows only AbortError, /ai-lab gated on the generative flag`

#### Hosted mode verification

- `32bceb4 feat: local JWKS and token minter for hosted mode verification`

#### Ingest safety

- `42dfd79 fix: licence text fallback matches whole phrases and denies all rights reserved`
- `664a18a fix: ingest never targets the library root, fails on Demucs failure, gates before the shortcut, dry run`

#### Deploy

- `a4e8b50 fix: deploy scripts that run, env names the code reads, drain on shutdown`

#### Security and CI

- `bd1c446 security: session cookie authenticates only GET and HEAD and is SameSite=Lax`
- `2cc961d fix: wordplay defines the nltk guard flag without nltk`

#### Docs

- `aef279f docs: move legacy docs to archive/docs/legacy`
- `03dcb8e docs: ledgers, README and runbook match the hosted-app code`

### Hosted product build

- `304da7e test: isolate outputs + data dirs for the whole suite`
- `4faac0f docs: scaffold app/ for the hosted-product layer`
- `3d06bf2 feat(license): PUBLIC_DOMAIN type + hosted-catalogue publish gate`
- `2f2833f feat(license): carry artist + CC deed URL through the library API`
- `f6c0c6c chore(config): drop download.default_source and spotify.redirect_uri`
- `2be8e01 docs: mark Phase 1 in progress, add hosted-migration status`
- `8c4f204 feat: remove user download paths, keep operator CC-catalog ingest`
- `a78e68c docs: record the Phase 1 backend teardown`
- `46aa166 test: 8-case classify_license source matrix, and fix the "archive" alias`
- `001f07c docs: tick the classify_license 8-case matrix test`
- `dbe2db1 feat: write CREDITS.txt + manifest.json credits[] next to every mix`
- `02bf0ea docs: tick mix credits (CREDITS.txt + manifest credits[])`
- `cc9de1a feat: frontend catalogue teardown; descope FMA + Musopen ingest`
- `18c4cda docs: tick Phase 1 (code); FMA/Musopen to the cut list`
- `f27aecb feat: gate generative routes behind a flag; drop unused runtime deps (Phase 2)`
- `c9f66ab docs: Phase 2 done — generative gate + slimmed runtime`
- `3b75618 docs: correct two Phase 2 claims (advisor review)`
- `653633a feat: user-scope outputs + job store via current_user contextvar`
- `4a40867 feat: user-scope crates/feedback + stem-split gate + spectral freeze (Phase 3b)`
- `d8b510b feat: Supabase RS256 JWT gate + cookie-auth SSE + uid migration (Phase 4 backend)`
- `9068435 feat: Supabase auth shell — RequireAuth, login page, Bearer + SSE cookie (Phase 4 commit 2)`
- `e4e894d refactor: SQLAlchemy Core engine for the job store (Phase 5 commit 1)`
- `cd0ccc8 refactor: SQLAlchemy Core for the crate + feedback stores (Phase 5 commit 2)`
- `b97b5d9 docs: Phase 5 done — SQLAlchemy Core, Postgres for jobs/crates/feedback`
- `093c8ab Phase 5 done: sync status table, per-crate song_count test, SHOWCASE`
- `b8317f2 Phase 6: per-user rendered-mix retention`
- `2919ec2 feat: legal pages, DMCA intake, account deletion/export, admin takedown`
- `8dbfe81 feat: production deploy — Caddy + systemd + Cloudflare Pages + CI`
- `fb1a5f9 feat: operator prep — bootstrap/backup, live_verify, catalog dry-run`

---

## [Unreleased] — Beat-quantized crossfade, job filtering, frontend health pass (historical, predates `hosted-app`)

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
  stem-type-specific timing exists; an adversarial audit disproved it in 2026-07 and `DEVELOPING.md`
  already recorded the correction. The README now describes the similarity-shaped curves.

---

## [Unreleased] — Initialize Library: dynamic live dashboard (historical, predates `hosted-app`)

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

## [Unreleased] — Hotfix: `_analyze_impl` import error on startup (historical, predates `hosted-app`)

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

## [Unreleased] — Reliability-First Modernization Pass (historical, predates `hosted-app`)

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

## API Surface Summary (after this pass) (historical, predates `hosted-app`)

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
