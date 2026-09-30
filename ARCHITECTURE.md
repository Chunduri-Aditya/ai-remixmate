# ARCHITECTURE

_How AI RemixMate is built, module by module. Function level detail lives in `docs/agents/UI_MAP.md` and `docs/agents/MIXING_MAP.md`; the dangerous routes and the rules live in `RULES.md`. Update in the same commit as any change to the flow, layout or stack._

## Flow

One two song remix, browser to file and back:

1. Browser: a page in `frontend/src/pages/` calls `frontend/src/lib/api.ts` through React Query, with a Bearer token in hosted mode; in dev the Vite proxy strips `/api` and forwards to the API on port 8000.
2. API: `scripts/api/main.py::app` runs the auth context, CORS and request id middleware, mounts the open `system` and `legal` routers, and mounts every other router through `scripts/api/routes.py::router` behind `require_user`.
3. Route: a router in `scripts/api/routers/` validates the body against `scripts/api/schemas.py`, checks the job cap and song names in `scripts/api/routers/_helpers.py`, then calls `scripts/api/jobs.py::create_job` and `scripts/api/jobs.py::submit_job`, which queue the task on a thread pool.
4. Job: a task in `scripts/api/task_modules/` runs in a worker thread and reports through `scripts/api/jobs.py::update_job`, which writes the job row (SQLite `jobs.db`, or Postgres) and emits an SSE frame through `scripts/api/routers/events.py::SSEBroadcaster`.
5. Engine: `scripts/api/task_modules/remix.py::_resolve_engine` picks the current or classic engine from `dj.mixing_engine` in `config.yaml`; the current path loads audio (`scripts/core/audio_source.py::load_source_audio`), analyses and plans (`scripts/core/dj_analysis.py`), moves the entry cue (`scripts/core/spectral_matching.py`), renders (`scripts/core/dj_engine.py::DJEngine`), eases the spectrum (`scripts/core/spectral_sync.py`) and masters (`scripts/core/mastering.py::master_mix`).
6. Outputs: the WAV, `CREDITS.txt` and `manifest.json` land in a session dir under `outputs/<uid>/`; the job result carries a `stream_url` served by `scripts/api/routers/library.py::stream_output`, and `scripts/core/mix_retention.py` sweeps old sessions.
7. Back in the browser: `frontend/src/hooks/useSSE.ts::useSSE` (or `frontend/src/hooks/useJobPoller.ts::useJobPoller` while SSE is down) updates `frontend/src/stores/appStore.ts::useAppStore`, and the page plays the result from the stream URL.

Songs reach the library offline, never through a route: `scripts/download.py::ingest_track` classifies the licence, downloads, splits stems with Demucs (`scripts/core/stems.py::separate_song_stems`) and analyses (`scripts/core/analysis_pipeline.py::run_song_analysis`) into one song dir.

## Stack

- Python 3.12 backend on FastAPI and Uvicorn: one process, one worker, long renders run as queued jobs with progress over SSE and polling as the fallback (`README.md`, `DEVELOPING.md` "SSE").
- Audio on librosa, numpy and scipy, stems from Demucs on PyTorch (MPS, CUDA or CPU), loudness per BS.1770 through pyloudnorm; hosted Demucs runs as an operator batch at ingest, with no GPU at request time (`app/DEPLOYMENT_PLAN.md`, "Target stack").
- Storage: song folders on disk (`library/` locally, `catalog/` for hosted mode) plus one SQLite file per store in `data/`, or Postgres through SQLAlchemy Core when `DATABASE_URL` is set, for the jobs, crates, feedback and legal stores (`scripts/core/db.py`; `README.md`, "Tech stack").
- Auth: Supabase RS256 tokens checked against the project JWKS rather than a shared HS256 secret (`app/DEPLOYMENT_PLAN.md`, Phase 4 decisions); off by default, so local use needs no account.
- Frontend: React, TypeScript and Vite, with React Query for server data and Zustand for client state, plain CSS on shared tokens, dark only (`docs/agents/UI_MAP.md`).
- Hosting target: Cloudflare Pages for the frontend and one VPS running uvicorn under systemd behind Caddy for TLS (`app/DEPLOYMENT_PLAN.md`); none of it is provisioned (`app/STATUS.md`, "Honest gaps").

## Layout

The full tree is `git ls-files`; `README.md` carries an annotated tree. Load bearing paths:

- `scripts/api/`: the FastAPI app: `main.py` (wiring, lifespan), `auth.py` (users and gates), `jobs.py` (job store, executor, SSE emit), `schemas.py`, `routers/` (one file per area), `task_modules/` (job bodies: remix, analysis, stems, generative, lab).
- `scripts/core/`: engine and stores: analysis (`dj_analysis.py`, `analysis_pipeline.py`, `key_detection.py`), render (`dj_engine.py`, `spectral_matching.py`, `spectral_sync.py`, `spectral_mixing.py`), `mastering.py`, `stems.py`, `library.py`, `setlist_planner.py`, `recommend.py`, the stores (`db.py`, `crates.py`, `feedback_store.py`, `legal_store.py`), `config.py` and `paths.py`.
- `scripts/core/classic_engine/`: the frozen classic engine, selected when `dj.mixing_engine` is `classic`, missing or unreadable (the dataclass default is `classic`; `config.yaml` sets `current`).
- `scripts/download.py`, `scripts/curate_catalog.py`, `scripts/purge_library.py`, `scripts/remixmate_cli.py`: operator CLIs for ingest, catalog curation, cleanup and the old bridge; `scripts/ui/` is the retired Streamlit UI.
- `frontend/src/`: `pages/`, `shell/` (routing frame, auth), `lib/` (API client, Supabase, player engine), `hooks/`, `stores/`, `styles/`, `types/`.
- `tests/`: the pytest suite, with storage isolation in `tests/conftest.py`; frontend tests sit in `__tests__` folders under `frontend/src/`.
- `config.yaml`: base config, overridden by `config.local.yaml`, then by `REMIXMATE_<SECTION>_<KEY>` env vars (`DEVELOPING.md`, "Running locally").
- `app/`: hosted mode ledger and plan; `app/deploy/` holds the systemd unit, Caddyfile, env template, bootstrap and backup scripts and `app/deploy/PROVISION.md`.
- Not product: `archive/` (history), `ai_remixmate_feature_lab/` (sandbox, imported by nothing).
- Local only data: `library/` (live songs), `catalog/` (CC seed catalog), `catalog_data/` (stores from the hosted checks), `data/` (local stores and indexes), `outputs/` (renders), `models/` (weights).

## Boundaries

- Auth mode: `REMIXMATE_AUTH_ENABLED` unset means every request runs as user `local` with no token; set, `scripts/api/auth.py::require_user` needs a valid Supabase token. The Bearer header works on any method, the `sb-access-token` cookie only on GET and HEAD. The frontend picks its own mode at build time from `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`, and nothing checks the two agree.
- Gates: `scripts/api/auth.py::require_admin` needs an email in `ADMIN_EMAILS` and always refuses with auth off; `scripts/api/auth.py::require_operator` equals admin with auth on and admits every local caller with auth off (OPS-03). Feature flags (`REMIXMATE_STEM_SPLIT_ENABLED`, `REMIXMATE_GENERATIVE_ENABLED`) turn routes into 503s; `GET /features` reports them and the UI hides controls, but the server gate is the enforcement.
- Tenancy: the middleware sets the `current_user` contextvar per request; jobs carry `meta.user_id` into worker threads; renders sit under `outputs/<uid>/`; SSE job frames go only to the owner.
- Untrusted input: JSON bodies and query params (Pydantic; some free strings are not validated, MX-07); song names in paths (`scripts/api/routers/_helpers.py::_require_song`); uploads through `POST /beat/upload` and `POST /setlist/import-csv`; the open DMCA form at `POST /legal/dmca`, rate limited per client IP, with `X-Forwarded-For` trusted only under `REMIXMATE_TRUST_PROXY=1`; bearer tokens and cookies.
- Ingest input: remote audio and licence text from ccMixter and the Internet Archive enter through `scripts/download.py` (operator CLI) and are classified by `scripts/core/license.py::classify_license`; the operator supplied name is not sanitized (OPS-05).
- External services: Supabase (JWKS, and user deletion on `DELETE /account`), Postgres, the Spotify API (`scripts/api/routers/spotify.py`), Genius through lyricsgenius (`scripts/core/wordplay.py`), Hugging Face model downloads for the generative stack (`scripts/core/model_manager.py`), and ccMixter and the Internet Archive at ingest.
