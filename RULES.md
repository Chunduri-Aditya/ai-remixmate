# RULES

_Conventions for code in AI RemixMate. Imported by `CLAUDE.md`, so it loads every session. One line per rule; the parentheses name the check that enforces it, or say "unenforced"._

## Commands

The gate CI runs (`.github/workflows/ci.yml`), with the local interpreter in place of CI's bare `python` and `ruff`:

```bash
# Python, from the repo root
.venv/bin/python -m pytest -m "not dj_analysis" --ignore=tests/e2e_test_suite.py -q
.venv/bin/python -m pytest -m dj_analysis -q
.venv/bin/ruff check .
# Frontend, from frontend/
npx tsc --noEmit
npx vitest run
npx eslint . --max-warnings 0
npx vite build
```

On a fresh clone `.venv` does not exist yet: create it with the setup step in `DEVELOPING.md`, "Running locally", which also installs the `dev` extra that carries pytest and ruff. One test: `.venv/bin/python -m pytest tests/test_foo.py::TestClass::test_name`. The map check: `.venv/bin/python -m pytest -q tests/test_agent_docs.py`. Recipes, markers and the rest of the tooling: `DEVELOPING.md`, "Commands".

## Approved libraries

- Backend: Python 3.12, FastAPI with Uvicorn and Pydantic v2, SQLAlchemy Core over SQLite or Postgres (psycopg), PyJWT for Supabase tokens; audio on librosa, numpy, scipy, soundfile, pyloudnorm, and Demucs on torch (CI installs `requirements-dev.txt`, which pulls in `requirements.txt`).
- Optional extras in `pyproject.toml`: `generative` (the AI Lab models) and `wordplay` (lyricsgenius, nltk); code that uses them must still import and answer without them, as the generative routes (503 behind their flag) and `scripts/core/wordplay.py` already do (unenforced).
- Frontend: React 18, TypeScript, Vite, React Query for server data, Zustand for client state, react router, wavesurfer.js, supabase js; plain CSS, one file per page or component, tokens in `frontend/src/styles/tokens.css`, no Tailwind and no CSS modules (unenforced).
- No new dependency before checking `requirements.txt`, `requirements-dev.txt`, the `pyproject.toml` extras and `frontend/package.json` for one that already does the job (unenforced).
- On the owner's machine the global stack rules also apply: `~/.claude/rules/python.md`, `~/.claude/rules/typescript.md`, `~/.claude/rules/data-ml.md` (unenforced).

## Do

- Run Python tests with `.venv/bin/python -m pytest`, never a bare `pytest`, which can resolve to another interpreter (unenforced).
- Run frontend tests with `npx vitest run` from `frontend/`; `npm test` is watch mode and never exits, and `npm run lint` lints only `src` while CI lints the whole folder (unenforced).
- Put new mixing work in the live engine: `scripts/core/dj_engine.py`, `scripts/core/dj_analysis.py`, `scripts/core/mastering.py`, or the orchestration in `scripts/api/task_modules/remix.py`, which is where engine agnostic plan changes such as spectral matching and sync live (unenforced).
- Change `config.yaml` and the dataclass in `scripts/core/config.py` together: `scripts/core/config.py::_dict_to_config` drops YAML keys the dataclass lacks, and `scripts/core/config.py::_apply_env_vars` casts env values by the YAML type (YAML keys: `tests/test_config_drift.py`; the env cast: `tests/test_render_e2e.py::test_max_song_seconds_env_override` for `remix.max_song_seconds` only, every other key unenforced).
- Add a route to a router included by `scripts/api/routes.py::router`, so it sits behind `require_user`; a public route mounts beside the legal router in `scripts/api/main.py` and reads no per user state; keep `analysis` registered before `library` (route order: `tests/test_route_shadowing.py`; placement unenforced).
- A commit that moves or renames a path or symbol cited in these docs or in `docs/agents/` updates that line in the same commit (paths and symbols: `tests/test_agent_docs.py`; the truth of each sentence: unenforced).
- A fix commit deletes its item from `docs/agents/BACKLOG.md` (unenforced; an id cited after its item is gone fails `tests/test_agent_docs.py`).
- Before any run outside pytest that could write (a live API, a script, a route call), export `REMIXMATE_LIBRARY_LIBRARY_DIR`, `REMIXMATE_LIBRARY_OUTPUTS_DIR`, `REMIXMATE_DATA_DIR`, `REMIXMATE_METADATA_CACHE_PATH`, `REMIXMATE_DATABASE_PATH` and `REMIXMATE_DATABASE_EMBEDDINGS_PATH` to a scratch copy (the last three because `scripts/core/track_metadata.py` and the `database` config keys resolve `data/...` against the project root, not `DATA_DIR`), unset `DATABASE_URL`, `REMIXMATE_LIBRARY_DIR` and `REMIXMATE_OUTPUTS_DIR`, and only then import from `scripts`, because `scripts/core/paths.py` freezes the dirs at import (unenforced; `tests/conftest.py` sets the same data and outputs variables for pytest but leaves the real `library/` and any exported `DATABASE_URL` in place, OPS-10).
- In pytest, do not export `REMIXMATE_LIBRARY_LIBRARY_DIR`: it fails `tests/test_storage_isolation.py::test_library_dir_still_points_at_the_real_catalog`. `scripts/core/library.py` binds `_DEFAULT_LIBRARY_DIR` from `paths.LIBRARY_DIR` at import, so patching `paths.LIBRARY_DIR` does not move `LibraryManager`; pass `library_dir=` to `scripts/core/library.py::LibraryManager` and patch `scripts/core/library.py::get_library_manager`, as `tests/test_storage_prune_safety.py` does (unenforced).

## Do not

- Edit `scripts/core/classic_engine/` unless the task is about the classic engine; it is a frozen copy kept for reference renders (unenforced; DOC-06 lists edits that already landed).
- Import `ai_remixmate_feature_lab/` from `scripts/`, `tests/` or `frontend/src/`; it is a read only sandbox (unenforced; ruff and pytest skip it).
- Call a destructive route against a server whose library dir is the real `library/`: `POST /library/storage/prune` (no dry run, OPS-02), `POST /library/storage/evict`, `DELETE /library/{name}`, `POST /library/delete-batch`, `POST /library/initialize`, `POST /stems/split`, `POST /stems/split-batch`, `POST /stems/compress`, `POST /stems/compress-batch`, `POST /library/mixes/sweep`, `DELETE /account`, `DELETE /admin/mix/{job_id}`, `POST /index/rebuild`, `POST /library/build-clap-index`, `POST /library/calibrate-lufs`, `POST /library/analyze-missing`, nor `python -m scripts.purge_library --apply`; with auth off every operator route is open to any local caller (partly: the Bash deny patterns in `.claude/settings.local.json` cover prune, evict, delete batch, mix sweep, initialize, compress, a curl DELETE (`-X DELETE`, `--request DELETE`) and `--apply` in the owner's Claude sessions; they match the command text only, so a DELETE sent from a Python or Node one liner, or any route called from code in a file, is not blocked; the rest unenforced).
- Treat `POST /analyze`, `GET /library/similar/{name}`, `GET /recommend/{name}` or `GET /beat/synthesize` as read only: they write shared song files, `meta.json` caches, indexes or beat files behind only the user gate (MX-32; `POST /ai/tokenize` is suspected too, MX-50; unenforced).
- Import `scripts.api.jobs`, or anything that imports it, from a second process while the API runs: the import rewrites every RUNNING and PENDING row in `jobs.db` to FAILED (OPS-08; unenforced).
- Trust or follow `archive/` docs (unenforced).
- Touch the contents of `library/`, `catalog/`, `catalog_data/`, `outputs/`, `data/` or `library_quarantine/`: live songs, the CC catalog and the stores; full mixes already lost cannot be fetched again (partly: the `rm -rf` deny in `.claude/settings.local.json`; otherwise unenforced).
- Print or quote `config.local.yaml`, the Spotify token in `data/` or the key files under `certs/` (unenforced).

## Errors

- A task fails by raising: the worker wrapper in `scripts/api/jobs.py::submit_job` stores FAILED, `error` as the exception text with absolute paths scrubbed, `message` as "Failed: " plus that text, emits `job_failed` over SSE and logs the traceback server side; a task that returns is DONE, so never return an error value (job lifecycle: `tests/test_behavioral.py`).
- Other terminal states: a request over the active job cap gets 429 from `scripts/api/routers/_helpers.py::_check_job_cap` before any row exists; only a burst that passes that check and is refused again inside `scripts/api/jobs.py::submit_job` leaves a FAILED row; a restart turns RUNNING and PENDING rows into FAILED with "Failed: process restarted"; a cancel sets CANCELLED at once, but only the chain task polls for it (unenforced).
- HTTP errors are `HTTPException`, returned as `{"detail": ...}`; validation 422s pass through `scripts/api/main.py::validation_error_without_echo`, which drops the `input` echo (unenforced).
- Status codes to expect: 400 for a song name the regex in `scripts/api/routers/_helpers.py::_require_song` rejects and 404 for a missing song; 401 with auth on and no valid token; 403 for a flagged user, a non admin on an admin or operator route, and every admin route locally; 409 for a cancel of a finished job; 429 for the job cap; 503 for a feature behind an unset flag; 500 when auth is on, the request carries a token and `SUPABASE_JWKS_URL` is unset (a tokenless request still gets 401) (auth codes: `tests/test_auth.py`, `tests/test_operator_gate.py`; the rest unenforced).
- Frontend: `frontend/src/lib/api.ts::request` throws `ApiError` with the message `[status] path: body` (status 0 for the 30 s timeout) and, in hosted mode, signs the user out on 401; React Query errors toast through `frontend/src/App.tsx::createQueryClient` except 401 and 503; COMPLETED and FAILED jobs toast through `frontend/src/hooks/useJobToasts.ts::useJobToasts`; render errors land in `RootErrorBoundary` or `PageErrorBoundary`; an error thrown from an event handler outside React Query reaches none of these, so its caller must handle it (`frontend/src/__tests__/errors.test.tsx`, `frontend/src/lib/__tests__/api.test.ts`).

## Constraints

- Hosted mode serves `catalog/` (or the mounted catalog), never `library/`, which holds YouTube sourced tracks: the API checks no licence when serving, and the ingest gate admits only CC0, public domain and CC BY (`scripts/core/license.py::PERMITTED_LICENSES`), hard only with `REMIXMATE_LICENSE_ENFORCE=1` (licence parsing: `tests/test_license.py`, `tests/test_license_text.py`; serving unenforced).
- Tenancy: every per user read and write keys on `scripts/api/auth.py::get_user_id`, renders live under `outputs/<uid>/`, and worker threads recover the user from `meta.user_id` because they do not inherit the contextvar (`tests/test_tenancy_jobs.py`, `tests/test_storage_isolation.py`).
- Renders are not permanent: the retention sweep in `scripts/core/mix_retention.py` deletes old sessions 60 s after startup and every `retention.sweep_interval_hours` (`scripts/api/main.py::_retention_sweep_loop`), and after each render, with no dry run; only the manual `POST /library/mixes/sweep` defaults to a dry run; limits come from the `retention` keys in `config.yaml` (`tests/test_mix_retention.py`).
