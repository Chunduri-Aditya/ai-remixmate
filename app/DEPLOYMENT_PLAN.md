# Deploy ai-remixmate as a multi-user hosted app (shared-catalog model)

> Canonical roadmap for the `hosted-app` branch. One phase at a time, top to
> bottom. A phase is **done** only when every box in its **Verify** block is
> ticked against real output. Update the Status table and commit at each phase
> boundary.

## Status

| # | Phase | State | Commit |
|---|---|---|---|
| 0 | Guardrails (test isolation fixture) | ☑ verified | 304da7e |
| 1 | Catalog teardown + enforced CC gate | ☑ verified (code; live ingest pending an operator run) | 3d06bf2, 2f2833f, f6c0c6c, 8c4f204, 46aa166, dbe2db1, + frontend teardown |
| 2 | Guard generative + slim the image | ☐ not started | |
| 3 | User contextvar + per-user mixes/jobs/crates/feedback | ☐ not started | |
| 4 | Auth + tenancy (Supabase) | ☐ not started | |
| 5 | SQLite → Postgres (jobs, crates, feedback, setlists) | ☐ not started | |
| 6 | Per-user mix retention | ☐ not started | |
| 7 | Deploy / infra | ☐ not started | |
| 8 | Legal / product surface | ☐ not started | |

Legend: ☐ not started · ◐ in progress · ☑ verified

**Phase 1 note:** all code is committed and the frontend/backend green bar passes.
Two Verify lines need a live network ingest run and are marked
`operator-pending` below — they get ticked on the first real catalogue fill, not
by a code change. Phase 3 does not depend on them.

**Green bar carried between every phase** (all three must pass before ticking a phase):
- `uvx ruff check <files changed this phase>` — clean (this is what "clean (4 files)"
  in the Phase 0 verify meant). **Repo-wide `uvx ruff check` is not a usable gate as
  it stands**: on a clean `HEAD` it reports ~1583 findings, dominated by pre-existing
  `W293` blank-line whitespace (~1100), `I001` import order (~200), and `F401` unused
  imports (~150) — debt this migration did not introduce. `E501` is correctly
  ignored, so the `pyproject.toml` rules are being read. Scope the check to the
  phase's own files until someone runs `ruff check --fix` repo-wide and moves the
  config to `[tool.ruff.lint]`. That cleanup is worth doing before the Phase 7 CI
  job — surfaced, not done here.
- safe pytest subset — green. The command (repo `.venv`, py3.12):
  ```
  .venv/bin/python -m pytest -m "not dj_analysis and not integration" \
    --ignore=tests/e2e_test_suite.py --ignore=tests/test_audio_source.py -q
  ```
  Baseline before Phase 0: **675 passed, 90 deselected, ~13s**. After Phase 0: **680
  passed**. After the Phase 1 licence slices + `config.yaml` alignment: **702 passed,
  90 deselected**. After the backend teardown: **701** — the −1 is a single
  parametrised case in `test_structured_logger_usage.py` that iterates source files,
  now one file shorter (`task_modules/download.py` deleted), not a lost assertion.
  After the licence matrix + mix-credits tests: **706 passed**.
- `cd frontend && npx vite build` — clean

---

## Context

Today ai-remixmate is a single-user local app: FastAPI single-process, in-process
`ThreadPoolExecutor(2)` + SQLite job store (`api.workers: 1` is load-bearing), a
69 GB library on plain filesystem, **no auth on any endpoint**, and `yt-dlp`
pulling commercial YouTube audio as a first-class feature. Its own docs say it is
"not appropriate as-is for a production service." The intended "deployment" today
is a static React build on GitHub Pages that calls `http://localhost:8000` on the
user's own machine.

Target: a **multi-user product**, catalog **cut to licensed CC audio only**,
hosting budget **~$10-30/mo**, generative AI features **out for v1**, music from
**one operator-curated shared catalog** (not per-user libraries).

**Why shared-catalog changes everything**
- Per-user storage math is unworkable: ~2 GB/user → ~16 songs after stem
  separation, too thin for key/BPM matching, ~1.5 h of CPU-Demucs to get there.
- Jamendo's API terms don't fit (verified): "non-commercial uses" only and
  "applications must not be specifically designed to cache the content" — but
  download→store→separate-stems is caching by design. Same error class as yt-dlp.
  Sources become **ccMixter, Free Music Archive, Internet Archive netlabels,
  Musopen** (CC-BY / CC0 / public domain — both license *and* platform terms
  intend download + derivatives).
- A shared catalog keeps `library/` a single global read-only tree (no per-user FS
  namespacing), keeps the *search* index global, and runs Demucs **once per track
  at operator ingest** — off the request path. This removes the three riskiest
  pieces of a per-user migration.

**Remaining tension:** multi-user at $10-30/mo is a **low-traffic private beta**,
not open signup. Keep the catalog small for v1 (~150 tracks, FLAC stems ≈ 35 GB),
waitlist signups.

**Target stack**

| Layer | Choice | Cost |
|---|---|---|
| Frontend | Cloudflare Pages (SPA, custom domain) | free |
| API | 1x Hetzner CX22 (2 vCPU / 4 GB) — bare `uvicorn` + `systemd` + `Caddy` TLS | ~€4.5/mo |
| Shared catalog storage | ~60 GB Hetzner volume, mounted read-only at the API | ~€3/mo |
| Auth + DB | Supabase free tier (Auth + Postgres 500 MB) | free |
| Per-user mixes (`outputs/`) | VPS disk + TTL/cap | included |
| Demucs | operator batch at ingest (local or any GPU box) — **no runtime GPU** | free |
| Backups | nightly `pg_dump` + `rclone` to Backblaze/R2 | ~free |

**Feature flags** (each phase stays inert until its flag flips):
`REMIXMATE_LICENSE_ENFORCE` (P1) · `REMIXMATE_GENERATIVE_ENABLED` (P2) ·
`REMIXMATE_SPECTRAL_LEARNING` (P3, `false` in prod) · `REMIXMATE_AUTH_ENABLED` (P4) ·
`DATABASE_URL` dialect (P5).
Default `user_id = "local"` keeps single-user behaviour byte-identical until P4.

---

## Phase 0 — Guardrails  ✅ done

**Goal:** a test fixture that isolates per-user paths, so every later phase has a
clean verify harness.

**Tasks**
- [x] `tests/conftest.py` storage-isolation shim: sets, to a session
      `TemporaryDirectory` **before the first `import scripts.*`** —
      `REMIXMATE_LIBRARY_OUTPUTS_DIR`, `REMIXMATE_DATA_DIR`, and (for the stores
      that bypass `paths.DATA_DIR` and hold their path as a config string)
      `REMIXMATE_METADATA_CACHE_PATH` / `REMIXMATE_DATABASE_PATH` /
      `REMIXMATE_DATABASE_EMBEDDINGS_PATH`. paths.py freezes its constants at
      import and consumers re-bind one level deeper (`jobs._DB_PATH = DATA_DIR /
      "jobs.db"`), so an after-the-fact reassign is useless. `LIBRARY_DIR` left
      global. A caller that pre-sets `REMIXMATE_LIBRARY_OUTPUTS_DIR` opts out.
- [x] `scripts/core/paths.py`: added the `REMIXMATE_DATA_DIR` override (it had
      none — only `library_dir` / `outputs_dir` had config/env hooks) via the
      existing `_resolve_dir()`; `import os`; header comment.
- [x] `scripts/core/audit.py`: `_AUDIT_PATH` was a bare CWD-relative
      `Path("data/audit.jsonl")` (broke when run from anywhere but the repo
      root) — now `DATA_DIR / "audit.jsonl"`.
- [x] Recorded the safe-subset command + baseline above.

**Verify**
- [x] Safe pytest subset green with the shim active — **680 passed** (675 + 5
      new), 90 deselected, no regressions, same 3 pre-existing warnings.
- [x] `tests/test_storage_isolation.py` (5 tests): `OUTPUTS_DIR` / `DATA_DIR` /
      `EMBEDDINGS_FILE` resolve outside the repo; `LIBRARY_DIR` still
      `<repo>/library`; `jobs`/`crates`/`feedback_store` `_DB_PATH` outside the
      repo; `cfg.metadata.cache_path` / `cfg.database.path` /
      `cfg.database.embeddings_path` / `audit._AUDIT_PATH` outside the repo; a
      real write under `OUTPUTS_DIR` lands outside the repo.
- [x] Test-first: the `cfg.*` / `audit` assertion was added and watched go **red**
      (`cfg.metadata.cache_path='data/metadata.db'` inside the repo) before the
      shim lines + audit.py fix turned it green.
- [x] Negative control: with all six `REMIXMATE_*` path vars forced to the repo
      dirs, 4/5 tests go red (the `LIBRARY_DIR` one stays green) — every guard
      can fail.
- [x] `find data outputs -newermt '-3 minutes'` after a full run → empty.
- [x] Green bar: `uvx ruff check` clean (4 files) · pytest 680 green · `vite build` ✓ 1.1s.

**Files:** `tests/conftest.py`, `tests/test_storage_isolation.py` (new),
`scripts/core/paths.py`, `scripts/core/audit.py`.

---

## Phase 1 — Catalog teardown + enforced CC gate

**Goal:** replace all download paths with an operator-only ingest tool over
CC-BY / CC0 / public-domain sources; every track in the catalog provably
publishable; attribution carried into the API response and every mix.

### Current state (as of 2026-08-31)

**Licence model + attribution — committed** (`3d06bf2`, `2f2833f`):
- [x] `LicenseType.PUBLIC_DOMAIN` added; `PERMITTED_LICENSES = {CC0, CC_BY,
      PUBLIC_DOMAIN}` frozenset; `is_publishable(info)`; `license_enforce_enabled()`
      reads `REMIXMATE_LICENSE_ENFORCE`.
- [x] `classify_license` extended with `ccmixter` / `fma` / `internetarchive` /
      `musopen` source branches.
- [x] `save_license()` persists `artist_name`, `license_ccurl`, source URL.
- [x] `SongInfo` gains `artist_name` + `license_ccurl`; `_song_info` reads both.

**Config alignment — committed** (`f6c0c6c`):
- [x] `scripts/core/config.py` + `config.yaml` (edited together — drift guard
      green): dropped `download.default_source` and `spotify.redirect_uri`.

**Backend teardown — committed** (`<this commit>`; import smoke checks on
`scripts.api.main` / `.routes` / `.tasks` / `scripts.download` all pass; safe
subset 701 passed — the −1 vs 702 is one parametrised lint case over the deleted
`task_modules/download.py`, not a lost assertion):
- [x] `git rm scripts/api/routers/downloads.py`,
      `scripts/api/task_modules/download.py`, `scripts/core/youtube_music.py`.
      Nothing imported them — the working-tree `routes.py` / `tasks.py` /
      `python_bridge.py` edits had already unwired all three.
- [x] `scripts/download.py` rewritten as the operator ingest CLI. ccMixter and
      Internet Archive fetchers **implemented**; Free Music Archive and Musopen are
      **stubs** (`# API unverified` in the docstring) — 2 of 4 sources real.
- [x] `scripts/api/routers/spotify.py` cut to metadata-only: `/spotify/status`,
      `/spotify/search`, `/spotify/track/{id}`. OAuth / callback / disconnect /
      playlists / top / import routes deleted (~6 endpoints, 252 → ~90 lines).
- [x] `scripts/api/routes.py`: `downloads` router no longer included.
- [x] `scripts/api/tasks.py`: `task_download` / `task_playlist_download`
      re-exports removed.
- [x] `scripts/core/python_bridge.py`: removed the `youtube_music` online
      auto-fetch of missing tracks.
- [x] `requirements.txt`: dropped `yt-dlp` and `ytmusicapi`; replaced the LEGAL
      NOTE block with a MUSIC SOURCING note. **`mutagen` kept** — still used by
      `cue_export.py` (Serato GEOB) and `library.py`; the task list below is
      wrong on this.
- [x] `scripts/core/analysis_pipeline.py`: stale `task_download` docstring
      reference updated to `scripts/download.py`.

**Phase 1 — additive work (all done):**
- [x] FMA + Musopen **descoped for v1** — fetchers removed from
      `scripts/download.py` (`_FETCHERS` now `ccmixter` / `archive`), docstring +
      cut list updated. `classify_license` keeps both source branches (tested), so
      re-adding a fetcher later needs no classifier change.
- [x] `classify_license` 8-case allow/deny matrix test — `tests/test_license.py::
      test_source_licence_matrix` (`46aa166`). One CC-BY + one CC-BY-NC per source,
      source-shaped inputs, asserts type **and** the `is_publishable` gate per
      cell. Proven to fail under 3 mutations (accept-NC, reject-BY, drop the
      `archive` alias — the latter fix shipped in the same commit).
- [x] `scripts/api/task_modules/remix.py`: `CREDITS.txt` + `manifest.json`
      `credits[]` — `dbe2db1`. `_write_mix_credits(out_dir, songs)` after the
      `sf.write` in `task_dj_remix` and `task_dj_chain`; per-track entry from
      `license.json` (artist / licence / deed URL / source), also returned under
      a `credits` key. A track with no readable licence still gets an
      `unknown`/`None` entry (never dropped). `task_remix_preview` excluded
      (throwaway audition). Guard: `tests/test_mix_credits.py`, mutation-tested.
- [x] Frontend teardown (frontend green bar: `tsc --noEmit` clean, `vitest run`
      35 passed, `vite build` clean):
      - `lib/api.ts`: `downloadApi` (single / batch / playlist / fromSpotify)
        removed. **No `catalogApi.search`** — no backend catalogue-search
        endpoint exists; `libraryApi.list` / `.get` already browse the catalogue.
      - `pages/Operations.tsx`: the single/batch/playlist download form, live
        download queue, `DownloadJobCard` / `DownloadTimer`, and the YouTube ToS
        copy are gone (672 → 329 lines). Keeps ProcessingQueue / Storage /
        Maintenance panels; a note points at `python -m scripts.download`.
      - `__tests__/smoke.test.ts`: `downloadApi` mock dropped; `/operations`
        route still renders.
      - `types/index.ts`: `SongInfo` gains `artist_name` / `license_type` /
        `license_ccurl` (backend already sends them).
      - `pages/LibraryAtlas.tsx`: licence badge + `by <artist> · licence` line
        in the song detail drawer.
      - Spotify *import* buttons in `MissionControl` / `MixVault` / `SetBuilder`:
        none existed — the plan's task list was stale (those `Download` icons are
        mix / setlist export, kept).

**Follow-ups noted, deferred** (not teardown, not blocking):
- `scripts/core/spotify.py` still carries dead OAuth methods (`get_auth_url`,
  `exchange_code`, `disconnect`, `is_connected`) and `spotify_token.json`
  persistence. Nothing live calls them. Slim in a later pass.
- `scripts/ui/app.py` (Streamlit, not deployed) still references `/download` and
  `/download-playlist` — dies with the `ui` service in Phase 2.
- `scripts/core/library.py:87` `_is_downloaded_track` reads `full.info.json`
  (a `yt-dlp` artefact). Library-hygiene helper; leave until it actually bites.
- `JobType.DOWNLOAD` kept in `schemas.py` — historical `jobs.db` rows still
  deserialise through it (matches the Phase 2 "keep job types, make unreachable"
  pattern).

**Tasks**
- [ ] `scripts/download.py` → operator CLI ingest tool. Delete the `yt_dlp` /
      `ytmusicapi` / SoundCloud branches (`search_track` L235, `download_track`
      L347, `download_playlist` L538, `_resolve_url`, `_source_from_url`).
- [ ] Add per-source fetchers: ccMixter, Free Music Archive, Internet Archive
      netlabels, Musopen. Each returns audio URL + raw license field + artist +
      source URL.
- [ ] **Extend `classify_license`** (`scripts/core/license.py`) to recognise each
      source's license expression: ccMixter query-API `license` field, FMA
      embedded license URL, Internet Archive item-metadata `licenseurl`, Musopen
      (mostly public domain, often no per-track field → map to `PUBLIC_DOMAIN`).
      Today it only knows `youtube_tos` / `jamendo` / a CC-URL parser; an
      unrecognised field returns `UNKNOWN` and the gate rejects it → the failure
      mode is a catalog that won't fill, not a leak.
- [ ] Ingest pipeline: fetch → `classify_license` → **gate** → Demucs
      (`scripts/core/stems.py`, run here) → `run_song_analysis` → write
      `meta.json` / `analysis.json` / `license.json` → register in catalog index.
- [ ] `scripts/api/routes.py`: remove the `downloads` router (or reduce to an
      `ADMIN_EMAILS`-gated `/admin/catalog/ingest`). Delete
      `scripts/api/task_modules/download.py`.
- [ ] `scripts/api/routers/spotify.py`: delete OAuth / import / playlist / top
      routes. Keep `/spotify/status`, `/spotify/search`, `/spotify/track/{id}`
      (app-credentials metadata only). Delete `spotify_token.json` handling —
      client-credentials token in memory only.
- [ ] `scripts/core/license.py`: confirm `LicenseType` has a `PUBLIC_DOMAIN`
      member (**add it if missing** — Musopen is public domain, not CC0).
      `PERMITTED = {CC0, CC_BY, PUBLIC_DOMAIN}` — **not** CC-BY-SA (viral
      share-alike infects the mix), **not** `*_NC` / `*_ND`. Add
      `is_publishable(info)`. Ingest aborts + cleans the song dir on a
      non-permitted license. Guard the gate behind `REMIXMATE_LICENSE_ENFORCE`
      (on for ingest/deploy, **off locally** so the existing dev library loads).
- [ ] `save_license()` (L245): persist `artist_name`, `license_ccurl`, source URL.
- [ ] **Extend `SongInfo`** (`scripts/api/schemas.py`) with `artist_name` and
      `license_ccurl`; read both in `_song_info` (`scripts/api/routers/_helpers.py:47-60`,
      which today extracts only `license_type` / `source`). The P1 frontend
      attribution line and the P8 mix credits both need these in the response.
- [x] `scripts/api/task_modules/remix.py`: after the `sf.write` in `task_dj_remix`
      and `task_dj_chain`, `_write_mix_credits()` writes `<out_dir>/CREDITS.txt`
      and a `credits[]` array in `<out_dir>/manifest.json` from each source song's
      `license.json` (`dbe2db1`). Preview render excluded.
- [x] `requirements.txt`: removed `yt-dlp`, `ytmusicapi` and the LEGAL NOTE block.
      **`mutagen` stays** — `cue_export.py` (Serato GEOB) and `library.py` both
      import it. The earlier "(verify unused)" note was wrong.
- [x] `config.yaml` + `scripts/core/config.py` **together** (`test_config_drift.py`
      green): dropped `download.default_source` and `spotify.redirect_uri`.
- [x] `frontend/src/pages/Operations.tsx` — download/playlist form + queue +
      `DownloadJobCard`/`DownloadTimer` + YouTube ToS copy removed; keeps the
      processing / storage / maintenance panels (672 → 329 lines).
- [x] `frontend/src/lib/api.ts`: `downloadApi.*` removed. `catalogApi.search` was
      **not** added — no backend endpoint; `libraryApi.list`/`.get` browse the
      catalogue.
- [x] Spotify *import* buttons: none present in `MissionControl` / `MixVault` /
      `SetBuilder` (task list was stale). Spotify *search* untouched.
- [x] License badge + `by <artist> · licence` line in `LibraryAtlas.tsx` song
      detail drawer; `SongInfo` TS type extended.

**Verify**
- [x] **`classify_license` matrix** — `tests/test_license.py::test_source_licence_matrix`,
      8 cells (CC-BY allow + CC-BY-NC deny per source), source-shaped inputs,
      asserts enum **and** gate per cell. Mutation-tested (accept-NC / reject-BY /
      drop `archive` alias all go red).
- [x] One known Musopen track → `classify_license` returns `PUBLIC_DOMAIN` and
      `is_publishable` is true — `test_musopen_without_any_licence_field_is_public_domain`.
- [ ] **operator-pending** — ingest a real ccMixter CC-BY track → `license.json`
      written, `is_publishable` true. Ticks on the first real catalogue fill.
- [ ] **operator-pending** — feed the ingest tool a CC-BY-NC track → aborts, song
      dir removed. (Gate logic itself is covered by the `classify_license` matrix +
      `is_publishable`; this line is the live end-to-end.)
- [x] `GET /library/<name>` response carries `artist_name` and `license_ccurl` —
      `tests/test_song_info_attribution.py` (3 passed).
- [x] Render a mix from two ingested tracks → `<out_dir>/CREDITS.txt` exists and
      `manifest.json` has a non-empty `credits[]` with artist + ccurl —
      `tests/test_mix_credits.py` (helper-level; a full render is outside the safe
      subset). Mutation-tested: dropping a missing-licence track, nulling the
      artist, and un-wiring `task_dj_chain` each turn it red.
- [x] `POST /download` / `/download-batch` / `/download-playlist` → 404 (router
      gone); `GET /library` → 200 (checked against a `TestClient`).
- [x] `tests/test_config_drift.py` green (26 passed); no runtime `yt_dlp` /
      `ytmusicapi` imports in `scripts/` (only legacy source-name string literals
      in `license.py`).
- [x] Green bar: safe subset 706 passed · `uvx ruff` clean on touched files ·
      frontend `tsc --noEmit` clean, `vitest run` 35 passed, `vite build` clean.

**Commits:** `f6c0c6c` (config) · `8c4f204` (backend teardown) · `46aa166`
(licence matrix + `archive` alias) · `dbe2db1` (mix credits) · frontend teardown +
FMA/Musopen descope (this batch). Original licence slices: `3d06bf2`, `2f2833f`.
**Rollback note:** large surface. The old `downloads` router / download UI live in
git history (`git show 2f2833f^:scripts/api/routers/downloads.py`).

---

## Phase 2 — Guard generative + slim the image

**Goal:** generative routes return 503 unless explicitly enabled; runtime image
drops the heavy ML deps it no longer needs; `demucs` leaves the request path.

**Tasks**
- [ ] `scripts/api/routers/generative.py`:
      `APIRouter(dependencies=[Depends(_generative_enabled)])` → 503 unless
      `REMIXMATE_GENERATIVE_ENABLED=true`. Keep job types defined but unreachable.
      **Do not touch** `dj_engine` / `classic_engine` / `_resolve_engine`.
- [ ] `requirements.txt`: drop `transformers`, `safetensors` (verify vs the demucs
      path), `streamlit`, `altair`, `lyricsgenius` + `nltk` (verify: setlist
      wordplay only). Move `demucs` to a `[project.optional-dependencies] ingest`
      extra — the runtime API no longer imports it.
- [ ] Drop the Streamlit `ui` service from `Dockerfile` and `docker-compose.yml`.
- [ ] `scripts/api/main.py:159-181`: replace the static
      `scripts/ui/static/index.html` at `/` and `/app` with a 302 to the frontend domain.

**Verify**
- [ ] API boots with `REMIXMATE_GENERATIVE_ENABLED` unset.
- [ ] `POST /style-transfer` → 503; `POST /dj-remix` and `POST /dj-chain` unaffected.
- [ ] `grep -rn "import demucs\|from demucs" scripts/api/` → no hits (request path
      is demucs-free; ingest tool keeps it).
- [ ] `pip check` clean after dep removal; fresh `pip install -r requirements.txt`
      in a scratch venv succeeds without torch-transformers.
- [ ] `GET /` → 302 to the frontend domain.
- [ ] Green bar.

**Commit:** `chore: guard generative routes behind a flag, drop unused runtime deps`

---

## Phase 3 — User contextvar + per-user mixes / jobs / crates / feedback

**Goal:** every write path is user-scoped, default `"local"`, single-user
behaviour byte-identical. No auth, no Postgres yet.

**Tasks**
- [ ] New `scripts/api/auth.py`: `current_user: ContextVar[str]` (default
      `"local"`), `get_user_id()`.
- [ ] `scripts/core/paths.py`: `user_outputs_dir() -> OUTPUTS_DIR / get_user_id()`;
      `session_dir()` routes through it. **`LIBRARY_DIR` untouched.**
- [ ] `scripts/api/jobs.py` — **the risky one**: `create_job()` stamps
      `meta["user_id"] = get_user_id()`; `_wrapper()` calls
      `current_user.set(job["meta"]["user_id"])` next to `set_job_id()`
      (`ThreadPoolExecutor` does not copy context). `list_jobs` / `_query_db`
      filter by user.
- [ ] `_check_job_cap` (`_helpers.py:150`) counts per-user.
- [ ] `scripts/core/crates.py` + `scripts/core/feedback_store.py` + setlist store:
      `user_id` column on every table in `_SCHEMA`; every function derives
      `user_id` from `get_user_id()` and adds `WHERE user_id = ?`.
- [ ] `audit.jsonl` (`scripts/core/audit.py`): one file, add a `user_id` field per record.
- [ ] **Genuinely global, no change** (read-only *search* over the shared catalog):
      `music_index.py` semantic index, `song_embeddings.json`, `crate_digger.py`
      CLAP index.
- [ ] **Freeze in prod, do not leave learning global** — `spectral_matching.py`
      learned weights train on 👍/👎 mix feedback, so a global model = user B's
      feedback reshapes user A's EQ (cross-tenant mutation via a feedback channel).
      Add `REMIXMATE_SPECTRAL_LEARNING` (default `true` locally, `false` in prod):
      when off, load the committed seed and ignore writes. Check whether
      `mastering.py` `stem_lufs_targets.json` is also feedback-written; if so, gate
      it the same way. Per-user personalisation stays deferred (cut list).

**Verify**
- [ ] Full suite with the P0 fixture — `"local"` behaviour byte-identical.
- [ ] **Allow test**: a test sets `current_user.set("u2")`, creates a crate + runs
      a render, then reads *its own* crate and mix back successfully.
- [ ] **Deny test**: with data owned by `u1`, `current_user.set("u2")` sees an
      empty `/crates`, `/jobs`, and no `u1` files under `outputs/`.
- [ ] **Contextvar tripwire**: submit a job as `u1`; assert the mix lands under
      `outputs/u1/`; the test **fails** if it lands under `outputs/local/` or `outputs/u2/`.
- [ ] With `REMIXMATE_SPECTRAL_LEARNING=false`, logging a 👎 does not change
      `spectral_match_model.json` on disk.
- [ ] Green bar.

**Commit:** `feat: user-scope outputs, jobs, crates, feedback via contextvar`
**Rollback note:** snapshot `data/` before running any local data through the new paths.

---

## Phase 4 — Auth + tenancy (Supabase)

**Goal:** unauthenticated requests get 401; two accounts are fully isolated except
the shared catalog.

**Tasks**
- [ ] `scripts/api/auth.py`: `require_user` dependency verifies the Supabase JWT
      (JWKS or project secret), sets `current_user`, else 401. Gated by
      `REMIXMATE_AUTH_ENABLED` (off locally → `"local"` fallback).
- [ ] `scripts/api/main.py:170`:
      `app.include_router(router, dependencies=[Depends(require_user)])`. Mount
      `system.router` (`/health*`) **separately, before that line**, so probes stay open.
- [ ] SSE: `/events/stream` authenticates via an **httpOnly secure cookie**
      (Supabase SSR cookie flow; `allow_credentials=True` already set at
      `main.py:130`). `EventSource` cannot send `Authorization`.
- [ ] `frontend/src/lib/api.ts:56`: add `credentials: 'include'` to every request.
- [ ] Frontend: `@supabase/supabase-js`; `frontend/src/shell/` gets `<RequireAuth>`
      + a login/signup page. EventSource hook unchanged with cookie auth.
- [ ] One-time migration script (own data, snapshot first): move `outputs/*` →
      `outputs/<your_uid>/`; stamp existing `jobs.db` / `remixmate.db` (crates) /
      `feedback.db` rows with `<your_uid>`. Catalog (`library/`) not namespaced.

**Verify**
- [ ] `REMIXMATE_AUTH_ENABLED=true`: unauthenticated request → 401; `/health/ready` → 200.
- [ ] Two Supabase accounts: `/library` identical (shared catalog); `/jobs`,
      `/crates`, `/feedback/summary`, `/outputs/*` disjoint.
- [ ] SSE stream connects with the session cookie, rejects without it.
- [ ] `REMIXMATE_AUTH_ENABLED` unset: full local suite still green.
- [ ] Green bar.

**Commit:** `feat: Supabase JWT auth gate + cookie-authenticated SSE`

---

## Phase 5 — SQLite → Postgres (jobs, crates, feedback, setlists)

**Goal:** the four per-user stores run on managed Postgres in prod, SQLite in
tests/local — both paths exercised.

**Tasks**
- [ ] New `scripts/core/db.py`: SQLAlchemy Core `Engine` from `DATABASE_URL`
      (`sqlite:///…` local + CI, `postgresql+psycopg://…` prod). Routers stay sync `def`.
- [ ] Replace the single connection helper in `jobs.py:83`, `crates.py:75`,
      `feedback_store.py:84` (+ setlist store) with `engine.begin()`.
- [ ] Wrap SQL in `sqlalchemy.text()`, convert `?` → `:name` (mechanical).
      `INSERT … ON CONFLICT DO UPDATE` valid in both dialects; guard
      `PRAGMA foreign_keys` to SQLite only.
- [ ] Document "verified deferred": `track_metadata.py` `MetadataCache` (global
      cache) and `scripts/core/database.py` `RemixMateDatabase` (CLI-only, not
      imported by the API) stay on SQLite.

**Verify**
- [ ] `DATABASE_URL=sqlite://…` → full suite green.
- [ ] Against a real Postgres (Supabase branch or local): job
      create/poll/list/cancel/retry, crate CRUD, feedback log/summarize all pass.
- [ ] Restart the process mid-job → history survives; RUNNING→FAILED rollback still fires.
- [ ] Green bar.

**Commit:** `refactor: SQLAlchemy Core engine, Postgres for jobs/crates/feedback/setlists`

---

## Phase 6 — Per-user mix retention

**Goal:** `outputs/` cannot grow unbounded; the shared catalog is a fixed asset.

**Tasks**
- [ ] Add a TTL sweep (mixes older than 30 days) + per-user cap (e.g. 50 mixes,
      LRU by session mtime), reusing the delete path in
      `scripts/api/routers/library.py` (`/library/delete-batch` internals, L310).
- [ ] `/library/storage` + `/library/storage/evict` (L190, L268) become per-user,
      report only `outputs/<uid>/`.

**Verify**
- [ ] Exceed the per-user cap → oldest *own* mix evicted, other users untouched.
- [ ] TTL sweep removes an artificially back-dated mix, keeps a fresh one.
- [ ] `/library/storage` shows only the caller's `outputs/` usage.
- [ ] Green bar.

**Commit:** `feat: TTL + per-user cap on rendered mixes`

---

## Phase 7 — Deploy / infra

**Goal:** the app is live behind TLS on the target stack, reproducible from a runbook.

**Tasks**
- [ ] Frontend on Cloudflare Pages; CI builds `frontend/dist`; set
      `VITE_API_BASE=https://api.<domain>`, `VITE_EVENTS_URL=…/events/stream`.
      Stop committing `frontend/dist/`.
- [ ] Hetzner CX22: bare `uvicorn` (1 worker) under `systemd`, `Caddy` for
      auto-TLS + reverse proxy. `apt install ffmpeg libsndfile1`; project venv;
      `pip install -r requirements.txt && pip install -e .` (no `demucs`).
- [ ] Mount the catalog volume **read-only** at `LIBRARY_DIR`.
- [ ] `/etc/remixmate.env` (chmod 600, from `app/deploy/remixmate.env.example`):
      `REMIXMATE_*` overrides, Supabase keys, Spotify app creds, `DATABASE_URL`,
      `REMIXMATE_AUTH_ENABLED=true`, `REMIXMATE_LICENSE_ENFORCE=true`,
      `REMIXMATE_SPECTRAL_LEARNING=false`. **Do not deploy `config.local.yaml`.**
- [ ] CORS (`scripts/api/main.py:119-152`): `_default_origins` → frontend domain
      only; **delete** the `_allow_private_network` middleware and the
      `chunduri-aditya.github.io` / `localhost:8501` entries. Keep `allow_credentials=True`.
- [ ] `systemd` `ExecStop` drain (`_executor.shutdown(wait=True)` already in the
      lifespan) with `TimeoutStopSec=600`. Document "a redeploy fails any running render."
- [ ] New `.github/workflows/ci.yml`: `uvx ruff check`; pytest safe subset on
      SQLite; `cd frontend && npx tsc --noEmit && npx vitest run && npx vite build`.
      Deploy job: SSH → `git pull` → `pip install` → `systemctl restart remixmate`
      + Pages deploy. Leave `pages.yml` / `codeql.yml` disabled.
- [ ] Catalog deploy: separate `rsync` of the built volume, not part of code deploy.
- [ ] Backups: nightly `pg_dump $DATABASE_URL | gzip` + `rclone` to Backblaze/R2.
- [ ] Fill `app/deploy/` — `Caddyfile`, `remixmate.service`, `remixmate.env.example`,
      `PROVISION.md`.

**Verify**
- [ ] `grep -rn "import demucs\|from demucs" scripts/api/` → no hits (re-check on the deploy box).
- [ ] Fresh VPS from the runbook boots the API behind Caddy TLS, catalog volume read-only.
- [ ] `/health/ready` → 200 from the public internet.
- [ ] Frontend on its domain: sign up, log in, render a mix from two catalog
      tracks end to end, download it with `CREDITS.txt`.
- [ ] Two concurrent `dj-remix` jobs on the box → RSS stays under box RAM, both
      complete. End-to-end render latency for two catalog tracks: `___ s`.
- [ ] `systemctl restart remixmate` drains in-flight then comes back healthy.
- [ ] `pg_dump` output restores into a scratch DB without error.
- [ ] CI green on a PR.

**Commit:** `feat: production deploy — Caddy + systemd + Cloudflare Pages + CI`

---

## Phase 8 — Legal / product surface

**Goal:** the public-facing legal minimum for a CC-catalog remix product.

**Tasks**
- [ ] Frontend routes + footer: `/legal/terms`, `/legal/privacy`, `/legal/dmca`,
      `/legal/acceptable-use`. Cover: data stored (email via Supabase, mixes,
      crates, feedback, audit log); DMCA/takedown procedure + agent contact; "Not
      affiliated with ccMixter, Free Music Archive, Internet Archive, Musopen,
      Spotify, or any rights holder"; Acceptable Use for user-published mixes;
      CC-BY attribution requirement (mixes carry `CREDITS.txt`); account deletion + export.
- [ ] `POST /legal/dmca` (public, rate-limited) → `dmca_reports` row + email.
- [ ] `DELETE /account` (auth) → purge `outputs/<uid>`, all `user_id` rows in
      jobs/crates/feedback/setlists, the Supabase user.
- [ ] `GET /account/export` → zip of the user's mix manifests + crates + favorites.
- [ ] Admin (`ADMIN_EMAILS`-gated, no UI): `DELETE /admin/mix/{job_id}`,
      `POST /admin/user/{id}/flag` (flagged → `require_user` returns 403).
- [ ] Add the `credits` list to the MixVault detail view and the export/share panel.

**Verify**
- [ ] All four footer links resolve and render content.
- [ ] `POST /legal/dmca` stores a row and sends the email.
- [ ] `DELETE /account` leaves zero rows and zero files for that uid.
- [ ] A flagged user gets 403 on every gated route.
- [ ] `DELETE /admin/mix/{job_id}` removes the output dir and marks the job.
- [ ] Green bar.

**Commit:** `feat: legal pages, DMCA intake, account deletion/export, admin takedown`

---

## Cut list (explicitly deferred past v1)

MusicGen / VampNet / DAC (routers guarded, code kept) · all Spotify OAuth + import
· all yt-dlp / YouTube / SoundCloud · **Free Music Archive + Musopen ingest
sources** (fetchers dropped from `scripts/download.py`; `classify_license` still
recognises both, so re-adding a fetcher needs no classifier change — v1 fills
from ccMixter + Internet Archive only) · per-user libraries and uploads · per-user
personalization of learned weights (spectral matching, LUFS targets) · Modal / any
runtime GPU · Redis / arq / multi-worker API / horizontal scaling · DJ-engine
`analyze()` reading cached `analysis.json` instead of recomputing per job (perf,
do only if render latency bites) · `scripts/core/database.py` Postgres migration
(CLI-only) · `track_metadata.py` Postgres migration · collaborative / shared
crates · admin dashboard UI · Streamlit UI (stays in repo, not deployed).

## Riskiest parts

1. **contextvar across the ThreadPoolExecutor boundary** (P3) — if `user_id`
   isn't stamped into job `meta` and re-set in `_wrapper()`, every render writes
   to the wrong user's `outputs/` *and all single-user tests still pass*. The
   tripwire test in P3 is mandatory.
2. **`classify_license` over four heterogeneous sources** (P1) — each expresses
   license differently; an unrecognised field silently rejects the track, so the
   catalog quietly won't fill. The 8-case matrix in P1 Verify is the guard.
3. **Catalog build** (P1) — per-source fetchers + license classification + a
   Demucs batch. Operator tooling, not user-facing, so it can start tiny and grow.
4. **SSE auth** (P4) — cookie-based is forced; it shapes the whole frontend auth flow.
5. **One-time `outputs/` + DB namespace migration** (P4) — irreversible if wrong;
   snapshot `data/` and `outputs/` first.
6. **Every redeploy force-fails in-flight jobs** (P7) — accepted with
   `api.workers: 1`; mitigate with the drain window, communicate to beta users.

## Critical files

| File | Role |
|---|---|
| `scripts/api/main.py` | auth-gate injection, CORS teardown, static-UI removal, `systemd` drain |
| `scripts/api/jobs.py` | contextvar re-set in `_wrapper`, `user_id` column/filter, SQLAlchemy engine |
| `scripts/api/routers/_helpers.py` | `_check_job_cap` → per-user; `_song_info` reads `artist_name` / `license_ccurl` |
| `scripts/api/schemas.py` | `SongInfo` gains `artist_name` + `license_ccurl` |
| `scripts/core/paths.py` | `REMIXMATE_DATA_DIR` override (P0); per-user `OUTPUTS_DIR` + `session_dir` (P3); `LIBRARY_DIR` untouched |
| `scripts/core/audit.py` | `_AUDIT_PATH` derives from `DATA_DIR` (P0); `user_id` field (P3) |
| `scripts/download.py` + `scripts/core/license.py` | operator ingest tool + `classify_license` per source + `{CC0, CC_BY, PUBLIC_DOMAIN}` gate |
| `scripts/api/routes.py` + `routers/{downloads,spotify,generative}.py` | router deletion / guarding |
| `scripts/core/{crates,feedback_store}.py` + setlist store | per-user column + Postgres |
| `scripts/core/{spectral_matching,mastering}.py` | `REMIXMATE_SPECTRAL_LEARNING` freeze in prod |
| `scripts/core/db.py` (new), `scripts/api/auth.py` (new) | SQLAlchemy engine, JWT dependency + contextvar |
| `frontend/src/pages/Operations.tsx`, `frontend/src/lib/api.ts`, `frontend/src/shell/` | download UI → catalog browser + auth shell |
| `scripts/core/config.py` + `config.yaml` | edited together (config-drift test) |
| `tests/conftest.py` | P0 storage-isolation shim (session TemporaryDirectory, 6 `REMIXMATE_*` path vars) |
| `app/deploy/` | Caddyfile, systemd unit, env template, provisioning runbook (P7) |

## Sources

- [Jamendo API Terms of Use](https://devportal.jamendo.com/api_terms_of_use)
- [Jamendo API — Jamendo Licensing Help Center](https://help-licensing.jamendo.com/hc/en-us/articles/20699346005661-Jamendo-API)
