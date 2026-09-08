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
| 2 | Guard generative + slim the image | ☑ verified | f27aecb |
| 3 | User contextvar + per-user mixes/jobs/crates/feedback | ☑ verified | 653633a (3a) + stem gate/audit/spectral/feedback/crates (3b) |
| 4 | Auth + tenancy (Supabase) | ◐ code done | RS256 JWKS gate + cookie-auth SSE + uid migration (backend) + auth shell (frontend); live Supabase run operator-pending |
| 5 | SQLite → Postgres (jobs, crates, feedback) | ☑ verified (code + real Postgres 16 container; live managed-Postgres run operator-pending) | SQLAlchemy Core engine factory `scripts/core/db.py`, `jobs`/`crates`/`feedback_store` on `text()` + named binds, multi-process `_load_from_db()` safety. No setlist store — `routers/setlist.py` is stateless |
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
  After the licence matrix + mix-credits tests: **706 passed**. After Phase 2
  (generative gate + root-route tests): **722 passed**. After Phase 3a
  (`test_user_scope.py`, 6 cases): **728 passed**. After Phase 3b (stem-split
  gate, audit, spectral freeze, feedback, crates — 15 more cases): **743 passed**.
  After Phase 4 backend (`test_auth.py` 13 + `test_migrate_uids.py` 4): **760 passed**.
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

**State:** ☑ done — `f27aecb`.

**Tasks**
- [x] `scripts/api/routers/generative.py`: the whole
      `APIRouter(dependencies=[Depends(_generative_enabled)])` → every `/ai/*`
      route 503s unless `REMIXMATE_GENERATIVE_ENABLED` is truthy. Flag read
      **per request** (not a module constant). Handlers + job types untouched.
      `dj_engine` / `classic_engine` / `_resolve_engine` not touched.
- [x] `requirements.txt`: `transformers` / `safetensors` / `huggingface-hub` →
      `pyproject.toml` `[generative]` extra (only `model_manager.py` /
      `codec_tokens.py` import them, lazily, behind the gate). `streamlit` /
      `altair` removed (`scripts/ui/` only). `lyricsgenius` / `nltk` → new
      `[wordplay]` extra (`wordplay.py` already `try/except`s them). **`demucs`
      kept in `requirements.txt`** — it is a `python -m demucs` subprocess,
      never imported; the "no imports on the request path" goal is already met.
      Original "move to an `ingest` extra" dropped: it would break `/stems/split`
      in a `requirements.txt`-only venv and changes nothing about the import
      surface.

**Exposure this left open — CLOSED in Phase 3b.** `POST /stems/split` and
`/stems/split-batch` let any caller trigger a CPU-Demucs separation at 2-5x
realtime on the single-worker box (`api.workers: 1`). Phase 3b put
`Depends(_stem_split_enabled)` on both — 503 unless `REMIXMATE_STEM_SPLIT_ENABLED`
is truthy (default off in prod). The `auto_analyze=True` path was moot: its only
callers were the download routers deleted in Phase 1, and `run_song_analysis`
reads existing stems without shelling Demucs. Phase 4's `require_user` layers on
top so an authenticated beta user still cannot amplify CPU load with the flag on.
Still open, out of scope here: `POST /library/initialize` batch-splits the whole
library — a maintenance route, left for Phase 4.
- [x] `docker-compose.yml`: `ui` service dropped. `Dockerfile`: `8501` EXPOSE
      and `models/` dir dropped.
- [x] `scripts/api/main.py`: `/` and `/app` no longer read
      `scripts/ui/static/index.html`. 302 to `REMIXMATE_FRONTEND_URL` when set,
      else a 200 JSON pointer at `/docs` (keeps local dev usable — a fake
      redirect target does not).

**Verify**
- [x] API imports with `REMIXMATE_GENERATIVE_ENABLED` unset; safe subset
      **722 passed** (+16 new).
- [x] `POST /ai/style-transfer` / `/ai/inpaint` / `GET /ai/models` → 503;
      `POST /dj-remix` and `POST /dj-chain` → 422 (reached their own handlers,
      not the gate). `tests/test_generative_gate.py` (11 cases, incl. a
      per-request-read check).
- [x] `grep -rn "import demucs\|from demucs" scripts/api/` → no hits.
- [x] `pip check` clean **against the pre-slim venv** (removing lines from
      `requirements.txt` does not uninstall — this does not validate the slimmed
      set). The slimmed set is covered by: importing `scripts.api.main` /
      `.routes` with all 7 removed packages shadowed to `None` succeeds → none
      are on the startup path. **Fresh-venv `pip install -r requirements.txt`
      not run** (slow, pulls torch) — the shadow-import test is the evidence.
      A fresh-install + `pip check` belongs in the Phase 7 CI job.
- [x] `GET /` → 302 to `REMIXMATE_FRONTEND_URL` when set, 200 JSON pointer when
      unset. `tests/test_root_route.py`.
- [x] Green bar: safe subset 722 · `uvx ruff` clean on `generative.py` +
      new tests (`main.py` keeps 1 pre-existing in-function `I001`).

**Commit:** `f27aecb` — `feat: gate generative routes behind a flag; drop unused runtime deps`

---

## Phase 3 — User contextvar + per-user mixes / jobs / crates / feedback

**Goal:** every write path is user-scoped, default `"local"`, single-user
behaviour byte-identical. No auth, no Postgres yet.

**State:** ☑ done — slice **3a** `653633a` (outputs + job store), slice **3b**
this batch (stem-split gate, audit, spectral freeze, feedback, crates).

**Tasks — slice 3a (`653633a`)**
- [x] New `scripts/api/auth.py`: `current_user: ContextVar[str]` (default
      `"local"`), `get_user_id()` / `set_user_id()`.
- [x] `scripts/core/paths.py`: `user_outputs_dir() -> OUTPUTS_DIR / get_user_id()`;
      `session_dir()` routes through it (lazy import of `scripts.api.auth`).
      **`LIBRARY_DIR` untouched.** 7 reachable output sites re-anchored
      (`task_modules/remix.py` ×3, `lab.py`, `routers/remix.py` ×2,
      `routers/library.py` serving route).
- [x] `scripts/api/jobs.py` — the risky one: `create_job()` stamps
      `meta["user_id"] = get_user_id()`; `_wrapper()` calls `current_user.set(...)`
      from the stamped meta next to `set_job_id()` (`ThreadPoolExecutor` does not
      copy context). `list_jobs` gains `user_id` (`None` → acting user, `"*"` →
      unscoped) applied on both the in-memory and SQLite paths.
- [x] `_check_job_cap` counts per-user — it calls `list_jobs()` with no arg, now
      scoped to the acting user. Covered by `test_active_job_cap_counts_per_user`.
- [x] `system.py` `/health` + `events.py` heartbeat pass `user_id="*"` (box telemetry).

**Tasks — slice 3b (this batch)**
- [x] **`scripts/api/routers/stems.py`**: `REMIXMATE_STEM_SPLIT_ENABLED` gate —
      **per-route** `dependencies=[Depends(_stem_split_enabled)]` on `/stems/split`
      and `/stems/split-batch` only (503 unless truthy, read per request). The two
      `/stems/compress*` routes are FLAC encode, never Demucs — left open. The
      `auto_analyze` Demucs half is **moot**: the only requests carrying it were on
      the download routers, deleted in Phase 1; `run_song_analysis` reads existing
      stems, never shells Demucs. `/library/initialize` (`task_initialize_library`)
      is still an unauthenticated batch-Demucs trigger — out of this slice's scope,
      Phase 4 `require_user` covers it.
- [x] `scripts/core/crates.py` + `scripts/core/feedback_store.py`:
      `user_id TEXT NOT NULL DEFAULT 'local'` column on every table; an idempotent
      `PRAGMA table_info` + `ALTER TABLE … ADD COLUMN` migration in `_init()` /
      `_init_db()` (verified against copies of the live DBs — the one real
      `song_favorites` row backfilled to `"local"`); every function derives the
      acting user via a lazy-import `_acting_user()` and adds `WHERE user_id = ?`;
      feedback's in-memory `_events` list is scoped too (jobs.py `_scoped` pattern),
      `list_events` / `summarize` gain a `user_id` arg with a `"*"` escape.
      **No persistent setlist store exists** — `setlist_planner.save(path)` writes
      to a caller-supplied path and the router is stateless, so the plan's
      "setlist store" line was vacuous.
      **Known gap (Phase 4):** `crates.name UNIQUE`, `song_tags` PK, and
      `song_favorites` PK are still user-blind (SQLite can't `ALTER` a
      constraint). Harmless while `get_user_id()` is always `"local"`; Phase 4
      must rebuild those three keys. Pinned by
      `test_crates_uniqueness_is_still_user_blind_phase4`. Severity differs:
      `create_crate` / `add_tag` hit `IntegrityError` and fail closed, but
      `set_favorite`'s `INSERT OR REPLACE` against the `(song_name)` PK would
      *silently rewrite the other user's row* — data loss, no error. Also:
      `feedback_store._MAX_IN_MEMORY = 5000` is a global cap filled newest-first
      across all users, and `summarize()` now reads per-user — after a restart a
      heavy user can evict a quiet user's whole summary until the next event.
- [x] `audit.jsonl` (`scripts/core/audit.py`): the existing `user` field now
      defaults to `_acting_user()` instead of the hard-coded `"local"`; an
      explicit `user=` still wins. (One file, one field — no schema.)
- [x] **Genuinely global, no change** (read-only *search* over the shared catalog):
      `music_index.py` semantic index, `song_embeddings.json`, `crate_digger.py`
      CLAP index.
- [x] **Freeze in prod** — `spectral_matching.py`: `REMIXMATE_SPECTRAL_LEARNING`
      (default on locally, off in prod). When off, `record_render` and
      `apply_feedback` no-op (`{"updated": False, "reason": …}`). Inference still
      runs — it uses `data/spectral_match_model.json` if present, else uniform
      band weights (`np.ones(24)`). `data/` is gitignored, so a fresh prod box
      scores every band equally; it never trains. **`mastering.py` `stem_lufs_targets.json`
      needs no gate** — it is corpus-derived from the shared read-only catalog
      via the operator endpoint at `library.py:446` (`analyze_library_stem_targets`
      measures LUFS across library stems), not feedback-written. Per-user
      personalisation stays deferred (cut list).

**Verify**
- [x] Safe subset with the P0 fixture — **743 passed**, 90 deselected (728 after
      3a, +15 in 3b). Single-user `"local"` behaviour unchanged.
- [x] **Allow / deny (jobs, 3a)** — `test_job_is_stamped_with_and_scoped_to_its_creator`,
      `test_filtered_query_is_also_user_scoped`, `test_legacy_rows_without_user_id_belong_to_local`.
- [x] **Deny (crates / feedback, 3b)** — `test_crates_tags_favorites_are_user_scoped`
      (u2 sees empty `list_crates` / `list_favorites` / `song_tags` / `crate_songs`,
      cannot delete or rename u1's crate), `test_feedback_events_are_user_scoped`
      (controlled 3-event set: u2 sees exactly its own 1, `user_id="*"` sees 3).
      Both mutation-tested (drop the `WHERE user_id` / `_scoped` filter → red).
- [x] **Contextvar tripwire (3a)** — `test_tripwire_job_artifact_lands_under_the_submitters_user_dir`:
      job submitted as `u1` lands under `outputs/u1/`; goes red if `_wrapper`
      stops re-`set`-ting `current_user`.
- [x] **Stem-split gate** — `test_stem_split_is_503_unless_enabled` (split +
      split-batch 503, compress routes reach their handler),
      `test_stem_split_enabled_reaches_handler`, `…falsy_values_stay_disabled`.
      Mutation-tested: drop the `dependencies=` → 503 case goes red (404).
- [x] **Spectral freeze** — `test_spectral_learning_frozen_is_a_noop`:
      non-vacuous (the render is recorded *while learning is on*, so the history
      entry exists; the frozen `apply_feedback` declines it because of the flag,
      not missing history). `record_render` no-op verified too. Mutation-tested.
- [x] **Migrations** — `test_feedback_migration_adds_user_id_and_backfills`,
      `test_crates_migration_adds_user_id_and_backfills`: build the pre-Phase-3
      table by hand (the P0 fixture always hands a fresh schema, so the migration
      branch needs an explicit test), run `_init*`, assert the column exists and
      rows backfilled to `"local"`.
- [x] Green bar: safe subset 743 · `uvx ruff` clean on the 6 touched files.

**Commits:** `653633a` (3a) · this commit (3b — stem gate / audit / spectral
freeze / feedback / crates).
**Rollback note:** `data/` was snapshotted to the scratchpad before the crates /
feedback migrations ran. The migrations are `ADD COLUMN … DEFAULT 'local'` only —
forward-safe, and a pre-Phase-3 binary still reads the table (it ignores the new
column).

---

## Phase 4 — Auth + tenancy (Supabase)

**Goal:** unauthenticated requests get 401; two accounts are fully isolated except
the shared catalog.

**State:** ◐ backend + frontend done; only the live-Supabase run (real
2-account isolation, browser cookie handshake) is operator-pending — needs a
provisioned project.

**Decisions (2026-09-06):** RS256 via the project **JWKS** (`SUPABASE_JWKS_URL`),
not the HS256 project secret. No Supabase project provisioned yet, so the backend
is flag-gated and tested with synthetic RSA-signed tokens; the real 2-account
isolation run and the browser cookie handshake are `operator-pending`.

**Tasks — backend (this batch)**
- [x] `scripts/api/auth.py`: `_verify_token` (Supabase RS256 against a lazily
      built, URL-keyed `PyJWKClient`; `aud="authenticated"`, `require=["exp","sub"]`);
      `_extract_token` (Bearer header, then a raw-JWT `sb-access-token` cookie for
      EventSource); `auth_context_middleware` sets `current_user` from the token
      `sub` per request (blocking verify runs in a threadpool, the `.set()` back
      in the request task); `require_user` dependency → 401 on a missing/bad
      token, 500 on a missing `SUPABASE_JWKS_URL`. All no-ops (`"local"`, no token
      needed) unless `REMIXMATE_AUTH_ENABLED` is truthy.
- [x] `scripts/api/main.py`: `app.middleware("http")(auth_context_middleware)`;
      `system.router` mounted **open** first, then
      `app.include_router(router, dependencies=[Depends(require_user)])`.
      `scripts/api/routes.py` no longer aggregates `system` (main.py owns it now).
      The contextvar is set in **middleware, not the dependency** — a sync
      dependency's `.set()` does not reach a sibling sync endpoint; the
      middleware's does (proven by `test_auth.py`'s sync `/whoami`).
- [x] SSE: `/events/stream` is inside the gated router, so `require_user` runs;
      `_extract_token` reads the `sb-access-token` cookie (`EventSource` can't
      send `Authorization`). `allow_credentials=True` already set at `main.py`.
- [x] `scripts/migrate_uids.py`: one-time `"local"` → real-uid migration. Dry run
      by default; `--apply` moves `outputs/local/*` → `outputs/<uid>/` (hard error
      on a colliding dir, before any DB write) and re-stamps every `"local"` row
      in `jobs.db` (`meta_json.user_id`), `remixmate.db` (4 crate tables), and
      `feedback.db`. Foreign-uid rows untouched. `tests/test_migrate_uids.py` (4).
- [x] `requirements.txt` + `pyproject.toml`: `PyJWT[crypto]==2.10.1` (pulls
      `cryptography` for RS256). Imported lazily — a no-auth deploy never loads it.

**Tasks — frontend (commit 2, done)**
- [x] `frontend/src/lib/api.ts`: every request now sends `credentials: 'include'`
      and, when a session exists, `Authorization: Bearer <access_token>` (from
      `getAccessToken()`). New `setUnauthorizedHandler` seam — a 401 from a gated
      route calls it (AuthProvider wires it to `signOut()`).
- [x] `@supabase/supabase-js@^2.115` added. `frontend/src/lib/supabase.ts` —
      client is `null` + `authConfigured=false` unless `VITE_SUPABASE_URL` +
      `VITE_SUPABASE_ANON_KEY` are both set, so local dev / tests are unchanged.
- [x] `frontend/src/shell/`: `authContext.ts` (context + `useAuth`, split per the
      `toastContext` pattern), `AuthProvider.tsx` (session lifecycle,
      `onAuthStateChange`, cookie mirror, 401 handler), `RequireAuth.tsx` (gate:
      passthrough when unconfigured, splash while loading, `<LoginPage>` when no
      session), `LoginPage.tsx` + `.css` (email/password sign-in + sign-up,
      design-token styling). `LeftRail` gains a sign-out button when
      `authConfigured && session`. Wired: `AuthProvider` in `main.tsx`,
      `RequireAuth` around `<Routes>` in `App.tsx`.
- [x] **Cookie decision:** no Supabase SSR chunked cookie. `syncSseCookie()`
      writes a single readable `sb-access-token` cookie (matches
      `auth.py::_DEFAULT_COOKIE_NAME`) kept in sync with the session
      (`AuthProvider`'s `onAuthStateChange`) — `SameSite=None; Secure` on https
      (cross-site prod topology), `SameSite=Lax` on http localhost. Not httpOnly
      (JS sets it; the token is in localStorage anyway via supabase-js) — a true
      httpOnly session cookie needs an SSR/edge proxy, deferred.
- [x] **The plan's "EventSource hook unchanged" was wrong.** `useSSE.ts` now
      opens `EventSource(url, { withCredentials: true })` — required for the
      cookie to ride a cross-origin stream (prod: Pages frontend → API on
      another domain). Harmless same-origin.
- [x] `AuthProvider` also `queryClient.clear()`s on `SIGNED_OUT` so a re-login
      on the same tab never shows the previous user's cached data
      (`staleTime` is 15s). `main.tsx` nests `AuthProvider` inside
      `QueryClientProvider` for this.

**Verify**
- [x] Synthetic-token matrix — `tests/test_auth.py` (11): valid Bearer → 200 +
      the sync handler sees `sub`; valid `sb-access-token` cookie → 200; no token
      → 401 (`WWW-Authenticate: Bearer`); expired / wrong-aud / wrong-key /
      malformed / no-`sub` → 401; `/health/live` + `/health/ready` open with auth
      on; auth on + no `SUPABASE_JWKS_URL` → 500. The 401 gate and the `audience=`
      arg were each mutation-tested red.
- [x] `REMIXMATE_AUTH_ENABLED` unset: safe subset **760 passed**, 90 deselected
      (743 + 11 auth + 4 migrate). `"local"` behaviour byte-identical.
- [ ] **operator-pending** — two real Supabase accounts: `/library` identical
      (shared catalog); `/jobs`, `/crates`, `/feedback/summary`, `/outputs/*`
      disjoint. Needs a live project.
- [ ] **operator-pending** — SSE stream connects with the browser session cookie,
      rejects without it. Needs the frontend auth flow + a live project.
- [x] Frontend gate: `tsc --noEmit` clean · `vitest run` **53 passed** (10 files;
      +18 — `auth.test.tsx` 7, `AuthProvider.test.tsx` 7, `supabase.test.ts` 4) ·
      `vite build` clean · `eslint --max-warnings 0` clean.
- [x] `<RequireAuth>` covered — `authConfigured=false` passes through
      (`supabase.test.ts` + `smoke.test.ts`); `=true` shows login with no
      session, splash while loading, children with a session, toggles to
      sign-up (`auth.test.tsx`). api.ts sends `Authorization` + `credentials`
      only when a token exists; a 401 fires the unauthorized handler.
- [x] `AuthProvider` lifecycle covered (`AuthProvider.test.tsx`, 7): initial
      cookie mirror + 401-handler registration; cookie re-mirrored on every
      auth-state change incl. token refresh (mutation-tested); cookie cleared +
      `queryClient.clear()` on `SIGNED_OUT` (not on other events); subscription
      + handler torn down on unmount.

**Commits:** `<backend batch>` (RS256 gate + SSE cookie + uid migration) ·
`<this batch>` (frontend auth shell).

---

## Phase 5 — SQLite → Postgres (jobs, crates, feedback) — DONE

**Goal:** the per-user stores run on managed Postgres in prod, SQLite in
tests/local — both paths exercised.

**What shipped**
- [x] New `scripts/core/db.py`: SQLAlchemy Core engine factory. `DATABASE_URL`
      unset → one SQLite engine **per store**, keyed on `str(_DB_PATH)` (keeps
      the three data files and the P0 storage-isolation fixtures' per-store path
      override working); `DATABASE_URL` set → one shared engine
      (`postgresql+psycopg://…`). Diverges from the "an Engine" wording on
      purpose — collapsing to one engine when unset breaks the isolation
      fixtures. NullPool on SQLite so `.connect()` stays connect-per-call.
      Routers stay sync `def`.
- [x] `jobs.py`, `crates.py`, `feedback_store.py`: `_get_conn()` / `_conn()`
      kept as the mockable seam but hand back a SQLAlchemy `Connection`. SQL
      wrapped in `text()`, `?` → `:name`, `jobs._query_db`'s dynamic `IN` list
      uses generated named binds.
- [x] Timestamp columns (`created_at`/`started_at`/`finished_at`/`added_at`/
      `favorited_at`/`position_sec`) → `DOUBLE PRECISION` on Postgres; float4
      `REAL` there scrambles every `ORDER BY <timestamp>` at epoch magnitudes.
- [x] `crates._SCHEMA` multi-statement string → one `CREATE` per `execute`.
      Phase 3b migration probe `PRAGMA table_info` → `inspect(conn).get_columns`
      (run on the init connection so it sees the uncommitted `CREATE`s on
      Postgres). `sqlite3.IntegrityError` → `sqlalchemy.exc.IntegrityError`.
      `INSERT OR REPLACE INTO song_favorites` → `INSERT … ON CONFLICT DO
      UPDATE`. `PRAGMA foreign_keys = ON` guarded to the SQLite dialect in
      `_conn()`.
- [x] `requirements.txt` / `pyproject.toml`: `SQLAlchemy==2.0.52`,
      `psycopg[binary]==3.2.13`. psycopg is imported by SQLAlchemy only when a
      postgres URL is built, so a SQLite-only install never loads it.
- **Verified deferred:** `track_metadata.py` `MetadataCache` (global cache) and
  `scripts/core/database.py` `RemixMateDatabase` (CLI-only, not imported by the
  API) stay on SQLite. There is **no setlist store** — `routers/setlist.py` is
  stateless (`setlist_planner.save(path)` writes to a caller-supplied path), so
  the original "+ setlists" scope line was vacuous.

**Verify**
- [x] `DATABASE_URL` unset → safe subset **762 passed** (per-store SQLite). A
      `DATABASE_URL=sqlite:///<file>` run confirmed the shared-engine branch:
      all six tables coexist in one file, all three stores functional.
- [x] Against a **real Postgres 16 container**: job
      create/poll/list/cancel/retry with user scoping, both enum
      serializations on the filtered read-through, crate CRUD + tags +
      favorites with user scoping, add/tag duplicate → `False`, repeat favorite
      → `ON CONFLICT` update, `delete_crate` FK cascade, the legacy-table
      migration (`inspect` + `ALTER` on both dialects), feedback
      log/list/summarize/restart.
- [x] Restart rollback: `_load_from_db()` rewrites any RUNNING **or** PENDING
      row to FAILED. A RUNNING row re-loaded comes back FAILED in memory **and**
      persisted. New regression test
      `test_running_jobs_roll_back_to_failed_on_restart`.
- [x] Mutation-tested: `jobs._query_db` named-bind conversion (divergent param
      key → 5 reds), the restart rollback guard (drop RUNNING → red),
      `crates` `IntegrityError` catch type (→ `RuntimeError` → red), the
      `song_favorites` `ON CONFLICT` clause (dropped → red).
- [x] Multi-process safety: two processes running `_load_from_db()` against the
      same Postgres at once — both exit clean, no deadlock, orphaned RUNNING and
      PENDING rows both end FAILED once (idempotent). This is the invariant the
      hosted multi-process deploy rests on.
- [x] `list_crates()` merged its two connections into one held read; per-crate
      `song_count` verified right across several crates of different sizes
      (`test_list_crates_song_count_is_per_crate`), on both dialects.
- [ ] **operator-pending:** point a hosted deploy at a managed Postgres
      (`DATABASE_URL` on the API service) and run the same flows against it.

**Commits:** `refactor: SQLAlchemy Core engine for the job store (Phase 5
commit 1)` · `refactor: SQLAlchemy Core for the crate + feedback stores (Phase
5 commit 2)`

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
      `VITE_API_BASE=https://api.<domain>`, `VITE_EVENTS_URL=…/events/stream`,
      `VITE_SUPABASE_URL=https://<ref>.supabase.co`, `VITE_SUPABASE_ANON_KEY=…`
      (both required to turn auth on client-side; must pair with the API's
      `REMIXMATE_AUTH_ENABLED=true` + `SUPABASE_JWKS_URL`). Stop committing
      `frontend/dist/`. API CORS must echo the Pages origin (not `*`) with
      `Access-Control-Allow-Credentials: true` for the cross-site SSE cookie.
- [ ] Hetzner CX22: bare `uvicorn` (1 worker) under `systemd`, `Caddy` for
      auto-TLS + reverse proxy. `apt install ffmpeg libsndfile1`; project venv;
      `pip install -r requirements.txt && pip install -e .` (`demucs` is in
      `requirements.txt` — it is only reached via the `REMIXMATE_STEM_SPLIT_ENABLED`
      route, off in prod; keeping it installed is cheap and avoids a broken
      `/stems/split` if the flag is ever flipped on for a debug session).
- [ ] `pip install -e .` **without** `[generative]` / `[wordplay]` — no
      transformers, no MusicGen weights on the box.
- [ ] Mount the catalog volume **read-only** at `LIBRARY_DIR`.
- [ ] `/etc/remixmate.env` (chmod 600, from `app/deploy/remixmate.env.example`):
      `REMIXMATE_*` overrides, Spotify app creds, `DATABASE_URL`,
      `REMIXMATE_AUTH_ENABLED=true`,
      `SUPABASE_JWKS_URL=https://<ref>.supabase.co/auth/v1/.well-known/jwks.json`
      (+ `SUPABASE_JWT_AUD` only if not `authenticated`; `SUPABASE_COOKIE_NAME` if
      the frontend uses a non-default session cookie), `REMIXMATE_LICENSE_ENFORCE=true`,
      `REMIXMATE_SPECTRAL_LEARNING=false`, `REMIXMATE_FRONTEND_URL=https://<domain>`,
      `REMIXMATE_STEM_SPLIT_ENABLED` unset, `REMIXMATE_GENERATIVE_ENABLED` unset.
      **Do not deploy `config.local.yaml`.**
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
