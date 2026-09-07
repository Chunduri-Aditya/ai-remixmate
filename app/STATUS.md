# Hosted migration status

Where the `hosted-app` branch stands against `app/DEPLOYMENT_PLAN.md`. This is the
delta on top of `PROJECT_OVERVIEW.md`, not a recap. Numbers are revised per
commit against a real command, not estimated.

## One line

Phases 0, 1, 2, and 3 are done in code. Phase 1: licence model, attribution,
backend + frontend teardown, licence matrix test, mix credits, FMA/Musopen
descoped — two Verify lines need a live ingest run (`operator-pending`). Phase 2:
`/ai/*` routes gated behind `REMIXMATE_GENERATIVE_ENABLED` (503 by default),
transformers/streamlit/nltk moved to pyproject extras, Streamlit `ui` service
dropped, root route → frontend redirect. Phase 3: per-user scoping via a
`current_user` contextvar (default `"local"`) — outputs, job store (3a),
stem-split gate, audit user, spectral-learning freeze, and the crates / feedback
SQLite stores (3b). Phase 4: **backend** Supabase auth — RS256/JWKS
`require_user` gate (`REMIXMATE_AUTH_ENABLED`, off → `"local"`), cookie-auth SSE,
`system.router` mounted open, `scripts/migrate_uids.py` one-time re-stamp.
Frontend auth shell (supabase-js, `<RequireAuth>`, login page, Bearer +
credentials on every request, SSE cookie mirror) is in. Only the live-Supabase
run is pending. Phases 5 through 8 untouched.

## Phase ledger

| # | Phase | State | Evidence |
|---|---|---|---|
| 0 | Guardrails (test isolation fixture) | done, verified | commit `304da7e`; `tests/test_storage_isolation.py` 5 tests; negative control drops 4/5 |
| 1 | Catalog teardown + enforced CC gate | done (code); live ingest operator-pending | `3d06bf2` `2f2833f` `f6c0c6c` `8c4f204` `46aa166` `dbe2db1` + frontend teardown; see below |
| 2 | Guard generative + slim the image | done, verified | `f27aecb`; `tests/test_generative_gate.py` (11) + `tests/test_root_route.py` (5); demucs kept (subprocess, not imported) |
| 3 | User contextvar + per-user data | done, verified | `653633a` (3a) + 3b batch; `tests/test_user_scope.py` (21); migrations verified on live-DB copies |
| 4 | Auth + tenancy (Supabase) | code done | backend: `tests/test_auth.py` (13, synthetic RS256, incl. real-app gate + role check) + `tests/test_migrate_uids.py` (4). frontend: `src/shell/{authContext,AuthProvider,RequireAuth,LoginPage}` + `src/lib/supabase.ts`; `auth.test.tsx` (7) + `AuthProvider.test.tsx` (7) + `supabase.test.ts` (4). Live 2-account isolation operator-pending. |
| 5 | SQLite to Postgres | not started | |
| 6 | Per-user mix retention | not started | |
| 7 | Deploy / infra | not started | |
| 8 | Legal / product surface | not started | |

## Green bar right now

| Check | Result | Note |
|---|---|---|
| safe pytest subset | 760 passed, 90 deselected, ~15s | 706 through Phase 1; +16 Phase 2; +6 Phase 3a; +15 Phase 3b; +13 `test_auth.py` +4 `test_migrate_uids.py` Phase 4 backend |
| import smoke | `scripts.api.main` / `.routes` / `.tasks` / `scripts.download` all import | proves nothing references the three deleted modules |
| `uvx ruff check` (repo-wide) | ~1580 findings on a clean `HEAD` | not a usable gate as it stands: pre-existing debt, ~1100 `W293` blank-line whitespace + ~200 `I001` + ~150 `F401`, none introduced by this migration. `E501` is correctly ignored so the config is being read. Run `ruff check --fix` repo-wide and move the config to `[tool.ruff.lint]` before Phase 7 CI. |
| `uvx ruff check` (teardown files) | ~54 findings, all pre-existing style | the teardown adds none; `analysis_pipeline.py` had 2 (`F401` numpy, `I001`) in code it did not touch |
| frontend gate | `tsc --noEmit` clean · `vitest run` 53 passed (10 files) · `vite build` clean · `eslint --max-warnings 0` clean | +18 in Phase 4 commit 2 (auth.test.tsx 7, AuthProvider.test.tsx 7, supabase.test.ts 4) |

The `config.yaml` / `config.py` pair was realigned (dropped
`download.default_source` and `spotify.redirect_uri` from both), fixing 3
`tests/test_config_drift.py` failures the half-applied Phase 1 tree had left red.

## Phase 1 detail

### Licence model + attribution — committed (`3d06bf2`, `2f2833f`)

- `LicenseType.PUBLIC_DOMAIN` added. `PERMITTED_LICENSES = {CC0, CC_BY,
  PUBLIC_DOMAIN}` as a frozenset. `is_publishable(info)` and
  `license_enforce_enabled()` (reads `REMIXMATE_LICENSE_ENFORCE`).
- `classify_license` gained `ccmixter` / `fma` / `internetarchive` / `musopen`
  source branches.
- `save_license()` persists `artist_name`, `license_ccurl`, source URL.
- `SongInfo` schema gained `artist_name` + `license_ccurl`; `_song_info` reads
  both, so `GET /library/<name>` can carry attribution.

### Config alignment — committed (`f6c0c6c`)

- `scripts/core/config.py` + `config.yaml` dropped `download.default_source` and
  `spotify.redirect_uri` together (`test_config_drift.py` green).

### Backend teardown — committed

- `git rm` of `scripts/api/routers/downloads.py`,
  `scripts/api/task_modules/download.py`, `scripts/core/youtube_music.py`.
  Nothing imported them: the `routes.py` / `tasks.py` / `python_bridge.py` edits
  had already unwired all three. Verified by import smoke on the app.
- `scripts/download.py` rewritten as the operator ingest CLI (`python -m
  scripts.download`). ccMixter and Internet Archive fetchers implemented; FMA and
  Musopen descoped for v1 (see below).
- `scripts/api/routers/spotify.py` cut to metadata only: `/spotify/status`,
  `/spotify/search`, `/spotify/track/{id}`. OAuth, callback, disconnect,
  playlists, top, import routes deleted. 252 lines to about 90.
- `scripts/api/routes.py` no longer includes the `downloads` router.
- `scripts/api/tasks.py` no longer re-exports `task_download` /
  `task_playlist_download`.
- `scripts/core/python_bridge.py` removed the YouTube Music online auto-fetch of
  missing tracks. `online` stays in the signature as a no-op.
- `requirements.txt` dropped `yt-dlp` and `ytmusicapi`; the LEGAL NOTE block
  became a MUSIC SOURCING note. **`mutagen` kept** — used by `cue_export.py`
  (Serato GEOB) and `library.py`.
- `scripts/core/analysis_pipeline.py` stale `task_download` docstring reference
  updated.

### Operator-pending (tick on the first real catalogue fill — no code change)

- Ingest a real ccMixter CC-BY track → `license.json` written, `is_publishable` true.
- Feed the ingest tool a CC-BY-NC track → aborts, song dir removed. (The gate
  logic is covered offline by the `classify_license` matrix + `is_publishable`;
  this line is the live end-to-end.)

### Done since the teardown

- `classify_license` 8-case source matrix — `tests/test_license.py::
  test_source_licence_matrix` (`46aa166`). CC-BY allow + CC-BY-NC deny per source
  (ccmixter / archive / fma / musopen), source-shaped inputs, asserts the enum
  and the `is_publishable` gate per cell. Proven to fail under 3 mutations. Same
  commit adds `"archive"` to the Internet Archive source aliases in `license.py`
  (the `download.py` fetcher emits `source="archive"`, previously unhandled).
- Mix credits — `dbe2db1`. `task_dj_remix` / `task_dj_chain` write
  `<out_dir>/CREDITS.txt` + a `credits[]` array in `manifest.json` from each
  source track's `license.json`, and return it under a `credits` key. A track
  with no readable licence still gets an entry (`unknown` / `None`) so an
  attribution line is never silently dropped. Preview render excluded (throwaway
  audition). Guard: `tests/test_mix_credits.py`, mutation-tested 3 ways.
- FMA + Musopen descoped for v1 — fetchers removed from `scripts/download.py`
  (`_FETCHERS` = ccMixter + Internet Archive), docstring + plan cut list updated.
  `classify_license` keeps both source branches, so a future fetcher needs no
  classifier change.
- Frontend teardown — `downloadApi` gone from `lib/api.ts`; `Operations.tsx`
  download form + queue removed (672 → 329 lines), keeps the processing / storage
  / maintenance panels; `SongInfo` TS type + a licence badge in `LibraryAtlas`;
  `smoke.test.ts` mock trimmed. No `catalogApi.search` (no backend endpoint). No
  Spotify import buttons existed. Frontend gate green.

## Phase 2 — committed (`f27aecb`)

- `scripts/api/routers/generative.py`: whole `APIRouter` carries
  `Depends(_generative_enabled)` → every `/ai/*` route 503s unless
  `REMIXMATE_GENERATIVE_ENABLED` is truthy (read per request). Handlers + job
  types untouched.
- `requirements.txt`: `transformers` / `safetensors` / `huggingface-hub` →
  `[generative]` extra; `streamlit` / `altair` removed; `lyricsgenius` / `nltk`
  → new `[wordplay]` extra. `demucs` **kept** (subprocess, never imported; still
  serves `/stems/split`).
- `docker-compose.yml`: `ui` service gone. `Dockerfile`: `8501` EXPOSE + `models/`
  dir gone. `main.py`: `/` and `/app` → 302 to `REMIXMATE_FRONTEND_URL`, else a
  200 JSON pointer.
- Guards: `tests/test_generative_gate.py` (11), `tests/test_root_route.py` (5).
  `pip check` clean; importing `scripts.api.main` with all 7 removed packages
  shadowed to `None` succeeds. Fresh-venv install skipped (slow); shadow test
  covers the risk.

### Deferred, noted (not blocking)

- `scripts/core/spotify.py` still carries dead OAuth methods and
  `spotify_token.json` persistence — nothing live calls them.
- `scripts/ui/app.py` (Streamlit) still references `/download`. The `ui` service
  is no longer in compose; `scripts/ui/` stays in the repo, undeployed.
- `scripts/core/library.py:87` `_is_downloaded_track` reads `full.info.json`, a
  `yt-dlp` artefact. Library-hygiene helper; left until it bites.
- `JobType.DOWNLOAD` kept in `schemas.py` for historical `jobs.db` rows.
- `POST /stems/split` / `/stems/split-batch` CPU-Demucs trigger — **closed in
  Phase 3b**: both routes carry `Depends(_stem_split_enabled)`, 503 unless
  `REMIXMATE_STEM_SPLIT_ENABLED` is truthy (off in prod). `auto_analyze` was moot
  (its download routers went in Phase 1). Still open: `POST /library/initialize`
  (`task_initialize_library`) batch-splits the whole library — a maintenance
  route, left for Phase 4 `require_user`.

## What is shippable today

Nothing is deployed. The single-user local app still runs as
`PROJECT_OVERVIEW.md` describes: transition engine, music understanding, setlist
intelligence, search, and the 9-page SPA. The migration has removed capability
(all download paths, generative routes behind a default-off flag) and added the
licence model + attribution, per-user data scoping, and the full auth path
(backend gate + frontend shell). Still missing: a live Supabase project,
Postgres, mix retention, any deploy artifact.

Phases 1–3 done; Phase 4 is code-complete both sides — backend RS256/JWKS
`require_user` + cookie-auth SSE + `scripts/migrate_uids.py`, frontend supabase-js
+ `<RequireAuth>` + login page + Bearer/credentials on every request. Next:
**Phase 5** — SQLite → Postgres for jobs/crates/feedback. Phase 4's live
2-account isolation run stays operator-pending on a provisioned project.

## Honest gaps against a real hosted product

- Auth is flag-gated (`REMIXMATE_AUTH_ENABLED` / `VITE_SUPABASE_*`, all off by
  default → every request is `"local"`). No Supabase project has been
  provisioned — the JWKS handshake and browser cookie flow are unverified
  against anything real (synthetic RSA tokens + jsdom only).
- Per-user scoping is column-only for crates/tags/favorites: `crates.name UNIQUE`
  and the `song_tags` / `song_favorites` PKs are still global. Phase 4 must
  rebuild those keys before a second real user exists (pinned by a test).
- Jobs, crates, feedback are SQLite with a load-bearing single-worker assumption
  (Phase 5). No persistent setlist store exists.
- `outputs/` grows without bound. No TTL, no cap (Phase 6).
- No Caddyfile, systemd unit, CI workflow, or provisioning runbook. `app/deploy/`
  holds only a README (Phase 7).
- No legal pages, DMCA intake, account deletion, or export (Phase 8).
- The v1 catalogue fills from ccMixter and Internet Archive only. FMA and Musopen
  are descoped (deliberately, not a gap) — `classify_license` still knows both, so
  a fetcher can be added later without touching the gate.
- The `uvx ruff` green bar only works scoped to changed files. Repo-wide it needs a
  `ruff check --fix` pass and a `[tool.ruff.lint]` config move before it can gate CI.

## Sources

- `app/DEPLOYMENT_PLAN.md` — the canonical 9-phase roadmap and per-phase verify blocks
- `PROJECT_OVERVIEW.md` — capability map and measured metrics for the single-user app
- `DEVELOPING.md` — 18 recorded gotchas
