# Hosted migration status

Where the `hosted-app` branch stands against `app/DEPLOYMENT_PLAN.md`. This is the
delta on top of `PROJECT_OVERVIEW.md`, not a recap. Every number below came from a
command on 2026-08-31, not an estimate.

## One line

Phases 0 and 1 are done. Phase 1 (catalog teardown + Creative Commons gate): all
code committed and green — licence model, attribution, backend teardown, licence
matrix test, mix credits, frontend teardown, FMA/Musopen descoped. Two Verify
lines need a live network ingest run (`operator-pending`, no code change). Phases
2 through 8 are untouched.

## Phase ledger

| # | Phase | State | Evidence |
|---|---|---|---|
| 0 | Guardrails (test isolation fixture) | done, verified | commit `304da7e`; `tests/test_storage_isolation.py` 5 tests; negative control drops 4/5 |
| 1 | Catalog teardown + enforced CC gate | done (code); live ingest operator-pending | `3d06bf2` `2f2833f` `f6c0c6c` `8c4f204` `46aa166` `dbe2db1` + frontend teardown; see below |
| 2 | Guard generative + slim the image | not started | |
| 3 | User contextvar + per-user data | not started | |
| 4 | Auth + tenancy (Supabase) | not started | |
| 5 | SQLite to Postgres | not started | |
| 6 | Per-user mix retention | not started | |
| 7 | Deploy / infra | not started | |
| 8 | Legal / product surface | not started | |

## Green bar right now

| Check | Result | Note |
|---|---|---|
| safe pytest subset | 706 passed, 90 deselected, ~12s | 702 after licence + config work; −1 from deleting `task_modules/download.py` (a parametrised lint case, not a lost assertion); +1 licence-matrix net; +4 `test_mix_credits.py` |
| import smoke | `scripts.api.main` / `.routes` / `.tasks` / `scripts.download` all import | proves nothing references the three deleted modules |
| `uvx ruff check` (repo-wide) | ~1580 findings on a clean `HEAD` | not a usable gate as it stands: pre-existing debt, ~1100 `W293` blank-line whitespace + ~200 `I001` + ~150 `F401`, none introduced by this migration. `E501` is correctly ignored so the config is being read. Run `ruff check --fix` repo-wide and move the config to `[tool.ruff.lint]` before Phase 7 CI. |
| `uvx ruff check` (teardown files) | ~54 findings, all pre-existing style | the teardown adds none; `analysis_pipeline.py` had 2 (`F401` numpy, `I001`) in code it did not touch |
| frontend gate | `tsc --noEmit` clean · `vitest run` 35 passed (7 files) · `vite build` clean (1.1s) | after the frontend teardown |

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

### Deferred, noted (not teardown, not blocking)

- `scripts/core/spotify.py` still carries dead OAuth methods and
  `spotify_token.json` persistence — nothing live calls them.
- `scripts/ui/app.py` (Streamlit, not deployed) still references `/download` —
  dies with the `ui` service in Phase 2.
- `scripts/core/library.py:87` `_is_downloaded_track` reads `full.info.json`, a
  `yt-dlp` artefact. Library-hygiene helper; left until it bites.
- `JobType.DOWNLOAD` kept in `schemas.py` for historical `jobs.db` rows.

## What is shippable today

Nothing is deployed. The single-user local app still runs exactly as
`PROJECT_OVERVIEW.md` describes: the transition engine, music understanding,
setlist intelligence, search, and the 9-page SPA are all intact on this branch.
The migration has removed capability (all download paths) and added the licence
model and attribution fields. It has not yet added auth, tenancy, Postgres, mix
retention, or any deploy artifact.

Phase 1 is done in code — a clean single-user CC-catalog remixer with no legal
exposure from the audio source, which is the precondition for Phases 3 through 8.
The only Phase 1 items left are the two live-ingest Verify lines, which tick on
the first real catalogue fill. Next is Phase 2 (guard generative routes, slim the
runtime image).

## Honest gaps against a real hosted product

- No auth on any endpoint. `REMIXMATE_AUTH_ENABLED` does not exist yet (Phase 4).
- No per-user data isolation. Every write path is global (Phase 3). The Phase 0
  fixture proves the test harness can isolate paths; the app cannot.
- Jobs, crates, feedback, and setlists are SQLite with a load-bearing
  single-worker assumption (Phase 5).
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
