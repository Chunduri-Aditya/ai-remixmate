# Hosted migration status

Where the `hosted-app` branch stands against `app/DEPLOYMENT_PLAN.md`. This is the
delta on top of `PROJECT_OVERVIEW.md`, not a recap. Every number below comes from a
command named beside it; test counts are `grep -c "def test_" <file>` for Python and
the per file count `npx vitest run` prints for the frontend.

## One line

Phases 0 to 8 are done in code and Phase 9 hardening has landed on `hosted-app`:
`git rev-list --count main..hosted-app` prints 53, and
`git rev-list --count fb1a5f9..hosted-app` prints 24 for the hardening run. Local
gates G1 to G3 are green at `bd1c446` (full suite 1960 passed and 1 skipped,
frontend 138 vitest tests, ruff clean). The CI replica in a fresh venv was red on one test
that needs `nltk` (only in the `wordplay` extra); `scripts/core/wordplay.py` now defines
the guard flag without `nltk`, and the replica is green. Nothing is deployed: the Supabase project, the Hetzner box, Cloudflare
Pages, SMTP, a managed Postgres run and a catalog of at least 5 tracks are operator
pending, and the branch is unpushed
(`git rev-list --count origin/hosted-app..hosted-app` prints 24).

## Phase ledger

| # | Phase | State | Evidence |
|---|---|---|---|
| 0 | Guardrails (test isolation fixture) | done, verified | `304da7e`, extended in `52cd9ad` and `a4e8b50`; `tests/test_storage_isolation.py` (9) |
| 1 | Catalog teardown and enforced CC gate | done in code; live catalog fill operator pending | `3d06bf2` `2f2833f` `f6c0c6c` `8c4f204` `46aa166` `dbe2db1` `cc9de1a`, hardened in `42dfd79` and `664a18a`; `tests/test_license.py` (8), `tests/test_license_text.py` (9), `tests/test_ingest_guards.py` (18), `tests/test_mix_credits.py` (4). Pending: `catalog/` holds 3 tracks against a floor of 5 (see gaps) |
| 2 | Guard generative routes and slim the image | done, verified | `f27aecb`; `tests/test_generative_gate.py` (4), `tests/test_root_route.py` (3), `tests/test_generative_urls.py` (3) |
| 3 | User contextvar and per user data | done, verified | `653633a` (3a), `4a40867` (3b); `tests/test_user_scope.py` (19), `tests/test_crate_keys.py` (8) |
| 4 | Auth and tenancy (Supabase) | done in code; live Supabase run operator pending | `d8b510b` backend, `9068435` frontend, local JWKS tool `32bceb4`; `tests/test_auth.py` (15 at `bd1c446`, 13 at `aef279f`), `tests/test_migrate_uids.py` (4), `tests/test_local_jwks.py` (8), `tests/test_tenancy_jobs.py` (22); vitest `auth.test.tsx` (7), `AuthProvider.test.tsx` (7), `supabase.test.ts` (8 at `bd1c446`, 7 at `aef279f`). Pending: no Supabase project exists, so no real two account run through the login UI |
| 5 | SQLite to Postgres (jobs, crates, feedback) | done in code; hosted managed Postgres run operator pending | `e4e894d` (jobs), `cd0ccc8` (crates, feedback); `scripts/core/db.py`; `tests/test_user_scope.py` (19), `tests/test_job_list_filters.py` (8). Pending: no managed Postgres is provisioned; the Postgres container path was not rerun in this pass because `docker info` exits 1 (daemon not running) |
| 6 | Per user mix retention | done, verified | `b8317f2` adds `scripts/core/mix_retention.py`, `GET /library/mixes/storage` and `POST /library/mixes/sweep` (`scripts/api/routers/library.py:316`, `:345`), and a sweep after each render; `a0d3a30` adds the lifespan background sweep of every `outputs/<uid>/` (`scripts/api/main.py:46`). `config.yaml:103` to `:107`: `mix_max_age_days: 30`, `mix_max_count: 50`, `sweep_after_render: true`, `sweep_interval_hours: 6`. Tests: `tests/test_mix_retention.py` (14), `tests/test_lifespan.py` (15) |
| 7 | Deploy and infra | done in code; live deploy operator pending | `8dbfe81` (Caddyfile, `remixmate.service`, `remixmate.env.example`, `PROVISION.md`, `.github/workflows/ci.yml`, CORS), `fb1a5f9` (`bootstrap.sh`, `backup.sh`, `scripts/system/live_verify.py`, catalog dry run), `a4e8b50` (scripts that run, env aliases, drain on shutdown), `13a6d0f` (CI installs dev requirements, Node 22); `tests/test_cors_origins.py` (4), `tests/test_deploy_scripts.py` (25), `tests/test_env_aliases.py` (10), `tests/test_live_verify.py` (17), `tests/test_curate_catalog.py` (6). Pending: no VPS, volume, DNS or Pages project exists; `/health/ready` from the public internet, a restart drain on the box, a `pg_dump` restore and CI green on a PR all wait on provisioning and a push |
| 8 | Legal and product surface | done in code; SMTP delivery operator pending | `2919ec2` (legal pages, `POST /legal/dmca`, `DELETE /account`, `GET /account/export`, admin takedown and flag routes), `a0d3a30` (account purge and admin hardening), `0f4b1f3` (account and legal pages, vault credits in the UI); `tests/test_legal_phase8.py` (7), `tests/test_account_lifecycle.py` (9), `tests/test_admin_takedown.py` (15), `tests/test_legal_text.py` (5); vitest `vault_account_legal.test.tsx` (10). Pending: DMCA email sends only when `REMIXMATE_SMTP_*` is set (`scripts/api/routers/legal.py:148`); no SMTP account is configured |
| 9 | Hardening run (2026-09-27 overnight) | done locally; CI replica green after the wordplay flag fix | 24 commits, `52cd9ad` to `aef279f`; see the Phase 9 section |

## Green bar right now

Run on 2026-09-30 on `hosted-app` with the tree equal to `bd1c446` (the cookie method
guard in `scripts/api/auth.py`, `tests/test_auth.py`, `frontend/src/lib/supabase.ts`
and `frontend/src/__tests__/supabase.test.ts`: 2 more backend tests, 1 more vitest)
plus docstring only edits in `tests/test_core_modules.py` and `tests/test_render_e2e.py`; at
`aef279f` alone the grep counts are 1100 `def test_` in total and 13 in
`tests/test_auth.py`.

G1 to G3 ran in the repo `.venv` (Python 3.12.7), which has drifted above the pins in
`requirements.txt`: `pip freeze` there shows fastapi 0.142.1, starlette 1.7.0,
uvicorn 0.54.0 and PyJWT 2.15.1 against pins 0.141.1, 1.3.1, 0.35.0 and 2.13.0. Only
the G4 replica venv (Python 3.12.7, exactly those pins, no `nltk`) runs the pinned
set.

| Gate | Command | Result |
|---|---|---|
| G1 full suite | `.venv/bin/python -m pytest tests -q -p no:cacheprovider` | exit 0; `1960 passed, 1 skipped, 55 warnings in 258.24s` |
| G2 types | `cd frontend && npx tsc --noEmit` | exit 0, no output |
| G2 tests | `cd frontend && npx vitest run` | exit 0; `Test Files 18 passed (18)`, `Tests 138 passed (138)` |
| G2 lint | `cd frontend && npx eslint . --max-warnings 0` | exit 0, no output |
| G2 build | `cd frontend && npx vite build` | exit 0; `built in 1.54s` |
| G3 ruff | `.venv/bin/ruff check .` | exit 0; `All checks passed!` |
| G4 CI replica install | fresh `python3.12 -m venv`, then `pip install -r requirements-dev.txt` and `pip install -e .` (venv created 2026-09-30 00:30, reused for this rerun) | exit 0 |
| G4 CI replica ruff | `ruff check .` in that venv | exit 0; `All checks passed!` |
| G4 CI replica pytest | `python -m pytest -m "not dj_analysis" --ignore=tests/e2e_test_suite.py -q` in that venv | exit 0; `1864 passed, 1 skipped, 96 deselected, 24 warnings in 41.83s` after the wordplay fix. Before it: exit 1, `1 failed, 1863 passed`, `tests/test_silent_failures.py::test_nltk_download_guarded` raised `AttributeError: ... no attribute '_ensure_nltk_data_done'` because `nltk` is not in the core install and `scripts/core/wordplay.py` set that flag only when `import nltk` succeeded |
| G4 CI replica engine tests | `python -m pytest -m dj_analysis -q` in that venv | exit 0; `96 passed, 1865 deselected, 13 warnings in 214.12s` |
| test functions | `grep -rc "def test_" tests \| awk -F: '{s+=$2} END {print s}'` | 1102 (1100 at `aef279f`) |
| runtime audit | `cd frontend && npm audit --omit=dev` | exit 1; 2 moderate, both in `react-router` 6.0.0 to 7.17.0 (see gaps) |

## Phase 9: hardening (2026-09-27 overnight)

Every commit in `git log --oneline fb1a5f9..hosted-app`, oldest first, with the test
file that guards it. Counts are file totals now, not only the tests the commit added.

| Commit | Fix | Guarded by |
|---|---|---|
| `52cd9ad` | storage is isolated in every pytest run unless a test opts out | `tests/test_storage_isolation.py` (9), `tests/conftest.py` |
| `13a6d0f` | CI installs dev requirements, runs the integration and engine tests, uses Node 22 | no test file; the G4 replica above runs the workflow's commands |
| `f3aa1f3` | removed features' dependencies leave the core install; ruff lint config | `tests/test_pyproject_contract.py` (8) |
| `3c5d3cb` | Python 3.12 image, complete `.dockerignore`, compose bound to loopback | no test file; image build unverified (`docker info` exits 1) |
| `e30495f` | `start.sh` uses `.venv`, binds loopback, cleans up its children; `bin/` archived | `tests/test_start_sh.py` (12) |
| `530ba73` | ruff safe autofix | G3 |
| `df66881` | `pre-commit` installed with pins that match the venv | no test file |
| `e7a053b` | remaining ruff findings fixed by hand | G3 |
| `62c7e58` | security bumps: pyjwt, multipart, requests, click, anyio, msgpack, `react-router-dom` | no test file; G1 and G2 run on the bumped set |
| `638779c` | fastapi 0.141.1 with starlette 1.3.1; route table test reads the flattened routes | `tests/test_route_shadowing.py` (9) |
| `a0d3a30` | tenancy and hardening: job ownership, operator gate, beat path confinement, per user crate keys, account purge, admin takedown, lifespan sweep and drain | `tests/test_tenancy_jobs.py` (22), `tests/test_operator_gate.py` (18), `tests/test_beat_paths.py` (9), `tests/test_crate_keys.py` (8), `tests/test_account_lifecycle.py` (9), `tests/test_admin_takedown.py` (15), `tests/test_hardening.py` (23), `tests/test_lifespan.py` (15), `tests/test_legal_phase8.py` (7), `tests/test_user_scope.py` (19) |
| `e251f22` | song B is stretched toward song A's tempo | `tests/test_render_e2e.py` (15) |
| `185bfc7` | pitch shift only when the Camelot pair is not already compatible | `tests/test_key_shift_rules.py` (10) |
| `23402fd` | every reader resolves audio through `audio_source` | `tests/test_audio_sources.py` (7) |
| `42dd8c2` | failures logged at DEBUG or swallowed now surface | `tests/test_silent_failures.py` (15) |
| `0f4b1f3` | `/features` endpoint, feature aware UI, jobs UX, vault credits, account and legal pages, API contract fixes | `tests/test_features_endpoint.py` (14), `tests/test_generative_urls.py` (3), `tests/test_job_meta_request.py` (5), `tests/test_legal_text.py` (5); vitest `features.test.tsx` (13), `jobsux.test.tsx` (6), `vault_account_legal.test.tsx` (10), `contracts.test.tsx` (37) |
| `842fc16` | root error boundary and a global request error toast | vitest `errors.test.tsx` (10) |
| `32bceb4` | local JWKS server and token minter for hosted mode checks | `tests/test_local_jwks.py` (8) |
| `9c3fcae` | uvicorn runs with `--no-proxy-headers` so the DMCA limiter keys on the socket peer | no test file; `start.sh:273` and `:278` carry the flag |
| `42dfd79` | licence text fallback matches whole phrases and denies "all rights reserved" | `tests/test_license_text.py` (9), `tests/test_license.py` (8) |
| `664a18a` | ingest never targets the library root, fails on a Demucs failure, gates before the shortcut, gains `--dry-run` | `tests/test_ingest_guards.py` (18) with `tests/fixtures/ingest/` |
| `d53592d` | crate ids match the API, deck teardown swallows only `AbortError`, `/ai-lab` follows the generative flag | vitest `crates.test.tsx` (1), `WaveformDeck.test.tsx` (2), `AppShell.test.tsx` (2), `features.test.tsx` (13) |
| `a4e8b50` | deploy scripts that run, env names the code reads, drain on shutdown, cookie domain | `tests/test_deploy_scripts.py` (25), `tests/test_env_aliases.py` (10), `tests/test_live_verify.py` (17), `tests/test_storage_isolation.py` (9); vitest `supabase.test.ts` (7) |
| `aef279f` | legacy docs moved to `archive/docs/legacy/` | docs only |

## Honest gaps (still open)

- **Supabase.** No project exists. Auth is proven only with synthetic RS256 tokens and
  the local JWKS tool; the two account isolation run through the real login UI and the
  browser session cookie on SSE are not done.
- **Hetzner, volume, DNS.** Not provisioned. `/health/ready` from the public internet,
  a `systemctl restart` drain and a `pg_dump` restore are unchecked.
- **Cloudflare Pages.** No project; the frontend is built locally only.
- **SMTP.** DMCA reports are stored; the email path is a logged stub until
  `REMIXMATE_SMTP_*` is set.
- **Managed Postgres.** No hosted instance. The container path was not rerun in this
  pass (`docker info` exits 1), and the Docker image build is unverified for the same
  reason.
- **Push, CI on a PR, merge.** `hosted-app` is 24 commits ahead of `origin/hosted-app`
  and 53 ahead of `main`; CI has not run on the hardened branch.
- **Catalog below the floor.** `ls catalog` lists 3 tracks against a floor of 5. Every
  ccMixter content download fails: `grep -c "403 Client Error: Forbidden"` over the
  ingest log in the backups directory prints 56.
- **Library full mixes.** `find library -maxdepth 2 -name full_enhanced.wav` finds 5
  files across 588 song folders; a prune call during the hosted checks removed the rest
  and they have no backup. Renders, recommend and the index read stems.
- **Frontend advisory.** `npm audit --omit=dev` reports 2 moderate findings in
  `react-router` (`GHSA-wrjc-x8rr-h8h6`, `GHSA-337j-9hxr-rhxg`); the fix is the
  `react-router-dom` 7 major.
- **Systemd unit proxy flag.** `app/deploy/remixmate.service` `ExecStart` lacks the
  `--no-proxy-headers` flag that `start.sh` passes.
- **Torch upgrade.** `requirements.txt` pins `torch==2.7.1` and `torchaudio==2.7.1`,
  coupled to Demucs.
- **Tier 3 engine items.** Not started: transition effects on the stem blend path,
  preview using the full render's window and stems, chain entry never before the
  previous exit, stem similarity on the transition windows, bridge intensity applied
  once, and a fade on a truncated last song in chain renders (the two song path fades
  a truncated song B at `scripts/api/task_modules/remix.py:436`; the chain loader at
  `:566` applies no fade).
- **Licence file.** `LICENSE` is MIT in a private repo meant for sale; a proprietary
  notice is undecided.
- The v1 catalog fills from ccMixter and Internet Archive only. FMA and Musopen are
  descoped on purpose, not a gap; `classify_license` still knows both.

## Sources

- `app/DEPLOYMENT_PLAN.md`: the phase roadmap and per phase verify blocks
- `app/deploy/PROVISION.md`: the operator runbook for the pending live steps
- `PROJECT_OVERVIEW.md`: capability map for the single user app
