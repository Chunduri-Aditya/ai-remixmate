# PRD

_What AI RemixMate is and for whom, on the `hosted-app` branch. Committed. Sources: `README.md`, `SHOWCASE.md`, `app/STATUS.md`, `app/DEPLOYMENT_PLAN.md`, `ai_remixmate_feature_lab/docs/00_product_direction.md`. TBD marks what no source says. Known defects live in `docs/agents/BACKLOG.md`, not here._

## Problem
- Automated DJ transitions that stop at a crossfade sound automated: beats drift apart, two basslines stack into mud, and amplitude complementary fades dip in loudness at the midpoint (`README.md`, "The parts that are actually interesting").
- A hosted, multi user version needs audio it may store, separate into stems and derive from. The single user app pulled commercial audio through a downloader, which fails that test; the hosted plan cuts the catalog to Creative Commons and public domain audio only (`app/DEPLOYMENT_PLAN.md`, "Context").
- Who has the problem beyond the author's portfolio and the planned beta: TBD.

## Users
- Listener, signed in to the hosted app: picks tracks from one shared catalog, previews and renders transitions and chains, plans setlists, and keeps renders in Mix Vault under per user storage with retention (`app/STATUS.md`, Phases 3, 4 and 6).
- Operator: curates the catalog with the ingest tool `scripts/download.py` (licence gate, Demucs, analysis), runs the operator gated routes such as stem split and storage prune, and acts on DMCA reports through the admin takedown routes in `scripts/api/routers/admin.py`, which have no UI (`DEVELOPING.md`, "Frontend pages"; `app/STATUS.md`, Phases 1 and 8).
- Local single user: with auth off every request runs as the `local` user and no token is needed (`README.md`, "Hosted mode").
- Audience at launch: a low traffic private beta with waitlist signups, not open signup (`app/DEPLOYMENT_PLAN.md`, "Context").
- What these users do today instead: TBD.

## Goals
- Mixes that do not sound automated: beat grid lock, a stem aware equal power crossfade with a bass swap, masking aware EQ, and mastering to the loudness target `audio.target_lufs` in `config.yaml` (`README.md`). How each step works: `docs/DJ_THEORY.md`, with the drift listed under DOC-02.
- A multi user hosted product: Creative Commons only catalog, per user tenancy, Supabase auth, Postgres, a small VPS behind Caddy, the UI on Cloudflare Pages, inside the hosting budget in `app/DEPLOYMENT_PLAN.md` "Context" (`SHOWCASE.md`, "Branches").
- A phase counts as done only when its Verify block in `app/DEPLOYMENT_PLAN.md` is ticked against real output; `app/STATUS.md` is the ledger.
- A measurable listening quality target for renders: TBD.

## Non goals
- Generative features in v1: the MusicGen, VampNet and DAC routers stay in the code behind `REMIXMATE_GENERATIVE_ENABLED` (`app/DEPLOYMENT_PLAN.md`, "Cut list").
- In app track acquisition: no YouTube or SoundCloud downloading and no Spotify OAuth or import; the download path is removed on this branch and `main` still carries it (`SHOWCASE.md`, "Known gaps"). Spotify stays as read only metadata lookups (`README.md`).
- Free Music Archive and Musopen ingest: descoped on purpose; v1 fills from ccMixter and the Internet Archive (`app/STATUS.md`, "Honest gaps").
- Per user libraries and uploads, per user learned weights, collaborative crates, an admin dashboard UI (`app/DEPLOYMENT_PLAN.md`, "Cut list").
- Runtime GPU, Redis, a multi worker API or horizontal scaling; Demucs runs once per track at operator ingest (`app/DEPLOYMENT_PLAN.md`, "Context" and "Cut list").
- Deploying the old Python UI under `scripts/ui/` (`app/DEPLOYMENT_PLAN.md`, "Cut list").
- Copying the surface of existing DJ software (turntable skins, branded layouts, hardware mapped assumptions). This comes from the frozen lab's direction doc; production imports none of the lab and rebuilt only the remix recipe concept in `scripts/core/remix_recipe.py` (`SHOWCASE.md`).

## Features
Each line names the check that proves it today. Where a backlog id sits beside a check, the check is green while the defect stands.

- Two song transition, `POST /dj-remix`, with preview `POST /dj-remix/preview`: `tests/test_render_e2e.py`, `tests/test_key_shift_rules.py`. Open defects: MX-02, MX-03, MX-17.
- Chains, `POST /dj-chain`: the chain cases in `tests/test_render_e2e.py`, including `tests/test_render_e2e.py::test_chain_middle_song_is_not_skipped`. Open defects: MX-09, MX-24.
- Compatibility and setlist planning, `POST /compatibility` and `/setlist/optimize`: `tests/test_compatibility_score.py`, `tests/test_setlist_planner.py`. Open defect: MX-04.
- Mastering: `tests/test_mastering.py`.
- Catalog ingest behind the licence gate, credits on every mix: `tests/test_license.py`, `tests/test_license_text.py`, `tests/test_ingest_guards.py`, `tests/test_mix_credits.py`.
- Auth, tenancy and the operator gate: `tests/test_auth.py`, `tests/test_tenancy_jobs.py`, `tests/test_user_scope.py`, `tests/test_operator_gate.py`.
- Feature flags for the UI, `GET /features`, and the generative gate: `tests/test_features_endpoint.py`, `tests/test_generative_gate.py`.
- Per user mix retention: `tests/test_mix_retention.py`.
- Legal pages, DMCA intake, account export and deletion, admin takedown: `tests/test_legal_phase8.py`, `tests/test_account_lifecycle.py`, `tests/test_admin_takedown.py`. Open defect: OPS-01.
- Web UI pages (`DEVELOPING.md`, "Frontend pages"): the vitest suites in the `__tests__` folders under `frontend/src/`; no browser suite exists. `tests/e2e_test_suite.py` is a legacy audio pipeline suite CI ignores and `tests/smoke_e2e.sh` is a live API smoke; neither drives the UI. Open defects: the UI ids in `docs/agents/BACKLOG.md`.
- Everything else the app does: `README.md`, "What it does".
