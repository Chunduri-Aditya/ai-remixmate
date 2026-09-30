# AI RemixMate: showcase

The short version and a local demo. For the capability map, endpoint list and the hard audio problems of the single user app on `main`, read [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md); it predates `hosted-app`, so its page count and download notes describe `main`. Hosted migration status lives in [`app/STATUS.md`](app/STATUS.md), and the plan in [`app/DEPLOYMENT_PLAN.md`](app/DEPLOYMENT_PLAN.md).

## What it is

A DJ engine built in Python and React that takes two tracks and renders a beat locked, key aware transition from Demucs stems, mastered to an integrated loudness target of −14 LUFS (`audio.target_lufs` in `config.yaml`). Around the transition engine sit a setlist planner (Camelot harmonic scoring, BPM continuity, energy arc shaping, a Markov model over historical setlists, Exportify CSV import), a lyric phrase wordplay layer over the Genius API, and a FastAPI backend with an async job queue, SQLite or Postgres persistence, plain text logging behind a log injection filter, an `X-Request-ID` on every response, and a JSONL audit trail (`data/audit.jsonl`).

The parts worth pointing at:

- **Beat grid lock** with the time stretch ratio compensated into the entry index, the error the ear catches and a waveform view does not. Song B is stretched toward song A's tempo, covered by an end to end render test (`tests/test_render_e2e.py`).
- **Equal power crossfading** with fade curves per stem (drums, bass, vocals fade separately), not equal gain.
- **MIR beyond `librosa.beat_track`**, all in production modules: a swappable beat tracker (`scripts/core/beat_tracker.py`, `analysis.beat_backend` set to `librosa`, `beat_this` or `auto`, where `auto` uses Beat This! from ISMIR 2024 when installed and falls back to librosa), Tonal Interval Vector harmonic scoring beside the Camelot wheel (`scripts/core/tiv_scoring.py`), and CLAP 512 dimension semantic crate search that falls back to the 35 dimension handcrafted index when `laion_clap` is not installed. `ai_remixmate_feature_lab/` is a frozen sandbox: production imports none of it, and only the plain language remix recipe concept was rebuilt from scratch in `scripts/core/remix_recipe.py`.

## Branches

| Branch | What it is | State |
|---|---|---|
| `main` | Single user local app: FastAPI in one process, a thread pool job queue, SQLite, a local music library on disk | 47 commits. Still carries the old download path and no auth; not merged with `hosted-app` |
| `hosted-app` | The multi user hosted product: Creative Commons only catalog, per user tenancy, Supabase auth, Postgres, a small VPS behind Caddy, the UI on Cloudflare Pages | 53 commits ahead of `main`. Phases 0 to 9 below |

## Hosted migration phases (`hosted-app`)

| Phase | What | Status |
|---|---|---|
| 0 | Test isolation guardrails | Done (`304da7e`); widened in `52cd9ad` so every pytest run isolates storage unless it opts out |
| 1 | Catalog teardown and enforced CC licence gate | Done in code. A seed catalog of 3 CC tracks was ingested live; growing it needs a new fetch path because ccMixter refuses content downloads |
| 2 | Generative routes behind a flag, slimmer runtime | Done |
| 3 | Per user data through a `current_user` contextvar | Done |
| 4 | Supabase RS256 JWKS auth and tenancy | Code done. Isolation checks passed in a local hosted mode simulation with a local JWKS and two users; the live Supabase project and the login UI flow are operator pending |
| 5 | SQLite to Postgres through SQLAlchemy Core | Code done, verified against a real Postgres 16 container (`app/DEPLOYMENT_PLAN.md` Phase 5 verify block); the managed Postgres run is operator pending |
| 6 | Per user mix retention | Done in code (`b8317f2`, `tests/test_mix_retention.py`, 14 tests) |
| 7 | Deploy and infra | Deploy files done in code (`8dbfe81`, `fb1a5f9`, `a4e8b50`: Caddy, systemd, bootstrap and backup scripts, CI); the VPS, DNS and Cloudflare Pages are operator pending |
| 8 | Legal and product surface | Done in code (`2919ec2`, `tests/test_legal_phase8.py`, 7 tests: legal pages, DMCA intake, account export and deletion, admin takedown); SMTP for DMCA email is operator pending |
| 9 | Hardening pass | Done: 24 commits after `fb1a5f9` covering toolchain and CI, dependency security bumps, tenancy and security fixes, render correctness, feature aware UI, a local JWKS minter, ingest guards and deploy script fixes. Commit list: `git log --oneline fb1a5f9..hosted-app` |

## Local demo (no cloud)

```bash
cd ~/Desktop/Projects/ai-remixmate

# backend and frontend gates
.venv/bin/python -m pytest tests -q
cd frontend && npx tsc --noEmit && npx vitest run && npx eslint . --max-warnings 0 && npx vite build && cd ..

# run it: API on 127.0.0.1:8000 (docs at /docs), React UI on :5173, DJ widget at /widget
./start.sh
```

Measured on 2026-09-30 on `hosted-app` at `bd1c446`, the cookie method guard (2 backend tests and 1 vitest more than `aef279f`):

| Check | Result |
|---|---|
| backend suite (`pytest tests -q`) | 1960 passed, 1 skipped |
| test functions under `tests/` | 1102 |
| frontend (`npx vitest run`) | 138 passed across 18 files |
| song folders in `library/` | 588 |
| one local two song render, `POST /dj-remix` to completed (from the hardening run report) | 81.7 s |
| one render over the CC seed catalog in hosted mode (same report) | 75.24 s |

## Done

- Transition engine: beat grid lock, equal power stem crossfade, phase vocoder time stretch, optional numpy bridge beat synthesis, LUFS mastering
- Setlist planner: Camelot harmonic scoring, energy arc shaping, Markov ordering, Exportify CSV import
- FastAPI backend, async job queue, SQLite or Postgres persistence, logging with a log injection filter, JSONL audit trail
- React and TypeScript UI: Mission Control, Library Atlas, Mix Deck, Set Builder, Signal Search, Mix Vault, Operations, account and legal pages, plus a floating DJ widget; AI Lab appears only when the generative flag is on
- `hosted-app` Phases 0 to 9 in code, as in the table above

## Known gaps

- **No live hosted deployment.** Supabase, the VPS, DNS, Cloudflare Pages, SMTP and managed Postgres are operator work with a monthly cost, not code changes.
- **Seed catalog is 3 tracks.** ccMixter refuses every content download, so catalog growth needs another fetch path or source.
- **Postgres** was verified on a container during Phase 5; the hosted mode simulation ran on SQLite.
- `main` still exposes the old download path with no auth on any endpoint; `hosted-app` is the answer and is not merged back.
- Engine items not built yet: transition effects on the stem blend path, preview parity with the full render, chain renders that hard cut a truncated last song (the two song path fades a truncated song B over 3 s at `scripts/api/task_modules/remix.py:436`; the chain loader at `:566` loads each song with `duration=max_s` and applies no fade).

## Layout

`~/Desktop/Projects/ai-remixmate`, default branch `main`, active branch `hosted-app`, repository <https://github.com/Chunduri-Aditya/ai-remixmate>
