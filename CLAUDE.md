# AI RemixMate — Claude Code Prompt

## What this project is

A real-time DJ engine built end-to-end in Python + React. Core capability: take two songs, analyze them for BPM/key/bar structure, lock downbeats at the sample level, time-stretch to match tempo, apply stem-aware crossfade (drums/bass/vocals fade independently), optionally synthesize a bridge beat, then master the output to −14 LUFS broadcast standard.

Wrapped around that: a FastAPI backend with async job queue + SQLite persistence, SSE live stream, and a React frontend (Vite + TypeScript) with 8 pages.

## Architecture

```
scripts/
├── api/
│   ├── main.py              # FastAPI app — lifespan, CORS, request-ID middleware
│   ├── jobs.py              # Write-through SQLite job store (data/jobs.db)
│   ├── routes.py            # Thin aggregator — includes all routers
│   ├── routers/             # 11 domain routers (system, library, downloads, stems,
│   │                        #   analysis, remix, generative, spotify, jobs, crates, setlist)
│   ├── task_modules/        # Async task functions: download, stems, remix, analysis,
│   │                        #   generative, lab
│   └── schemas.py           # Pydantic request/response models
├── core/
│   ├── dj_engine.py         # DJ renderer — beat-grid lock, stem crossfade, EQ fade
│   ├── dj_analysis.py       # Song structure analysis, transition planning
│   ├── mastering.py         # ITU-R BS.1770-4 LUFS metering + true-peak limiter
│   ├── beat_synth.py        # Procedural bridge beat generator (6 genre presets)
│   ├── setlist_planner.py   # Weighted greedy set optimizer (Camelot, BPM, energy arc)
│   ├── music_index.py       # 35-dim embedding index (JSON-persisted, no FAISS)
│   ├── style_transfer.py    # AI style transfer via DAC tokens
│   ├── inpainting.py        # Audio inpainting via VampNet
│   ├── stems.py             # Demucs stem separation wrapper
│   ├── library.py           # Song library management
│   ├── paths.py             # Canonical path constants (outputs/, models/, data/)
│   └── config.py            # YAML config loader (config.yaml → config.local.yaml → env)
frontend/src/
├── pages/                   # 8 pages: MissionControl, LibraryAtlas, MixDeck,
│                            #   SetBuilder, SignalSearch, AILab, MixVault, Operations
├── shell/                   # AppShell (3-zone grid), LeftRail, RightInspector
├── stores/appStore.ts       # Zustand: nav, health, live jobs, activity log
├── lib/api.ts               # Thin fetch wrapper for all API namespaces
├── hooks/useSSE.ts          # SSE connection + job store hydration
├── types/index.ts           # TypeScript types mirroring FastAPI schemas
└── styles/tokens.css        # Full design token system (colors, spacing, type, radius)
```

## Key design decisions

**Beat-grid lock** — `dj_engine.py` computes bar-grid phases at cue points and applies a sample-level correction (clamped to ±half a bar). Entry sample index compensates for the stretch ratio: `entry_sample_b = int(entry_time_b * sr / stretch_ratio)`.

**Stem crossfade** — When Demucs stems exist, each stem fades on its own envelope. Drums/bass from Song B come in early. Vocals are delayed. This is what makes outputs sound intentional vs. automated.

**Dynamic EQ** — Low-shelf filter pulls Song A's bass as Song B's rises. Prevents bass-clash from dual-track playback.

**Embedding index** — 35-dimensional numpy vectors (BPM, key, energy, spectral features). JSON-persisted, no external vector DB. `/library/similar/{name}` returns k nearest neighbours.

**Job store** — SQLite write-through with in-memory dict for O(1) reads. Jobs recorded as RUNNING at process start are rolled back to FAILED on restart. Cancel via `DELETE /jobs/{id}`.

**SSE** — `GET /events/stream` pushes heartbeat, job_created, job_updated, job_completed, job_failed, library_changed. Frontend falls back to polling when SSE disconnects.

## Running locally

**Always activate the venv first** — all Python tooling (pytest, uvicorn, scripts) lives inside `.venv/`:

```bash
source .venv/bin/activate    # activate once per terminal session
```

Then:

```bash
./start.sh               # install deps + start FastAPI (8000) + React (5173)
./start.sh --skip-setup  # skip pip/npm install on fast restarts
./start.sh api           # API only (for GitHub Pages widget mode)
```

React UI:    http://localhost:5173  
API docs:    http://localhost:8000/docs  
SSE stream:  http://localhost:8000/events/stream

```bash
docker compose up        # containerized (GPU not forwarded by default)
```

If the venv doesn't exist yet (first-time setup), run `./start.sh setup` — it creates `remix-env/` automatically.

Config override: copy `config.yaml` → `config.local.yaml` and edit. Env vars override both: `REMIXMATE_<SECTION>_<KEY>`.

## API surface

| Namespace | Key endpoints |
|---|---|
| Health | `GET /health/live`, `/health/ready` |
| Library | `GET /library`, `GET /library/{name}`, `DELETE /library/{name}`, `POST /library/init` |
| Downloads | `POST /download`, `/download-playlist`, `/spotify/import` |
| Stems | `POST /stems/split`, `/stems/split-batch` |
| Analysis | `POST /analyze`, `/compatibility`, `GET /recommend/{name}`, `/library/similar/{name}` |
| Remix | `POST /dj-remix`, `/dj-remix/preview`, `/dj-chain` |
| AI | `POST /ai/style-transfer`, `/ai/inpaint`, `/ai/tokenize`, `GET /ai/models` |
| Jobs | `GET /jobs`, `GET /jobs/{id}`, `DELETE /jobs/{id}` (cancel) |
| Crates | Full CRUD on `/crates`, `/crates/{id}/songs` |
| Tags | `/library/{name}/tags`, `/tags` |
| Favorites | `/favorites`, `/favorites/{name}` |

All mutating endpoints return `{ job_id: string }` (202 Accepted). Poll `/jobs/{id}` or subscribe to SSE for updates.

## Frontend pages

| Page | Route | Status |
|---|---|---|
| Mission Control | `/mission-control` | ✅ Full — stats, live jobs, quick actions |
| Library Atlas | `/library-atlas` | ✅ Full — sortable table, search, filters, favorites |
| Mix Deck | `/mix-deck` | ✅ Full — dual deck, compat score, preview + remix |
| Set Builder | `/set-builder` | ✅ Full — pool, ordered set, energy arc, chain remix |
| Signal Search | `/signal-search` | ✅ Full — similarity search, score rings, breakdown |
| AI Lab | `/ai-lab` | ✅ Full — style transfer, inpaint, tokenize |
| Mix Vault | `/mix-vault` | ✅ Full — audio player, download, metadata |
| Operations | `/operations` | ✅ Full — single/batch/playlist downloads |
| DJ Widget | `/widget` | ✅ Floating PiP window for live mixing |

## Design system

All tokens in `frontend/src/styles/tokens.css`. Key values:

- **Accents**: amber (`--color-amber-500: #f59e0b`), ice (`--color-ice-400: #38bdf8`), green (`--color-green-500: #34d399`), crimson (`--color-crimson-500: #f87171`), violet (`--color-violet-400: #a78bfa`)
- **Surfaces**: void `#05050a` → base `#09090b` → surface `#111113` → elevated `#18181b` → overlay `#1c1c20`
- **Fonts**: Space Grotesk (display), Inter (UI), JetBrains Mono (mono)
- **Shell**: 64px left rail + variable right inspector (316px) + canvas

New pages follow the `PageBase.css` layout pattern (`page-base` → `page-base__header` → `page-base__body`). Each page gets its own CSS file. No Tailwind.

## Testing

```bash
pytest                          # all tests
pytest -m "not dj_analysis"     # skip librosa-dependent tests
pytest tests/test_core_modules.py
bash tests/smoke_e2e.sh         # live smoke test (API must be running)
```

Tests requiring librosa are marked `@pytest.mark.dj_analysis` and auto-skipped if librosa fails to initialize (numba cache issues on some machines).

## Common tasks

**Add a new API endpoint:**
1. Add route to the appropriate router in `scripts/api/routers/`
2. Add task function to `scripts/api/task_modules/` if async work is needed
3. Create a job via `job_store.create_job()` + `job_store.submit_job()`
4. Add the API client call to `frontend/src/lib/api.ts`

**Add a new frontend page:**
1. Create `src/pages/MyPage.tsx` + `src/pages/MyPage.css`
2. Add a `NavItem` entry to `shell/LeftRail.tsx`
3. Add a `NavDestination` union type in `src/types/index.ts`
4. Add a `<Route>` in `shell/AppShell.tsx`

**Run a one-off analysis:**
```python
from scripts.core.dj_analysis import analyze_structure
structure = analyze_structure("path/to/song.mp3")
```

**Check job queue:**
```bash
curl http://localhost:8000/jobs | python3 -m json.tool
```

## Gotchas

- `job_store.submit_job()` runs the task in a `ThreadPoolExecutor`. Don't do async I/O inside task functions — use `asyncio.run()` if you need it.
- The embedding index is rebuilt via `POST /index/rebuild`. It must be current before `/library/similar` returns meaningful results.
- Demucs runs on CPU by default. Set `separation.device: mps` in config for Apple Silicon. Full stem separation takes 2–5× real time on CPU.
- `config.local.yaml` is gitignored. Never commit API keys to `config.yaml`.
- The React Vite proxy rewrites `/api/*` → `http://localhost:8000/*` in dev. In production (GitHub Pages), set `VITE_API_BASE=http://localhost:8000`.
- Job progress is 0–100 in SSE frames and in the normalized frontend `Job` type. The REST `/jobs` endpoint returns 0–1 — `normalizeJob()` in `api.ts` handles the conversion.

---

# Session Update: June 26, 2026 — Merge Complete ✅

## What Happened

**Three branches consolidated into main via safe audit & merge:**

| Branch | Status | Result |
|--------|--------|--------|
| `batch7-render-client-hardening` | 4 commits ahead | ✅ Merged (136 files: UI + backend hardening) |
| `claude/recursing-leakey` | Identical to main | 🗑 Deleted (no new work) |
| `claude/elegant-williamson` | 2 commits behind | 🗑 Deleted (already ancestor) |

**Verification:** 167 tests pass ✅ | Frontend builds clean ✅ | Zero conflicts ✅

## Current State

```
Branch: main
Merge commit: b8bab12
Files changed: 136 (55 frontend, 17 backend, 64 tests/config/docs)
API: Running, SSE streaming, job queue functional
Tests: All green (pytest 167/167 pass)
Ready: Yes, deployable
```

## Next Phase: Improvements Planning

**Using Claude Sonnet for rapid analysis & implementation.**

### Recommended Focus Areas (Priority Order)

1. **Test Coverage** (High impact, quick win)
   - Identify untested modules: `task_instrument_lab()`, error paths in routers
   - Write tests for edge cases in stem crossfade
   - Target: >90% coverage
   - Effort: 4-6 hours

2. **Performance Verification** (Medium impact, data-driven)
   - Measure job throughput (target: >50 jobs/sec)
   - Verify SSE latency (<100ms)
   - Profile stem separation bottleneck
   - Effort: 2-3 hours

3. **Error Handling & UX** (Medium impact, user-facing)
   - Better error messages in API responses
   - User-friendly failure states in UI
   - Guidance on common issues (missing library songs, etc.)
   - Effort: 3-4 hours

4. **Documentation** (Low impact, high value long-term)
   - API endpoint guide (Swagger is auto-generated)
   - Component storybook (React components)
   - Deployment guide (Docker, environment setup)
   - Effort: 6-8 hours

5. **Type Safety** (Ongoing, prevent regression)
   - Automated sync between Pydantic ↔ TypeScript schemas
   - CI check for schema drift
   - Effort: 3-4 hours

### How to Use Sonnet

**Brief Sonnet with this context:**
```
You're working on AI RemixMate (real-time DJ engine).

CURRENT STATE:
- Main branch: 136 files, all tests pass
- 8-page React frontend + FastAPI backend (12 routers, job queue)
- See ARCHITECTURE.md for system design

YOUR TASK:
1. Audit [area: test-coverage|performance|error-handling|docs|type-safety]
2. Identify gaps with specific examples
3. Propose concrete improvements
4. Estimate effort (hours)
5. Provide implementation steps

See CLAUDE_CODE_TESTING_PROMPT.md Part 6 for gap analysis.
Use ARCHITECTURE.md as reference.
```

**Sonnet is great for:**
- Fast pattern recognition (finding untested code paths)
- Writing focused, practical improvements (tests, error messages)
- Refactoring with clarity (no overthinking, action-oriented)
- Rapid prototyping (test coverage, performance profiling)

## Supporting Documents

**Root-level reference files:**

1. **IMPROVEMENTS.md** ← **START HERE for new work**
   - Gap analysis vs. Spotify/Apple Music (June 2026)
   - Maps every gap to the exact file + function that needs to change
   - Stage 1 (wiring), Stage 2 (new implementations), Stage 3 (publication)
   - Prioritized checklist with effort estimates

2. **ARCHITECTURE.md**
   - Complete system map: 7 backend layers, 8 frontend layers
   - Dependency graph, design patterns, end-to-end flows
   - Use: Understand connections, trace data flow

3. **CLAUDE_CODE_TESTING_PROMPT.md** (in `/outputs/`)
   - 8 executable test sections
   - Architecture verification, integration tests, smoke tests, e2e flows, coverage analysis, performance tests
   - Use: Identify gaps, verify code quality

4. **TESTING_GUIDE.md** (in `/outputs/`)
   - How to run each test section
   - Recommended test runs (5 min to 60 min)
   - Cheat sheet of common commands
   - Use: Quick reference for testing

5. **MERGE_STRATEGY.md** (in `/outputs/`)
   - Safe branch merging process
   - Conflict resolution, automated merge script
   - Use: Reference for future multi-branch merges

## Quick Start Commands

### Start Development
```bash
# Terminal 1: Backend
python -m scripts.api.main

# Terminal 2: Frontend
cd frontend && npm run dev

# Terminal 3: Tests (watch)
pytest tests/ -v --looponfail
```

### Verify State
```bash
# Tests
pytest tests/ -q  # 167 tests pass

# Build
cd frontend && npm run build

# API health
curl http://localhost:8000/health/live
```

### Create Improvement Branch
```bash
# New feature branch
git checkout -b improvements/[name]

# Work + commit
git add . && git commit -m "Improve: [description]"

# Merge back when done
git checkout main
git merge improvements/[name]
```

## Key Architectural Points (for Sonnet)

**How things connect:**

User click → Page calls API → FastAPI router validates + creates job (202 Accepted) → Job submitted to ThreadPoolExecutor → Task runs in background, updates job store → SSE broadcasts job updates → useSSE() hook syncs to Zustand appStore → Pages subscribed to appStore re-render → User sees progress in real-time → Task completes → SSE broadcasts job_completed → UI updates with result

**Core patterns:**
- Job queue (ThreadPoolExecutor + SQLite) decouples requests from execution
- SSE + Zustand keep UI in sync without polling
- Stem-aware crossfade + beat-grid lock make output sound intentional
- 35-D embeddings enable fast similarity search (no external DB)
- TypeScript types mirror Pydantic schemas (manual sync required)

**Common pitfalls:**
- Schema drift between Pydantic and TypeScript (must keep in sync)
- Job queries not cached in memory (always O(1) via dict)
- Concurrent updates → write-through SQLite prevents race conditions

## Version Info

```
Last updated: June 26, 2026
Merge status: Complete ✅ (3 branches consolidated)
Test status: 167/167 pass ✅
Build status: Clean (frontend + backend)
API status: Running, SSE functional, job queue operational
Next phase: Improvements planning (Sonnet-assisted)
```

---

**Ready to proceed. Brief Claude Sonnet with this context for rapid improvements planning.**
