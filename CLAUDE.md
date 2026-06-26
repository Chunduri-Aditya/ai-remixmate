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
