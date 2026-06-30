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
pytest                          # all tests (226 as of June 28, 2026)
pytest -m "not dj_analysis"     # skip librosa/numba-dependent tests (safe in all envs)
pytest tests/test_behavioral.py # behavioral correctness tests (36 — catch audible bugs)
pytest tests/test_core_modules.py
bash tests/smoke_e2e.sh         # live smoke test (API must be running)
```

Tests requiring librosa are marked `@pytest.mark.dj_analysis` and auto-skipped if librosa fails to initialize (numba cache issues on some machines).

`tests/test_behavioral.py` contains the **behavioral** test suite added June 28, 2026. These assert *what you hear*, not just dtype/shape. Every test there corresponds to a bug that was invisible to the shape-only suite. Don't remove them — they're the regression guard for the D1 correctness fixes.

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

**Logging** — `StructuredLogger` (from `logging_utils.get_logger`) has signature `warning(msg, extra_dict=None)`, NOT `warning(msg, *args)`. Never call `logger.warning("...%s", exc)` — pass `exc` as an f-string or into a dict: `logger.warning(f"...{exc}")`. The second positional arg is taken as the `extra` dict; passing an exception crashes at the `for key in extra:` line in Python's logging internals.

**SSE from worker threads** — Always capture the event loop in `lifespan` with `asyncio.get_running_loop()` and close over it. Never call `asyncio.get_event_loop()` from inside a ThreadPoolExecutor worker — raises `RuntimeError` on Python 3.12 with no running loop in thread.

**Camelot semitone table** — The canonical source is `CAMELOT` + `NOTE_NAMES` in `key_detection.py`. Don't maintain a separate inline dict anywhere else — it will drift and produce wrong pitch shifts. `pitch_shift_for_camelot()` now derives from those constants.

**JobResponse.status** — The REST serializer (`job_to_response`) and SSE both now emit uppercase normalized values: `PENDING | RUNNING | COMPLETED | FAILED | CANCELLED`. The internal `JobStatus` enum still uses lowercase `done` etc. — `_norm_status()` bridges them. The `JobResponse.status` field is typed as `str`, not `JobStatus`.

---

# Session Update: June 29, 2026 (session 3) — Stage 3 Improvements + Library Atlas Bug Fix ✅

## What Happened

Fixed Library Atlas crash and implemented all three Stage 3 engineering improvements.

| Item | Status | Files |
|------|--------|-------|
| Library Atlas crash (`new Set()` on non-iterable) | ✅ | `frontend/src/lib/api.ts` |
| Stage 3C: rekordbox XML + Serato GEOB cue export | ✅ | `cue_export.py` (new), `routers/library.py` |
| Stage 3B: Essentia energy arc upgrade | ✅ | `energy_profiler.py` (new), `setlist_planner.py` |
| Stage 3A: Crate Digger — CLAP 512-D semantic search | ✅ | `crate_digger.py` (new), `routers/analysis.py` |
| 26 cue_export tests | ✅ | `tests/test_cue_export.py` (new) |
| 25 energy_profiler tests | ✅ | `tests/test_energy_profiler.py` (new) |
| 22 crate_digger tests | ✅ | `tests/test_crate_digger.py` (new) |

### Bug Fixed

**`favoritesApi.list` shape mismatch** (`frontend/src/lib/api.ts:240`)
- Backend `/favorites` returns `{ songs: string[], count: number }` — an object
- Frontend typed and consumed it as `string[]` directly
- `new Set(favorites)` → `TypeError: object is not iterable (Symbol.iterator)`
- Fix: `.then((r) => r.songs ?? [])` in the `list` call unwraps the payload

### Key additions

**`scripts/core/cue_export.py` (new):**
- `export_rekordbox_xml()` — rekordbox 6+ compatible XML with HOT CUE (up to 8)
  and MEMORY CUE (phrase boundaries) markers; TEMPO beat-grid element included
- `export_serato_markers()` — GEOB Serato Markers2 ID3 tag writer for .mp3 files
  (requires `pip install mutagen`); reverse-engineered binary format with cos-scaled slots
- `export_cues()` — dispatcher for both formats; format string is case-insensitive
- `_build_serato_markers2_payload()` — pure-struct binary builder (no mutagen for tests)

**`routers/library.py` (updated):**
- `GET /library/{name}/export-cues?fmt=rekordbox|serato` — reads `analysis.json` for
  phrase boundaries + BPM, writes temp file, streams as attachment, cleans up via
  `BackgroundTask`. 422 returned when Serato requested without .mp3 source.

**`scripts/core/energy_profiler.py` (new):**
- `EnergyFeatures` dataclass: rms_energy, spectral_centroid, dynamic_range, arousal, valence, backend
- `_numpy_profile()` — always-available backend: RMS + spectral centroid blend → arousal proxy
- `_essentia_profile()` — optional Essentia backend; tries MusiCNN ArousalValence model,
  falls back to feature blend when essentia-tensorflow absent
- `profile_energy(audio, sr, backend="auto")` — dispatches by backend/availability
- `enrich_track_node(track, audio, sr)` — sets `track.energy` and `track.arousal_predicted`

**`scripts/core/setlist_planner.py` (updated):**
- `TrackNode.arousal_predicted: Optional[float] = None` — new field (Stage 3B)
- `transition_cost()` energy path: prefers `arousal_predicted` over `.energy` when set

**`scripts/core/crate_digger.py` (new):**
- `CrateResult` dataclass: name, score, bpm, key, camelot, energy, backend
- `CrateDigger` class — thread-safe, lazy-loaded singleton
  - `index_library(library_dir, progress_cb, force)` — embeds all songs via CLAP;
    falls back to 35-D music_index.py when laion_clap absent; incremental (skips existing)
  - `find_similar(query_name, query_audio, query_text, k, camelot_filter, bpm_range)` —
    cosine similarity over unit-norm embeddings; supports combined audio+text queries;
    keyword fallback for text when CLAP absent
  - `_text_keyword_fallback()` — word-overlap matching on song names
  - `get_stats()` — n_songs, backend, dim, built_at
  - Index persisted at `data/clap_index.npy` + `data/clap_index_meta.json`
- `get_digger()` — module-level singleton

**`routers/analysis.py` (updated):**
- `GET /library/crate-search?q=&name=&k=&camelot=&bpm_min=&bpm_max=` — CLAP semantic search
- `POST /library/build-clap-index?force=` — async job to build/update CLAP index

## Current State

```
Branch: main
Tests: 374+ pass ✅ (11 pre-existing failures in TestGenreDetection + TestRateLimiting, unchanged)
New files: scripts/core/cue_export.py, scripts/core/energy_profiler.py, scripts/core/crate_digger.py,
           tests/test_cue_export.py, tests/test_energy_profiler.py, tests/test_crate_digger.py
Next: Stage 4 items — Vocal Analyzer (CREPE), CUE-DETR, B-Roll matching
      Or: router TestClient smoke tests for new endpoints
      See IMPROVEMENTS_V2.md for full roadmap
```

### Gotchas added

```
**favoritesApi shape** — GET /favorites returns {songs, count}, not a plain array.
Frontend must unwrap .songs. Fixed in api.ts; do not re-introduce the raw get<string[]>.

**Serato GEOB requires .mp3** — export_serato_markers() raises ValueError for WAV/FLAC.
The Serato binary format has no FLAC/WAV equivalent. Use rekordbox for non-MP3 tracks.

**CLAP auto-download** — _load_clap_model() calls model.load_ckpt() with no path,
which auto-downloads ~300 MB to ~/.cache/. Set models.clap_model in config.yaml to a
local path to avoid re-downloading. First call takes 30–60 s on slow connections.

**CLAP index vs music_index** — data/clap_index.npy is separate from data/music_index.json.
Both can coexist. /library/similar/{name} still uses music_index.py (35-D).
/library/crate-search uses crate_digger.py (CLAP 512-D or 35-D fallback).

**enrich_track_node side effect** — modifies track.energy in-place. If you still want
the original Spotify energy after enrichment, save it before calling enrich_track_node.
```

---

# Session Update: June 29, 2026 (session 2) — Stage 2 Improvements ✅

## What Happened

Implemented IMPROVEMENTS_V2.md Stage 2 (beat tracking + bar-grid cue snapping + stem bass muting).

| Item | Status | Files |
|------|--------|-------|
| BeatTracker Protocol interface + BeatResult dataclass | ✅ | `beat_tracker.py` (new) |
| LibrosaBeatTracker — wraps librosa, estimates downbeats | ✅ | `beat_tracker.py` |
| BeatThisTracker — Beat This! (ISMIR 2024), falls back gracefully | ✅ | `beat_tracker.py` |
| `get_tracker(backend)` / `get_configured_tracker()` factory | ✅ | `beat_tracker.py` |
| `SongStructure.downbeat_times` field | ✅ | `dj_analysis.py` |
| `_analyze_impl` → `get_configured_tracker().track()` | ✅ | `dj_analysis.py` |
| `_snap_to_bar_grid()` — SSM boundaries snapped to 8/16/32-bar grid | ✅ | `dj_analysis.py` |
| `_detect_phrase_boundaries()` now accepts + uses downbeat_times | ✅ | `dj_analysis.py` |
| `_stem_bass_ramp()` — cosine taper true stem muting at swap point | ✅ | `dj_engine.py` |
| `render_stem_blend()` bass path → `_stem_bass_ramp()` (no IIR bleed) | ✅ | `dj_engine.py` |
| 24 BeatTracker tests (15 pass, 9 skip on machines without librosa) | ✅ | `tests/test_beat_tracker.py` (new) |
| 9 bar-grid snapping tests | ✅ | `tests/test_beat_tracker.py` |
| 21 stem bass ramp behavioral tests | ✅ | `tests/test_stem_bass_ramp.py` (new) |

### Key additions

**`scripts/core/beat_tracker.py` (new):**
- `BeatResult` dataclass: `beat_times`, `beat_frames`, `downbeat_times`, `bpm`, `sr`, `backend`
- `BeatResult.nearest_downbeat(t)` / `nearest_downbeat_at_or_after(t)` — bar-aware helpers
- `LibrosaBeatTracker.track()` — downbeats estimated as beat[::4] from librosa output
- `BeatThisTracker.track()` — model-predicted downbeats via `beat-this` package; falls back to librosa if not installed; writes temp WAV → reads beats → cleans up
- `get_tracker(backend)` — accepts `"librosa"`, `"beat_this"`, `"auto"` (prefers beat_this if available)
- `get_configured_tracker()` — reads `config.yaml: analysis.beat_backend`

**`scripts/core/dj_analysis.py` (updated):**
- `SongStructure.downbeat_times: list` field added (default `[]`)
- `_analyze_impl()` replaced bare `librosa.beat.beat_track()` with `get_configured_tracker().track()`
- `struct.downbeat_times` populated from `BeatResult.downbeat_times`
- `_snap_to_bar_grid(boundary_times, downbeat_times, preferred_lengths_bars=[8,16,32])` — pure-numpy bar-grid snapper; scores candidates by phrase-length alignment + proximity; deduplicates + sorts output
- `_detect_phrase_boundaries()` accepts `downbeat_times` kwarg; passes raw SSM boundaries through `_snap_to_bar_grid()` when available

**`scripts/core/dj_engine.py` (updated):**
- `_stem_bass_ramp(stem_a_bass, stem_b_bass, swap_sample, ramp_samples)` — module-level DSP helper
  - Song A: cosine taper 1→0 over ramp window ending at swap_sample; hard zero after
  - Song B: hard zero before swap_sample; cosine taper 0→1 over ramp window after
  - `ramp_samples=0` → hard instantaneous cut (backward compatible)
  - cos² (Hann) taper: derivative=0 at endpoints → no audible click on sustained bass
- `render_stem_blend()` bass path: removed inline linspace fade; now calls `_stem_bass_ramp(ramp_samples=int(0.10*sr))` and `continue`s past generic stem A/B slice code

## Current State

```
Branch: main
Tests: 305+ pass ✅
New files: scripts/core/beat_tracker.py, tests/test_beat_tracker.py, tests/test_stem_bass_ramp.py
Next: Stage 3 — paper draft (ISMIR 2027 target), router/endpoint smoke tests, openapi-typescript CI
      See IMPROVEMENTS_V2.md for the full staged roadmap
```

### Gotchas added

```
**BeatThisTracker temp WAV** — BeatThisTracker writes audio to a tmp WAV at 16kHz
(beat-this expected sample rate), runs inference, then os.unlink().  On crash the
temp file is not cleaned up.  Use LibrosaBeatTracker if temp-file hygiene matters.

**_snap_to_bar_grid search window** — defaults to ±2 bars.  If a raw SSM boundary
falls more than 2 bars from any downbeat (unusual), it is returned unchanged rather
than snapped.  Extremely short songs (<8 bars) may produce no snapping.

**_stem_bass_ramp ramp clamping** — ramp_samples is clamped to
min(swap_sample, n - swap_sample) to prevent array overruns on short transitions.
At swap_sample=1 the ramp degrades to a hard cut regardless of requested width.
```

---

# Session Update: June 29, 2026 — Stage 1 Improvements ✅

## What Happened

Implemented IMPROVEMENTS_V2.md Stage 1 (analysis foundation upgrades) and wrote comprehensive
tests. All new code passes; no regressions in existing suite.

| Item | Status | Files |
|------|--------|-------|
| FxNorm per-stem-type LUFS normalization | ✅ | `mastering.py`, `dj_engine.py`, `routers/library.py` |
| TIV harmonic scoring (Bernardes et al. 2016) | ✅ | `tiv_scoring.py` (new), `dj_analysis.py` |
| 28 new mastering tests (FxNorm) | ✅ | `tests/test_mastering.py` |
| 25 new TIV tests | ✅ | `tests/test_tiv_scoring.py` (new) |
| IMPROVEMENTS_V2.md staged plan | ✅ | `IMPROVEMENTS_V2.md` (new) |

### Key additions

**`mastering.py`:**
- `analyze_library_stem_targets(library_dir)` — scans all stems, returns corpus-mean LUFS per type
- `normalize_stems_to_corpus_targets(stems, targets)` — per-stem-type LUFS normalization (FxNorm scheme)
- `load_stem_targets()` / `save_stem_targets()` — JSON cache at `data/stem_lufs_targets.json`
- `_STEM_FALLBACK_TARGETS` — empirical defaults: drums -18.5, bass -21.0, vocals -19.5, other -21.5

**`dj_engine.py`:** `render_stem_blend()` now calls `normalize_stems_to_corpus_targets()` with
corpus targets (falls back to flat -20 LUFS if cache absent).

**`routers/library.py`:** `POST /library/calibrate-lufs` — async job that runs corpus analysis
and writes `data/stem_lufs_targets.json`. Run once after library setup.

**`tiv_scoring.py` (new):**
- Pure-numpy TIV implementation — no git submodule; paper math is self-contained
- `tiv_from_chroma(chroma)` → complex (6,) TIV vector
- `tiv_harmonic_score(chroma_a, chroma_b)` → float [0,1]
- `compare_tiv_vs_camelot(...)` → {tiv_score, camelot_adjacent, camelot_distance}
- `all_key_compatibility_matrix()` → 24×24 symmetric matrix

**`dj_analysis.py`:** `TransitionPlan.tiv_compatibility: Optional[float]` — populated when
`song.mean_chroma` is available from `analyze_structure()`.

## Current State

```
Branch: main
Tests: 279+/279+ pass  (226 existing + 25 TIV + ~28 new mastering; 8 skip on machines without soundfile)
New files: scripts/core/tiv_scoring.py, tests/test_tiv_scoring.py, IMPROVEMENTS_V2.md
Next: Stage 2A — Beat This! BeatTracker interface
      Stage 2B — bar-grid cue point snapping
      Stage 2C — true stem-level bass muting
      (See IMPROVEMENTS_V2.md for full roadmap)
```

### Gotchas added

```
**Per-stem LUFS targets** — data/stem_lufs_targets.json must be generated via
POST /library/calibrate-lufs before normalize_stems_to_corpus_targets() uses
corpus-derived values.  Without it, falls back to _STEM_FALLBACK_TARGETS.

**TIV scoring** — tiv_scoring.py is self-contained (no TIVlib git submodule).
Implements Bernardes et al. 2016 TIS math directly (30 lines of numpy).
tiv_harmonic_score(C major, G major) ≈ 0.52 — this is correct TIS behavior
(tonal-center distance, not note-overlap).
Calibrated values with binary scale templates:
  same-key = 1.00 | relative major/minor (same note set) = 1.00
  adjacent 5th ≈ 0.52 | tritone ≈ 0.47
All 24×24 key pairs land in [0.36, 1.0] — the TIV floor with binary templates
is ~0.36, not 0.0 (zero would require maximally anti-correlated tonal content).
```

---

# Session Update: June 28, 2026 — Audit Bug Fixes ✅

## What Happened

Deep audit pass across correctness, schema drift, test coverage, and production readiness.
Seven bugs fixed; 36 behavioral tests added.

| Bug | Severity | File | Fix |
|-----|----------|------|-----|
| Song B audible in "solo A" first half | **critical** | `dj_engine.py:_apply_dynamic_eq_fade` | Zero head for `direction="in"`; match `render_chain` semantics |
| Clash-path bass swap disabled at worst moment | major | `dj_analysis.py:plan_transition` | Compute consonance/shortening **before** `EQPlan` construction |
| Camelot semitone table wrong (A-ring +1, B-ring 1B–7B −4) | major | `key_detection.py:pitch_shift_for_camelot` | Derive from `CAMELOT`+`NOTE_NAMES`; delete inline dict |
| SSE broadcasts silently dropped on Python 3.12 | major | `api/main.py:_sync_emit` | Capture loop with `get_running_loop()` in lifespan; close over it |
| SQLite connection leak (one per progress tick) | major | `api/jobs.py` | `contextlib.closing()` on every `_get_conn()` |
| ETA `ZeroDivisionError` when elapsed=0 | minor | `api/jobs.py:update_job` | `max(elapsed, 1e-6)` |
| `_cancelled` set grows unbounded | minor | `api/jobs.py` | `discard(job_id)` on completion and failure |
| REST status `"done"` ≠ SSE `"COMPLETED"` | critical | `api/jobs.py`, `api/schemas.py` | `job_to_response` uses `_norm_status`; `JobResponse.status: str` |
| `logger.warning("...%s", exc)` crashes StructuredLogger | — | `api/jobs.py` (8 sites) | All converted to f-strings |
| Dead LP coefficients computed but never applied | minor | `dj_engine.py:render` | Removed |
| `swap_sample` division by zero if `transition_bars=0` | minor | `dj_engine.py:render` | `max(transition_bars, 1)` guard |

## Current State

```
Branch: main
Commit: 5ef1b87
Tests: 226/226 pass (190 existing + 36 new behavioral)
New test file: tests/test_behavioral.py
Next: Stage 3B — paper draft (ISMIR 2027 target)
      Router/endpoint smoke tests (routers/ still zero TestClient coverage)
      openapi-typescript in CI (schema drift prevention)
      setlist_planner tests (flagship feature, still zero coverage)
```

## Remaining Test Gaps (from audit Dimension 3)

| Module | Status |
|--------|--------|
| `mastering.py` known-signal tests | ✅ `test_mastering.py` exists |
| `key_detection` Camelot/harmonic utils | ✅ Added to `test_behavioral.py` |
| `jobs.py` lifecycle | ✅ Added to `test_behavioral.py` |
| render() B-silence behavioral | ✅ Added to `test_behavioral.py` |
| `setlist_planner.py` | ❌ Zero tests — `MarkovTransitionModel`, `optimize`, `export_csv` |
| `routers/` + `task_modules/` | ❌ Zero `TestClient` endpoint tests |
| `render_chain()` behavioral | ❌ Only shape tests exist |

---

# Session Update: June 27, 2026 — Improvements Complete ✅

## What Happened

All Stage 1, Stage 2, and Stage 3A improvements from IMPROVEMENTS.md implemented and committed.

| Gap | Status | Commit |
|-----|--------|--------|
| 1A Per-stem LUFS normalization | ✅ | `dj_engine.py:render_stem_blend()` |
| 1B Bass swap envelope | ✅ | `dj_engine.py:render_stem_blend()` per-stem loop |
| 1C Chain render stem path | ✅ | `render_chain(stems_dirs=)` |
| 2A Pitch shift audio | ✅ | `key_detection.py:pitch_shift_audio()` + `TransitionPlan.suggested_pitch_shift` |
| 2B SSM-novelty boundaries | ✅ | `dj_analysis.py:_detect_phrase_boundaries()` |
| 2C EDMA/EDMM key profiles | ✅ | `detect_key(profile=)` + `/analyze?key_profile=` |
| 2D Psychoacoustic consonance | ✅ | `key_detection.py:psychoacoustic_consonance()` |
| 3A GiantSteps benchmark | ✅ | `scripts/benchmarks/giantsteps_eval.py` |

**Test status: 190/190 pass** (up from 167 at session start)

## Current State

```
Branch: main
Tests: 190/190 pass
New modules: pitch_shift_audio, psychoacoustic_consonance, _detect_phrase_boundaries,
             normalize_stems_to_target (wired), giantsteps_eval (synthetic + real modes)
Next: Stage 3B — paper draft (ISMIR 2027 target)
      Stage 4 candidate — global energy-arc TSP optimizer (setlist_planner.py)
```

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
pytest tests/ -q -m "not dj_analysis"  # 226 tests pass

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
Last updated: June 29, 2026 (third session)
Branch: main
Tests: 374+ pass ✅ (use: pytest -m "not dj_analysis and not integration" --ignore=tests/e2e_test_suite.py --ignore=tests/test_audio_source.py)
Build status: Clean (frontend + backend); Library Atlas crash fixed
API status: SSE functional, job queue operational; new endpoints:
            GET /library/{name}/export-cues, GET /library/crate-search, POST /library/build-clap-index
Next phase: Stage 4 items (Vocal Analyzer, CUE-DETR, B-Roll matching)
            Router TestClient smoke tests for new endpoints
            openapi-typescript CI (schema drift prevention)
            See IMPROVEMENTS_V2.md for the full staged roadmap
```
