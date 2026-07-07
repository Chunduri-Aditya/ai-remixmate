# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A real-time DJ engine built end-to-end in Python + React. Core capability: take two songs, analyze
them for BPM/key/bar structure, lock downbeats at the sample level, time-stretch to match tempo,
apply a stem-aware crossfade (drums/bass/vocals fade independently), optionally synthesize a bridge
beat, then master the output to −14 LUFS broadcast standard (ITU-R BS.1770-4).

Wrapped around the transition engine: a setlist planner (Camelot wheel + BPM continuity + energy arc
+ Markov blending), a lyric-matching "wordplay" layer (Genius API), a FastAPI backend with an async
job queue + SQLite persistence + SSE live stream, and a React frontend (Vite + TypeScript) with 8
pages plus a floating PiP DJ widget.

## Running locally

Activate the venv first — all Python tooling (pytest, uvicorn, scripts) lives in `.venv/` (Python 3.12):

```bash
source .venv/bin/activate
```

```bash
./start.sh               # install deps + start FastAPI (8000) + React (5173)
./start.sh --skip-setup  # skip pip/npm install on fast restarts
./start.sh api           # API only (for GitHub Pages widget mode)
./start.sh ui            # legacy Streamlit UI on :8501
docker compose up        # containerized (GPU not forwarded by default)
```

- React UI: http://localhost:5173 · API docs: http://localhost:8000/docs · SSE: http://localhost:8000/events/stream
- Config precedence: `config.yaml` (base, committed) → `config.local.yaml` (gitignored overrides) →
  env vars `REMIXMATE_<SECTION>_<KEY>` (e.g. `REMIXMATE_AUDIO_SAMPLE_RATE=48000`). Never commit keys
  to `config.yaml`.

## Commands

**Backend tests:**
```bash
pytest                                                    # full suite
pytest -m "not dj_analysis"                               # skip librosa/numba-dependent tests
pytest -m "not dj_analysis and not integration" \
  --ignore=tests/e2e_test_suite.py --ignore=tests/test_audio_source.py   # ~462 tests, safe on any machine
pytest tests/test_behavioral.py                           # behavioral correctness (asserts what you *hear*)
pytest tests/test_foo.py::TestClass::test_name            # single test
bash tests/smoke_e2e.sh                                   # live smoke test (API must already be running)
```
Tests requiring librosa are marked `@pytest.mark.dj_analysis` and auto-skip (see `tests/conftest.py`)
if librosa/numba fails to initialize (happens on some machines due to numba cache issues) — this is
not a real failure. `tests/test_behavioral.py` is the regression guard for audible-bug fixes; don't
delete tests there just because they look redundant with shape-only tests elsewhere.

**Lint / format / typecheck (Python):**
```bash
ruff check .          # pyflakes + pycodestyle + isort, line-length 100
black .               # formatter, line-length 100
mypy scripts/         # ignore_missing_imports=True (audio libs lack stubs)
pre-commit run --all-files
```

**Frontend** (`cd frontend`):
```bash
npm run dev           # vite dev server
npm run build          # tsc && vite build
npm run lint            # eslint, --max-warnings 0
npm run typecheck       # tsc --noEmit
npm test                # vitest
```

## Architecture

```
scripts/
├── api/
│   ├── main.py              # FastAPI app — lifespan captures event loop, CORS, request-ID middleware
│   ├── routes.py             # Aggregates 12 routers into one APIRouter; ORDER MATTERS (see Gotchas)
│   ├── jobs.py                # Write-through SQLite job store (data/jobs.db) + in-memory dict
│   ├── schemas.py             # Pydantic request/response models
│   ├── routers/                # system, library, downloads, stems, analysis, remix, generative,
│   │                           #   spotify, jobs, crates, events, setlist
│   └── task_modules/           # Async task functions run inside submit_job()'s ThreadPoolExecutor
├── core/                        # ~46 modules — the audio engine
│   ├── dj_engine.py             # Renderer: beat-grid lock, stem crossfade, dynamic EQ, bass ramp
│   ├── dj_analysis.py           # SongStructure/TransitionPlan, phrase-boundary + bar-grid snapping
│   ├── classic_engine/          # Vendored AI_REMIXMATE_BEST_MIXING_ALGO mix path (frozen — see
│   │                            #   Key design decisions → Two mixing engines)
│   ├── beat_tracker.py          # BeatTracker protocol: librosa backend + optional Beat This! (ISMIR 2024)
│   ├── key_detection.py         # Krumhansl-Schmuckler + Camelot wheel (canonical CAMELOT/NOTE_NAMES)
│   ├── tiv_scoring.py            # Tonal Interval Vector harmonic scoring (Bernardes et al. 2016)
│   ├── mastering.py              # ITU-R BS.1770-4 LUFS + true-peak limiter + per-stem-type LUFS norm
│   ├── beat_synth.py              # Procedural bridge-beat synthesis, 6 genre presets, Strudel export
│   ├── stems.py                   # Demucs stem separation wrapper
│   ├── setlist_planner.py          # Greedy optimizer (Camelot/BPM/energy) + Markov blending
│   ├── energy_profiler.py           # Essentia arousal/valence, numpy fallback
│   ├── wordplay.py                   # Genius API lyric n-gram matching between adjacent setlist tracks
│   ├── cue_export.py                  # rekordbox XML + Serato GEOB cue export
│   ├── crate_digger.py                 # CLAP 512-D semantic search over the library (35-D fallback)
│   ├── music_index.py                   # 35-dim numpy embedding index, JSON-persisted, no FAISS
│   ├── style_transfer.py / inpainting.py # DAC-token style transfer / VampNet inpainting (optional[generative] extra)
│   ├── library.py                        # Song library index, self-healing adopt/unregister
│   ├── gpu.py                              # MPS / CUDA / CPU auto-detection
│   ├── paths.py                             # Canonical path constants + sanitize_song_name()
│   └── config.py                             # YAML config loader, see precedence above
└── ui/app.py                     # Streamlit dashboard (legacy; React is primary)

frontend/src/
├── pages/                # 8 pages: MissionControl, LibraryAtlas, MixDeck, SetBuilder,
│                         #   SignalSearch, AILab, MixVault, Operations (+ Widget)
├── shell/                # AppShell (3-zone grid), LeftRail, RightInspector
├── stores/appStore.ts    # Zustand: nav, health, live jobs, activity log
├── lib/api.ts             # Thin fetch wrapper for all API namespaces; normalizeJob() lives here
├── hooks/useSSE.ts         # SSE connection + job store hydration; falls back to polling on disconnect
└── types/index.ts           # TypeScript types mirroring FastAPI Pydantic schemas (kept in sync manually)

tests/                    # pytest + librosa-probe conftest guard + e2e/smoke suites
ai_remixmate_feature_lab/ # Isolated experimental area — imports nothing from scripts/, not wired to UI by design
```

The runtime data layout (gitignored): `library/` for downloaded songs, `outputs/` for rendered mixes,
`data/` for SQLite stores + JSON caches + the audit log, `models/` for Demucs/CLAP weights.

## Key design decisions

**Two mixing engines** — `dj.mixing_engine` in config selects the render path for the remix/chain/
preview tasks. `classic` (the default) is `scripts/core/classic_engine/`, a frozen vendored copy of
the April 2026 AI_REMIXMATE_BEST_MIXING_ALGO snapshot (its own analysis, planner, renderer, and
mastering at −8 LUFS) — kept byte-faithful to the reference renders, so never fold current-engine
fixes into it. `current` is the live `dj_engine.py` path described below. Only
`scripts/api/task_modules/remix.py::_resolve_engine()` consumes the flag; analysis endpoints,
setlist planning, and library metadata always use the current modules. Job results carry an
`engine` field naming the path that rendered them.

**Beat-grid lock** — `dj_engine.py` computes bar-grid phases at cue points and applies a sample-level
correction (clamped to ±half a bar). Entry sample index compensates for the stretch ratio:
`entry_sample_b = int(entry_time_b * sr / stretch_ratio)`.

**Stem-aware crossfade** — When Demucs stems exist, each stem fades on its own envelope. Drums/bass
from Song B come in early; vocals are delayed. `_stem_bass_ramp()` does a true stem-level cosine-taper
bass mute at the swap point (no IIR bleed) rather than a linear crossfade. The standard
(medium-similarity) crossfade is equal-power (cos/sin), not equal-gain — two different songs are
uncorrelated, so amplitude-complementary curves dip ~3 dB at the midpoint. The stem-summed transition
window splices into the surrounding original mixes through ~30 ms equal-gain seam blends
(`_seam_blend()` — equal-gain there because seam material is correlated).

**Dynamic EQ** — A low-shelf filter pulls Song A's bass down as Song B's rises, preventing bass-clash
from simultaneous dual-track playback.

**Masking-aware multiband EQ** — `spectral_mixing.py` (Hafezi & Reiss 2015 offline variant) ducks a
track's mid/high bands where they mask content the other track needs, bidirectionally, during stem
transitions. Gated by `dj.masking_eq_enabled` (the A/B kill switch); the bass band is always
protected so `_stem_bass_ramp()`/the low-shelf stay the sole authority over low end. Gain curves are
tapered to 0 dB at window edges so the processed window splices transparently.

**Learned spectral cue matching** — `spectral_matching.py` scores Song B's phrase-boundary entry
candidates by mel-band *trajectory* similarity to Song A's outro (frame-by-frame spectral shape over
the whole window — one melspec per song, sliced per candidate) and moves
`plan.entry_bar_b`/`entry_time_b` to the best fit before rendering (engine-agnostic — it adjusts the
plan in `task_modules/remix.py`, never the engines, so the frozen classic package stays untouched).
The per-band weights train online from 👍/👎 verdicts (`POST /dj-remix/feedback`, one-shot per render
session; UI buttons in MixDeck's Mix Ready card and the static UI result panel). State lives in
`data/spectral_match_model.json` + `spectral_match_history.json`; gated by `dj.spectral_matching`.

**Dynamic spectral sync** — `spectral_sync.py` runs post-render/pre-master on full remix outputs: a
time-varying multiband EQ over the overlap window that eases the mix's spectrum from A's outro
profile to B's profile (cosine target ramp, per-band gains capped at `dj.spectral_sync_max_db`,
temporally smoothed, edge-tapered to 0 dB, bass band below `dj.bass_crossover_hz` untouched — the
engine's bass swap keeps low-end authority). Full renders only: previews are the bare overlap
segment and chains don't expose window positions. Gated by `dj.spectral_sync`.

**Harmonic compatibility** — Two independent systems exist and are used for different things: Camelot
wheel + `camelot_modulation()` classification (9 named modulation types, each with a psychoacoustic
cost) for setlist planning; `tiv_scoring.py`'s Tonal Interval Vector for a continuous [0.36, 1.0]
harmonic score (the TIV floor with binary chroma templates is ~0.36, not 0). Don't conflate the two —
Camelot answers "is this move common in DJ practice," TIV answers "how tonally close are these keys."

**Job store** — SQLite write-through with an in-memory dict for O(1) reads. Jobs recorded as RUNNING
at process start are rolled back to FAILED on restart. Cancel via `DELETE /jobs/{id}`.

**SSE** — `GET /events/stream` pushes heartbeat, job_created, job_updated, job_completed, job_failed,
library_changed. Frontend falls back to polling when SSE disconnects.

**Embedding index vs. CLAP index** — `data/music_index.json` (35-D numpy, always available) backs
`/library/similar/{name}`. `data/clap_index.npy` (512-D CLAP, optional) backs `/library/crate-search`
text/semantic queries. They coexist and are rebuilt independently.

## Common tasks

**Add a new API endpoint:**
1. Add the route to the appropriate router in `scripts/api/routers/`.
2. Add a task function in `scripts/api/task_modules/` if it needs to run async.
3. Create + submit the job via `job_store.create_job()` / `job_store.submit_job()`.
4. Add the client call to `frontend/src/lib/api.ts`.

**Add a new frontend page:**
1. Create `src/pages/MyPage.tsx` + `src/pages/MyPage.css` (follow the `PageBase.css` layout pattern —
   `page-base` → `page-base__header` → `page-base__body`; no Tailwind).
2. Add a `NavItem` in `shell/LeftRail.tsx`.
3. Add a `NavDestination` union member in `src/types/index.ts`.
4. Add a `<Route>` in `shell/AppShell.tsx`.

**Design tokens** live in `frontend/src/styles/tokens.css` (colors, spacing, type scale, radius) —
check there before hardcoding a color or spacing value.

**Stem work** (Demucs separation, separation config, missing/failed stems, library pruning) — the
`stem-processing` project skill (`.claude/skills/stem-processing/`) carries the procedures and
auto-fires; its rules are authoritative for those tasks.

## Gotchas

- **New config.yaml keys need a matching dataclass field.** `_dict_to_config()` in
  `scripts/core/config.py` silently drops any YAML key not declared on the section's dataclass
  (and any top-level section not declared on `AppConfig`) — guarded `getattr` reads then always
  return their defaults with no error. `tests/test_config_drift.py` guards this; add the field
  and the YAML key in the same change.
- **Router registration order matters.** `routes.py` registers `analysis` before `library` on purpose
  — `library`'s dynamic `GET /library/{name}` would otherwise shadow `analysis`'s static
  `GET /library/crate-search` (FastAPI matches routes in registration order). If a new static route
  under an existing dynamic-prefix router 404s or hits the wrong handler, check registration order
  first, and add a case to `tests/test_route_shadowing.py`.
- **`job_store.submit_job()` runs the task in a `ThreadPoolExecutor`.** Don't do async I/O inside task
  functions; use `asyncio.run()` if you need it there.
- **Job progress units differ by transport.** SSE frames and the frontend's normalized `Job` type use
  0–100. The REST `GET /jobs`/`GET /jobs/{id}` endpoints return 0–1. `normalizeJob()` in `api.ts`
  handles the conversion — don't re-derive it ad hoc in a component.
- **`JobResponse.status` is `str`, not the `JobStatus` enum.** The internal enum is lowercase
  (`pending`, `running`, `done`, `failed`, `cancelled`); REST/SSE both normalize to uppercase
  (`PENDING|RUNNING|COMPLETED|FAILED|CANCELLED`) via `_norm_status()` in `jobs.py`.
- **`StructuredLogger.warning(msg, extra=None)`** takes a dict as its second arg, not printf-style
  varargs. Never call `logger.warning("...%s", exc)` — the exception gets treated as the `extra` dict
  and crashes inside Python's logging internals. Use an f-string or pass `exc` inside a dict.
- **SSE from worker threads** — the event loop is captured once in `main.py`'s lifespan via
  `asyncio.get_running_loop()` and closed over by `_sync_emit`. Never call `asyncio.get_event_loop()`
  from inside a `ThreadPoolExecutor` worker — raises `RuntimeError` on Python 3.12 with no running
  loop in that thread.
- **Camelot semitone table** — `CAMELOT` + `NOTE_NAMES` in `key_detection.py` are the single source of
  truth; `pitch_shift_for_camelot()` derives from them. Don't add a second inline mapping — it will
  drift and produce wrong pitch shifts.
- **`favoritesApi.list`** — backend `GET /favorites` returns `{ songs: string[], count: number }`, not
  a bare array. The frontend call unwraps `.songs`; don't reintroduce a raw `get<string[]>`.
- **Serato cue export requires `.mp3`** — `export_serato_markers()` raises `ValueError` for WAV/FLAC;
  the GEOB binary format has no equivalent container. Use rekordbox XML export for non-MP3 tracks.
- **CLAP model auto-downloads ~300MB** on first `crate_digger` use unless `models.clap_model` points
  to a local checkpoint in config. First call takes 30–60s on a slow connection.
- **Per-stem LUFS targets** (`data/stem_lufs_targets.json`) must be generated once via
  `POST /library/calibrate-lufs` before `normalize_stems_to_corpus_targets()` uses corpus-derived
  values; otherwise it silently falls back to `_STEM_FALLBACK_TARGETS`.
- **Demucs runs on CPU by default** — set `separation.device: mps` in config for Apple Silicon. Full
  separation takes 2–5× real time on CPU.
- **`config.local.yaml` is gitignored** — never commit API keys to `config.yaml`.
- **Vite proxy** rewrites `/api/*` → `http://localhost:8000/*` in dev. In production (GitHub Pages),
  set `VITE_API_BASE=http://localhost:8000` since there's no dev proxy.
- **`ai_remixmate_feature_lab/`** is intentionally isolated — it has its own test suite and imports
  nothing from `scripts/`; it is not wired into the UI by design, not because it's abandoned.

## Where history lives

Don't add dated "Session Update" logs to this file — that's what `CHANGELOG.md` and
`SESSION_HANDOFF.md` are for. `IMPROVEMENTS.md` / `IMPROVEMENTS_V2.md` track the staged feature
roadmap; `ARCHITECTURE.md` has a deeper system map if you need more than the summary above.
