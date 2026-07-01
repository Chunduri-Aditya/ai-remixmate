# AI RemixMate — Context

> Single source of truth for current project state. Verified against the actual codebase
> (`ls`/`grep`/`pytest`), not just against other docs — several other `.md` files in this repo
> describe stale architecture (Streamlit-primary, old API shapes) or duplicate content under
> misleading names. Where this doc and another doc disagree, trust this one; if this doc and the
> code disagree, trust the code and fix this doc.
>
> Last verified: June 30, 2026. Read `FURTHER_STUDIES.md` for everything NOT yet built.

---

## 1. What this project is

A real-time DJ engine, Python + React, single maintainer, personal portfolio project. Give it two
songs: it analyzes BPM/key/bar structure, time-stretches and phase-locks Song B's downbeat onto
Song A's bar grid at the sample level, renders a stem-aware crossfade (drums/bass/vocals fade on
independent envelopes when Demucs stems exist), optionally synthesizes a procedural bridge beat,
and masters the output to ITU-R BS.1770-4 loudness standards with a true-peak limiter.

Wrapped around the transition engine: a setlist planner (Camelot-aware greedy optimizer + Markov
blending + energy arc shaping), a lyric-matching "wordplay" layer (Genius API), AI generative
tools (style transfer, inpainting, tokenization via DAC/MusicGen/VampNet), semantic search (35-D
handcrafted vectors + optional CLAP 512-D), and a job-queue-backed FastAPI service with a React
frontend.

**Current primary frontend is React (Vite + TypeScript), not Streamlit.** The root `README.md`
still describes Streamlit as primary and React as "next" — that's stale. React has 9 fully-built
pages; Streamlit (`scripts/ui/app.py`) is legacy, reachable only via `./start.sh ui`. This doc
reflects the real current state; `README.md` needs a pass to catch up (see `FURTHER_STUDIES.md`).

---

## 2. Architecture (verified against `ls`, June 30 2026)

```
scripts/
├── api/
│   ├── main.py                 # FastAPI app — lifespan, CORS, request-ID middleware
│   ├── jobs.py                 # Write-through SQLite job store (data/jobs.db)
│   ├── routes.py                # Thin aggregator — includes all routers
│   ├── schemas.py               # Pydantic request/response models
│   ├── routers/                 # 13 routers: system, library, downloads, stems, analysis,
│   │                            #   remix, generative, spotify, jobs, crates, setlist,
│   │                            #   events, _helpers
│   └── task_modules/            # download, stems, remix, analysis, generative, lab
├── core/                        # 41 modules — see table below for the ones that matter
└── ui/app.py                    # Legacy Streamlit dashboard

frontend/src/
├── pages/      # 9 pages: MissionControl, LibraryAtlas, MixDeck, SetBuilder, SignalSearch,
│               #   AILab, MixVault, Operations, Widget
├── components/ # WaveformDeck, TransitionTimeline, CamelotWheel, RemixControls,
│               #   StemsPlayerRow, Toast, ShortcutsModal, PageErrorBoundary
├── shell/      # AppShell (3-zone grid), LeftRail, RightInspector
├── stores/appStore.ts   # Zustand: nav, health, live jobs, activity log
├── lib/api.ts            # Typed fetch wrappers for every API namespace
├── hooks/useSSE.ts       # SSE connection + job store hydration
└── types/index.ts        # TypeScript types mirroring FastAPI schemas (synced manually)

tests/            # 20 test files, ~520+ tests total (see Section 6)
```

### Core modules that matter

| Module | What it does | Status |
|---|---|---|
| `dj_engine.py` | Beat-grid lock, stem crossfade, dynamic EQ fade, render/render_chain/render_stem_blend | Implemented, behaviorally tested |
| `dj_analysis.py` | `SongStructure`, `TransitionPlan`, `analyze_structure()`, `plan_transition()`, bar-grid snapping | Implemented |
| `beat_tracker.py` | `LibrosaBeatTracker` / `BeatThisTracker` + `resolve_bpm_octave()` octave-error correction | Implemented (octave correction added June 30) |
| `mastering.py` | ITU-R BS.1770-4 LUFS + true-peak limiter with independent attack/release coefficients + FxNorm per-stem-type normalization | Implemented (attack/release split added June 30) |
| `key_detection.py` | Camelot wheel (single source of truth: `CAMELOT`/`NOTE_NAMES`), pitch shift, psychoacoustic consonance | Implemented |
| `tiv_scoring.py` | Tonal Interval Vector harmonic scoring (Bernardes 2016), self-contained (no TIVlib submodule) | Implemented |
| `track_metadata.py` | `MetadataClient.compatibility_score()` — SetFlow formula (harmonic/beat/energy/genre/timbral − vocal_clash) | Implemented (rewired to match `docs/DJ_THEORY.md` June 30) |
| `setlist_planner.py` | Weighted **greedy** optimizer (50/30/20 harmonic/BPM/energy), Markov blending, Exportify CSV import, energy arc shaping | Implemented — still greedy, not a global optimizer (see FURTHER_STUDIES) |
| `energy_profiler.py` | numpy + optional Essentia arousal/valence backends, feeds `TrackNode.arousal_predicted` | Implemented |
| `crate_digger.py` | CLAP 512-D semantic search with 35-D fallback | Implemented |
| `cue_export.py` | rekordbox XML + Serato GEOB ID3 cue export | Implemented |
| `wordplay.py` | Genius API lyric matching, NLTK n-gram similarity for mixtape-style transitions | Implemented |
| `vocal_analyzer.py` | CREPE F0 + vibrato/phrase/dynamics analysis on vocal stems | **Module implemented, NOT wired to any API route or to `compatibility_score()`'s vocal_clash_penalty** — see FURTHER_STUDIES |
| `music_index.py` | 35-dim numpy embedding index, JSON-persisted, no FAISS | Implemented |
| `style_transfer.py` | MusicGen-melody-conditioned style transfer | Implemented |
| `inpainting.py` | VampNet audio inpainting | Implemented |
| `codec_tokens.py` | DAC token extraction | Implemented |
| `beat_synth.py` | Procedural bridge-beat synth, 6 genre presets, Strudel export | Implemented |
| `analysis_pipeline.py` | Shared `run_song_analysis()` — writes `meta.json` + `analysis.json` | Implemented |
| `music_index.py` quick-index path | `_quick_features()` — fast 30s partial analysis on library ingest | Implemented, now octave-corrected (was the root cause of cached-vs-render BPM mismatches) |

**Not present anywhere in the codebase** (don't go looking): `scripts/ml/cue_detector.py` (CUE-DETR), `scripts/core/broll_matcher.py` (B-roll matching). Both are future work, see `FURTHER_STUDIES.md`.

---

## 3. API surface (verified against `scripts/api/routers/*.py`)

| Namespace | Key endpoints |
|---|---|
| Health/SSE | `GET /health/live`, `/health/ready`, `GET /events/stream` |
| Library | `GET /library`, `GET /library/{name}`, `DELETE /library/{name}`, `POST /library/init`, `POST /library/storage/evict` |
| Downloads | `POST /download`, `/download-playlist`, `/spotify/import`, `/spotify/import-playlist` |
| Stems | `POST /stems/split`, `/stems/split-batch`, `/stems/compress`, `/stems/compress-batch` |
| Analysis | `POST /analyze`, `POST /compatibility`, `GET /recommend/{name}`, `GET /library/similar/{name}` |
| Crate search | `GET /library/crate-search`, `POST /library/build-clap-index` |
| Cue export | `GET /library/{name}/export-cues?fmt=rekordbox\|serato` |
| LUFS calibration | `POST /library/calibrate-lufs` |
| Remix | `POST /dj-remix`, `POST /dj-remix/preview`, `POST /dj-chain` |
| Generative | `POST /ai/style-transfer`, `/ai/inpaint`, `/ai/tokenize`, `GET /ai/models` |
| Setlist | `POST /setlist/plan`, `GET /setlist/{id}`, `POST /setlist/import-csv`, `GET /setlist/camelot/modulation` |
| Jobs | `GET /jobs`, `GET /jobs/{id}`, `DELETE /jobs/{id}` (cancel) |
| Crates | Full CRUD on `/crates`, `/crates/{id}/songs` |
| Tags | `/library/{name}/tags`, `/tags` |
| Favorites | `GET /favorites` → `{ songs: string[], count: number }`, `/favorites/{name}` |

All mutating endpoints return `{ job_id: string }` (202 Accepted). Poll `/jobs/{id}` or subscribe to
SSE. `compatibility_score()`'s response now also includes `genre_proximity`, `timbral_similarity`,
`vocal_clash_penalty` (added June 30 — see Section 5).

**No endpoint exists yet for vocal_analyzer.py's output** — it's a fully-built, fully-tested module
with zero API surface. See `FURTHER_STUDIES.md`.

---

## 4. Frontend pages (all 9 are fully built, not stubs)

| Page | Route | Notes |
|---|---|---|
| Mission Control | `/mission-control` | Stats, live jobs, quick actions |
| Library Atlas | `/library-atlas` | Sortable table, search, filters, favorites |
| Mix Deck | `/mix-deck` | Dual deck, BPM/key/energy (per-song detail fetch, not bulk list), compat score, similar-song suggestions, waveform, preview + remix |
| Set Builder | `/set-builder` | Pool, ordered set, energy arc chart, chain remix |
| Signal Search | `/signal-search` | Similarity search, score rings, breakdown |
| AI Lab | `/ai-lab` | Style transfer, inpaint, tokenize |
| Mix Vault | `/mix-vault` | Audio player, download, metadata |
| Operations | `/operations` | Single/batch/playlist downloads |
| DJ Widget | `/widget` | Floating Document-Picture-in-Picture window for live mixing, deployable standalone to GitHub Pages against a local API |

Design tokens: `frontend/src/styles/tokens.css`. No Tailwind. Layout pattern for new pages:
`page-base` → `page-base__header` → `page-base__body`. Shell: 64px left rail + 316px right
inspector + canvas.

---

## 5. What changed most recently (June 30, 2026 session — DSP correctness + scoring formula)

Fixed three confirmed-live bugs from `REMIX_QUALITY_INSIGHTS.md` (findings #1, #2/#3, #4 — all now
closed):

1. **BPM octave errors.** Three independent raw `librosa.beat.beat_track()` call sites
   (`music_index.py:_quick_features()`, `track_metadata.py:LocalAnalysisProvider.from_file()`,
   `beat_tracker.py`'s two tracker classes) had zero octave-error handling, meaning the same song
   could get cached at one octave (e.g. 63.8 BPM) and re-analyzed at render time at another (129.2
   BPM). Added `beat_tracker.resolve_bpm_octave()` — autocorrelation-evidence scoring with a
   log-normal prior toward typical dance tempo (~122 BPM) — wired into all relevant call sites.
   Verified with synthetic click-track tests; naive "highest autocorrelation peak wins" is itself
   octave-biased (caught this directly: it flipped a correct 120 BPM reading to 60 BPM on a clean
   synthetic signal), which is why the prior term exists.
2. **Stretch-ratio clamp logging.** `dj_engine.py:_time_stretch()` now logs a distinct warning when
   a clamp hugs the `_MIN_STRETCH`/`_MAX_STRETCH` boundary (signature of a BPM octave error) vs a
   routine mid-band clamp.
3. **Limiter attack/release.** `mastering.py:apply_limiter()` used one symmetric `release_ms` time
   constant for both gain-reduction engagement and recovery, so a fast transient took the same
   ~50ms to attenuate as a fully-limited section took to recover — the look-ahead window's warning
   was wasted, and the hard-clip safety net became the primary limiter on dense transient material
   instead of a rare fallback. Split into independent fast-attack (~2ms) / slow-release (~50ms)
   coefficients via a numba-JIT'd sequential envelope follower (`_smooth_gain_envelope`, falls back
   to pure Python if numba unavailable). Verified zero clipping on a dense-transient stress test
   that previously produced a 227,029-sample clipping defect.
4. **`compatibility_score()` formula.** Replaced an undocumented `key*0.45 + bpm*0.40 + energy*0.15`
   formula with the actual SetFlow weights documented in `docs/DJ_THEORY.md` section 3:
   `0.35*harmonic_match + 0.25*beat_alignment + 0.15*energy_smoothness + 0.15*genre_proximity +
   0.10*timbral_similarity − vocal_clash_penalty`. Added `vocal_clash_penalty` (was nowhere in the
   codebase despite being named as the single most audible DJ-mixing failure mode) and
   `genre_proximity`/`timbral_similarity` (new `TrackMetadata.vocal_density` /
   `spectral_centroid_hz` fields, populated via Demucs-stem energy ratio when available or a
   spectral-band proxy otherwise). Also fixed: the `/compatibility` router was never reading
   `energy_mean` back from cached `meta.json`, so every library-song pair silently compared energy
   0.5-vs-0.5 (always "perfect") regardless of actual energy difference.

Verification: rebuilt a minimal dependency set in the sandbox (numpy/scipy/numba/librosa/pydantic/
fastapi/pytest — the project's own `.venv`/`remix-env` were broken symlinks in that environment).
417 pre-existing tests pass unmodified; added 39 new tests across `test_mastering.py`,
`test_beat_tracker.py`, and new `test_compatibility_score.py` (reproducing the exact tritone-clash
and 2.45x-BPM-gap bugs found live in an earlier session). One pre-existing, unrelated test failure
surfaced (`total_bars` property-setter bug in `test_behavioral.py::TestClashPathBassSwap`) — not
touched, flagged for whoever picks it up.

---

## 6. Testing

```bash
source .venv/bin/activate   # or remix-env/ — always activate first

pytest                                                                          # everything
pytest -m "not dj_analysis and not integration" \
  --ignore=tests/e2e_test_suite.py --ignore=tests/test_audio_source.py          # SAFE default — run this
pytest tests/test_behavioral.py     # 36 behavioral correctness tests — never remove
bash tests/smoke_e2e.sh             # live E2E (needs API running)
```

Tests needing librosa/numba are marked `@pytest.mark.dj_analysis`, auto-skipped if the environment
can't init them (`tests/conftest.py` probes once at collection time). 20 test files exist, covering
mastering, beat tracking, TIV scoring, cue export, energy profiler, crate digger, setlist planner,
stem bass ramp, vocal analyzer, dj_engine guards, compatibility score, router smoke tests, and the
36-test behavioral suite. `tests/test_router_smoke.py` exists (TestClient endpoint coverage) but
its breadth against every Stage 2/3 endpoint hasn't been audited — see `FURTHER_STUDIES.md`.

No CI workflow runs pytest or TypeScript checks (`.github/workflows/` has only `codeql.yml` and
`pages.yml`). Tests are run manually, not gated on push/PR.

---

## 7. Running locally

```bash
source .venv/bin/activate      # always first — all tooling lives in here
./start.sh                     # install deps + start API (8000) + React UI (5173)
./start.sh --skip-setup        # fast restart, no reinstall
./start.sh api                 # API only (for GitHub Pages widget mode)
./start.sh ui                  # legacy Streamlit (8501) — not the primary UI anymore
docker compose up              # containerized (GPU not forwarded by default)
```

React UI `http://localhost:5173` · API docs `http://localhost:8000/docs` · SSE
`http://localhost:8000/events/stream`.

Config: `config.yaml` → `config.local.yaml` (gitignored, never commit keys) → env vars
(`REMIXMATE_<SECTION>_<KEY>`). Key knobs: `analysis.beat_backend` (librosa/beat_this/auto),
`separation.device` (cpu/mps/cuda), `generative.musicgen_model`, `output.target_lufs` (-14.0
default for general output; `/dj-remix` specifically targets -8.0 LUFS, intentional — don't
"fix" this without checking `task_modules/remix.py`), `library.max_size_gb` (LRU eviction).

---

## 8. Gotchas (verified still true)

**Job system.** `job_store.submit_job()` runs tasks in a `ThreadPoolExecutor` — never do async I/O
inside task functions. SSE event loop is captured once in `lifespan` via
`asyncio.get_running_loop()` and closed over; never call `asyncio.get_event_loop()` from inside a
worker thread (raises `RuntimeError` on Python 3.12 with no running loop in-thread). Job progress
is 0.0–1.0 internally and in REST `/jobs`, 0–100 in SSE frames and the frontend's normalized `Job`
type — `normalizeJob()` in `api.ts` bridges this; don't double-multiply. `JobResponse.status` is
typed `str`, normalized uppercase (`PENDING|RUNNING|COMPLETED|FAILED|CANCELLED`); the internal
`JobStatus` enum is lowercase (`done` etc.) — `_norm_status()` bridges them.

**Logging.** `StructuredLogger.warning(msg, extra_dict=None)` — never `warning(msg, *args)`. The
second positional arg is read as the `extra` dict; passing an exception there crashes inside
Python's logging internals. Always f-string the exception in.

**SQLite.** Always wrap connections in `contextlib.closing()`.

**Camelot wheel.** Single source of truth is `CAMELOT` + `NOTE_NAMES` in `key_detection.py`. Don't
maintain a separate inline semitone table anywhere — that's exactly how the A-ring/B-ring bug
happened (fixed in the June 28 audit pass).

**BPM.** As of June 30, `beat_tracker.resolve_bpm_octave()` is the single place octave-error
correction happens — if you add a new raw `librosa.beat.beat_track()` call anywhere, route its
output through this function or you'll reintroduce the cached-vs-render mismatch bug.

**Frontend ↔ backend schema sync.** `frontend/src/types/index.ts` mirrors Pydantic schemas
*manually* — no codegen, no CI check. When you add/change a Pydantic field, update the TS type in
the same change or you'll get a silent runtime shape mismatch (this has happened at least twice:
`favoritesApi.list` returning `{songs, count}` typed as `string[]`, and `MixDeck` needing
per-song-detail fetch because the bulk list endpoint omits analysis fields).

**MixDeck BPM/key/energy.** `libraryApi.list()` (bulk) does NOT include analysis fields. Always use
`libraryApi.get(name)` (per-song detail) for BPM/key/energy/camelot.

**Demucs.** CPU by default; `separation.device: mps` for Apple Silicon (2-5x real time on CPU).

**Serato cue export.** `.mp3` only — raises `ValueError` for WAV/FLAC (no Serato GEOB equivalent
exists for those containers). Use rekordbox format for non-MP3 tracks.

**CLAP auto-download.** First call downloads ~300MB to `~/.cache/`. Set `models.clap_model` in
config to a local path to avoid re-downloading on every fresh environment.

**`enrich_track_node`** modifies `track.energy` in place — save the original Spotify energy first
if you need it after calling this.

---

## 9. Doc map (what to actually read, and what's stale)

| File | Status |
|---|---|
| **`CONTEXT.md`** (this file) | Current. Start here. |
| **`FURTHER_STUDIES.md`** | Current. Everything not yet built. |
| `SELF_IMPROVING_DJ_RESEARCH.md` | Current — standalone research proposal, referenced (not duplicated) from `FURTHER_STUDIES.md`. |
| `docs/DJ_THEORY.md` | Current, evergreen — the musicological/DSP reference (Camelot theory, SetFlow formula, phrase alignment, bass-swap timing). Cite this for "why," not "what's done." |
| `config.yaml` | Current — every tunable parameter, read the inline comments. |
| `PREREQUISITES.md` | Mostly current — correctly identifies React as primary, Streamlit as legacy (more accurate than root `README.md` on this point). |
| `SECURITY.md` | Current, static policy doc. |
| `README.md` (root) | **Stale on one point** — still calls Streamlit primary / React "next." Everything else (quick start, tech stack, "why I built this") is fine. Needs a one-section update, not a rewrite. |
| `CLAUDE.md` | Running changelog, mostly accurate but additive-only (never prunes). Good chronological record, bad single source of truth — that's what this file is for now. |
| `AGENT_PROMPT.md` / `AGENTS.md` | Agent-onboarding prompts, mostly accurate as of their write time but already had at least one stale claim (listed Stage 4A Vocal Analyzer as "not started" when `scripts/core/vocal_analyzer.py` already exists with full test coverage — just unwired to any API route). Treat as secondary to this file. |
| `AUDIT.md` + (mislabeled) old `CONTEXT.md` content | **Superseded.** All D1-critical findings were fixed in the June 28 "Audit Bug Fixes" pass per `CLAUDE.md`'s changelog. The old `CONTEXT.md` was actually audit-handoff notes (companion to `AUDIT.md`), not project context — that's why it got overwritten by this file. |
| `IMPROVEMENTS.md`, `IMPROVEMENTS_V2.md` | **Mostly superseded.** Stages 1-3 of both are done. Stage 4 (Vocal Analyzer, CUE-DETR, B-Roll, global energy-arc optimizer) is partially done — see `FURTHER_STUDIES.md` for the accurate remaining slice. |
| `REMIX_QUALITY_INSIGHTS.md`, `SESSION_HANDOFF.md` | **Mostly superseded.** Findings #1-4 fixed June 30 (Section 5 above). A few SESSION_HANDOFF items still need live browser verification — carried into `FURTHER_STUDIES.md`. |
| `claude-code-improvements.md` through `-7.md` (7 files) | **Superseded.** These are individual batch task-prompts from past sessions ("Batch 2 Improvements," "Batch 7: Render-Path & Client Hardening," etc.). Cross-checked against the actual frontend (`WaveformDeck.tsx`, `TransitionTimeline.tsx` both exist) and `dj_engine.py` (`_safe_ratio`/`_MIN_STRETCH` clamps exist) — confirmed implemented. Historical record only. |
| `docs/COMPASS_ARTIFACT.md` | **Mislabeled/duplicate.** `docs/README.md`'s own index describes this as "vision, scope, what's in and out" — it is not. The actual file content is a near-duplicate of `docs/TOKENIZATION_ROADMAP.md` (same title, same content: MIDI/spectral/neural tokenization reference). Needs a decision: delete, or rename+rewrite as the vision doc it's supposed to be. |
| `docs/TOKENIZATION_ROADMAP.md` | Reference material (tokenization math/papers/libraries), not a strict status doc. The generative pipeline it informed (DAC tokens, MusicGen style transfer, VampNet inpainting) is implemented and exposed via `/ai/*`. Keep as background reading, not a task list. |
| `docs/GUIDE.md`, `docs/QUICK_REFERENCE.md`, `docs/COMPREHENSIVE_DOCUMENTATION.md`, `docs/DOCUMENTATION_INDEX.md` | **Stale — describe the Streamlit-era architecture in deep technical detail** (`docs/GUIDE.md` alone has 5 direct Streamlit references in an architecture-overview context). Not verified line-by-line against current React-era code; treat as historical/legacy reference, not current documentation. |
| `docs/AI_RemixMate_Launch_Plan.md`, `docs/PORTFOLIO_INTEGRATION.md` | Business/marketing docs, not implementation status — orthogonal to this consolidation, left untouched. |
| `CHANGELOG.md` | Historical, accurate as a log, last entry dated April 2026 — doesn't cover anything from May/June. |

See `FURTHER_STUDIES.md` Section "Doc hygiene" for the specific cleanup recommendation (which of
the superseded files above are safe to delete vs. worth keeping as historical record).
