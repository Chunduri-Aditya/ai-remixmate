# Developing AI RemixMate

Practical notes for working in this codebase: how to run it, the design decisions that are not
obvious from the code, and the traps that have each cost real debugging time at least once.

For the project front door see [README.md](README.md). For hosted mode (auth, tenancy, deploy)
see [app/README.md](app/README.md) and [app/DEPLOYMENT_PLAN.md](app/DEPLOYMENT_PLAN.md). For what
changed when, see [CHANGELOG.md](CHANGELOG.md). Older design documents live in
`archive/docs/legacy/` and describe earlier states of the code.

---

## Running locally

All Python tooling (pytest, uvicorn, scripts) lives in `.venv/` on Python 3.12. `./start.sh`
creates it on first run; activate it for direct commands:

```bash
source .venv/bin/activate
```

```bash
./start.sh               # install deps, then API (127.0.0.1:8000) + React UI (5173)
./start.sh frontend      # same as the default
./start.sh setup         # install or update packages only
./start.sh api           # API only
./start.sh stop          # stop everything start.sh started
./start.sh --skip-setup  # skip pip/npm install on fast restarts
./start.sh --dev         # API with --reload on scripts/
./start.sh --no-open     # do not open a browser tab
./start.sh --https       # serve with the local certs in certs/
docker compose up        # API only, published on 127.0.0.1:8000
```

The API binds 127.0.0.1 unless `--host ADDR` says otherwise. The old `ui` and `both` modes exit
with status 2.

- React UI: http://localhost:5173 · API docs: http://127.0.0.1:8000/docs · SSE:
  http://127.0.0.1:8000/events/stream
- Config precedence: `config.yaml` (base, committed), then `config.local.yaml` (gitignored
  overrides), then env vars `REMIXMATE_<SECTION>_<KEY>` (for example
  `REMIXMATE_AUDIO_SAMPLE_RATE=48000`). Never commit keys to `config.yaml`.
- Auth is off by default: with `REMIXMATE_AUTH_ENABLED` unset every request runs as the `local`
  user. Hosted mode setup is in `app/deploy/PROVISION.md`.

## Commands

**Backend tests** (collected counts from `pytest --collect-only -q` on 2026-09-30 at `bd1c446`):
```bash
pytest                                                    # full suite, 1961 tests
pytest -m "not dj_analysis"                               # skip librosa/numba dependent tests
pytest -m "not dj_analysis and not integration" \
  --ignore=tests/e2e_test_suite.py --ignore=tests/test_audio_source.py   # 1759 tests, safe on any machine
pytest tests/test_behavioral.py                           # 48 behavioral tests (asserts what you hear)
pytest tests/test_foo.py::TestClass::test_name            # single test
bash tests/smoke_e2e.sh                                   # live smoke test (API must already be running on :8000)
```
The suite holds 1102 `def test_` functions (`grep -rc "def test_" tests --include='*.py'`; 1100 at `aef279f`); the
collected count is higher because of parametrized cases. Tests requiring librosa are marked
`@pytest.mark.dj_analysis` and auto skip (see `tests/conftest.py`) when librosa or numba fails to
initialize, which happens on some machines because of numba cache issues; that is not a real
failure. Tests marked `integration` start the full FastAPI app through `TestClient`.
`tests/test_behavioral.py` is the regression guard for audible bug fixes; do not delete tests
there because they look redundant with shape only tests elsewhere.

**Lint, format, typecheck (Python):**
```bash
ruff check .          # pyflakes + pycodestyle + isort (rules E, F, W, I)
black .               # formatter, line-length 100
mypy scripts/         # ignore_missing_imports=True (audio libs lack stubs)
pre-commit run --all-files
```

**Frontend** (`cd frontend`, Node 22 as in CI):
```bash
npm run dev           # vite dev server
npm run build         # tsc && vite build
npm run lint          # eslint, --max-warnings 0
npm run typecheck     # tsc --noEmit
npm test              # vitest
```

## Key design decisions

**Two mixing engines.** `dj.mixing_engine` in config selects the render path for the remix,
chain and preview tasks. `config.yaml` sets `current`; the dataclass default in
`scripts/core/config.py` is still `classic`, so a config that omits the key renders classic.
`classic` is `scripts/core/classic_engine/`, a frozen vendored copy of the
AI_REMIXMATE_BEST_MIXING_ALGO snapshot with its own analysis, planner, renderer and mastering
(`CLASSIC_DJ_TARGET_LUFS = -8.0`), kept faithful to the reference renders, so never fold current
engine fixes into it. `current` is the live `dj_engine.py` path described below and masters to
`audio.target_lufs` (-14.0 in `config.yaml`), so its mixes are quieter than classic ones. Only
`scripts/api/task_modules/remix.py::_resolve_engine()` consumes the flag; analysis endpoints,
setlist planning and library metadata always use the current modules. Job results carry an
`engine` field naming the path that rendered them.

**Beat grid lock.** `dj_engine.py` computes bar grid phases at cue points and applies a sample
level correction clamped to half a bar either way. The entry sample index compensates for the
stretch ratio: `entry_sample_b = int(plan.entry_time_b * sr / safe_ratio)`, then
`_clamp_entry_sample()`.

**Stem aware crossfade.** When Demucs stems exist, each stem fades on its own envelope. `bass`
gets a dedicated swap ramp (`_stem_bass_ramp()`, a true stem level cosine taper mute at the swap
point with no IIR bleed). Every other stem (drums, vocals, other) goes through
`_stem_crossfade_curves(trans_samples, similarity)`, which is driven by that stem pair's
similarity score, not by stem identity: no code path keys entry timing to "this is vocals, delay
it". Read "stem aware" as "each stem gets its own similarity shaped curve", not "drums and bass
always come in ahead of vocals"; the latter would need stem type specific offset logic that does
not exist. The standard (medium similarity) crossfade is equal power (cos/sin), not equal gain,
because two different songs are uncorrelated and amplitude complementary curves dip at the
midpoint. The stem summed transition window splices into the surrounding original mixes through
equal gain seam blends of about 30 ms (`_seam_blend()`, `min(int(0.030 * sr), trans_samples // 8)`);
equal gain there because seam material is correlated.

**Stepped (beat quantized) crossfade.** `transition_effect="stepped"` replaces the similarity
curves with a staircase: `_stepped_crossfade_curves()` holds one level per beat
(`plan.transition_bars * 4` plateaus) instead of sweeping. Increments are equal in **power**
(`fade_out[k] = sqrt(1 - k/(N-1))`), not amplitude; equal amplitude steps put both curves at 0.5
mid transition and reintroduce the dip that `TestEqualPowerCrossfade` in
`tests/test_behavioral.py` exists to catch. The `N-1` divisor is what lands the last plateau on
exactly (0.0, 1.0), which the `_seam_blend()` handoff assumes. Three things to know: it
**bypasses loop extension** (a staircase on tiled loop audio is not "fade Song A down"), it does
**not** touch bass (`_stem_bass_ramp()` still owns the low end), and it only applies on the stem
path; `render()` and `render_chain()` treat `"stepped"` as `"auto"` for the exit effect so a stem
less song does not silently lose its echo or filter. `steps=None` is the default everywhere else
and leaves the similarity bands unchanged. Guarded by `tests/test_stepped_crossfade.py`.

**Dynamic EQ.** A low shelf filter pulls Song A's bass down as Song B's rises, preventing bass
clash from simultaneous dual track playback.

**Masking aware multiband EQ.** `spectral_mixing.py` (Hafezi and Reiss 2015 offline variant)
ducks a track's mid and high bands where they mask content the other track needs,
bidirectionally, during stem transitions. Gated by `dj.masking_eq_enabled` (the A/B kill switch);
the bass band is always protected so `_stem_bass_ramp()` and the low shelf stay the sole
authority over low end. Gain curves taper to 0 dB at window edges so the processed window splices
transparently.

**Learned spectral cue matching.** `spectral_matching.py` scores Song B's phrase boundary entry
candidates by mel band *trajectory* similarity to Song A's outro (frame by frame spectral shape
over the whole window; one melspec per song, sliced per candidate) and moves
`plan.entry_bar_b` / `entry_time_b` to the best fit before rendering. It is engine agnostic: it
adjusts the plan in `task_modules/remix.py`, never the engines, so the frozen classic package
stays untouched. The per band weights train online from thumbs up or down verdicts
(`POST /dj-remix/feedback`, one per render session; buttons in MixDeck's Mix Ready card). State
lives in `data/spectral_match_model.json` and `spectral_match_history.json`; gated by
`dj.spectral_matching`.

**Dynamic spectral sync.** `spectral_sync.py` runs after render and before mastering on full
remix outputs: a time varying multiband EQ over the overlap window that eases the mix's spectrum
from A's outro profile to B's profile (cosine target ramp, per band gains capped at
`dj.spectral_sync_max_db`, temporally smoothed, edge tapered to 0 dB, bass band below
`dj.bass_crossover_hz` untouched so the engine's bass swap keeps low end authority). Full renders
only: previews are the bare overlap segment and chains do not expose window positions. Gated by
`dj.spectral_sync`.

**Harmonic compatibility.** Two independent systems exist and serve different purposes: the
Camelot wheel plus `camelot_modulation()` classification (named modulation types, each with a
psychoacoustic cost) for setlist planning, and `tiv_scoring.py`'s Tonal Interval Vector for a
continuous harmonic score. Do not conflate them: Camelot answers "is this move common in DJ
practice", TIV answers "how tonally close are these keys".

**Job store.** SQLite write through with an in memory dict for O(1) reads. Jobs recorded as
RUNNING or PENDING at process start are rolled back on restart. Cancel via `DELETE /jobs/{id}`.
The dict holds only the newest 500 rows (`_load_from_db`), so it is **not** the full record: an
unfiltered `list_jobs()` reads it, but a call passing `job_type` or `status` reads through to
SQLite, because a filtered lookup answered from memory silently omits everything older. That is
what once made Mix Vault render "Vault is empty" over real mixes. Filters apply before the limit.

**SSE.** `GET /events/stream` pushes `heartbeat`, `job_created`, `job_updated`,
`job_completed`, `job_failed`, `job_cancelled`, `library_changed` and `system_status`. The
frontend falls back to polling when SSE disconnects.

**Embedding index vs. CLAP index.** `data/music_index.json` (35 dim vectors, `VECTOR_DIMS = 35`,
always available) backs `/library/similar/{name}`. `data/clap_index.npy` (512 dim CLAP, optional)
backs `/library/crate-search` text and semantic queries. They coexist and are rebuilt
independently.

**Auth gate.** `main.py` mounts `system.router` (`/health*`, `/features`) and `legal.router` open, then
mounts the aggregated router from `routes.py` with `dependencies=[Depends(require_user)]`. A route
added to any router in `routes.py` is therefore behind auth automatically; a route that must stay
public has to be mounted beside `legal.router` and must not read per user data.

## Common tasks

**Add a new API endpoint:**
1. Add the route to the appropriate router in `scripts/api/routers/` (it inherits `require_user`
   through `routes.py`).
2. Add a task function in `scripts/api/task_modules/` if it needs to run async.
3. Create and submit the job via `job_store.create_job()` / `job_store.submit_job()`.
4. Add the client call to `frontend/src/lib/api.ts`.

**Add a new frontend page:**
1. Create `src/pages/MyPage.tsx` and `src/pages/MyPage.css` (follow the `PageBase.css` layout
   pattern: `page-base`, then `page-base__header`, then `page-base__body`; no Tailwind).
2. Add a `NavItem` in `shell/LeftRail.tsx`.
3. Add a `NavDestination` union member in `src/types/index.ts`.
4. Add a `<Route>` in `shell/AppShell.tsx`.

Pages that must render without sign in (like `/legal`) go in `src/App.tsx` outside
`<RequireAuth>`, not in `AppShell.tsx`.

**Design tokens** live in `frontend/src/styles/tokens.css` (colors, spacing, type scale,
radius); check there before hardcoding a color or spacing value.

**Stem work** (Demucs separation, separation config, missing or failed stems, library pruning):
a local `stem-processing` skill at `.claude/skills/stem-processing/` carries the procedures on the
maintainer's machine. It is not in the repository (`.gitignore:101` ignores `.claude/`, and
`git ls-files .claude` lists nothing), so a clone has no such directory; this file and
`scripts/core/stems.py` are the tracked sources.

## Frontend pages

| Page | Route | What it does |
|---|---|---|
| Mission Control | `/mission-control` | System health, active jobs, library stats, recent activity (Zustand + SSE) |
| Library Atlas | `/library-atlas` | Song table with search, filters, favorites, per song actions |
| Mix Deck | `/mix-deck` | Dual deck, BPM and key display, compatibility score, transition preview, full remix job |
| Set Builder | `/set-builder` | Sequence songs into a set, energy arc view, chain remix |
| Signal Search | `/signal-search` | Similarity search over the 35 dim embedding index |
| AI Lab | `/ai-lab` | Style transfer, inpainting, tokenization |
| Mix Vault | `/mix-vault` | Completed mixes: in page player, mix and credits files, metadata (BPM, Camelot, harmonic score, LUFS) |
| Operations | `/operations` | Processing status, storage (size cap, prune, eviction), maintenance (index builds, calibration, reanalysis) |
| Account | `/account` | Who is signed in, export of the user's data, account deletion gated by typing `DELETE` (off in local mode) |
| Legal | `/legal`, `/legal/:page` | Public, no sign in: Terms, Privacy, DMCA (with the report form), Acceptable Use |
| DJ Widget | `/widget` | Compact recommendation panel with a Document Picture in Picture pop out |

The catalogue is operator curated: there is no in app track acquisition. Tracks enter through
the operator ingest tool `scripts/download.py` (ccMixter and Internet Archive, licence check,
Demucs, analysis).

## Design system

All tokens in `frontend/src/styles/tokens.css`. Key values:

- **Accents**: amber (`--color-amber-500: #f59e0b`), ice (`--color-ice-400: #38bdf8`), green
  (`--color-green-500: #34d399`), crimson (`--color-crimson-500: #f87171`), violet
  (`--color-violet-400: #a78bfa`)
- **Surfaces**: `--color-bg-void` `#05050a`, `--color-bg-base` `#09090b`, `--color-bg-surface`
  `#111113`, `--color-bg-elevated` `#18181b`, `--color-bg-overlay` `#1c1c20`
- **Fonts**: Space Grotesk (display), Inter (UI), JetBrains Mono (mono)
- **Shell**: 64px left rail (`--shell-rail-width`) + 316px right inspector
  (`--shell-inspector-width`) + canvas

New pages follow the `PageBase.css` layout pattern. Each page gets its own CSS file. No Tailwind.

## Stem processing (Demucs)

Separating tracks into stems and keeping the library clean. CPU separation is slow; plan for it.

**Where things live**

- Config: `config.yaml` `separation:` block, `model` (`htdemucs` | `htdemucs_ft` | `mdx_extra`)
  and `device`. Storage keys `keep_raw_after_separation` and `prune_on_download` sit in the
  `library:` block; the AI model memory budget `max_vram_gb` (12.0) sits in `generative:`.
- Device: `scripts/core/stems.py::_detect_device()` calls `scripts/core/gpu.py::get_device()`,
  which picks MPS on Apple Silicon, then CUDA, then CPU. Force one with `REMIXMATE_DEVICE`.
- Ingest plus separation: `scripts/download.py`, the operator catalogue ingest tool.
- Bridge integration: `scripts/remixmate_cli.py` `generate_stems_and_params(base_song,
  match_song, out_dir)`.
- Cleanup: `python -m scripts.purge_library` lists songs with no `full.wav`, no
  `full_enhanced.wav` and no stems (dry run); `--apply` deletes them.

**Rules**

- Respect `generative.max_vram_gb` so separation and the other AI models do not run out of memory
  together.
- Missing stems are not a crash. A song folder with no source audio and no stems is a failed or
  shell ingest; route it through `purge_library` rather than silently producing an empty remix.
- Stem paths and mix params come from the CLI helper, never from hardcoded guesses.
- Never commit keys. `config.local.yaml` is gitignored; secrets live there.

**Before calling stem work done**

- Did the log line `[stems] Device:` name the device you expected?
- Is the memory budget respected?
- Are missing stem songs handled rather than silently mixed?
- Do stem paths come from the CLI helper?

## Gotchas

- **New config.yaml keys need a matching dataclass field.** `_dict_to_config()` in
  `scripts/core/config.py` silently drops any YAML key not declared on the section's dataclass
  (and any top level section not declared on `AppConfig`); guarded `getattr` reads then always
  return their defaults with no error. `tests/test_config_drift.py` guards this; add the field
  and the YAML key in the same change.
- **Router registration order matters.** `routes.py` registers `analysis` before `library` on
  purpose: `library`'s dynamic `GET /library/{name}` would otherwise shadow `analysis`'s static
  `GET /library/crate-search` (FastAPI matches routes in registration order). If a new static
  route under an existing dynamic prefix router 404s or hits the wrong handler, check
  registration order first, and add a case to `tests/test_route_shadowing.py`.
- **`job_store.submit_job()` runs the task in a `ThreadPoolExecutor`.** Do not do async I/O
  inside task functions; use `asyncio.run()` if you need it there.
- **Job progress units differ by transport.** SSE frames and the frontend's normalized `Job` type
  use 0 to 100. The REST `GET /jobs` and `GET /jobs/{id}` endpoints return 0 to 1.
  `normalizeJob()` in `api.ts` handles the conversion; do not derive it again in a component.
- **`JobResponse.status` is `str`, not the `JobStatus` enum.** The internal enum is lowercase
  (`pending`, `running`, `done`, `failed`, `cancelled`); REST and SSE both normalize to uppercase
  (`PENDING|RUNNING|COMPLETED|FAILED|CANCELLED`) via `_norm_status()` in `scripts/api/jobs.py`.
- **`data/jobs.db` stores two spellings of the same enums.** Older rows persisted the repr
  (`JobType.DJ_REMIX`, `JobStatus.DONE`), newer rows the value (`dj_remix`, `done`); both forms
  are live in the same table. A raw `WHERE job_type = 'dj_remix'` returns a plausible undercount.
  Always normalize both sides through `_type_value()` / `_status_value()` in
  `scripts/api/jobs.py`, which collapse either form. `normalizeJob()` in
  `frontend/src/lib/api.ts` does the same for the client. `tests/test_job_list_filters.py`
  guards this.
- **`StructuredLogger.warning(msg, extra=None)`** takes a dict as its second arg, not printf
  style varargs. Never call `logger.warning("...%s", exc)`: the exception is treated as the
  `extra` dict and crashes inside Python's logging internals. Use an f string or pass `exc`
  inside a dict.
- **Two logging setups exist and only one runs.** `configure_structured_logging()` in
  `logging_utils.py` builds the `StructuredJsonFormatter` / `StructuredConsoleFormatter`
  handlers and has no callers (`grep -rn configure_structured_logging scripts` returns only its
  own `def`). The path that runs is `configure_logging()` in `config.py`, called from `main.py`'s
  lifespan, which uses `logging.basicConfig` with a plain format string. Anything attached to the
  structured formatters is dead code that still looks wired: attach handler level behaviour in
  `config.configure_logging()` instead, which is where `LogInjectionFilter` (CR/LF scrubbing
  against forged log lines) lives. `tests/adversarial/test_log_injection.py` asserts that path
  installs it.
- **SSE from worker threads.** The event loop is captured once in `main.py`'s lifespan via
  `asyncio.get_running_loop()` and closed over by `_sync_emit`. Never call
  `asyncio.get_event_loop()` from inside a `ThreadPoolExecutor` worker; it raises `RuntimeError`
  on Python 3.12 with no running loop in that thread.
- **Camelot semitone table.** `CAMELOT` and `NOTE_NAMES` in `key_detection.py` are the single
  source of truth; `pitch_shift_for_camelot()` derives from them. Do not add a second inline
  mapping; it will drift and produce wrong pitch shifts.
- **`favoritesApi.list`.** Backend `GET /favorites` returns `{ songs: string[], count: number }`,
  not a bare array. The frontend call unwraps `.songs`; do not reintroduce a raw
  `get<string[]>`.
- **Serato cue export requires `.mp3`.** `export_serato_markers()` raises `ValueError` for any
  other suffix; the GEOB tag format has no equivalent container. Use rekordbox XML export for non
  MP3 tracks.
- **CLAP model downloads its checkpoint** on first `crate_digger` use unless `models.clap_model`
  points to a local checkpoint in config, so the first call is slow.
- **Per stem LUFS targets** (`data/stem_lufs_targets.json`) must be generated once via
  `POST /library/calibrate-lufs` before `normalize_stems_to_corpus_targets()` uses corpus derived
  values; otherwise it silently falls back to `_STEM_FALLBACK_TARGETS` in `mastering.py`.
- **Demucs auto selects MPS on Apple Silicon.** No caller passes `separation.device` from config
  into `separate_song_stems()`, so the `device` key in `config.yaml` has no effect; every run
  uses `device="auto"`, which resolves through `gpu.get_device()` (MPS, then CUDA, then CPU). To
  force a device set `REMIXMATE_DEVICE=cpu|mps|cuda`. If Demucs fails on MPS or CUDA,
  `stems.py` retries once on CPU, which is much slower.
- **`config.local.yaml` is gitignored.** Never commit API keys to `config.yaml`.
- **Production frontend is Cloudflare Pages.** In dev, Vite proxies `/api/*` (prefix stripped),
  `/events` and `/outputs` to `http://localhost:8000`. The Pages build has no proxy, so set
  `VITE_API_BASE=https://api.<domain>` and `VITE_EVENTS_URL=https://api.<domain>/events/stream`
  in the Pages project (plus the `VITE_SUPABASE_*` and `VITE_COOKIE_DOMAIN` vars); the full list
  is in `app/deploy/PROVISION.md` section 10. The API serves no UI.
- **`ai_remixmate_feature_lab/`** is intentionally isolated: it has its own test suite and
  imports nothing from `scripts/`; it is not wired into the UI by design, not because it is
  abandoned.
