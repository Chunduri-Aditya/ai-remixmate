# Developing AI RemixMate

Practical notes for working in this codebase: how to run it, the design decisions that are not
obvious from the code, and the traps that have each cost real debugging time at least once.

For the system map see [ARCHITECTURE.md](ARCHITECTURE.md). For what changed when, see
[CHANGELOG.md](CHANGELOG.md).

---

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

## Key design decisions

**Two mixing engines** — `dj.mixing_engine` in config selects the render path for the remix/chain/
preview tasks. **`current` is the default as of 2026-08-27** (it was `classic` before that; the
switch moved mastering from −8 to −14 LUFS, so every mix rendered after it is quieter).
`classic` is `scripts/core/classic_engine/`, a frozen vendored copy of
the April 2026 AI_REMIXMATE_BEST_MIXING_ALGO snapshot (its own analysis, planner, renderer, and
mastering at −8 LUFS) — kept byte-faithful to the reference renders, so never fold current-engine
fixes into it. `current` is the live `dj_engine.py` path described below. Only
`scripts/api/task_modules/remix.py::_resolve_engine()` consumes the flag; analysis endpoints,
setlist planning, and library metadata always use the current modules. Job results carry an
`engine` field naming the path that rendered them.

**Beat-grid lock** — `dj_engine.py` computes bar-grid phases at cue points and applies a sample-level
correction (clamped to ±half a bar). Entry sample index compensates for the stretch ratio:
`entry_sample_b = int(entry_time_b * sr / stretch_ratio)`.

**Stem-aware crossfade** — When Demucs stems exist, each stem fades on its own envelope. `bass` gets a
dedicated swap-ramp path (see below); every other stem (drums/vocals/other) goes through
`_stem_crossfade_curves(trans_samples, similarity)`, which is driven purely by that stem pair's
similarity score, not by stem identity — there is no code path that keys entry timing to "this is
vocals, delay it" specifically. (Confirmed by adversarial behavioral audit, 2026-07: measured
per-stem B-onset timing with distinct test tones — drums, vocals, and other entered the mix at the
identical timestamp under equal similarity.) Read "stem-aware" as "each stem gets its own
similarity-shaped curve," not "drums/bass are always brought in ahead of vocals" — the latter would
need real stem-type-specific offset logic that doesn't currently exist. `_stem_bass_ramp()` does a
true stem-level cosine-taper bass mute at the swap point (no IIR bleed) rather than a linear
crossfade. The standard
(medium-similarity) crossfade is equal-power (cos/sin), not equal-gain — two different songs are
uncorrelated, so amplitude-complementary curves dip ~3 dB at the midpoint. The stem-summed transition
window splices into the surrounding original mixes through ~30 ms equal-gain seam blends
(`_seam_blend()` — equal-gain there because seam material is correlated).

**Stepped (beat-quantized) crossfade** — `transition_effect="stepped"` replaces the similarity
curves with a staircase: `_stepped_crossfade_curves()` holds one level per beat
(`plan.transition_bars * 4`) instead of sweeping. Increments are equal in **power**
(`fade_out[k] = sqrt(1 - k/(N-1))`), not amplitude — equal-amplitude steps put both curves at 0.5
mid-transition and reintroduce the ~3 dB dip that `TestEqualPowerCrossfade` exists to catch. The
`N-1` divisor is what lands the last plateau on exactly (0.0, 1.0); dividing by `N` leaves A at
`sqrt(1/64) ≈ 0.125` and breaks the handoff `_seam_blend()` assumes. Three things to know: it
**bypasses loop extension** (a staircase on tiled loop audio is not "fade Song A down"), it does
**not** touch bass (`_stem_bass_ramp()` still `continue`s past the curve), and it only applies on
the stem path — `render()`/`render_chain()` treat `"stepped"` as `"auto"` for the exit effect so a
stem-less song does not silently lose its echo/filter. `steps=None` is the default everywhere else
and leaves the similarity bands byte-identical. Guarded by `tests/test_stepped_crossfade.py`.

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
The dict holds only the newest 500 rows (`_load_from_db`), so it is **not** the full record: an
unfiltered `list_jobs()` reads it, but a call passing `job_type`/`status` reads through to SQLite
instead, because a filtered lookup answered from memory silently omits everything older. That is
what made Mix Vault render "Vault is empty" over 35 real mixes. Filters apply before the limit.

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

## Stem processing (Demucs)

Separating tracks into stems and keeping the library clean. CPU is slow — plan for it.

**Where things live**

- Config: `config.yaml` → `separation:` block — `model` (`htdemucs` | `htdemucs_ft` | `mdx_extra`),
  `device` (`auto` | `cpu` | `cuda` | `mps`), `keep_raw_after_separation`, `prune_on_download`,
  `max_vram_gb`.
- Download plus optional separation: `scripts/download.py` (skips the split if `demucs` is absent).
- Bridge integration: `scripts/remixmate_cli.py` → `generate_stems_and_params(base, match, out_dir)`.
- Cleanup: `scripts/purge_library.py` flags songs with neither a full mix nor stems.

**Rules**

- Set the device deliberately. On Apple Silicon use `device: mps`; Demucs on CPU runs 2–5× real
  time, so batch and expect latency.
- Respect `max_vram_gb` (default 12) so separation and the other AI models do not OOM each other.
- Missing stems are not a crash. A song folder with no `full.wav` and no stems is a failed or shell
  download — route it through `purge_library.py` rather than silently producing an empty remix.
- Stem paths and mix params come from the CLI helper, never from hardcoded guesses.
- Never commit keys. `config.local.yaml` is gitignored; secrets live there.

**Before calling stem work done**

- Does the device match the machine (`mps` on Apple Silicon)?
- Is the VRAM budget respected?
- Are missing-stem songs handled rather than silently mixed?
- Do stem paths come from the CLI helper?

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
- **`data/jobs.db` stores two spellings of the same enums.** Older rows persisted the repr
  (`JobType.DJ_REMIX`, `JobStatus.DONE`), newer rows the value (`dj_remix`, `done`); both forms are
  live in the same table. A raw `WHERE job_type = 'dj_remix'` returns a plausible-looking undercount
  (29 of 35 completed remixes, as of 2026-08-27). Always normalize both sides through
  `_type_value()` / `_status_value()` in `scripts/api/jobs.py` — they already collapse either form.
  `normalizeJob()` in `frontend/src/lib/api.ts` does the same for the client.
  `tests/test_job_list_filters.py` guards this.
- **`StructuredLogger.warning(msg, extra=None)`** takes a dict as its second arg, not printf-style
  varargs. Never call `logger.warning("...%s", exc)` — the exception gets treated as the `extra` dict
  and crashes inside Python's logging internals. Use an f-string or pass `exc` inside a dict.
- **Two logging setups exist and only one runs.** `configure_structured_logging()` in
  `logging_utils.py` builds the `StructuredJsonFormatter` / `StructuredConsoleFormatter`
  handlers, and has zero callers (`grep -rn configure_structured_logging .` returns only its
  own `def`, as of 2026-08-27). The path that actually runs is `configure_logging()` in
  `config.py`, called from `main.py`'s lifespan, which uses `logging.basicConfig` with a plain
  format string. So anything attached to the structured formatters is dead code that still
  looks wired: attach handler-level behaviour in `config.configure_logging()` instead, which
  is where `LogInjectionFilter` (CR/LF scrubbing, guards against forged log lines) lives.
  `tests/adversarial/test_log_injection.py` asserts that path installs it.
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

