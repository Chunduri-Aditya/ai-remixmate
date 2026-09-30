# Mixing map

*Agent map of the mixing side of AI RemixMate: how a song becomes analysis, how two or more songs become a mastered mix, and where each step lives. As of 72e2368. Every cited path and `path::symbol` is checked against the tree by `tests/test_agent_docs.py`; whether a sentence about the code is true is a review job, and where this file and the code disagree, the code wins.*

## How to use this map

- The live engine is `scripts/core/dj_engine.py` with its planner in `scripts/core/dj_analysis.py`. The classic engine under `scripts/core/classic_engine/` is frozen reference code. `dj.mixing_engine` in `config.yaml` picks one per job (`current` today); see "Engine selection and the classic engine".
- Start from "End to end flow", find the hop you need, then jump to its section. Each section names the function that owns a step, the data it reads and writes, and the traps.
- Before fixing anything that looks wrong, look for its id in `docs/agents/BACKLOG.md`. Problems are cited here as ids (MX, OPS, DOC, UI) plus a few words; the backlog holds the evidence and a fix sketch.
- Commands, recipes, gotchas and design decisions live in `DEVELOPING.md`. Music theory lives in `docs/DJ_THEORY.md`. The hosted deployment ledger lives in `app/STATUS.md`. None of that is restated here.
- Numbers in this file are code constants or config values unless they carry "(as of 72e2368, `command`)".

## End to end flow

One two song remix, `POST /dj-remix`, on the current engine. The chain and preview variants differ at the hops noted in "Rendering" and "Job lifecycle".

| Hop | Function | What happens | Writes |
|---|---|---|---|
| 1 | `scripts/api/routers/remix.py::dj_remix` | Validates `scripts/api/schemas.py::DJRemixRequest`; `scripts/api/routers/_helpers.py::_check_job_cap`, then `scripts/api/routers/_helpers.py::_require_song` per song, then `scripts/api/routers/remix.py::_confine_beat_path` for a bridge beat file | nothing |
| 2 | `scripts/api/jobs.py::create_job` | Stamps `meta.user_id`, `meta.song_a`, `meta.song_b`, `meta.request`; emits `job_created` | a row in `data/jobs.db` (or Postgres) |
| 3 | `scripts/api/jobs.py::submit_job` | Repeats the cap under the lock, queues the task on the worker pool | FAILED row on refusal (429) |
| 4 | `scripts/api/task_modules/remix.py::task_dj_remix` | Calls `scripts/api/task_modules/remix.py::_resolve_engine` for the analyse, plan, render and master callables | nothing |
| 5 | `scripts/core/audio_source.py::load_source_audio` | `full.wav`, else `full_enhanced.wav`, else the sum of whatever stems exist; mono, 44100 Hz, capped at `remix.max_song_seconds` via `scripts/api/task_modules/remix.py::_max_song_seconds` | nothing |
| 6 | `scripts/core/dj_analysis.py::analyze_structure` | Fresh analysis of the loaded audio; the cached `meta.json` is never read here | nothing |
| 7 | `scripts/core/dj_analysis.py::plan_transition` | Builds `scripts/core/dj_analysis.py::TransitionPlan` | nothing |
| 8 | `scripts/core/spectral_matching.py::apply_spectral_matching` | May move `plan.entry_bar_b` and `plan.entry_time_b` when `dj.spectral_matching` is on | nothing |
| 8b | `task_dj_remix` bridge beat | `bridge_beat_mode` `auto`: `scripts/core/beat_synth.py::render_beat` at `plan.bpm_a` for `max(4, transition_bars)` bars; `file`: `scripts/api/task_modules/remix.py::_user_beat_file` then `librosa.load`; any other value means no beat (MX-07). Either branch swallows every exception, renders without the beat and finishes DONE; only `result.bridge_beat` (`auto:failed(...)`, `file:failed(...)`) says so | nothing |
| 9 | `scripts/api/task_modules/remix.py::_should_use_stem_blend` | Stems on disk for both songs and `0.5 * harmonic_score + 0.5 * spectral match score` (or `harmonic_score` alone) at or above `dj.transition_confidence_threshold` picks the stem path; with the current scorer's 0.89 floor the gate cannot fail (MX-27), and a negative `harmonic_score` (the -1.0 not computed sentinel) skips it | nothing |
| 10 | `scripts/core/dj_engine.py::DJEngine.render_stem_blend` or `scripts/core/dj_engine.py::DJEngine.render` | Produces the full mono mix | nothing |
| 11 | `scripts/core/spectral_sync.py::apply_spectral_sync` | Multiband EQ over the transition window on the rendered array | nothing |
| 12 | `scripts/core/mastering.py::master_mix` | LUFS normalise to `audio.target_lufs`, true peak limit, `QualityReport` | nothing |
| 13 | `scripts/api/task_modules/remix.py::_fade_out_tail` | 3 s fade, only when song B was cut at the load cap (MX-38) | nothing |
| 14 | `task_dj_remix`, `scripts/api/task_modules/remix.py::_write_mix_credits` | Writes the mix as mono PCM_24 plus `CREDITS.txt` and `manifest.json` | `outputs/<uid>/dj_<sid>/` |
| 15 | `scripts/core/spectral_matching.py::record_render` | Stores the render's spectral info for later feedback learning | `data/spectral_match_history.json` |
| 15b | `task_dj_remix` transition score | `scripts/core/music_intelligence.py::compute_transition_score` on the two `music_vector`s into the `ts_*` result fields, inside `except Exception: pass`, so a DONE result can lack them with no log line | nothing |
| 16 | `scripts/api/task_modules/remix.py::_sweep_user_mixes` | Retention sweep of the caller's `outputs/<uid>/`; `scripts/api/jobs.py::mark_expired` sets `result.expired = True` on the swept sessions' jobs, and Mix Vault (`frontend/src/pages/MixVault.tsx`) hides those jobs | deletes old session dirs, updates job rows |
| 17 | inner `_wrapper` of `submit_job` | DONE, progress 1.0, relativized result, `job_completed` over SSE | job row |

Analysis that is cached for the UI and scoring (`meta.json`, `analysis.json`) is a separate flow; see "Analysis".

## Job lifecycle

Owner: `scripts/api/jobs.py`. `DEVELOPING.md` "Key design decisions" (Job store, SSE) and "Gotchas" (progress units, status spelling, two enum spellings in `data/jobs.db`) cover the design; this is the hop list.

1. Route validates, then `_check_job_cap` counts the caller's active jobs through `scripts/api/jobs.py::list_jobs` (in memory, scoped to the current user) and `scripts/api/jobs.py::_is_active`. Not atomic; its docstring says so.
2. `create_job` stamps `meta.user_id` because executor threads do not inherit contextvars.
3. `submit_job` repeats the cap under `_lock` against `_MAX_ACTIVE_JOBS`. A refusal marks the job FAILED and raises 429, so a refused submit still leaves a FAILED row. Otherwise the id goes into `_queued` and `_wrapper` goes to `_executor` (`ThreadPoolExecutor`, `max_workers=2`).
4. `_wrapper` drops the id from `_queued`, returns if cancelled or purged before start, sets the owner as current user, marks RUNNING, calls `fn(job_id=job_id, **kwargs)`.
5. The task reports with `scripts/api/jobs.py::update_job`: progress clamped to 0..1, `eta_sec` by linear extrapolation, absolute paths scrubbed from `message` and `error` (`scripts/api/jobs.py::_scrub`), `result` relativized (`scripts/api/jobs.py::_relativize_result`), row written, SSE frame emitted.
6. Return sets DONE; an exception sets FAILED with the scrubbed error. A purged owner (account deletion) skips persistence and deletes `outputs/<uid>/` again (`scripts/api/jobs.py::_discard_purged_output`). A cancelled id skips persistence.

Persistence and reads:
- `scripts/api/jobs.py::_get_conn` goes through `scripts/core/db.py` to SQLite at `data/jobs.db`, or Postgres when `DATABASE_URL` is set. One row per job, `result_json` and `meta_json` as JSON text.
- `scripts/api/jobs.py::_load_from_db` runs at import time, loads the newest 500 rows and rewrites every RUNNING or PENDING row to FAILED. Importing the module from a second process therefore fails the live API's jobs (OPS-08). `tests/conftest.py` redirects the data dir through `REMIXMATE_DATA_DIR`; see OPS-10 for `DATABASE_URL`.
- Filtered `list_jobs` reads the database; unfiltered reads memory. `GET /jobs/{id}` and `DELETE /jobs/{id}` read memory only (`scripts/api/routers/jobs.py::_owned_job`), so an old job can be listable with a filter yet 404 by id.

Two wire shapes for one job: REST uses `scripts/api/jobs.py::job_to_response` into `scripts/api/schemas.py::JobResponse` (`job_type`, progress 0..1, epoch floats, `eta_sec`, no `meta`, see UI-10). SSE uses `scripts/api/jobs.py::_job_to_response_dict` (`type`, progress 0..100, ISO strings, `meta`, no `eta_sec`). Both uppercase status and map `done` to `COMPLETED` (`scripts/api/jobs.py::_norm_status`).

SSE: `scripts/api/main.py::lifespan` registers `_sync_emit`, which schedules `scripts/api/routers/events.py::SSEBroadcaster.broadcast` on the serving loop. A job frame goes only to queues whose user equals `meta.user_id`; frames without `meta` go to everyone. A full client queue is dropped and its stream ends. Event type comes from the stored status: `done` gives `job_completed`, `failed` gives `job_failed`, `cancelled` gives `job_cancelled`, else `job_updated`.

Cancellation: `scripts/api/jobs.py::cancel_job` (`DELETE /jobs/{job_id}`) sets CANCELLED at once for PENDING and RUNNING. A PENDING job never runs. Only `task_dj_chain` polls `scripts/api/jobs.py::is_cancelled` (between song loads and analyses); remix and preview keep rendering and writing output (MX-08), and later `update_job` calls keep emitting `job_cancelled` frames (MX-35).

Job types: `scripts/api/schemas.py::JobType` holds `download`, `dj_remix`, `separate`, `analyze`. Nothing in `scripts/` creates `download` or `separate` jobs; keep the values, because legacy rows with them must still validate for `GET /jobs`. `meta` tells jobs apart:

| JobType | Distinguishing `meta` | Route |
|---|---|---|
| `dj_remix` | `song_a`, `song_b`, `request` | `POST /dj-remix` |
| `dj_remix` | `song_a`, `song_b`, `preview: true`, `request` | `POST /dj-remix/preview` |
| `dj_remix` | `songs`, `n`, `request` | `POST /dj-chain` |
| `dj_remix` | `type: "instrument_lab"`, `songs`, `mode` | `POST /instrument-lab` |
| `analyze` | `song`, `key_profile`, no `type` | `POST /analyze` |
| `analyze` | `type` in stem split, compress, analyze missing, scan broken, calibrate, initialize, index rebuild, CLAP index, generative | stems, library, analysis, generative routers |

There is no `dj_chain` job type on the backend (UI-20). `meta.request` is `req.model_dump(mode="json")` with the bridge beat path confined; it reaches clients only over SSE.

Result fields to keep stable (the frontend and retention read them): `output` (relative `<uid>/<dir>/<file>`), `session_id` (bare 8 hex id), `stream_url` (served by `scripts/api/routers/library.py::stream_output` from the caller's own outputs dir), `engine`, `tempo_ratio` (the plan's `tempo_shift_ratio`, `bpm_a / bpm_b`; the preview route docstring says the inverse, DOC-04), `lufs`, `peak_dbfs`, `quality_passed`, `quality_notes`, `timeline` (`scripts/api/task_modules/remix.py::_transition_timeline`), `credits`, `stem_blend` (MX-51). `POST /dj-remix/feedback` (`scripts/api/routers/remix.py::dj_remix_feedback`) finds a session through `scripts/api/routers/remix.py::_owns_session`, which accepts `dj_<id>` or bare `<id>` dirs only, so chains cannot take feedback. Feedback context backfill: MX-06.

Retention: `scripts/core/mix_retention.py::sweep_dir` treats every immediate subdirectory of `outputs/<uid>/` as a session (`dj_`, `chain_`, preview `<sid>`, `beat_`, `lab_`, `gen_`). Pass one removes dirs older than `retention.mix_max_age_days`; pass two keeps the newest `retention.mix_max_count`. Callers: `_sweep_user_mixes` after every render (unless `retention.sweep_after_render` is false), `scripts/api/main.py::_retention_sweep_loop` every `retention.sweep_interval_hours`, and `POST /library/mixes/sweep` (dry run by default). `mark_expired` writes `result.expired = True` and keeps the row, so any job list must filter on that field or it shows dead download links. Traps: previews and uploaded beats count toward the cap, so heavy previewing evicts full mixes; a queued remix whose `beat_` dir was swept renders without the beat; `mark_expired` is not scoped by user. See OPS-18.

## Analysis

Two separate analysis paths exist and they never share results:

- Cached analysis, `scripts/core/analysis_pipeline.py::run_song_analysis`: writes `meta.json` and `analysis.json` into `library/<song>/` and upserts the similarity index. Feeds Library Atlas, cue export, `/compatibility`, `/recommend`, `/library/similar` and setlist planning.
- Render analysis, `scripts/core/dj_analysis.py::analyze_structure` called from the remix tasks on the loaded (capped) audio with no `key_profile` and no `stems_dir`. Editing `meta.json` or `analysis.json` never changes a render.

Entry points to the cached path:

| Entry | Handler | Task | Notes |
|---|---|---|---|
| `POST /analyze?key_profile=` | `scripts/api/routers/analysis.py::analyze_song` (job cap, no operator gate, DOC-01) | `scripts/api/task_modules/analysis.py::task_analyze` | ANALYZE job |
| `POST /library/analyze-missing` | `scripts/api/routers/library.py::analyze_missing` (operator) | `scripts/api/task_modules/analysis.py::task_analyze_missing` | songs where `has_analysis` is false, or all with `force`; double upsert (MX-37) |
| Operator ingest | `scripts/download.py::ingest_track` with `auto_analyze=True` | inline | |
| `POST /compatibility` | `scripts/api/routers/analysis.py::check_compatibility` | synchronous | analyses a song as a side effect when `has_analysis` is false |
| `GET /analysis/v1/{name}/structure` | `scripts/api/routers/analysis.py::song_structure_v1` | synchronous | calls `analyze_structure` on the full track, caches nothing (MX-15) |

Inside `run_song_analysis`:

1. `scripts/core/analysis_pipeline.py::_load_song_audio`: mono 44100 Hz (hardcoded, not `audio.sample_rate`), first 120 s (MX-01). Source order `full.wav`, else a sum of stems (`.flac` before `.wav`); it skips `full_enhanced.wav` (MX-18).
2. `scripts/core/genre.py::detect_genre` on the first 60 s (MX-14).
3. `analyze_structure(audio, sr, key_profile, stems_dir=song_dir)`: tracker, bar grid, per bar RMS and centroid, `scripts/core/dj_analysis.py::_label_sections`, `scripts/core/dj_analysis.py::_detect_phrase_boundaries`, `scripts/core/key_detection.py::detect_key`, `scripts/core/key_detection.py::detect_key_segments` mapped onto sections, then `scripts/core/music_intelligence.py::compute_track_vector`, which overwrites the key fields and fills energy, danceability, vocal and chord fields.
4. `scripts/core/recommend.py::write_meta_cache` then an inline merge into `meta.json`, both inside `except Exception: pass` (MX-19).
5. `analysis.json` write, then `scripts/core/music_index.py::MusicIndex.upsert_song`.

`task_analyze` rewrites the `ValueError` a clip with no full bar raises into `no beats detected in '<song>'` when fewer than 4 beats exist.

`scripts/core/analysis_pipeline.py::has_analysis` is the "analysed" predicate everywhere: both files exist and `meta.json` has a non null `bpm`.

### Files

`meta.json` keys written by `run_song_analysis` (merged, `None` dropped): `bpm` (octave corrected, 0.1 rounding), `genre`, `key`, `mode`, `camelot` (sharps only), `energy_mean`, `energy_std`, `danceability`, `beat_strength`, `tempo_stability`, `vocal_density`, `vocal_mean_pitch_hz`, `vocal_voiced_fraction`, `spectral_centroid_hz`, `chroma_vector` (12 floats). Not persisted: key confidence, duration, drop position, sections, bars, beats, phrase boundaries, per phrase keys. Other writers: `scripts/core/recommend.py::get_or_cache_bpm` (bpm only, from a 10 s clip) and `MusicIndex.upsert_song` through `scripts/core/music_index.py::_quick_features`, which writes raw RMS into `energy_mean` (a different scale). Read routes that write: MX-32.

`analysis.json` is `{bpm, phrase_boundaries, duration, key, camelot}`, written only by `run_song_analysis`. Readers: the song info builder in `scripts/api/routers/_helpers.py` (duration shown in the UI), `scripts/api/routers/library.py::export_cues`, and the CLAP index build in `scripts/core/crate_digger.py` (MX-40). Its `phrase_boundaries` are energy section starts, not the SSM result (MX-10); its `duration` never exceeds 120 s (MX-01).

### Beat grid and bars

- Tracker: `scripts/core/beat_tracker.py::get_configured_tracker` reads `analysis.beat_backend` (default `auto`; `config.yaml` has no `analysis:` block, so only `config.local.yaml` or `REMIXMATE_ANALYSIS_BEAT_BACKEND` changes it). `auto` picks `scripts/core/beat_tracker.py::BeatThisTracker` when `import beat_this` works, else `scripts/core/beat_tracker.py::LibrosaBeatTracker`. The repo `.venv` has no `beat_this` package, so local runs use librosa.
- `scripts/core/beat_tracker.py::BeatResult`: `beat_times`, `beat_frames` (hop 512), `downbeat_times`, `bpm`, `sr`, `backend`.
- BPM goes through `scripts/core/beat_tracker.py::resolve_bpm_octave` (onset autocorrelation, log normal prior centred at 122 BPM, range 70 to 180). Only the scalar changes; beat times keep the tracker's density (MX-20).
- Downbeats on the librosa path are `beat_frames[::4]` from the first detected beat. No downbeat phase detection happens by default.
- Bar grid in `analyze_structure`: bar `i` spans `beat_times[4i]` to `beat_times[4i+4]`, one entry per complete group of four beats. It ignores `downbeat_times` (MX-54). Trailing beats past the last full bar carry a bar index with no `bars` entry.
- `SongStructure.bar_start_time` falls back to `bar * 4 * 60 / bpm` outside the grid; `SongStructure.phrase_boundary` rounds up to a multiple of 8 counted from the first detected beat, not from a musical phrase start.
- Genre, danceability and the phrase detector fallback each run their own `librosa.beat.beat_track`; the genre BPM feature is not octave corrected.

### Sections and phrases

- `_label_sections`: per bar RMS normalised to the loudest bar, smoothed over 4 bars. Above mean + 0.2 is `drop` (centroid above 0.65) or `chorus`; below mean minus 0.15 is `intro` (first 20 percent), `outro` (last 20 percent) or `break`; a rise above 0.08 is `build`; else `verse`. Early and late quiet bars are forced to intro and outro. Runs under `min_section_bars=4` are dropped, leaving gaps (MX-55).
- `scripts/core/dj_analysis.py::Section` carries type, bar range (end exclusive), times, average energy and centroid, plus per phrase key fields ("" and 0.0 mean not computed). Mixable exits are outro, break, verse; mixable entries are intro, break, verse.
- `_detect_phrase_boundaries`: beat synchronous `chroma_cqt`, cosine self similarity, Foote checkerboard novelty over 8 beats, peaks at least 8 beats apart, then `scripts/core/dj_analysis.py::_snap_to_bar_grid` to a downbeat within two bars, preferring bars at multiples of 8 (MX-41). Returns `[]` on error or under 16 beats. Stored as `SongStructure.phrase_boundaries` in seconds.
- The only consumer of phrase boundaries is `plan_transition`. The structure endpoint returns `sections`, `key_segments` and `downbeat_times`, not phrase boundaries.
- Per phrase key: `detect_key_segments` on 30 s windows at 50 percent overlap; each section takes the segment active at its midpoint.

### Key and Camelot

Canonical table: `scripts/core/key_detection.py::CAMELOT` with `scripts/core/key_detection.py::NOTE_NAMES` (sharps; A minor is 8A, C major is 8B). `DEVELOPING.md` "Gotchas" calls it the single source of truth. Other tables that must match it:

| Table | Matches? | Live callers |
|---|---|---|
| `scripts/core/music_intelligence.py::CAMELOT` | identical copy | `scripts/core/music_intelligence.py::detect_key` |
| `scripts/core/track_metadata.py::_CAMELOT` via `scripts/core/track_metadata.py::key_to_camelot` | same codes, C sharp minor missing (MX-65) | external metadata lookups, legacy CLI |
| `scripts/core/spotify.py::spotify_key_to_camelot` | not audited | Spotify track parsing |
| `scripts/core/musical_analysis.py::CAMELOT_WHEEL`, `scripts/core/metrics.py::AudioMetrics._init_camelot_wheel` | no (MX-46) | legacy CLI, `scripts/core/python_bridge.py`, `scripts/core/real_optimizer.py`, `tests/e2e_test_suite.py` |
| `frontend/src/components/CamelotWheel.tsx::CAMELOT` | no (UI-01) | Mix Deck, Widget |

Key detectors, all Krumhansl style correlation:

| Detector | Used by | Notes |
|---|---|---|
| `scripts/core/key_detection.py::detect_key` | `analyze_structure`, `detect_key_segments` | HPSS, CQT at 36 bins per octave; profiles `ks`, `edma`, `edmm`, `auto` (centroid picks edmm or edma); invalid profile raises, which `analyze_structure` swallows (MX-12); any other failure returns C major 8B at confidence 0.0 (MX-28) |
| `scripts/core/music_intelligence.py::detect_key` | `compute_track_vector` without a key tuple, `_quick_features` | KS only, no HPSS |
| `scripts/core/track_metadata.py::LocalAnalysisProvider.from_file` | legacy CLI | loudest chroma bin |
| `scripts/core/musical_analysis.py::MusicalAnalyzer._detect_key_chroma`, `scripts/core/metrics.py::AudioMetrics.detect_key` | legacy CLI, e2e suite | mislabel most keys (MX-46) |

Confidence is `(best_r + 1) / 2`, and the best of the 24 correlations is never negative, so a successful detection reports at least 0.5 (MX-22).

Harmony scorers disagree on the same pair; pick the one your caller already uses:

| Scorer | Rule | Callers |
|---|---|---|
| `scripts/core/key_detection.py::psychoacoustic_consonance` | Sethares roughness between tonics, ignores mode; floor issue MX-27 | `plan_transition` `harmonic_score`, which drives the clash cut, the auto exit effect and the stem blend gate |
| `scripts/core/music_intelligence.py::_camelot_move_analysis`, `scripts/core/music_intelligence.py::camelot_harmonic_score` | DJ_THEORY move table; letters ignored for adjacency (MX-62) | `scripts/core/music_intelligence.py::compute_transition_score` (render `ts_*` fields), classic plan |
| `scripts/core/recommend.py::_camelot_score` | same table, 0.5 for unknown | `GET /recommend/{name}` |
| `scripts/core/track_metadata.py::MetadataClient.compatibility_score` with `scripts/core/track_metadata.py::camelot_distance` | any ring change scores 0 and vetoes (MX-04) | `POST /compatibility`, `GET /compatibility/matches/{name}` |
| `scripts/core/key_detection.py::camelot_modulation` | named moves with a cost | `scripts/core/setlist_planner.py::transition_cost`, `GET /setlist/camelot/modulation` |
| `scripts/core/key_detection.py::pitch_shift_for_camelot` | 0 when compatible, else smallest shift up to 3 semitones | `plan_transition` `suggested_pitch_shift` |
| `scripts/core/tiv_scoring.py::tiv_harmonic_score` | TIV distance | `plan_transition` only when a structure has `mean_chroma`, which nothing sets (MX-34) |

### Track vector and similarity

- `scripts/core/music_intelligence.py::MusicVector` from `compute_track_vector`: bpm, key fields, energy curve and stats, drop position, spectral features, danceability triple, vocal density and pitch, chord sequence, chroma. Stored on `SongStructure.music_vector`; `task_dj_remix` reuses it for `compute_transition_score`. The vocals stem is resolved through `scripts/core/audio_source.py::stem_files` by folder name under the library dir, not by the passed path.
- `scripts/core/music_index.py::_build_vector` builds the vector for `GET /library/similar/{name}` (weighted cosine, `scripts/core/music_index.py::_DEFAULT_WEIGHTS`). Key term and drop dimension: MX-61.
- `GET /library/crate-search` and `POST /library/build-clap-index` use `scripts/core/crate_digger.py`; see `DEVELOPING.md` "Embedding index vs. CLAP index".

## Transition planning

`plan_transition` in `scripts/core/dj_analysis.py` decides every field of `TransitionPlan`. Background: `docs/DJ_THEORY.md` sections 4 and 6; `DEVELOPING.md` "Key design decisions".

| Field | Rule | Consumed by |
|---|---|---|
| `transition_bars` | Request value (the API always passes one; schema default 16, range 8 to 64). With `None`, which no production caller passes: 8 when the BPM ratio differs from 1 by more than 0.08, else `dj.default_transition_bars` (MX-67). Always rounded up to a multiple of 8 (12 becomes 16; the docstring says nearest, DOC-04). Cut to 8 when `harmonic_score` is below 0.35 and bars are 16 or more | all renderers, stepped step count, bass swap |
| `transition_seconds` | `transition_bars * 4 * 60 / bpm_a`, recomputed after the clash cut | window length everywhere |
| `exit_bar_a`, `exit_time_a` | `SongStructure.best_exit_bar` capped at `total_bars - transition_bars`, then overridden by the last phrase boundary before 85 percent of duration, snapped to a phrase | window start in A, spectral sync window |
| `entry_bar_b`, `entry_time_b` | `SongStructure.best_entry_bar`, then the first phrase boundary after 0 s, then spectral matching when on | window start in stretched B |
| `bpm_a`, `bpm_b` | tracker BPM | stretch, bar length, echo spacing |
| `tempo_shift_ratio` | `bpm_a / bpm_b`, a rate: above 1 speeds B up | `_time_stretch`, entry index |
| `harmonic_score` | `psychoacoustic_consonance` on the local section key at the cue when its confidence is at least 0.35, else the whole track key; 0.5 if the call raises | clash cut, auto exit effect, stem gate |
| `tiv_compatibility` | always `None` (MX-34) | `scripts/core/remix_recipe.py` only |
| `suggested_pitch_shift` | `pitch_shift_for_camelot` on whole track codes | `render_stem_blend` only |
| `eq` | `scripts/core/dj_analysis.py::EQPlan`, built after the clash cut | renderers |

`EQPlan`: `hp_start_hz` and `hp_end_hz` from `dj.hp_filter_start_hz` and `dj.hp_filter_end_hz` (the B high pass ramp, inaudible, MX-42); `bass_swap_bar = transition_bars // 2`; `bass_crossover_hz` from `dj.bass_crossover_hz` (A high pass after the swap); `hp_ramp_bars` and the fade bar fields are metadata no renderer reads.

Planning traps:
- Bars are `beat index // 4` from the first detected beat, so exits and entries need not start on a real downbeat (MX-54).
- `scripts/core/dj_engine.py` defines its own `_HP_START_HZ`, `_HP_END_HZ`, `_BASS_CROSSOVER` and never reads them (MX-34); the values that matter are the module constants in `scripts/core/dj_analysis.py`, read from config at import, so a config change needs a restart.
- `preset` reaches the tasks only as audit metadata (MX-33). `transition_effect` and the bridge options are free strings (MX-07).
- The clash cut and the 0.35 branches depend on a scorer that stays high (MX-27).

## Rendering

All three renderers live on `scripts/core/dj_engine.py::DJEngine`. They duplicate the entry index and grid lock code; `render` and `render_chain` also duplicate the exit effect block, which `render_stem_blend` lacks (MX-13). A fix in one is not a fix in the others.

### render

`DJEngine.render(track_a, track_b, plan, full_output, bridge_beat, bridge_gain, transition_effect)`: full renders without the stem path, every preview, and the fallback inside `render_stem_blend` when no stem loads. Window length T is `transition_seconds`.

| Span | Song A | Song B |
|---|---|---|
| before window | untouched | absent |
| 0 to T/2 | full level, exit effect over the whole window | multiplied by zero |
| T/2 (bass swap) | high pass at 150 Hz from here on (`scripts/core/dj_engine.py::_butter_highpass`) | enters with full bass |
| T/2 to T | 3 band fade out (`DJEngine._apply_dynamic_eq_fade`, fixed 150 Hz and 2500 Hz splits) | 3 band fade in; only the mid band is equal power (MX-56) |
| after window | gone | rest of stretched B |

- Exit effect: `auto` and `stepped` resolve to `filter` (`scripts/core/dj_engine.py::_apply_filter_sweep`, low pass sweep) when `harmonic_score` is in [0, 0.35), else `echo` (`scripts/core/dj_engine.py::_apply_echo_out`: four echoes one beat apart at `bpm_a`, each 0.4 times the previous, peak normalised only above 1). `reverb` is `scripts/core/dj_engine.py::_apply_reverb_tail`. Any other string means no effect (MX-07).
- Bridge beat: tiled, sine arch over the window, times `bridge_gain` (clamped to [0, 1]).
- Input guards raise `ValueError` for `transition_seconds` outside (0, 300], non positive BPM or non finite audio.
- Output is peak normalised to 0.891 (about -1 dBFS).

### render_chain

`DJEngine.render_chain(tracks, plans, bridge_beats, bridge_gain, transition_effect, stems_dirs)`, called by `scripts/api/task_modules/remix.py::task_dj_chain` for 2 to 8 songs.
- Every song after the first is stretched once to song 1's BPM; cue times map through the `_orig_to_stretched` closure inside `render_chain`; window length uses song 1's BPM.
- Per pair: same grid lock, exit effect, B high pass ramp and A high pass after the swap as `render`, but a plain equal power cos and sin over the second half, no 3 band fade (MX-57).
- Segments: pre A, window, each middle song's body up to the next exit (MX-24), last song's tail (MX-39). Peak normalise to 0.891.
- The stem branch is dead because the task never passes `stems_dirs` (MX-09). If you wire it, the window would come from `render_stem_blend` on unstretched tracks while the body comes from the chain's stretched audio. Traps for the fix: the classic `scripts/core/classic_engine/dj_engine.py::DJEngine.render_chain` takes no `stems_dirs`, and `tests/test_render_e2e.py::test_chain_stops_between_songs_once_cancelled` runs chains on classic, so pass the argument only when `eng.name == "current"`; the branch gate is `Path(stems_dirs[i]).is_dir()`, and a song dir always exists, so pass `None` for songs that fail the `has_stems` check `task_dj_remix` does, or `render_stem_blend` quietly falls back to `render`; the branch passes neither `transition_effect` nor the bridge beat to `render_stem_blend` and applies no `_should_use_stem_blend` gate.

### render_stem_blend

`DJEngine.render_stem_blend(track_a, track_b, plan, stems_dir_a, stems_dir_b, full_output, bridge_beat, bridge_gain, transition_effect)`:

1. `scripts/core/dj_engine.py::_load_stem` for vocals, drums, bass, other (FLAC before WAV, resampled). If either side loads none, return `render(...)`.
2. Stretch B and every B stem by `tempo_shift_ratio`.
3. Pitch shift B and its stems by `suggested_pitch_shift` when nonzero and at most 3. This transposes the rest of song B too.
4. Per stem LUFS with `scripts/core/mastering.py::normalize_stems_to_corpus_targets` and `scripts/core/mastering.py::load_stem_targets` (`load_stem_targets` reads `data/stem_lufs_targets.json` merged over `scripts/core/mastering.py::_STEM_FALLBACK_TARGETS`; without the file the fallbacks are per stem, drums -18.5, bass -21.0, vocals -19.5, other -21.5 LUFS, never the flat -20 the `dj_engine.py` comment claims, DOC-04).
5. Entry index, clamp, grid lock, same arithmetic as `render`.
6. Masking EQ on the vocals, drums and other windows: `scripts/core/spectral_mixing.py::unmask_stem_pair` with `protect_bands=[0]`, keys `dj.masking_eq_enabled`, `dj.masking_eq_strength`, `dj.masking_eq_max_atten_db`.
7. `scripts/core/dj_engine.py::_stem_similarity` picks a curve from `scripts/core/dj_engine.py::_stem_crossfade_curves` (MX-58); `stepped` uses `scripts/core/dj_engine.py::_stepped_crossfade_curves` instead.
8. Bass: `scripts/core/dj_engine.py::_stem_bass_ramp` at `bass_swap_bar`, 100 ms taper ending at the swap for A and starting there for B.
9. Bridge beat, then scale the window only if its peak exceeds 1.
10. 30 ms equal gain `scripts/core/dj_engine.py::_seam_blend` at both edges, concatenate, scale only if the full peak exceeds 1.

| Stem similarity | Curve | Equal power |
|---|---|---|
| at least 0.75 | A holds then fades, looped from its last 2 bars (skipped under `stepped`); B linear in | no, sums above unity mid window |
| 0.40 to 0.75 | cos and sin over the whole window | yes |
| below 0.40 | A out over the first half, B in over the second | no, silent midpoint by design |
| `stepped` | square root staircase, one plateau per beat | yes |
| bass stem | swap at `bass_swap_bar` | never doubled, briefly absent |

Stem path traps: exit effects are never applied (MX-13); the window is LUFS normalised stems between raw full mix audio (MX-25); `stem_blend` in the result reports the gate, not the path (MX-51).

### Differences at a glance

| | `render` | `render_chain` | `render_stem_blend` | preview (`render`, `full_output=False`) |
|---|---|---|---|---|
| Caller | `task_dj_remix` without stem path | `task_dj_chain` | `task_dj_remix` when the gate passes | `task_remix_preview` |
| B audible from | T/2 | T/2 | window start, curve dependent | T/2 |
| Gain curve | 3 band dynamic fade | cos and sin, second half | per stem by similarity | as `render` |
| Bass | A high pass after swap | A high pass after swap | stem mute and ramp | as `render` |
| Exit effect | yes | yes | no | yes |
| Pitch shift | no | no | yes | no |
| Load cap | `remix.max_song_seconds` | same | same for the full mixes; the four stems are loaded whole by `scripts/core/dj_engine.py::_load_stem` and stretched (and pitch shifted) at full length before the window is sliced | none (MX-17) |
| Final level | peak 0.891 | peak 0.891 | scale only above 1 | peak 0.891 |
| After | spectral sync, master at `audio.target_lufs` | master at -8 LUFS, no sync | spectral sync, master | nothing, PCM_16 |

Preview parity: the preview loads whole files at A's native rate while the full render loads the capped audio at 44100 Hz, so the 85 percent exit threshold differs and the preview can audition a different exit bar (MX-17). It never takes the stem path, never pitch shifts, and skips sync and mastering. The Mix Deck preview also drops effect and preset (UI-09).

## Beat alignment and tempo

- `scripts/core/dj_engine.py::_time_stretch` clamps through `scripts/core/dj_engine.py::_safe_ratio` to [`_MIN_STRETCH`, `_MAX_STRETCH`] (0.5 to 2.0; non finite or non positive becomes 1.0), warns near a bound (an octave error signature), and returns the input unchanged when the ratio is within 0.02 of 1 (MX-23). Otherwise `scripts/core/gpu.py::gpu_time_stretch`, then librosa.
- Entry index: `entry_time_b * sr / ratio`, then `scripts/core/dj_engine.py::_clamp_entry_sample` backs the entry up so a full window exists. The division happens even inside the no stretch band (MX-02); `render_chain` has the same arithmetic.
- Grid lock: B moves by the difference of the two bar phases, each measured from sample 0 of its file, wrapped to half a bar. Since both cues are already tracker bar starts, a nonzero correction moves B off its own bar start (MX-03). The block is copied in all three renderers; `DEVELOPING.md` "Beat grid lock" describes it without this caveat.
- `tempo_shift_ratio` is `bpm_a / bpm_b` and only B is stretched. The classic engine had this inverted until e251f22 (see "Engine selection").
- GPU stretch: runs only when torch imports and `scripts/core/gpu.py::get_device` is not `cpu` (`REMIXMATE_DEVICE` overrides). On MPS it raises and falls back to librosa silently (MX-59). Also used by `scripts/core/instrument_lab.py`, `scripts/core/musical_analysis.py` and `scripts/core/real_optimizer.py`.

## Spectral matching and sync

Design notes: `DEVELOPING.md` "Key design decisions" (Learned spectral cue matching, Dynamic spectral sync). Both are engine agnostic and run from the task module, so they apply to classic renders too.

Matching, `scripts/core/spectral_matching.py::choose_entry_bar` (gated by `dj.spectral_matching`):
- A's profile is the `transition_seconds` before its exit, not the audio that plays in the window.
- Candidates are B's bars 0, 8, 16 and so on up to `scripts/core/spectral_matching.py::_MAX_CANDIDATES`, compared as mel trajectories (`scripts/core/spectral_matching.py::mel_trajectory`, `scripts/core/spectral_matching.py::trajectory_similarity`) in B's original time (the docstring claim is DOC-04), weighted by `scripts/core/spectral_matching.py::SpectralMatchModel` weights, times `scripts/core/spectral_matching.py::_drop_alignment_factor`.
- Learning: `record_render` and `scripts/core/spectral_matching.py::apply_feedback` write `data/spectral_match_model.json` and `data/spectral_match_history.json` at `dj.spectral_matching_lr`; `REMIXMATE_SPECTRAL_LEARNING` off freezes it. Any exception leaves the plan untouched.

Sync, `scripts/core/spectral_sync.py::sync_transition_spectrum` via `apply_spectral_sync` (gated by `dj.spectral_sync`):
- Window starts at `exit_time_a` on the rendered timeline for `transition_seconds`. The target morphs by a raised cosine from the 8 s before to the 8 s after (MX-63).
- Gains capped at `dj.spectral_sync_max_db`, damped on high flux frames, smoothed, tapered to 0 dB at both edges, skipped when the window does not fit.
- The bass guard is meant to leave bands at or below `dj.bass_crossover_hz` alone; at the configured value it protects none (MX-29).
- Not run for chains or previews.

## Mastering

Background: `docs/DJ_THEORY.md` "6. EQ and Frequency Management" (Gain Staging).

| Caller | Function | Target |
|---|---|---|
| `task_dj_remix`, current | `scripts/core/mastering.py::master_mix` | `audio.target_lufs` in `config.yaml`, fallback in `_resolve_engine` |
| `task_dj_remix`, classic | `scripts/core/classic_engine/mastering.py::master_mix` | `scripts/core/classic_engine/__init__.py::CLASSIC_DJ_TARGET_LUFS` |
| `task_dj_chain`, either engine | the engine's `master_mix` | literal -8.0 at the call, ignores config |
| `task_remix_preview` | none | peak 0.891, 16 bit |
| `scripts/core/style_transfer.py::_master_and_save`, `scripts/core/inpainting.py::run_inpainting` | current `master_mix` | literal -14.0 |
| `scripts/core/instrument_lab.py::render_combo` | none | peak 0.891, LUFS measured only |

`config.local.yaml` sets neither `audio.target_lufs` nor `dj.mixing_engine` (as of 72e2368, `grep -c target_lufs config.local.yaml` and `grep -c mixing_engine config.local.yaml` both printed 0), although a comment in `_resolve_engine` suggests -8.0 there for club use.

Current `master_mix` steps:
1. `scripts/core/mastering.py::normalize_to_lufs`: one gain to the target, clamped to plus or minus 20 dB, skipped at or below -70 LUFS.
2. `scripts/core/mastering.py::apply_limiter`: 4x oversampled look ahead peak envelope, 2 ms attack and 50 ms release through `scripts/core/mastering.py::_smooth_gain_envelope` (numba with a Python fallback), clip at the ceiling, decimate, then one corrective gain if the true peak is still over. Ceiling -1 dBFS.
3. Report with `scripts/core/mastering.py::compute_lufs` (own BS.1770 gating, not pyloudnorm). `scripts/core/mastering.py::QualityReport` fields: `lufs_integrated`, `lufs_target`, `lufs_gain_applied`, `peak_dbfs`, `has_clipping`, `clip_count`, `dynamic_range_db`, `passed`, `notes`. Clipping is judged on the input to `master_mix` (MX-44). `passed` is false on input clipping, output LUFS more than 3 LU off target, or a peak over the ceiling. The truncation fade runs after the report (MX-38). The UI ignores `quality_passed` (UI-15).

Classic `master_mix`: same normalise step, then a sample peak limiter with one time constant in a per sample Python loop (slow on long mixes) and a hard clip; clipping verdict: MX-21.

Per stem loudness (current engine only): corpus targets come from `POST /library/calibrate-lufs` (`scripts/api/routers/library.py::calibrate_stem_lufs`, operator only) through `scripts/core/mastering.py::analyze_library_stem_targets` and `scripts/core/mastering.py::save_stem_targets`. `scripts/core/mastering.py::normalize_stems_to_target` has no production caller.

## Stems

Procedures: `DEVELOPING.md` "Stem processing (Demucs)" and the `stem-processing` skill; that `DEVELOPING.md` section is wrong in three places (DOC-03).

| Route | Guards | Task |
|---|---|---|
| `POST /stems/split` (`scripts/api/routers/stems.py::stem_split`) | `require_operator`, `scripts/api/routers/stems.py::_stem_split_enabled` (503 unless `REMIXMATE_STEM_SPLIT_ENABLED`), job cap, `_require_song` | `scripts/api/task_modules/stems.py::task_stem_split` |
| `POST /stems/split-batch` (`scripts/api/routers/stems.py::stem_split_batch`) | same | `scripts/api/task_modules/stems.py::task_batch_stem_split` |
| `POST /stems/compress`, `POST /stems/compress-batch` | `require_operator` | `scripts/api/task_modules/stems.py::task_compress_stems`, `scripts/api/task_modules/stems.py::task_batch_compress_stems` |
| `POST /library/initialize` (`scripts/api/routers/library.py::initialize_library`) | operator, split flag | `scripts/api/task_modules/analysis.py::task_initialize_library` (split, compress, index) |

Other callers of `scripts/core/stems.py::separate_song_stems`: operator ingest through `scripts/download.py::_separate_stems` (always `enhance=False`) and the module CLI. API splits default to `enhance=True`, so API and ingest stems come from different input processing. `scripts/core/stems.py::separate_batch` has no caller.

`separate_song_stems`:
1. Source from `scripts/core/audio_source.py::resolve_source_file` (`full.wav`, then `full_enhanced.wav`); stems alone cannot be split again.
2. Load mono 44100 Hz. `preview_seconds` requires an explicit `output_dir` so a preview never lands in the library.
3. Enhance when asked: `scripts/core/audio_enhance.py::enhance_audio` (filters, gate, compression, LUFS -14, then `scripts/core/audio_enhance.py::apply_true_peak_limiter`, MX-52), written to `full_enhanced.wav`.
4. Demucs as a subprocess, model from `scripts/api/schemas.py::StemSplitRequest` (OPS-13); a GPU failure retries once on CPU.
5. Move the four WAVs into the song dir, normalise each with `scripts/core/audio_enhance.py::enhance_stems` (per stem LUFS targets), rewrite mono 24 bit; a failure here is swallowed.
6. Success when at least one stem exists (MX-49).

Interpreter (`scripts/core/stems.py::_venv_python`, docstring drift OPS-15): the first existing of the shebang interpreter of a `demucs` script on `PATH`, `remix-env/bin/python3`, `remix-env/bin/python`, then `sys.executable`. On this checkout `remix-env/` exists (gitignored) with its own demucs and torch, so Demucs normally runs there, not in `.venv`. Deleting `remix-env/` moves Demucs to `sys.executable`. Device: `scripts/core/stems.py::_detect_device` through `get_device`. The `separation:` block in `config.yaml` is read by no stem code (OPS-14).

Pruning, `scripts/api/task_modules/stems.py::_prune_after_split` after every successful split: gated by `library.prune_on_download` (fails closed), then `scripts/core/library.py::LibraryManager.prune_raw` and `scripts/core/library.py::LibraryManager.prune_enhanced`, which delete `full.wav` and `full_enhanced.wav` only when all four stems pass `scripts/core/library.py::_stem_is_healthy` and `library.keep_raw_after_separation` is false. `config.yaml` sets `prune_on_download: true` and `keep_raw_after_separation: false`, so a clean split leaves stems only. Consequences: no resplit with another model, batch discovery skips the song, render source becomes a stem sum whose balance differs from the original (MX-30), and generative requests on `full` fail (MX-31).

Compression rewrites each WAV as 24 bit FLAC and unlinks the WAV by default; a stem whose FLAC exists is skipped (MX-64). `stem_files`, both engines' `_load_stem` prefer `.flac`. The remix gate treats a song as having stems when any one stem file exists. WAV only readers: `scripts/core/codec_tokens.py::tokenize_stems` (MX-53), `scripts/core/paths.py::vocals_path`, `scripts/core/paths.py::other_path`.

## Generative and instrument lab

| Feature | Route | Gate | Task | Core |
|---|---|---|---|---|
| Style transfer | `POST /ai/style-transfer` | `scripts/api/routers/generative.py::_generative_enabled` (503 unless `REMIXMATE_GENERATIVE_ENABLED`) | `scripts/api/task_modules/generative.py::task_style_transfer` | `scripts/core/style_transfer.py::run_style_transfer`, MusicGen melody (`generative.musicgen_model`) via `scripts/core/model_manager.py::get_manager` |
| Inpainting | `POST /ai/inpaint` | same | `scripts/api/task_modules/generative.py::task_inpaint_transition` | `run_inpainting`, DAC and VampNet with `scripts/core/inpainting.py::_inpaint_cosine_fallback` |
| Tokenize | `POST /ai/tokenize` | same, no operator check (MX-50) | `scripts/api/task_modules/generative.py::task_tokenize_stems` | `tokenize_stems`, EnCodec or DAC loaded directly |
| Model status | `GET /ai/models` | same | none | `ModelManager.status` |
| Instrument Lab | `POST /instrument-lab` in `scripts/api/routers/remix.py` | signed in user only (MX-26) | `scripts/api/task_modules/lab.py::task_instrument_lab` | `scripts/core/instrument_lab.py::render_all_combos` |

- The `/ai/*` stack is off unless the flag is set; its packages sit in the `generative` extra of `pyproject.toml`, and VampNet is not on PyPI (see `requirements.txt`). The AI Lab page hides itself when the flag is off.
- Generative output: the core writes into shared staging dirs under the outputs root, then `scripts/api/task_modules/generative.py::_publish_output` moves the file to `gen_<job_id>/` in the caller's outputs dir. `tokenize_stems` writes `tokens/` into the shared library song dir.
- Instrument Lab `render_combo`: anchor tempo from the drums source song (MX-60), stems via `scripts/core/dj_engine.py::_load_stem`, stretch with `gpu_time_stretch` when BPMs differ by more than 1, sum, trim, peak normalise, write 16 bit WAV under `lab_<id>/`. Mode `targeted` renders single stem swaps; any other mode renders every combination (MX-26). Labels: MX-43; use `mapping`, not letters.

## Engine selection and the classic engine

`_resolve_engine` runs once per job in `task_dj_remix`, `task_dj_chain` and `task_remix_preview` and returns `scripts/api/task_modules/remix.py::MixEngine` (`analyze`, `plan_transition`, `DJEngine`, `master_mix`, `dj_target_lufs`, `name`). It reads `dj.mixing_engine`; any exception, missing key or unknown value yields `classic`. `scripts/core/config.py::DJConfig` defaults to `"classic"`; `config.yaml` sets `current`; `REMIXMATE_DJ_MIXING_ENGINE` overrides both. The docstring still calls classic the default (DOC-05). The engine name lands in each job result as `engine`. Analysis endpoints, setlist planning, `scripts/core/remix_recipe.py` and the CLI bridge always use the current modules.

| Aspect | classic | current |
|---|---|---|
| Beat tracking | `librosa.beat.beat_track` | `get_configured_tracker` |
| Harmonic score | `camelot_harmonic_score` on whole song codes | `psychoacoustic_consonance` on the local section key |
| Bass swap vs clash cut | swap computed before the cut (MX-11) | after the cut |
| Stretch clamp | none | `_safe_ratio` |
| Medium stem curve | raised cosine, equal gain | cos and sin, equal power |
| Stem LUFS, masking EQ, bass ramp, pitch shift, stepped, seam blend, entry clamp | absent | present |
| Limiter | sample peak, Python loop | 4x true peak |
| Remix loudness | -8 LUFS | `audio.target_lufs` |

The frozen rule: `scripts/core/classic_engine/__init__.py` and `DEVELOPING.md` say the package is a byte faithful copy. `git log --oneline -- scripts/core/classic_engine` shows later commits after the vendoring commit f240073: 9e93c05 (a docstring), 530ba73 and e7a053b (lint), and e251f22 (tempo ratio flipped to `bpm_a / bpm_b` and the `render_chain` pre stretch flipped). Classic output no longer matches the reference renders (DOC-06). Classic imports from live code only `scripts/core/config.py`, `scripts/core/gpu.py` and `scripts/core/music_intelligence.py`.

Rule: do not edit `scripts/core/classic_engine/` unless the task is explicitly about classic. Shared behaviour goes in `scripts/api/task_modules/remix.py` (as spectral matching and sync do) or in the current modules.

## Legacy and sandbox code

Never edit these unless asked.

| Code | Who reaches it | Edit? |
|---|---|---|
| `scripts/core/python_bridge.py::PythonBridge` | CLI `remixmate bridge` (`scripts/remixmate_cli.py::cmd_bridge`) and its own `main`; no test (the `tests/test_core_modules.py` docstring names a bridge compatibility test that does not exist); `PythonBridge.dj_remix` has no caller (MX-47) | only for CLI bugs |
| `scripts/core/python_mixer.py::PythonMixer` | only `python_bridge.py` | no |
| `scripts/core/pro_audio_chain.py::ProfessionalAudioChain` | only `tests/e2e_test_suite.py`, which CI ignores | no |
| `scripts/core/musical_analysis.py`, `scripts/core/metrics.py`, `scripts/core/real_optimizer.py` | legacy CLI, bridge, e2e suite | no (MX-46) |
| `scripts/core/remix_recipe.py::generate_remix_recipe` | only `tests/test_remix_recipe.py`, no route | safe to extend; reads the current `TransitionPlan` |
| `scripts/remixmate_cli.py::generate_stems_and_params` | no caller | no |
| `scripts/core/stems.py::separate_batch`, `DJEngine.render_from_stems` in both engines | no caller (MX-34) | no |
| `ai_remixmate_feature_lab/` | nothing in `scripts/`, `tests/` or `frontend/src` imports it; excluded from ruff and pytest | read only reference |
| `archive/` | nothing in `scripts/`, `tests/` or `frontend/src` imports it | read only history |

## Set planning and recommendation

Music rules: `docs/DJ_THEORY.md` sections 2, 3 and 9.

| Route | Handler | Core |
|---|---|---|
| `POST /setlist/optimize` | `scripts/api/routers/setlist.py::optimize_setlist` (no track cap, MX-16) | `scripts/core/setlist_planner.py::SetlistPlanner.optimize` |
| `POST /setlist/import-csv` | `scripts/api/routers/setlist.py::import_exportify_csv` | `scripts/core/setlist_planner.py::parse_exportify_csv`, then `optimize` |
| `GET /setlist/camelot/modulation` | `scripts/api/routers/setlist.py::camelot_modulation` | `scripts/core/key_detection.py::camelot_modulation` |
| `POST /setlist/wordplay` | `scripts/api/routers/setlist.py::find_wordplay` | `scripts/core/wordplay.py::find_wordplay_pairs` |
| `GET /recommend/{name}` | `scripts/api/routers/analysis.py::recommend_songs` | `scripts/core/recommend.py::get_recommendations` |
| `GET /compatibility/matches/{name}` | `scripts/api/routers/analysis.py::compatibility_matches` | `scripts/core/recommend.py::top_compatibility_matches` (409 without cached analysis) |

- `transition_cost` is `w_harmonic * harmonic + w_bpm * bpm + w_energy * energy`, lower is better. Harmonic from `camelot_modulation` (0.5 when a key is missing); BPM cost `|delta| / 30` capped at 1; energy mixes distance from the arc target (`scripts/core/setlist_planner.py::_arc_target_energy`, MX-48) and the jump from the previous track.
- `optimize` is greedy from an arc dependent anchor. Both routes construct `SetlistPlanner()` bare, so `markov_weight` has no effect. Wordplay bonus: MX-36. `start_track` is not exposed over HTTP. The frontend caller `frontend/src/lib/api.ts::setlistApi` reads only `setlist[].name`.
- `get_recommendations` takes the source BPM from `meta.json` or a 10 s estimate that it writes back (MX-32), estimates a bounded number of uncached songs per call, and ranks with `scripts/core/recommend.py::_composite_score` (MX-05).

## Config keys

All in `config.yaml` unless noted; dataclasses in `scripts/core/config.py`. `DEVELOPING.md` "Gotchas" covers adding a key (it needs a dataclass field).

| Key | File | Reader | Effect |
|---|---|---|---|
| `dj.mixing_engine` | `config.yaml`, env `REMIXMATE_DJ_MIXING_ENGINE` | `_resolve_engine` | `current` or `classic`; anything else is classic |
| `audio.target_lufs` | `config.yaml` | `_resolve_engine` | full remix loudness on current; not chains, previews or generative |
| `dj.default_transition_bars` | `config.yaml` | `scripts/core/dj_analysis.py` at import, only when `plan_transition` receives `transition_bars=None`; no production caller does | none on API renders (MX-67): every route passes `req.transition_bars`, whose default is the pydantic 16 in `scripts/api/schemas.py::DJRemixRequest`, `scripts/api/schemas.py::DJPreviewRequest` and `scripts/api/schemas.py::DJChainRequest`; change those to change the default window. `tests/test_core_modules.py::TestConfig.test_dj_values` pins the key to 8, 16 or 32 |
| `dj.hp_filter_start_hz`, `dj.hp_filter_end_hz` | `config.yaml` | `scripts/core/dj_analysis.py` at import | B high pass ramp (MX-42) |
| `dj.bass_crossover_hz` | `config.yaml` | `scripts/core/dj_analysis.py`, `scripts/core/spectral_sync.py` | A high pass after the swap; sync bass guard (MX-29) |
| `dj.transition_confidence_threshold` | `config.yaml` | `task_dj_remix` stem gate | stem path cutoff, unreachable on the current engine (MX-27); its comment claims 0.35 (DOC-04) |
| `dj.spectral_matching`, `dj.spectral_matching_lr` | `config.yaml` | `scripts/core/spectral_matching.py` | entry bar choice, learning rate |
| `dj.spectral_sync`, `dj.spectral_sync_max_db` | `config.yaml` | `scripts/core/spectral_sync.py` | post render EQ on or off, gain cap |
| `dj.masking_eq_enabled`, `dj.masking_eq_strength`, `dj.masking_eq_max_atten_db` | `config.yaml` | `scripts/core/dj_engine.py` at import (restart to change), passed to `unmask_stem_pair` from `render_stem_blend` | stem window masking EQ; `strength` 0.0 is full 1x ducking, not off (the multiplier is `2**strength`), only `masking_eq_enabled: false` disables it |
| `remix.max_song_seconds` | `config.yaml` | `_max_song_seconds` | render load cap; the cache window is a separate hardcoded 120 s |
| `remix.max_pitch_shift_semitones` | `config.yaml` | nothing (OPS-11) | `pitch_shift_for_camelot` hardcodes 3 |
| `analysis.beat_backend` | `scripts/core/config.py::AnalysisConfig` default, env `REMIXMATE_ANALYSIS_BEAT_BACKEND` | `get_configured_tracker` | `auto`, `librosa`, `beat_this`; else raises |
| `library.prune_on_download`, `library.keep_raw_after_separation` | `config.yaml` | `_prune_after_split`, `LibraryManager.prune_raw` | delete sources after a clean split |
| `library.auto_evict_on_download` | `config.yaml` | nothing (OPS-12) | none |
| `separation.*` | `config.yaml` | nothing (OPS-14) | none; the model comes from the request |
| `retention.*` | `config.yaml`, `scripts/core/config.py::RetentionConfig` | `scripts/core/mix_retention.py` | mix age, count, sweep timing |
| `generative.musicgen_model` | `config.yaml` | model manager | style transfer model |
| `metadata.*` | `config.yaml` | `scripts/core/track_metadata.py::MetadataClient` | external lookup for `/compatibility` songs outside the library |
| `library.library_dir` | `config.yaml` | `scripts/core/paths.py` | where `meta.json` and `analysis.json` land |

Env flags read per request: `REMIXMATE_STEM_SPLIT_ENABLED`, `REMIXMATE_GENERATIVE_ENABLED`. Also: `REMIXMATE_SPECTRAL_LEARNING`, `REMIXMATE_DEVICE`. Not read by analysis: `audio.sample_rate`.

## Judging a mixing change

CI (`.github/workflows/ci.yml`) runs pytest twice: once with `-m "not dj_analysis"` ignoring `tests/e2e_test_suite.py`, once with `-m dj_analysis`. Most analysis tests carry the `dj_analysis` marker and skip when librosa or numba fails to import. Commands to run them locally: `DEVELOPING.md` "Commands".

| Change | Run |
|---|---|
| `scripts/api/task_modules/remix.py` orchestration or result fields | `tests/test_render_e2e.py`, `tests/test_mix_credits.py`, `tests/test_spectral_sync.py`, `tests/test_job_meta_request.py`, `tests/test_lifespan.py` |
| Render paths in `scripts/core/dj_engine.py` | `tests/test_behavioral.py`, `tests/test_dj_engine_guards.py`, `tests/test_stem_bass_ramp.py`, `tests/test_stepped_crossfade.py`, `tests/adversarial/test_audio_engine_correctness.py`, then `tests/test_render_e2e.py` |
| Analysis or planning | `tests/test_core_modules.py`, `tests/test_beat_tracker.py`, `tests/test_behavioral.py`, `tests/test_spectral_matching.py`, `tests/test_render_e2e.py` |
| Cached analysis files | `tests/test_audio_sources.py`, `tests/test_silent_failures.py`, `tests/test_structure_endpoint.py` |
| Harmony scoring | `tests/test_compatibility_score.py`, `tests/test_compatibility_matches.py`, `tests/test_key_shift_rules.py`, `tests/test_tiv_scoring.py` |
| Classic engine | `tests/test_classic_engine.py`, `tests/test_render_e2e.py` |
| Mastering or LUFS | `tests/test_mastering.py` |
| Stems | `tests/test_stems.py`, `tests/test_prune_after_split_wiring.py`, `tests/test_batch_skip_existing.py`, `tests/test_stem_split_batch_discovery.py`, `tests/test_initialize_library_accounting.py` |
| Generative and lab | `tests/test_generative_gate.py`, `tests/test_generative_urls.py`, `tests/test_hardening.py` |
| Job store, SSE, cancel | `tests/test_behavioral.py`, `tests/test_job_list_filters.py`, `tests/test_tenancy_jobs.py`, `tests/test_hardening.py`, `tests/test_account_lifecycle.py`, `tests/test_admin_takedown.py` |
| Retention | `tests/test_mix_retention.py`, `tests/test_lifespan.py` |
| Storage prune | `tests/test_storage_prune_safety.py`: its fixture builds `LibraryManager(library_dir=tmp)`, registers songs and patches `scripts.core.library.get_library_manager`, then calls the route function directly; reuse it rather than touching `library/` (OPS-02) |
| Setlist | `tests/test_setlist_planner.py`, `tests/test_hardening.py`, `tests/test_silent_failures.py` |

LUFS targets: full remix on current masters to `audio.target_lufs`; on classic to `CLASSIC_DJ_TARGET_LUFS`; chains to a literal -8.0 on both engines; previews are not mastered. The verdict lands in `quality_passed` and `quality_notes`.

Audio on this machine (local only, never in CI):
- `library/` holds 588 entries (as of 72e2368, `ls library | wc -l`).
- 5 of them keep a `full.wav` (as of 72e2368, `ls library/*/full.wav 2>/dev/null | wc -l`) and 5 a `full_enhanced.wav` (as of 72e2368, `ls library/*/full_enhanced.wav 2>/dev/null | wc -l`).
- 588 vocals stem files (as of 72e2368, `ls library/*/vocals.* 2>/dev/null | wc -l`) and 583 `analysis.json` files (as of 72e2368, `ls library/*/analysis.json 2>/dev/null | wc -l`).
- `catalog/` holds 3 song dirs (as of 72e2368, `ls catalog | wc -l`); each holds four stem WAVs, `analysis.json`, `meta.json` and `license.json`, no `full.wav` (as of 72e2368, `ls catalog/*/`).

So nearly every local render loads through the stem sum branch of `load_source_audio`, not an original mix (MX-30). `tests/test_render_e2e.py` builds its own synthetic library and depends on none of this. Never run a destructive route against `library/` to test a gate (OPS-02).

## How to change a transition safely

1. Pick the renderer. `render`, `render_chain` and `render_stem_blend` duplicate the entry index and grid lock code; `render` and `render_chain` also duplicate the exit effect block, which `render_stem_blend` lacks (MX-13). A fix in one is not a fix in the others; the preview is `render`. Check whether classic needs the change at all (usually not).
2. Keep these invariants:
   - medium and stepped stem curves satisfy `fade_out**2 + fade_in**2 == 1`; seams stay equal gain;
   - bass is never present from both songs at once (`_stem_bass_ramp` on the stem path, A high pass from `bass_swap_bar` elsewhere);
   - `bass_swap_bar` is computed after the clash cut;
   - `tempo_shift_ratio` stays `bpm_a / bpm_b`, and B is stretched, never A;
   - peak normalisation runs after the bridge beat layer; loudness is set only in `master_mix`;
   - `scripts/core/spectral_matching.py` and `scripts/core/spectral_sync.py` touch only `plan.entry_bar_b`, `plan.entry_time_b` and the rendered array.
3. Keep the result fields listed under "Job lifecycle" stable; the frontend reads them.
4. Run the render row of the table above, then `tests/test_render_e2e.py` (both engines). Several files carry the `dj_analysis` marker, which CI runs as its own step.
5. For timing changes, add a test that plants a click track with a known downbeat offset in each song and asserts the offsets line up in the output. None exists today (MX-02, MX-03, MX-23).
6. Listen on a real song only after the tests pass, and remember most local songs are stem sums.

## Tests and gaps

Covered: ratio and BPM guards (`tests/test_dj_engine_guards.py::TestSafeRatio`, `tests/test_dj_engine_guards.py::TestTimeStretch`); entry clamps (`tests/test_dj_engine_guards.py::TestRenderIndexClamp`); B silent in the first half (`tests/test_behavioral.py::TestRenderBSilence`); equal power on the medium and stepped curves (`tests/test_behavioral.py::TestEqualPowerCrossfade`, `tests/test_stepped_crossfade.py::TestSteppedCurve`); bass swap after the clash cut (`tests/test_behavioral.py::TestClashPathBassSwap`); seams and masking (`tests/test_behavioral.py::TestSeamBlend`, `tests/test_behavioral.py::TestMaskingAwareMultibandEQ`); stem LUFS (`tests/test_core_modules.py::TestStemLufsNormalization`); planner rounding and ratio direction (`tests/test_core_modules.py::TestDJTransitionPlanner`); structure, per phrase keys, profiles and consonance (`tests/test_core_modules.py::TestDJStructureAnalysis`, `tests/test_core_modules.py::TestPerPhraseKeyDetection`, `tests/test_core_modules.py::TestKeyDetectionProfiles`, `tests/test_core_modules.py::TestPsychoacousticConsonance`); job lifecycle, filters, ownership and SSE scoping; retention; remix and chain end to end on synthetic audio with chain cancel, load cap and fade.

Gaps (from the sections' greps over `tests/`):
- Timing: no downbeat phase test after the grid lock, nothing in the no stretch band, no `gpu_time_stretch` test. The guard fixture uses the inverted ratio (MX-66).
- Clash: covered only with `psychoacoustic_consonance` patched to a low value in `tests/test_behavioral.py` (MX-27).
- Render: no chain with stems, no preview versus full parity, no check of exit effects on the stem path, no check that spectral sync leaves the bass band alone.
- Analysis: nothing tests `_label_sections` directly, `has_analysis`, `_camelot_move_analysis`, `compute_transition_score`, `_camelot_score`, `_build_vector`, relative key scoring in `compatibility_score`, or the cached `duration`.
- Jobs and planning: no test of `get_recommendations` ranking, cancel of a running remix or preview, unknown `transition_effect` values, `markov_weight`, `JobResponse` with `meta`, or `scripts/api/jobs.py::retry_job` (no caller either, MX-34).
- Stems and lab: no real Demucs run, no `_venv_python` order test, no classic clipping verdict test, no Instrument Lab render, no generative source on a pruned song, no `PythonBridge` test.

## Contradictions with docs/DJ_THEORY.md

The code wins; fix the doc or mark the section as theory.

- Bass swap at bar T*0.75, stretch limits of 3 to 5 BPM, 4 to 8 bar clash transitions, and same number A and B pairs called both relative and parallel: DOC-02.
- "Why clashing keys physically hurt" and "Camelot Move Selection Algorithm" assume a harmonic scorer that reaches low values; the current engine's does not (MX-27).
- "Key Detection Pipeline" calls confidence the normalised correlation; the code keeps every success at or above 0.5 (MX-22). It also describes KS profiles, while the default `key_profile=auto` picks between the EDM profiles.
- "The Seven Camelot Moves (Scored)" names +7 an energy boost; `camelot_modulation` uses `energy_boost` for plus or minus 2 and reserves `modulated_boost` for a +7 branch that cannot fire.
- "7. ML Foundations" lists downbeat detection via a DBN; the default tracker assumes the first beat is a downbeat (MX-54).
- Same number A and B scoring differs across scorers (MX-04, MX-62).
- `DEVELOPING.md` drift on the low shelf, stem config and harmonic systems: DOC-03.
