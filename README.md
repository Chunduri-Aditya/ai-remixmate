# 🎛️ AI RemixMate

> **An AI-powered DJ engine** — download tracks, analyse their structure, and render seamless DJ mixes with beat-locked crossfades, stem-aware mixing, RAG-powered smart search, and a full REST API + live web UI.

**Version:** 0.2.0 · **Python:** 3.10+ · **Last updated:** 2026-04-06

---

## What it does

AI RemixMate is a full-stack audio engineering project built in Python. Give it two songs, and it:

1. **Analyses both tracks** — detects BPM, key, energy, genre, and bar structure using librosa
2. **Plans a DJ transition** — picks exit and entry cue points at musical phrase boundaries, computes the tempo ratio for time-stretching
3. **Beat-grid locks the handoff** — aligns Song B's downbeat to Song A's bar grid at the crossover point (sample-level precision)
4. **Renders a stem-aware crossfade** — optionally uses Demucs-separated stems so drums/bass/vocals can be faded independently
5. **Applies dynamic EQ fade** — low-end frequencies from Song A are shelved down as Song B's low-end rises, preventing bass frequency stacking
6. **Optionally layers a bridge beat** — synthesised from scratch (numpy/scipy, no sample files) using genre-matched drum patterns and a bell-curve volume envelope
7. **Outputs compressed FLAC or WAV** — normalised to −1 dBFS, 16-bit PCM, with optional lossless FLAC compression (≈ 60 % smaller files)

All driven by a **FastAPI backend** with async job queue + SQLite persistence, structured JSON logging, an immutable audit trail, and a **Streamlit web UI** with a dark DJ-booth theme.

---

## Features

### Core Audio Engine
- **Stem-aware mixing** — Demucs-separated vocals, drums, bass, and "other" can be faded independently across the transition window
- **Dynamic EQ fade** — low-shelf filter transitions prevent bass frequency collision between outgoing and incoming tracks
- **Procedural beat synthesis** — 6 genre presets (techno, house, hip-hop, trap, DnB, ambient) built from sine-envelope kick, filtered-noise snare, and highpass hi-hats — pure numpy, zero sample files
- **FLAC compression** — all rendered mixes can be saved as lossless FLAC for ≈ 60 % smaller library footprint
- **Beat-grid lock + time-stretch** — sample-level bar-grid phase correction and librosa phase-vocoder tempo matching
- **GPU acceleration** — `gpu.py` auto-detects Apple Silicon MPS, NVIDIA CUDA, or falls back to CPU; accelerates STFT, cosine similarity (50–100x for large libraries), time-stretching, and resampling
- **ITU-R BS.1770-4 mastering** — integrated LUFS measurement, K-weighting, true-peak brick-wall limiter with look-ahead
- **Pre-Demucs enhancement** — DC offset removal, noise gate, soft-knee compression, air EQ, LUFS normalization pipeline

### Library & Discovery
- **Song library management** — download from YouTube Music via `yt-dlp`, optional Demucs stem separation
- **Genre detection** — multi-feature classifier (spectral centroid, MFCC, zero-crossing rate, RMS energy) across 8 genres
- **Compatibility scoring** — BPM, key (Camelot wheel), and energy scoring before you commit to a mix
- **35-dim vector similarity index** — numpy embedding index (`music_index.py`, JSON-persisted) for semantic similarity search — no FAISS dependency
- **Smart Search** — natural-language queries ("dark progressive techno 128 BPM") routed through the vector index via weighted cosine similarity
- **BPM-cache recommendations** — sub-second suggestions on warm cache, invalidated automatically on library change
- **Crates, Tags & Favorites** — named song groups, per-song tags, and favorites all persisted to SQLite (`data/remixmate.db`)

### API & Infrastructure
- **Async job queue** — `ThreadPoolExecutor`-backed with ETA estimation, progress reporting, and structured logging per job
- **SQLite job persistence** — `data/jobs.db`; jobs survive process restarts; in-flight jobs automatically rolled back to `FAILED` on restart; in-memory dict for O(1) reads with write-through to disk
- **Job cancellation & retry** — `DELETE /jobs/{id}` cancels pending/running jobs; failed/cancelled jobs can be cloned and re-submitted via `retry_job()`
- **Rate limiting** — configurable `max_active_jobs` cap with HTTP 429 responses
- **Health endpoints** — `/health/live` (liveness → `{"status": "ok"}`) and `/health/ready` (readiness + dependency checks)
- **Immutable audit trail** — every download, remix, deletion, and library change is appended to `data/audit.jsonl`
- **Structured JSON logging** — request IDs and job IDs propagated through all log records via `contextvars`
- **Strudel code export** — generate a live-coding pattern for [strudel.cc](https://strudel.cc) from any bridge beat
- **Spotify integration** — import individual tracks or full playlists via OAuth; all imports create proper job records

### UI

> **Supported frontend:** Streamlit on port 8501 (`scripts/ui/app.py`) is the primary, fully-supported interface.
> The static HTML UI at `/` (port 8000) is internal/experimental and may lag behind Streamlit features.

- **Dark DJ-booth theme** — animated equalizer bars, vinyl rotation, gradient transitions
- **Real-time progress bars** — `@st.fragment`-based polling with live ETA display
- **Quick Actions dashboard** — one-click navigation to Analyze / Mix / Search from the Home page
- **Recent Mixes panel** — last 6 rendered outputs with inline audio preview on the Home page
- **My Mixes history page** — full rendered output browser with type filter, date sort, and inline audio player
- **Playlist download** — queue multiple tracks for sequential batch download
- **Instrument Lab** — generate hybrid stems combos (vocals from A, drums from B, bass from C)

---

## Architecture

```
ai-remixmate/
├── scripts/
│   ├── api/                         # FastAPI backend
│   │   ├── main.py                  # App factory, lifespan context manager, CORS, middleware
│   │   ├── routes.py                # Thin aggregator — includes all domain routers
│   │   ├── jobs.py                  # SQLite-backed async job store + ETA + cancel/retry
│   │   ├── schemas.py               # Pydantic request/response models
│   │   │
│   │   ├── routers/                 # Domain-split route handlers (11 files)
│   │   │   ├── _helpers.py          # Shared: _require_song, _stem_file, _check_job_cap
│   │   │   ├── system.py            # /health, /health/live, /health/ready
│   │   │   ├── library.py           # /library/*, /outputs/*
│   │   │   ├── downloads.py         # /download, /download-playlist
│   │   │   ├── stems.py             # /stems/*
│   │   │   ├── analysis.py          # /analyze, /compatibility, /recommend, /library/similar, /index/*
│   │   │   ├── remix.py             # /dj-remix, /dj-remix/preview, /dj-chain, /beat/*, /instrument-lab
│   │   │   ├── generative.py        # /ai/* (style transfer, inpainting, tokenize)
│   │   │   ├── spotify.py           # /spotify/* (OAuth, import, playlist)
│   │   │   ├── jobs.py              # /jobs, /jobs/{id}, DELETE /jobs/{id}
│   │   │   └── crates.py            # /crates/*, /tags/*, /favorites/*
│   │   │
│   │   └── task_modules/            # Long-running task functions (7 files)
│   │       ├── download.py          # task_download (yt-dlp + Demucs)
│   │       ├── stems.py             # task_stem_split, task_stem_compress
│   │       ├── remix.py             # task_dj_remix, task_remix_preview, task_dj_chain
│   │       ├── analysis.py          # task_analyze, task_compatibility
│   │       ├── generative.py        # task_style_transfer, task_inpaint
│   │       ├── lab.py               # task_instrument_lab
│   │       └── __init__.py          # Re-exports (tasks.py kept as backward-compat shim)
│   │
│   ├── core/                        # Audio engine + infrastructure (30+ modules)
│   │   ├── dj_engine.py             # Transition renderer (beat-grid lock, crossfade, bridge mix)
│   │   ├── dj_analysis.py           # Analysis & planning: Beat, Section, SongStructure, TransitionPlan
│   │   ├── beat_synth.py            # Procedural drum synthesiser + Strudel code generator
│   │   ├── stems.py                 # Demucs stem separation + stem-aware crossfade mixer
│   │   ├── gpu.py                   # Centralized GPU acceleration (MPS / CUDA / CPU fallback)
│   │   ├── mastering.py             # ITU-R BS.1770-4 LUFS mastering + true-peak limiter
│   │   ├── audio_enhance.py         # Pre-Demucs enhancement pipeline (gate, compress, EQ)
│   │   ├── music_index.py           # 35-dim numpy vector index for semantic search (JSON-persisted)
│   │   ├── music_intelligence.py    # Smart query parser + recommendation routing
│   │   ├── recommend.py             # BPM-cache recommendation engine
│   │   ├── genre.py                 # Multi-feature genre classifier
│   │   ├── library.py               # Song library CRUD
│   │   ├── crates.py                # SQLite-backed crates, tags, and favorites
│   │   ├── paths.py                 # Centralised path management
│   │   ├── download.py              # yt-dlp wrapper + Demucs integration
│   │   ├── audit.py                 # Immutable JSONL audit log
│   │   ├── logging_utils.py         # Structured JSON logging + request/job ID context
│   │   ├── spotify.py               # Spotify API + OAuth client
│   │   ├── style_transfer.py        # AI style transfer between songs
│   │   ├── inpainting.py            # Audio inpainting / gap-fill
│   │   ├── instrument_lab.py        # Hybrid stem combination engine
│   │   ├── generative_remix.py      # Generative remix logic
│   │   ├── key_detection.py         # Advanced harmonic key detection
│   │   ├── musical_analysis.py      # Extended musical feature extraction
│   │   └── [+ more modules]         # features, codec_tokens, database, metrics, model_manager, etc.
│   │
│   └── ui/
│       └── app.py                   # Streamlit web interface (primary UI, port 8501)
│
├── tests/                           # pytest suite (80+ tests)
│   ├── conftest.py                  # librosa probe guard + clean_env fixture
│   ├── test_core_modules.py         # DJEngine, DJAnalysis, mastering, GPU (@pytest.mark.dj_analysis)
│   ├── test_new_features.py         # Audit, logging, ETA, rate limiting, health, RAG, schemas
│   └── e2e_test_suite.py            # End-to-end integration tests
│
├── config.yaml                      # All tuneable parameters
├── config.local.yaml                # Local overrides (gitignored)
├── pyproject.toml                   # Package metadata + dev tooling (v0.2.0)
├── requirements.txt                 # Production dependencies
├── Dockerfile                       # Canonical paths: /app/outputs, /app/models, /app/data
├── docker-compose.yml               # REMIXMATE_API_URL env var; health check → /health/live
├── start.sh                         # One-command launcher
├── run_overnight.sh                 # Overnight pipeline (full library batch processing)
├── check.sh                         # Preflight dependency check
│
├── library/                         # [runtime] Downloaded songs
├── outputs/                         # [runtime] Rendered mixes
├── data/                            # [runtime] jobs.db, remixmate.db, audit.jsonl, embeddings
└── models/                          # [runtime] ML model weights
```

---

## Quick start

### Prerequisites

- Python 3.10+
- `ffmpeg` — `brew install ffmpeg` or `apt install ffmpeg`

### Install

```bash
git clone https://github.com/Chunduri-Aditya/ai-remixmate.git
cd ai-remixmate

python -m venv remix-env
source remix-env/bin/activate     # Windows: remix-env\Scripts\activate

pip install -e ".[dev]"
```

### Run

```bash
./start.sh          # starts both API (port 8000) and UI (port 8501)
./start.sh --https  # starts with TLS (auto-generates certs)
./start.sh api      # API only
./start.sh ui       # UI only
./start.sh stop     # kill both
```

### Overnight pipeline (process entire library)

```bash
./run_overnight.sh                      # default model: htdemucs
./run_overnight.sh --model htdemucs_ft  # higher quality model
# Logs to pipeline.log — check progress: tail -f pipeline.log
```

- **Web UI** → http://localhost:8501
- **API docs** → http://localhost:8000/docs

### Docker (optional)

```bash
docker build -t ai-remixmate .
docker run -p 8000:8000 -p 8501:8501 \
  -e REMIXMATE_API_URL=http://localhost:8000 \
  -v $(pwd)/library:/app/library \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data:/app/data \
  ai-remixmate
```

---

## Usage

### Download a track

```bash
curl -X POST http://localhost:8000/download \
  -H "Content-Type: application/json" \
  -d '{"query": "Anyma Voices In My Head", "separate": false}'
```

Or use the **Download & Library** tab in the web UI. Enable **Stem Separation** to let Demucs split the track into vocals/drums/bass/other — required for stem-aware mixing.

### Smart Search (RAG)

```bash
curl "http://localhost:8000/library/similar?query=dark+progressive+techno+128bpm&limit=5"
```

Uses the 35-dimensional numpy vector index to rank library songs by semantic similarity to your natural-language query. No FAISS — pure numpy weighted cosine similarity, JSON-persisted.

### Check compatibility

```bash
curl "http://localhost:8000/compatibility?song_a=Anyma+-+Voices+In+My+Head&song_b=Dom+Dolla+-+Define"
```

Returns BPM score, key score, energy score, and Camelot wheel positions.

### Remix preview (audition before full render)

```bash
curl -X POST http://localhost:8000/dj-remix/preview \
  -H "Content-Type: application/json" \
  -d '{"song_a": "Anyma - Voices In My Head", "song_b": "Dom Dolla - Define"}'
```

Renders **only the crossfade window** — fast audition before committing to a full mix job. Returns `exit_bar_a`, `entry_bar_b`, BPM data, harmonic score, Camelot positions, tempo ratio, and a `stream_url`.

### Create a DJ mix

```bash
curl -X POST http://localhost:8000/dj-remix \
  -H "Content-Type: application/json" \
  -d '{
    "song_a": "Anyma - Voices In My Head",
    "song_b": "Dom Dolla - Define",
    "transition_bars": 16,
    "preset": "auto",
    "bridge_beat_mode": "auto",
    "bridge_beat_genre": "techno",
    "bridge_beat_intensity": 0.38,
    "use_stems": true,
    "output_format": "flac"
  }'

# Poll for result and ETA
curl http://localhost:8000/jobs/{job_id}
# → {"status": "running", "progress": 0.42, "eta_sec": 18, ...}

# Cancel if needed
curl -X DELETE http://localhost:8000/jobs/{job_id}
```

### DJ Chain (3+ songs)

```bash
curl -X POST http://localhost:8000/dj-chain \
  -H "Content-Type: application/json" \
  -d '{"songs": ["Track A", "Track B", "Track C"], "transition_bars": 16, "use_stems": false}'
```

Chains N songs into a single continuous mix, applying the same beat-grid-locked transition between every adjacent pair.

### Crates, Tags & Favorites

```bash
# Create a crate
curl -X POST http://localhost:8000/crates \
  -H "Content-Type: application/json" \
  -d '{"name": "Late Night Set"}'

# Add a song to a crate
curl -X POST http://localhost:8000/crates/{crate_id}/songs \
  -d '{"name": "Anyma - Voices In My Head"}'

# Tag a song
curl -X POST "http://localhost:8000/library/Anyma+-+Voices+In+My+Head/tags" \
  -d '{"tag": "dark"}'

# Favorite a song
curl -X POST http://localhost:8000/favorites/Anyma+-+Voices+In+My+Head
```

### Get a Strudel bridge beat

```bash
curl "http://localhost:8000/beat/synthesize?bpm=128&genre=techno"
```

Returns a rendered WAV (`audio_url`) and `strudel_code` — paste into [strudel.cc](https://strudel.cc) to hear and tweak live.

### Health check

```bash
curl http://localhost:8000/health/live   # {"status": "ok"}
curl http://localhost:8000/health/ready  # {"status": "ready", "library_songs": 12, ...}
```

---

## API reference

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | General health check |
| `GET` | `/health/live` | Liveness probe → `{"status": "ok"}` |
| `GET` | `/health/ready` | Readiness + dependency checks |

### Library

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/library` | List all songs in the library |
| `GET` | `/library/{name}` | Get metadata for a single song |
| `DELETE` | `/library/{name}` | Remove a song from the library |
| `GET` | `/library/similar` | Semantic search via 35-dim vector index |
| `POST` | `/library/init` | Initialise / re-scan library |
| `GET` | `/outputs/{session}/{filename}` | Serve a specific mix output |

### Downloads

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/download` | Download a track from YouTube Music |
| `POST` | `/download-playlist` | Download multiple tracks (batch queue) |

### Stems

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/stems/split` | Separate a song into stems via Demucs |
| `POST` | `/stems/split-batch` | Batch stem separation |
| `POST` | `/stems/compress` | Compress stem files |
| `POST` | `/stems/compress-batch` | Batch compress stems |

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Analyse BPM, genre, key, and structure |
| `GET` | `/compatibility` | Score two songs for mix compatibility |
| `GET` | `/recommend/{name}` | Top BPM-matched songs from library |
| `GET` | `/index/stats` | Vector index statistics |
| `POST` | `/index/rebuild` | Rebuild the vector similarity index |

### Remix

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/dj-remix` | Start a full DJ mix render job |
| `POST` | `/dj-remix/preview` | Audition crossfade window only (fast, no full render) |
| `POST` | `/dj-chain` | Start an N-song chain mix job |
| `GET` | `/beat/synthesize` | Synthesise a bridge beat + get Strudel code |
| `POST` | `/beat/upload` | Upload a custom bridge beat WAV |
| `POST` | `/instrument-lab` | Hybrid stem combination (vocals-A + drums-B + ...) |
| `GET` | `/instrument-lab/songs` | List songs with stems available for lab |

### Generative AI

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ai/style-transfer` | Transfer audio style between tracks |
| `POST` | `/ai/inpaint` | Fill audio gaps / inpaint segments |
| `POST` | `/ai/tokenize` | Tokenize audio for model input |
| `GET` | `/ai/models` | List available AI models |

### Spotify

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/spotify/auth-url` | Get OAuth authorization URL |
| `GET` | `/spotify/callback` | OAuth callback handler |
| `GET` | `/spotify/playlists` | List user playlists |
| `POST` | `/spotify/import` | Import a single track (creates proper job record) |
| `POST` | `/spotify/import-playlist` | Import all tracks from a playlist |

### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/jobs` | List recent jobs (newest first) |
| `GET` | `/jobs/{job_id}` | Poll job status, progress, ETA, and result |
| `DELETE` | `/jobs/{job_id}` | Cancel a pending or running job (409 if already complete) |

### Crates / Tags / Favorites

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/crates` | Create a new crate |
| `GET` | `/crates` | List all crates |
| `PATCH` | `/crates/{id}` | Rename a crate |
| `DELETE` | `/crates/{id}` | Delete a crate |
| `GET` | `/crates/{id}/songs` | List songs in a crate |
| `POST` | `/crates/{id}/songs` | Add a song to a crate |
| `DELETE` | `/crates/{id}/songs/{name}` | Remove a song from a crate |
| `GET` | `/tags` | List all tags |
| `GET` | `/tags/{tag}/songs` | List songs with a specific tag |
| `GET` | `/library/{name}/tags` | Get all tags for a song |
| `POST` | `/library/{name}/tags` | Add a tag to a song |
| `DELETE` | `/library/{name}/tags/{tag}` | Remove a tag from a song |
| `GET` | `/favorites` | List all favorited songs |
| `POST` | `/favorites/{name}` | Favorite a song |
| `DELETE` | `/favorites/{name}` | Unfavorite a song |

Full interactive docs at `/docs` (Swagger UI) or `/redoc`.

---

## How the transition works

```
         [── Song A: full volume ──────────────][─── A fades ───][─── B rises ───][── B: full ──]
                                                 ← transition window (N bars) →
                                                 Bridge beat:    ╭──────────────╮  (bell curve)
```

**Sequential handoff** — Song A holds at full for the first half of the transition window, then cosine-fades to silence. Song B starts from zero and cosine-rises to full in the second half. They're never both loud simultaneously — no +6 dB overlap distortion.

**Beat-grid lock** — The engine computes Song A's bar-grid phase at the exit cue and Song B's at the entry cue, then applies a sample-level correction (clamped to ±half-bar) to land B's downbeat on A's grid.

**Time-stretch correction** — Song B is time-stretched to match Song A's BPM via librosa's phase vocoder. The entry sample index accounts for this: `entry_sample_b = int(entry_time_b × sr / stretch_ratio)`.

**Stem-aware mode** — When `use_stems=true` and Demucs stems are available, the engine fades each stem independently. Drums and bass from Song B can be introduced earlier in the window, while vocals are delayed — mimicking real DJ mixing technique.

**Dynamic EQ fade** — A low-shelf filter progressively reduces Song A's bass frequencies while boosting Song B's across the transition, preventing the "bass clash" that makes automated mixes sound amateur.

---

## Runtime data

All runtime paths are managed by `scripts/core/paths.py`. Docker volumes and the Dockerfile use these same canonical paths.

| Directory / File | Purpose |
|------------------|---------|
| `library/` | Downloaded songs — one subdirectory per song: `full.wav`, `meta.json`, `license.json` |
| `outputs/` | Rendered mixes — one subdirectory per session ID |
| `data/jobs.db` | SQLite job store — survives restarts; in-flight jobs roll back to `FAILED` |
| `data/remixmate.db` | SQLite store for crates, tags, and favorites |
| `data/audit.jsonl` | Immutable append-only audit log |
| `models/` | ML model weights (Demucs, etc.) |

---

## Audit trail

Every significant operation is recorded in `data/audit.jsonl`:

```json
{"ts": "2025-03-22T18:45:01.234Z", "action": "download_complete", "resource": "Anyma - Voices In My Head", "job_id": "f47a...", "meta": {"duration_sec": 12.4, "format": "wav"}}
{"ts": "2025-03-22T18:46:30.891Z", "action": "dj_remix_start",    "resource": "Anyma - Voices In My Head → Dom Dolla - Define", "job_id": "9c1b...", "meta": {"transition_bars": 16, "use_stems": true}}
{"ts": "2025-03-22T18:46:55.102Z", "action": "dj_remix_complete", "resource": "Anyma - Voices In My Head → Dom Dolla - Define", "job_id": "9c1b...", "meta": {"output_path": "outputs/...", "quality_score": 0.87}}
```

Read it programmatically:

```python
from scripts.core.audit import read_audit
entries = read_audit(limit=50, action_filter="dj_remix_complete")
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Audio analysis | librosa, numpy, scipy |
| GPU acceleration | PyTorch (MPS/CUDA), torchaudio |
| Stem separation | Demucs (Meta AI) |
| Semantic search | numpy (35-dim weighted cosine similarity, JSON-persisted) |
| Download | yt-dlp, ytmusicapi |
| API | FastAPI, Uvicorn, Pydantic v2 |
| UI | Streamlit |
| Audio I/O | soundfile, pydub |
| Mastering | ITU-R BS.1770-4 LUFS, true-peak limiter |
| Persistence | SQLite (`jobs.db` for jobs, `remixmate.db` for crates/tags/favorites) |
| Spotify | spotipy / OAuth 2.0 |
| Security | TLS/HTTPS, CORS lockdown, path traversal prevention |
| Logging | Python logging + contextvars (structured JSON) |
| Testing | pytest, pytest-asyncio |
| Packaging | pyproject.toml (PEP 517), v0.2.0 |

---

## Tests

```bash
pytest tests/ -v
pytest tests/ -x              # stop on first failure
pytest tests/ --tb=short      # compact tracebacks
pytest -m "not dj_analysis"   # skip librosa-dependent tests
```

| Test file | Coverage area | Count |
|-----------|---------------|-------|
| `test_core_modules.py` | DJEngine/DJAnalysis, mastering, GPU detection (`@pytest.mark.dj_analysis`) | 59 |
| `test_new_features.py` | Audit log, structured logging, job ETA, rate limiting, health, RAG index, schemas | 25+ |
| `e2e_test_suite.py` | End-to-end integration tests | varies |

**`conftest.py`** provides: a `_probe_librosa()` guard that auto-skips `dj_analysis`-marked tests when librosa/numba can't initialize (instead of crashing), and a `clean_env` fixture that clears all `REMIXMATE_*` env vars before each test.

---

## Configuration

`config.yaml` exposes all tuneable parameters. Override locally with `config.local.yaml` (gitignored):

```yaml
library:
  path: library/
  cap_gb: 20.0

audio:
  sample_rate: 22050
  transition_bars: 16
  output_format: wav       # or "flac" for lossless compression

bridge_beat:
  default_intensity: 0.38
  default_genre: auto

api:
  max_active_jobs: 3        # Rate limit: max concurrent job submissions
  worker_threads: 2         # ThreadPoolExecutor size

stems:
  enabled: false            # Default: off (requires Demucs, slow)
  model: htdemucs           # Demucs model name
```

---

## Changelog highlights

| Phase | What changed |
|-------|--------------|
| UI: Init Library (2026-04-06) | Auto-polling stats fragment (`run_every=3`), SVG progress rings, pipeline stage stepper, animated cards |
| Hotfix (2026-04-06) | `_analyze_impl` alias added to `dj_engine.py` — fixes startup `ImportError` caused by Phase 4 rename |
| Deployment clarity | Streamlit declared primary UI; static HTML is internal/experimental; `REMIXMATE_API_URL` env var added to Docker |
| Runtime fixes | `/health/live` returns `{"status": "ok"}`; Streamlit reads `REMIXMATE_API_URL` correctly; Spotify import uses job store |
| Test stabilization | `conftest.py` librosa probe; `@pytest.mark.dj_analysis` guard; `@app.on_event` → `lifespan` context manager |
| Module splitting | `routes.py` → 11 routers in `routers/`; `tasks.py` → 7 task modules in `task_modules/`; `dj_engine.py` split into `dj_engine.py` + `dj_analysis.py` |
| Job persistence | SQLite `data/jobs.db`; `cancel_job()` + `retry_job()`; `DELETE /jobs/{id}` endpoint |
| UX additions | `POST /dj-remix/preview` for fast crossfade audition; full crates / tags / favorites system via `data/remixmate.db` |

Full history in [CHANGELOG.md](CHANGELOG.md).

---

## Disclaimer

This project is for educational and research purposes. Users are responsible for ensuring their use of downloaded audio complies with applicable copyright laws. All music rights remain with original creators and rights holders.

---

## License

MIT — see [LICENSE](LICENSE).

---

**Aditya Chunduri** · [GitHub](https://github.com/Chunduri-Aditya) · chunduri@usc.edu
