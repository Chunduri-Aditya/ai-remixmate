# AI RemixMate — Complete Guide

> GPU-accelerated AI DJ engine with stem separation, harmonic mixing, and professional mastering.
> Built in pure Python. No external DAW required.

---

## Quick Start

```bash
# 1. Verify everything is ready
bash check.sh

# 2. Launch (starts both API + UI)
./start.sh

# 3. Open in browser
#    Local:  http://localhost:8501
#    Phone:  http://<your-lan-ip>:8501  (shown in terminal)
```

To stop everything: `./start.sh stop`

---

## Architecture Overview

AI RemixMate is split into three layers:

```
┌──────────────────────────────────────┐
│  Streamlit UI  (port 8501) PRIMARY   │   ← Supported user interface
│  Static HTML   (port 8000 at /)      │   ← Internal/experimental only
├──────────────────────────────────────┤
│  FastAPI Backend (port 8000)         │   ← REST API, job queue, async tasks
├──────────────────────────────────────┤
│  Core Engine (30+ modules)           │   ← Audio processing, ML, GPU ops
└──────────────────────────────────────┘
```

**Frontend model:** The Streamlit UI (`scripts/ui/app.py`, port 8501) is the **primary supported interface**. The static HTML UI served by FastAPI at `http://localhost:8000/` is an internal/experimental frontend — it may lag behind the Streamlit feature set and is not the documented run path.

The API runs long tasks (stem splitting, remixes, downloads) as background jobs. You can poll job status from the UI or via `GET /jobs/{job_id}`. Up to 4 concurrent jobs run at once.

### Runtime directories

All runtime paths follow `scripts/core/paths.py`:

| Directory | Purpose |
|-----------|---------|
| `library/` | Downloaded songs (one subdirectory per song) |
| `outputs/` | Rendered mixes (one subdirectory per session ID) |
| `data/` | SQLite DB, embeddings, audit log (`data/audit.jsonl`) |
| `models/` | ML model weights |

Docker volumes and the Dockerfile use these same canonical paths.

---

## UI Pages

### 🏠 Home
Dashboard with quick-action cards. Jump straight into any workflow — download, remix, analyze. Shows library stats (total songs, songs with stems, disk usage) and API health status.

### 📚 Library
Browse all downloaded songs. Each song card shows name, file size, available stems, and license type. Paginated (25/50/100 per page) with search. Click any song to see details, play audio, or trigger stem separation.

### ⬇️ Download
Download tracks from YouTube Music. Paste a URL or type a search query.

- **Single track**: Search by name or paste a YouTube/YouTube Music URL. Optionally auto-split stems after download.
- **Playlist**: Paste a playlist URL to download all tracks (limit up to 200). Each track shows individual progress.

Downloaded songs land in `library/<Artist - Title>/` with `full.wav`, `meta.json`, and `license.json`.

### 🔗 Compatibility
Check if two songs mix well together *before* downloading. Enter artist + title for both songs and get an instant compatibility score based on:

- **BPM match** — How close their tempos are (±6% is ideal)
- **Key compatibility** — Camelot wheel harmony (same key, adjacent keys, relative major/minor)
- **Energy match** — Vibe alignment

No downloads needed — uses metadata APIs (GetSongBPM, Last.fm, MusicBrainz) with local librosa fallback.

### 🎛️ DJ Remix
Create a DJ-style transition between two songs from your library.

**Parameters you control:**
- **Transition length**: 8, 16, 32, or 64 bars
- **Genre preset**: techno, house, hip-hop, trap, dnb, ambient, or auto-detect
- **Bridge beat**: none, auto-synthesized, or upload your own WAV
- **Transition effect**: auto, echo, filter, reverb, or none

The engine handles phrase detection, beat alignment, EQ crossfading, and harmonic matching automatically. Output renders as a WAV file you can download or play in-browser.

### 🎚️ DJ Chain
Build a continuous multi-song DJ mix (2–8 songs). Same controls as DJ Remix, but chained — the engine plans transitions between each consecutive pair, locking phrases and managing energy flow across the full set.

### 🧪 Instrument Lab
The wildest feature. Pick 2+ songs that have stems and generate hybrid combinations — vocals from Song A over drums from Song B with bass from Song C. The engine:

- Time-stretches all stems to a common BPM
- Normalizes gain across stems
- Generates either "targeted" (single-stem swaps) or "all" (full permutation) combinations
- Renders each combination as a playable WAV

### 🔍 Analyze
Deep-dive into any library song's musical DNA:

- **Genre detection** — rule-based classifier across 10 genres (techno, house, hip-hop, trap, dnb, ambient, pop, rock, jazz, classical)
- **Structure analysis** — section breakdown (intro, verse, chorus, drop, break, outro) with timestamps
- **Musical features** — BPM, key, Camelot code, energy curve, danceability, vocal density, spectral characteristics

### 🔮 Smart Search
Find songs in your library using the 35-dimensional RAG vector index. Search by similarity to a reference song — the index weights BPM (40%), key/mode (35%), energy (10%), rhythm complexity (8%), and vocal/timbre character (7%).

### 🗜️ Compress Library
Convert all WAV stems to lossless FLAC (~50% space savings, identical quality). Runs as a batch background job. Can optionally delete the original WAVs after compression.

### 🚀 Initialize Library
One-shot pipeline for setting up or refreshing the entire library:
1. Stem-split all songs that don't have stems yet (Demucs)
2. Compress WAV stems to FLAC
3. Rebuild the RAG vector index

### 🎨 Visualizer
Audio visualization and waveform inspection for library tracks.

### 🎵 My Mixes
Browse previously rendered remixes and DJ chains, organized by session. Play, download, or delete old mixes.

---

## API Reference

All endpoints are documented at `http://localhost:8000/docs` (Swagger UI) when the API is running.

### System
| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check — returns library size + running job count |
| `GET /health/live` | Lightweight liveness probe |
| `GET /health/ready` | Readiness probe — checks thread pool + library access |

### Library Management
| Endpoint | Description |
|---|---|
| `GET /library` | List songs with pagination, search, and stats |
| `GET /library/names` | All song names as a flat array (lightweight) |
| `GET /library/{name}` | Single song metadata (size, stems, license) |
| `GET /library/{name}/audio` | Stream the full mix WAV |
| `GET /library/{name}/stems/{stem}` | Stream a single stem (vocals/drums/bass/other) — serves FLAC or WAV |
| `DELETE /library/{name}` | Remove a song from the library |
| `POST /library/initialize` | Full pipeline: stem-split → compress → index |

### Downloads
| Endpoint | Description |
|---|---|
| `POST /download` | Download a single track (URL or search query) |
| `POST /download-playlist` | Download an entire playlist (up to 200 tracks) |

### Analysis
| Endpoint | Description |
|---|---|
| `POST /compatibility` | Instant BPM + key + energy compatibility check |
| `POST /analyze` | Genre + structure analysis (queued job) |
| `GET /recommend/{name}` | Top compatible songs for a given track |
| `GET /library/similar/{name}` | Similar songs via 35-dim vector index |

### Stem Processing
| Endpoint | Description |
|---|---|
| `POST /stems/split` | Demucs stem separation for one song |
| `POST /stems/split-batch` | Batch stem separation (entire library or list) |
| `POST /stems/compress` | Compress one song's stems to FLAC |
| `POST /stems/compress-batch` | Batch compress all stems to FLAC |

### Remix & Mixing
| Endpoint | Description |
|---|---|
| `POST /dj-remix` | DJ transition between two songs |
| `POST /dj-chain` | Multi-song continuous DJ mix (2–8 songs) |
| `POST /instrument-lab` | Stem-swap experiments |
| `GET /instrument-lab/songs` | Songs eligible for Instrument Lab |

### Beat Synthesis
| Endpoint | Description |
|---|---|
| `GET /beat/synthesize` | Generate a procedural drum loop (by BPM, genre, bars, intensity) |
| `POST /beat/upload` | Upload a custom beat WAV for use as bridge beat |

### Outputs
| Endpoint | Description |
|---|---|
| `GET /outputs/{session_id}/{filename}` | Stream a rendered mix |

### Vector Index
| Endpoint | Description |
|---|---|
| `GET /index/stats` | RAG index statistics |
| `POST /index/rebuild` | Rebuild the vector index |

### Job Management
| Endpoint | Description |
|---|---|
| `GET /jobs` | List recent jobs (default 20, max 100) |
| `GET /jobs/{job_id}` | Poll job status, progress, ETA, result |

**Job lifecycle**: `pending` → `running` → `done` or `failed`

---

## Core Engine Modules

### Audio Pipeline
| Module | What it does |
|---|---|
| `gpu.py` | Centralized GPU detection — MPS (Apple Silicon), CUDA (NVIDIA), CPU fallback |
| `stems.py` | Demucs stem separation (vocals, drums, bass, other) |
| `audio_enhance.py` | Pre-Demucs enhancement — DC removal, noise gate, compression, LUFS normalization |
| `mastering.py` | Final mastering — ITU-R BS.1770-4 LUFS, true-peak limiter, clipping detection |
| `pro_audio_chain.py` | Professional processing chain — sidechain compression, reverb, 24-bit export |

### Music Intelligence
| Module | What it does |
|---|---|
| `music_intelligence.py` | Per-track feature vectors: BPM, key, Camelot, energy, danceability, vocal density |
| `musical_analysis.py` | Key detection, beat grid alignment, pitch shifting, Camelot compatibility |
| `genre.py` | Rule-based genre classifier across 10 genres |
| `features.py` | Librosa feature extraction (tempo, chroma, MFCC) |
| `music_index.py` | 35-dim RAG vector index for sub-millisecond similarity search |
| `recommend.py` | Fast BPM-based song recommendations |

### DJ Engine
| Module | What it does |
|---|---|
| `dj_engine.py` | Phrase detection, section analysis, transition planning, EQ curve rendering |
| `python_mixer.py` | Pure Python audio mixing with loudness measurement |
| `beat_synth.py` | Procedural drum synthesis (kick, snare, hi-hat) across 6 genres |
| `instrument_lab.py` | Stem swapping and combination across songs |

### Infrastructure
| Module | What it does |
|---|---|
| `library.py` | Library management — inventory, deduplication (SHA-256), pruning, LRU eviction |
| `database.py` | SQLite store for metadata, feature vectors, embeddings, lyrics |
| `track_metadata.py` | Metadata lookup from GetSongBPM, Last.fm, MusicBrainz + local fallback |
| `license.py` | License tracking for all downloaded songs |
| `audit.py` | Immutable JSONL audit log for all operations |
| `config.py` | Configuration loader (env vars, YAML, defaults) |
| `paths.py` | Canonical path layout for library, outputs, data, models |
| `logging_utils.py` | Structured JSON logging with request ID tracing |

### Optimization
| Module | What it does |
|---|---|
| `real_optimizer.py` | Random search optimizer with constraint repair for mix parameters |
| `optimization_proxy.py` | ML prediction + constraint enforcement for remix parameters |
| `advanced_remix.py` | Advanced remix engine with quality certification |
| `metrics.py` | Objective quality metrics — tempo error, key match, LUFS, clipping, SI-SDR |

---

## File Structure

```
ai-remixmate/
├── start.sh                    # Launch script (API + UI)
├── check.sh                    # Readiness checker
├── Dockerfile                  # Docker deployment
├── requirements.txt            # Python dependencies
│
├── scripts/
│   ├── api/                    # FastAPI backend
│   │   ├── main.py             #   App factory + CORS
│   │   ├── routes.py           #   All 30+ API endpoints
│   │   ├── schemas.py          #   Pydantic request/response models
│   │   ├── tasks.py            #   Background job implementations
│   │   └── jobs.py             #   Job store + status tracking
│   │
│   ├── core/                   # Engine (30 modules)
│   │   ├── gpu.py              #   GPU detection + acceleration
│   │   ├── dj_engine.py        #   DJ transition engine
│   │   ├── stems.py            #   Demucs stem separation
│   │   ├── mastering.py        #   Final mastering chain
│   │   ├── music_index.py      #   RAG vector similarity index
│   │   └── ...                 #   (25 more modules)
│   │
│   ├── ui/
│   │   └── app.py              # Streamlit frontend (4200+ lines)
│   │
│   ├── download.py             # YouTube Music downloader
│   ├── remixmate_cli.py        # CLI interface
│   └── legacy/                 # Older modules (kept for reference)
│
├── library/                    # Downloaded songs
│   └── Artist - Title/
│       ├── full.wav            #   Full mix
│       ├── vocals.flac         #   Vocal stem
│       ├── drums.flac          #   Drum stem
│       ├── bass.flac           #   Bass stem
│       ├── other.flac          #   Other/synth stem
│       ├── meta.json           #   BPM, key, genre metadata
│       └── license.json        #   Source + license type
│
├── outputs/                    # Rendered mixes (by session ID)
├── data/                       # SQLite DB, embeddings, audit logs
├── models/                     # ML model checkpoints
├── certs/                      # TLS certificates (optional HTTPS)
├── tests/                      # Unit + integration + e2e tests
└── .streamlit/config.toml      # Streamlit theme + config
```

---

## GPU Acceleration

The engine auto-detects your GPU on startup:

| Hardware | Backend | What gets accelerated |
|---|---|---|
| Apple Silicon (M1/M2/M3/M4) | MPS (Metal) | STFT, cosine similarity, time stretching, resampling, Demucs inference |
| NVIDIA GPU | CUDA | Same as above |
| No GPU | CPU | Everything still works, just slower for Demucs (~2x) |

Override with: `REMIXMATE_DEVICE=cpu ./start.sh`

---

## Security

- **Path traversal protection** — All library paths sanitized with `Path.resolve()` + containment check
- **CORS locked** — API only accepts requests from `localhost:8501` origins
- **XSRF protection** — Enabled in Streamlit config
- **No shell injection** — All subprocess calls use list args (no `shell=True`)
- **File upload validation** — 100MB limit + audio format whitelist
- **Audit logging** — Immutable JSONL log of all operations

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REMIXMATE_DEVICE` | auto | Force GPU backend: `mps`, `cuda`, or `cpu` |
| `REMIXMATE_HTTPS` | (unset) | Set to `1` or `true` to force HTTPS mode |
| `REMIXMATE_LIBRARY_CAP_GB` | 50 | Max library size in GB |
| `JAMENDO_CLIENT_ID` | (unset) | Jamendo API key for Creative Commons tracks |

### start.sh Options

```bash
./start.sh              # Start API + UI (HTTP)
./start.sh --https      # Start with HTTPS (auto-generates certs)
./start.sh api          # API only
./start.sh ui           # UI only
./start.sh stop         # Kill all processes
./start.sh --no-https   # Force HTTP even if certs exist
```

---

## Typical Workflows

### Building a DJ Set
1. **Download** → Download 5–10 tracks from YouTube Music
2. **Analyze** → Check compatibility between pairs
3. **Stem Split** → Run Demucs on all tracks (batch)
4. **Compress** → Convert stems to FLAC to save space
5. **DJ Chain** → Build a continuous mix with automatic transitions
6. **Download mix** → Export the rendered WAV

### Creative Stem Mashups
1. **Download** two songs you want to mashup
2. **Stem Split** both tracks
3. **Instrument Lab** → Generate combinations (vocals from A + drums from B, etc.)
4. **Listen** → Browse and play all generated combinations
5. **Export** the ones you like

### Quick Compatibility Check
1. Go to **Compatibility** page
2. Enter artist + title for two songs (no download needed)
3. Get instant BPM, key, and energy compatibility scores
4. If compatible → download both → remix

---

## Troubleshooting

**"Cannot reach API at localhost:8000"**
→ The API server isn't running. Run `./start.sh` (starts both API and UI together).

**Stems showing as empty / not found**
→ Run stem separation first: go to the song in Library → click "Split Stems". Or use the batch endpoint.

**Demucs is slow**
→ Expected: 2–5 minutes per song on CPU, ~1 minute on GPU. Make sure GPU is detected (`bash check.sh` shows GPU status).

**Port already in use**
→ Run `./start.sh stop` first, then `./start.sh` again.

**Browser shows HTTPS warning**
→ Self-signed cert issue. Use HTTP mode instead: `./start.sh --no-https`
