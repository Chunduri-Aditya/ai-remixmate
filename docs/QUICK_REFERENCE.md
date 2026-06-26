# AI RemixMate — Quick Reference Guide

## 🚀 Getting Started (5 minutes)

```bash
# 1. Clone & setup
git clone https://github.com/Chunduri-Aditya/ai-remixmate.git
cd ai-remixmate

# 2. Create virtual environment
python -m venv remix-env
source remix-env/bin/activate    # Windows: remix-env\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Verify setup
bash bin/check.sh

# 5. Launch (both API + UI)
./start.sh

# 6. Open in browser
# Streamlit UI: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

---

## 📁 Project Structure at a Glance

```
ai-remixmate/
├── scripts/
│   ├── api/                  # FastAPI backend
│   │   ├── main.py          # App factory
│   │   ├── routes.py        # 30+ endpoints
│   │   ├── jobs.py          # Job queue + ETA
│   │   ├── tasks.py         # Long-running work
│   │   └── schemas.py       # Pydantic models
│   │
│   ├── core/                # Audio engine
│   │   ├── dj_engine.py     # Transition planning
│   │   ├── stems.py         # Demucs separation
│   │   ├── gpu.py           # GPU detection
│   │   ├── music_index.py   # RAG vector search
│   │   ├── mastering.py     # LUFS processing
│   │   ├── audio_enhance.py # Pre-Demucs chain
│   │   ├── beat_synth.py    # Drum synthesis
│   │   ├── genre.py         # Genre detection
│   │   ├── library.py       # Library management
│   │   └── ... (8 more core modules)
│   │
│   └── ui/                  # Streamlit frontend
│       └── app.py           # 5000+ lines
│
├── tests/                   # 80+ tests
├── config.yaml             # Base config
├── pyproject.toml          # Package metadata
├── start.sh                # One-command launcher
├── run_pipeline.sh         # Demucs pipeline
└── run_overnight.sh        # Sleep-proof runner
```

---

## 🎵 Core Modules Summary

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| **dj_engine.py** | DJ transitions | `DJEngine`, `plan_transition()`, `render()` |
| **stems.py** | Demucs integration | `separate_song_stems()`, `StemResult` |
| **gpu.py** | GPU acceleration | `get_device()`, `to_tensor()`, `gpu_stft()` |
| **mastering.py** | LUFS mastering | `master_mix()`, `compute_lufs()` |
| **audio_enhance.py** | Pre-processing | `enhance_audio()`, `EnhanceOptions` |
| **beat_synth.py** | Drum synthesis | `render_beat()`, 6 genre presets |
| **music_index.py** | RAG search | `MusicIndex`, 35-dim vector space |
| **music_intelligence.py** | Feature vectors | `MusicVector`, `TransitionScore` |
| **recommend.py** | BPM matching | `get_recommendations()` |
| **genre.py** | Genre detection | `auto_preset()`, `GenrePreset` |
| **library.py** | Library CRUD | `LibraryManager`, deduplication, LRU eviction |
| **audit.py** | Audit logging | `log_audit()`, immutable JSONL |
| **logging_utils.py** | Structured logging | `get_logger()`, context vars |

---

## 🌐 API Endpoints (Quick Reference)

### Health
```
GET /health           # Liveness + readiness
GET /health/live      # Liveness only
GET /health/ready     # Readiness (dependencies)
```

### Library
```
GET /library                    # List all songs
GET /library/{name}             # Song detail
GET /library/similar?query=...  # RAG semantic search
DELETE /library/{name}          # Remove song
POST /library/initialize        # Batch process
```

### Download & Separation
```
POST /download           # Download track (auto-stems)
POST /playlist-download  # Batch download
POST /stem-split         # Single separation
POST /batch-stem-split   # Batch separation
```

### Discovery
```
POST /compatibility      # Instant BPM/key/energy check
POST /analyze           # Genre + structure analysis
GET /genres             # Supported genres
```

### DJ Mixing
```
POST /dj-remix          # Render transition
POST /dj-chain          # Advanced remix
POST /instrument-lab    # Stem editor
POST /bridge-beat       # Solo bridge beat
```

### Jobs
```
GET /jobs               # List recent jobs
GET /jobs/{job_id}      # Poll specific job
```

---

## 💻 Common Command Examples

### Download & Separate
```bash
curl -X POST http://localhost:8000/download \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Anyma - Voices In My Head",
    "separate": true
  }'
```

### Check Compatibility
```bash
curl "http://localhost:8000/compatibility?song_a=Anyma+-+Voices&song_b=Dom+Dolla+-+Define"
```

### Semantic Song Search
```bash
curl "http://localhost:8000/library/similar?query=dark+progressive+techno+128bpm&limit=5"
```

### Render DJ Mix
```bash
curl -X POST http://localhost:8000/dj-remix \
  -H "Content-Type: application/json" \
  -d '{
    "song_a": "Anyma - Voices In My Head",
    "song_b": "Dom Dolla - Define",
    "transition_bars": 16,
    "preset": "auto",
    "bridge_beat_mode": "auto",
    "output_format": "wav"
  }'
```

### Batch Stem Split
```bash
curl -X POST http://localhost:8000/batch-stem-split \
  -H "Content-Type: application/json" \
  -d '{
    "model": "htdemucs_ft",
    "enhance": true,
    "delete_wav": true
  }'
```

---

## ⚙️ Configuration

### Quick Config Override

Create `config.local.yaml`:

```yaml
# Example: faster model, smaller library cap
separation:
  model: htdemucs

library:
  max_size_gb: 100.0

logging:
  level: DEBUG
```

### Environment Variables

```bash
export REMIXMATE_DEVICE=mps              # Force GPU
export REMIXMATE_AUDIO_SAMPLE_RATE=48000 # 48 kHz
export REMIXMATE_API_PORT=9000           # Custom port
```

---

## 🎯 Workflows

### Workflow 1: Download → Mix
```
1. POST /download → get job_id
2. GET /jobs/{job_id} → poll until done
3. POST /compatibility → check BPM/key match
4. POST /dj-remix → render mix → outputs/remix_xyz.wav
```

### Workflow 2: Overnight Library Processing
```bash
./bin/run_overnight.sh --model htdemucs_ft
# Logs to pipeline.log
# Prevents sleep (macOS)
# Completion notification when done
```

### Workflow 3: Semantic Search → Smart Mix
```
1. GET /library/similar?query=dark+tech+128 → top-5 songs
2. POST /compatibility → verify BPM match
3. POST /dj-remix → render with bridge beat
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000/8501 in use | `./start.sh stop` then `./start.sh` |
| GPU not detected | `export REMIXMATE_DEVICE=cpu` |
| Demucs not found | `pip install demucs==4.0.1` |
| Memory issues | Use `htdemucs` model instead of `htdemucs_ft` |
| Slow music index | Rebuild: `POST /library/rebuild-index` |

---

## 🐳 Docker

### Build & Run
```bash
docker build -t ai-remixmate .
docker run -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/library:/app/library \
  -v $(pwd)/outputs:/app/outputs \
  ai-remixmate
```

### Docker Compose
```bash
docker compose up
# Access: http://localhost:8501
```

---

## 📊 Key Features at a Glance

### Audio Engine
- ✅ Beat-grid lock (sample-level precision)
- ✅ Stem-aware mixing (independent vocal/drum/bass fades)
- ✅ Dynamic EQ (bass frequency swap)
- ✅ Procedural beat synthesis (6 genre presets)
- ✅ ITU-R BS.1770-4 LUFS mastering
- ✅ True-peak brick-wall limiter

### Library & Search
- ✅ Demucs stem separation (4 models: htdemucs, htdemucs_ft, mdx_extra)
- ✅ FAISS RAG vector index (35-dim feature space)
- ✅ Semantic similarity search (natural language queries)
- ✅ Audio fingerprinting (SHA-256 deduplication)
- ✅ LRU eviction (smart storage management)

### API & Infrastructure
- ✅ FastAPI with async job queue
- ✅ ETA estimation (progress tracking)
- ✅ Structured JSON logging (request ID tracing)
- ✅ Immutable JSONL audit trail
- ✅ GPU auto-detection (MPS/CUDA/CPU)

### UI
- ✅ Dark DJ-booth theme
- ✅ Real-time progress bars
- ✅ Inline audio preview
- ✅ Semantic search interface
- ✅ Mobile-friendly responsive design

---

## 📦 Dependencies at a Glance

### Audio Processing
- `librosa` — Music analysis
- `demucs` — Stem separation
- `pydub`, `soundfile` — Audio I/O
- `scipy`, `numpy` — Signal processing
- `torch`, `torchaudio` — GPU acceleration

### Search & ML
- `sentence-transformers` — Semantic embeddings
- `scikit-learn` — Feature extraction

### Web
- `fastapi` — API framework
- `uvicorn` — ASGI server
- `streamlit` — UI framework

### Download
- `yt-dlp` — YouTube downloader
- `ytmusicapi` — YouTube Music search

---

## 🎛️ Genre Presets

| Genre | BPM Range | Kick | Snare | Hi-Hat | Characteristics |
|-------|-----------|------|-------|--------|-----------------|
| Techno | 120–130 | 4-on-the-floor | 2&4 | Straight 16ths | Hypnotic, industrial |
| House | 120–130 | 4-on-the-floor | 2&4 | Offbeat 8ths | Soulful, groovy |
| Hip-Hop | 85–100 | 1&3 | 2&4 | Swung 16ths | Boom-bap, breakbeats |
| Trap | 140–180 | Sparse | Snappy (3) | Dense | Aggressive, hi-energy |
| DnB | 160–180 | Fast syncopated | 2.5&4 | Dense | Liquid/dark |
| Ambient | 60–100 | None | None | Slow pad | Atmospheric, meditative |

---

## 📈 Music Vector Space (35 Dimensions)

```
[0]    BPM (normalized)
[1-12] Key one-hot (C…B)
[13]   Mode (major/minor)
[14]   Energy (mean)
[15]   Energy (std dev)
[16]   Drop position
[17]   Danceability
[18]   Beat strength
[19]   Tempo stability
[20]   Vocal density
[21]   Spectral centroid (norm)
[22]   Spectral rolloff (norm)
[23-34] Chroma features (normalized)
```

**Default Weights**:
- BPM: 40% (most important for DJ mixing)
- Key: 30% (harmonic compatibility via Camelot wheel)
- Energy: 10% (dynamics matching)
- Rhythm: 8% (groove compatibility)
- Other: 12% (vocal clash, spectral, chroma)

---

## 🔐 File Locations

| File | Purpose | Location |
|------|---------|----------|
| Songs | Downloaded tracks | `library/{song_name}/` |
| Stems | Separated audio | `library/{song_name}/{vocals,drums,bass,other}.{wav,flac}` |
| Mixes | Rendered outputs | `outputs/{session}/remix_*.{wav,flac}` |
| Index | Music vector index | `data/music_index.json` |
| Cache | BPM metadata | `library/{song_name}/meta.json` |
| Audit | Action log | `data/audit.jsonl` |
| Config | Base settings | `config.yaml` |
| Config (Local) | Overrides | `config.local.yaml` (gitignored) |

---

## 🔗 Links

- **GitHub**: https://github.com/Chunduri-Aditya/ai-remixmate
- **API Docs**: http://localhost:8000/docs (when running)
- **ReDoc**: http://localhost:8000/redoc (when running)

---

## 📋 Checklist: Pre-Production Deployment

- [ ] Install all dependencies: `pip install -r requirements.txt`
- [ ] Run validation: `bash bin/check.sh`
- [ ] Create `config.local.yaml` with production settings
- [ ] Set up HTTPS: `./start.sh --https` (generates certs)
- [ ] Configure CORS in `config.local.yaml` (tighten origins)
- [ ] Set logging level to INFO or WARNING
- [ ] Enable audit logging (already on by default)
- [ ] Test API endpoints with sample songs
- [ ] Verify job queue doesn't exceed max_active_jobs
- [ ] Set up monitoring for `/health/ready` endpoint
- [ ] Configure disk space alerts (library max_size_gb)
- [ ] Deploy via Docker Compose (production-ready)

---

## 🎓 Learning Path

**Beginner** (30 minutes):
1. Run `./start.sh`
2. Download a track via UI
3. Check compatibility
4. Render a simple mix

**Intermediate** (2 hours):
1. Explore API docs at `/docs`
2. Test endpoints via curl
3. Review `config.yaml`
4. Check job queue status

**Advanced** (1 day):
1. Deep dive into `dj_engine.py`
2. Understand music_index vector space
3. Modify genre presets
4. Run overnight library processing
5. Deploy Docker Compose

---

## 💡 Pro Tips

1. **Semantic Search**: Use natural language! "dark techno 128 bpm" finds better matches than exact song names.

2. **FLAC Compression**: Save ~60% disk space. Lossless quality preserved. Use `--no-compress` flag to skip.

3. **Bridge Beat**: Synthesized drum fills add energy during transitions. Try "auto" mode for best results.

4. **Overnight Processing**: Use `./bin/run_overnight.sh` on Mac to prevent sleep. On Linux, ensure no screensaver.

5. **Job Polling**: For long jobs, check `/jobs/{job_id}` every 5–10 seconds. Don't hammer the server.

6. **GPU Acceleration**: Apple M1/M2/M3 users get 50–100x speedup on music index searches. Set `REMIXMATE_DEVICE=mps`.

7. **Memory Pressure**: Use `htdemucs` model (4 GB) instead of `htdemucs_ft` (6 GB) on constrained systems.

8. **Audit Trail**: Every action is logged to `data/audit.jsonl`. Useful for compliance and debugging.

---

## 📞 Support

For issues:
1. Check `/health/ready` endpoint for dependency issues
2. Review logs in `logs/remixmate.log` (if file logging enabled)
3. Run `bash bin/check.sh` to validate environment
4. Check GitHub Issues: https://github.com/Chunduri-Aditya/ai-remixmate/issues

