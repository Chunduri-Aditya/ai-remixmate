# AI RemixMate Documentation Index

This directory contains comprehensive documentation for the AI RemixMate project — a full-stack Python audio engineering application.

---

## 📚 Documentation Files

### 1. **AI_REMIXMATE_COMPREHENSIVE_DOCUMENTATION.md** (2178 lines)
**The complete technical reference**

Contents:
- Executive summary
- Project structure
- Core audio modules (13 modules × 250–2000 lines each)
- API modules (FastAPI backend)
- UI module (Streamlit)
- Shell scripts
- Configuration system
- Dependencies
- Docker setup
- Testing suite
- Public API endpoints (30+)
- Key workflows
- Performance tuning
- Troubleshooting
- Deployment guides
- Legal notes
- File reference index
- Quick reference section

**Use this when**: You need deep technical details, function signatures, class definitions, or how components interact.

---

### 2. **REMIXMATE_QUICK_REFERENCE.md** (250 lines)
**Fast lookup guide for common tasks**

Contents:
- 5-minute getting started
- Project structure overview
- Core modules summary table
- API endpoints quick reference
- Common command examples (curl)
- Configuration quick start
- Common workflows (3 main workflows)
- Troubleshooting table
- Docker quick commands
- Key features summary
- Genre presets table
- Music vector space explanation
- File locations reference
- Learning path (beginner → advanced)
- Pro tips
- Support links

**Use this when**: You need quick answers, want to run a command, or just need a reminder about endpoints.

---

### 3. **DOCUMENTATION_INDEX.md** (this file)
**Navigation guide and project overview**

---

## 🗂️ How to Use This Documentation

### By Role

#### **Software Engineer / Developer**
1. Start: `REMIXMATE_QUICK_REFERENCE.md` → "Getting Started" section
2. Reference: `AI_REMIXMATE_COMPREHENSIVE_DOCUMENTATION.md` → relevant section
3. Deep Dive: Project source files with module docstrings
4. Test: `tests/` directory for usage examples

#### **DevOps / Deployment**
1. Start: `REMIXMATE_QUICK_REFERENCE.md` → "Docker" section
2. Reference: `AI_REMIXMATE_COMPREHENSIVE_DOCUMENTATION.md` → "Deployment" section
3. Config: `config.yaml` in project root
4. Check: `check.sh` script for validation

#### **Audio Engineer / Music Producer**
1. Start: `REMIXMATE_QUICK_REFERENCE.md` → "5-minute getting started"
2. Reference: `REMIXMATE_QUICK_REFERENCE.md` → "Common workflows"
3. Explore: Web UI at http://localhost:8501 (user-friendly)
4. Customize: Genre presets in `scripts/core/genre.py`

#### **Architect / System Designer**
1. Reference: `AI_REMIXMATE_COMPREHENSIVE_DOCUMENTATION.md` → "Project Structure"
2. Deep Dive: Architecture section + module descriptions
3. API: `AI_REMIXMATE_COMPREHENSIVE_DOCUMENTATION.md` → "Public API Endpoints"
4. Extend: See module examples for extension patterns

---

## 🎯 Common Questions (Quick Answers)

### How do I start the application?
See: `REMIXMATE_QUICK_REFERENCE.md` → "Getting Started"
Command: `./start.sh`

### How do I download and separate a song?
See: `REMIXMATE_QUICK_REFERENCE.md` → "Common Command Examples"
Or: API endpoint `POST /download` in comprehensive docs

### How do I render a DJ mix?
See: `REMIXMATE_QUICK_REFERENCE.md` → "Workflows" → "Workflow 1"
API: `POST /dj-remix` endpoint

### How do I process my entire library overnight?
See: `REMIXMATE_QUICK_REFERENCE.md` → "Getting Started"
Command: `./run_overnight.sh`

### What's the difference between different Demucs models?
See: `AI_REMIXMATE_COMPREHENSIVE_DOCUMENTATION.md` → "Performance Tuning" → "Stem Separation Models"

### How does the music search work?
See: `AI_REMIXMATE_COMPREHENSIVE_DOCUMENTATION.md` → "music_index.py" section
Brief: Uses 35-dimensional vector space with FAISS for semantic similarity

### Can I customize the beat synthesis?
See: `AI_REMIXMATE_COMPREHENSIVE_DOCUMENTATION.md` → "beat_synth.py" section
Also: `scripts/core/beat_synth.py` with 6 genre presets

### How do I deploy to production?
See: `AI_REMIXMATE_COMPREHENSIVE_DOCUMENTATION.md` → "Deployment" section
Options: Docker, Docker Compose, or Systemd service

### What GPU devices are supported?
See: `AI_REMIXMATE_COMPREHENSIVE_DOCUMENTATION.md` → "gpu.py" section
Supported: Apple Silicon MPS, NVIDIA CUDA, CPU fallback

### How do I check if everything is installed correctly?
Command: `bash check.sh`
This validates Python, dependencies, system tools, GPU, project structure, config, library, ports, and syntax.

---

## 📊 Project Statistics

| Metric | Count |
|--------|-------|
| Total Lines of Code (core) | ~30,000 |
| Core Modules | 13 |
| API Endpoints | 30+ |
| Test Cases | 80+ |
| Configuration Sections | 10 |
| Supported Genres | 10 |
| Music Vector Dimensions | 35 |
| Dependencies | 40+ |
| Python Files | 50+ |

---

## 🔑 Key Architecture Points

### Three-Layer Architecture

```
┌─────────────────────────────────────┐
│  Streamlit UI (scripts/ui/app.py)   │  User-facing web interface
├─────────────────────────────────────┤
│  FastAPI Backend (scripts/api/)     │  REST API, job queue, routing
│  - main.py, routes.py, jobs.py,     │
│  - tasks.py, schemas.py             │
├─────────────────────────────────────┤
│  Audio Engine (scripts/core/)       │  Audio processing algorithms
│  - 13 specialized modules            │
│  - DJ transitions, stem separation   │
│  - Mastering, GPU acceleration       │
└─────────────────────────────────────┘
```

### Data Flow (Download → Analyze → Mix)

```
User Input → FastAPI Route → Job Queue → Task Function → Audio Engine
   ↓              ↓              ↓             ↓              ↓
Browser       /download    ThreadPoolExecutor  task_download  Demucs
(Streamlit)    handler       (2 workers)       (job thread)   separation
                                                              ↓
                                                    library/song/
                                                    {vocals,drums,
                                                     bass,other}
```

### Music Matching Algorithm

```
Song A → Feature Vector (35 dims) ──┐
                                      ├→ Cosine Similarity (FAISS) → Top-K Results
Song B → Feature Vector (35 dims) ──┘

Weighted Similarity = Σ(weight_i × similarity_i)
  - BPM (40%): Primary tempo constraint
  - Key (30%): Camelot wheel harmonic compatibility
  - Energy (10%): Dynamics continuity
  - Other (20%): Rhythm, spectral, vocal density
```

---

## 🔍 Module Dependency Graph

```
┌────────────────────────────────────────────────────────┐
│                   FastAPI Routes                        │
│                  (30+ endpoints)                        │
└──────────┬──────────────────────────────────────────────┘
           │
           ├─ Job Queue (jobs.py)
           │  └─ Task Functions (tasks.py)
           │     ├─ task_download
           │     ├─ task_stem_split
           │     ├─ task_dj_remix
           │     └─ task_analyze
           │        └─ Core Audio Engine
           │           ├─ dj_engine.py ─────┐
           │           ├─ stems.py          │
           │           ├─ music_index.py    ├─ GPU acceleration (gpu.py)
           │           ├─ music_intelligence.py
           │           ├─ mastering.py      │
           │           ├─ audio_enhance.py  │
           │           ├─ beat_synth.py ────┘
           │           ├─ genre.py
           │           ├─ recommend.py
           │           ├─ library.py
           │           └─ audio I/O (librosa, soundfile)
           │
           └─ Infrastructure
              ├─ logging_utils.py (structured JSON logs)
              ├─ audit.py (immutable audit trail)
              ├─ config.py (centralized config)
              └─ paths.py (path management)

Streamlit UI (app.py)
│
├─ requests → FastAPI routes (via HTTP)
├─ Job polling (GET /jobs/{id})
└─ File serving (outputs, audio preview)
```

---

## 📋 Module Purpose Quick Reference

### Audio Processing (Core)
| Module | What It Does |
|--------|-------------|
| dj_engine.py | Plans DJ transitions, renders beats to grids |
| stems.py | Calls Demucs, handles separation output |
| gpu.py | Detects GPU, provides acceleration |
| mastering.py | LUFS measurement, gain normalization, limiting |
| audio_enhance.py | Pre-processing (gate, compress, EQ, normalize) |
| beat_synth.py | Generates drums from scratch (no samples) |

### Search & Discovery (Core)
| Module | What It Does |
|--------|-------------|
| music_index.py | Builds 35-dim vector space, FAISS search |
| music_intelligence.py | Computes features per track, transition scores |
| recommend.py | BPM-based matching with caching |
| genre.py | Detects genre, returns preset parameters |

### Library & Storage (Core)
| Module | What It Does |
|--------|-------------|
| library.py | Song CRUD, deduplication, LRU eviction |
| download.py | yt-dlp wrapper, track fetching |
| audit.py | Immutable JSONL logging |

### Web Backend (API)
| Module | What It Does |
|--------|-------------|
| main.py | FastAPI app factory, middleware, startup |
| routes.py | HTTP endpoint definitions (30+) |
| jobs.py | Job queue, ETA estimation, threading |
| tasks.py | Actual work (download, remix, separate) |
| schemas.py | Pydantic request/response models |

### Web Frontend (UI)
| Module | What It Does |
|--------|-------------|
| app.py | Streamlit interface, pages, interactions |

### Infrastructure
| Module | What It Does |
|--------|-------------|
| config.py | Config loading, typing, validation |
| logging_utils.py | Structured JSON logging, context vars |
| paths.py | Centralized path management |

---

## 🔗 Cross-References

### If you want to understand...

**How beat-locking works**:
- `dj_engine.py` → `analyze_structure()` → `Beat` dataclass
- See: Comprehensive docs → "dj_engine.py" section

**How semantic search works**:
- `music_index.py` → Vector layout (35 dims)
- `music_intelligence.py` → Feature computation
- See: Comprehensive docs → "music_index.py" and "music_intelligence.py"

**How GPU acceleration works**:
- `gpu.py` → `get_device()` function
- All modules import from gpu.py for tensor operations
- See: Comprehensive docs → "gpu.py" section

**How jobs are queued**:
- `jobs.py` → `create_job()`, `update_job()`, `run_job()`
- `tasks.py` → Individual task functions
- See: Comprehensive docs → "jobs.py" and "tasks.py"

**How audio is enhanced**:
- `audio_enhance.py` → 7-stage processing pipeline
- Called from `stems.py` before Demucs
- See: Comprehensive docs → "audio_enhance.py" section

**How beats are synthesized**:
- `beat_synth.py` → Instruments (`_kick`, `_snare`, `_hihat`, etc.)
- 6 genre presets (techno, house, hiphop, trap, dnb, ambient)
- See: Comprehensive docs → "beat_synth.py" section

---

## 🚀 Getting Started Paths

### Path 1: Just Want to Use It (Non-Technical)
1. Read: `REMIXMATE_QUICK_REFERENCE.md` → "Getting Started"
2. Run: `./start.sh`
3. Open: http://localhost:8501
4. Use: Web UI (intuitive interface)

### Path 2: Want to Understand the Architecture
1. Read: Comprehensive docs → "Project Structure"
2. Read: Comprehensive docs → "Core Audio Modules" (skim all 13)
3. Read: Comprehensive docs → "API Modules"
4. Explore: Source code (docstrings are helpful)

### Path 3: Want to Deploy to Production
1. Read: `REMIXMATE_QUICK_REFERENCE.md` → "Docker"
2. Read: Comprehensive docs → "Deployment" section
3. Create: `config.local.yaml` with production settings
4. Deploy: Via Docker Compose or Systemd

### Path 4: Want to Extend or Modify
1. Read: Comprehensive docs → entire document
2. Study: Source files with attention to:
   - Function signatures (inputs/outputs)
   - Class definitions (data structures)
   - Docstrings (purpose and behavior)
3. Check: `tests/` for usage examples
4. Modify: Start with one module (e.g., `genre.py`)

### Path 5: Want to Understand Music Theory Aspects
1. Read: `REMIXMATE_QUICK_REFERENCE.md` → "Genre Presets"
2. Read: `REMIXMATE_QUICK_REFERENCE.md` → "Music Vector Space"
3. Read: Comprehensive docs → "music_intelligence.py" (Camelot wheel, Krumhansl)
4. Read: Comprehensive docs → "dj_engine.py" (phrase detection, transitions)

---

## 🆘 Troubleshooting Navigation

| Problem | Where to Look |
|---------|---------------|
| Won't start | `check.sh` script, then comprehensive docs → "Troubleshooting" |
| Wrong audio output | Comprehensive docs → "mastering.py" or "audio_enhance.py" |
| Slow music search | Comprehensive docs → "Performance Tuning" → "Music Index" |
| GPU not working | Quick reference → "Troubleshooting" or comprehensive docs → "gpu.py" |
| Download fails | Check yt-dlp availability, comprehensive docs → "download.py" |
| Memory issues | Comprehensive docs → "Performance Tuning" → "Stem Separation Models" |
| API errors | Comprehensive docs → "Public API Endpoints" → validate request format |
| Stems bad quality | Comprehensive docs → "stems.py" → try `htdemucs_ft` model |

---

## 📞 Support & Resources

### In-Code Resources
- **Docstrings**: Every module has detailed docstrings explaining purpose and usage
- **Type Hints**: All functions use type hints for clarity
- **Config**: `config.yaml` with comments explaining each setting
- **Tests**: `tests/` directory with 80+ usage examples

### External Resources
- **GitHub**: https://github.com/Chunduri-Aditya/ai-remixmate
- **API Docs**: http://localhost:8000/docs (when running)
- **Dependencies**: See `requirements.txt` and `pyproject.toml`

### Validation Tools
- **Readiness Check**: `bash check.sh` validates entire environment
- **Health Endpoints**: `/health`, `/health/live`, `/health/ready`
- **API Testing**: `curl` examples in quick reference

---

## 📝 Documentation Maintenance

These documentation files were auto-generated from the project source code as of **March 24, 2026**.

To keep documentation up-to-date:
1. Update docstrings in source files
2. Update `config.yaml` with new options
3. Regenerate comprehensive docs by reviewing source
4. Update quick reference with new endpoints

---

## 🎓 Learning Resources

### Concepts to Understand

**Audio & Signal Processing**:
- ITU-R BS.1770-4 (LUFS standard) → See mastering.py
- STFT (Short-Time Fourier Transform) → See librosa usage
- Beat Detection → See dj_engine.py
- Stem Separation → See stems.py

**Music Theory**:
- Camelot Wheel (harmonic mixing) → See music_intelligence.py
- Key Detection → Krumhansl-Schmuckler profiles in music_intelligence.py
- BPM/Tempo → See recommend.py

**Web Architecture**:
- FastAPI basics → See main.py, routes.py
- Async job queues → See jobs.py
- WebSocket streaming (future) → Already prepared in routes.py

**Vector Search**:
- FAISS indexing → See music_index.py
- Embedding spaces → 35-dim vector layout in music_index.py
- Cosine similarity → Weighted scoring in music_intelligence.py

---

## ✅ Documentation Checklist

This documentation package includes:

- ✅ Executive summary
- ✅ Complete architecture overview
- ✅ 13 core audio module descriptions
- ✅ 5 API module descriptions
- ✅ 1 UI module description
- ✅ Function signatures with docstrings
- ✅ Class definitions with fields
- ✅ Configuration system explanation
- ✅ 30+ API endpoints documented
- ✅ 3 complete workflows
- ✅ Troubleshooting guide
- ✅ Deployment instructions
- ✅ Docker setup
- ✅ Testing overview
- ✅ Performance tuning tips
- ✅ Quick reference guide
- ✅ Command examples
- ✅ File location reference
- ✅ Learning paths by role
- ✅ Pro tips section
- ✅ Index/navigation document

---

## 🎉 Conclusion

This documentation provides everything needed to:
- **Use** the application (UI)
- **Integrate** the API (developers)
- **Deploy** to production (DevOps)
- **Extend** with new features (engineers)
- **Understand** the architecture (architects)

Start with the appropriate section for your role, use the quick reference for common tasks, and dive into comprehensive documentation when you need deep details.

Happy remixing! 🎵

