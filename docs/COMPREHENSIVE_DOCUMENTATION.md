# AI RemixMate — Comprehensive Technical Documentation

## Executive Summary

AI RemixMate is a full-stack Python audio engineering project that renders seamless DJ transitions between two songs. It combines:
- **Audio Engine**: Beat-grid locking, stem-aware mixing, dynamic EQ, procedural beat synthesis
- **Library Management**: Song download, Demucs stem separation, FAISS RAG vector indexing
- **API Backend**: FastAPI with async job queue, structured logging, immutable audit trails
- **Web UI**: Streamlit dark DJ-booth interface with real-time progress tracking
- **GPU Acceleration**: Auto-detects Apple Silicon MPS, NVIDIA CUDA, CPU fallback

Version: 0.2.0 | Python: 3.10+ | License: MIT

---

## Project Structure

```
ai-remixmate/
├── scripts/
│   ├── api/                          # FastAPI backend
│   │   ├── main.py                   # App factory, CORS, middleware, startup/shutdown
│   │   ├── routes.py                 # HTTP endpoints (30+ endpoints)
│   │   ├── jobs.py                   # In-memory async job queue, ETA estimation
│   │   ├── tasks.py                  # Long-running task functions
│   │   └── schemas.py                # Pydantic request/response models
│   │
│   ├── core/                         # Audio engine + infrastructure
│   │   ├── dj_engine.py              # Transition planner + renderer (2075 lines)
│   │   ├── beat_synth.py             # Procedural drum synthesizer
│   │   ├── stems.py                  # Demucs integration
│   │   ├── gpu.py                    # Device detection (MPS/CUDA/CPU)
│   │   ├── mastering.py              # ITU-R BS.1770-4 LUFS mastering
│   │   ├── audio_enhance.py          # Pre-Demucs enhancement pipeline
│   │   ├── music_index.py            # FAISS RAG vector index
│   │   ├── music_intelligence.py     # Feature vector computation, transition scoring
│   │   ├── recommend.py              # BPM-cache recommendation engine
│   │   ├── genre.py                  # Multi-feature genre classifier
│   │   ├── library.py                # Song library CRUD + deduplication
│   │   ├── audit.py                  # Immutable JSONL audit log
│   │   ├── logging_utils.py          # Structured JSON logging
│   │   ├── config.py                 # Centralized config (typed dataclasses)
│   │   ├── paths.py                  # Path management
│   │   └── download.py               # yt-dlp wrapper
│   │
│   ├── ui/                           # Streamlit frontend
│   │   ├── app.py                    # Main Streamlit app (5000+ lines)
│   │   └── static/                   # Static assets (CSS, JS)
│   │
│   └── legacy/                       # Archived prototype scripts
│
├── tests/                            # pytest suite (80+ tests)
│   ├── test_new_features.py          # Unit/integration tests for v2 modules
│   └── test_core_modules.py          # Smoke tests for core audio modules
│
├── docs/                             # Documentation
│   ├── PORTFOLIO_INTEGRATION.md      # Portfolio setup guide
│   ├── portfolio-card.html           # Embeddable portfolio component
│   └── portfolio-card.jsx            # React portfolio component
│
├── .github/                          # GitHub configuration
│   ├── FUNDING.yml                   # Sponsorship info
│   └── ISSUE_TEMPLATE/               # Issue templates
│
├── .streamlit/                       # Streamlit configuration
│   └── config.toml                   # Theme, server, browser settings
│
├── config.yaml                       # Base configuration (override via config.local.yaml)
├── pyproject.toml                    # Package metadata, dev tools, build config
├── requirements.txt                  # Production dependencies
├── Dockerfile                        # Multi-stage Docker build
├── docker-compose.yml                # Docker Compose orchestration
├── start.sh                          # One-command launcher (HTTP/HTTPS)
├── check.sh                          # Readiness check script
├── run_pipeline.sh                   # Overnight Demucs pipeline runner
├── run_overnight.sh                  # Sleep-proof pipeline (macOS caffeinate)
├── README.md                         # Project overview
├── GUIDE.md                          # Extended user guide
├── library/                          # [runtime] Downloaded songs
├── outputs/                          # [runtime] Rendered mixes
└── data/                             # [runtime] Indices, cache, audit log
```

---

## 1. Core Audio Modules

### 1.1 dj_engine.py (2075 lines)

**Purpose**: DJ-style transition planning and rendering.

**Key Classes**:

```python
@dataclass
class Beat:
    """Single beat in a track."""
    index: int              # 0-based beat number
    time: float             # seconds from start
    bar: int                # which bar (0-based)
    beat_in_bar: int        # 1–4 (position within bar)

@dataclass
class SongStructure:
    """Analysed structure of a track."""
    bpm: float
    beats: List[Beat]
    bars: List[Bar]
    sections: Dict[str, Section]  # intro, verse, chorus, drop, break, outro
    onset_times: List[float]
    estimated_duration: float

@dataclass
class TransitionPlan:
    """Plan for mixing Song A into Song B."""
    exit_bar_a: int              # last bar of Song A
    entry_bar_b: int             # first bar of Song B
    transition_bars: int         # overlap length (8/16/32)
    overlap_start_seconds: float  # when crossfade begins in A
    hp_freq_curve: Callable      # high-pass freq over time
    bass_swap_point: float        # seconds into transition for bass handover

class DJEngine:
    """Main transition renderer."""

    def analyze_structure(
        audio: np.ndarray,
        sr: int,
        bpm: Optional[float] = None,
    ) -> SongStructure

    def plan_transition(
        structure_a: SongStructure,
        structure_b: SongStructure,
    ) -> TransitionPlan

    def render(
        vocals_a: np.ndarray,
        instrumentals_b: np.ndarray,
        sr: int,
        plan: TransitionPlan,
    ) -> np.ndarray  # Rendered mix
```

**Features**:
- Phrase detection at 8/16/32-bar boundaries
- Section analysis (intro, verse, chorus, drop, break, outro)
- EQ plan with high-pass sweep and bass swap timing
- Sample-level beat-grid alignment
- Time-stretching for tempo matching
- Crossfade with dynamic EQ

**Config Parameters** (from config.yaml):
```yaml
dj:
  default_transition_bars: 16   # 8 | 16 | 32
  hp_filter_start_hz: 400.0     # High-pass start frequency
  hp_filter_end_hz: 80.0        # High-pass end frequency
  bass_crossover_hz: 150.0      # Bass/mid crossover
```

---

### 1.2 stems.py (450+ lines)

**Purpose**: Canonical Demucs stem separation with enhancement and normalization.

**Key Functions**:

```python
@dataclass
class StemResult:
    """Result of stem separation."""
    success: bool
    song_name: str
    wav: Optional[Path]          # Full mix (if kept)
    stems: Dict[str, Path]       # vocals, drums, bass, other
    enhance_info: Optional[Dict] = None
    error: Optional[str] = None

def separate_song_stems(
    song_name: str,
    enhance: bool = True,
    model: str = "htdemucs",
    no_backup: bool = False,
) -> StemResult
```

**Key Features**:
- Resolves correct Python executable (venv-first)
- Optional audio enhancement before separation
- Handles all Demucs output directory variations
- Normalizes each stem to recommended LUFS target
- Supports multiple Demucs models: `htdemucs`, `htdemucs_ft`, `mdx_extra`

**CLI Usage**:
```bash
python -m scripts.core.stems "Song Name"
python -m scripts.core.stems "Song Name" --model htdemucs_ft --no-enhance
```

---

### 1.3 gpu.py (300+ lines)

**Purpose**: Centralized GPU device detection and acceleration.

**Key Functions**:

```python
@lru_cache(maxsize=1)
def get_device() -> str:
    """
    Returns: "mps" | "cuda" | "cpu"

    Priority:
      1. REMIXMATE_DEVICE env var override
      2. Apple Silicon MPS (macOS arm64)
      3. NVIDIA CUDA (torch.cuda.is_available)
      4. CPU fallback
    """

def to_tensor(arr: np.ndarray) -> torch.Tensor
def to_numpy(t: torch.Tensor) -> np.ndarray
def gpu_stft(audio_tensor: torch.Tensor, n_fft: int = 2048) -> torch.Tensor
```

**Environment Variables**:
```bash
REMIXMATE_DEVICE=cpu      # Force CPU
REMIXMATE_DEVICE=mps      # Force Apple Silicon
REMIXMATE_DEVICE=cuda     # Force NVIDIA
```

---

### 1.4 mastering.py (400+ lines)

**Purpose**: ITU-R BS.1770-4 mastering chain.

**Key Functions**:

```python
def compute_lufs(
    audio: np.ndarray,
    sr: int,
) -> float
    """Returns integrated loudness in LUFS."""

def master_mix(
    mix_array: np.ndarray,
    sr: int = 44100,
    target_lufs: float = -14.0,
) -> Tuple[np.ndarray, QualityReport]
    """
    Full mastering chain:
      1. LUFS measurement
      2. Gain-only normalization
      3. True-peak brick-wall limiter (-1 dBFS)
      4. Clipping detection
    """

@dataclass
class QualityReport:
    lufs_integrated: float      # Final loudness
    peak_dbfs: float           # Maximum sample value
    clipping_detected: bool
    gain_applied_db: float
```

**Standards**:
- Streaming: -14 LUFS (Spotify, Apple Music, YouTube)
- DJ mix: -8 LUFS (louder, club playback)
- Ceiling: -1 dBFS (headroom for D/A)

---

### 1.5 audio_enhance.py (400+ lines)

**Purpose**: Pre-Demucs enhancement pipeline.

**Enhancement Chain**:
1. DC-offset removal + subsonic high-pass (20 Hz Butterworth)
2. Brick-wall low-pass at 20 kHz (anti-aliasing)
3. Noise gate (silence passages below threshold)
4. Soft-knee RMS compression (3:1, -20 dBFS threshold)
5. Air EQ (+1.5 dB @ 12 kHz)
6. LUFS normalization to -14 LUFS
7. True-peak limiter at -1 dBFS

**Key Classes**:

```python
@dataclass
class EnhanceOptions:
    """Knobs for enhancement chain."""
    hp_filter: bool = True
    lp_filter: bool = True
    noise_gate: bool = True
    compression: bool = True
    air_eq: bool = True
    lufs_target: float = -14.0
    true_peak_ceil: float = -1.0
    # ... stage parameters ...

@dataclass
class EnhanceReport:
    rms_before_db: float
    rms_after_db: float
    peak_before_db: float
    peak_after_db: float
    lufs_before: float
    lufs_after: float
    gain_applied_db: float
    clipped: bool
    stages_applied: List[str]

def enhance_audio(
    audio_array: np.ndarray,
    sr: int = 44100,
    options: EnhanceOptions = EnhanceOptions(),
) -> Tuple[np.ndarray, EnhanceReport]

def enhance_stems(
    stems: Dict[str, np.ndarray],
    sr: int = 44100,
    options: EnhanceOptions = EnhanceOptions(),
) -> Dict[str, np.ndarray]
```

---

### 1.6 beat_synth.py (600+ lines)

**Purpose**: Procedural drum beat synthesizer (no sample files).

**Instruments** (pure numpy/scipy):
- **Kick**: Sine with pitch envelope (150→50 Hz) + click transient
- **Snare**: Bandpass-filtered noise + 180 Hz sine body
- **Hihat**: Highpass-filtered noise, fast decay (closed)
- **Openhat**: Same, slower decay

**Genre Presets**:

```python
def render_beat(
    bpm: float = 128.0,
    genre: str = "techno",  # techno|house|hiphop|trap|dnb|ambient|auto
    bars: int = 16,
    sr: int = 22050,
) -> np.ndarray

def strudel_code(
    bpm: float = 128.0,
    genre: str = "techno",
) -> str  # Returns live-coding pattern for strudel.cc
```

**Genre Patterns**:
- **techno**: 4-on-the-floor kick, 2&4 snare, straight 16th hats
- **house**: 4-on-the-floor kick, 2&4 snare, offbeat 8th hats
- **hiphop**: Boom-bap kick (1&3), snare (2&4), swung 16th hats
- **trap**: Sparse kick, steady 32nd hats, snare (3)
- **dnb**: Fast syncopated kick, snare (2.5&4), dense hats
- **ambient**: Just open hats (no kick/snare)
- **auto**: Alias for techno (safe default)

---

### 1.7 music_index.py (700+ lines)

**Purpose**: FAISS RAG vector index for semantic song similarity.

**Vector Layout** (35 dimensions):

| Dim(s) | Name | Formula | Range |
|--------|------|---------|-------|
| 0 | bpm_norm | bpm / 200.0 | [0, 1] |
| 1–12 | key_onehot[12] | One-hot for C…B | {0,1} |
| 13 | mode | 1.0 = major, 0.0 = minor | [0, 1] |
| 14 | energy_mean | RMS energy | [0, 1] |
| 15 | energy_std | RMS energy spread | [0, 1] |
| 16 | drop_norm | drop_pos / duration | [0, 1] |
| 17 | danceability | beat_strength × beat_regularity | [0, 1] |
| 18 | beat_strength | Onset confidence | [0, 1] |
| 19 | tempo_stability | BPM variance | [0, 1] |
| 20 | vocal_density | Demucs vocal stem energy | [0, 1] |
| 21 | centroid_norm | centroid_hz / 8000 | [0, 1] |
| 22 | rolloff_norm | rolloff_hz / 16000 | [0, 1] |
| 23–34 | chroma[12] | Mean chroma, L2-normalized | [0, 1] |

**Default Segment Weights**:

```python
weights = {
    'bpm': 0.40,              # Tempo is DJ #1 constraint
    'key_onehot': 0.30,       # Harmonic compatibility (Camelot)
    'mode': 0.05,             # Major/minor feel
    'energy': 0.10,           # Dynamics match
    'rhythm': 0.08,           # Groove compatibility
    'vocal': 0.03,            # Avoid vocal clash
    'spectral': 0.02,         # Timbre (tie-breaker)
    'chroma': 0.02,           # Harmonic colour
}
```

**Public API**:

```python
def get_index() -> MusicIndex
    """Module-level singleton."""

class MusicIndex:
    def upsert_song(self, song_name: str) -> None
        """Add/refresh one song."""

    def remove_song(self, song_name: str) -> None
        """Remove from index."""

    def search(
        self,
        song_name: str,
        k: int = 5,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]
        """Top-k similar songs with score breakdown."""

    def rebuild(
        self,
        library_dir: Path,
        callback: Optional[Callable] = None,
    ) -> None
        """Rebuild entire library index."""

    def get_stats(self) -> Dict[str, Any]
        """Index size, last rebuild time."""

# Index file: data/music_index.json (relative to project root)
```

---

### 1.8 music_intelligence.py (800+ lines)

**Purpose**: Music feature vector computation and transition scoring.

**Key Classes**:

```python
@dataclass
class MusicVector:
    """Per-track feature vector."""
    bpm: float
    bpm_std: float                # Tempo stability
    key: str                       # Note name (C, C#, ..., B)
    mode: str                      # "major" | "minor"
    camelot: str                   # e.g., "8B" (harmonic wheel)
    energy_curve: np.ndarray       # Normalised RMS over time
    drop_position: float           # Time of loudest spike (seconds)
    spectral_brightness: Dict[str, float]  # centroid, rolloff, contrast
    danceability: float            # [0, 1]
    beat_strength: float           # [0, 1]
    vocal_density: float           # [0, 1]
    chord_progression: List[str]   # Simplified chords

@dataclass
class TransitionScore:
    """Transition compatibility score."""
    overall: float                 # 0–1 (weighted sum)
    beat_alignment: float          # BPM + phase match
    harmonic_match: float          # Camelot wheel distance
    energy_smoothness: float       # Curve continuity
    vocal_clash: float             # Vocal density penalty
    breakdown: Dict[str, float]    # Per-component scores
```

**Key Functions**:

```python
def compute_track_vector(
    audio: np.ndarray,
    sr: int,
    bpm: Optional[float] = None,
    stems_dir: Optional[Path] = None,
) -> MusicVector

def compute_transition_score(
    vec_a: MusicVector,
    vec_b: MusicVector,
) -> TransitionScore
    """
    Scoring formula:
    overall = 0.35 × beat_alignment
            + 0.35 × harmonic_match
            + 0.20 × energy_smoothness
            + 0.10 × (1 − vocal_clash)
            − vocal_clash_penalty
    """
```

**Camelot Wheel** (harmonic mixing):

```python
CAMELOT = {
    'C major': '8B',   'C# major': '3B',  'D major': '10B',  ...
    'A minor': '8A',   'A# minor': '3A',  'B minor': '10A',  ...
    # ... all 12 keys in both major and minor modes
}
```

**Krumhansl-Schmuckler Profiles** (tonal hierarchy):

```python
_MAJOR_PROF = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                         2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROF = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                         2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
```

---

### 1.9 recommend.py (250+ lines)

**Purpose**: Fast BPM-based song recommendation (with caching).

**Strategy**:
- Each song directory gets a lightweight `meta.json` cache
- Cache contains at minimum `{"bpm": <float>}`
- On cold call, processes up to `MAX_UNCACHED_PER_CALL = 120` songs
- Subsequent calls are instant

**Key Functions**:

```python
def get_recommendations(
    source_dir: Path,
    library_dir: Path,
    limit: int = 5,
    bpm_tolerance: float = 0.05,  # 5% tolerance
) -> List[Dict[str, Any]]
    """
    Returns:
    [
        {"name": "...", "bpm": 128.0, "bpm_score": 0.97, "overall": 0.97},
        ...
    ]
    """

_CACHE_FILE = "meta.json"
MAX_UNCACHED_PER_CALL = 120
EARLY_EXIT_THRESHOLD = 10  # High-quality matches before scanning all
```

---

### 1.10 genre.py (500+ lines)

**Purpose**: Multi-feature genre detection and preset lookup.

**Supported Genres**:
- house, techno, hiphop, trap, pop, rnb, dnb, ambient, rock, jazz

**Detection Features**:
- BPM range (primary classifier)
- Spectral centroid (brightness/hardness)
- Low-energy ratio (bass heaviness)
- Dynamic range (compressed vs dynamic)
- Zero-crossing rate (roughness)
- Chroma variance (harmonic richness)

**Key Classes**:

```python
@dataclass
class GenrePreset:
    """Full mix configuration for one genre."""
    genre: str
    display_name: str
    description: str

    # Loudness
    lufs_target: float
    true_peak_ceiling: float

    # Vocal chain
    vocal_gain_db: float
    vocal_hp_filter_hz: float       # High-pass cutoff
    vocal_reverb_send: float        # 0–1

    # Instrumental chain
    inst_gain_db: float
    inst_sidechain_amount: float    # Kick→bass sidechain depth (0–1)
    inst_compression_ratio: float   # 1:1 – 8:1

    # Global EQ / dynamics
    low_shelf_hz: float
    low_shelf_gain_db: float        # Negative = cut
    high_shelf_hz: float
    high_shelf_gain_db: float

    # Stereo / space
    stereo_width: float             # 0–1 (0 = mono, 1 = wide)

    # BPM tolerance
    bpm_range: Tuple[float, float]

def auto_preset(
    audio: np.ndarray,
    sr: int,
) -> GenrePreset
    """Detects best-matching genre and returns its preset."""

GENRE_PRESETS: Dict[str, GenrePreset]
```

---

### 1.11 library.py (550+ lines)

**Purpose**: Smart library management with deduplication, pruning, LRU eviction.

**Features**:
1. **Inventory**: Fast O(1) lookups via lightweight `.index.json`
2. **Deduplication**: Audio fingerprint (SHA-256 of first 30s PCM)
3. **Pruning**: Delete `full.wav` after stem separation (saves ~60% space)
4. **LRU eviction**: When library > max_size_gb, remove least-recently-accessed `full.wav` files
5. **License lookup**: Wrapper around scripts.core.license

**Key Classes**:

```python
@dataclass
class SongEntry:
    """One entry in library index."""
    name: str
    path: str                           # str(song_dir)
    size_bytes: int = 0
    has_full_wav: bool = False
    stems_available: List[str] = field(default_factory=list)
    fingerprint: Optional[str] = None   # SHA-256 of first 30s PCM
    last_accessed: float = field(default_factory=time.time)
    source: Optional[str] = None        # "youtube", "jamendo", etc.

class LibraryManager:
    def touch(self, song_name: str) -> None
        """Mark as accessed."""

    def get_size_gb(self) -> float

    def prune_raw(self, song_name: str) -> None
        """Delete full.wav if stems exist."""

    def evict_lru(self) -> None
        """Free space if over cap."""

    def add_song(self, song_name: str, metadata: Dict) -> None

    def remove_song(self, song_name: str) -> None
```

**Config** (from config.yaml):

```yaml
library:
  max_size_gb: 50.0                  # Trigger eviction
  keep_raw_after_separation: false   # Delete full.wav
  prune_on_download: true            # Auto-prune on sep
```

---

### 1.12 audit.py (150+ lines)

**Purpose**: Immutable JSONL audit log.

**Every operation is recorded**:
- Downloads, remixes, deletions, library modifications
- Tamper-evident append-only log at `data/audit.jsonl`

**Key Functions**:

```python
def log_audit(
    action: str,                           # "download_start", "remix_complete", etc.
    *,
    resource: Optional[str] = None,        # Song name, file path, etc.
    user: Optional[str] = None,            # Who initiated it
    job_id: Optional[str] = None,          # Associated job ID
    metadata: Optional[Dict[str, Any]] = None,  # Extra context
) -> None

def read_audit(
    limit: int = 100,
    action_filter: Optional[str] = None,
) -> List[Dict[str, Any]]
    """Read recent entries (newest first)."""

# Example entry:
{
    "ts": "2024-03-22T14:35:22.123456Z",
    "epoch": 1711101322.123456,
    "action": "download_start",
    "resource": "Anyma - Voices",
    "user": "local",
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "meta": {"source": "youtube", "query": "..."}
}
```

---

### 1.13 logging_utils.py (350+ lines)

**Purpose**: Structured JSON logging with request/job ID context variables.

**Key Features**:
- Automatic JSON serialization
- Request-scoped trace IDs via `contextvars`
- Job ID propagation
- Human-readable console output (dev)
- JSON-formatted output (production)

**Key Functions**:

```python
_request_id_context: contextvars.ContextVar[Optional[str]]
_job_id_context: contextvars.ContextVar[Optional[str]]

def set_request_id(request_id: str) -> None
def get_request_id() -> Optional[str]
def set_job_id(job_id: str) -> None
def get_job_id() -> Optional[str]

def get_logger(name: str) -> StructuredLogger

class StructuredJsonFormatter(logging.Formatter):
    """Outputs structured JSON with metadata."""

    # Output format:
    # {
    #     "timestamp": "2024-03-22T14:35:22.123456Z",
    #     "level": "INFO",
    #     "logger_name": "scripts.api.routes",
    #     "message": "Remix task started",
    #     "request_id": "550e8400-e29b-41d4-a716-446655440000",
    #     "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    #     "extra": {...}
    # }
```

---

## 2. API Modules (FastAPI Backend)

### 2.1 main.py (150 lines)

**Purpose**: FastAPI app factory, middleware setup, static file mounting.

**Key Features**:
- **Request ID Middleware**: Assigns UUID to each request, propagates via contextvars
- **CORS Middleware**: Configurable origin list (dev: all origins, prod: tightened)
- **Request ID Header**: Adds `X-Request-ID` to all responses
- **Static File Mount**: Serves UI assets from `scripts/ui/static/`
- **Health Endpoint**: Liveness check at startup

**API Info**:

```python
app = FastAPI(
    title="AI RemixMate API",
    version="0.2.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc
)
```

**Startup/Shutdown**:

```python
@app.on_event("startup")
async def startup():
    configure_logging()
    ensure_directories()
    logger.info("AI RemixMate API started")

@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown — wait for in-flight jobs."""
    _executor.shutdown(wait=True, cancel_futures=False)
```

---

### 2.2 routes.py (1000+ lines)

**Purpose**: All HTTP endpoint handlers.

**Endpoint Categories**:

#### Health Checks
```
GET  /health              — Liveness + readiness check
GET  /health/live         — Liveness only
GET  /health/ready        — Readiness (dependency checks)
```

#### Library Management
```
GET  /library                    — List all songs + stats
GET  /library/{name}             — Single song detail
GET  /library/{name}/audio       — Stream full.wav or vocal stem
DELETE /library/{name}           — Remove song
POST /library/initialize         — Batch stem split + compress + index
POST /library/batch-compress     — FLAC compression pipeline
POST /library/rebuild-index      — Rebuild music_index
```

#### Discovery & Compatibility
```
GET  /library/similar?query=...  — RAG semantic search
POST /compatibility              — Instant Camelot + BPM check
POST /analyze                    — Genre + structure analysis (queued)
GET  /genres                     — List supported genres
```

#### Download & Separation
```
POST /download             — Download track (queued)
POST /playlist-download    — Batch download (queued)
POST /stem-split           — Single song separation (queued)
POST /batch-stem-split     — Batch separation (queued)
```

#### DJ Mixing & Advanced
```
POST /dj-remix             — Render DJ transition (queued)
POST /dj-chain             — Advanced remix with custom params (queued)
POST /instrument-lab       — Stem editor + remix (queued)
POST /bridge-beat          — Generate solo bridge beat (instant)
```

#### Job Tracking
```
GET  /jobs                 — List recent jobs
GET  /jobs/{job_id}        — Poll specific job
```

#### Outputs
```
GET  /outputs/{session}/{file}   — Stream rendered mix
```

---

### 2.3 jobs.py (200+ lines)

**Purpose**: In-memory async job queue with ETA estimation.

**Architecture**:
- ThreadPoolExecutor (2 workers) for CPU-bound tasks
- No Redis/Celery (fits single-process dev server)
- In-memory state (lost on restart)
- For production scale-out: swap executor for task queue

**Key Functions**:

```python
_jobs: Dict[str, Dict[str, Any]] = {}  # Job store
_executor = ThreadPoolExecutor(max_workers=2)

def create_job(job_type: JobType, meta: Optional[Dict] = None) -> str
    """Register new job, return ID."""

def get_job(job_id: str) -> Optional[Dict]

def list_jobs(limit: int = 50) -> List[Dict]
    """Most recent jobs, newest first."""

def update_job(
    job_id: str,
    *,
    status: Optional[JobStatus] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    eta_sec: Optional[float] = None,
    result: Optional[Dict] = None,
    error: Optional[str] = None,
) -> None

def run_job(
    job_id: str,
    task_fn: Callable,
    **kwargs,
) -> None
    """Execute task in executor thread, handle errors."""

def job_to_response(job_dict: Dict) -> JobResponse
    """Convert job dict to Pydantic model."""
```

**Job States**:

```python
class JobStatus(str, Enum):
    PENDING  = "pending"    # Queued, not yet running
    RUNNING  = "running"    # Currently executing
    DONE     = "done"       # Completed successfully
    FAILED   = "failed"     # Errored out
```

---

### 2.4 tasks.py (1000+ lines)

**Purpose**: Long-running task functions executed in job threads.

**Task Function Signature**:

```python
def task_xxx(
    job_id: str,
    **kwargs,
) -> Dict[str, Any]
    """
    Execute work, call update_job() for progress.
    Return result dict on success.
    Exceptions bubble up to job runner (marks job FAILED).
    """
```

**Key Tasks**:

#### task_download

```python
def task_download(
    job_id: str,
    query: str,
    name: Optional[str],
    separate: bool = True,  # Always True now
) -> Dict[str, Any]
    """Download track with auto-Demucs separation."""
```

#### task_stem_split

```python
def task_stem_split(
    job_id: str,
    song_name: str,
    model: str = "htdemucs",
    enhance: bool = True,
) -> Dict[str, Any]
    """Single song Demucs separation."""
```

#### task_batch_stem_split

```python
def task_batch_stem_split(
    job_id: str,
    song_names: Optional[List[str]] = None,
    model: str = "htdemucs",
    enhance: bool = True,
    delete_wav: bool = True,
) -> Dict[str, Any]
    """Batch separation (all library if song_names empty)."""
```

#### task_dj_remix

```python
def task_dj_remix(
    job_id: str,
    song_a: str,
    song_b: str,
    transition_bars: int = 16,
    preset: str = "auto",
    bridge_beat_mode: str = "auto",
    output_format: str = "wav",
) -> Dict[str, Any]
    """Render DJ transition with optional bridge beat."""
```

#### task_analyze

```python
def task_analyze(
    job_id: str,
    song_name: str,
    force: bool = False,
) -> Dict[str, Any]
    """Genre + structure analysis, cache metadata."""
```

#### task_initialize_library

```python
def task_initialize_library(
    job_id: str,
    enhance: bool = True,
    model: str = "htdemucs",
    delete_wav: bool = True,
    run_compress: bool = True,
    run_index: bool = True,
) -> Dict[str, Any]
    """Batch stem split → FLAC compress → rebuild index."""
```

---

### 2.5 schemas.py (350+ lines)

**Purpose**: Pydantic request/response models.

**Enums**:

```python
class JobStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"

class JobType(str, Enum):
    DOWNLOAD   = "download"
    DJ_REMIX   = "dj_remix"
    SEPARATE   = "separate"
    ANALYZE    = "analyze"
```

**Response Models**:

```python
class SongInfo(BaseModel):
    name: str
    size_mb: float
    has_full_wav: bool
    stems: List[str] = []
    license_type: Optional[str] = None
    source: Optional[str] = None
    last_accessed: Optional[float] = None

class CompatibilityResult(BaseModel):
    song_a: str
    song_b: str
    compatible: bool
    overall: float          # 0–1
    bpm_score: float
    key_score: float
    energy_score: float
    bpm_a: float
    bpm_b: float
    camelot_a: Optional[str]
    camelot_b: Optional[str]
    genre_a: Optional[str]
    genre_b: Optional[str]

class GenreResult(BaseModel):
    song: str
    genre: str
    confidence: float
    runner_up: Optional[str] = None
    bpm: Optional[float] = None

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    job_type: JobType
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: float        # 0–1
    message: str
    result: Optional[Dict] = None
    error: Optional[str] = None
    eta_sec: Optional[float] = None

class LibraryListResponse(BaseModel):
    songs: List[SongInfo]
    stats: LibraryStats

class LibraryStats(BaseModel):
    total_songs: int
    total_size_gb: float
    songs_with_stems: int
    songs_with_full_wav: int
```

**Request Models**:

```python
class DownloadRequest(BaseModel):
    query: str
    name: Optional[str] = None
    separate: bool = True

class DJRemixRequest(BaseModel):
    song_a: str
    song_b: str
    transition_bars: int = 16
    preset: str = "auto"
    bridge_beat_mode: str = "auto"
    output_format: str = "wav"

class CompatibilityRequest(BaseModel):
    song_a: str
    song_b: str

class AnalyzeRequest(BaseModel):
    song_name: str
    force: bool = False

class StemSplitRequest(BaseModel):
    song_name: str
    model: str = "htdemucs"
    enhance: bool = True

class BatchStemSplitRequest(BaseModel):
    song_names: Optional[List[str]] = None
    model: str = "htdemucs"
    enhance: bool = True
    delete_wav: bool = True
```

---

## 3. UI Module (Streamlit)

### 3.1 app.py (5000+ lines)

**Purpose**: Dark DJ-booth themed Streamlit web interface.

**Key Sections**:

#### Configuration & Setup
```python
def _api_host() -> str           # localhost for server-side calls
def _lan_ip() -> str             # Machine LAN IP for browser-facing links
def _detect_https() -> bool      # Check for cert.pem, key.pem

_LAN_IP = _lan_ip()
_HTTPS = os.environ.get("REMIXMATE_HTTPS", "").lower() in ("1", "true", "yes")
_PROTO = "https" if _HTTPS else "http"
API = f"{_PROTO}://localhost:8000"
API_PUBLIC = f"{_PROTO}://{_LAN_IP}:8000"
```

#### Visual Theme
- **Page**: Wide layout, expanded sidebar
- **Colors**: Neon purples (#c084fc), dark background (#0a0a0e)
- **Animations**: Equalizer bars, vinyl rotation, gradient transitions, liquid morphs, rainbow borders

#### Pages
1. **Home / Dashboard**
   - Quick Actions (Analyze, Mix, Search)
   - Recent Mixes (last 6 outputs with audio preview)
   - Library Stats

2. **Download & Library**
   - Single track download with optional stem separation
   - Playlist batch download
   - Library browser (filter, sort by date, size, stems available)
   - Song detail view

3. **Smart Search (RAG)**
   - Natural-language query input
   - Semantic similarity ranking
   - One-click mix launch

4. **My Library**
   - Full song inventory with metadata
   - License type + source display
   - Quick-delete + audio preview

5. **Create Mix**
   - Song A / Song B selector
   - Transition bars (8/16/32)
   - Preset selector (auto, radio, club, ambient)
   - Bridge beat mode (off, auto, solo)
   - Output format (WAV, FLAC)
   - Real-time progress with ETA

6. **Compatibility Check**
   - Instant BPM, key, energy check
   - Camelot wheel visualization
   - No download required

7. **Advanced / Instrument Lab**
   - Per-stem gain/pan/reverb controls
   - Custom EQ chain
   - Preview individual stems

8. **My Mixes**
   - Rendered output history browser
   - Type filter, date sort
   - Inline audio player
   - Download links

**Key Components**:

```python
def _inject_css() -> None
    """Global CSS with keyframe animations."""

def render_header() -> None
    """Logo, title, tabs."""

def render_quick_actions() -> None
    """Card-based quick navigation."""

def render_recent_mixes() -> None
    """Last 6 outputs with preview."""

@st.fragment
def render_job_progress(job_id: str, session: str) -> None
    """Live polling with progress bar + ETA."""

def api_call(
    method: str,
    endpoint: str,
    json: Optional[Dict] = None,
    files: Optional[Dict] = None,
) -> Optional[Dict]
    """Wrapper around requests library."""
```

---

## 4. Shell Scripts

### 4.1 start.sh (210 lines)

**Purpose**: One-command launcher for API + UI (HTTP/HTTPS).

**Usage**:

```bash
./start.sh              # Start both (HTTP)
./start.sh --https      # Start both (HTTPS, generate certs if needed)
./start.sh api          # API only
./start.sh ui           # UI only
./start.sh stop         # Kill both
```

**Features**:
- Kills stale processes on ports 8000, 8501
- Auto-generates SSL certificates if needed
- Detects LAN IP for phone/tablet access
- Logs PID of each process
- Waits for both to exit (Ctrl+C stops both)

---

### 4.2 check.sh (155 lines)

**Purpose**: Readiness validation before launch.

**Checks**:
1. Python 3 available
2. Core packages (streamlit, fastapi, librosa, torch, scipy, numpy)
3. Optional packages (demucs, yt_dlp, ytmusicapi, sentence_transformers)
4. System tools (ffmpeg, ffprobe)
5. GPU detection (CUDA, MPS, or CPU)
6. Project structure (key files present)
7. Configuration files exist
8. Library inventory (song count, stems)
9. Ports free (8000, 8501)
10. Syntax validation (routes.py, app.py, tasks.py)

**Output**:

```
╔══════════════════════════════════════╗
║   AI RemixMate — Readiness Check     ║
╚══════════════════════════════════════╝

📦  Python
  ✅  Python 3.11.x
📦  Core Dependencies
  ✅  streamlit
  ✅  fastapi
  ... (all pass/fail/warn)
🌐  Ports
  ✅  Port 8000 is free
  ✅  Port 8501 is free

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ 28 passed   ❌ 0 failed   ⚠️ 2 warnings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  🚀  Ready to launch!  Run:  ./start.sh
```

---

### 4.3 run_pipeline.sh (190 lines)

**Purpose**: Fire-and-forget Demucs pipeline for entire library.

**Steps**:
1. Stem split all songs (Demucs)
2. FLAC compression (optional)
3. Rebuild music index (optional)
4. Delete raw WAV files (optional)

**Usage**:

```bash
./bin/run_pipeline.sh                      # Default: htdemucs model
./bin/run_pipeline.sh --model htdemucs_ft  # Higher quality
./bin/run_pipeline.sh --no-compress        # Skip FLAC compression
./bin/run_pipeline.sh --no-enhance         # Skip audio enhancement
./bin/run_pipeline.sh --keep-wav           # Don't delete full.wav
```

**Polling**:

```bash
tail -f pipeline.log    # Watch progress in real-time
```

**Features**:
- Continuous polling (15s intervals)
- ETA calculation based on progress
- Elapsed time tracking
- Result summary on completion
- Comprehensive error reporting

---

### 4.4 run_overnight.sh (68 lines)

**Purpose**: Sleep-proof pipeline runner (macOS specific).

**Features**:
- Starts API if not already running
- Uses `caffeinate` to prevent system sleep
- Wraps `run_pipeline.sh`
- Plays completion sound + notification on success
- Logs everything

**Usage**:

```bash
./bin/run_overnight.sh
./bin/run_overnight.sh --model htdemucs_ft
```

---

## 5. Configuration

### 5.1 config.yaml (100 lines)

**Sections**:

```yaml
# Audio processing
audio:
  sample_rate: 44100          # 44100 or 48000 Hz
  bit_depth: 24               # 16 or 24 bits
  target_lufs: -14.0          # Streaming standard
  channels: 1                 # 1 = mono, 2 = stereo

# Remix engine
remix:
  default_preset: radio       # radio | club | ambient
  optimizer_iterations: 50    # Random-search iterations
  max_pitch_shift_semitones: 2
  beat_alignment_tolerance_ms: 40

# Stem separation
separation:
  model: htdemucs             # htdemucs | htdemucs_ft | mdx_extra
  device: auto                # auto | cpu | cuda | mps

# Download settings
download:
  default_source: auto        # auto | youtube | jamendo
  jamendo_client_id: b6747d04
  audio_format: wav
  audio_quality: 0            # 0 = best
  no_playlist_by_default: true

# Metadata caching
metadata:
  getsongbpm_api_key: ""      # Get at https://getsongbpm.com/api
  lastfm_api_key: ""
  cache_path: data/metadata.db
  cache_ttl_days: 30

# DJ engine
dj:
  default_transition_bars: 16 # 8 | 16 | 32
  hp_filter_start_hz: 400.0
  hp_filter_end_hz: 80.0
  bass_crossover_hz: 150.0

# Library management
library:
  max_size_gb: 50.0           # LRU eviction trigger
  keep_raw_after_separation: false
  prune_on_download: true

# Database
database:
  path: data/remixmate.db
  embeddings_path: data/song_embeddings.json

# API
api:
  host: 0.0.0.0
  port: 8000
  workers: 1
  reload: false               # true in dev
  cors_origins:
    - "*"                     # Tighten in prod

# Logging
logging:
  level: INFO                 # DEBUG | INFO | WARNING | ERROR
  format: "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
  file: null                  # null = stdout; or "logs/remixmate.log"
```

**Override Methods** (priority order):
1. `config.local.yaml` (highest priority, gitignored)
2. Environment variables: `REMIXMATE_<SECTION>_<KEY>`
3. `config.yaml` (default, lowest priority)

### 5.2 .streamlit/config.toml

```toml
[server]
headless = true
port = 8501
address = "0.0.0.0"
maxUploadSize = 200         # MB
enableCORS = false
enableXsrfProtection = true

[theme]
primaryColor = "#c084fc"
backgroundColor = "#0a0a0e"
secondaryBackgroundColor = "#12121c"
textColor = "#e8e8f0"
font = "sans serif"

[browser]
gatherUsageStats = false
```

---

## 6. Dependencies

### 6.1 pyproject.toml

```toml
[project]
name = "remixmate"
version = "0.2.0"
description = "AI-powered music remix engine"
requires-python = ">=3.10"

[project.dependencies]
# Audio core
librosa>=0.11
soundfile>=0.12
pydub>=0.25
demucs>=4.0

# Speech-to-text
openai-whisper>=20250625

# ML / embeddings
torch>=2.0
torchaudio>=2.0
scikit-learn>=1.7
sentence-transformers>=5.0

# Signal processing
numpy>=2.0
scipy>=1.16

# Data
pandas>=2.0
PyYAML>=6.0

# Download
yt-dlp>=2025.1
ytmusicapi>=1.10
mutagen>=1.47

# Web / API
fastapi>=0.115
uvicorn[standard]>=0.35
python-multipart>=0.0.20

# Progress / CLI
tqdm>=4.67
click>=8.2
streamlit>=1.47
requests>=2.32
tenacity>=9.0

[project.optional-dependencies]
dev = [
    "pytest>=8.4",
    "pytest-asyncio>=0.25",
    "black>=25.1",
    "ruff>=0.12",
    "mypy>=1.16",
    "pre-commit>=4.2",
]
```

### 6.2 requirements.txt (85 lines)

**Pinned versions** for reproducible builds:

```
# Audio core
librosa==0.11.0
soundfile==0.13.1
pydub==0.25.1
soxr==0.5.0.post1
audioread==3.0.1

# Stem separation
demucs==4.0.1

# ML
torch==2.7.1
torchaudio==2.7.1
scikit-learn==1.7.1
sentence-transformers==5.0.0
transformers==4.53.3

# Signal processing
numpy==2.2.6
scipy==1.16.0
numba==0.61.2

# Data
pandas==2.3.1
PyYAML==6.0.2

# Download
yt-dlp==2025.7.21
ytmusicapi==1.10.2
mutagen==1.47.0

# Web
fastapi==0.115.14
uvicorn[standard]==0.35.0
python-multipart==0.0.20

# UI
streamlit==1.47.0
altair==5.5.0

# Utilities
requests==2.32.4
tenacity==9.1.2
tqdm==4.67.1
click==8.2.1
```

---

## 7. Docker

### 7.1 Dockerfile (36 lines)

```dockerfile
FROM python:3.11-slim AS base

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -s /bin/bash remixmate \
    && mkdir -p /app/library /app/data /app/output /app/mixes \
    && chown -R remixmate:remixmate /app
USER remixmate

# Expose API + Streamlit ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start API server
CMD ["python", "-m", "scripts.api.main"]
```

### 7.2 docker-compose.yml (37 lines)

```yaml
version: "3.9"

services:
  api:
    build: .
    container_name: remixmate-api
    ports:
      - "8000:8000"
    volumes:
      - ./library:/app/library
      - ./data:/app/data
      - ./output:/app/output
      - ./mixes:/app/mixes
    environment:
      - REMIXMATE_API_HOST=0.0.0.0
      - REMIXMATE_API_PORT=8000
      - REMIXMATE_LOGGING_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  ui:
    build: .
    container_name: remixmate-ui
    command: streamlit run scripts/ui/app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - REMIXMATE_API_URL=http://api:8000
    depends_on:
      api:
        condition: service_healthy
    restart: unless-stopped
```

---

## 8. Testing

### 8.1 Test Suite (test_new_features.py + test_core_modules.py)

**Coverage**:

```
Unit Tests:
  - Audit module (log_audit, read_audit, filtering)
  - Structured logging (context vars, formatters)
  - Job store ETA calculation + lifecycle
  - DJEngine input validation
  - Rate limiting

Integration Tests:
  - Full job lifecycle (create → submit → poll → done/failed)
  - job_to_response() round-trip

Smoke Tests:
  - Health endpoint responses
  - Config system (all sections load as dataclasses)
  - Genre detection (synthetic audio)
  - DJ structure analysis (synthetic audio)
  - Compatibility scoring

Run:
  python -m pytest tests/test_new_features.py -v
  python -m pytest tests/test_core_modules.py -v
```

**Synthetic Audio Helper**:

```python
def _make_audio(
    duration: float = 5.0,
    bpm: float = 128.0,
    freq: float = 440.0,
    sr: int = 44100,
) -> np.ndarray:
    """Sine wave with BPM-aligned amplitude envelope."""
    # Returns: ndarray, shape (sr * duration,), dtype float32
```

---

## 9. Public API Endpoints

### 9.1 Health & Status

```
GET /health
GET /health/live
GET /health/ready
```

### 9.2 Library Operations

```
GET /library                              List all songs + stats
GET /library/{name}                       Song detail
GET /library/{name}/audio                 Stream full.wav
GET /library/similar?query=...&limit=5    RAG semantic search
DELETE /library/{name}                    Remove song

POST /library/initialize                  Batch process entire library
POST /library/batch-compress              FLAC compression
POST /library/rebuild-index               Rebuild music_index
```

### 9.3 Download & Separation

```
POST /download              Single track download (auto-stems)
POST /playlist-download     Batch download
POST /stem-split            Single song separation
POST /batch-stem-split      Batch separation
```

### 9.4 Discovery

```
POST /compatibility         BPM + key + energy check (instant)
POST /analyze               Genre + structure analysis (queued)
GET /genres                 List supported genres
```

### 9.5 DJ Mixing

```
POST /dj-remix              Render DJ transition (queued)
POST /dj-chain              Advanced remix (queued)
POST /instrument-lab        Stem editor + remix (queued)
POST /bridge-beat           Solo bridge beat (instant)
```

### 9.6 Job Tracking

```
GET /jobs                   List recent jobs
GET /jobs/{job_id}          Poll specific job
```

### 9.7 Outputs

```
GET /outputs/{session}/{file}   Stream rendered mix
```

---

## 10. Key Workflows

### 10.1 Download → Analyze → Mix

```
1. POST /download
   {
     "query": "Anyma - Voices In My Head",
     "separate": true
   }
   → Returns job_id

2. GET /jobs/{job_id}  [Poll until done]
   → Returns stems at library/Anyma - Voices In My Head/

3. POST /compatibility
   {
     "song_a": "Anyma - Voices In My Head",
     "song_b": "Dom Dolla - Define"
   }
   → Returns bpm_score, key_score, energy_score, camelot positions

4. POST /dj-remix
   {
     "song_a": "Anyma - Voices In My Head",
     "song_b": "Dom Dolla - Define",
     "transition_bars": 16,
     "preset": "auto",
     "bridge_beat_mode": "auto",
     "output_format": "wav"
   }
   → Returns job_id → Poll until done → outputs/remix_xyz.wav
```

### 10.2 Overnight Library Processing

```bash
./bin/run_overnight.sh --model htdemucs_ft
```

This:
1. Starts API if not running
2. Prevents system sleep (caffeinate on macOS)
3. Batch stems splits entire library (htdemucs_ft model)
4. Compresses stems to FLAC
5. Rebuilds music_index (RAG)
6. Deletes full.wav files
7. Logs progress to pipeline.log
8. Plays completion notification

---

## 11. Configuration & Customization

### 11.1 Create config.local.yaml

```bash
cp config.yaml config.local.yaml
# Edit config.local.yaml with your settings
```

**Example customizations**:

```yaml
# Faster model for quicker processing
separation:
  model: htdemucs            # Instead of htdemucs_ft

# Larger library (more eviction)
library:
  max_size_gb: 200.0

# Tighter API security
api:
  cors_origins:
    - "http://localhost:8501"
    - "http://127.0.0.1:8501"

# Debug logging
logging:
  level: DEBUG
  file: "logs/remixmate.log"
```

### 11.2 Environment Variables

Override any config value:

```bash
export REMIXMATE_AUDIO_SAMPLE_RATE=48000
export REMIXMATE_DJ_DEFAULT_TRANSITION_BARS=32
export REMIXMATE_API_PORT=9000
export REMIXMATE_DEVICE=mps    # Force GPU device
```

### 11.3 Streaming Standards

The default target is **-14 LUFS** (Spotify, Apple Music, YouTube).

To adjust:

```yaml
audio:
  target_lufs: -8.0    # DJ mix (louder)
  # or
  target_lufs: -11.0   # Podcasts
```

---

## 12. Performance Tuning

### 12.1 GPU Acceleration

All core modules auto-detect GPU:

```python
from scripts.core.gpu import get_device
device = get_device()  # "mps" | "cuda" | "cpu"
```

**Performance Impact**:
- STFT: 2–10x faster on GPU
- Cosine similarity (music index): 50–100x faster
- Time-stretching: 3–5x faster
- Resampling: 2–3x faster

### 12.2 Stem Separation Models

| Model | Speed | Quality | GPU Memory |
|-------|-------|---------|------------|
| htdemucs | Fast | Good | 4 GB |
| htdemucs_ft | Medium | Excellent | 6 GB |
| mdx_extra | Slow | Best | 12 GB |

### 12.3 Music Index Optimization

```python
from scripts.core.music_index import get_index

# On first call, rebuild (slow)
index = get_index()
index.rebuild(LIBRARY_DIR)

# Subsequent calls are instant (from memory)
results = index.search("Anyma - Voices", k=10)
```

Vector searches are **sub-millisecond** on 100+ songs.

### 12.4 Library Eviction

When `library.max_size_gb` is exceeded:
1. LRU (least-recently-accessed) `full.wav` files are deleted first
2. Stems are preserved (so re-mixing works)
3. Size check runs after each separation

---

## 13. Troubleshooting

### 13.1 Port Already in Use

```bash
# Find process on port 8000
lsof -ti tcp:8000

# Kill it
kill -9 <PID>

# Or use start.sh
./start.sh stop
./start.sh
```

### 13.2 GPU Not Detected

```bash
# Force CPU
export REMIXMATE_DEVICE=cpu
./start.sh

# Check GPU
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.backends.mps.is_available())"
```

### 13.3 Demucs Not Found

```bash
# Ensure venv is activated
source remix-env/bin/activate

# Reinstall Demucs
pip install demucs==4.0.1

# Check installation
python -m demucs --help
```

### 13.4 Memory Issues (Stem Separation)

```yaml
# Use faster model that uses less memory
separation:
  model: htdemucs    # Not htdemucs_ft
```

Or process library in batches:

```bash
./bin/run_pipeline.sh --model htdemucs
```

---

## 14. Deployment

### 14.1 Docker

```bash
docker build -t ai-remixmate .
docker run -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/library:/app/library \
  -v $(pwd)/outputs:/app/outputs \
  ai-remixmate
```

### 14.2 Docker Compose

```bash
docker compose up
```

Accesses: `http://localhost:8501`

### 14.3 HTTPS Setup

```bash
./start.sh --https
# Auto-generates certs in certs/ directory
```

For custom certs:

```bash
cp your-cert.pem certs/cert.pem
cp your-key.pem certs/key.pem
./start.sh --https
```

### 14.4 Systemd Service (Linux)

Create `/etc/systemd/system/remixmate.service`:

```ini
[Unit]
Description=AI RemixMate
After=network.target

[Service]
Type=simple
User=remixmate
WorkingDirectory=/home/remixmate/ai-remixmate
ExecStart=/home/remixmate/ai-remixmate/remix-env/bin/python -m scripts.api.main
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable & start:

```bash
systemctl enable remixmate
systemctl start remixmate
systemctl status remixmate
```

---

## 15. Legal & Attribution

### 15.1 Download Sources

**yt-dlp** (Unlicense):
- Personal use / local remix: ✅
- Educational demos / portfolio: ✅
- Production SaaS: ⚠️ (Consult lawyer)

**Legal alternatives for production**:
- **Jamendo**: Official API, CC licensed
- **Free Music Archive**: CC licensed
- **Musopen**: Public-domain classical
- **Audionautix / Incompetech**: CC BY

### 15.2 Demucs

Meta's Demucs is licensed under CC-BY-NC (non-commercial).

For commercial use, contact Meta for licensing.

---

## 16. File Reference Index

| Path | Lines | Purpose |
|------|-------|---------|
| scripts/api/main.py | 150 | FastAPI app factory |
| scripts/api/routes.py | 1000+ | HTTP endpoints |
| scripts/api/jobs.py | 200+ | Job queue + ETA |
| scripts/api/tasks.py | 1000+ | Long-running tasks |
| scripts/api/schemas.py | 350+ | Pydantic models |
| scripts/core/dj_engine.py | 2075 | Transition planner/renderer |
| scripts/core/stems.py | 450+ | Demucs integration |
| scripts/core/gpu.py | 300+ | GPU detection |
| scripts/core/mastering.py | 400+ | LUFS mastering |
| scripts/core/audio_enhance.py | 400+ | Enhancement pipeline |
| scripts/core/beat_synth.py | 600+ | Beat synthesis |
| scripts/core/music_index.py | 700+ | FAISS RAG index |
| scripts/core/music_intelligence.py | 800+ | Feature vectors |
| scripts/core/recommend.py | 250+ | BPM recommendation |
| scripts/core/genre.py | 500+ | Genre detection |
| scripts/core/library.py | 550+ | Library management |
| scripts/core/audit.py | 150+ | Audit logging |
| scripts/core/logging_utils.py | 350+ | Structured logging |
| scripts/ui/app.py | 5000+ | Streamlit UI |
| pyproject.toml | 83 | Package metadata |
| requirements.txt | 85 | Dependencies |
| Dockerfile | 36 | Docker build |
| docker-compose.yml | 37 | Docker orchestration |
| config.yaml | 100 | Configuration |
| start.sh | 210 | Launcher |
| check.sh | 155 | Readiness check |
| run_pipeline.sh | 190 | Demucs pipeline |
| run_overnight.sh | 68 | Sleep-proof runner |
| tests/test_new_features.py | 500+ | Unit/integration tests |
| tests/test_core_modules.py | 400+ | Smoke tests |

---

## 17. Quick Reference

### Clone & Setup

```bash
git clone https://github.com/Chunduri-Aditya/ai-remixmate.git
cd ai-remixmate
python -m venv remix-env
source remix-env/bin/activate    # Windows: remix-env\Scripts\activate
pip install -e ".[dev]"
```

### Validate & Launch

```bash
bash check.sh                          # Readiness check
./start.sh                             # Start API + UI
```

### Process Library Overnight

```bash
./bin/run_overnight.sh --model htdemucs_ft
tail -f pipeline.log                   # Watch progress
```

### Access Points

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| API Docs | http://localhost:8000/docs |
| API ReDoc | http://localhost:8000/redoc |

### Common Tasks

```bash
# Download a track
curl -X POST http://localhost:8000/download \
  -H "Content-Type: application/json" \
  -d '{"query": "Anyma - Voices", "separate": true}'

# Check compatibility
curl "http://localhost:8000/compatibility?song_a=Anyma+-+Voices&song_b=Dom+Dolla+-+Define"

# Search similar songs
curl "http://localhost:8000/library/similar?query=dark+progressive+techno&limit=5"

# Render a DJ mix
curl -X POST http://localhost:8000/dj-remix \
  -H "Content-Type: application/json" \
  -d '{"song_a": "Anyma - Voices", "song_b": "Dom Dolla - Define", "transition_bars": 16}'
```

---

## Final Notes

**AI RemixMate** is a comprehensive, production-ready audio engineering project demonstrating:

- ✅ Full-stack Python architecture (API + UI + Audio engine)
- ✅ GPU-accelerated audio processing (MPS/CUDA/CPU fallback)
- ✅ Async job queue with ETA estimation
- ✅ Structured logging & audit trails
- ✅ RAG semantic search (FAISS vector index)
- ✅ Professional mastering chain (ITU-R BS.1770-4)
- ✅ Docker orchestration
- ✅ Comprehensive test suite (80+ tests)
- ✅ Dark UI with smooth animations

All code is modular, well-documented, and designed for easy extension.

