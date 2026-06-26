# AI RemixMate — Architecture Map

Comprehensive mapping of backend layers, frontend layers, data flow, feature dependencies, and module import graph.

---

## Table of Contents
1. [Backend Layer Structure](#backend-layer-structure)
2. [Frontend Layer Structure](#frontend-layer-structure)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Feature-to-Endpoint Mapping](#feature-to-endpoint-mapping)
5. [Module Dependencies & Import Graph](#module-dependencies--import-graph)
6. [API Surface & Schemas](#api-surface--schemas)
7. [Key Design Patterns](#key-design-patterns)

---

## Backend Layer Structure

### Layer 1: HTTP Entry Point (`scripts/api/main.py`)
**Purpose**: FastAPI application bootstrap, lifespan management, middleware configuration, static UI serving.

**Key Components**:
- **Lifespan Manager**: `lifespan()` async context manager
  - On startup: Configure logging, ensure directories, register SSE hook, start heartbeat background task
  - On shutdown: Gracefully drain ThreadPoolExecutor, cancel in-flight jobs
  - Wires SSE broadcaster into job store via `register_sse_hook()`
  - Signature: `(event_type: str, job_dict: dict) -> None` (sync callback for thread-safe emission)

- **Request ID Middleware**: `request_id_middleware()`
  - Assigns UUID to each request via `X-Request-ID` header
  - Stored in context variable via `set_request_id()` for distributed tracing

- **CORS Configuration**: Allows origins (localhost:5173 for Vite dev, localhost:8501 for Streamlit, GitHub Pages static frontend)

- **Private Network Access**: Handles Chrome preflight for GitHub Pages → local API communication

- **Static UI Serving** (experimental): Serves `/ui/static/index.html` at root and `/app`

**Imports**:
```python
from scripts.api.routes import router
from scripts.core.config import cfg, configure_logging
from scripts.core.logging_utils import get_logger, set_request_id
from scripts.core.paths import ensure_directories
from scripts.api.routers.events import start_heartbeat, broadcaster
from scripts.api.jobs import register_sse_hook, _executor
```

**Exports**:
- `app: FastAPI` — The ASGI application instance

---

### Layer 2: Route Aggregator (`scripts/api/routes.py`)
**Purpose**: Combine 11 domain-specific routers into a single APIRouter that main.py includes.

**Included Routers** (in order):
1. `system` — Health checks, system info
2. `library` — Song library CRUD + stats
3. `downloads` — Single/batch/playlist downloads, Spotify import
4. `stems` — Demucs stem separation, batch ops
5. `analysis` — BPM/key detection, compatibility scoring, recommendations
6. `remix` — DJ transition rendering, chain remixing, preview
7. `generative` — Style transfer, inpainting, tokenization, AI model status
8. `spotify` — Spotify integration endpoints
9. `jobs` — Job status polling, cancellation, retry
10. `crates` — User-defined song collections (CRUD on `/crates`)
11. `events` — Server-sent events stream + heartbeat

**Architecture**:
```python
router = APIRouter()
router.include_router(system.router)      # GET /health/live, GET /health/ready
router.include_router(library.router)     # GET /library, GET /library/{name}, …
router.include_router(downloads.router)   # POST /download, POST /download-batch, …
router.include_router(stems.router)       # POST /stems/split, POST /stems/split-batch
router.include_router(analysis.router)    # POST /analyze, GET /recommend/{name}, …
router.include_router(remix.router)       # POST /dj-remix, POST /dj-chain, …
router.include_router(generative.router)  # POST /ai/style-transfer, POST /ai/inpaint, …
router.include_router(spotify.router)     # Spotify auth/import routes
router.include_router(jobs.router)        # GET /jobs, GET /jobs/{id}, DELETE /jobs/{id}
router.include_router(crates.router)      # Full CRUD /crates, /crates/{id}/songs
router.include_router(events.router)      # GET /events/stream (SSE), heartbeat
router.include_router(setlist.router)     # Set builder endpoints
```

---

### Layer 3: Domain Routers (in `scripts/api/routers/`)
Each router exposes endpoints, calls task functions, and returns job creation responses (`{ job_id }`).

**Example Flow (Download Router)**:
```
POST /download
  ↓
request body: DownloadRequest(query, name, separate)
  ↓
create_job(JobType.DOWNLOAD, meta={...})
  ↓
submit_job(job_id, download_task, query=..., separate=...)
  ↓
return { job_id }
```

Router responsibilities:
- **Request validation** (via Pydantic schemas)
- **Job creation** (SQLite + in-memory cache)
- **Async task submission** (ThreadPoolExecutor)
- **Response formatting** ({ job_id } for async, direct result for sync endpoints)

---

### Layer 4: Task Modules (in `scripts/api/task_modules/`)
Long-running CPU-bound operations executed in ThreadPoolExecutor workers.

**Key Task Functions**:
- `download_task()` — YouTube Music fetch, audio extraction, M4A→WAV conversion
- `stems_task()` — Demucs stem separation (vocals/drums/bass/other)
- `remix_task()` — DJ transition rendering (beat-grid lock, stem crossfade, mastering)
- `analysis_task()` — Song structure analysis (beat tracking, section labeling, BPM detection)
- `style_transfer_task()` — MusicGen-style audio synthesis
- `inpaint_task()` — VampNet bridge beat generation

**Task Pattern**:
```python
def task_fn(arg1, arg2, job_id):
    job = job_store.get_job(job_id)
    try:
        job_store.update_job(job_id, status=RUNNING, message="Processing...")
        result = core_module.process(arg1, arg2)  # Call core module
        job_store.update_job(job_id, status=DONE, result={...})
    except Exception as e:
        job_store.update_job(job_id, status=FAILED, error=str(e))
```

---

### Layer 5: Job Store (`scripts/api/jobs.py`)
**Purpose**: Durable async job queue with SQLite persistence and in-memory cache.

**Architecture**:
- **Write-through cache**: In-memory `_jobs` dict + SQLite `jobs.db` table
- **Executor**: `ThreadPoolExecutor(max_workers=2)` named `remixmate-worker`
- **Job lifecycle**:
  ```
  PENDING → RUNNING → DONE (or FAILED/CANCELLED)
  ```
- **Durability**: On process restart, jobs marked RUNNING are rolled back to FAILED

**Public API**:
```python
create_job(job_type: JobType, meta: dict) -> str               # Returns job_id
get_job(job_id: str) -> Optional[Dict]
list_jobs(limit: int = 50) -> List[Dict]
update_job(job_id, *, status=None, progress=None, ...)        # Write-through to SQLite
submit_job(job_id, fn, **kwargs) -> None                       # Enqueue in ThreadPoolExecutor
cancel_job(job_id) -> bool                                     # Mark CANCELLED
retry_job(job_id, fn, **kwargs) -> Optional[str]               # Clone job, return new_id
register_sse_hook(callback) -> None                            # Wire callback for SSE emit
job_to_response(job: Dict) -> JobResponse                      # Normalize to Pydantic model
```

**SSE Integration**:
```python
_sse_hook: Optional[Callable] = None
# Registered at startup via register_sse_hook(_sync_emit)
# Called from update_job() when status/progress changes
# _sync_emit schedules broadcaster.broadcast() to all connected clients
```

**Database Schema**:
```sql
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    job_type TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    progress REAL NOT NULL DEFAULT 0.0,
    message TEXT NOT NULL DEFAULT '',
    result_json TEXT,
    error TEXT,
    eta_sec INTEGER,
    meta_json TEXT NOT NULL DEFAULT '{}'
)
```

---

### Layer 6: Pydantic Schemas (`scripts/api/schemas.py`)
**Purpose**: Centralized request/response models for API contract + OpenAPI docs.

**Key Model Groups**:

**1. Enums**:
```python
JobStatus = "pending" | "running" | "done" | "failed" | "cancelled"
JobType = "download" | "dj_remix" | "separate" | "analyze"
```

**2. Job Models**:
```python
JobResponse(
    job_id: str,
    status: JobStatus,
    job_type: JobType,
    created_at: float,
    started_at: Optional[float],
    finished_at: Optional[float],
    progress: float,           # 0.0–1.0
    message: str,
    result: Optional[Dict],
    error: Optional[str],
    eta_sec: Optional[int]
)
```

**3. Request Models**:
- `DownloadRequest(query, name, separate)`
- `BatchDownloadRequest(queries[], separate)`
- `PlaylistDownloadRequest(url, separate, limit)`
- `DJRemixRequest(song_a, song_b, transition_bars, preset, bridge_beat_*)`
- `DJChainRequest(songs[], transition_bars, ...)`
- `DJPreviewRequest(song_a, song_b, transition_bars, transition_effect)`
- `StemSplitRequest(song, enhance, model)`
- `CompatibilityRequest(song_a, song_b, artist_a, artist_b)`
- `AnalyzeRequest(song)`
- `StyleTransferRequest(song_name, description, duration_sec, ...)`
- `InpaintRequest(song_a, song_b, mask_type, ...)`
- `TokenizeRequest(song_name, codec, bandwidth)`

**4. Response Models**:
- `SongInfo(name, size_mb, has_full_wav, stems[], license_type, source, last_accessed)`
- `CompatibilityResult(song_a, song_b, compatible, overall, bpm_score, key_score, energy_score, bpm_a, bpm_b, camelot_a, camelot_b, genre_a, genre_b)`
- `GenreResult(song, genre, confidence, runner_up, bpm)`
- `MusicVectorResponse(key, mode, camelot, key_confidence, danceability, vocal_density, spectral_centroid_hz, chord_sequence, drop_position_sec)`
- `TransitionScoreResponse(overall, beat_alignment, harmonic_match, energy_smoothness, vocal_clash, camelot_a/b, key_a/b, recommended_transition_bars, notes)`
- `QualityReportResponse(lufs_integrated, lufs_target, lufs_gain_applied, peak_dbfs, has_clipping, clip_count, dynamic_range_db, passed, notes)`
- `LibraryStats(total_songs, total_size_gb, cap_gb, within_cap, songs_with_stems)`
- `SimilarSongResult(name, score, bpm, key, mode, camelot, genre, danceability, vocal_density, breakdown)`
- `SimilarSongsResponse(source, similar[], engine)`
- `ModelStatusResponse(device, mps_allocated_gb, max_vram_gb, models[])`

---

### Layer 7: Core Modules (in `scripts/core/`)

#### `dj_engine.py` — Transition Rendering
**Purpose**: Apply planned metadata to actual audio samples and produce final crossfade.

**Key Data Structures** (imported from `dj_analysis.py`):
```python
@dataclass Beat:
    index: int              # 0-based beat number
    time: float             # seconds from start
    bar: int                # which bar this beat falls in
    beat_in_bar: int        # 1–4 position within bar

@dataclass Section:
    type: str               # "intro" | "verse" | "chorus" | "drop" | "break" | "outro" | "build"
    start_bar: int
    end_bar: int
    start_time: float
    end_time: float
    avg_energy: float       # 0–1
    avg_spectral: float     # 0–1 brightness
    
    @property is_mixable_exit() -> bool
    @property is_mixable_entry() -> bool

@dataclass SongStructure:
    bpm: float
    duration: float         # seconds
    beats: List[Beat]
    bars: List[Tuple[float, float]]       # (start_s, end_s)
    sections: List[Section]

@dataclass EQPlan:
    # Envelope curves for bass/mids/highs during crossfade
    # Specified as (time_offset, gain_db) keypoints

@dataclass TransitionPlan:
    exit_bar_a: int                   # Which bar Song A exits
    entry_bar_b: int                  # Which bar Song B enters
    transition_bars: int              # Overlap duration (typically 16)
    stretch_ratio: float              # Time-stretch factor for Song B
    exit_time_a: Optional[float]      # Seconds in Song A's timeline
    entry_time_b: Optional[float]     # Seconds in Song B's timeline
```

**Public Functions**:
```python
analyze_structure(audio: np.ndarray, sr: int) -> SongStructure
    # Beat tracking, section labeling, BPM detection

plan_transition(structure_a: SongStructure, structure_b: SongStructure) -> TransitionPlan
    # Compute mix points, harmonic compatibility, EQ curves

class DJEngine:
    def render(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        plan: TransitionPlan,
        sr: int
    ) -> np.ndarray
        # Apply beat-grid lock, stem-aware crossfade, EQ, mastering
        # Return final mixed audio
```

**DSP Helpers**:
```python
_butter_highpass(cutoff_hz, sr, order=4) -> Tuple  # scipy.signal.butter coefficients
_butter_lowpass(cutoff_hz, sr, order=4) -> Tuple
```

**Config Fallbacks** (from `scripts.core.config`):
```python
_HP_START_HZ     = 400.0      # High-pass filter start frequency
_HP_END_HZ       = 80.0       # High-pass filter end frequency
_BASS_CROSSOVER  = 150.0      # Bass/mids crossover point
```

---

#### `dj_analysis.py` — Structure Analysis
**Purpose**: Analyze raw audio to extract beat grids, sections, and transition plans.

**Imports**:
```python
from scripts.core.config import cfg  # Read default_transition_bars, hp_filter settings
```

**Public Functions**:
```python
analyze_structure(audio: np.ndarray, sr: int) -> SongStructure
    # Librosa-based beat tracking + tempogram peak detection for BPM
    # Onset detection + novelty function segmentation for sections
    # Return SongStructure with beats, bars, sections

plan_transition(structure_a: SongStructure, structure_b: SongStructure) -> TransitionPlan
    # Choose harmonic-compatible mix points
    # Compute stretch_ratio = bpm_a / bpm_b
    # Find bar-aligned exit and entry points
    # Generate EQ curves for bass/mids/highs crossfade
```

---

#### `library.py` — Library Management
**Purpose**: Smart song inventory with deduplication, LRU eviction, audio fingerprinting.

**Key Data Structure**:
```python
@dataclass SongEntry:
    name: str
    path: str                           # str(song_dir)
    size_bytes: int = 0
    has_full_wav: bool = False
    stems_available: List[str] = []     # ['vocals', 'drums', 'bass', 'other']
    fingerprint: Optional[str] = None   # SHA-256 of first 30s PCM
    last_accessed: float                # time.time()
    source: Optional[str] = None        # "youtube", "jamendo", etc.
    license_type: Optional[str] = None
```

**Public Methods** (LibraryManager class):
```python
class LibraryManager:
    __init__(library_dir=None, max_size_gb=20.0)
    
    touch(name: str) -> None
        # Mark song as accessed (update last_accessed timestamp)
    
    get_size_gb() -> float
        # Total library disk usage
    
    prune_raw(name: str) -> None
        # Delete full.wav if stems exist (save ~60% space)
    
    evict_lru() -> None
        # When library > max_size_gb, remove full.wav files for LRU songs
        # Stems are preserved so remixing still works
    
    add_song_entry(entry: SongEntry) -> None
    get_entry(name: str) -> Optional[SongEntry]
    list_all() -> List[SongEntry]
    
    fingerprint_audio(audio_path: Path) -> str
        # SHA-256 of first 30 seconds PCM
    
    is_duplicate(fingerprint: str) -> bool
        # Check if fingerprint already in index
```

**Config** (from `scripts.core.config`):
```python
_MAX_SIZE_GB: float = 20.0              # Default library size cap
_KEEP_RAW: bool = False                 # Keep full.wav after separation
_PRUNE_ON_DL: bool = True               # Auto-prune after download
```

---

#### Other Core Modules (Summarized)
- `mastering.py` — ITU-R BS.1770-4 LUFS metering + true-peak limiter (outputs −14 LUFS)
- `beat_synth.py` — Procedural bridge beat generator (6 genre presets: techno/house/hiphop/trap/dnb/ambient)
- `setlist_planner.py` — Weighted greedy set optimizer (Camelot wheel, BPM, energy arc)
- `music_index.py` — 35-dimensional embedding index (JSON-persisted, no FAISS)
- `style_transfer.py` — AI style transfer via DAC tokens
- `inpainting.py` — Audio inpainting via VampNet
- `stems.py` — Demucs stem separation wrapper
- `paths.py` — Canonical path constants (outputs/, models/, data/, library/)
- `config.py` — YAML config loader (config.yaml → config.local.yaml → env vars)

---

## Frontend Layer Structure

### Layer 1: Entry Point & Routing (`frontend/src/App.tsx`)
**Purpose**: Top-level router that branches into shell (main UI) or widget (floating PiP).

```tsx
<Routes>
  <Route path="/widget" element={<Widget />} />
  <Route path="*" element={<AppShell />} />
</Routes>
```

**Imports**:
```typescript
import { Suspense, lazy } from 'react'
import { Routes, Route } from 'react-router-dom'
import { AppShell } from './shell/AppShell'
const Widget = lazy(() => import('@/pages/Widget'))
```

---

### Layer 2: App Shell — 3-Zone Layout (`frontend/src/shell/AppShell.tsx`)
**Purpose**: Main layout container with LeftRail + Canvas + RightInspector.

**Architecture**:
```
┌─────────────────────────────────────────┐
│         AppShell (CSS Grid)             │
├────┬──────────────────────────┬─────────┤
│    │                          │         │
│Left│   Router Canvas (8 pages)│Inspector│
│Rail│                          │(jobs/   │
│    │     • mission-control    │activity)│
│ 64 │     • library-atlas      │         │
│ px │     • mix-deck           │  316px  │
│    │     • set-builder        │variable │
│    │     • signal-search      │         │
│    │     • ai-lab             │         │
│    │     • mix-vault          │         │
│    │     • operations         │         │
│    │                          │         │
└────┴──────────────────────────┴─────────┘
```

**Key Hooks & Features**:
```tsx
export function AppShell() {
  // SSE connection (persists across page navigation)
  useSSE()
  
  // Polling fallback when SSE unavailable
  const sseConnected = useAppStore((s) => s.sseConnected)
  useJobPoller(!sseConnected)
  
  // Job completion toast notifications
  useJobToasts()
  
  // Keyboard shortcuts (Cmd+K for help modal)
  useKeyboardShortcuts()
  
  // Error boundaries per page
  const inspectorOpen = useAppStore((s) => s.inspectorOpen)
  const [showShortcuts, setShowShortcuts] = useState(false)
}
```

**Lazy-Loaded Pages** (code-split):
```tsx
const MissionControl = lazy(() => import('@/pages/MissionControl'))
const LibraryAtlas   = lazy(() => import('@/pages/LibraryAtlas'))
const MixDeck        = lazy(() => import('@/pages/MixDeck'))
const SetBuilder     = lazy(() => import('@/pages/SetBuilder'))
const SignalSearch   = lazy(() => import('@/pages/SignalSearch'))
const AILab          = lazy(() => import('@/pages/AILab'))
const MixVault       = lazy(() => import('@/pages/MixVault'))
const Operations     = lazy(() => import('@/pages/Operations'))
```

**Imports**:
```typescript
import { useAppStore } from '@/stores/appStore'
import { useSSE } from '@/hooks/useSSE'
import { useJobPoller } from '@/hooks/useJobPoller'
import { useJobToasts } from '@/hooks/useJobToasts'
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'
import { ToastProvider } from '@/components/Toast'
import { PageErrorBoundary } from '@/components/PageErrorBoundary'
```

---

### Layer 3: Global State Store (`frontend/src/stores/appStore.ts`)
**Purpose**: Zustand store for navigation, connection health, live jobs, activity log, inspector state.

**Key State Interfaces**:
```typescript
interface AppState {
  // Navigation
  activeNav: NavDestination
  setActiveNav: (dest: NavDestination) => void
  
  // Health & Connection
  apiHealth: 'unknown' | 'ok' | 'degraded' | 'down'
  sseConnected: boolean
  uptimeSeconds: number
  machineProfile: MachineProfile | null
  setApiHealth: (h: HealthStatus['status'] | 'unknown') => void
  setSseConnected: (v: boolean) => void
  setUptimeSeconds: (v: number) => void
  setMachineProfile: (p: MachineProfile) => void
  
  // Live Jobs (from SSE or polling)
  jobs: Record<string, Job>    // keyed by job_id
  upsertJob: (job: Job) => void
  removeJob: (id: string) => void
  setJobs: (jobs: Job[]) => void
  
  // Completion Log (timestamps of COMPLETED jobs, last 60 min)
  completionLog: number[]
  logCompletion: () => void
  
  // Activity Log (recent SSE events, human-readable)
  activityLog: ActivityEntry[]
  pushActivity: (entry: Omit<ActivityEntry, 'id' | 'ts'>) => void
  clearActivity: () => void
  
  // Inspector Panel
  inspectorTab: 'jobs' | 'activity' | 'system'
  setInspectorTab: (tab: AppState['inspectorTab']) => void
  inspectorOpen: boolean
  toggleInspector: () => void
}

interface ActivityEntry {
  id: string
  ts: string                    // ISO timestamp
  level: 'info' | 'success' | 'warn' | 'error'
  message: string
  job_id?: string
}
```

**Convenience Selectors**:
```typescript
selectActiveJobs(state) -> Job[]        // Filter by RUNNING | PENDING
selectRecentJobs(state) -> Job[]        // Sort by updated_at, take 20
```

---

### Layer 4: TypeScript Types (`frontend/src/types/index.ts`)
**Purpose**: Shared type definitions mirroring FastAPI schemas.

**Key Types**:

**Jobs**:
```typescript
type JobStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED'

interface Job {
  job_id: string
  status: JobStatus
  type: string
  progress: number              // 0–100 (normalized from API's 0–1)
  message: string
  created_at: string            // ISO
  updated_at: string            // ISO
  result?: Record<string, unknown>
  error?: string
  meta?: Record<string, unknown>
}
```

**Library**:
```typescript
interface SongInfo {
  name: string
  path: string
  has_stems: boolean
  has_analysis: boolean
  bpm?: number
  key?: string
  camelot?: string
  duration?: number             // seconds
  genre?: string
  energy?: number               // 0–1
  embedding?: number[]
  stems?: string[]              // ['vocals', 'drums', 'bass', 'other']
}

interface LibraryStats {
  total_songs: number
  indexed_songs: number
  stems_split: number
  total_size_mb: number
}
```

**Analysis & Mixing**:
```typescript
interface TransitionPlan {
  exit_bar_a: number
  entry_bar_b: number
  transition_bars: number
  stretch_ratio: number
  exit_time_a?: number          // seconds in Song A's timeline
  entry_time_b?: number         // seconds in Song B's timeline
  key_compatible?: boolean
}

interface CompatibilityResult {
  song_a: string
  song_b: string
  compatible: boolean
  overall: number               // 0–1 composite score
  bpm_score: number             // 0–1
  key_score: number             // 0–1
  energy_score: number          // 0–1
  bpm_a: number
  bpm_b: number
  camelot_a?: string
  camelot_b?: string
  genre_a?: string
  genre_b?: string
  transition_plan?: TransitionPlan
}

interface DJRemixRequest {
  song_a: string
  song_b: string
  transition_bars?: number
  preset?: string
  transition_effect?: string
  bridge_beat_mode?: string
  bridge_beat_genre?: string
  bridge_beat_intensity?: number
}
```

**Navigation**:
```typescript
type NavDestination =
  | 'mission-control'
  | 'library-atlas'
  | 'mix-deck'
  | 'set-builder'
  | 'signal-search'
  | 'ai-lab'
  | 'mix-vault'
  | 'operations'
  | 'widget'
```

**SSE Events**:
```typescript
type SSEEventType =
  | 'heartbeat'
  | 'job_created'
  | 'job_updated'
  | 'job_completed'
  | 'job_failed'
  | 'job_cancelled'
  | 'library_changed'
  | 'system_status'

interface SSEEvent<T = unknown> {
  type: SSEEventType
  data: T
  ts: string                    // ISO timestamp
}
```

---

### Layer 5: API Client (`frontend/src/lib/api.ts`)
**Purpose**: Thin fetch wrapper for all API endpoints, job normalization, type safety.

**Base URL Resolution**:
```typescript
const BASE = import.meta.env.VITE_API_BASE || '/api'
// In dev: Vite proxy rewrites /api/* → http://localhost:8000/*
// In static builds: VITE_API_BASE points to local API, e.g. http://localhost:8000
```

**Generic Request Function**:
```typescript
async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<T>
  // Throws ApiError on non-2xx or timeout
  // Automatic JSON serialization/deserialization
```

**HTTP Helpers**:
```typescript
const get  = <T>(path: string) => request<T>('GET', path)
const post = <T>(path: string, body?: unknown) => request<T>('POST', path, body)
const del  = <T>(path: string) => request<T>('DELETE', path)
const patch = <T>(path: string, body?: unknown) => request<T>('PATCH', path, body)
```

**Job Normalization** (converts REST shape → frontend shape):
```typescript
export function normalizeJob(raw: RawJob): Job
  // Maps status values: "done" → "COMPLETED"
  // Converts progress: 0–1 → 0–100
  // Converts timestamps: epoch → ISO
  // Result: canonical frontend Job type
```

**API Namespaces** (first 150 lines shown):
```typescript
export const healthApi = {
  live:  () => get<HealthStatus>('/health/live'),
  ready: () => get<HealthStatus>('/health/ready'),
}

export const libraryApi = {
  list:    () => get<{ songs: SongInfo[] }>('/library?per_page=5000'),
  get:     (name: string) => get<SongInfo>(`/library/${encodeURIComponent(name)}`),
  delete:  (name: string) => del<void>(`/library/${encodeURIComponent(name)}`),
  stats:   () => get<LibraryStats>('/library/init'),
  initRun: (opts: Record<string, unknown>) => post<{ job_id: string }>('/library/init', opts),
}

export const downloadApi = {
  single:      (query: string, name?: string) =>
    post<RawJob>('/download', { query, name }).then(normalizeJob),
  batch:       (queries: string[]) =>
    post<RawJob[]>('/download-batch', { queries }).then((js) => js.map(normalizeJob)),
  fromSpotify: (url: string) =>
    post<{ job_id: string }>('/spotify/import', { url }),
  playlist:    (url: string, limit?: number) =>
    post<RawJob>('/download-playlist', { url, limit }).then(normalizeJob),
}
```

---

## Data Flow Architecture

### Request → API → Task → Update Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Frontend (React)                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Page Component calls api.downloadApi.single(query)               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                   POST /download
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│ Backend (FastAPI)                                                │
├─────────────────────────────────────────────────────────────────┤
│ downloads.router                                                 │
│   @router.post("/download")                                      │
│   async def handle_download(req: DownloadRequest):               │
│     job_id = job_store.create_job(JobType.DOWNLOAD, ...)        │
│     job_store.submit_job(job_id, download_task, query=...)      │
│     return { job_id }  ← immediate 202 Accepted               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │ ThreadPoolExecutor.submit()         │
        │ download_task(query, job_id)        │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │ task_modules.download_task()        │
        │   job_store.update_job(            │
        │     status=RUNNING,                │
        │     progress=10%,                  │
        │     message="Fetching from YouTube"│
        │   )  ← writes to jobs.db + RAM    │
        │       triggers SSE broadcast       │
        │   audio = youtube_download(query) │
        │   full_wav_path = convert_m4a()   │
        │   if separate:                     │
        │     demucs_stems(full_wav_path)   │
        │   job_store.update_job(            │
        │     status=DONE,                   │
        │     result={path, stems}           │
        │   )                                │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────────────┐
        │ Job Store → SQLite                         │
        │   UPDATE jobs SET                          │
        │     status='done',                         │
        │     result_json='{...}',                   │
        │     finished_at=1234567.89                 │
        │   WHERE job_id='...'                       │
        └─────────────────┬──────────────────────────┘
                          │
        ┌─────────────────▼──────────────────────────┐
        │ SSE Hook: _sync_emit()                     │
        │   Schedules broadcaster.broadcast(         │
        │     'job_completed',                       │
        │     {job_id, status, result, ...}          │
        │   )                                        │
        └─────────────────┬──────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│ Browser (EventSource)                                            │
├─────────────────────────────────────────────────────────────────┤
│ GET /events/stream (Server-Sent Events)                          │
│   event: job_completed                                           │
│   data: {job_id, status, result, ...}                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────▼──────────────────────────┐
        │ useSSE() hook                              │
        │   on message (SSE event):                  │
        │     normalizeJob() → canonical Job type   │
        │     appStore.upsertJob(job)  ← Zustand   │
        │     pushActivity(message)                 │
        └─────────────────┬──────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│ UI Re-render (React Components)                                  │
├─────────────────────────────────────────────────────────────────┤
│ useAppStore((s) => s.jobs[job_id])  ← Hook selector            │
│   re-renders page showing updated progress                       │
│ useJobToasts()                                                   │
│   creates toast notification "Download complete"                │
└─────────────────────────────────────────────────────────────────┘
```

### Alternative Flow: Job Polling (SSE Unavailable)

```
Frontend (useJobPoller hook)
  ↓
  Every 2–5s: GET /jobs, GET /jobs/{id}
  ↓
  normalizeJob() → canonical Job type
  ↓
  appStore.upsertJob(job)
  ↓
  UI re-renders with updated progress
```

---

## Feature-to-Endpoint Mapping

### Feature: Download Track from YouTube Music

**Frontend Initiation**:
- Page: `Operations.tsx`
- Component: `<DownloadForm />`
- Action: `downloadApi.single(query, name)`

**Backend Route**:
- Router: `scripts/api/routers/downloads.py`
- Endpoint: `POST /download`
- Request: `DownloadRequest(query, name, separate=True)`
- Response: `{ job_id }`

**Task Execution**:
- Task: `scripts/api/task_modules/download.py :: download_task()`
- Updates: Job progress (0→100%), message, result on completion
- Calls:
  - `youtube_dl` or `yt-dlp` wrapper → M4A file
  - `ffmpeg` → WAV conversion
  - (if `separate`) `Demucs` stem separation → vocals, drums, bass, other

**Job Store**:
- Table: `jobs` in `data/jobs.db`
- Persists: job_id, status, job_type, progress, result, error

**SSE Broadcast**:
- Event: `job_updated` (progress 0–100%)
- Final: `job_completed` (with result path)

**Frontend Reception**:
- Hook: `useSSE()` in `AppShell.tsx`
- Fallback: `useJobPoller()` every 2s
- State: `appStore.upsertJob(job)`
- Display: `RightInspector` jobs tab, `useJobToasts()` notification

---

### Feature: Mix Two Songs with DJ Transition

**Frontend Initiation**:
- Page: `MixDeck.tsx`
- Component: `<RemixForm />`
- Action: `remixApi.dj(request)` or `remixApi.preview(request)`

**Backend Route**:
- Router: `scripts/api/routers/remix.py`
- Endpoint: `POST /dj-remix` (full) or `POST /dj-remix/preview` (transition only)
- Request: `DJRemixRequest(song_a, song_b, transition_bars, preset, bridge_beat_mode, ...)`
- Response: `{ job_id }`

**Task Execution**:
- Task: `scripts/api/task_modules/remix.py :: remix_task()`
- Flow:
  1. Load Song A + Song B from library (full.wav or stems)
  2. Call `dj_analysis.analyze_structure(audio_a, sr)` → `SongStructure`
  3. Call `dj_analysis.analyze_structure(audio_b, sr)` → `SongStructure`
  4. Call `dj_analysis.plan_transition(struct_a, struct_b)` → `TransitionPlan`
  5. Apply stem-aware crossfade (if stems exist)
  6. Optional: synthesize bridge beat via `beat_synth` module
  7. Apply EQ curves (bass/mids/highs fade independently)
  8. Master output to −14 LUFS via `mastering.py`
  9. Write output WAV to `outputs/` directory
  10. Update job result with output path + metadata

**Core Modules Called**:
- `dj_analysis.py` — beat/section detection, transition planning
- `dj_engine.py` — rendering, beat-grid lock, crossfade
- `beat_synth.py` — optional bridge beat generation
- `mastering.py` — LUFS metering + limiting
- `stems.py` — load Demucs stems if available

**Job Store**:
- Type: `REMIX`
- Progress: 0% (startup) → 25% (structure analysis) → 50% (DSP rendering) → 75% (mastering) → 100% (done)
- Result: `{ output_path: "/path/to/mix.wav", duration_sec: 120.5, lufs: -14.0, ... }`

**SSE Broadcast**:
- Events: `job_updated` (progress), then `job_completed`

**Frontend Reception**:
- State: `appStore.upsertJob(job)`
- Display: `MixDeck.tsx` progress bar, waveform preview
- Download: `MixVault.tsx` fetches output file via `GET /mix/{job_id}/download`

---

### Feature: Search Similar Songs

**Frontend Initiation**:
- Page: `SignalSearch.tsx`
- Component: `<SimilaritySearch />`
- Action: `libraryApi.similar(song_name)`

**Backend Route**:
- Router: `scripts/api/routers/analysis.py`
- Endpoint: `GET /library/similar/{name}`
- Response: `SimilarSongsResponse(source, similar[], engine)`

**Core Module**:
- `music_index.py` — 35-dimensional embedding index
- Compares query song embedding against library using cosine similarity
- Returns k=10 nearest neighbors with per-dimension breakdowns (BPM, key, energy, rhythm, timbre)

**Data Structure**:
```typescript
interface SimilarSongResult {
  name: string
  score: float              // 0–1 cosine similarity
  bpm?: number
  key?: string
  camelot?: string
  genre?: string
  breakdown?: {             // Per-dimension scores
    bpm_sim: float
    key_sim: float
    energy_sim: float
    rhythm_sim: float
    timbre_sim: float
  }
}
```

**Frontend Reception**:
- Page: `SignalSearch.tsx`
- Display: Rings showing multi-dimensional similarity breakdown
- Action: Click result to add to `SetBuilder`

---

### Feature: Build a DJ Set

**Frontend Initiation**:
- Page: `SetBuilder.tsx`
- Components: `<SongPool />`, `<OrderedSet />`, `<EnergyCurve />`
- Actions:
  1. Drag songs from pool to ordered set
  2. Auto-optimize via `setlist_planner`
  3. Chain remix all

**Backend Route** (optimization):
- Router: `scripts/api/routers/setlist.py`
- Endpoint: `POST /setlist/optimize`
- Request: `{ songs: [names], preset: "camelot" | "energy" }`
- Response: `{ optimized_order: [names], energy_arc: [(bar, energy_level), ...] }`

**Core Module**:
- `setlist_planner.py` — weighted greedy optimizer
  - Minimizes Camelot key shift (±1 semitone preferred)
  - Builds energy arc (low → peak → low)
  - Sorts by BPM adjacency

**Chain Remix Endpoint**:
- Router: `scripts/api/routers/remix.py`
- Endpoint: `POST /dj-chain`
- Request: `DJChainRequest(songs[], transition_bars, bridge_beat_mode, ...)`
- Response: `{ job_id }`

**Task Execution**:
- Task: `scripts/api/task_modules/remix.py :: chain_remix_task()`
- For each adjacent pair (song_i, song_{i+1}):
  1. Render transition (same as DJ remix, but without master output)
  2. Concatenate transition + song_{i+1} head
- Final output: concatenated set with seamless transitions

**Frontend Reception**:
- Display: `SetBuilder` energy curve, order changes
- Download: Final set file from `MixVault`

---

## Module Dependencies & Import Graph

### Backend Import Tree

```
scripts/api/main.py
  ├── scripts.api.routes
  │   └── [All 11 routers]
  │       └── scripts.api.jobs (each router uses for job creation)
  │           ├── scripts.api.schemas (JobResponse, JobStatus, JobType)
  │           └── scripts.core.paths (DATA_DIR, job persistence)
  │
  ├── scripts.core.config (cfg, configure_logging)
  │   └── YAML loader + env var override
  │
  ├── scripts.core.logging_utils (get_logger, set_request_id)
  │
  ├── scripts.core.paths (ensure_directories)
  │
  └── scripts.api.routers.events
      ├── start_heartbeat() → background task
      └── broadcaster → SSE emission to clients

scripts/api/routers/remix.py
  ├── scripts.api.jobs (job creation, progress updates)
  ├── scripts.api.schemas (DJRemixRequest, DJPreviewRequest, JobResponse)
  ├── scripts.api.task_modules.remix (remix_task)
  └── scripts.core.paths (output directory)

scripts/api/task_modules/remix.py
  ├── scripts.api.jobs (update_job)
  ├── scripts.core.dj_analysis (analyze_structure, plan_transition)
  ├── scripts.core.dj_engine (DJEngine class)
  ├── scripts.core.beat_synth (optional bridge beat)
  ├── scripts.core.mastering (LUFS metering + limiting)
  ├── scripts.core.stems (load Demucs stems if available)
  ├── scripts.core.library (LibraryManager, song loading)
  └── soundfile (WAV I/O)

scripts/core/dj_engine.py
  ├── scripts.core.dj_analysis (SongStructure, TransitionPlan, Beat, Section, EQPlan)
  │   └── [Data classes + analysis functions]
  ├── scipy.signal (butter filters for EQ)
  ├── numpy (DSP operations)
  └── scripts.core.config (optional HP/bass filter settings)

scripts/core/dj_analysis.py
  ├── numpy
  ├── librosa (beat tracking, onset detection, spectral features)
  ├── scipy (novelty function segmentation)
  └── scripts.core.config (default_transition_bars, etc.)

scripts/core/library.py
  ├── hashlib (SHA-256 fingerprinting)
  ├── json (index.json persistence)
  ├── soundfile, numpy (audio I/O, fingerprint computation)
  └── scripts.core.config (max_size_gb, keep_raw, prune_on_dl)

scripts/api/schemas.py
  └── pydantic (BaseModel, Field, Enum)
      [No internal imports — self-contained]

scripts/api/jobs.py
  ├── sqlite3 (jobs.db persistence)
  ├── concurrent.futures (ThreadPoolExecutor)
  ├── scripts.api.schemas (JobResponse, JobStatus, JobType)
  └── scripts.core.logging_utils (get_logger, set_job_id)
```

### Frontend Import Tree

```
frontend/src/App.tsx
  ├── react-router-dom (Routes, Route)
  └── shell/AppShell.tsx
      └── [Pages + Hooks + Stores]

shell/AppShell.tsx
  ├── useSSE() hook
  │   ├── lib/api.ts (EVENTS_URL for EventSource)
  │   └── stores/appStore.ts (upsertJob, pushActivity, setSseConnected)
  │
  ├── useJobPoller() hook
  │   ├── lib/api.ts (jobsApi.list())
  │   └── stores/appStore.ts (setJobs)
  │
  ├── useJobToasts() hook
  │   ├── stores/appStore.ts (selectActiveJobs)
  │   └── components/Toast.tsx (createToast)
  │
  └── shell/LeftRail.tsx
      ├── useAppStore (activeNav, setActiveNav)
      └── [NavItem components]

pages/MixDeck.tsx
  ├── lib/api.ts (analysisApi.compatibility, remixApi.dj, remixApi.preview)
  ├── stores/appStore.ts (useAppStore)
  ├── types/index.ts (CompatibilityResult, DJRemixRequest, Job)
  ├── components/DualDeck.tsx (visualization)
  ├── components/RemixForm.tsx (parameter input)
  └── hooks/useJob.ts (poll job status)

pages/LibraryAtlas.tsx
  ├── lib/api.ts (libraryApi.list, libraryApi.similar)
  ├── types/index.ts (SongInfo, SimilarTrack)
  ├── components/SongTable.tsx (sortable, filterable)
  ├── components/SimilarityBreakdown.tsx (radar chart)
  └── stores/appStore.ts (favorites via metadata)

pages/SignalSearch.tsx
  ├── lib/api.ts (libraryApi.similar)
  ├── types/index.ts (SimilarTrack)
  └── components/ScoreRings.tsx (multi-dimensional similarity viz)

pages/SetBuilder.tsx
  ├── lib/api.ts (remixApi.chain, setlistApi.optimize)
  ├── types/index.ts (DJChainRequest)
  ├── components/SongPool.tsx
  ├── components/OrderedSet.tsx
  ├── components/EnergyCurve.tsx
  └── stores/appStore.ts (upsertJob for chain progress)

pages/AILab.tsx
  ├── lib/api.ts (aiApi.styleTransfer, aiApi.inpaint, aiApi.tokenize, aiApi.models)
  ├── types/index.ts (StyleTransferRequest, InpaintRequest, TokenizeRequest)
  └── components/[various AI controls]

pages/MixVault.tsx
  ├── lib/api.ts (mixApi.list, mixApi.download, healthApi.live)
  ├── components/AudioPlayer.tsx
  └── components/MetadataPanel.tsx

pages/Operations.tsx
  ├── lib/api.ts (downloadApi.single, downloadApi.batch, downloadApi.fromSpotify, stemsApi.split)
  ├── components/DownloadForm.tsx
  ├── components/BatchUpload.tsx
  └── stores/appStore.ts (job monitoring)

stores/appStore.ts
  ├── zustand (create store)
  └── types/index.ts (Job, NavDestination, HealthStatus, MachineProfile, ActivityEntry)

lib/api.ts
  ├── types/index.ts (All API request/response types)
  └── [No component dependencies]

types/index.ts
  └── [Pure TypeScript — no imports]
```

---

## API Surface & Schemas

### Request/Response Pair Examples

**Download Single Track**:
```
POST /download
Request:  DownloadRequest(query: str, name: Optional[str], separate: bool = True)
Response: { job_id: str }  [202 Accepted]

Poll:     GET /jobs/{job_id}
Response: JobResponse(
  status: JobStatus,
  progress: 0.0–1.0,
  result: { path: str, stems: [str], ... }
)

SSE:      event: job_completed
          data: { job_id, status, result }
```

**Analyze Song Structure**:
```
POST /analyze
Request:  AnalyzeRequest(song: str)
Response: { job_id: str }

Result:   {
  bpm: float,
  sections: [{ type, start_bar, end_bar, avg_energy, avg_spectral }],
  beats: [{ index, time, bar, beat_in_bar }]
}
```

**Check Compatibility**:
```
POST /compatibility
Request:  CompatibilityRequest(song_a: str, song_b: str, ...)
Response: CompatibilityResult(
  compatible: bool,
  overall: 0.0–1.0,
  bpm_score: 0.0–1.0,
  key_score: 0.0–1.0,
  energy_score: 0.0–1.0,
  camelot_a: Optional[str],
  camelot_b: Optional[str],
  transition_plan: Optional[TransitionPlan]
)
```

**Render DJ Transition**:
```
POST /dj-remix
Request:  DJRemixRequest(
  song_a: str,
  song_b: str,
  transition_bars: int = 16,
  preset: str = "auto",
  bridge_beat_mode: str = "none",
  ...
)
Response: { job_id: str }

Result:   {
  output_path: str,
  duration_sec: float,
  lufs: float,
  metadata: { bpm_a, bpm_b, key_a, key_b, ... }
}
```

**Server-Sent Events Stream**:
```
GET /events/stream
Heartbeat (10s):  event: heartbeat
                  data: { uptime_seconds, active_jobs, machine_profile }

Job Update:       event: job_updated
                  data: { job_id, status, progress, message }

Job Complete:     event: job_completed
                  data: { job_id, status, result, ... }

Library Change:   event: library_changed
                  data: { added: [names], removed: [names], updated: [names] }
```

---

## Key Design Patterns

### 1. Job Queue Pattern (ThreadPoolExecutor + SQLite)

**Problem**: Long-running tasks (5–60s) block FastAPI event loop.

**Solution**:
- Submit task to `ThreadPoolExecutor(max_workers=2)`
- Return job UUID immediately (202 Accepted)
- Client polls `/jobs/{id}` or subscribes to SSE for updates
- Job state persisted to SQLite (survives process restart)
- Progress updates broadcast via SSE

**Benefit**: Non-blocking API, durable state, live progress streaming.

---

### 2. Stem-Aware Crossfade (Independent Envelope Curves)

**Problem**: Naive audio crossfade (linear mix) sounds artificial when bass/vocals clash.

**Solution**:
- If Demucs stems exist, fade each stem independently:
  - Drums/bass from Song B enter early (establish groove)
  - Vocals from Song B enter late (avoid overlap clash)
  - Low-shelf EQ pulls Song A's bass as Song B's rises
  
**Benefit**: Intentional, pro-sounding transitions; feels less "automated."

---

### 3. Beat-Grid Lock (Sample-Level Alignment)

**Problem**: Time-stretching to match BPM can phase-shift beat boundaries.

**Solution**:
- Compute bar-grid phases at cue points
- Apply sample-level correction (clamped to ±half bar)
- Entry sample index compensates for stretch ratio: `entry_sample_b = int(entry_time_b * sr / stretch_ratio)`

**Benefit**: Beat alignment within samples; DJ-quality sync.

---

### 4. Write-Through Cache (Job Store)

**Problem**: High-frequency job status reads need to be fast; writes must be durable.

**Solution**:
- In-memory `_jobs` dict for O(1) reads
- Every mutation (status/progress change) immediately flushed to SQLite
- On startup, table loaded back into RAM

**Benefit**: O(1) reads (fast API responses), ACID guarantees (durable on crash).

---

### 5. SSE with Fallback Polling

**Problem**: Network flakiness, older browsers, shared hosting may block SSE.

**Solution**:
- Primary: `useSSE()` subscribes to `GET /events/stream`
- Fallback: `useJobPoller()` polls `/jobs` every 2s if SSE disconnects
- Same normalized data structure (Job type) fed to Zustand

**Benefit**: Always-connected UI; works in constrained environments.

---

### 6. Page Lazy-Loading with Error Boundaries

**Problem**: Large component bundle causes initial paint delay; one error crashes entire app.

**Solution**:
- Each page in AppShell is lazy-loaded: `const Page = lazy(() => import('@/pages/Page'))`
- Each page wrapped in `<PageErrorBoundary>` with fallback UI
- Suspense fallback shows spinner

**Benefit**: Fast initial load; isolated error handling per page.

---

### 7. Zustand Store + React Query Pattern

**Problem**: State vs. server cache; Zustand for client state (nav, jobs, health), React Query for server data (library, analysis results).

**Solution**:
- **Zustand** (`appStore`): Navigation state, live jobs from SSE, connection health
- **React Query** (hooks in pages): Song library, analysis results, cached API responses

**Benefit**: Clear separation of concerns; efficient cache invalidation.

---

### 8. API Client with Type-Safe Request/Response

**Problem**: Manual fetch calls lack type safety; hard to keep frontend/backend schemas in sync.

**Solution**:
- All API calls funneled through `lib/api.ts`
- Each namespace (libraryApi, remixApi, etc.) defines request/response types
- Job normalization (`normalizeJob()`) converts REST shape (0–1 progress, epoch timestamps) to frontend shape (0–100, ISO strings)

**Benefit**: One place to catch API contract breaks; IDE autocomplete; compile-time safety.

---

### 9. Config with YAML + Env Var Overrides

**Problem**: Hardcoding settings is inflexible; secrets should not be in repo.

**Solution**:
- Base: `config.yaml` (in repo)
- Local override: `config.local.yaml` (gitignored)
- Env vars: `REMIXMATE_<SECTION>_<KEY>` override both

**Example**:
```yaml
# config.yaml
dj:
  default_transition_bars: 16
  hp_filter_start_hz: 400.0
```

```bash
# .env
REMIXMATE_DJ_DEFAULT_TRANSITION_BARS=32
```

**Benefit**: Safe for CI/CD, local dev, production without credential exposure.

---

### 10. Music Embedding Index (35-D, No FAISS)

**Problem**: FAISS is overkill for small library; slow startup on edge hardware (Raspberry Pi, old laptops).

**Solution**:
- Extract 35-dimensional feature vector per song:
  - BPM, key, energy, spectral centroid, danceability, vocal density, etc.
- Store as JSON (human-readable, no binary deps)
- Similarity: cosine distance between vectors
- Index rebuild via `POST /index/rebuild` (async job)

**Benefit**: No C++ deps; JSON is portable; simple cosine similarity is fast enough for <10k songs.

---

## Summary Table: Files & Responsibilities

| File | Layer | Responsibility |
|------|-------|---|
| `main.py` | HTTP | FastAPI bootstrap, middleware, lifespan |
| `routes.py` | Router Aggregator | Combines 11 domain routers |
| `routers/*.py` | Domain Routes | Request validation, job creation |
| `task_modules/*.py` | Async Tasks | CPU-bound work (5–60s) |
| `jobs.py` | Job Store | SQLite + in-memory cache, ThreadPoolExecutor |
| `schemas.py` | Data Models | Pydantic request/response shapes |
| `dj_engine.py` | Core DSP | Transition rendering, beat-grid lock |
| `dj_analysis.py` | Core Analysis | Beat tracking, section labeling, planning |
| `library.py` | Core Data | Song inventory, fingerprinting, LRU eviction |
| `config.py` | Config | YAML + env var loader |
| `App.tsx` | Frontend Entry | Top-level router (shell vs. widget) |
| `AppShell.tsx` | Layout | 3-zone grid, page routing, hook mounting |
| `appStore.ts` | Client State | Zustand: nav, jobs, health, activity |
| `types/index.ts` | Type Defs | Shared TypeScript types |
| `api.ts` | API Client | Fetch wrapper, job normalization, namespaces |
| `pages/*.tsx` | Feature Pages | 8 pages (MissionControl, MixDeck, etc.) |

---

## End-to-End Example: "Mix Two Songs"

**User Action**:
```
MixDeck.tsx: Select "Daft Punk - One More Time" + "Disclosure - Latch"
Click "Remix" button
```

**Frontend**:
```typescript
// MixDeck.tsx
const handleRemix = async (songA, songB) => {
  const response = await remixApi.dj({
    song_a: songA,
    song_b: songB,
    transition_bars: 16,
    bridge_beat_mode: "auto"
  })
  // { job_id: "abc-123" }
  // Component polls /jobs/abc-123 or waits for SSE
}
```

**Backend - Route**:
```python
# routers/remix.py
@router.post("/dj-remix")
async def handle_dj_remix(req: DJRemixRequest):
    job_id = job_store.create_job(JobType.DJ_REMIX, meta=req.dict())
    job_store.submit_job(job_id, remix_task, song_a=req.song_a, song_b=req.song_b, ...)
    return { "job_id": job_id }  # 202 Accepted
```

**Backend - Task**:
```python
# task_modules/remix.py
def remix_task(song_a, song_b, job_id, **kwargs):
    job_store.update_job(job_id, status=RUNNING, progress=0, message="Loading...")
    
    # Load audio
    audio_a, sr = librosa.load(library.song_dir(song_a) / "full.wav")
    audio_b, sr = librosa.load(library.song_dir(song_b) / "full.wav")
    
    job_store.update_job(job_id, progress=25, message="Analyzing structure...")
    
    # Analyze
    struct_a = dj_analysis.analyze_structure(audio_a, sr)
    struct_b = dj_analysis.analyze_structure(audio_b, sr)
    
    job_store.update_job(job_id, progress=50, message="Planning transition...")
    
    # Plan
    plan = dj_analysis.plan_transition(struct_a, struct_b)
    
    job_store.update_job(job_id, progress=75, message="Rendering mix...")
    
    # Render
    engine = DJEngine()
    output = engine.render(audio_a, audio_b, plan, sr)
    
    # Master
    output, lufs = mastering.master_to_target(output, sr, target_lufs=-14)
    
    # Write output
    output_path = OUTPUT_DIR / f"mix-{job_id[:8]}.wav"
    sf.write(output_path, output, sr)
    
    job_store.update_job(
        job_id,
        status=DONE,
        progress=100,
        result={
            "output_path": str(output_path),
            "duration_sec": len(output) / sr,
            "lufs": lufs,
            "metadata": { "bpm_a": struct_a.bpm, "bpm_b": struct_b.bpm, ... }
        }
    )
```

**Job Store**:
```
jobs.db: INSERT / UPDATE
  job_id='abc-123',
  status='done',
  progress=1.0,
  result_json='{"output_path": "...", "duration_sec": 120.5, ...}',
  finished_at=1700000000.123
```

**SSE**:
```
event: job_updated
data: { job_id: "abc-123", status: "running", progress: 0.25, message: "Analyzing..." }

event: job_updated
data: { job_id: "abc-123", status: "running", progress: 0.75, message: "Rendering mix..." }

event: job_completed
data: { job_id: "abc-123", status: "done", result: { ... }, progress: 1.0 }
```

**Frontend Reception**:
```typescript
// useSSE() hook receives SSE message
// Normalizes and updates Zustand store
appStore.upsertJob({
  job_id: "abc-123",
  status: "COMPLETED",
  progress: 100,
  result: { output_path: "...", duration_sec: 120.5, ... },
  updated_at: "2024-11-15T12:34:56Z"
})

// MixDeck.tsx re-renders with result
// Show download button, metadata panel, audio player
```

**UI Display**:
- MixDeck shows "Mix complete!" notification
- RightInspector jobs tab shows "✓ Mix complete (120.5s, -14 LUFS)"
- MixVault page now lists the output file
- Click to download or play preview

---

**End of Architecture Map**
