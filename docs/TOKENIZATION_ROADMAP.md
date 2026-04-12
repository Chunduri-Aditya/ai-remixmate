# AI RemixMate: Music Tokenization Integration Roadmap

**Version:** 1.0
**Date:** March 2026
**Duration:** 10 weeks
**Target:** Production-grade music tokenization and generative remix capabilities

---

## Executive Summary

This roadmap outlines a phased approach to integrate advanced music tokenization and generative capabilities into AI RemixMate. The implementation progresses from enhanced harmonic analysis through neural codec tokenization to full generative remixing with production hardening. The architecture leverages state-of-the-art models (EnCodec, DAC, VampNet, MusicGen) while maintaining efficient resource utilization and clean API design.

**Key Outcomes:**
- Semantic music representation via neural codecs
- Generative remix capabilities with creative transitions
- Production-grade stability with VRAM management
- 6 new API endpoints + 1 new UI page
- ~25 KB additional tokens per song library entry

---

## Phase 1: Enhanced Harmonic Analysis
**Duration:** Week 1-2
**Effort:** 40 hours
**GPU Budget:** CPU only (librosa)

### Objectives
- Replace basic Krumhansl-Schmuckler key detection with CQT chroma-based analysis
- Apply Harmonic-Percussive Source Separation (HPSS) preprocessing
- Upgrade vector space from 35-dim to incorporate CQT chroma features
- Improve key detection accuracy for diverse genres (EDM, hip-hop, classical)

### Deliverables

#### New Module: `scripts/core/key_detection.py`
**Purpose:** Advanced harmonic analysis pipeline

```python
# Core components:
# - CQTAnalyzer: Constant-Q transform feature extraction
# - HPSSPreprocessor: Harmonic-Percussive source separation
# - ChromaKeyDetector: Chroma-based key detection (replaces Krumhansl)
# - KeyConfidenceEstimator: Multi-frame confidence scoring
# - VectorSpaceUpgrade: 35-dim → 64-dim (35 existing + 24 CQT chroma + 5 confidence metrics)
```

**Key Dependencies:**
- librosa >= 0.10.0 (for CQT and HPSS)
- numpy >= 1.24.0
- scipy >= 1.10.0

**Algorithm Details:**
- CQT with bins_per_octave=24 (10 octaves, C1-C11)
- HPSS margin parameter: 2.0 (default librosa)
- Chroma normalization: L2-norm for stability
- Key estimation: Correlation with 24 major/minor profiles (Gomez 2006)
- Confidence: Cross-correlation peak sharpness

#### Files to Modify

**`music_intelligence.py`**
- Add `HarmonicAnalyzer` class
- Integrate HPSS preprocessing into audio loading pipeline
- Update `get_key_confidence()` to return tuple: (key, mode, confidence_scores)
- Backward compatibility: Maintain existing API, add `use_legacy=False` parameter

**`music_index.py`**
- Extend vector representation from 35→64 dimensions
- Migration function: `upgrade_legacy_vectors()`
- Schema change: Add columns `cqt_chroma_features` (24-dim float32), `confidence_metrics` (5-dim float32)
- Index regeneration tool: `python -m scripts.tools.reindex_harmonic_features`

#### API & Output Format

```python
# music_intelligence.analyze_harmony(audio_path: str) → dict
{
    "key": "C",
    "mode": "major",
    "confidence": 0.94,
    "cqt_chroma": [0.12, 0.05, 0.88, ...],  # 24-dim
    "confidence_metrics": {
        "profile_correlation": 0.94,
        "sharpness": 0.87,
        "multi_frame_agreement": 0.91,
        "chromatic_purity": 0.89,
        "energy_consistency": 0.85
    },
    "harmonic_complexity": 0.63,  # 0-1 scale
    "processing_time_ms": 342
}
```

#### Testing Strategy
- Unit tests: 12 test cases (golden reference audio files, edge cases)
- Regression tests: Validate 35-dim features remain unchanged
- Accuracy benchmark: Compare against Essentia key detector on standard dataset (MedleyDB)
- Performance: < 500ms per 3-minute song on CPU

#### Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Performance regression on large libraries | High | Implement optional async reindexing; cache results |
| CQT feature instability for noisy audio | Medium | Add preprocessing filter; validate on diverse genres |
| Compatibility with existing search indices | High | Comprehensive migration script; fallback to legacy vectors |
| Memory usage during HPSS | Medium | Process in chunks; stream-based implementation |

#### Success Metrics
- Key detection accuracy: ≥90% on validation set (MedleyDB)
- Processing speed: ≤500ms per song
- Vector quality: ≥95% cosine similarity consistency
- Zero breaking changes to existing API

---

## Phase 2: Neural Codec Tokenization
**Duration:** Week 3-4
**Effort:** 60 hours
**GPU Budget:** ~500 MB (EnCodec) or ~1 GB (DAC)

### Objectives
- Tokenize audio stems using neural codecs (EnCodec or DAC)
- Store discrete tokens in library filesystem
- Enable future generative workflows with discrete audio tokens
- Support both lossy (EnCodec) and high-fidelity (DAC) codec chains

### Deliverables

#### New Module: `scripts/core/codec_tokens.py`
**Purpose:** Neural codec tokenization pipeline

```python
# Core components:
# - CodecFactory: Unified interface for EnCodec/DAC selection
# - StemTokenizer: Per-stem tokenization after Demucs
# - TokenStorage: Efficient token serialization (.npy, .safetensors)
# - TokenDecoder: Reconstruct audio from tokens (validation)
# - TokenStatistics: Compression ratio, entropy metrics
```

**Codec Selection Matrix:**

| Codec | Bandwidth | Quality | Speed | Inference Memory |
|-------|-----------|---------|-------|------------------|
| EnCodec (6 kbps) | 6 kbps | Good (lossy) | Fast | ~500 MB |
| EnCodec (24 kbps) | 24 kbps | Very good | Fast | ~500 MB |
| DAC (9 kbps) | 9 kbps | Excellent (near-lossless) | Slower | ~1 GB |

**Recommended:** EnCodec (24 kbps) for balance; DAC optional for mastering-grade projects

**Key Dependencies:**
- encodec >= 0.1.1 (Facebook Research)
- hydra-core >= 1.3.0 (for EnCodec config)
- safetensors >= 0.3.0 (for token serialization)
- Or: audiocraft-musicgen >= 1.0.0 (includes encodec)

#### Files to Modify

**`music_intelligence.py`**
- Add `CodecTokenizer` class
- Integrate tokenization into post-Demucs pipeline
- Async tokenization support for large batches
- Return token stats in analysis output

**`music_index.py`**
- Add columns: `codec_type` (str), `codec_tokens_path` (str), `token_shape` (tuple), `compression_ratio` (float)
- Index query: `find_by_token_similarity(seed_tokens, top_k=10)`
- Token-level search API (TBD for Phase 3)

**New file: `scripts/utils/token_serializer.py`**
- Serialization: Token arrays → .npz or .safetensors
- Deserialization with version control
- Compression: Optional gzip for storage efficiency

#### Directory Structure

```
library/
├── {song_name}/
│   ├── audio/
│   │   ├── original.wav
│   │   ├── stems.npz          # Demucs output (drums, bass, other, vocals)
│   │   └── stems_resampled/
│   │       ├── drums_44.1k.wav
│   │       ├── bass_44.1k.wav
│   │       ├── other_44.1k.wav
│   │       └── vocals_44.1k.wav
│   ├── tokens/                # NEW
│   │   ├── metadata.json      # Codec params, token shapes
│   │   ├── drums_tokens.safetensors
│   │   ├── bass_tokens.safetensors
│   │   ├── other_tokens.safetensors
│   │   └── vocals_tokens.safetensors
│   ├── analysis/
│   └── metadata.json
```

#### API Endpoints

**POST /tokenize**
```python
# Request
{
    "song_name": "uptown_funk",
    "codec": "encodec_24k",      # or "dac_9k"
    "recompute": false            # Skip if exists
}

# Response (201)
{
    "song_name": "uptown_funk",
    "codec": "encodec_24k",
    "token_shapes": {
        "drums": [75000, 8],      # [time_steps, codebook_dim]
        "bass": [75000, 8],
        "other": [75000, 8],
        "vocals": [75000, 8]
    },
    "compression_ratio": 0.12,    # 12% of original WAV size
    "storage_bytes": 2400000,
    "processing_time_seconds": 45,
    "token_paths": {
        "drums": "library/uptown_funk/tokens/drums_tokens.safetensors",
        ...
    }
}
```

**GET /library/{name}/tokens**
```python
# Response (200)
{
    "song_name": "uptown_funk",
    "codec_type": "encodec_24k",
    "token_stats": {
        "drums": {
            "shape": [75000, 8],
            "entropy": 6.2,        # bits per token
            "unique_codes": 240
        },
        ...
    },
    "total_tokens": 300000,
    "storage_bytes": 2400000,
    "created_at": "2026-03-15T10:30:00Z"
}
```

#### Tokenization Pipeline

```
Audio File (44.1 kHz)
    ↓
[Demucs Separation] → {drums, bass, other, vocals}.wav
    ↓
[Resample to codec sample rate]
    ↓
[EnCodec/DAC Encode] → Discrete token sequences
    ↓
[Serialize & Store] → tokens/
    ↓
[Validate] → Reconstruct audio, compute SI-SDR
```

#### Testing Strategy
- Unit tests: 8 test cases (different codecs, stem combinations)
- Integration tests: Full tokenization pipeline for 5 reference songs
- Validation: Reconstruct audio from tokens, measure SI-SDR ≥ 15 dB (EnCodec)
- Storage efficiency: Verify compression ratio ~10-15%
- Concurrent tokenization: Test 4 songs in parallel without OOM

#### Token Storage Considerations

**Per-song overhead:**
- Metadata: ~5 KB
- 4 stems × 75k tokens × 8 bytes/token ≈ 2.4 MB
- **Total per song:** ~2.5 MB (compressed ~0.3 MB with gzip)

**Scalability:**
- 1000-song library: ~2.5 GB uncompressed, ~300 MB compressed
- Incremental tokenization: Only new songs, ~1-2 min per song

#### Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Out of memory (GPU) during tokenization | High | Process stems sequentially, add OOM handler |
| Token reconstruction quality varies by codec | Medium | Validate each codec; store reference metrics |
| Storage bloat for large libraries | Medium | Implement compression; optional token quantization |
| Codec library version incompatibility | High | Pin dependencies; store codec version in metadata |

#### Success Metrics
- Tokenization time: ≤2 min per song (GPU)
- Token reconstruction SI-SDR: ≥15 dB (EnCodec 24 kbps)
- Storage compression: 10-15% of original size
- API availability: 99.5% uptime
- No data loss during serialization

---

## Phase 3: Generative Remix
**Duration:** Week 5-8
**Effort:** 120 hours
**GPU Budget:** 4-8 GB per model (sequential loading)

### Objectives
- Implement VampNet for creative inpainting/transitions between songs
- Add MusicGen melody conditioning for style transfer
- Enable seamless remix generation with harmonic consistency
- Create "AI Remix Lab" UI page for user-facing generation

### Deliverables

#### New Module: `scripts/core/generative_remix.py`
**Purpose:** Generative remix orchestration

```python
# Core components:
# - VampNetInpainter: Bridging transitions between stems
# - MusicGenStyleTransfer: Melody conditioned generation
# - RemixSequencer: Temporal alignment and blending
# - PromptBuilder: Semantic remix descriptions
# - GenerationQueue: Async GPU task management (Phase 4)
```

**Model Stack:**
- **VampNet** (Meta/Facebook): 0.5 B params, code2seq masked token prediction
- **MusicGen** (Meta): 3.9 B params (large), melody/style conditioning
- Both support 30-second sequences at 16 kHz

**Key Dependencies:**
- audiocraft >= 1.0.0 (includes VampNet, MusicGen)
- julius >= 0.2.0 (audio resampling)
- einops >= 0.7.0 (tensor operations)

#### New Module: `scripts/core/remix_synthesis.py`
**Purpose:** Audio-level remix composition

```python
# Core components:
# - StemAligner: Temporal alignment across sources
# - HarmonicBlender: Harmonic-aware crossfading
# - DynamicsCompressor: Level matching between stems
# - MixdownEngine: Multitrack to stereo master
```

#### Files to Modify

**`music_intelligence.py`**
- Add `RemixGenerator` class
- Prompt engineering for VampNet/MusicGen
- Generation quality metrics (SI-SDR, perceptual loss)

**`music_index.py`**
- Add columns: `generation_history` (JSON), `remix_ancestry` (list of song references)

**New file: `scripts/api/generative_routes.py`**
- FastAPI/Flask routes for generation endpoints
- Async task handling with status polling

#### Directory Structure (Updated)

```
library/
├── {song_name}/
│   ├── audio/
│   ├── tokens/
│   ├── generated/          # NEW
│   │   ├── inpaints/
│   │   │   ├── {remix_id}_inpaint.wav
│   │   │   └── {remix_id}_metadata.json
│   │   └── transfers/
│   │       ├── {remix_id}_transfer.wav
│   │       └── {remix_id}_metadata.json
│   └── metadata.json
```

#### Use Case 1: Inpainting (Smooth Transitions)

**Scenario:** Bridge drums from Song A to bass from Song B over 8 seconds

```python
# Workflow
Song A (drums) [0-30s]
    ↓
[VampNet Inpainting] (5-13s transition window)
    ↓
Song B (bass) [13-30s]
```

**API Endpoint: POST /generate/inpaint**

```python
# Request
{
    "source_song": "uptown_funk",
    "source_stem": "drums",              # drums | bass | other | vocals
    "target_song": "levitating",
    "target_stem": "bass",
    "transition_duration_seconds": 8,
    "position_in_source": 15,            # Start inpainting at 15s
    "num_generations": 3,                # Generate 3 variants
    "temperature": 0.9,                  # 0.7-1.5 range
    "top_p": 0.95                        # Nucleus sampling
}

# Response (202 Accepted)
{
    "task_id": "inpaint_abc123def456",
    "status": "queued",
    "estimated_time_seconds": 120,
    "position_in_queue": 2
}

# Polling: GET /tasks/{task_id}
{
    "task_id": "inpaint_abc123def456",
    "status": "completed",
    "results": [
        {
            "variant_idx": 0,
            "audio_path": "library/uptown_funk/generated/inpaints/inpaint_abc123def456_v0.wav",
            "metrics": {
                "si_sdr": 16.3,
                "perceptual_loss": 0.42,
                "genre_consistency": 0.87
            },
            "generation_params": {...}
        },
        ...
    ]
}
```

#### Use Case 2: Style Transfer (Melody Conditioning)

**Scenario:** Transfer melody from Song A to Song B's harmonic context

```python
# Workflow
Song A Melody (extraction or user-provided)
    ↓
[MusicGen Conditioner] (with Song B key/tempo context)
    ↓
Generated audio (Song B style, Song A melody)
```

**API Endpoint: POST /generate/style-transfer**

```python
# Request
{
    "reference_song": "uptown_funk",     # Source melody
    "reference_stem": "vocals",
    "target_song": "levitating",         # Target harmonic context
    "style_strength": 0.8,               # 0.5-1.0 (how much to preserve target style)
    "duration_seconds": 30,
    "num_generations": 2,
    "seed": 42                           # For reproducibility
}

# Response (202 Accepted) → Results follow same polling pattern

{
    "task_id": "transfer_xyz789",
    "status": "completed",
    "results": [
        {
            "variant_idx": 0,
            "audio_path": "library/uptown_funk/generated/transfers/transfer_xyz789_v0.wav",
            "metrics": {
                "melody_preservation": 0.91,
                "harmonic_consistency": 0.88,
                "style_adherence": 0.85
            }
        },
        ...
    ]
}
```

#### UI Page: "AI Remix Lab"

**Location:** `/remix-lab` route

**Features:**
1. **Inpainting Interface**
   - Dual song selector (source/target)
   - Per-stem selection (visual stem cards with waveform previews)
   - Transition duration slider (2-15 seconds)
   - Interactive timeline scrubber for position selection
   - Generation parameters: temperature, top_p, num_variants
   - Real-time task status with progress bar

2. **Style Transfer Interface**
   - Reference song/stem selector
   - Target song selector
   - Style strength slider (visual "preserve reference" ↔ "preserve target")
   - Melody visualization (extracted from reference)
   - Advanced: Manual melody editing

3. **Results Gallery**
   - Grid of generated audio cards
   - Waveform visualization
   - Playback with A/B comparison
   - Metric badges (SI-SDR, perceptual loss, genre consistency)
   - Download button (WAV, MP3)
   - Save to library button (create new remix entry)

4. **History & Sharing**
   - Recent generations (last 24 hours)
   - Remix ancestry tree (visual: Song A + Song B → Generated Remix)
   - Shareable remix links

#### Generation Quality Metrics

```python
# SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
# Measures alignment with reference signal, invariant to amplitude scaling
# Target: ≥15 dB (audibly transparent)

# Perceptual Loss (VGGish-based)
# Euclidean distance in learned audio feature space
# Target: <0.5 (on 0-1 scale)

# Genre/Style Consistency
# Classifier confidence in expected genre for generated audio
# Target: ≥0.85

# Melody Preservation (for style transfer)
# DTW distance between reference and generated pitch contours
# Target: <5% normalized distance
```

#### Orchestration & Load Balancing

```
User Request
    ↓
[Task Queue] (Redis, Phase 4)
    ↓
[GPU Worker] (1-4 GPUs available)
    ├─ Load VampNet model (2 GB, ~10s)
    ├─ Run inference (30-120s depending on model)
    ├─ Cache model for next task
    └─ Return results
    ↓
[Results Storage] → library/{song}/generated/
```

#### Generation Configuration File

**File:** `config/generation_config.yaml`

```yaml
models:
  vampnet:
    model_id: "meta-v1"
    sample_rate: 16000
    max_duration_seconds: 30
    vram_mb: 2048
    config:
      temperature: 0.9
      top_p: 0.95

  musicgen:
    model_id: "facebook/musicgen-large"
    sample_rate: 16000
    max_duration_seconds: 30
    vram_mb: 4096
    conditioning:
      - "description"
      - "melody"
      - "harmony"
    config:
      temperature: 0.85
      top_k: 250

generation:
  max_concurrent_tasks: 2
  timeout_seconds: 300
  quality_metrics:
    - "si_sdr"
    - "perceptual_loss"
    - "style_consistency"
```

#### Testing Strategy

- Unit tests: 6 test cases (inpainting, style transfer, edge cases)
- Integration tests: End-to-end generation (inpainting + style transfer)
- Model validation: Pre-trained weights verification
- Quality benchmarks: Compare against baseline remixes
- Load testing: 5 concurrent generation tasks
- UI testing: Browser compatibility (Chrome, Firefox, Safari)

#### Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Generation quality varies significantly | High | Implement ensemble voting; expose quality metrics to user |
| VRAM explosion (OOM) | High | Strict model lifecycle; implement model unloading; queue management (Phase 4) |
| Long inference times (>2 min) | Medium | Add progressive inference; streaming results; user expectations management |
| Generation not novel (exact memorization) | Medium | Eval on out-of-distribution prompts; add diversity metrics |
| UI rendering lag (large result galleries) | Medium | Lazy loading; pagination; virtual scrolling |

#### Success Metrics

- Inpainting quality: SI-SDR ≥15 dB, <120s generation time
- Style transfer: melody preservation ≥90%, harmonic consistency ≥85%
- Generation diversity: <70% cosine similarity between variants
- UI responsiveness: <100ms interaction latency
- No OOM errors in normal operation

#### Dependencies Added

```
audiocraft>=1.0.0
julius>=0.2.0
einops>=0.7.0
fastapi>=0.104.0 (if not present)
pydantic>=2.0.0 (if not present)
```

---

## Phase 4: Production Hardening
**Duration:** Week 9-10
**Effort:** 80 hours
**GPU Budget:** Same as Phase 3 (efficient reuse)

### Objectives
- Implement robust VRAM management and model caching
- Add async task queue (Celery/Redis) for reliable GPU job processing
- Integrate quality metrics (SI-SDR, ViSQOL) for result validation
- Build A/B comparison UI and monitoring dashboard
- Ensure 99.5% uptime and graceful degradation

### Deliverables

#### New Module: `scripts/core/gpu_manager.py`
**Purpose:** VRAM lifecycle and model caching

```python
# Core components:
# - ModelCache: LRU cache with VRAM tracking
# - VRAMMonitor: Real-time GPU memory tracking
# - ModelLoader: Atomic load/unload with safety checks
# - PreemptionHandler: Graceful interruption on resource contention
```

**Logic:**
```
Model A in VRAM (2 GB)
    ↓ [Task requires Model B (3 GB), only 2 GB available]
    ↓ [Check: Model A not active]
    ├─ Save Model A state (if needed)
    ├─ Unload Model A (free 2 GB)
    ├─ Load Model B (3 GB)
    ├─ Run Model B
    ├─ Unload Model B
    └─ Restore Model A
```

#### New Module: `scripts/core/task_queue.py`
**Purpose:** Async GPU task orchestration

```python
# Core components:
# - TaskQueue: Redis-backed job queue
# - GPUWorker: Celery task executor
# - TaskMonitor: Status tracking and ETA estimation
# - FailureRecovery: Automatic retry with exponential backoff
```

**Architecture:**

```
Web API
  ↓
[FastAPI Request Handler]
  ├─ Validate input
  ├─ Create Task (Redis)
  └─ Return task_id (202)
  ↓
[Task Queue (Redis)]
  ├─ Task A [pending]
  ├─ Task B [pending]
  └─ Task C [pending]
  ↓
[Celery Workers] (1-4 GPU workers)
  ├─ Worker 1: Task A [running]
  ├─ Worker 2: Task B [running]
  ├─ Worker 3: [idle]
  └─ Worker 4: [idle]
  ↓
[Results Storage]
  ├─ Task A [completed] → Results saved
  └─ Task B [completed] → Results saved
```

**Task Status States:**

```
pending → running → processing_output → completed
                 ↓
              failed → retrying → completed/permanent_failure
                 ↓
              cancelled
```

#### New Module: `scripts/core/audio_quality_metrics.py`
**Purpose:** Quality validation pipeline

```python
# Core components:
# - SISDRCalculator: Scale-invariant SDR
# - VisQOLProxy: Perceptual quality (speech/music proxy)
# - IntonationValidator: Pitch stability check
# - HarmonicConsistencyValidator: Key/mode adherence
```

**Validation Flow:**

```
Generated Audio
    ↓
[SI-SDR vs. seed] → Must be ≥15 dB for vocals, ≥12 dB for instruments
    ↓
[ViSQOL proxy] → Must be <0.5 (0=perfect, 1=worst)
    ↓
[Intonation stability] → Must be <5% variance in detected pitch
    ↓
[Harmonic consistency] → Must detect key within 1 semitone
    ↓
[PASS/FAIL] → Metrics attached to result
```

#### Files to Modify

**`app.py` or main server file**
- Add Celery initialization
- Add task status polling routes
- Add quality metrics to response objects
- Error handling for worker failures

**`config/production.yaml` (NEW)**
```yaml
redis:
  host: "localhost"
  port: 6379
  db: 0

celery:
  broker: "redis://localhost:6379/0"
  backend: "redis://localhost:6379/1"
  worker_concurrency: 2           # Adjust per GPU count
  task_time_limit: 300            # 5 minutes max per task
  task_soft_time_limit: 280

gpu:
  monitor_interval_seconds: 10
  vram_safety_margin_mb: 256
  model_cache_size_gb: 8
  enable_model_pooling: true

quality_metrics:
  si_sdr_threshold_vocals: 15.0
  si_sdr_threshold_instruments: 12.0
  visqol_threshold: 0.5
  run_validation: true
  parallel_validation: false      # Avoid VRAM contention
```

#### API Endpoints (Updated)

**GET /tasks/{task_id}** (Enhanced)
```python
{
    "task_id": "inpaint_abc123",
    "status": "completed",
    "progress": {
        "current_step": 10,
        "total_steps": 10,
        "percentage": 100
    },
    "results": [
        {
            "variant_idx": 0,
            "audio_path": "...",
            "metrics": {
                "si_sdr": 16.3,
                "visqol_score": 0.38,
                "intonation_stability": 0.98,
                "harmonic_consistency": 0.91,
                "quality_assessment": "excellent",
                "validation_passed": true
            }
        }
    ],
    "created_at": "2026-03-30T10:00:00Z",
    "completed_at": "2026-03-30T10:02:00Z",
    "duration_seconds": 120
}
```

**POST /comparison (NEW)**
```python
# Create A/B comparison
{
    "name": "remix_comparison_001",
    "description": "VampNet inpaint variants",
    "samples": [
        {
            "label": "Variant A (temp=0.7)",
            "task_id": "inpaint_abc123",
            "variant_idx": 0
        },
        {
            "label": "Variant B (temp=0.9)",
            "task_id": "inpaint_abc123",
            "variant_idx": 1
        }
    ]
}

# Response (201)
{
    "comparison_id": "comp_xyz789",
    "url": "/compare/comp_xyz789"
}
```

**GET /monitoring/gpu (NEW)**
```python
{
    "timestamp": "2026-03-30T10:05:00Z",
    "gpus": [
        {
            "gpu_id": 0,
            "vram_used_mb": 5200,
            "vram_total_mb": 8192,
            "vram_percentage": 63.5,
            "current_model": "musicgen-large",
            "active_task_id": "transfer_xyz789",
            "temperature_celsius": 62
        }
    ],
    "queue": {
        "pending_tasks": 3,
        "running_tasks": 1,
        "average_wait_time_seconds": 45,
        "eta_next_completion": "2026-03-30T10:06:30Z"
    }
}
```

#### UI: Monitoring Dashboard

**Location:** `/admin/monitoring`

**Components:**
1. **Real-time GPU Status**
   - VRAM usage gauge (per GPU)
   - Active task indicator with progress
   - Temperature monitoring
   - Model cache status

2. **Task Queue Metrics**
   - Pending/Running/Completed tasks
   - Average generation time
   - Error rate
   - Queue depth chart (15-minute history)

3. **Quality Metrics Leaderboard**
   - Top-performing variants by SI-SDR
   - Genre-specific performance
   - Model comparison (VampNet vs. MusicGen)

4. **System Health**
   - Worker availability status
   - Redis connectivity
   - Disk usage (for generated files)
   - Error logs and alerts

#### UI: A/B Comparison Page

**Location:** `/compare/{comparison_id}`

**Features:**
- Dual audio players (synchronized)
- Waveform visualization (overlay possible)
- Quality metrics side-by-side
- Voting/preference tracking (for model improvement)
- Download comparison bundle (.zip)

#### Deployment Configuration

**File:** `docker-compose.prod.yaml`

```yaml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s

  celery_worker:
    build: .
    command: celery -A scripts.tasks.worker worker --loglevel=info --concurrency=2
    depends_on:
      - redis
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
    volumes:
      - ./library:/app/library
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]

  api_server:
    build: .
    command: uvicorn scripts.api.main:app --host 0.0.0.0 --port 8000
    depends_on:
      - redis
    ports:
      - "8000:8000"
    volumes:
      - ./library:/app/library
      - ./logs:/app/logs
    environment:
      - REDIS_URL=redis://redis:6379

volumes:
  redis_data:
```

#### Testing Strategy

- Unit tests: Task queue (10 tests), VRAM manager (8 tests), quality metrics (6 tests)
- Integration tests: Full pipeline with mock Celery worker
- Load tests: 10 concurrent tasks, verify no VRAM leaks
- Failure scenario tests: Worker crash, Redis disconnect, OOM handling
- UI tests: Monitoring dashboard responsiveness, A/B comparison accuracy

#### Monitoring & Alerting

**Prometheus Metrics (NEW):**
```
remixmate_gpu_vram_used_bytes
remixmate_gpu_vram_allocated_bytes
remixmate_task_queue_length
remixmate_task_duration_seconds (histogram)
remixmate_generation_quality_si_sdr (histogram)
remixmate_worker_health (1=healthy, 0=unhealthy)
remixmate_redis_connection_errors_total
```

**Logging:**
- Centralized logs to `/logs` directory
- Log rotation: 100 MB per file, keep 10 files
- Structured logging (JSON) for easy parsing
- Task-level logging with unique task_id

#### Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Redis data loss (crash) | High | Enable RDB + AOF persistence; backups |
| Worker deadlock | High | Task time limits; health checks; auto-restart |
| Model cache coherence | Medium | Atomic load/unload; versioning; checksum validation |
| Monitoring overhead | Low | Metrics batching; 10s update intervals |

#### Success Metrics

- 99.5% uptime (≤3.6 hours downtime/month)
- Task queue latency: p50 <10s, p99 <60s
- No VRAM leaks: Consistent memory after 100 sequential tasks
- Quality validation: 95%+ of generated audio passes metrics
- A/B comparison agreement: >85% inter-rater reliability

---

## Cross-Phase Dependencies

### Dependency Graph
```
Phase 1 (Harmonic Analysis)
    ↓
Phase 2 (Codec Tokens) ← Depends on Phase 1 for metadata
    ↓
Phase 3 (Generative Remix) ← Depends on Phase 1, 2
    ↓
Phase 4 (Production Hardening) ← Spans all phases
```

### Data Migration Strategy

**Phase 1→2 Transition:**
- No breaking changes to Phase 1 APIs
- Optional tokenization per song
- Existing songs continue to work without tokens

**Phase 2→3 Transition:**
- Tokens must exist for inpainting (auto-tokenize if missing)
- Fallback to non-tokenized generation if insufficient VRAM

**Phase 3→4 Transition:**
- Task queue becomes mandatory for generation
- Backward compatibility: Sync API calls queued internally
- No end-user experience changes

### Shared Infrastructure

**Vector Database:** Phase 1's 64-dim vectors enable Phase 3 remix suggestions
**Stem Library:** Phase 2 reuses Demucs stems from library initialization
**Model Zoo:** Centralized model management (Phase 4) across all phases

---

## Technical Specifications

### Supported Audio Formats
- **Input:** WAV (16/24 bit), MP3, FLAC, OGG
- **Output:** WAV (24-bit), MP3 (320 kbps)
- **Streaming:** HLS for real-time preview (Phase 3+)

### Sample Rates
- Analysis: Native sample rate (44.1 kHz typical)
- Demucs: Resampled to 44.1 kHz
- Tokenization: Codec-specific (16 kHz for EnCodec/DAC)
- Generation: 16 kHz (upsampled to 44.1 kHz for output)

### Latency Targets

| Operation | Target | Budget |
|-----------|--------|--------|
| Key detection | <500 ms | CPU |
| Tokenization (4 stems) | 2-3 min | GPU |
| Inpainting generation | 30-120 s | GPU |
| Style transfer generation | 60-180 s | GPU |
| A/B comparison upload | <5 s | Disk I/O |

### Storage Estimates

**Per 1000-song library:**
- Stems (Demucs): ~500 GB
- Codec tokens (compressed): ~300 MB
- Generated remixes (30 variants/song): ~150 GB
- Metadata & indices: ~50 MB
- **Total: ~650 GB** (scales linearly)

### Compute Requirements

**CPU:**
- Minimum: 4 cores, 8 GB RAM
- Recommended: 8 cores, 16 GB RAM

**GPU:**
- Minimum: 1× RTX 3080 (10 GB VRAM) → 1 concurrent task
- Recommended: 2× RTX 4090 (24 GB each) → 4 concurrent tasks
- Excellent: 4× RTX 4090 → 8+ concurrent tasks (data center grade)

**Storage:**
- Minimum: 1 TB SSD (OS + libraries)
- Recommended: 4 TB NVMe RAID-1 (high-I/O workloads)

---

## Success Criteria & KPIs

### Phase 1
- Key detection accuracy: ≥90% (vs. ground truth)
- API stability: Zero breaking changes
- Performance: ≤500ms per song

### Phase 2
- Token reconstruction quality: SI-SDR ≥15 dB
- Storage efficiency: 10-15% compression ratio
- Integration: Zero downtime for library queries

### Phase 3
- Generation quality: SI-SDR ≥15 dB (vocals), ≥12 dB (instruments)
- User experience: <200ms UI latency during generation
- Diversity: <70% cosine similarity between variants
- Coverage: Support 50+ language prompts for MusicGen

### Phase 4
- System uptime: 99.5%
- Task queue latency: p99 <60s
- VRAM stability: No leaks after 100+ sequential tasks
- Monitoring: <5% CPU overhead from metrics collection

---

## Risk Register & Mitigation Plans

### Critical Risks

**1. Model Size & VRAM Explosion**
- **Impact:** Unable to run concurrent generation tasks
- **Probability:** High
- **Mitigation:** Phase 4 strict VRAM management; quantization (8-bit models); model distillation as fallback

**2. Audio Quality Regression**
- **Impact:** Generated audio perceived as lower quality than baseline
- **Probability:** Medium
- **Mitigation:** Extensive A/B testing; human evaluation panel; quality metrics validation

**3. Ecosystem Dependency Risk**
- **Impact:** Meta discontinues audiocraft support or introduces breaking changes
- **Probability:** Low-Medium
- **Mitigation:** Pin exact versions; maintain fork of critical components; community-maintained alternatives (Stable Audio, Jukebox)

### Operational Risks

**4. GPU Worker Crashes**
- **Impact:** Task data loss, user experience degradation
- **Probability:** Medium
- **Mitigation:** Automatic restart; task retry logic; persistent task queue (Redis AOF)

**5. Library Data Corruption**
- **Impact:** Loss of index, tokens, or generated audio
- **Probability:** Low
- **Mitigation:** Automated backups (daily); integrity checks (SHA256 hashing); version control for metadata

### Business/UX Risks

**6. Generation Times Too Long**
- **Impact:** Users abandon generative features due to latency
- **Probability:** Medium
- **Mitigation:** Progressive inference; streaming audio preview; set expectations upfront

---

## Testing Strategy

### Test Coverage by Phase

**Phase 1:** 12 unit tests, 5 integration tests, 1 accuracy benchmark
**Phase 2:** 8 unit tests, 4 integration tests, 1 codec validation suite
**Phase 3:** 6 unit tests, 8 integration tests, 2 subjective quality evaluations
**Phase 4:** 15 unit tests, 10 integration tests, 3 load tests

### Continuous Integration

**Pipeline:**
```
Code commit
    ↓
[Unit tests] (5 min)
    ↓
[Integration tests] (15 min)
    ↓
[Build Docker image] (5 min)
    ↓
[Deploy to staging] (2 min)
    ↓
[Smoke tests] (5 min)
    ↓
[Approved] → Deploy to production
```

### Gold Standard Reference Data

Maintain repository of audio files with ground truth for:
- Key detection (MedleyDB subset, 50 songs)
- Tokenization quality (reference SI-SDR scores)
- Generation quality (human-rated baseline remixes)

---

## Rollout & Deployment Strategy

### Phased Production Release

**Phase 1 Rollout:**
- Week 2: Deploy to internal testing
- Week 3: Gradual rollout (10% → 50% → 100% users)

**Phase 2 Rollout:**
- Week 4: Optional tokenization (non-blocking if fails)
- Weeks 4-5: Background tokenization of library
- Week 6: Enable token-based search (Phase 3 prep)

**Phase 3 Rollout:**
- Week 5: Close beta (50 power users)
- Week 7: Limited release (feature flag)
- Week 8: Full release with "AI Remix Lab" page

**Phase 4 Rollout:**
- Week 9: Celery queue as optional backend
- Week 10: Migrate all generation to queue
- Weeks 10+: Monitoring dashboards available to operators

### Rollback Plan

- Phase 1: Revert to legacy Krumhansl key detection (0 min downtime)
- Phase 2: Disable tokenization endpoint, keep existing tokens (5 min)
- Phase 3: Disable generation endpoints, keep UI (10 min)
- Phase 4: Disable queue, run sync inference (30 min)

---

## Documentation & Knowledge Transfer

### Deliverables per Phase

- **Code Documentation:** Docstrings + module README.md files
- **API Documentation:** OpenAPI/Swagger specs + human-readable guides
- **Architecture Diagrams:** Mermaid/Lucidchart diagrams in `/docs`
- **Operational Runbooks:** Troubleshooting guides for each phase
- **Configuration Guides:** env.example files with explanations

### Knowledge Transfer Sessions

- Phase 1 handoff: 2-hour walkthrough + Q&A (internal team)
- Phase 2 handoff: 2-hour session + token format spec review
- Phase 3 handoff: 4-hour session (generation API design, model capabilities)
- Phase 4 handoff: 3-hour session (operations, monitoring, incident response)

---

## Budget & Resource Allocation

### Engineering Hours by Phase

| Phase | Analysis | Development | Testing | Documentation |
|-------|----------|-------------|---------|---------------|
| 1 | 8 h | 20 h | 10 h | 2 h |
| 2 | 12 h | 35 h | 10 h | 3 h |
| 3 | 20 h | 70 h | 20 h | 10 h |
| 4 | 15 h | 50 h | 12 h | 3 h |
| **Total** | **55 h** | **175 h** | **52 h** | **18 h** |

**Total Project Effort:** ~300 engineering hours (≈7.5 weeks at 40 h/week)

### Hardware Investment

- **Phase 1-2:** 1× GPU (RTX 3080, $700) for tokenization
- **Phase 3:** 2× GPU (RTX 4090, $4000 each) for generation
- **Phase 4:** Monitoring infrastructure (Redis, logging) ~$200/month cloud costs

---

## Future Enhancements (Post-Phase 4)

### Potential Phase 5 Additions

- **Fine-tuning:** Train MusicGen on user library for personalized generation
- **MIDI Export:** Generate MIDI representations of remixes
- **Collaborative Remixing:** Multi-user real-time remix editing
- **Video Sync:** Automatic video remix generation with stems
- **Licensning Integration:** Automatic rights management for generated content

---

## Appendix A: Dependencies & Version Pins

```
# Core
librosa==0.10.0
numpy==1.24.0
scipy==1.10.0

# Codecs & Generative
audiocraft>=1.0.0
encodec>=0.1.1
julius>=0.2.0

# API & Async
fastapi>=0.104.0
pydantic>=2.0.0
celery>=5.3.0
redis>=5.0.0

# Storage & Serialization
safetensors>=0.3.0
pyyaml>=6.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0

# Monitoring
prometheus-client>=0.17.0
```

---

## Appendix B: Configuration Template

```yaml
# config/tokenization_config.yaml

analysis:
  cqt:
    bins_per_octave: 24
    n_octaves: 10
    threshold_db: -120
  hpss:
    margin: 2.0
    power: 2.0

codecs:
  default: "encodec_24k"
  options:
    encodec_24k:
      bandwidth: "24k"
      sample_rate: 16000
    dac_9k:
      model_path: "meta-av/dac-musicgen"
      sample_rate: 16000

generation:
  vampnet:
    temperature: 0.9
    top_p: 0.95
    top_k: 0
  musicgen:
    temperature: 0.85
    top_k: 250
    top_p: 0.95

quality_thresholds:
  si_sdr_vocals: 15.0
  si_sdr_instruments: 12.0
  visqol: 0.5
```

---

**Document Version:** 1.0
**Last Updated:** 2026-03-30
**Maintainer:** AI RemixMate Engineering Team
**Status:** Ready for Development Kickoff
