"""
scripts/api/schemas.py — Pydantic request/response models for the RemixMate API.

All JSON shapes in and out of the API are defined here. Keeping them separate
from the route logic makes it easy to generate OpenAPI docs and write tests.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    DOWNLOAD   = "download"
    DJ_REMIX   = "dj_remix"
    SEPARATE   = "separate"
    ANALYZE    = "analyze"


# ---------------------------------------------------------------------------
# Shared / utility
# ---------------------------------------------------------------------------

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
    overall: float = Field(..., ge=0.0, le=1.0)
    bpm_score: float = Field(..., ge=0.0, le=1.0)
    key_score: float = Field(..., ge=0.0, le=1.0)
    energy_score: float = Field(..., ge=0.0, le=1.0)
    bpm_a: float
    bpm_b: float
    camelot_a: Optional[str] = None
    camelot_b: Optional[str] = None
    genre_a: Optional[str] = None
    genre_b: Optional[str] = None


class GenreResult(BaseModel):
    song: str
    genre: str
    confidence: float
    runner_up: Optional[str] = None
    bpm: Optional[float] = None


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    job_type: JobType
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: float = Field(0.0, ge=0.0, le=1.0, description="0.0–1.0")
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    eta_sec: Optional[int] = Field(None, description="Estimated seconds remaining")


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class DownloadRequest(BaseModel):
    query: str = Field(..., description="YouTube Music URL or search query")
    name: Optional[str] = Field(None, description="Override output filename")
    separate: bool = Field(
        True,
        description="Run Demucs after download — always True by default (auto-stem pipeline)",
    )


class BatchDownloadRequest(BaseModel):
    queries: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="List of search queries or URLs — one track each",
    )
    separate: bool = Field(
        True,
        description="Run Demucs on every track after download (default: True)",
    )


class PlaylistDownloadRequest(BaseModel):
    url: str = Field(..., description="YouTube / YouTube Music playlist URL")
    separate: bool = Field(
        True,
        description="Run Demucs on every track after download (default: True)",
    )
    limit: Optional[int] = Field(
        None, ge=1, le=200,
        description="Max number of tracks to download (default: all)",
    )


class StemSplitRequest(BaseModel):
    song: str = Field(..., description="Song name in library (must have full.wav)")
    enhance: bool = Field(True, description="Run audio enhancement chain before Demucs")
    model: str = Field("htdemucs", description="Demucs model name (htdemucs | htdemucs_ft | mdx_extra)")


class BatchStemSplitRequest(BaseModel):
    songs: Optional[list] = Field(
        None,
        description="List of song names to split. Pass null/empty to split ALL library songs.",
    )
    enhance: bool = Field(True, description="Run audio enhancement before each split")
    model: str = Field("htdemucs", description="Demucs model")
    skip_existing: bool = Field(True, description="Skip songs that already have vocals.wav")


class DJRemixRequest(BaseModel):
    song_a: str = Field(..., description="Name of first song (must exist in library)")
    song_b: str = Field(..., description="Name of second song (must exist in library)")
    transition_bars: int = Field(16, ge=8, le=64, description="8 | 16 | 32")
    preset: str = Field("auto", description="Genre preset or 'auto'")

    # Bridge beat options
    bridge_beat_mode: str = Field(
        "none",
        description="'none' | 'auto' (Python synthesis) | 'file' (user-supplied WAV)",
    )
    bridge_beat_genre: str = Field(
        "auto",
        description="Genre preset for auto beat synthesis (techno/house/hiphop/trap/dnb/ambient)",
    )
    bridge_beat_intensity: float = Field(
        0.38,
        ge=0.0, le=1.0,
        description="Peak gain of the bridge beat layer (0 = silent, 1 = full)",
    )
    bridge_beat_path: Optional[str] = Field(
        None,
        description="Absolute path to a user-uploaded WAV (used when mode='file')",
    )
    transition_effect: str = Field(
        "auto",
        description="Effect on the outgoing track: 'auto' | 'echo' | 'filter' | 'reverb' | 'none'",
    )


class DJChainRequest(BaseModel):
    songs: List[str] = Field(
        ...,
        min_length=2,
        max_length=8,
        description="Ordered list of 2–8 song names (must exist in library)",
    )
    transition_bars: int = Field(16, ge=8, le=64, description="8 | 16 | 32 bars per transition")
    preset: str = Field("auto", description="Genre preset or 'auto'")

    # Bridge beat options (applied to every transition in the chain)
    bridge_beat_mode: str = Field("none", description="'none' | 'auto' | 'file'")
    bridge_beat_genre: str = Field("auto", description="Genre preset for auto beat synthesis")
    bridge_beat_intensity: float = Field(0.38, ge=0.0, le=1.0)
    bridge_beat_path: Optional[str] = Field(None, description="WAV path for mode='file'")
    transition_effect: str = Field(
        "auto",
        description="Effect on the outgoing track: 'auto' | 'echo' | 'filter' | 'reverb' | 'none'",
    )


class DJPreviewRequest(BaseModel):
    """Lightweight request to render a transition-only preview (no full mix)."""
    song_a: str = Field(..., description="Name of first song (must exist in library)")
    song_b: str = Field(..., description="Name of second song (must exist in library)")
    transition_bars: int = Field(16, ge=8, le=64, description="8 | 16 | 32 bars for the preview window")
    transition_effect: str = Field(
        "auto",
        description="Effect on the outgoing track: 'auto' | 'echo' | 'filter' | 'reverb' | 'none'",
    )


class CompatibilityRequest(BaseModel):
    song_a: str = Field(..., description="Song name or search query")
    song_b: str = Field(..., description="Song name or search query")
    artist_a: str = ""
    artist_b: str = ""


class AnalyzeRequest(BaseModel):
    song: str = Field(..., description="Song name (must exist in library)")


# ---------------------------------------------------------------------------
# Music intelligence result models
# ---------------------------------------------------------------------------

class MusicVectorResponse(BaseModel):
    """Detailed per-track music intelligence features."""
    key:               Optional[str]        = None
    mode:              Optional[str]        = None   # "major" | "minor"
    camelot:           Optional[str]        = None   # e.g. "8B"
    key_confidence:    Optional[float]      = None   # 0–1
    danceability:      Optional[float]      = None   # 0–1
    vocal_density:     Optional[float]      = None   # 0–1
    spectral_centroid_hz: Optional[float]   = None
    chord_sequence:    List[str]            = []
    drop_position_sec: Optional[float]      = None


class QualityReportResponse(BaseModel):
    """Post-mastering quality report."""
    lufs_integrated:   float
    lufs_target:       float
    lufs_gain_applied: float
    peak_dbfs:         float
    has_clipping:      bool
    clip_count:        int
    dynamic_range_db:  float
    passed:            bool
    notes:             List[str] = []


class TransitionScoreResponse(BaseModel):
    """AI-computed transition compatibility score."""
    overall:          float = Field(..., ge=0.0, le=1.0)
    beat_alignment:   float = Field(..., ge=0.0, le=1.0)
    harmonic_match:   float = Field(..., ge=0.0, le=1.0)
    energy_smoothness: float = Field(..., ge=0.0, le=1.0)
    vocal_clash:      float = Field(..., ge=0.0, le=1.0)
    vocal_clash_penalty: float = 0.0
    camelot_a:        Optional[str] = None
    camelot_b:        Optional[str] = None
    key_a:            Optional[str] = None
    key_b:            Optional[str] = None
    recommended_transition_bars: int = 16
    notes:            List[str] = []


# Add transition_effect option to DJRemixRequest (extend existing model below)

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

class LibraryStats(BaseModel):
    total_songs: int
    total_size_gb: float
    cap_gb: float
    within_cap: bool
    songs_with_stems: int


class LibraryListResponse(BaseModel):
    stats: LibraryStats
    songs: List[SongInfo]


# ---------------------------------------------------------------------------
# RAG index similarity
# ---------------------------------------------------------------------------

class SimilarSongBreakdown(BaseModel):
    """Per-dimension similarity sub-scores for a single search result."""
    bpm_sim:    float = Field(..., ge=0, le=1)
    key_sim:    float = Field(..., ge=0, le=1)
    energy_sim: float = Field(..., ge=0, le=1)
    rhythm_sim: float = Field(..., ge=0, le=1)
    timbre_sim: float = Field(..., ge=0, le=1)


class SimilarSongResult(BaseModel):
    """One result from the RAG vector similarity search."""
    name:          str
    score:         float = Field(..., ge=0, le=1, description="Weighted cosine similarity")
    bpm:           Optional[float] = None
    key:           Optional[str]   = None
    mode:          Optional[str]   = None
    camelot:       Optional[str]   = None
    genre:         Optional[str]   = None
    danceability:  Optional[float] = None
    vocal_density: Optional[float] = None
    breakdown:     Optional[SimilarSongBreakdown] = None


class SimilarSongsResponse(BaseModel):
    source:    str
    similar:   List[SimilarSongResult]
    engine:    str = "rag_vector"


# ---------------------------------------------------------------------------
# Stem compression
# ---------------------------------------------------------------------------

class CompressStemsRequest(BaseModel):
    song:       str  = Field(..., description="Song name in library")
    delete_wav: bool = Field(True, description="Delete WAV source after successful FLAC encode")


class BatchCompressStemsRequest(BaseModel):
    songs:          Optional[List[str]] = Field(
        None,
        description="Song names to compress. Pass null for all library songs.",
    )
    delete_wav:     bool = Field(True,  description="Delete WAV files after FLAC encode")
    skip_existing:  bool = Field(True,  description="Skip songs that already have FLAC stems")


# ---------------------------------------------------------------------------
# Library initialisation pipeline
# ---------------------------------------------------------------------------

class InitializeLibraryRequest(BaseModel):
    enhance:      bool = Field(True,        description="Enhance audio before stem split")
    model:        str  = Field("htdemucs",  description="Demucs model to use")
    delete_wav:   bool = Field(True,        description="Delete WAV stems after FLAC encode")
    run_compress: bool = Field(True,        description="Run FLAC compression after splitting")
    run_index:    bool = Field(True,        description="Rebuild RAG index after compression")


# ---------------------------------------------------------------------------
# Instrument Lab — stem swap experiments
# ---------------------------------------------------------------------------

class InstrumentLabRequest(BaseModel):
    songs:            List[str] = Field(..., description="2+ song names (must have stems)")
    mode:             str  = Field("targeted", description="'all' for full permutation, 'targeted' for single-stem swaps")
    swap_stems:       Optional[List[str]] = Field(None, description="Which stems to swap (targeted mode only)")
    target_duration:  Optional[float] = Field(None,  description="Trim each combo to N seconds (None = full length)")
    include_pure:     bool = Field(False, description="Include combos where all stems come from one song")


# ---------------------------------------------------------------------------
# AI Studio — Style Transfer (MusicGen)
# ---------------------------------------------------------------------------

class StyleTransferRequest(BaseModel):
    song_name: str = Field(..., description="Library song to use as melody source")
    description: str = Field(
        ...,
        description=(
            "Text style prompt for MusicGen. "
            "Example: 'dark melodic techno 128 BPM heavy sub-bass melancholic chords'"
        ),
    )
    duration_sec: float = Field(
        15.0, ge=4.0, le=30.0,
        description="Length of generated audio in seconds (max 30 on 16 GB RAM)",
    )
    source_stem: str = Field(
        "full",
        description="Stem to extract melody from: 'full' | 'vocals' | 'other' | 'bass'",
    )
    source_start_sec: float = Field(0.0, ge=0.0, description="Offset into source audio")
    source_clip_sec: float = Field(
        15.0, ge=4.0, le=30.0,
        description="How many seconds of source audio to use as melody reference",
    )
    guidance_scale: float = Field(3.0, ge=1.0, le=10.0, description="CFG guidance strength")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(250, ge=1, le=1000)
    seed: Optional[int] = Field(None, description="Random seed (null = random)")
    output_format: str = Field("wav", description="'wav' or 'flac'")


class StyleTransferResponse(BaseModel):
    """Response from a completed style transfer job."""
    song_name: str
    description: str
    output_path: Optional[str] = None
    audio_url: Optional[str] = None
    duration_sec: float = 0.0
    lufs: float = 0.0
    source_key: Optional[str] = None
    source_camelot: Optional[str] = None
    generation_time_sec: float = 0.0
    method: str = "musicgen-melody"


# ---------------------------------------------------------------------------
# AI Studio — VampNet Inpainting
# ---------------------------------------------------------------------------

class InpaintRequest(BaseModel):
    song_a: str = Field(..., description="Outgoing track (tail will be used as prefix context)")
    song_b: str = Field(..., description="Incoming track (head will be used as suffix context)")
    mask_type: str = Field(
        "prefix_suffix",
        description=(
            "Mask strategy: "
            "'prefix_suffix' (keep A tail + B head, inpaint gap) | "
            "'periodic' (rhythmic anchor every P tokens) | "
            "'beat_driven' (keep beat positions) | "
            "'compression' (keep coarse codebooks only)"
        ),
    )
    prefix_bars: int = Field(4, ge=1, le=16, description="Bars from song_a end to use as prefix")
    suffix_bars: int = Field(4, ge=1, le=16, description="Bars from song_b start to use as suffix")
    periodic_prompt: int = Field(4, ge=1, le=16, description="Keep every N-th token (periodic mode)")
    n_codebooks_to_keep: int = Field(3, ge=1, le=8, description="Coarse codebooks to preserve (compression mode)")
    sampling_steps: int = Field(36, ge=8, le=128, description="VampNet decoding passes (more = better quality)")
    rand_mask_intensity: float = Field(0.7, ge=0.0, le=1.0, description="Fraction of non-anchor tokens to mask")
    source_stem: str = Field("full", description="Stem to use: 'full' | 'vocals' | 'other' | 'bass' | 'drums'")
    bpm: Optional[float] = Field(None, ge=40.0, le=250.0, description="BPM override (null = auto-detect)")
    output_format: str = Field("wav", description="'wav' or 'flac'")


class InpaintResponse(BaseModel):
    """Response from a completed inpainting job."""
    song_a: str
    song_b: str
    output_path: Optional[str] = None
    audio_url: Optional[str] = None
    method: str = "vampnet"    # "vampnet" | "cosine_fallback"
    duration_sec: float = 0.0
    lufs: float = 0.0
    generation_time_sec: float = 0.0
    metadata: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# AI Studio — Tokenization
# ---------------------------------------------------------------------------

class TokenizeRequest(BaseModel):
    song_name: str = Field(..., description="Library song to tokenize")
    codec: str = Field("encodec", description="'encodec' (24kHz) or 'dac' (44.1kHz)")
    bandwidth: float = Field(6.0, description="EnCodec bandwidth in kbps (1.5, 3, 6, 12, 24)")


class TokenizeResponse(BaseModel):
    song_name: str
    codec: str
    stems: List[str] = []
    total_tokens: int = 0
    token_rate_hz: float = 0.0
    num_codebooks: int = 0
    codebook_size: int = 0
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# AI Studio — Model status
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    name: str
    loaded: bool
    vram_estimate_gb: float
    description: str
    last_used: float = 0.0


class ModelStatusResponse(BaseModel):
    device: str
    mps_allocated_gb: float = 0.0
    max_vram_gb: float = 12.0
    models: List[ModelInfo] = []
