"""
scripts/core/config.py — Centralised configuration loader for AI RemixMate.

Priority (highest → lowest):
  1. Environment variables  REMIXMATE_<SECTION>_<KEY>  (e.g. REMIXMATE_AUDIO_SAMPLE_RATE)
  2. config.local.yaml      (user overrides, gitignored)
  3. config.yaml            (project defaults, committed)

Usage:
  from scripts.core.config import cfg

  sr  = cfg.audio.sample_rate       # 44100
  tgt = cfg.audio.target_lufs       # -14.0
  mdl = cfg.separation.model        # "htdemucs"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # PyYAML — already in requirements

# ---------------------------------------------------------------------------
# Config file locations
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent   # project root

_BASE_CONFIG   = _ROOT / "config.yaml"
_LOCAL_CONFIG  = _ROOT / "config.local.yaml"  # gitignored; user overrides


# ---------------------------------------------------------------------------
# Typed config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    sample_rate: int   = 44100
    bit_depth: int     = 24
    target_lufs: float = -14.0
    channels: int      = 1


@dataclass
class RemixConfig:
    default_preset: str              = "radio"
    optimizer_iterations: int        = 50
    max_pitch_shift_semitones: float = 2.0
    beat_alignment_tolerance_ms: float = 40.0
    # Load cap (seconds) per song for task_dj_remix / task_dj_chain; env
    # override REMIXMATE_REMIX_MAX_SONG_SECONDS.
    max_song_seconds: int            = 180


@dataclass
class SeparationConfig:
    model: str  = "htdemucs"
    device: str = "auto"
    # DJ_ENGINE_IMPROVEMENT_PLAN.md §2.2 — default window (seconds) for
    # preview-tier separation (live preview / PiP widget), used by callers
    # that pass preview_seconds to separate_song_stems(); has no effect on
    # full-quality library separation, which always processes the whole
    # track regardless of this value.
    preview_seconds: float = 60.0


@dataclass
class DownloadConfig:
    # Catalogue ingest is operator-only (scripts/download.py). jamendo_client_id
    # is retained for a possible future Jamendo *metadata* lookup; it is not used
    # for audio download (Jamendo's API terms forbid caching).
    jamendo_client_id: str    = "b6747d04"
    audio_format: str         = "wav"
    audio_quality: int        = 0
    no_playlist_by_default: bool = True


@dataclass
class MetadataConfig:
    getsongbpm_api_key: str = ""
    lastfm_api_key: str     = ""
    cache_path: str         = "data/metadata.db"
    cache_ttl_days: int     = 30


@dataclass
class DJConfig:
    # "classic" = vendored AI_REMIXMATE_BEST_MIXING_ALGO render path
    # (scripts/core/classic_engine/); "current" = live engine. Consumed by
    # scripts/api/task_modules/remix.py::_resolve_engine().
    mixing_engine: str           = "classic"
    default_transition_bars: int = 16
    hp_filter_start_hz: float    = 400.0
    hp_filter_end_hz: float      = 80.0
    bass_crossover_hz: float     = 150.0
    # Masking-aware multiband EQ (scripts/core/spectral_mixing.py), read by
    # dj_engine.py. masking_eq_enabled is the A/B kill switch — strength does
    # NOT go to zero effect at 0.0 (Eq.(2)'s multiplier is 2**strength, so
    # 0.0 still applies a full 1x masking gain); see masking_gain_curve()'s
    # docstring. Use masking_eq_enabled=False to actually disable this.
    masking_eq_enabled: bool         = True
    masking_eq_strength: float       = 0.0
    masking_eq_max_atten_db: float   = 6.0
    # Learned spectrogram cue matching (scripts/core/spectral_matching.py) —
    # picks B's entry phrase by mel-band fit to A's outro; band weights train
    # on POST /dj-remix/feedback verdicts at spectral_matching_lr per step.
    spectral_matching: bool          = True
    spectral_matching_lr: float      = 0.2
    # Dynamic in-transition spectral sync (scripts/core/spectral_sync.py) —
    # post-render multiband EQ easing the overlap's spectrum A→B over time.
    spectral_sync: bool              = True
    spectral_sync_max_db: float      = 4.0
    # Confidence gate on the render path (DJ_ENGINE_IMPROVEMENT_PLAN.md §2.1):
    # below this threshold, task_dj_remix falls back to the plain render()
    # crossfade instead of committing to the full stem-aware blend, even when
    # both songs' stems are on disk. Reuses the same 0.35 "harmonic clash"
    # cutoff plan_transition() already applies to shorten transition_bars —
    # see scripts/api/task_modules/remix.py::_should_use_stem_blend().
    transition_confidence_threshold: float = 0.30


@dataclass
class LibraryConfig:
    max_size_gb: float              = 20.0   # GB cap before LRU eviction
    keep_raw_after_separation: bool = False  # keep full.wav after stems produced
    prune_on_download: bool         = True   # auto-prune after each separation
    auto_evict_on_download: bool    = True   # run evict_lru() after every download (can
                                              # silently delete OTHER old songs — surfaced
                                              # in download job results either way)
    library_dir: str                = ""     # "" = default <project_root>/library.
                                              # Set to an absolute path to store the
                                              # library on an external drive or NAS mount
                                              # (e.g. "/Volumes/MusicDrive/remixmate-library").
                                              # NOT object storage (S3/GCS) — every script
                                              # in this codebase opens stems with plain
                                              # pathlib/soundfile calls, so this only
                                              # supports filesystem-mounted locations.
    outputs_dir: str                = ""     # "" = default <project_root>/outputs.


@dataclass
class RetentionConfig:
    # Bounds a user's outputs/<uid>/ folder — see scripts/core/mix_retention.py.
    # The shared catalog is a fixed operator asset; per-user rendered output is
    # not, so it gets a TTL and a count cap.
    mix_max_age_days: int      = 30    # remove a mix dir whose newest mtime is older
    mix_max_count: int         = 50    # then keep at most this many, oldest evicted first
    sweep_after_render: bool   = True  # run the sweep on the acting user after each render
    sweep_interval_hours: float = 6    # lifespan task sweeps every outputs/<uid>/ this often


@dataclass
class DatabaseConfig:
    path: str             = "data/remixmate.db"
    embeddings_path: str  = "data/song_embeddings.json"


@dataclass
class ApiConfig:
    host: str              = "0.0.0.0"
    port: int              = 8000
    workers: int           = 1
    reload: bool           = False
    # Explicit allowlist — no wildcard. Add origins in config.yaml if needed.
    # Wildcard CORS is flagged by CodeQL (CWE-942 / partial CSRF exposure).
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:8501",   # Streamlit UI (default port)
        "http://127.0.0.1:8501",
        "http://localhost:8000",   # API self-calls / Swagger UI
        "http://127.0.0.1:8000",
    ])


@dataclass
class LoggingConfig:
    level: str          = "INFO"
    format: str         = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    file: Optional[str] = None


@dataclass
class GenerativeConfig:
    # Consumed via model_manager._get_cfg_value("generative", ...).
    max_vram_gb: float                = 12.0
    strategy: str                     = "sequential"   # sequential | greedy
    musicgen_model: str               = "facebook/musicgen-melody"
    musicgen_offload_after_use: bool  = True
    vampnet_offload_after_use: bool   = True
    encodec_model: str                = "facebook/encodec_24khz"
    encodec_bandwidth: float          = 6.0
    dac_model: str                    = "44khz"
    max_generation_sec: float         = 30.0
    default_guidance_scale: float     = 3.0
    default_temperature: float        = 1.0


@dataclass
class SpotifyConfig:
    # scripts/core/spotify.py reads the raw YAML itself (plus SPOTIFY_* env
    # vars); declared here so the section survives _dict_to_config() and the
    # drift guard in tests/test_config_drift.py covers it. Metadata-only now —
    # user OAuth (and redirect_uri) went with the catalogue teardown.
    client_id: str     = ""
    client_secret: str = ""


@dataclass
class AnalysisConfig:
    # Read by beat_tracker.py / energy_profiler.py via guarded getattr.
    beat_backend: str   = "auto"
    energy_backend: str = "auto"


@dataclass
class ModelsConfig:
    # Read by crate_digger.py / energy_profiler.py via guarded getattr.
    clap_model: Optional[str]   = None
    musicnn_path: Optional[str] = None


@dataclass
class AppConfig:
    """Root config object — access via  cfg.<section>.<key>"""
    audio:      AudioConfig      = field(default_factory=AudioConfig)
    remix:      RemixConfig      = field(default_factory=RemixConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    download:   DownloadConfig   = field(default_factory=DownloadConfig)
    metadata:   MetadataConfig   = field(default_factory=MetadataConfig)
    dj:         DJConfig         = field(default_factory=DJConfig)
    library:    LibraryConfig    = field(default_factory=LibraryConfig)
    retention:  RetentionConfig  = field(default_factory=RetentionConfig)
    database:   DatabaseConfig   = field(default_factory=DatabaseConfig)
    api:        ApiConfig        = field(default_factory=ApiConfig)
    logging:    LoggingConfig    = field(default_factory=LoggingConfig)
    generative: GenerativeConfig = field(default_factory=GenerativeConfig)
    spotify:    SpotifyConfig    = field(default_factory=SpotifyConfig)
    analysis:   AnalysisConfig   = field(default_factory=AnalysisConfig)
    models:     ModelsConfig     = field(default_factory=ModelsConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _apply_env_vars(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override any config value via environment variables.

    Naming convention:  REMIXMATE_<SECTION>_<KEY>
    Examples:
      REMIXMATE_AUDIO_SAMPLE_RATE=48000
      REMIXMATE_DOWNLOAD_JAMENDO_CLIENT_ID=my_key
      REMIXMATE_API_PORT=9000
      REMIXMATE_SEPARATION_MODEL=htdemucs_ft
    """
    prefix = "REMIXMATE_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, key = parts
        if section not in raw:
            raw[section] = {}
        # Cast to correct Python type based on existing value
        existing = raw[section].get(key)
        if isinstance(existing, bool):
            raw[section][key] = env_val.lower() in ("1", "true", "yes")
        elif isinstance(existing, int):
            try:
                raw[section][key] = int(env_val)
            except ValueError:
                pass
        elif isinstance(existing, float):
            try:
                raw[section][key] = float(env_val)
            except ValueError:
                pass
        else:
            raw[section][key] = env_val
    return raw


def _dict_to_config(raw: Dict[str, Any]) -> AppConfig:
    """Populate the typed AppConfig from a raw dictionary."""
    import dataclasses
    import typing

    def _fill(cls, data: Dict) -> Any:
        kwargs = {}
        # get_type_hints() resolves string annotations (from __future__ import annotations)
        # to actual class objects — critical for nested dataclass detection.
        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            hints = {}

        for f in dataclasses.fields(cls):
            val = data.get(f.name)
            if val is None:
                continue
            # Use resolved type, not the raw string annotation
            actual_type = hints.get(f.name, f.type)
            if dataclasses.is_dataclass(actual_type) and isinstance(val, dict):
                kwargs[f.name] = _fill(actual_type, val)
            else:
                kwargs[f.name] = val
        return cls(**kwargs)

    return _fill(AppConfig, raw)


def load_config() -> AppConfig:
    """Load, merge, and return the complete application configuration."""
    base  = _load_yaml(_BASE_CONFIG)
    local = _load_yaml(_LOCAL_CONFIG)
    merged = _deep_merge(base, local)
    merged = _apply_env_vars(merged)
    return _dict_to_config(merged)


# ---------------------------------------------------------------------------
# Module-level singleton — import  cfg  from here
# ---------------------------------------------------------------------------

cfg: AppConfig = load_config()


# ---------------------------------------------------------------------------
# Convenience accessors (avoids chaining for the most-used values)
# ---------------------------------------------------------------------------

def sample_rate() -> int:
    return cfg.audio.sample_rate

def bit_depth() -> int:
    return cfg.audio.bit_depth

def target_lufs() -> float:
    return cfg.audio.target_lufs

def optimizer_iterations() -> int:
    return cfg.remix.optimizer_iterations

def demucs_model() -> str:
    return cfg.separation.model

def jamendo_client_id() -> str:
    return cfg.download.jamendo_client_id


# ---------------------------------------------------------------------------
# Logging bootstrap (call once at app startup)
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Set up root logger from config. Call this in main entry points."""
    from scripts.core.logging_utils import LogInjectionFilter

    lc = cfg.logging
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if lc.file:
        log_path = Path(lc.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    # User-controlled text (song names, tags, search queries) reaches log calls
    # across scripts/core/. Scrub CR/LF here so no call site can forge a line.
    for _h in handlers:
        _h.addFilter(LogInjectionFilter())
    logging.basicConfig(
        level=getattr(logging, lc.level.upper(), logging.INFO),
        format=lc.format,
        handlers=handlers,
    )
