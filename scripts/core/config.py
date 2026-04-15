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


@dataclass
class SeparationConfig:
    model: str  = "htdemucs"
    device: str = "auto"


@dataclass
class DownloadConfig:
    default_source: str       = "auto"
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
    default_transition_bars: int = 16
    hp_filter_start_hz: float    = 400.0
    hp_filter_end_hz: float      = 80.0
    bass_crossover_hz: float     = 150.0


@dataclass
class LibraryConfig:
    max_size_gb: float              = 20.0   # GB cap before LRU eviction
    keep_raw_after_separation: bool = False  # keep full.wav after stems produced
    prune_on_download: bool         = True   # auto-prune after each separation


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
class AppConfig:
    """Root config object — access via  cfg.<section>.<key>"""
    audio:      AudioConfig      = field(default_factory=AudioConfig)
    remix:      RemixConfig      = field(default_factory=RemixConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    download:   DownloadConfig   = field(default_factory=DownloadConfig)
    metadata:   MetadataConfig   = field(default_factory=MetadataConfig)
    dj:         DJConfig         = field(default_factory=DJConfig)
    library:    LibraryConfig    = field(default_factory=LibraryConfig)
    database:   DatabaseConfig   = field(default_factory=DatabaseConfig)
    api:        ApiConfig        = field(default_factory=ApiConfig)
    logging:    LoggingConfig    = field(default_factory=LoggingConfig)


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
    lc = cfg.logging
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if lc.file:
        log_path = Path(lc.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=getattr(logging, lc.level.upper(), logging.INFO),
        format=lc.format,
        handlers=handlers,
    )
