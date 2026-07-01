"""
Path constants and utilities for AI RemixMate.

New canonical layout (v0.3+):
  library/<SongName>/        ← everything for one song lives here
    full.wav                 ← original downloaded WAV
    vocals.wav               ← Demucs vocal stem
    drums.wav / bass.wav / other.wav
    lyrics.txt               ← Whisper transcription

  outputs/<session_id>/      ← finished remixes + session artefacts
    mix.wav / report.json / manifest.json

  data/                      ← database + embeddings (no model weights)
    remixmate.db
    song_embeddings.json

  models/                    ← actual ML model checkpoints (Demucs, etc.)

Backwards-compatible aliases are preserved so existing code keeps working
without any changes until it is migrated to the new names.
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Song-name character policy — SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------
# Every place in the codebase that (a) sanitizes a raw title into an on-disk
# song name, or (b) validates a song name before using it in a filesystem
# path, must agree on the same allowed character set. Before this constant
# existed, scripts/download.py:_sanitize() (the real on-disk name source for
# every /download and /download-playlist job) used a BLOCKLIST — it only
# stripped Windows-reserved chars (\ / : * ? " < > |) and let everything
# else through, including commas, periods, semicolons, exclamation marks,
# and apostrophes. Meanwhile scripts/api/routers/_helpers.py used a much
# stricter ALLOWLIST that rejected most of that punctuation. Net effect: a
# perfectly normal download (e.g. a track titled "Don't Say" or "What Do
# You Mean, Pt. 2!") would land on disk fine, then get a permanent 400 Bad
# Request on every single API call that touches it afterward (audio stream,
# detail, similar, analyze, delete, stems) — with the frontend's query
# retry logic looping on the dead request forever. Both _sanitize() and the
# allowlist regex must derive from this one definition so they can't drift
# apart again.
#
# Explicitly EXCLUDED (filesystem/shell-dangerous, never allowed): path
# separators \ /, null bytes and other control characters, and the
# remaining Windows-reserved characters : * ? " < > |
SONG_NAME_SAFE_CHARS = r"\w\s\-\(\)\[\]&'’,.;!–—"
SONG_NAME_RE = re.compile(rf"^[{SONG_NAME_SAFE_CHARS}]{{1,120}}$")


def sanitize_song_name(name: str) -> str:
    """
    Turn a raw title into a song name guaranteed to pass SONG_NAME_RE.

    Collapses whitespace, strips any character outside the shared allowed
    set, then truncates to 120 chars. Used by the download pipeline so
    names written to disk can never produce a song that later 400s when
    read back through _require_song().
    """
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(rf"[^{SONG_NAME_SAFE_CHARS}]+", "", name)
    return name[:120].strip("._ ")

# ---------------------------------------------------------------------------
# Storage location overrides — read once at process start. Set
# library.library_dir / library.outputs_dir in config.yaml,
# config.local.yaml, or REMIXMATE_LIBRARY_LIBRARY_DIR /
# REMIXMATE_LIBRARY_OUTPUTS_DIR to move storage onto an external drive or
# NAS mount. A relative value resolves against PROJECT_ROOT; an absolute
# value (e.g. "/Volumes/MusicDrive/remixmate-library") is used as-is.
# This is a filesystem path only — NOT object storage (S3/GCS); every
# script in this codebase reads/writes stems with plain pathlib/soundfile
# calls, so the target must be locally mounted.
# ---------------------------------------------------------------------------
def _resolve_dir(configured: str, default_name: str) -> Path:
    if configured:
        p = Path(configured).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p)
    return PROJECT_ROOT / default_name


try:
    from scripts.core.config import cfg as _cfg
    _LIBRARY_DIR_OVERRIDE = getattr(getattr(_cfg, "library", None), "library_dir", "") or ""
    _OUTPUTS_DIR_OVERRIDE = getattr(getattr(_cfg, "library", None), "outputs_dir", "") or ""
except Exception:
    _LIBRARY_DIR_OVERRIDE = ""
    _OUTPUTS_DIR_OVERRIDE = ""

# ---------------------------------------------------------------------------
# New canonical directories (v0.3)
# ---------------------------------------------------------------------------
LIBRARY_DIR  = _resolve_dir(_LIBRARY_DIR_OVERRIDE, "library")    # per-song folder
OUTPUTS_DIR  = _resolve_dir(_OUTPUTS_DIR_OVERRIDE, "outputs")    # remix sessions
DATA_DIR     = PROJECT_ROOT / "data"        # db + embeddings
MODELS_DIR   = PROJECT_ROOT / "models"      # ML model weights

EMBEDDINGS_FILE = DATA_DIR / "song_embeddings.json"

# ---------------------------------------------------------------------------
# Backwards-compatible aliases (map old names → new locations)
# ---------------------------------------------------------------------------
AUDIO_IN   = LIBRARY_DIR          # was audio_input/
AUDIO_OUT  = OUTPUTS_DIR          # was audio_output/
OUTPUT_DIR = OUTPUTS_DIR          # was output/
SEPARATED  = LIBRARY_DIR          # was separated/htdemucs/ — stems now live in library/<name>/
STEMS_DIR  = LIBRARY_DIR          # was stems/
LYRICS_DIR = LIBRARY_DIR          # lyrics now live inside each song's folder


# ---------------------------------------------------------------------------
# Per-song helpers
# ---------------------------------------------------------------------------

def song_dir(song_name: str) -> Path:
    """Return the canonical library directory for a song."""
    return LIBRARY_DIR / song_name


def full_wav_path(song_name: str) -> Path:
    """Path to the original downloaded WAV for a song."""
    return song_dir(song_name) / "full.wav"


def vocals_path(song_name: str) -> Path:
    """Path to the Demucs vocal stem for a song."""
    return song_dir(song_name) / "vocals.wav"


def drums_path(song_name: str) -> Path:
    """Path to the Demucs drum stem for a song."""
    return song_dir(song_name) / "drums.wav"


def bass_path(song_name: str) -> Path:
    """Path to the Demucs bass stem for a song."""
    return song_dir(song_name) / "bass.wav"


def other_path(song_name: str) -> Path:
    """Path to the Demucs 'other' stem for a song."""
    return song_dir(song_name) / "other.wav"


def lyrics_path(song_name: str) -> Path:
    """Path to the Whisper lyrics transcription for a song."""
    return song_dir(song_name) / "lyrics.txt"


# ---------------------------------------------------------------------------
# Output / session helpers
# ---------------------------------------------------------------------------

def session_dir(session_id: str) -> Path:
    """Return the output directory for a remix session."""
    return OUTPUTS_DIR / session_id


def embeddings_path() -> Path:
    """Path to the song embeddings JSON file."""
    return EMBEDDINGS_FILE


# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    """Create all canonical directories if they don't already exist."""
    for d in (LIBRARY_DIR, OUTPUTS_DIR, DATA_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
