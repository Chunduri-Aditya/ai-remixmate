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

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# New canonical directories (v0.3)
# ---------------------------------------------------------------------------
LIBRARY_DIR  = PROJECT_ROOT / "library"     # per-song folder
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"     # remix sessions
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
