"""
scripts/api/routers/_helpers.py — Shared helper functions for all domain routers.

These utilities are used by multiple routers for common operations like
song validation, stem file lookup, and rate limiting.
"""

from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from scripts.api import jobs as job_store
from scripts.api.schemas import SongInfo
from scripts.core.paths import LIBRARY_DIR


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _stem_file(song_dir: Path, stem: str) -> Path | None:
    """Return the stem file path (prefers .flac, falls back to .wav), or None."""
    flac = song_dir / f"{stem}.flac"
    if flac.exists():
        return flac
    wav = song_dir / f"{stem}.wav"
    if wav.exists():
        return wav
    return None


def _song_info(song_dir: Path) -> SongInfo:
    size = sum(f.stat().st_size for f in song_dir.rglob("*") if f.is_file())
    stems = [
        s for s in ("vocals", "drums", "bass", "other")
        if _stem_file(song_dir, s) is not None
    ]
    license_type = None
    source = None
    try:
        import json
        lic_path = song_dir / "license.json"
        if lic_path.exists():
            lic = json.loads(lic_path.read_text())
            license_type = lic.get("license_type")
            source = lic.get("source")
    except Exception:
        pass

    return SongInfo(
        name=song_dir.name,
        size_mb=round(size / 1_048_576, 2),
        has_full_wav=(song_dir / "full.wav").exists(),
        stems=stems,
        license_type=license_type,
        source=source,
    )


def _require_song(name: str) -> Path:
    """Return the song dir or raise 404.  Prevents path traversal."""
    # Sanitize: strip path separators and parent references
    safe = Path(name).name  # extracts final component, strips ../ etc.
    if not safe or safe != name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid song name")
    d = (LIBRARY_DIR / safe).resolve()
    # Ensure resolved path is still inside library
    if not str(d).startswith(str(LIBRARY_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid song name")
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Song not found in library: {name!r}")
    return d


# ---------------------------------------------------------------------------
# Rate limiting / active job caps
# ---------------------------------------------------------------------------

_MAX_ACTIVE_JOBS = 4  # Hard cap on concurrent background jobs


def _check_job_cap() -> None:
    """Raise 429 if the user already has too many active jobs running."""
    active = [j for j in job_store.list_jobs() if j["status"] == "running"]
    if len(active) >= _MAX_ACTIVE_JOBS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too many active jobs ({len(active)}/{_MAX_ACTIVE_JOBS}). "
                f"Wait for one to finish before submitting another."
            ),
        )
