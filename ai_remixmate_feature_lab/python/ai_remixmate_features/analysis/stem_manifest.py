from __future__ import annotations

from pathlib import Path

STEM_TYPES = ("vocals", "drums", "bass", "other", "instrumental", "full")
EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".aiff")


def build_stem_manifest(track_id: str, stems_dir: str) -> dict:
    """Build a JSON-serializable manifest for available stem files.

    TODO: Add duration/sample-rate probing when production audio IO helpers are integrated.
    """
    if not track_id.strip():
        raise ValueError("track_id is required")
    root = Path(stems_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"stems directory not found: {stems_dir}")
    stems = []
    for stem_type in STEM_TYPES:
        found = next((root / f"{stem_type}{ext}" for ext in EXTENSIONS if (root / f"{stem_type}{ext}").exists()), None)
        stems.append({"type": stem_type, "path": str(found) if found else "", "available": found is not None})
    return {"trackId": track_id, "stemsDir": str(root), "stems": stems}
