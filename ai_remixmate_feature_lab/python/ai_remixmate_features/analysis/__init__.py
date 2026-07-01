from __future__ import annotations

from typing import Any

from .beatgrid import estimate_beatgrid
from .bpm import estimate_bpm
from .chroma_key import estimate_chroma_key
from .energy_curve import extract_energy_curve
from .mfcc_features import extract_mfcc_features
from .sections import estimate_sections
from .stem_manifest import build_stem_manifest
from .waveform_summary import summarize_waveform


def export_track_analysis(track_id: str, title: str, duration_sec: float, bpm: float, beatgrid: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Build a JSON-serializable TrackAnalysis-like dictionary.

    TODO: Adapt this shape to the production AI RemixMate analysis pipeline after
    schema review. This function intentionally performs no repository writes.
    """
    if not track_id.strip():
        raise ValueError("track_id is required")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    return {
        "id": track_id,
        "title": title,
        "durationSec": float(duration_sec),
        "bpm": float(bpm),
        "beatgrid": beatgrid,
        **kwargs,
    }


__all__ = [
    "estimate_bpm",
    "estimate_beatgrid",
    "estimate_chroma_key",
    "extract_mfcc_features",
    "extract_energy_curve",
    "estimate_sections",
    "build_stem_manifest",
    "summarize_waveform",
    "export_track_analysis",
]
