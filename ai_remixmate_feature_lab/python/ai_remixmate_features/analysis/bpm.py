from __future__ import annotations

from pathlib import Path


def _require_audio_path(audio_path: str) -> Path:
    path = Path(audio_path)
    if not str(audio_path).strip():
        raise ValueError("audio_path is required")
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")
    return path


def _load_librosa():
    try:
        import librosa  # type: ignore
    except ImportError as exc:
        raise ImportError("estimate_bpm requires librosa. Install the lab with the audio extra or run in the main RemixMate environment.") from exc
    return librosa


def estimate_bpm(audio_path: str) -> dict:
    """Estimate BPM for an audio file with librosa and return JSON data.

    TODO: Compare octave resolution with production scripts.core.beat_tracker before integration.
    """
    path = _require_audio_path(audio_path)
    librosa = _load_librosa()
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0] if hasattr(tempo, "__len__") else tempo)
    return {"audioPath": str(path), "bpm": round(bpm, 3), "sampleRate": int(sr), "confidence": 0.5, "source": "librosa.beat.beat_track"}
