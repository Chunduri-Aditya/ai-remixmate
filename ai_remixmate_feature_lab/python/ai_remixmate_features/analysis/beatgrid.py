from __future__ import annotations

from .bpm import _load_librosa, _require_audio_path


def estimate_beatgrid(audio_path: str) -> dict:
    """Estimate beat timestamps and simple downbeats for an audio file.

    TODO: Replace every-fourth-beat downbeat fallback with production downbeat phase logic.
    """
    path = _require_audio_path(audio_path)
    librosa = _load_librosa()
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = [float(t) for t in librosa.frames_to_time(frames, sr=sr)]
    bpm = float(tempo[0] if hasattr(tempo, "__len__") else tempo)
    downbeats = beat_times[::4]
    return {"audioPath": str(path), "bpm": round(bpm, 3), "beatTimes": beat_times, "downbeats": downbeats, "beatsPerBar": 4, "confidence": 0.5}
