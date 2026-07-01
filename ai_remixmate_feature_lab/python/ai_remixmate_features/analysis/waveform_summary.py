from __future__ import annotations

from .bpm import _load_librosa, _require_audio_path


def summarize_waveform(audio_path: str, buckets: int = 256) -> dict:
    """Return downsampled peak and RMS waveform values.

    TODO: Persist summaries beside production analysis metadata for fast UI loading.
    """
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    path = _require_audio_path(audio_path)
    librosa = _load_librosa()
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    if len(y) == 0:
        return {"audioPath": str(path), "sampleRate": int(sr), "durationSec": 0.0, "peaks": [], "rms": []}
    size = max(1, len(y) // buckets)
    peaks = []
    rms = []
    for start in range(0, len(y), size):
        chunk = y[start:start + size]
        if len(chunk) == 0:
            continue
        peaks.append(float(abs(chunk).max()))
        rms.append(float((chunk * chunk).mean() ** 0.5))
    return {"audioPath": str(path), "sampleRate": int(sr), "durationSec": float(len(y) / sr), "peaks": peaks[:buckets], "rms": rms[:buckets]}
