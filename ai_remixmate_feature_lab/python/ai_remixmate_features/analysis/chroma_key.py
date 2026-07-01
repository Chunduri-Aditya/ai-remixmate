from __future__ import annotations

from .bpm import _load_librosa, _require_audio_path

KEYS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def estimate_chroma_key(audio_path: str) -> dict:
    """Estimate a simple chroma key summary.

    TODO: Replace max-chroma heuristic with production Krumhansl/Camelot scoring.
    """
    path = _require_audio_path(audio_path)
    librosa = _load_librosa()
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    means = chroma.mean(axis=1)
    idx = int(means.argmax())
    confidence = float(means[idx] / (means.sum() + 1e-9))
    return {"audioPath": str(path), "key": KEYS[idx], "mode": "unknown", "camelot": None, "confidence": round(confidence, 4), "chroma": [float(v) for v in means]}
