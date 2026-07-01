from __future__ import annotations

from .bpm import _load_librosa, _require_audio_path


def extract_energy_curve(audio_path: str) -> dict:
    """Extract a normalized RMS energy curve.

    TODO: Align buckets with production section and waveform summaries.
    """
    path = _require_audio_path(audio_path)
    librosa = _load_librosa()
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    rms = librosa.feature.rms(y=y)[0]
    max_value = float(rms.max()) if len(rms) else 0.0
    curve = [float(v / max_value) if max_value > 0 else 0.0 for v in rms]
    return {"audioPath": str(path), "sampleRate": int(sr), "energyCurve": curve, "meanEnergy": float(sum(curve) / len(curve)) if curve else 0.0}
