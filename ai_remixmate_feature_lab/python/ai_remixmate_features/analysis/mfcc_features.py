from __future__ import annotations

from .bpm import _load_librosa, _require_audio_path


def extract_mfcc_features(audio_path: str) -> dict:
    """Extract MFCC mean/std features for timbre compatibility.

    TODO: Compare dimensions with the production music index before integration.
    """
    path = _require_audio_path(audio_path)
    librosa = _load_librosa()
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    means = mfcc.mean(axis=1)
    stds = mfcc.std(axis=1)
    return {"audioPath": str(path), "mfccMean": [float(v) for v in means], "mfccStd": [float(v) for v in stds], "vector": [float(v) for v in list(means) + list(stds)]}
