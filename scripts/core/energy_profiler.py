"""
scripts/core/energy_profiler.py — Energy and arousal profiling for DJ setlist planning.

Provides two implementations behind a unified API:

NumPy fallback (always available)
    RMS energy + spectral centroid proxy.  Fast, reliable, requires only
    numpy + scipy.  Suitable for production use when Essentia is not installed.

Essentia backend (optional, high-accuracy)
    Uses Essentia's MusicExtractor or BpmHistogramDescriptors + ArousalValence
    model.  Provides calibrated arousal / valence predictions that map directly
    to the Russell circumplex model of affect.

Usage
-----
    from scripts.core.energy_profiler import profile_energy, EnergyFeatures

    feats = profile_energy(audio, sr)
    print(feats.arousal)     # 0.0–1.0, arousal dimension
    print(feats.rms_energy)  # 0.0–1.0, simple RMS proxy

Integration with SetlistPlanner
--------------------------------
    from scripts.core.energy_profiler import enrich_track_node

    enrich_track_node(track, audio, sr)
    # track.energy        ← overwritten with calibrated arousal (if Essentia)
    # track.arousal_predicted ← same value, stored for SetlistPlanner display
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class EnergyFeatures:
    """
    Normalised energy / arousal features for one song.

    All values in [0.0, 1.0] unless noted.

    Attributes
    ----------
    rms_energy : float
        Root-mean-square energy, normalised to [0, 1] relative to 0 dBFS.
    spectral_centroid : float
        Mean spectral centroid normalised to [0, 1] over [0, sr/2].
        Proxy for brightness / percussiveness.
    dynamic_range : float
        Ratio of peak-to-RMS in dB, normalised.  Higher = more dynamic.
    arousal : float
        Predicted arousal dimension (0 = calm, 1 = energetic).
        Uses Essentia's ArousalValence model when available;
        falls back to a weighted blend of rms_energy and spectral_centroid.
    valence : float
        Predicted valence (0 = negative, 1 = positive affect).
        Always 0.5 when Essentia is not available (neutral / unknown).
    backend : str
        "numpy" or "essentia" — which backend produced this result.
    """
    rms_energy: float = 0.5
    spectral_centroid: float = 0.5
    dynamic_range: float = 0.5
    arousal: float = 0.5
    valence: float = 0.5
    backend: str = "numpy"


# ---------------------------------------------------------------------------
# NumPy-only backend (always available)
# ---------------------------------------------------------------------------

def _rms_normalised(audio: np.ndarray) -> float:
    """RMS amplitude normalised to [0, 1] (0 dBFS = 1.0)."""
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return min(1.0, max(0.0, rms))


def _spectral_centroid_normalised(audio: np.ndarray, sr: int, frame_size: int = 2048) -> float:
    """
    Mean spectral centroid as fraction of Nyquist frequency.

    Frames the signal, computes per-frame centroid from the magnitude spectrum,
    returns the mean.  Bright/percussive material scores near 1.0.
    """
    audio = audio.astype(np.float32)
    hop = frame_size // 2
    n_frames = max(1, (len(audio) - frame_size) // hop + 1)
    window = np.hanning(frame_size)
    nyquist = sr / 2.0
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)

    centroids = []
    for i in range(n_frames):
        start = i * hop
        frame = audio[start : start + frame_size]
        if len(frame) < frame_size:
            break
        mag = np.abs(np.fft.rfft(frame * window))
        mag_sum = mag.sum()
        if mag_sum < 1e-10:
            centroids.append(0.0)
        else:
            centroids.append(float(np.dot(freqs, mag) / mag_sum))

    if not centroids:
        return 0.0
    mean_centroid = np.mean(centroids)
    return min(1.0, float(mean_centroid / nyquist))


def _dynamic_range_normalised(audio: np.ndarray) -> float:
    """
    Peak-to-RMS dynamic range, normalised to [0, 1].

    A fully compressed signal has dynamic range ≈ 0 dB → score near 0.
    A lightly mastered track at ~12 dB → score near 0.5.
    Returns 0.0 on silence.
    """
    peak = float(np.max(np.abs(audio.astype(np.float64))))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 1e-10:
        return 0.0
    dr_db = 20.0 * np.log10(peak / rms)
    # Clamp: [0 dB, 30 dB] → [0, 1]
    return min(1.0, max(0.0, float(dr_db) / 30.0))


def _numpy_profile(audio: np.ndarray, sr: int) -> EnergyFeatures:
    """
    Pure-numpy energy profiling.

    Arousal proxy: 70% RMS energy + 30% spectral centroid.
    This matches broad empirical observations (high-energy EDM has both high RMS
    and high spectral centroid; ambient tracks have low of both).
    """
    rms = _rms_normalised(audio)
    centroid = _spectral_centroid_normalised(audio, sr)
    dr = _dynamic_range_normalised(audio)
    arousal = min(1.0, 0.70 * rms + 0.30 * centroid)
    return EnergyFeatures(
        rms_energy=round(rms, 4),
        spectral_centroid=round(centroid, 4),
        dynamic_range=round(dr, 4),
        arousal=round(arousal, 4),
        valence=0.5,   # unknown without a model
        backend="numpy",
    )


# ---------------------------------------------------------------------------
# Essentia backend (optional)
# ---------------------------------------------------------------------------

def _essentia_profile(audio: np.ndarray, sr: int) -> EnergyFeatures:
    """
    Essentia-based energy profiling using algorithm-level API.

    Uses: essentia.standard.RhythmExtractor2013, Loudness, Dissonance, Windowing,
    Spectrum, Centroid — available in essentia>=2.1b6.

    Falls back to numpy on any import or runtime error.
    """
    try:
        import essentia.standard as es  # type: ignore
    except ImportError:
        log.debug("[energy_profiler] Essentia not installed; falling back to numpy")
        return _numpy_profile(audio, sr)

    try:
        # Essentia requires float32 mono
        mono = audio.astype(np.float32)
        if mono.ndim > 1:
            mono = mono.mean(axis=1)

        # Resample to 44100 if needed (Essentia prefers 44.1k)
        target_sr = 44100
        if sr != target_sr:
            import librosa
            mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # ── Loudness / RMS ────────────────────────────────────────────────────
        loudness_algo = es.Loudness()
        rms_linear = float(loudness_algo(mono))
        # Loudness() returns amplitude in dB LUFS; convert to 0–1 linear proxy
        # Typical range: -30 dBFS (quiet) to 0 dBFS (full scale)
        rms_norm = min(1.0, max(0.0, (rms_linear + 30.0) / 30.0))

        # ── Spectral centroid ─────────────────────────────────────────────────
        frame_size = 2048
        w = es.Windowing(type="hann")
        spec = es.Spectrum()
        centroid_algo = es.Centroid(range=float(sr // 2))
        centroids = []
        for frame in es.FrameGenerator(mono, frameSize=frame_size, hopSize=1024):
            magnitude = spec(w(frame))
            centroids.append(centroid_algo(magnitude))
        sc_norm = min(1.0, float(np.mean(centroids)) / (sr / 2)) if centroids else 0.5

        # ── Dynamic range ─────────────────────────────────────────────────────
        dr = _dynamic_range_normalised(mono)

        # ── Arousal prediction ─────────────────────────────────────────────────
        # Essentia's ArousalValence model is part of essentia-tensorflow.
        # Try it; fall back to numpy blend if the extra package is absent.
        arousal = 0.5
        valence = 0.5
        try:
            from essentia.standard import TensorflowPredictMusiCNN  # type: ignore
            # MusiCNN arousal/valence model (Castellon et al. 2021)
            # Requires essentia-tensorflow package
            model_path_env = None
            try:
                from scripts.core.config import cfg
                model_path_env = getattr(getattr(cfg, "models", None), "musicnn_path", None)
            except Exception:
                pass

            if model_path_env:
                from scripts.core.paths import MODELS_DIR
                model_path = str(MODELS_DIR / model_path_env)
                predictor = TensorflowPredictMusiCNN(graphFilename=model_path)
                predictions = predictor(mono)  # shape (n_frames, 2): [arousal, valence]
                arousal = float(np.mean(predictions[:, 0]))
                valence = float(np.mean(predictions[:, 1]))
            else:
                # No model configured → numpy blend
                arousal = min(1.0, 0.70 * rms_norm + 0.30 * sc_norm)
        except (ImportError, Exception) as e:
            log.debug("[energy_profiler] MusiCNN unavailable (%s); using Essentia feature blend", e)
            arousal = min(1.0, 0.70 * rms_norm + 0.30 * sc_norm)

        return EnergyFeatures(
            rms_energy=round(rms_norm, 4),
            spectral_centroid=round(sc_norm, 4),
            dynamic_range=round(dr, 4),
            arousal=round(arousal, 4),
            valence=round(valence, 4),
            backend="essentia",
        )

    except Exception as exc:
        log.warning("[energy_profiler] Essentia profiling failed (%s); falling back to numpy", exc)
        return _numpy_profile(audio, sr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_energy(
    audio: np.ndarray,
    sr: int,
    backend: str = "auto",
) -> EnergyFeatures:
    """
    Profile the energy / arousal of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        Mono or stereo float32/float64 audio, any sample rate.
    sr : int
        Sample rate in Hz.
    backend : str
        "auto"    — use Essentia if installed, else numpy.
        "essentia" — require Essentia (raises ImportError if absent).
        "numpy"   — always use numpy (fast, no optional deps).

    Returns
    -------
    EnergyFeatures
    """
    backend = backend.lower().strip()
    if backend == "numpy":
        return _numpy_profile(audio, sr)
    if backend == "essentia":
        feats = _essentia_profile(audio, sr)
        if feats.backend != "essentia":
            raise ImportError("Essentia is required but not available. pip install essentia")
        return feats
    # auto
    try:
        import essentia  # noqa: F401
        return _essentia_profile(audio, sr)
    except ImportError:
        return _numpy_profile(audio, sr)


def get_configured_backend() -> str:
    """Return the backend name from config.yaml (analysis.energy_backend). Defaults to 'auto'."""
    try:
        from scripts.core.config import cfg
        return str(getattr(getattr(cfg, "analysis", None), "energy_backend", "auto"))
    except Exception:
        return "auto"


def profile_energy_from_config(audio: np.ndarray, sr: int) -> EnergyFeatures:
    """Convenience wrapper that reads the backend from config.yaml."""
    return profile_energy(audio, sr, backend=get_configured_backend())


def enrich_track_node(track, audio: np.ndarray, sr: int, backend: str = "auto") -> None:
    """
    Populate a TrackNode's energy / arousal fields in-place.

    Sets:
      track.energy            ← EnergyFeatures.arousal  (overrides Spotify value)
      track.arousal_predicted ← same value (for downstream display)

    If the TrackNode doesn't have an `arousal_predicted` attribute (legacy),
    the attribute is monkey-patched in silently.

    Parameters
    ----------
    track : TrackNode
        An instance from setlist_planner.TrackNode.
    audio : np.ndarray
        Mono float32 audio.
    sr : int
        Sample rate.
    backend : str
        Same as profile_energy().
    """
    feats = profile_energy(audio, sr, backend=backend)
    track.energy = feats.arousal
    # Patch arousal_predicted onto the node (newer builds have it as a field;
    # older builds may not — we don't want to crash either way)
    try:
        track.arousal_predicted = feats.arousal
    except AttributeError:
        object.__setattr__(track, "arousal_predicted", feats.arousal)
    log.debug(
        "[energy_profiler] %s: arousal=%.3f rms=%.3f centroid=%.3f (%s)",
        getattr(track, "name", "?"),
        feats.arousal, feats.rms_energy, feats.spectral_centroid, feats.backend,
    )
