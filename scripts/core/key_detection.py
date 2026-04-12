"""
scripts/core/key_detection.py — Enhanced CQT Chroma-Based Key Detection

A specialised module for accurate musical key and mode detection using
Constant-Q Transform (CQT) chromagrams and Krumhansl-Schmuckler tonal
hierarchy profiles. Provides harmonic mixing helpers (Camelot wheel distance,
compatibility checking, pitch shift calculation).

Features:
  • Accurate key detection via CQT chroma (7 octaves, 36 bins/octave)
  • Harmonic Percussive Source Separation (HPSS) preprocessing
  • Complete 24-key correlation scoring (all major + minor modes)
  • Segment-wise key detection for tracks with key changes
  • Camelot wheel distance and compatibility utilities
  • Normalized 12-dimensional chroma vectors for music indexing

Usage:
    from scripts.core.key_detection import detect_key, detect_key_segments, KeyResult
    result = detect_key(audio, sr=22050)
    print(f"Detected: {result.key_name} {result.mode} ({result.camelot})")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"All correlations: {result.correlation_scores}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Musical constants ─────────────────────────────────────────────────────────

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Camelot Wheel — complete notation (outer ring B = major, inner ring A = minor)
CAMELOT: Dict[str, str] = {
    'C major':  '8B',   'G major':  '9B',   'D major': '10B',  'A major': '11B',
    'E major': '12B',   'B major':  '1B',   'F# major': '2B',  'C# major': '3B',
    'G# major': '4B',   'D# major': '5B',   'A# major': '6B',  'F major':  '7B',
    'A minor':  '8A',   'E minor':  '9A',   'B minor': '10A',  'F# minor': '11A',
    'C# minor': '12A',  'G# minor': '1A',   'D# minor': '2A',  'A# minor': '3A',
    'F minor':  '4A',   'C minor':  '5A',   'G minor':  '6A',  'D minor':  '7A',
}

# Krumhansl-Schmuckler tonal hierarchy profiles (normalised)
# These represent the likelihood of each semitone being the tonic in major/minor keys
_MAJOR_PROF = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                         2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64)
_MINOR_PROF = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                         2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)
_MAJOR_PROF /= _MAJOR_PROF.sum()
_MINOR_PROF /= _MINOR_PROF.sum()


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class KeyResult:
    """
    Result of musical key detection.

    Attributes:
        key_name: Root note name (e.g., 'A', 'C#')
        mode: Either 'major' or 'minor'
        camelot: Camelot wheel notation (e.g., '8A', '11B')
        confidence: Detection confidence [0.0, 1.0]
        correlation_scores: Dict mapping all 24 keys to correlation coefficients
        chroma_vector: 12-dimensional normalized chroma feature vector
    """
    key_name: str
    mode: str
    camelot: str
    confidence: float
    correlation_scores: Dict[str, float] = field(default_factory=dict)
    chroma_vector: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """JSON-serialisable dict."""
        return {
            'key_name': self.key_name,
            'mode': self.mode,
            'camelot': self.camelot,
            'confidence': float(self.confidence),
            'correlation_scores': {k: float(v) for k, v in self.correlation_scores.items()},
            'chroma_vector': [float(x) for x in self.chroma_vector],
        }


# ── Chroma extraction ─────────────────────────────────────────────────────────

def get_chroma_vector(audio: np.ndarray, sr: int,
                      method: str = 'cqt') -> np.ndarray:
    """
    Extract a normalized 12-dimensional chroma feature vector from audio.

    Useful for music indexing and similarity computations.

    Args:
        audio: Audio time-series (mono)
        sr: Sample rate (Hz)
        method: 'cqt' (high-res, accurate) or 'stft' (faster, less accurate)

    Returns:
        12-dimensional normalized chroma vector (sum = 1.0)
    """
    try:
        import librosa

        if method == 'cqt':
            # High-resolution CQT chroma: 7 octaves, 36 bins/octave
            chroma = librosa.feature.chroma_cqt(
                y=audio, sr=sr,
                n_octaves=7,
                bins_per_octave=36,
                fmin=librosa.note_to_hz('C1')
            )
        else:  # 'stft'
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

        # Average across time, then normalize
        c = chroma.mean(axis=1)
        c = np.clip(c, 0, None)  # ensure non-negative
        c_norm = c.sum()
        if c_norm > 1e-10:
            c = c / c_norm
        else:
            c = np.ones(12) / 12.0

        return c.astype(np.float32)

    except Exception as exc:
        log.warning('Chroma extraction failed: %s', exc)
        return np.ones(12, dtype=np.float32) / 12.0


def _apply_hpss(audio: np.ndarray, sr: int,
                 margin: float = 2.0) -> np.ndarray:
    """
    Apply Harmonic-Percussive Source Separation (HPSS) to extract the harmonic content.

    Args:
        audio: Audio time-series
        sr: Sample rate
        margin: HPSS margin parameter (higher = more aggressive separation)

    Returns:
        Harmonic component of the audio
    """
    try:
        import librosa

        D = librosa.stft(audio)
        H, P = librosa.decompose.hpss(D, margin=margin)
        harmonic = librosa.istft(H)
        return harmonic
    except Exception as exc:
        log.debug('HPSS preprocessing failed: %s. Using original audio.', exc)
        return audio


# ── Key detection ─────────────────────────────────────────────────────────────

def detect_key(audio: np.ndarray, sr: int,
               method: str = 'cqt') -> KeyResult:
    """
    Detect the musical key and mode of an audio signal.

    Uses CQT chroma with Harmonic-Percussive Source Separation (HPSS) preprocessing
    for robustness, then correlates the mean chroma against all 24 Krumhansl-Schmuckler
    rotated profiles (12 major + 12 minor keys).

    Args:
        audio: Audio time-series (mono)
        sr: Sample rate (Hz)
        method: 'cqt' (default, high-res) or 'stft' (faster, less accurate)

    Returns:
        KeyResult with key_name, mode, camelot, confidence, and full correlation scores
    """
    try:
        import librosa

        # Apply harmonic preprocessing for robustness
        harmonic = _apply_hpss(audio, sr, margin=2.0)

        # Extract chroma
        if method == 'cqt':
            chroma = librosa.feature.chroma_cqt(
                y=harmonic, sr=sr,
                n_octaves=7,
                bins_per_octave=36,
                fmin=librosa.note_to_hz('C1')
            )
        else:  # 'stft'
            chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr)

        # Average across time and normalize
        c = chroma.mean(axis=1)
        c = np.clip(c, 0, None)
        c_norm = c.sum()
        if c_norm > 1e-10:
            c = c / c_norm
        else:
            c = np.ones(12) / 12.0

        chroma_vector = c.astype(np.float32)

        # Compute correlations against all 24 keys
        correlation_scores: Dict[str, float] = {}
        best_val, best_key, best_mode = -np.inf, 0, 'major'

        for k in range(12):
            key_note = NOTE_NAMES[k]

            # Major key correlation
            rotated_major = np.roll(_MAJOR_PROF, k)
            maj_corr = float(np.corrcoef(c, rotated_major)[0, 1])
            if np.isnan(maj_corr):
                maj_corr = 0.0
            correlation_scores[f'{key_note} major'] = maj_corr

            if maj_corr > best_val:
                best_val, best_key, best_mode = maj_corr, k, 'major'

            # Minor key correlation
            rotated_minor = np.roll(_MINOR_PROF, k)
            min_corr = float(np.corrcoef(c, rotated_minor)[0, 1])
            if np.isnan(min_corr):
                min_corr = 0.0
            correlation_scores[f'{key_note} minor'] = min_corr

            if min_corr > best_val:
                best_val, best_key, best_mode = min_corr, k, 'minor'

        key_name = NOTE_NAMES[best_key]
        camelot = CAMELOT.get(f'{key_name} {best_mode}', '8B')
        confidence = float(np.clip((best_val + 1) / 2, 0.0, 1.0))

        return KeyResult(
            key_name=key_name,
            mode=best_mode,
            camelot=camelot,
            confidence=confidence,
            correlation_scores=correlation_scores,
            chroma_vector=chroma_vector.tolist(),
        )

    except Exception as exc:
        log.error('Key detection failed: %s', exc)
        return KeyResult(
            key_name='C',
            mode='major',
            camelot='8B',
            confidence=0.0,
            correlation_scores={},
            chroma_vector=[1/12] * 12,
        )


def detect_key_segments(audio: np.ndarray, sr: int,
                        segment_seconds: float = 30.0) -> List[KeyResult]:
    """
    Perform segment-wise key detection for tracks with key changes.

    Divides the audio into overlapping segments and detects the key in each,
    useful for identifying key modulations or key changes during breakdowns.

    Args:
        audio: Audio time-series (mono)
        sr: Sample rate (Hz)
        segment_seconds: Duration of each segment in seconds (default: 30.0)

    Returns:
        List of KeyResult objects, one per segment
    """
    segment_samples = int(segment_seconds * sr)
    if len(audio) < segment_samples:
        # Audio is shorter than one segment — return single detection
        return [detect_key(audio, sr)]

    results: List[KeyResult] = []
    # 50% overlap between segments
    hop_samples = segment_samples // 2

    start = 0
    while start < len(audio):
        end = min(start + segment_samples, len(audio))
        segment = audio[start:end]

        # Pad if this is the last segment and too short
        if len(segment) < segment_samples // 2:
            break

        result = detect_key(segment, sr)
        results.append(result)

        start += hop_samples

    return results if results else [detect_key(audio, sr)]


# ── Camelot wheel utilities ───────────────────────────────────────────────────

def camelot_distance(a: str, b: str) -> int:
    """
    Compute distance on the Camelot wheel between two Camelot codes.

    Distance represents the minimum number of "steps" on the wheel:
      • 0: same key
      • 1: adjacent keys (safe/smooth transition)
      • 2+: further apart (less ideal for harmonic mixing)

    Args:
        a: Camelot code (e.g., '8A', '11B')
        b: Camelot code

    Returns:
        Distance (0-6, with 6 being opposite)
    """
    try:
        # Parse Camelot codes: format is "<number><letter>" where number is 1-12
        num_a = int(a[:-1])
        letter_a = a[-1]
        num_b = int(b[:-1])
        letter_b = b[-1]

        # Within-ring distance (number difference on the wheel, min 0-6)
        num_dist = min(abs(num_a - num_b), 12 - abs(num_a - num_b))

        # Cross-ring distance: same letter = 0, different = 1 (but adds to number distance)
        if letter_a == letter_b:
            return num_dist
        else:
            # Different rings: add 1 step to cross the ring
            return num_dist + 1

    except (ValueError, IndexError):
        return 12  # Invalid codes


def camelot_compatible(a: str, b: str) -> bool:
    """
    Check if two Camelot codes are harmonically compatible for mixing.

    Compatible if:
      • Same key (distance 0)
      • Adjacent keys (distance 1)
      • Same number, different letter (relative major/minor)

    Args:
        a: Camelot code (e.g., '8A')
        b: Camelot code (e.g., '8B')

    Returns:
        True if keys are harmonically compatible
    """
    dist = camelot_distance(a, b)
    return dist <= 1


def pitch_shift_for_camelot(source: str, target: str) -> int:
    """
    Calculate the semitone shift needed to transpose from source to target Camelot position.

    Useful for beatmatching at a consistent pitch.

    Args:
        source: Source Camelot code (e.g., '8A')
        target: Target Camelot code (e.g., '11B')

    Returns:
        Number of semitones to shift (positive = up, negative = down, 0 = no shift)
    """
    try:
        # Parse Camelot codes
        num_s = int(source[:-1])
        letter_s = source[-1]
        num_t = int(target[:-1])
        letter_t = target[-1]

        # Map Camelot numbers to root notes: 8B = C major, 9B = G major, etc.
        # Camelot uses a circle of fifths offset
        camelot_to_semitone = {
            '1A': 9,   '1B': 7,    # G# minor, B major
            '2A': 4,   '2B': 2,    # D# minor, F# major
            '3A': 11,  '3B': 9,    # B minor, C# major
            '4A': 6,   '4B': 4,    # F minor, G# major
            '5A': 1,   '5B': 11,   # C minor, D# major
            '6A': 8,   '6B': 6,    # G minor, A# major
            '7A': 3,   '7B': 1,    # D minor, F major
            '8A': 10,  '8B': 0,    # A minor, C major
            '9A': 5,   '9B': 7,    # E minor, G major
            '10A': 0,  '10B': 2,   # B minor, D major
            '11A': 7,  '11B': 9,   # F# minor, A major
            '12A': 2,  '12B': 4,   # C# minor, E major
        }

        semitone_s = camelot_to_semitone.get(source, 0)
        semitone_t = camelot_to_semitone.get(target, 0)

        shift = semitone_t - semitone_s
        # Normalise to ±6 semitones (prefer smaller transposition)
        if shift > 6:
            shift -= 12
        elif shift < -6:
            shift += 12

        return int(shift)

    except (ValueError, KeyError):
        return 0
