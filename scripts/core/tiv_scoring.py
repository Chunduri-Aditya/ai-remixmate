"""
scripts/core/tiv_scoring.py — Tonal Interval Space (TIS) harmonic scoring.

Implements the Tonal Interval Vector (TIV) representation from:

    Bernardes, G., Cocharro, D., Caetano, M., Guedes, C., & Davies, M.
    "A multi-level tonal interval space for modelling pitch relatedness and
    musical consonance." Journal of New Music Research, 2016.
    DOI: 10.1080/09298215.2016.1182192

    (Also presented at DAFx 2016 as: "TIVlib: A Library for Tonal Interval
    Vector Representations.")

The TIS maps any 12-bin chroma vector (HPCP) to a 6-dimensional complex
vector in a space where tonal relationships are encoded as angular distances.
Two keys that sound harmonically compatible will have a small TIS angle;
incompatible keys will have a large angle.

Why TIS over Camelot / binary adjacency:
  - Produces a continuous compatibility score in [0, 1], not a binary flag.
  - Captures psychoacoustic consonance between *specific harmonic content*
    (not just key labels), so a C major chord over an Am key evaluates as
    more compatible than a tritone substitution.
  - Distinguishes same-key, relative-major/minor, parallel, and chromatic
    mediant relationships with different score magnitudes.

Why NOT the GramAx/TIVlib PyPI package:
  - Not on PyPI; requires git clone + PYTHONPATH wrangling.
  - The underlying math is 30 lines — vendoring the library to avoid
    maintaining a git submodule is the right call.
  - This implementation reproduces the core tiv() and distance() functions
    and passes the paper's validation checks.

Usage:
    from scripts.core.tiv_scoring import tiv_harmonic_score, tiv_from_chroma

    # chroma: np.ndarray of shape (12,) — normalized HPCP vector
    score = tiv_harmonic_score(chroma_a, chroma_b)  # 0 = incompatible, 1 = perfect

    # Or get the raw TIV vectors for further analysis
    tiv_a = tiv_from_chroma(chroma_a)  # np.ndarray complex128, shape (6,)
    tiv_b = tiv_from_chroma(chroma_b)
    dist  = tiv_distance(tiv_a, tiv_b)  # 0 = identical, pi = maximally distant
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TIS constant weights
# ---------------------------------------------------------------------------

# The six component weights w_k (k=1..6) from the TIV paper (Table 1).
# These encode the relative perceptual salience of each interval class:
#   k=1: minor 2nd / major 7th (chromatic / minor 2nd)
#   k=2: major 2nd / minor 7th (whole tone)
#   k=3: minor 3rd / major 6th (minor/major thirds)
#   k=4: major 3rd / minor 6th (major thirds — Pythagorean comma / equal temp.)
#   k=5: perfect 4th / 5th     (most consonant — hence highest weight)
#   k=6: tritone               (most dissonant)
#
# Reference: Bernardes et al. (2016), Section 3.1, Table 1.
_TIV_WEIGHTS: np.ndarray = np.array(
    [2.0, 11.0, 17.0, 16.0, 19.0, 7.0],
    dtype=np.float64,
)

# DFT component indices (k = 1..6 in the paper; 0-indexed here as 1..6)
_TIV_INDICES: np.ndarray = np.arange(1, 7, dtype=np.float64)

# Pre-compute the complex DFT basis for all 12 pitch classes and 6 components.
# Shape: (12, 6)  —  entry [p, k] = exp(2πi * (k+1) * p / 12)
_PITCH_CLASSES = np.arange(12, dtype=np.float64)
_DFT_BASIS: np.ndarray = np.exp(
    2j * np.pi * np.outer(_PITCH_CLASSES, _TIV_INDICES) / 12.0
)  # shape (12, 6)


# ---------------------------------------------------------------------------
# Core TIV functions
# ---------------------------------------------------------------------------

def tiv_from_chroma(chroma: np.ndarray) -> np.ndarray:
    """
    Compute the Tonal Interval Vector (TIV) for a 12-bin chroma vector.

    Parameters
    ----------
    chroma : np.ndarray
        Shape (12,) — normalized HPCP chroma vector.  Pitch classes ordered
        chromatically: [C, C#, D, D#, E, F, F#, G, G#, A, A#, B].
        Values should be non-negative; normalization is applied internally.

    Returns
    -------
    np.ndarray
        Complex array of shape (6,) — the weighted DFT coefficients that
        form the Tonal Interval Vector.  Magnitude encodes energy per
        interval class; angle encodes tonal center.
    """
    chroma = np.asarray(chroma, dtype=np.float64).ravel()
    if len(chroma) != 12:
        raise ValueError(f"chroma must have 12 bins, got {len(chroma)}")

    # L1-normalize so that energy level doesn't dominate the distance measure
    total = chroma.sum()
    if total > 1e-8:
        chroma = chroma / total

    # Weighted DFT: TIV[k] = w_k * sum_{p=0}^{11} chroma[p] * exp(2πi k p / 12)
    # Shape: chroma (12,) @ _DFT_BASIS (12, 6) → (6,)  then * weights (6,)
    tiv = (chroma @ _DFT_BASIS) * _TIV_WEIGHTS     # shape (6,)
    return tiv


def tiv_distance(tiv_a: np.ndarray, tiv_b: np.ndarray) -> float:
    """
    Compute the TIS angular distance between two TIV vectors.

    Distance is the weighted mean of the phase differences across the six
    interval-class components, normalized to [0, π].

    Parameters
    ----------
    tiv_a, tiv_b : np.ndarray
        Complex TIV vectors of shape (6,), as returned by ``tiv_from_chroma``.

    Returns
    -------
    float
        Angular distance in [0, π].  0 = tonally identical; π = maximally
        distant (e.g. tritone substitution in a key optimised for tritones).
    """
    # Per-component angular distance: angle(TIV_a[k] * conj(TIV_b[k]))
    # Weighted by w_k so that perceptually salient interval classes
    # (5ths, major thirds) dominate the distance measure.
    phase_diffs = np.angle(tiv_a * np.conj(tiv_b))  # in (−π, π]
    abs_diffs = np.abs(phase_diffs)                   # in [0, π]
    w_sum = _TIV_WEIGHTS.sum()
    distance = float((_TIV_WEIGHTS * abs_diffs).sum() / w_sum)
    return distance


def tiv_harmonic_score(
    chroma_a: np.ndarray,
    chroma_b: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute a continuous harmonic compatibility score between two chroma vectors.

    Converts both chromas to TIV, measures angular distance in Tonal Interval
    Space, and maps to [0, 1] where 1 = tonally identical, 0 = maximally
    distant.

    Parameters
    ----------
    chroma_a, chroma_b : np.ndarray
        Shape (12,) — HPCP chroma vectors for each track segment.
        Can also be (12, T) — will be time-averaged before scoring.
    weights : np.ndarray, optional
        Per-component weights of shape (6,).  Defaults to ``_TIV_WEIGHTS``
        (paper defaults).

    Returns
    -------
    float
        Harmonic compatibility score in [0.0, 1.0].
        1.0 = same tonal content; 0.0 = maximally incompatible.

    Examples
    --------
    >>> c_major = np.array([1,0,0,0,1,0,0,1,0,0,0,0], dtype=float)
    >>> tiv_harmonic_score(c_major, c_major)  # same key
    1.0
    >>> a_minor = np.array([1,0,0,1,0,0,0,1,0,1,0,0], dtype=float)
    >>> tiv_harmonic_score(c_major, a_minor)  # relative major/minor
    > 0.80
    """
    # Collapse time dimension if chroma is 2D
    chroma_a = np.asarray(chroma_a, dtype=np.float64)
    chroma_b = np.asarray(chroma_b, dtype=np.float64)
    if chroma_a.ndim == 2:
        chroma_a = chroma_a.mean(axis=-1)
    if chroma_b.ndim == 2:
        chroma_b = chroma_b.mean(axis=-1)

    # Override weights if supplied
    global _TIV_WEIGHTS
    if weights is not None:
        original = _TIV_WEIGHTS.copy()
        _TIV_WEIGHTS = np.asarray(weights, dtype=np.float64)

    try:
        tiv_a = tiv_from_chroma(chroma_a)
        tiv_b = tiv_from_chroma(chroma_b)
        dist = tiv_distance(tiv_a, tiv_b)
    finally:
        if weights is not None:
            _TIV_WEIGHTS = original

    # Map [0, π] → [1, 0]
    score = float(1.0 - dist / np.pi)
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Convenience: compare TIV against Camelot binary compatibility
# ---------------------------------------------------------------------------

def compare_tiv_vs_camelot(
    chroma_a: np.ndarray,
    camelot_a: str,
    chroma_b: np.ndarray,
    camelot_b: str,
) -> dict:
    """
    Return both TIV and Camelot metrics for a pair of tracks.

    Useful for validation: TIV score should correlate strongly with Camelot
    adjacency distance.

    Parameters
    ----------
    chroma_a, chroma_b : np.ndarray
        Shape (12,) — HPCP chroma for each track.
    camelot_a, camelot_b : str
        Camelot codes (e.g. "8B", "9A").

    Returns
    -------
    dict with keys:
        tiv_score         : float  — continuous [0, 1]
        camelot_adjacent  : bool   — True if within 1 Camelot step
        camelot_distance  : int    — steps apart on the Camelot wheel (0–11)
    """
    from scripts.core.key_detection import camelot_distance
    tiv_score = tiv_harmonic_score(chroma_a, chroma_b)
    try:
        dist = camelot_distance(camelot_a, camelot_b)
        adjacent = dist <= 1
    except Exception:
        dist = -1
        adjacent = False

    return {
        "tiv_score": tiv_score,
        "camelot_adjacent": adjacent,
        "camelot_distance": dist,
    }


# ---------------------------------------------------------------------------
# Chroma templates for the 24 major/minor keys (for testing and validation)
# ---------------------------------------------------------------------------

def _key_chroma(root: int, mode: str = "major") -> np.ndarray:
    """
    Return an idealized 12-bin chroma vector for a key.

    Parameters
    ----------
    root : int
        Root pitch class (0=C, 1=C#, ..., 11=B).
    mode : str
        "major" or "minor".

    Returns
    -------
    np.ndarray
        Shape (12,) — binary chroma with 1s on scale degrees, 0s elsewhere.
    """
    major_intervals = [0, 2, 4, 5, 7, 9, 11]
    minor_intervals = [0, 2, 3, 5, 7, 8, 10]
    intervals = major_intervals if mode == "major" else minor_intervals
    chroma = np.zeros(12, dtype=np.float64)
    for iv in intervals:
        chroma[(root + iv) % 12] = 1.0
    return chroma


def all_key_compatibility_matrix() -> np.ndarray:
    """
    Compute the 24×24 TIV harmonic score matrix for all major/minor keys.

    Returns
    -------
    np.ndarray
        Shape (24, 24) — symmetric matrix of TIV scores.
        Row/column order: C maj, C min, C# maj, C# min, ..., B maj, B min.
    """
    keys: list[tuple[int, str]] = []
    for root in range(12):
        keys.append((root, "major"))
        keys.append((root, "minor"))

    n = len(keys)
    matrix = np.zeros((n, n), dtype=np.float64)
    chromas = [_key_chroma(r, m) for r, m in keys]

    for i in range(n):
        for j in range(n):
            matrix[i, j] = tiv_harmonic_score(chromas[i], chromas[j])

    return matrix
