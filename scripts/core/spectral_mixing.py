"""
scripts/core/spectral_mixing.py — Masking-aware multiband EQ for DJ transitions.

Generalizes the bass-only dynamic EQ in dj_engine.py (_stem_bass_ramp() and
_apply_dynamic_eq_fade()'s 3-band crossover) to the rest of the spectrum:
during a crossfade, two tracks (or two stems of the same type — vocals,
drums, other) can each occupy frequency bands the other needs, producing
audible masking that a flat amplitude crossfade doesn't address. This module
detects that per-band collision and applies a smoothed, capped attenuation to
whichever track is "wasting" a band the other track needs there.

Algorithm reference: Sina Hafezi & Joshua D. Reiss, "Autonomous Multitrack
Equalization Based on Masking Reduction," J. Audio Eng. Soc. 63(5):312-323,
May 2015, DOI 10.17743/jaes.2015.0021 (Queen Mary University of London) —
verified as a real, published paper (not fabricated) before this module was
written. This is the OFFLINE variant of their system: this is an offline
render engine (no live-performance latency budget), so we deliberately do NOT
adopt their real-time constraints (causal-only envelope followers, capped
lookahead, small hop sizes chosen for latency). Instead:
  - whole-window two-pass analysis (look at the entire transition before
    rendering any of it), and
  - zero-phase smoothing (scipy.signal.filtfilt) over the gain-vs-time curve,
which are both illegal in a real-time system but strictly better here.

Equation mapping (their offline variant, adapted):
  Eq. (1) — masking metric: M_AB(f,t) = X_A(f,t) - X_B(f,t) dB, counted only
            where band f is NON-essential to masker A (not among A's
            rank_thresh loudest bands at time t) but IS essential to maskee B
            (among B's rank_thresh loudest bands at time t).
  Eq. (2) — gain: B = -2**strength * M, clamped to +/- max_atten_db.
  Eq. (4)/(5) — temporal smoothing: EMA with tau (default 2s), or (offline-only
            improvement) zero-phase filtfilt over the same time constant.

Deliberately excluded from this module (see the research brief this was built
from): phase-vocoder cross-synthesis / spectral morphing (solves a different
problem — timbre blending, not masking, and introduces phasiness), and any
GAN/learned model (needs training data + a deep-learning stack this project
doesn't otherwise depend on).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

# 5-band DJ-standard split (Hz). Index 0 is the band _stem_bass_ramp() and the
# existing dynamic low-shelf already own exclusively — always pass
# protect_bands=[0] (or more) from dj_engine.py so this module never competes
# with that proven bass-clash fix.
BANDS_HZ: Tuple[Tuple[float, float], ...] = (
    (20.0, 200.0),      # sub/bass — protected by default at the dj_engine call site
    (200.0, 500.0),     # low-mid
    (500.0, 2000.0),    # mid
    (2000.0, 6000.0),   # high-mid — the vocal-presence "2-4 kHz" collision zone
    (6000.0, 20000.0),  # high
)

_DEFAULT_N_FFT = 2048
_DEFAULT_HOP = 512


def compute_band_energies(
    stem: np.ndarray,
    sr: int,
    n_fft: int = _DEFAULT_N_FFT,
    hop: int = _DEFAULT_HOP,
    bands: Sequence[Tuple[float, float]] = BANDS_HZ,
) -> np.ndarray:
    """
    STFT -> per-band, per-frame RMS energy (dB). Returns shape [n_bands, n_frames].

    Each band's energy is the RMS magnitude across the FFT bins falling
    inside that band's frequency range, converted to dB (linear-Hz
    DJ-standard split, not Bark/ERB — see module docstring for why that's
    sufficient for pass 1).

    IMPORTANT: this averages in the LINEAR magnitude/power domain, then
    converts to dB — not the other way around. A band can span hundreds of
    FFT bins where only one or two actually carry a tone; averaging
    already-logged dB values across those bins lets the many near-silent
    bins (each near the -160 dB floor) drag the mean down to near-silence
    even when the band clearly has audible energy. Caught by
    tests/test_behavioral.py::TestMaskingAwareMultibandEQ during initial
    verification — the first version of this function did exactly that.
    """
    import librosa

    if stem is None or len(stem) == 0:
        return np.zeros((len(bands), 0), dtype=np.float32)

    D = librosa.stft(np.asarray(stem, dtype=np.float32), n_fft=n_fft, hop_length=hop)
    mag = np.abs(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    n_frames = mag.shape[1]
    out = np.zeros((len(bands), n_frames), dtype=np.float32)
    for i, (lo, hi) in enumerate(bands):
        bin_mask = (freqs >= lo) & (freqs < hi)
        if not np.any(bin_mask):
            continue
        band_power = np.mean(mag[bin_mask] ** 2, axis=0)
        out[i] = 20.0 * np.log10(np.sqrt(band_power) + 1e-8)
    return out


def _rank_within_track(band_energies_db: np.ndarray) -> np.ndarray:
    """
    Per-frame rank of each band's energy within one track (rank 1 = loudest).
    Shape in == shape out: [n_bands, n_frames].
    """
    n_bands, n_frames = band_energies_db.shape
    if n_bands == 0 or n_frames == 0:
        return np.zeros_like(band_energies_db, dtype=np.int32)
    # argsort descending along the band axis -> rank 1 for the loudest band
    order = np.argsort(-band_energies_db, axis=0)
    band_positions = np.argsort(order, axis=0)  # position of each band in the sort
    return (band_positions + 1).astype(np.int32)


def masking_gain_curve(
    masker_db: np.ndarray,
    maskee_db: np.ndarray,
    strength: float = 0.0,
    max_atten_db: float = 6.0,
    rank_thresh: int = 3,
) -> np.ndarray:
    """
    Hafezi & Reiss Eq.(1)/(2): per-band, per-frame attenuation (dB, <= 0) to
    apply to the MASKER, where it dominates a band that's non-essential to
    itself (not among its rank_thresh loudest bands at that frame) but IS
    essential to the maskee (among the maskee's rank_thresh loudest bands).

    masker_db / maskee_db: [n_bands, n_frames] dB magnitude (compute_band_energies()).
    Returns: [n_bands, n_frames] gain in dB, always <= 0, magnitude <= max_atten_db.
    """
    n_bands = min(masker_db.shape[0], maskee_db.shape[0])
    n_frames = min(masker_db.shape[1], maskee_db.shape[1])
    if n_bands == 0 or n_frames == 0:
        return np.zeros((n_bands, n_frames), dtype=np.float32)

    masker_db = masker_db[:n_bands, :n_frames]
    maskee_db = maskee_db[:n_bands, :n_frames]

    masker_rank = _rank_within_track(masker_db)
    maskee_rank = _rank_within_track(maskee_db)

    # Band is non-essential to masker (rank > rank_thresh) but essential to
    # maskee (rank <= rank_thresh) — Hafezi & Reiss's essentiality condition.
    essential_to_maskee = maskee_rank <= rank_thresh
    non_essential_to_masker = masker_rank > rank_thresh
    masking_region = essential_to_maskee & non_essential_to_masker

    # Eq.(1): M = X_masker - X_maskee (dB), only where masker actually louder
    # in that band (a negative M would mean "no masking", so clip at 0 before
    # gain conversion).
    m = np.clip(masker_db - maskee_db, 0.0, None)
    m = np.where(masking_region, m, 0.0)

    # Eq.(2): B = -2**strength * M, clamped to +/- max_atten_db.
    gain_db = -(2.0 ** strength) * m
    gain_db = np.clip(gain_db, -abs(max_atten_db), 0.0)
    return gain_db.astype(np.float32)


def smooth_gain_curve(
    gain_db: np.ndarray,
    sr: int,
    hop: int = _DEFAULT_HOP,
    tau_s: float = 2.0,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Hafezi & Reiss Eq.(4)/(5): temporal smoothing of the per-band gain curve
    to avoid zipper artifacts from frame-to-frame jumps.

    zero_phase=True (default, offline-only — illegal in a real-time system):
      scipy.signal.filtfilt with a single-pole lowpass matched to tau_s.
    zero_phase=False: causal EMA, y[n] = (1-a)x[n] + a*y[n-1],
      a = exp(-1 / (tau_s * frame_rate)).

    gain_db: [n_bands, n_frames]. Returns the same shape.
    """
    if gain_db.size == 0:
        return gain_db

    frame_rate = sr / float(hop)
    alpha = float(np.exp(-1.0 / max(tau_s * frame_rate, 1e-6)))

    if not zero_phase:
        out = np.empty_like(gain_db)
        out[:, 0] = gain_db[:, 0]
        for t in range(1, gain_db.shape[1]):
            out[:, t] = (1.0 - alpha) * gain_db[:, t] + alpha * out[:, t - 1]
        return out

    try:
        from scipy.signal import filtfilt

        # Single-pole IIR lowpass matched to the same time constant, applied
        # zero-phase (forward-backward) — no lag, which a causal EMA can't do.
        b = [1.0 - alpha]
        a = [1.0, -alpha]
        # filtfilt's default padlen is 3 * max(len(a), len(b)) = 6 and it
        # requires the signal to be strictly longer than that; fall back to
        # EMA for very short transition windows.
        if gain_db.shape[1] <= 3 * max(len(a), len(b)):
            return smooth_gain_curve(gain_db, sr, hop, tau_s, zero_phase=False)
        return filtfilt(b, a, gain_db, axis=1).astype(np.float32)
    except (ImportError, ValueError):
        return smooth_gain_curve(gain_db, sr, hop, tau_s, zero_phase=False)


def _taper_edges(
    gain_db: np.ndarray,
    sr: int,
    hop: int = _DEFAULT_HOP,
    edge_s: float = 0.05,
) -> np.ndarray:
    """
    Scale a [n_bands, n_frames] gain curve toward 0 dB at both window edges
    with a linear ramp over ``edge_s`` seconds (in frames). Guarantees the
    processed window is bit-identical-in-level to the unprocessed stem at the
    splice boundaries, whatever the masking analysis said about frame 0.
    """
    n_frames = gain_db.shape[1]
    edge_frames = max(2, int(round(edge_s * sr / hop)))
    if n_frames < 2 * edge_frames:
        edge_frames = max(1, n_frames // 2)

    taper = np.ones(n_frames, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, edge_frames + 1, dtype=np.float32)[:-1]
    taper[:edge_frames] = ramp
    taper[n_frames - edge_frames:] = ramp[::-1]
    return gain_db * taper[None, :]


def apply_multiband_gain(
    stem: np.ndarray,
    sr: int,
    gain_db: np.ndarray,
    n_fft: int = _DEFAULT_N_FFT,
    hop: int = _DEFAULT_HOP,
    bands: Sequence[Tuple[float, float]] = BANDS_HZ,
) -> np.ndarray:
    """
    Apply a time-varying per-band gain (dB, [n_bands, n_frames]) to `stem` via
    STFT magnitude multiply + istft. Phase is preserved from the original
    signal (only magnitude is scaled), matching the resynthesis approach used
    by DJtransGAN and standard STFT-domain EQ.
    """
    import librosa

    if stem is None or len(stem) == 0 or gain_db.size == 0:
        return stem

    D = librosa.stft(np.asarray(stem, dtype=np.float32), n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(D), np.angle(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    n_frames = min(D.shape[1], gain_db.shape[1])
    linear_gain = np.ones_like(mag)
    for i, (lo, hi) in enumerate(bands):
        if i >= gain_db.shape[0]:
            break
        bin_mask = (freqs >= lo) & (freqs < hi)
        if not np.any(bin_mask):
            continue
        band_gain_db = gain_db[i, :n_frames]
        band_gain_lin = 10.0 ** (band_gain_db / 20.0)
        linear_gain[np.ix_(bin_mask, range(n_frames))] = band_gain_lin[None, :]

    D_out = mag * linear_gain * np.exp(1j * phase)
    out = librosa.istft(D_out, hop_length=hop, length=len(stem))
    return out.astype(np.float32)


def unmask_stem_pair(
    track_a: Optional[np.ndarray],
    track_b: Optional[np.ndarray],
    sr: int,
    strength: float = 0.0,
    max_atten_db: float = 6.0,
    rank_thresh: int = 3,
    protect_bands: Optional[List[int]] = None,
    n_fft: int = _DEFAULT_N_FFT,
    hop: int = _DEFAULT_HOP,
    bands: Sequence[Tuple[float, float]] = BANDS_HZ,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Full pipeline for one stem-type pair (e.g. vocals_a vs vocals_b) over a
    transition window. Reduces masking BIDIRECTIONALLY: track A is ducked in
    bands where it masks something B needs, and symmetrically B is ducked
    where it masks something A needs. This differs slightly from a single
    fixed masker/maskee assignment — during a crossfade neither track is
    exclusively "the masker" for the whole window (dominance flips as the
    fade progresses), so both directions are computed and applied.

    protect_bands: indices into `bands` to exclude entirely (gain forced to
    0 dB) — pass [0] (or more) to guarantee this module never touches the
    sub/bass range, leaving that exclusively to dj_engine.py's
    _stem_bass_ramp() / dynamic low-shelf.

    Returns (processed_a, processed_b). Either element is None (unchanged)
    if the corresponding input was None.
    """
    if track_a is None and track_b is None:
        return None, None
    if track_a is None or track_b is None:
        # Nothing to compare against — pass through unchanged.
        return track_a, track_b
    if len(track_a) == 0 or len(track_b) == 0:
        return track_a, track_b

    a_bands = compute_band_energies(track_a, sr, n_fft=n_fft, hop=hop, bands=bands)
    b_bands = compute_band_energies(track_b, sr, n_fft=n_fft, hop=hop, bands=bands)

    if a_bands.shape[1] == 0 or b_bands.shape[1] == 0:
        return track_a, track_b

    # A as masker, B as maskee -> attenuation to apply to A.
    gain_a = masking_gain_curve(a_bands, b_bands, strength=strength,
                                max_atten_db=max_atten_db, rank_thresh=rank_thresh)
    gain_a = smooth_gain_curve(gain_a, sr, hop=hop)

    # B as masker, A as maskee -> attenuation to apply to B.
    gain_b = masking_gain_curve(b_bands, a_bands, strength=strength,
                                max_atten_db=max_atten_db, rank_thresh=rank_thresh)
    gain_b = smooth_gain_curve(gain_b, sr, hop=hop)

    if protect_bands:
        for idx in protect_bands:
            if 0 <= idx < gain_a.shape[0]:
                gain_a[idx, :] = 0.0
            if 0 <= idx < gain_b.shape[0]:
                gain_b[idx, :] = 0.0

    # Taper the gain curves to exactly 0 dB at the window edges (~50 ms
    # ramp). Callers splice the processed window back into an unprocessed
    # stem; without this, a band can be attenuated up to max_atten_db on the
    # very first frame, producing a level step right at the splice point.
    gain_a = _taper_edges(gain_a, sr, hop)
    gain_b = _taper_edges(gain_b, sr, hop)

    processed_a = apply_multiband_gain(track_a, sr, gain_a, n_fft=n_fft, hop=hop, bands=bands)
    processed_b = apply_multiband_gain(track_b, sr, gain_b, n_fft=n_fft, hop=hop, bands=bands)

    # apply_multiband_gain resynthesizes at the original length via istft's
    # length= param, but guard against off-by-one drift before handing back
    # to callers that expect exact-length arrays.
    processed_a = processed_a[: len(track_a)] if processed_a is not None else track_a
    processed_b = processed_b[: len(track_b)] if processed_b is not None else track_b

    return processed_a, processed_b
