"""
scripts/core/audio_enhance.py — Pre-Demucs audio enhancement pipeline.

Applies a lightweight signal-processing chain to maximise Demucs stem
quality before separation and normalises each individual stem afterward.

Enhancement chain (in order):
  1. DC-offset removal + subsonic high-pass (20 Hz Butterworth)
  2. Brick-wall low-pass at 20 kHz  (anti-aliasing / remove digital noise)
  3. Noise gate  (silence passages below threshold)
  4. Gentle soft-knee RMS compression  (3:1, -20 dBFS threshold)
  5. "Air" high-shelf boost  (+1.5 dB @ 12 kHz) for presence / clarity
  6. LUFS normalisation to -14 LUFS  (Streaming / Demucs sweet-spot)
  7. True-peak limiter at -1 dBFS

All processing is CPU-only (numpy + scipy); no GPU required.

Usage:
    from scripts.core.audio_enhance import enhance_audio, EnhanceOptions, enhance_stems
    enhanced, report = enhance_audio(audio_array, sr=44100)
    stem_dict = enhance_stems({'vocals': voc_arr, 'drums': drm_arr, ...}, sr)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EnhanceOptions:
    """Knobs for the enhancement chain.  All can be disabled individually."""
    # Stage toggles
    hp_filter:     bool = True      # 20 Hz subsonic cut
    lp_filter:     bool = True      # 20 kHz anti-alias cut
    noise_gate:    bool = True      # silence very quiet passages
    compression:   bool = True      # gentle RMS compression
    air_eq:        bool = True      # high-shelf presence boost
    lufs_target:   float = -14.0    # set to 0 to skip normalisation
    true_peak_ceil: float = -1.0    # dBFS ceiling, 0 to skip

    # Stage parameters
    hp_cutoff_hz:        float = 20.0
    lp_cutoff_hz:        float = 20_000.0
    gate_threshold_db:   float = -70.0   # gate opens above this
    gate_release_ms:     float = 150.0
    comp_threshold_db:   float = -20.0
    comp_ratio:          float = 3.0
    comp_attack_ms:      float = 10.0
    comp_release_ms:     float = 150.0
    comp_knee_db:        float = 6.0
    comp_makeup_db:      float = 2.0
    air_freq_hz:         float = 12_000.0
    air_gain_db:         float = 1.5


@dataclass
class EnhanceReport:
    rms_before_db:   float = 0.0
    rms_after_db:    float = 0.0
    peak_before_db:  float = 0.0
    peak_after_db:   float = 0.0
    lufs_before:     float = 0.0
    lufs_after:      float = 0.0
    gain_applied_db: float = 0.0
    clipped:         bool  = False
    stages_applied:  list  = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio ** 2))
    return float(20 * np.log10(max(rms, 1e-9)))


def _peak_db(audio: np.ndarray) -> float:
    return float(20 * np.log10(max(np.max(np.abs(audio)), 1e-9)))


def _butter_filter(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float,
    btype: str,
    order: int = 4,
) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    nyq = sr / 2.0
    norm_cut = np.clip(cutoff_hz / nyq, 1e-4, 0.999)
    sos = butter(order, norm_cut, btype=btype, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 1 — Subsonic HP filter
# ---------------------------------------------------------------------------

def apply_hp_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 20.0) -> np.ndarray:
    """Remove DC offset and infrasonic rumble below `cutoff_hz`."""
    return _butter_filter(audio, sr, cutoff_hz, btype="highpass", order=4)


# ---------------------------------------------------------------------------
# Stage 2 — Anti-alias LP filter
# ---------------------------------------------------------------------------

def apply_lp_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 20_000.0) -> np.ndarray:
    """Remove digital noise above `cutoff_hz`."""
    if cutoff_hz >= sr / 2 * 0.98:
        return audio  # nothing to cut at this sample rate
    return _butter_filter(audio, sr, cutoff_hz, btype="lowpass", order=2)


# ---------------------------------------------------------------------------
# Stage 3 — Noise gate
# ---------------------------------------------------------------------------

def apply_noise_gate(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -70.0,
    release_ms: float = 150.0,
) -> np.ndarray:
    """
    Mute samples whose short-term RMS envelope is below `threshold_db`.
    Uses a one-pole smoother for the gain envelope.
    """
    hop       = max(1, int(sr * 0.010))      # 10 ms frames
    threshold = 10.0 ** (threshold_db / 20.0)
    release   = np.exp(-1.0 / (sr * release_ms / 1000.0 / hop))

    out    = audio.copy()
    gain   = 1.0
    n_hops = (len(audio) + hop - 1) // hop

    for i in range(n_hops):
        start = i * hop
        end   = min(start + hop, len(audio))
        rms   = float(np.sqrt(np.mean(out[start:end] ** 2)))
        target = 1.0 if rms >= threshold else 0.0
        gain   = release * gain + (1.0 - release) * target
        out[start:end] *= gain

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 4 — Gentle soft-knee RMS compressor
# ---------------------------------------------------------------------------

def apply_compression(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -20.0,
    ratio: float = 3.0,
    attack_ms: float = 10.0,
    release_ms: float = 150.0,
    knee_db: float = 6.0,
    makeup_db: float = 2.0,
) -> np.ndarray:
    """
    Soft-knee RMS compressor.
    All gain-decision math is in dB; application is linear.
    """
    hop        = max(1, int(sr * 0.005))   # 5 ms frames
    attack_c   = np.exp(-1.0 / (sr * attack_ms  / 1000.0 / hop))
    release_c  = np.exp(-1.0 / (sr * release_ms / 1000.0 / hop))
    makeup     = 10.0 ** (makeup_db / 20.0)
    knee_half  = knee_db / 2.0

    out        = audio.copy()
    n_hops     = (len(audio) + hop - 1) // hop
    gain_db    = 0.0   # smoothed gain reduction in dB

    for i in range(n_hops):
        start = i * hop
        end   = min(start + hop, len(audio))
        rms   = float(np.sqrt(np.mean(out[start:end] ** 2)))
        lvl   = 20.0 * np.log10(max(rms, 1e-9))

        # Soft-knee gain computation
        diff = lvl - threshold_db
        if diff < -knee_half:
            target_gr = 0.0
        elif diff > knee_half:
            target_gr = (diff * (1.0 - 1.0 / ratio))
        else:
            k = diff + knee_half
            target_gr = k * k / (2.0 * knee_db) * (1.0 - 1.0 / ratio)

        # Smooth the gain reduction
        coeff  = attack_c if target_gr > gain_db else release_c
        gain_db = coeff * gain_db + (1.0 - coeff) * target_gr

        out[start:end] *= (10.0 ** (-gain_db / 20.0)) * makeup

    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 5 — "Air" high-shelf EQ
# ---------------------------------------------------------------------------

def apply_air_eq(
    audio: np.ndarray,
    sr: int,
    freq_hz: float = 12_000.0,
    gain_db: float = 1.5,
) -> np.ndarray:
    """
    Gentle high-shelf boost for presence and clarity.
    Implemented as a 1st-order shelving filter (bilinear transform).
    """
    try:
        from scipy.signal import sosfilt
        # Bilinear transform high-shelf coefficients
        omega = 2.0 * np.pi * freq_hz / sr
        A     = 10.0 ** (gain_db / 40.0)
        wd    = np.tan(omega / 2.0)
        k     = wd / np.sqrt(A)

        # Normalised coefficients for a 1st-order high shelf
        b0 = A * (k + A) / (k + 1.0 / A)
        b1 = A * (A - k) / (k + 1.0 / A)
        a0 = 1.0
        a1 = (k - 1.0 / A) / (k + 1.0 / A)

        # pack as 1-section SOS (a0 is always 1)
        sos = np.array([[b0, b1, 0.0, 1.0, a1, 0.0]])
        return sosfilt(sos, audio).astype(np.float32)
    except Exception:
        return audio


# ---------------------------------------------------------------------------
# Stage 6+7 — LUFS normalise + true-peak limiter
# ---------------------------------------------------------------------------

def _quick_lufs(audio: np.ndarray, sr: int) -> float:
    """Fast approximate LUFS (no gating) for normalisation."""
    try:
        from scripts.core.mastering import compute_lufs
        return compute_lufs(audio, sr)
    except Exception:
        rms = np.sqrt(np.mean(audio ** 2))
        return float(20.0 * np.log10(max(rms, 1e-9)) - 0.691)


def apply_lufs_normalize(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -14.0,
    max_gain_db: float = 15.0,
) -> Tuple[np.ndarray, float]:
    """Return (normalised_audio, gain_applied_db)."""
    current = _quick_lufs(audio, sr)
    gain_db = float(np.clip(target_lufs - current, -max_gain_db, max_gain_db))
    gain    = 10.0 ** (gain_db / 20.0)
    return (audio * gain).astype(np.float32), gain_db


def apply_true_peak_limiter(audio: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Brick-wall peak limiter."""
    ceil = 10.0 ** (ceiling_db / 20.0)
    peak = np.max(np.abs(audio))
    if peak > ceil:
        audio = audio * (ceil / peak)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Master enhancement function
# ---------------------------------------------------------------------------

def enhance_audio(
    audio: np.ndarray,
    sr: int,
    opts: Optional[EnhanceOptions] = None,
) -> Tuple[np.ndarray, EnhanceReport]:
    """
    Run the full enhancement chain on a mono float32 array.

    Returns (enhanced_audio, EnhanceReport).
    """
    if opts is None:
        opts = EnhanceOptions()

    report          = EnhanceReport()
    report.rms_before_db  = _rms_db(audio)
    report.peak_before_db = _peak_db(audio)
    report.lufs_before    = _quick_lufs(audio, sr)

    out = audio.copy().astype(np.float32)

    if opts.hp_filter:
        try:
            out = apply_hp_filter(out, sr, opts.hp_cutoff_hz)
            report.stages_applied.append("hp_filter")
        except Exception as e:
            log.warning("HP filter failed: %s", e)

    if opts.lp_filter:
        try:
            out = apply_lp_filter(out, sr, opts.lp_cutoff_hz)
            report.stages_applied.append("lp_filter")
        except Exception as e:
            log.warning("LP filter failed: %s", e)

    if opts.noise_gate:
        try:
            out = apply_noise_gate(out, sr, opts.gate_threshold_db, opts.gate_release_ms)
            report.stages_applied.append("noise_gate")
        except Exception as e:
            log.warning("Noise gate failed: %s", e)

    if opts.compression:
        try:
            out = apply_compression(
                out, sr,
                threshold_db=opts.comp_threshold_db,
                ratio=opts.comp_ratio,
                attack_ms=opts.comp_attack_ms,
                release_ms=opts.comp_release_ms,
                knee_db=opts.comp_knee_db,
                makeup_db=opts.comp_makeup_db,
            )
            report.stages_applied.append("compression")
        except Exception as e:
            log.warning("Compression failed: %s", e)

    if opts.air_eq:
        try:
            out = apply_air_eq(out, sr, opts.air_freq_hz, opts.air_gain_db)
            report.stages_applied.append("air_eq")
        except Exception as e:
            log.warning("Air EQ failed: %s", e)

    if opts.lufs_target != 0:
        try:
            out, gain_db = apply_lufs_normalize(out, sr, opts.lufs_target)
            report.gain_applied_db = gain_db
            report.stages_applied.append(f"lufs_norm({opts.lufs_target:.0f})")
        except Exception as e:
            log.warning("LUFS normalisation failed: %s", e)

    if opts.true_peak_ceil != 0:
        try:
            out = apply_true_peak_limiter(out, opts.true_peak_ceil)
            report.stages_applied.append("true_peak_limiter")
        except Exception as e:
            log.warning("True-peak limiter failed: %s", e)

    report.rms_after_db  = _rms_db(out)
    report.peak_after_db = _peak_db(out)
    report.lufs_after    = _quick_lufs(out, sr)
    report.clipped       = bool(np.any(np.abs(out) >= 0.999))

    return out, report


# ---------------------------------------------------------------------------
# Per-stem normalisation after Demucs
# ---------------------------------------------------------------------------

def enhance_stems(
    stems: Dict[str, np.ndarray],
    sr: int,
    stem_targets: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Normalise each stem to its recommended LUFS target.

    Default targets (per stem):
      vocals  →  -14 LUFS  (clear, full presence)
      drums   →  -16 LUFS  (punchy, not overpowering)
      bass    →  -18 LUFS  (tight, leave headroom)
      other   →  -18 LUFS  (instruments / pads)
    """
    defaults = {
        "vocals": -14.0,
        "drums":  -16.0,
        "bass":   -18.0,
        "other":  -18.0,
    }
    targets = {**defaults, **(stem_targets or {})}

    enhanced: Dict[str, np.ndarray] = {}
    for stem_name, audio in stems.items():
        target = targets.get(stem_name, -14.0)
        try:
            opts = EnhanceOptions(
                hp_filter=False,
                lp_filter=False,
                noise_gate=False,
                compression=False,
                air_eq=False,
                lufs_target=target,
                true_peak_ceil=-1.0,
            )
            out, _ = enhance_audio(audio, sr, opts)
            enhanced[stem_name] = out
        except Exception as e:
            log.warning("Stem enhance failed for %s: %s", stem_name, e)
            enhanced[stem_name] = audio

    return enhanced
