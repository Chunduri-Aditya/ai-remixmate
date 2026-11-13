"""
scripts/core/mastering.py — Mastering Engine

Implements the full post-render mastering chain:
  1. ITU-R BS.1770-4 integrated loudness measurement (LUFS)
  2. LUFS normalisation (gain-only, no dynamic processing)
  3. True-peak brick-wall limiter with look-ahead
  4. Make up gain: bounded re-limiting passes that restore the loudness
     the limiter removed
  5. Clipping detector
  6. master_mix() — full chain in one call

Standards:
  Streaming:   -14 LUFS  (Spotify, Apple Music, YouTube normalised)
  DJ mix:       -8 LUFS  (louder, matches club playback expectation)
  Ceiling:      -1 dBFS  (headroom for D/A conversion)

Usage:
    from scripts.core.mastering import master_mix, compute_lufs, QualityReport

    mastered, report = master_mix(mix_array, sr=44100, target_lufs=-14.0)
    print(f"Final: {report.lufs_integrated:.1f} LUFS  peak: {report.peak_dbfs:.1f} dBFS")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

log = logging.getLogger(__name__)

# ── K-weighting filter coefficient cache ─────────────────────────────────────
_KW_CACHE: Dict[int, Tuple] = {}


def _kweight_coefficients(sr: int) -> Tuple:
    """
    Return (b1, a1, b2, a2) for the ITU-R BS.1770-4 K-weighting filter.

    Two cascaded stages:
      Stage 1 — High-shelving pre-filter  (+4 dB above ~1.5 kHz)
      Stage 2 — RLB high-pass filter      (−3 dB at ~38 Hz)

    Coefficients are derived analytically so they work at any sample rate.
    """
    if sr in _KW_CACHE:
        return _KW_CACHE[sr]

    # Stage 1: high-shelving pre-filter (+4 dB shelf above ~1.5 kHz).
    # Reference: Giannoulis, Massberg & Reiss (JAES 2013) analytic K-weighting
    # coefficients / ITU-R BS.1770.  The shelf gain Vh must appear on the
    # numerator's constant and middle terms (with Vb = Vh**0.4996 on the K/Q
    # term); that is what makes H(z) → Vh (+4 dB) at Nyquist and → 1 (0 dB) at
    # DC — an actual high shelf.  The previous numerator used 1.0 on those
    # terms, collapsing the shelf to 0 dB at HF (a mid-bell, not a shelf) and
    # making compute_lufs deviate from a BS.1770 reference meter by several LU
    # in a frequency-dependent way.  Verified against the published 48 kHz
    # reference coefficients (b0≈1.53512, a2≈0.73248).
    db = 3.99984385397
    f0 = 1681.97437553
    Q  = 0.7071752369
    K  = np.tan(np.pi * f0 / sr)
    Vh = 10 ** (db / 20.0)       # full shelf gain (linear)
    Vb = Vh ** 0.4996            # mid-band coupling term (Giannoulis)

    b1 = np.array([
        Vh + Vb / Q * K + K ** 2,
        2.0 * (K ** 2 - Vh),
        Vh - Vb / Q * K + K ** 2,
    ])
    a1 = np.array([
        1.0 + 1.0 / Q * K + K ** 2,
        2.0 * (K ** 2 - 1.0),
        1.0 - 1.0 / Q * K + K ** 2,
    ])
    b1 = b1 / a1[0]
    a1 = a1 / a1[0]

    # Stage 2: RLB high-pass (2nd-order, fc = 38.13547 Hz, Q = 0.50033).
    # BS.1770's RLB stage is NOT a maximally-flat Butterworth (Q = 0.7071) —
    # scipy.signal.butter places the poles at the wrong Q and leaves a ~2–3 LU
    # error on deep-bass content vs a reference meter.  Derive the poles
    # analytically at the spec Q so the response matches BS.1770 (verified
    # against the published 48 kHz reference a = [1, -1.99004745, 0.99007225]).
    fc  = 38.13547
    Qhp = 0.5003270373238773
    Khp = np.tan(np.pi * fc / sr)
    a0_hp = 1.0 + Khp / Qhp + Khp ** 2
    b2 = np.array([1.0, -2.0, 1.0]) / a0_hp
    a2 = np.array([
        a0_hp,
        2.0 * (Khp ** 2 - 1.0),
        1.0 - Khp / Qhp + Khp ** 2,
    ]) / a0_hp

    _KW_CACHE[sr] = (b1, a1, b2, a2)
    return b1, a1, b2, a2


def _apply_kweight(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply ITU-R BS.1770 K-weighting filter chain to a mono signal."""
    b1, a1, b2, a2 = _kweight_coefficients(sr)
    tmp = signal.lfilter(b1, a1, audio.astype(np.float64))
    return signal.lfilter(b2, a2, tmp).astype(np.float32)


# ── LUFS measurement ──────────────────────────────────────────────────────────

def compute_lufs(
    audio:                np.ndarray,
    sr:                   int,
    block_size:           float = 0.400,    # seconds — ITU-R standard
    overlap:              float = 0.750,    # 75 % overlap — ITU-R standard
    gate_threshold_lufs:  float = -70.0,    # absolute gate
) -> float:
    """
    ITU-R BS.1770-4 integrated loudness in LUFS.

    Args:
        audio:   Mono float32 signal, range [-1, +1]
        sr:      Sample rate

    Returns:
        LUFS value as a float (e.g. -14.3).  Returns -70.0 for silence.
    """
    try:
        weighted  = _apply_kweight(audio, sr)
        block_len = int(sr * block_size)
        hop_len   = max(1, int(block_len * (1.0 - overlap)))

        if len(weighted) < block_len:
            return -70.0

        n_blocks = (len(weighted) - block_len) // hop_len + 1
        ms = np.array([
            float(np.mean(weighted[i * hop_len: i * hop_len + block_len] ** 2))
            for i in range(n_blocks)
        ])
        ms = np.maximum(ms, 1e-20)

        # Absolute gate (−70 LUFS)
        abs_gate_ms  = 10.0 ** ((gate_threshold_lufs + 0.691) / 10.0)
        ms_gated     = ms[ms > abs_gate_ms]
        if len(ms_gated) == 0:
            return -70.0

        # Relative gate (−10 LU below ungated integrated)
        ungated_lufs = -0.691 + 10.0 * np.log10(ms_gated.mean())
        rel_gate_ms  = 10.0 ** ((ungated_lufs - 10.0 + 0.691) / 10.0)
        ms_final     = ms_gated[ms_gated > rel_gate_ms]
        if len(ms_final) == 0:
            ms_final = ms_gated

        return float(-0.691 + 10.0 * np.log10(ms_final.mean()))

    except Exception as exc:
        log.warning('LUFS computation failed: %s', exc)
        # Fallback: calibrated RMS approximation (not ITU-R compliant)
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        return 20.0 * np.log10(max(rms, 1e-10)) - 3.0


def normalize_to_lufs(
    audio:        np.ndarray,
    sr:           int,
    target_lufs:  float = -14.0,
    max_gain_db:  float = 20.0,
) -> Tuple[np.ndarray, float]:
    """
    Apply a single linear gain to reach `target_lufs`.

    Returns:
        (normalised_audio, gain_db_applied)
    """
    current = compute_lufs(audio, sr)
    if current > -70.0:
        gain_db  = float(np.clip(target_lufs - current, -max_gain_db, max_gain_db))
        gain_lin = 10.0 ** (gain_db / 20.0)
        return (audio.astype(np.float32) * gain_lin).astype(np.float32), gain_db
    return audio.copy(), 0.0


# ── Clipping detection ────────────────────────────────────────────────────────

def detect_clipping(
    audio:          np.ndarray,
    threshold_db:   float = -0.3,
) -> Tuple[bool, int, float]:
    """
    Detect digital clipping (samples at or near 0 dBFS).

    Returns:
        (has_clipping, clip_sample_count, peak_dbfs)
    """
    thresh   = 10.0 ** (threshold_db / 20.0)
    count    = int((np.abs(audio) >= thresh).sum())
    peak_db  = float(20.0 * np.log10(float(np.abs(audio).max()) + 1e-10))
    return count > 0, count, peak_db


# ── True-peak limiter ─────────────────────────────────────────────────────────

def _smooth_gain_envelope(
    reduction:    np.ndarray,
    alpha_attack: float,
    alpha_release: float,
) -> np.ndarray:
    """
    Sample-accurate two-coefficient envelope follower (gain[i] = a*gain[i-1]
    + (1-a)*reduction[i], where `a` switches between a fast attack
    coefficient and a slow release coefficient depending on direction).

    REMIX_QUALITY_INSIGHTS.md finding #4: the previous implementation
    smoothed gain reduction with a single `release_ms` time constant in
    *both* directions via scipy.signal.lfilter (a linear, time-invariant
    filter). A transient needing full reduction took the same ~50ms to ramp
    down as a fully-limited section took to ramp back up to unity — during
    that attack lag the unattenuated peak passed through and got caught by
    the hard-clip safety net instead of being smoothly limited, which is
    audible distortion, not loudness control. Real look-ahead limiters use a
    fast attack (near-instant) and a slow release (avoids pumping) —
    asymmetric by design, which is why this can't be one lfilter call.

    The recursion is inherently sequential (the coefficient at each sample
    depends on whether gain is dropping or recovering), so it isn't
    expressible as a single LTI filter. Numba JIT-compiles the loop to
    native speed (numba is already a transitive dependency via librosa);
    falls back to a pure-Python loop — slower, but still correct — if numba
    is unavailable, same graceful-degradation pattern used elsewhere in this
    codebase (e.g. BeatThisTracker → LibrosaBeatTracker).
    """
    red = reduction.astype(np.float64)
    n = red.shape[0]

    try:
        from numba import njit  # noqa: PLC0415

        @njit(cache=True)
        def _run(r, a_atk, a_rel):
            out = np.empty(r.shape[0], dtype=np.float64)
            g = 1.0
            for i in range(r.shape[0]):
                target = r[i]
                a = a_atk if target < g else a_rel
                g = a * g + (1.0 - a) * target
                out[i] = g
            return out

        gain = _run(red, float(alpha_attack), float(alpha_release))
    except Exception:
        gain = np.empty(n, dtype=np.float64)
        g = 1.0
        for i in range(n):
            target = red[i]
            a = alpha_attack if target < g else alpha_release
            g = a * g + (1.0 - a) * target
            gain[i] = g

    return gain


def apply_limiter(
    audio:          np.ndarray,
    ceiling_db:     float = -1.0,
    lookahead_ms:   float = 3.0,
    release_ms:     float = 50.0,
    attack_ms:      float = 2.0,
    sr:             int   = 44100,
    oversample:     int   = 4,
) -> np.ndarray:
    """
    Brick-wall true-peak limiter with look-ahead gain reduction.

    Algorithm:
      1. Oversample the signal 4x (ITU-R's own recommended factor for
         true-peak detection)
      2. Compute peak envelope with a look-ahead window (maximum filter) on
         the oversampled signal — where it exceeds ceiling, compute the
         required gain reduction
      3. Smooth gain curve with independent attack/release time constants —
         fast attack so the look-ahead window's advance warning is actually
         used, slow release so gain recovery doesn't pump (see
         _smooth_gain_envelope docstring for why these must differ)
      4. Apply gain and hard-clip AT the oversampled rate (so the safety net
         can see inter-sample peaks, unlike a native-rate clip), then
         decimate the fully-limited oversampled audio back to native rate
      5. Verify: re-measure the decimated output's own true peak (one more
         oversampled pass) and, if still over ceiling, apply one exact
         corrective linear gain. Decimation's reconstruction filter rings
         slightly on the sharp corner a hard clip creates (classic
         Gibbs-type behavior), so step 4's clip alone measurably isn't
         airtight — two earlier versions of this fix (oversampling only the
         gain envelope; then oversampled clip without verification) each
         left a fraction of a dB of overshoot on razor-sharp transient
         material. This step makes the ceiling a guarantee by measurement
         rather than by assuming the DSP chain landed under it.

    True-peak detection: a signal can measure "safe" using only its discrete
    sample values (np.abs(audio).max() below ceiling) while its
    reconstructed analog waveform still overshoots between samples — this is
    exactly what ITU-R BS.1770-4 Annex 2 "true peak" measures, and what
    downstream lossy encoders / D/A conversion actually see. The previous
    version of this function detected and limited on sample-domain peaks
    only, despite being named/documented as a true-peak limiter — measured
    up to +2 dB of inter-sample overshoot above ceiling on hot transient
    material even though every sample individually reported within ceiling.
    Oversampling by 4x (ITU-R's own recommended factor) before peak
    detection makes those inter-sample peaks visible to the detector, so the
    gain reduction that gets computed actually accounts for them.

    Args:
        ceiling_db:   True-peak ceiling (default -1 dBFS)
        lookahead_ms: Anticipation window (default 3 ms)
        attack_ms:    Gain-reduction attack time constant (default 2 ms —
                      fast enough that look-ahead-detected peaks are caught
                      before they pass through)
        release_ms:   Gain-recovery release time constant (default 50 ms —
                      slow enough to avoid audible pumping)
        oversample:   Oversampling factor for true-peak detection (default
                      4x, matching ITU-R BS.1770-4 Annex 2). Set to 1 to
                      restore the old sample-peak-only behavior.
    """
    ceiling = 10.0 ** (ceiling_db / 20.0)
    audio = audio.astype(np.float32)
    os_factor = max(1, int(oversample))

    if os_factor > 1:
        audio_os = signal.resample_poly(audio, os_factor, 1).astype(np.float32)
    else:
        audio_os = audio
    sr_os = sr * os_factor

    la_samp  = max(1, int(sr_os * lookahead_ms  / 1000.0))
    att_samp = max(1, int(sr_os * attack_ms  / 1000.0))
    rel_samp = max(1, int(sr_os * release_ms / 1000.0))

    peak_env = np.abs(audio_os)
    peak_env = maximum_filter1d(peak_env, size=la_samp * 2)

    # Per-sample target gain, computed at the oversampled rate so
    # inter-sample peaks factor into the reduction amount.
    reduction = np.where(
        peak_env > ceiling,
        ceiling / np.maximum(peak_env, 1e-10),
        1.0,
    ).astype(np.float32)

    alpha_attack  = float(np.exp(-1.0 / att_samp))
    alpha_release = float(np.exp(-1.0 / rel_samp))
    gain_os = _smooth_gain_envelope(reduction, alpha_attack, alpha_release)
    gain_os = np.clip(gain_os, 0.0, 1.0).astype(np.float32)

    # Gain application AND the safety-net clip both happen at the
    # oversampled rate, where inter-sample peaks are actually visible —
    # clipping only after decimating back to native rate would miss the
    # exact peaks this function exists to catch.
    limited_os = (audio_os * gain_os).astype(np.float32)
    limited_os = np.clip(limited_os, -ceiling, ceiling)

    if os_factor > 1:
        # Polyphase decimation back to native rate — standard sample-rate
        # conversion with a built-in anti-aliasing filter.
        limited = signal.resample_poly(limited_os, 1, os_factor).astype(np.float32)
        if len(limited) < len(audio):
            limited = np.pad(limited, (0, len(audio) - len(limited)), mode="edge")
        elif len(limited) > len(audio):
            limited = limited[: len(audio)]
    else:
        limited = limited_os

    limited = np.clip(limited, -ceiling, ceiling)

    if os_factor > 1:
        # Decimation's reconstruction filter measurably reintroduces
        # overshoot right at a hard-clipped edge (the clip creates a sharp
        # corner — decimation's FIR filter rings on it, classic Gibbs
        # behavior). Rather than guess a safety-margin dB to pre-compensate
        # for that (fragile — tuned to one test signal, wrong for another),
        # verify the actual output's true peak directly and, if it's still
        # over ceiling, apply one exact corrective linear gain. This is a
        # guarantee by measurement, not by hoping the DSP chain landed
        # under threshold — the correction is a fraction of a dB in
        # practice since the bulk of the reduction already happened above.
        verify_os = signal.resample_poly(limited, os_factor, 1)
        true_peak = float(np.max(np.abs(verify_os))) + 1e-12
        if true_peak > ceiling:
            limited = (limited * (ceiling / true_peak)).astype(np.float32)

    return np.clip(limited, -ceiling, ceiling)


# ── Quality report ────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    """Post-render mastering quality report."""
    lufs_integrated:    float = 0.0
    lufs_target:        float = -14.0
    lufs_gain_applied:  float = 0.0
    peak_dbfs:          float = 0.0
    has_clipping:       bool  = False
    clip_count:         int   = 0
    dynamic_range_db:   float = 0.0
    passed:             bool  = True
    notes:              List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


# ── Master chain ──────────────────────────────────────────────────────────────

# Make up gain after limiting. The limiter removes loudness from the level
# normalisation reached (gain reduction on transients, plus its whole file
# true peak correction), and nothing else restores it: a click plus pad track
# mastered to -14 LUFS landed near -17.7. master_mix re-limits the normalised
# signal with extra gain until it is within the tolerance of that level.
# Read at call time inside master_mix, so a test can patch them.
_MAKEUP_MAX_PASSES = 3
_MAKEUP_TOLERANCE_LU = 0.3
_MAKEUP_MAX_DB = 6.0


def master_mix(
    audio:        np.ndarray,
    sr:           int,
    target_lufs:  float = -14.0,
    ceiling_db:   float = -1.0,
) -> Tuple[np.ndarray, QualityReport]:
    """
    Full post-render mastering chain.

    Steps:
      1. Measure input LUFS
      2. Apply linear gain to reach target_lufs
      3. Apply true-peak brick-wall limiter (ceiling_db)
      4. Make up: while the limited output sits more than
         _MAKEUP_TOLERANCE_LU under the level step 2 reached, re-limit the
         normalised signal with extra gain before the limiter (at most
         _MAKEUP_MAX_PASSES passes, at most _MAKEUP_MAX_DB in total; a pass
         that gains under 0.05 LU stops the loop and is discarded). The
         returned audio is always a direct limiter output, so the true-peak
         ceiling still holds.
      5. Measure output peak + clipping
      6. Build QualityReport (lufs_integrated is measured on the returned
         audio; lufs_gain_applied is normalise gain plus make up gain)

    Args:
        audio:       Mono float32 mix
        sr:          Sample rate
        target_lufs: -14 LUFS for streaming, -8 LUFS for DJ mix
        ceiling_db:  True-peak ceiling (-1 dBFS is standard)

    Returns:
        (mastered_audio, QualityReport)
    """
    report = QualityReport(lufs_target=target_lufs)

    # 1. Normalise
    input_lufs = compute_lufs(audio, sr)
    normalised, gain_db = normalize_to_lufs(audio, sr, target_lufs=target_lufs)

    # 2. Limit
    mastered = apply_limiter(normalised, ceiling_db=ceiling_db, sr=sr)
    measured = compute_lufs(mastered, sr)

    # 3. Make up gain toward the level normalisation reached (never the raw
    # target). Make up is capped at 20 dB minus the normalise gain, so the
    # total gain never exceeds the 20 dB clamp in normalize_to_lufs. Every
    # test is written as "if not x > y: break" so a NaN stops the loop.
    goal = input_lufs + gain_db
    makeup_db = 0.0
    makeup_cap_db = min(_MAKEUP_MAX_DB, 20.0 - gain_db)
    if input_lufs > -70.0:
        for _ in range(_MAKEUP_MAX_PASSES):
            deficit = goal - measured
            if not deficit > _MAKEUP_TOLERANCE_LU:
                break
            trial_db = min(makeup_db + deficit, makeup_cap_db)
            if not trial_db > makeup_db:
                break
            cand = apply_limiter(
                normalised * np.float32(10.0 ** (trial_db / 20.0)),
                ceiling_db=ceiling_db, sr=sr,
            )
            cand_lufs = compute_lufs(cand, sr)
            if not cand_lufs > measured + 0.05:
                break
            mastered, measured, makeup_db = cand, cand_lufs, trial_db

    report.lufs_gain_applied = gain_db + makeup_db

    # 4. Post-analysis
    report.lufs_integrated = measured
    _, _, peak             = detect_clipping(mastered, threshold_db=ceiling_db)
    # Clipping is judged on the pre limiter mix: the limiter holds its output
    # AT the ceiling by design, so a ">= ceiling" count after limiting flagged
    # every render the limiter touched.
    clipping, count, _     = detect_clipping(audio)
    report.has_clipping    = clipping
    report.clip_count      = count
    report.peak_dbfs       = peak

    # 5. Dynamic range (simplified: peak − RMS)
    rms_db = float(20.0 * np.log10(
        float(np.sqrt(np.mean(mastered.astype(np.float64) ** 2))) + 1e-10
    ))
    report.dynamic_range_db = float(peak - rms_db)

    # 6. Verdict
    issues: List[str] = []
    if clipping:
        issues.append(
            f'⚠️  {count} sample(s) at or near 0 dBFS before limiting'
        )
    lufs_err = abs(report.lufs_integrated - target_lufs)
    if lufs_err > 3.0:
        issues.append(
            f'⚠️  Final LUFS {report.lufs_integrated:.1f} deviates '
            f'{lufs_err:.1f} LU from target {target_lufs:.1f}'
        )
    if peak > ceiling_db:
        issues.append(
            f'⚠️  True peak {peak:.1f} dBFS exceeds ceiling {ceiling_db:.1f} dBFS'
        )

    report.notes  = issues
    report.passed = len(issues) == 0

    log.info(
        'Master  %.1f→%.1f LUFS  gain=+%.1f dB  makeup=+%.1f dB  '
        'peak=%.1f dBFS  DR=%.1f dB  %s',
        input_lufs, report.lufs_integrated,
        gain_db, makeup_db, peak, report.dynamic_range_db,
        '✅ PASS' if report.passed else '⚠️  WARN',
    )
    return mastered.astype(np.float32), report


# ── Per-stem LUFS analysis ────────────────────────────────────────────────────

@dataclass
class StemLoudnessProfile:
    """
    Loudness profile for a single stem or full track.

    Used for:
      • Conditioning generative models on energy level
      • Normalizing per-stem levels before mixing
      • Detecting energy envelope for AI-driven transitions
    """
    stem: str                           # "vocals" | "drums" | "bass" | "other" | "full"
    lufs_integrated: float = 0.0       # ITU-R BS.1770-4 integrated loudness
    lufs_short_term_max: float = 0.0   # Peak short-term LUFS (3 s window)
    lufs_short_term_min: float = 0.0   # Quietest short-term LUFS
    peak_dbfs: float = 0.0             # True peak
    rms_db: float = 0.0                # RMS level
    dynamic_range_db: float = 0.0      # Peak − RMS
    energy_envelope: List[float] = field(default_factory=list)  # Short-term LUFS series


def analyze_stem_lufs(
    audio: np.ndarray,
    sr: int,
    stem: str = "full",
    window_sec: float = 3.0,
    hop_sec: float = 1.0,
) -> StemLoudnessProfile:
    """
    Compute a full loudness profile for a single audio array.

    Returns a StemLoudnessProfile with integrated LUFS, short-term LUFS
    envelope (useful for energy conditioning), RMS, and true peak.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 audio.
    sr : int
        Sample rate.
    stem : str
        Name tag for the stem (informational only).
    window_sec : float
        Short-term window length in seconds (default 3.0 per ITU standard).
    hop_sec : float
        Hop between short-term windows.

    Returns
    -------
    StemLoudnessProfile
    """
    profile = StemLoudnessProfile(stem=stem)

    if len(audio) == 0:
        return profile

    audio_f = audio.astype(np.float32)

    # Integrated LUFS
    profile.lufs_integrated = compute_lufs(audio_f, sr)

    # True peak
    _, _, profile.peak_dbfs = detect_clipping(audio_f)

    # RMS
    rms = float(np.sqrt(np.mean(audio_f ** 2))) + 1e-10
    profile.rms_db = float(20.0 * np.log10(rms))
    profile.dynamic_range_db = float(profile.peak_dbfs - profile.rms_db)

    # Short-term LUFS envelope (sliding window)
    w_samples = int(window_sec * sr)
    h_samples = max(1, int(hop_sec * sr))

    if len(audio_f) >= w_samples:
        short_term: List[float] = []
        for start in range(0, len(audio_f) - w_samples + 1, h_samples):
            window = audio_f[start : start + w_samples]
            st_lufs = compute_lufs(window, sr)
            short_term.append(st_lufs)
        if short_term:
            profile.energy_envelope = short_term
            profile.lufs_short_term_max = max(short_term)
            profile.lufs_short_term_min = min(short_term)

    log.debug(
        "[mastering] Stem '%s': %.1f LUFS (integrated), %.1f dBFS peak, %.1f dB DR",
        stem, profile.lufs_integrated, profile.peak_dbfs, profile.dynamic_range_db,
    )
    return profile


def analyze_stems_lufs(
    stem_audio: Dict[str, Tuple[np.ndarray, int]],
) -> Dict[str, StemLoudnessProfile]:
    """
    Compute loudness profiles for multiple stems.

    Parameters
    ----------
    stem_audio : Dict[str, Tuple[np.ndarray, int]]
        Mapping of stem_name → (audio_array, sample_rate).

    Returns
    -------
    Dict[str, StemLoudnessProfile]
    """
    profiles: Dict[str, StemLoudnessProfile] = {}
    for stem_name, (audio, sr) in stem_audio.items():
        profiles[stem_name] = analyze_stem_lufs(audio, sr, stem=stem_name)
    return profiles


def normalize_stems_to_target(
    stem_audio: Dict[str, Tuple[np.ndarray, int]],
    target_lufs: float = -20.0,
) -> Dict[str, np.ndarray]:
    """
    Normalize each stem to a common target LUFS, returning normalized arrays.

    Using -20 LUFS per-stem before mixing prevents inter-stem level collisions.
    The final mix is then mastered to -14 LUFS.

    Prefer normalize_stems_to_corpus_targets() when corpus-derived per-stem-type
    targets are available — it produces more natural stem balances.

    Parameters
    ----------
    stem_audio : Dict[str, Tuple[np.ndarray, int]]
        stem_name → (audio, sr)
    target_lufs : float
        Per-stem target in LUFS. Default -20.0 (leaves headroom for summing).

    Returns
    -------
    Dict[str, np.ndarray]  — normalized audio arrays, same sample rates
    """
    result: Dict[str, np.ndarray] = {}
    for stem_name, (audio, sr) in stem_audio.items():
        normalized, _ = normalize_to_lufs(audio, sr, target_lufs=target_lufs)
        result[stem_name] = normalized
    return result


# ── FxNorm-Automix-style per-stem-type corpus normalization ──────────────────
#
# Reference: Steinmetz et al. "Automatic Multitrack Mixing with a
#            Context-Aware Loudness Model" (Sony, ISMIR 2022 / FxNorm-Automix).
#
# Key insight: drums, bass, vocals, and other stems have very different
# frequency content and perceived loudness at the same LUFS.  Using one flat
# -20 LUFS target for all stems produces mix imbalances that are invisible in
# LUFS meters but clearly audible.
#
# Solution: measure the mean integrated LUFS of each stem type across the whole
# library (the "corpus target"), then normalize each stem to *its type's* corpus
# mean.  This preserves the natural loudness relationships between stem types
# while preventing inter-stem collisions.

#: Fallback per-stem targets used when the library has < 3 songs of any type.
#: Derived empirically from commercial EDM/pop productions.
_STEM_FALLBACK_TARGETS: Dict[str, float] = {
    "drums":  -18.5,   # transient-heavy; naturally louder integrated
    "bass":   -21.0,   # sustained sub; high energy but fewer transients
    "vocals": -19.5,   # dynamic; moderate integrated LUFS
    "other":  -21.5,   # chords/pads; typically the softest stem
}


def analyze_library_stem_targets(
    library_dir: Optional[Path] = None,
    min_songs: int = 3,
) -> Dict[str, float]:
    """
    Compute per-stem-type mean integrated LUFS across the library.

    Iterates every song directory that has Demucs stems and measures integrated
    LUFS for each stem type.  Returns a dict of stem_name → mean LUFS target.
    Stem types with fewer than `min_songs` measurements fall back to
    ``_STEM_FALLBACK_TARGETS``.

    Results should be cached to ``data/stem_lufs_targets.json`` and reloaded on
    subsequent calls; this function performs full audio I/O and is slow.

    Parameters
    ----------
    library_dir : Path, optional
        Path to the library root.  Defaults to ``scripts.core.paths.LIBRARY_DIR``.
    min_songs : int
        Minimum measurements required before trusting corpus mean (default 3).

    Returns
    -------
    Dict[str, float]
        stem_name → mean integrated LUFS across the library corpus.
    """
    if library_dir is None:
        from scripts.core.paths import LIBRARY_DIR
        library_dir = LIBRARY_DIR

    stem_names = ("drums", "bass", "vocals", "other")
    measurements: Dict[str, List[float]] = {s: [] for s in stem_names}

    song_dirs = sorted(d for d in library_dir.iterdir() if d.is_dir())
    if not song_dirs:
        log.warning("[mastering] Library is empty — using fallback stem targets")
        return dict(_STEM_FALLBACK_TARGETS)

    import soundfile as sf

    for song_dir in song_dirs:
        for stem in stem_names:
            for ext in (".wav", ".flac"):
                stem_path = song_dir / f"{stem}{ext}"
                if not stem_path.exists():
                    continue
                try:
                    audio, sr = sf.read(str(stem_path), dtype="float32", always_2d=False)
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    if len(audio) < sr:           # skip clips < 1 second
                        break
                    lufs = compute_lufs(audio, sr)
                    if np.isfinite(lufs) and lufs > -70.0:  # skip silence / -inf
                        measurements[stem].append(lufs)
                except Exception as exc:
                    log.debug("[mastering] Skipping %s/%s: %s", song_dir.name, stem, exc)
                break  # found a file for this stem; don't try next ext

    targets: Dict[str, float] = {}
    for stem in stem_names:
        vals = measurements[stem]
        if len(vals) >= min_songs:
            targets[stem] = float(np.mean(vals))
            log.info(
                "[mastering] Corpus target  %-8s  %.1f LUFS  (n=%d)",
                stem, targets[stem], len(vals),
            )
        else:
            targets[stem] = _STEM_FALLBACK_TARGETS[stem]
            log.info(
                "[mastering] Fallback target %-8s  %.1f LUFS  (only %d samples)",
                stem, targets[stem], len(vals),
            )
    return targets


def load_stem_targets(cache_path: Optional[Path] = None) -> Dict[str, float]:
    """
    Load per-stem-type LUFS targets from cache, or fall back to built-in defaults.

    Parameters
    ----------
    cache_path : Path, optional
        Path to ``stem_lufs_targets.json``.  Defaults to ``data/stem_lufs_targets.json``.

    Returns
    -------
    Dict[str, float]
        stem_name → LUFS target.  Always returns a complete dict with all four
        stem types.
    """
    if cache_path is None:
        from scripts.core.paths import DATA_DIR
        cache_path = DATA_DIR / "stem_lufs_targets.json"

    if cache_path.exists():
        try:
            import json
            with open(cache_path) as f:
                cached = json.load(f)
            # Merge with fallbacks so new stem types are always covered
            return {**_STEM_FALLBACK_TARGETS, **cached}
        except Exception as exc:
            log.warning(f"[mastering] Failed to load stem targets cache: {exc}")

    return dict(_STEM_FALLBACK_TARGETS)


def save_stem_targets(targets: Dict[str, float], cache_path: Optional[Path] = None) -> None:
    """Persist per-stem-type LUFS targets to JSON cache."""
    if cache_path is None:
        from scripts.core.paths import DATA_DIR
        cache_path = DATA_DIR / "stem_lufs_targets.json"
    import json
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(targets, f, indent=2)
    log.info("[mastering] Saved stem LUFS targets → %s", cache_path)


def normalize_stems_to_corpus_targets(
    stem_audio: Dict[str, Tuple[np.ndarray, int]],
    targets: Optional[Dict[str, float]] = None,
    fallback_lufs: float = -20.0,
    true_peak_ceiling: float = -1.0,
) -> Dict[str, np.ndarray]:
    """
    Normalize each stem to its corpus-derived per-stem-type LUFS target.

    This is the FxNorm-Automix-style improvement over ``normalize_stems_to_target``.
    Instead of applying one flat LUFS value to all stems, each stem type is
    normalized to the mean LUFS of *that type* across the library corpus.

    Parameters
    ----------
    stem_audio : Dict[str, Tuple[np.ndarray, int]]
        stem_name → (audio_array, sample_rate)
    targets : Dict[str, float], optional
        Per-stem-type LUFS targets.  If None, loads from
        ``data/stem_lufs_targets.json`` or falls back to built-in defaults.
    fallback_lufs : float
        Target LUFS for stem types not found in ``targets`` (default -20.0).
    true_peak_ceiling : float
        Maximum true peak allowed per stem in dBFS (default -1.0).
        Applied after LUFS normalization to prevent inter-stem clipping.

    Returns
    -------
    Dict[str, np.ndarray]
        Normalized audio arrays keyed by stem name.
    """
    if targets is None:
        targets = load_stem_targets()

    result: Dict[str, np.ndarray] = {}

    for stem_name, (audio, sr) in stem_audio.items():
        if len(audio) == 0:
            result[stem_name] = audio
            continue

        audio_f = audio.astype(np.float32)
        target = targets.get(stem_name, fallback_lufs)

        normalized, gain_db = normalize_to_lufs(audio_f, sr, target_lufs=target)

        # True-peak ceiling: if any sample exceeds ceiling after normalization,
        # apply a gentle brick-wall limit to avoid clipping when stems are summed.
        peak = float(np.max(np.abs(normalized)))
        ceiling_lin = 10.0 ** (true_peak_ceiling / 20.0)
        if peak > ceiling_lin:
            clip_gain = ceiling_lin / peak
            normalized = normalized * clip_gain
            log.debug(
                "[mastering] True-peak clip: stem '%s' clamped by %.1f dB",
                stem_name, 20.0 * np.log10(clip_gain),
            )

        result[stem_name] = normalized
        log.debug(
            "[mastering] Corpus-norm: stem='%s' target=%.1f LUFS gain=%.1f dB",
            stem_name, target, gain_db,
        )

    return result
