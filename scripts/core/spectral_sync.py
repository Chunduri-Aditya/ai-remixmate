"""
scripts/core/spectral_sync.py — dynamic spectral syncing across the transition.

Post-render, pre-master stage: a time-varying multiband EQ over the rendered
transition window that eases the mix's spectral balance from Song A's outro
profile to Song B's post-entry profile. Each STFT frame is nudged toward an
interpolated target (cosine ramp A→B across the window), so the spectrum never
jumps at the handover — it *syncs* continuously while the two songs overlap.

Design rules (mirroring spectral_mixing.py's conventions):
  - The bass band below dj.bass_crossover_hz is never touched — the engines'
    bass swap stays the sole authority over low end.
  - Per-band gains are capped (±spectral_sync_max_db) and smoothed over time,
    and taper to 0 dB at the window edges so the processed window splices
    transparently into the untouched mix.
  - Engine-agnostic: operates on the rendered output timeline (classic or
    current), so the frozen classic package stays untouched.
  - Any internal failure returns the mix unchanged — sync never breaks renders.

Gated by config `dj.spectral_sync`; strength via `dj.spectral_sync_max_db`.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

N_BANDS = 24            # mel bands — same resolution as spectral_matching
_N_FFT = 2048
_HOP = 512
_EDGE_TAPER_S = 0.25    # gains ramp to 0 dB this close to the window edges
_SMOOTH_ALPHA = 0.8     # temporal EMA on band gains (higher = smoother)
_PROFILE_CTX_S = 8.0    # context used to profile A's outro / B's continuation


def _mel_filterbank(sr: int, n_fft: int, n_bands: int):
    import librosa
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_bands, fmax=sr / 2)


def _band_profiles(spec_power: np.ndarray, mel_fb: np.ndarray) -> np.ndarray:
    """Power spectrogram (bins × frames) → per-frame band distributions (bands × frames)."""
    band = mel_fb @ spec_power
    total = band.sum(axis=0, keepdims=True)
    total[total <= 1e-12] = 1e-12
    return band / total


def sync_transition_spectrum(
    mix: np.ndarray,
    sr: int,
    overlap_start_s: float,
    overlap_dur_s: float,
    bass_crossover_hz: float = 150.0,
    max_db: float = 4.0,
) -> Tuple[np.ndarray, Optional[Dict]]:
    """Spectrally sync the transition window of a rendered mix.

    Target trajectory: cosine ramp from the mean band profile of the material
    just before the window (A-dominant) to the profile just after it
    (B-dominant). Each frame's band energies are nudged toward the target.

    Returns (mix, info) — info is None when the window doesn't fit the mix.
    Output length always equals input length; audio outside the window is
    bit-identical.
    """
    import librosa

    n = len(mix)
    w_start = int(overlap_start_s * sr)
    w_len = int(overlap_dur_s * sr)
    if w_len <= _N_FFT * 4 or w_start < 0 or w_start + w_len > n:
        return mix, None

    ctx = int(_PROFILE_CTX_S * sr)
    pre = mix[max(0, w_start - ctx): w_start]
    post = mix[w_start + w_len: min(n, w_start + w_len + ctx)]
    window = mix[w_start: w_start + w_len].astype(np.float32)

    mel_fb = _mel_filterbank(sr, _N_FFT, N_BANDS)

    def _mean_profile(seg: np.ndarray) -> Optional[np.ndarray]:
        if len(seg) < _N_FFT * 2:
            return None
        s = np.abs(librosa.stft(seg.astype(np.float32), n_fft=_N_FFT, hop_length=_HOP)) ** 2
        return _band_profiles(s, mel_fb).mean(axis=1)

    prof_a = _mean_profile(pre)
    prof_b = _mean_profile(post)
    if prof_a is None or prof_b is None:
        return mix, None
    # Previews have no pre/post context inside the segment itself — callers
    # pass the window as the whole mix and we bail above; guarded anyway.

    stft = librosa.stft(window, n_fft=_N_FFT, hop_length=_HOP)
    power = np.abs(stft) ** 2
    frames = power.shape[1]
    current = _band_profiles(power, mel_fb)                      # bands × frames

    # Cosine ramp 0→1 across the window: target morphs A-outro → B-profile
    r = (1.0 - np.cos(np.linspace(0.0, np.pi, frames))) / 2.0
    target = prof_a[:, None] * (1.0 - r)[None, :] + prof_b[:, None] * r[None, :]

    gain_cap = 10.0 ** (max_db / 20.0)
    band_gain = np.sqrt(np.clip(target / np.maximum(current, 1e-12), 0.0, None))
    band_gain = np.clip(band_gain, 1.0 / gain_cap, gain_cap)

    # Bass authority stays with the engine's bass swap: no gain below crossover
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)
    band_centers = mel_fb @ freqs / np.maximum(mel_fb.sum(axis=1), 1e-12)
    band_gain[band_centers <= bass_crossover_hz, :] = 1.0

    # Transient preservation: drum hits and drops are the frames with high
    # spectral flux — shaping them dulls the punch that makes a drop land.
    # Scale each frame's gain deviation down by its (normalized) onset flux,
    # so steady-state material gets the full sync EQ and transients pass
    # through nearly untouched.
    flux = np.maximum(power[:, 1:] - power[:, :-1], 0.0).sum(axis=0)
    flux = np.concatenate([[0.0], flux])
    flux_ref = float(np.percentile(flux, 95))
    if flux_ref > 1e-12:
        protect = 1.0 - np.clip(flux / flux_ref, 0.0, 1.0)      # 0 = transient
        band_gain = 1.0 + (band_gain - 1.0) * protect[None, :]

    # Temporal smoothing (EMA) so gains never zip
    for i in range(1, frames):
        band_gain[:, i] = _SMOOTH_ALPHA * band_gain[:, i - 1] + (1 - _SMOOTH_ALPHA) * band_gain[:, i]

    # Edge taper → 0 dB at both window ends (transparent splice)
    edge_frames = max(1, int(_EDGE_TAPER_S * sr / _HOP))
    taper = np.ones(frames)
    ramp = np.linspace(0.0, 1.0, edge_frames)
    taper[:edge_frames] = ramp
    taper[-edge_frames:] = np.minimum(taper[-edge_frames:], ramp[::-1])
    band_gain = 1.0 + (band_gain - 1.0) * taper[None, :]

    # Bands → bins: each bin takes the gain of its band mix (normalized weights)
    fb_norm = mel_fb / np.maximum(mel_fb.sum(axis=0, keepdims=True), 1e-12)
    bin_gain = fb_norm.T @ band_gain                              # bins × frames

    synced = librosa.istft(stft * bin_gain, hop_length=_HOP, length=w_len)

    out = mix.copy()
    out[w_start: w_start + w_len] = synced.astype(mix.dtype)

    adj_db = 20.0 * np.log10(np.maximum(band_gain, 1e-6))
    info = {
        "mean_abs_gain_db": round(float(np.abs(adj_db).mean()), 3),
        "max_gain_db": round(float(adj_db.max()), 2),
        "min_gain_db": round(float(adj_db.min()), 2),
        "frames": int(frames),
        # frames whose gains were mostly bypassed to protect an onset
        "transient_frames": int((flux > 0.5 * flux_ref).sum()) if flux_ref > 1e-12 else 0,
    }
    return out, info


def apply_spectral_sync(mix: np.ndarray, sr: int, plan) -> Tuple[np.ndarray, Optional[Dict]]:
    """Config-gated wrapper for the remix tasks (full-output timeline).

    The overlap window sits at plan.exit_time_a for plan.transition_seconds
    (both engines place it there — Song A plays unstretched up to its exit).
    Returns (mix, info); (mix, None) when disabled or inapplicable.
    """
    try:
        from scripts.core.config import cfg
        dj = getattr(cfg, "dj", None)
        if not bool(getattr(dj, "spectral_sync", True)):
            return mix, None
        bass_hz = float(getattr(dj, "bass_crossover_hz", 150.0))
        max_db = float(getattr(dj, "spectral_sync_max_db", 4.0))
    except Exception:
        bass_hz, max_db = 150.0, 4.0

    try:
        out, info = sync_transition_spectrum(
            mix, sr,
            overlap_start_s=float(plan.exit_time_a),
            overlap_dur_s=float(plan.transition_seconds),
            bass_crossover_hz=bass_hz,
            max_db=max_db,
        )
        if info:
            log.info(
                "Spectral sync: %d frames, mean |gain| %.2f dB (%.1f…%.1f dB)",
                info["frames"], info["mean_abs_gain_db"],
                info["min_gain_db"], info["max_gain_db"],
            )
        return out, info
    except Exception as exc:  # sync is an enhancement, never a failure mode
        log.warning(f"spectral sync skipped: {exc}")
        return mix, None
