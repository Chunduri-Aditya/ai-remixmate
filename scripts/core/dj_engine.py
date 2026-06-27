"""
scripts/core/dj_engine.py — DJ-style transition rendering for AI RemixMate.

This module implements the rendering phase of DJ-style transitions:

  TRANSITION RENDERER — apply the planned EQ curves to the actual audio
                        and produce the final crossfade.

For analysis and planning, see dj_analysis.py which provides:
  - PHRASE DETECTION (SongStructure, analyze_structure)
  - SECTION ANALYSIS (Section, _label_sections)
  - TRANSITION PLANNING (TransitionPlan, plan_transition)

Architecture note:
  Analysis (dj_analysis.py) works on raw audio (with librosa) to produce
  metadata SongStructure and TransitionPlan objects. Rendering (this module)
  applies those plans to the audio samples using scipy/numpy DSP.

Usage:
  from scripts.core.dj_engine import DJEngine
  from scripts.core.dj_analysis import analyze_structure, plan_transition

  # Analyse downloaded audio
  structure_a = analyze_structure(vocals_a, sr)
  structure_b = analyze_structure(instrumentals_b, sr)

  # Plan the transition
  plan = plan_transition(structure_a, structure_b)
  print(f"Mix at bar {plan.exit_bar_a}, overlap {plan.transition_bars} bars")

  # Render
  engine = DJEngine()
  mix = engine.render(vocals_a, instrumentals_b, plan, sr)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Re-export data structures and analysis functions for backward compatibility
from scripts.core.dj_analysis import (
    Beat,
    Section,
    SongStructure,
    EQPlan,
    TransitionPlan,
    analyze_structure,
    plan_transition,
)

# Mastering utilities — used for per-stem LUFS normalization in render_stem_blend()
from scripts.core.mastering import normalize_stems_to_target
from scripts.core.key_detection import pitch_shift_audio

# _analyze_impl was renamed to analyze_structure during the dj_analysis module split.
# This alias preserves backward compatibility for all existing callers.
_analyze_impl = analyze_structure

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (for rendering only)
# ---------------------------------------------------------------------------

try:
    from scripts.core.config import cfg as _cfg
    _HP_START_HZ: float     = getattr(getattr(_cfg, "dj", None), "hp_filter_start_hz", 400.0)
    _HP_END_HZ: float       = getattr(getattr(_cfg, "dj", None), "hp_filter_end_hz", 80.0)
    _BASS_CROSSOVER: float  = getattr(getattr(_cfg, "dj", None), "bass_crossover_hz", 150.0)
except Exception:
    _HP_START_HZ     = 400.0
    _HP_END_HZ       = 80.0
    _BASS_CROSSOVER  = 150.0


# ---------------------------------------------------------------------------
# DSP helpers (rendering support)
# ---------------------------------------------------------------------------

def _butter_highpass(cutoff_hz: float, sr: int, order: int = 4) -> Tuple:
    """Return scipy butter high-pass filter coefficients."""
    try:
        from scipy.signal import butter
        nyq = sr / 2.0
        normal_cutoff = max(0.001, min(0.999, cutoff_hz / nyq))
        return butter(order, normal_cutoff, btype="high", analog=False)
    except ImportError:
        return None, None


def _butter_lowpass(cutoff_hz: float, sr: int, order: int = 4) -> Tuple:
    """Return scipy butter low-pass filter coefficients."""
    try:
        from scipy.signal import butter
        nyq = sr / 2.0
        normal_cutoff = max(0.001, min(0.999, cutoff_hz / nyq))
        return butter(order, normal_cutoff, btype="low", analog=False)
    except ImportError:
        return None, None


def _apply_filter(audio: np.ndarray, b, a) -> np.ndarray:
    """Apply a scipy filter to audio; return input unchanged if scipy absent."""
    if b is None or a is None:
        return audio
    try:
        from scipy.signal import sosfilt, butter
        from scipy.signal import lfilter
        return lfilter(b, a, audio).astype(audio.dtype)
    except Exception:
        return audio


def _make_hp_ramp(total_samples: int, sr: int,
                  start_hz: float, end_hz: float,
                  audio: np.ndarray) -> np.ndarray:
    """
    Apply a high-pass filter to audio with a cutoff that ramps linearly
    from start_hz → end_hz over the duration of audio.

    Implemented by dividing audio into small chunks and filtering each chunk
    with a progressively lower cutoff.
    """
    num_chunks = 16
    chunk_size = total_samples // num_chunks
    if chunk_size < 512:
        # Too short to ramp — just apply a fixed HP at start_hz
        b, a = _butter_highpass(start_hz, sr)
        return _apply_filter(audio, b, a)

    out = np.zeros_like(audio)
    for i in range(num_chunks):
        t = i / max(num_chunks - 1, 1)         # 0 → 1
        cutoff = start_hz + t * (end_hz - start_hz)
        b, a = _butter_highpass(max(cutoff, 10.0), sr)
        s = i * chunk_size
        e = s + chunk_size if i < num_chunks - 1 else total_samples
        out[s:e] = _apply_filter(audio[s:e], b, a)

    return out


# ---------------------------------------------------------------------------
# Numeric guard helpers — single chokepoint for render-path inputs
# ---------------------------------------------------------------------------

# Stretch ratios outside this band are either mis-detected BPM or numeric
# garbage.  librosa will happily waste minutes of CPU (or crash) on them.
_MIN_STRETCH = 0.5   # B is at most 2× faster than A
_MAX_STRETCH = 2.0   # B is at most 2× slower than A


def _safe_ratio(ratio: float) -> float:
    """Clamp a time-stretch ratio to a musically sane, finite band.

    Guards against 0, negative, NaN, and inf — any of which corrupts the
    render. Out-of-band values are clamped (not raised) so a single bad BPM
    detection degrades gracefully to a 'close enough' mix instead of failing
    the whole job.
    """
    if ratio is None or not math.isfinite(ratio) or ratio <= 0.0:
        return 1.0
    return float(min(max(ratio, _MIN_STRETCH), _MAX_STRETCH))


def _safe_bpm(bpm: float, default: float = 120.0) -> float:
    """Return a finite, positive BPM, falling back to `default`."""
    if bpm is None or not math.isfinite(bpm) or bpm <= 0.0:
        return default
    return float(bpm)


def _time_stretch(audio: np.ndarray, ratio: float) -> np.ndarray:
    """
    Time-stretch audio by ratio using librosa's phase vocoder.
    ratio > 1.0 → speed up (higher BPM); ratio < 1.0 → slow down.
    Returns input unchanged if ratio is within 2% of 1.0.
    """
    raw = ratio
    ratio = _safe_ratio(ratio)
    if raw != ratio and raw is not None and math.isfinite(raw):
        log.info("Clamped stretch ratio %.3f → %.3f", raw, ratio)
    if abs(ratio - 1.0) < 0.02:
        return audio
    try:
        from scripts.core.gpu import gpu_time_stretch
        return gpu_time_stretch(audio, rate=ratio)
    except ImportError:
        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=ratio)
        except Exception as e:
            log.warning("Time stretch failed (ratio=%.3f): %s", ratio, e)
            return audio
    except Exception as e:
        log.warning("Time stretch failed (ratio=%.3f): %s", ratio, e)
        return audio


# ---------------------------------------------------------------------------
# Smart transition effects
# ---------------------------------------------------------------------------

def _apply_echo_out(
    audio: np.ndarray,
    sr: int,
    bpm: float,
    decay: float = 0.4,
    n_echoes: int = 4,
) -> np.ndarray:
    """
    Add beat-sync'd ping-pong delay echoes to signal Song A is closing.

    Each echo is placed one beat apart and attenuated by `decay` per step,
    creating a natural "DJ echo throw" effect at the end of the outgoing track.

    Args:
        bpm:      Tempo of the outgoing track (controls echo spacing).
        decay:    Amplitude decay factor per echo (0.4 ≈ −8 dB/echo).
        n_echoes: Number of echo repetitions (default 4 = one bar at 4/4).
    """
    if bpm <= 0 or len(audio) == 0:
        return audio
    beat_samples = int(sr * 60.0 / bpm)
    result = audio.astype(np.float32).copy()
    for i in range(1, n_echoes + 1):
        delay = i * beat_samples
        gain  = decay ** i
        if delay < len(result):
            result[delay:] += audio[:len(result) - delay] * gain
    peak = float(np.abs(result).max())
    if peak > 1.0:
        result /= peak
    return result


def _apply_filter_sweep(
    audio: np.ndarray,
    sr: int,
    direction: str = "out",
    start_hz: float = 18000.0,
    end_hz: float = 300.0,
    num_chunks: int = 32,
) -> np.ndarray:
    """
    Sweep a low-pass filter cutoff across the entire audio clip.

    direction='out': cutoff closes (start_hz → end_hz), darkening the sound
                     as Song A exits (classic "filter out" DJ technique).
    direction='in':  cutoff opens (end_hz → start_hz), brightening Song B
                     as it enters.
    """
    try:
        from scipy.signal import butter, lfilter
    except ImportError:
        return audio

    n = len(audio)
    if n < 512:
        return audio

    if direction == "in":
        start_hz, end_hz = end_hz, start_hz

    chunk_size = max(128, n // num_chunks)
    out = np.empty_like(audio)
    nyq = sr / 2.0

    pos = 0
    chunk_idx = 0
    while pos < n:
        t = chunk_idx / max(num_chunks - 1, 1)
        cutoff = float(np.clip((start_hz + t * (end_hz - start_hz)) / nyq, 0.001, 0.999))
        end    = min(pos + chunk_size, n)
        try:
            b, a = butter(2, cutoff, btype="low")
            out[pos:end] = lfilter(b, a, audio[pos:end]).astype(np.float32)
        except Exception:
            out[pos:end] = audio[pos:end]
        pos        += chunk_size
        chunk_idx  += 1

    return out.astype(np.float32)


def _apply_reverb_tail(
    audio: np.ndarray,
    sr: int,
    reverb_time: float = 1.2,
    wet: float = 0.25,
    seed: int = 42,
) -> np.ndarray:
    """
    Add a simple diffuse-reverb tail to the audio.

    Uses a decaying noise impulse response (a reasonable approximation of
    a large room/hall). The wet/dry mix keeps the original transients clear
    while adding a sense of space as Song A fades out.

    Args:
        reverb_time: RT60 approximation in seconds (default 1.2 s).
        wet:         Wet/dry mix 0–1 (default 0.25 = 25 % reverb).
        seed:        RNG seed for reproducibility.
    """
    try:
        from scipy.signal import fftconvolve
    except ImportError:
        return audio

    rng    = np.random.default_rng(seed)
    ir_len = int(sr * reverb_time)
    ir     = rng.standard_normal(ir_len).astype(np.float64)
    ir    *= np.exp(-3.0 * np.arange(ir_len) / ir_len)
    ir    /= np.abs(ir).max() + 1e-10

    wet_sig = fftconvolve(audio.astype(np.float64), ir, mode="full")
    wet_sig = wet_sig[:len(audio)].astype(np.float32)

    result = audio * (1.0 - wet) + wet_sig * wet
    peak   = float(np.abs(result).max())
    if peak > 1.0:
        result /= peak
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Stem-aware mixing helpers
# ---------------------------------------------------------------------------

_STEM_NAMES = ("vocals", "drums", "bass", "other")


def _load_stem(stem_dir: "Path", stem: str, sr: int) -> "Optional[np.ndarray]":
    """
    Load a stem audio file from a song directory.
    Prefers FLAC (compressed, lossless) over WAV.
    Returns mono float32 array, or None if not found.
    """
    try:
        import soundfile as sf
        from pathlib import Path as _Path
        d = _Path(stem_dir)
        for ext in (".flac", ".wav"):
            p = d / f"{stem}{ext}"
            if p.exists():
                data, file_sr = sf.read(str(p), dtype="float32", always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                # Resample if necessary
                if file_sr != sr:
                    try:
                        from scripts.core.gpu import gpu_resample
                        data = gpu_resample(data, orig_sr=file_sr, target_sr=sr)
                    except (ImportError, Exception):
                        try:
                            import librosa
                            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
                        except Exception:
                            pass
                return data.astype(np.float32)
    except Exception as exc:
        log.debug("_load_stem %s/%s: %s", stem_dir, stem, exc)
    return None


def _stem_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute a compact 5-dim feature vector for stem similarity comparison:
      [rms_norm, centroid_norm, zcr, beat_strength, spectral_flatness]

    All dims are in [0,1].  Used for cosine similarity between two stems.
    """
    feats = np.zeros(5, dtype=np.float32)
    if audio is None or len(audio) == 0:
        return feats
    # Use at most 30 s for speed
    clip = audio[:sr * 30] if len(audio) > sr * 30 else audio
    try:
        import librosa

        # RMS energy (normalised to [0,1] using a reference of 0.1)
        rms = float(np.sqrt(np.mean(clip ** 2)))
        feats[0] = float(np.clip(rms / 0.10, 0, 1))

        # Spectral centroid (norm to 8 kHz)
        try:
            from scripts.core.gpu import gpu_stft
            S = np.abs(gpu_stft(clip))
        except (ImportError, Exception):
            S = np.abs(librosa.stft(clip))
        feats[1] = float(np.clip(
            np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)) / 8000.0, 0, 1
        ))

        # Zero-crossing rate (norm, typical speech ≈ 0.05)
        feats[2] = float(np.clip(
            np.mean(librosa.feature.zero_crossing_rate(clip)) / 0.10, 0, 1
        ))

        # Beat strength
        onset_env = librosa.onset.onset_strength(y=clip, sr=sr)
        _, beats  = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        if len(beats) >= 4:
            feats[3] = float(np.clip(
                np.mean(onset_env[beats]) / (np.max(onset_env) + 1e-8), 0, 1
            ))
        else:
            feats[3] = 0.3

        # Spectral flatness (0 = tonal, 1 = noise-like)
        feats[4] = float(np.clip(
            np.mean(librosa.feature.spectral_flatness(S=S)), 0, 1
        ))
    except Exception:
        pass
    return feats


def _stem_similarity(stem_a: "Optional[np.ndarray]",
                     stem_b: "Optional[np.ndarray]",
                     sr: int) -> float:
    """
    Cosine similarity between two stem feature vectors.
    Returns a value in [0, 1]:
      > 0.75  → stems sound similar → use extended blend
      0.40-0.75 → moderate similarity → standard crossfade
      < 0.40  → stems are very different → sharp handoff
    """
    if stem_a is None or stem_b is None:
        return 0.5   # unknown → neutral
    fa = _stem_features(stem_a, sr)
    fb = _stem_features(stem_b, sr)
    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
    if na < 1e-8 or nb < 1e-8:
        return 0.5
    return float(np.clip(np.dot(fa, fb) / (na * nb), 0.0, 1.0))


def _extract_rhythmic_loop(
    audio: np.ndarray,
    bpm: float,
    sr: int,
    bars: int = 2,
) -> np.ndarray:
    """
    Extract the last ``bars`` bars from ``audio`` and return them as a
    loopable segment.

    This is the "loop extension" the DJ uses when holding a drum pattern
    while the next song fades in — the outgoing drum loop keeps playing
    cleanly on the bar grid instead of stopping abruptly.
    """
    bpm = _safe_bpm(bpm)
    bar_samples  = int(round(sr * 60.0 / bpm * 4))
    loop_samples = bars * bar_samples
    if len(audio) < loop_samples:
        # Not enough audio — tile what we have
        repeats = int(np.ceil(loop_samples / max(len(audio), 1)))
        return np.tile(audio, repeats)[:loop_samples].astype(np.float32)
    return audio[-loop_samples:].astype(np.float32)


def _stem_crossfade_curves(
    trans_samples: int,
    similarity: float,
) -> tuple:
    """
    Compute (fade_out_a, fade_in_b) envelope arrays for a single stem pair
    based on their similarity score.

    High similarity  (> 0.75): long blend — both stems are audible for most
                                of the transition ("extend the loop" feel)
    Medium (0.40-0.75):        standard cosine crossfade
    Low    (< 0.40):           quick handoff — A exits fast, B enters late
                                to create contrast rather than mud

    Both envelopes are float32 numpy arrays of length ``trans_samples``.
    """
    n = trans_samples
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)

    if similarity >= 0.75:
        # Extended blend: A holds full until 40%, fades out by 90%
        #                 B fades in from 10%, full by 60%
        fade_out_a = np.where(t < 0.40, 1.0,
                     np.where(t < 0.90, 1.0 - (t - 0.40) / 0.50, 0.0)
                     ).astype(np.float32)
        fade_in_b  = np.where(t < 0.10, 0.0,
                     np.where(t < 0.60, (t - 0.10) / 0.50, 1.0)
                     ).astype(np.float32)

    elif similarity >= 0.40:
        # Standard cosine crossfade
        fade_out_a = (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)
        fade_in_b  = (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)

    else:
        # Sharp handoff: A exits by 50%, B enters after 50%
        fade_out_a = np.where(t < 0.50, 1.0 - t / 0.50, 0.0).astype(np.float32)
        fade_in_b  = np.where(t < 0.50, 0.0, (t - 0.50) / 0.50).astype(np.float32)

    return fade_out_a, fade_in_b


# ---------------------------------------------------------------------------
# Transition renderer
# ---------------------------------------------------------------------------

class DJEngine:
    """
    Renders a planned DJ transition between two audio clips.

    The engine applies:
      1. Tempo matching (time-stretch Song B to match Song A's BPM)
      2. High-pass filter ramp on Song B (introduce without bass clash)
      3. Bass swap at the midpoint (cut A's bass, release B's bass)
      4. Volume crossfade (A fades out, B fades in)

    Both clips should be mono numpy arrays at the same sample rate.
    """

    def __init__(self, sr: int = 44100) -> None:
        self.sr = sr

    def _apply_dynamic_eq_fade(
        self,
        audio: np.ndarray,
        fade_samples: int,
        direction: str = "out",
    ) -> np.ndarray:
        """
        Frequency-selective fade that mimics professional DJ EQ sweeps.

        Instead of a flat volume fade across all frequencies, this:
        - Fades highs FIRST (brightness disappears early)
        - Fades mids at normal rate
        - Fades bass LAST (energy stays grounded longer)

        For "in" direction, bass enters first, highs bloom in (natural entry)
        This prevents the harsh bass clash that makes amateur mixes sound bad.
        """
        try:
            import scipy.signal as sig
        except ImportError:
            # Degrade gracefully when scipy is absent — simple cosine envelope
            t = np.linspace(0, np.pi / 2, len(audio), dtype=np.float32)
            if direction == "out":
                return (audio * np.cos(t)).astype(np.float32)
            return (audio * np.sin(t)).astype(np.float32)

        if len(audio) < fade_samples or fade_samples < 256:
            # Too short for frequency splitting — fall back to simple envelope
            t = np.linspace(0, np.pi / 2, len(audio), dtype=np.float32)
            if direction == "out":
                return audio * np.cos(t)
            else:
                return audio * np.sin(t)

        # Design 3-band crossover filters
        nyq = self.sr / 2.0
        low_cut = min(150.0 / nyq, 0.95)    # Below 150 Hz = bass/sub
        mid_cut = min(2500.0 / nyq, 0.95)   # 150-2500 Hz = mids/vocals

        # Low-pass for bass
        sos_low = sig.butter(4, low_cut, btype='low', output='sos')
        # Band-pass for mids
        sos_mid = sig.butter(4, [low_cut, mid_cut], btype='band', output='sos')
        # High-pass for highs
        sos_high = sig.butter(4, mid_cut, btype='high', output='sos')

        # Split audio into 3 frequency bands
        low  = sig.sosfilt(sos_low, audio).astype(np.float32)
        mid  = sig.sosfilt(sos_mid, audio).astype(np.float32)
        high = sig.sosfilt(sos_high, audio).astype(np.float32)

        # Build per-band fades with different speeds
        t = np.linspace(0.0, np.pi / 2.0, fade_samples, dtype=np.float32)

        if direction == "out":
            # Exiting: highs vanish first, bass lingers
            fade_high = np.cos(np.clip(t * 1.4, 0, np.pi / 2))   # Fast fade
            fade_mid  = np.cos(t)                                    # Normal
            fade_low  = np.cos(np.clip(t * 0.65, 0, np.pi / 2))  # Slow fade
        else:
            # Entering: bass arrives first, highs bloom last
            fade_high = np.sin(np.clip(t * 0.65, 0, np.pi / 2))  # Slow rise
            fade_mid  = np.sin(t)                                    # Normal
            fade_low  = np.sin(np.clip(t * 1.4, 0, np.pi / 2))   # Fast rise

        # Apply per-band envelopes
        # Only the tail (fade_samples) portion gets the fade
        result = np.copy(audio)
        tail_start = len(audio) - fade_samples

        # Replace tail with frequency-separated faded version
        low_tail  = low[tail_start:]  * fade_low
        mid_tail  = mid[tail_start:]  * fade_mid
        high_tail = high[tail_start:] * fade_high

        result[tail_start:] = low_tail + mid_tail + high_tail

        return result

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def render(
        self,
        track_a: np.ndarray,
        track_b: np.ndarray,
        plan: TransitionPlan,
        full_output: bool = True,
        bridge_beat: Optional[np.ndarray] = None,
        bridge_gain: float = 0.38,
        transition_effect: str = "auto",
    ) -> np.ndarray:
        """
        Render the full DJ mix of track_a into track_b.

        Parameters
        ----------
        track_a : np.ndarray
            Outgoing track (mono float32).
        track_b : np.ndarray
            Incoming track (mono float32).
        plan : TransitionPlan
            Output of plan_transition().
        full_output : bool
            If True, include pre-transition content from A and post-transition
            content from B to produce a full continuous mix.
        bridge_beat : np.ndarray, optional
            A drum/beat loop (mono float32) to layer over the transition.
            Tiled to fill the window; shaped with a sin-arch envelope so it
            rises from silence at the start, peaks at the midpoint, and falls
            back to silence at the end.  Produces a "DJ drops a loop" feel.
        transition_effect : str
            Smart effect applied to Song A's transition window to signal the
            exit.  Options:
              "none"        — no extra processing
              "echo"        — beat-sync echo throw
              "filter"      — low-pass filter sweep (closes over the window)
              "reverb"      — diffuse reverb tail
              "auto"        — choose based on harmonic_score and plan length
                             (echo for matched keys, filter for clashes)
        bridge_gain : float
            Peak gain of the bridge beat layer (default 0.38 ≈ −8 dBFS
            below peaks of A and B, so it supports but doesn't overpower).

        Returns
        -------
        np.ndarray — mixed audio (mono float32).
        """
        # --- Input validation ---
        if len(track_a) == 0 or len(track_b) == 0:
            raise ValueError("Both tracks must have non-zero length")

        if plan.transition_seconds <= 0 or plan.transition_seconds > 300:
            raise ValueError(
                f"Transition must be 0 < seconds ≤ 300; got {plan.transition_seconds}"
            )

        if plan.bpm_a <= 0 or plan.bpm_b <= 0:
            raise ValueError(f"BPM must be > 0; got A={plan.bpm_a}, B={plan.bpm_b}")

        if not np.isfinite(track_a).all():
            raise ValueError("Track A contains NaN or Inf values")
        if not np.isfinite(track_b).all():
            raise ValueError("Track B contains NaN or Inf values")

        if bridge_beat is not None:
            if len(bridge_beat) == 0:
                bridge_beat = None
            elif not np.isfinite(bridge_beat).all():
                raise ValueError("Bridge beat contains NaN or Inf values")

        if not (0.0 <= bridge_gain <= 1.0):
            log.warning("bridge_gain %.2f out of [0, 1] — clamping", bridge_gain)
            bridge_gain = max(0.0, min(1.0, bridge_gain))

        sr = self.sr

        # --- Tempo match: time-stretch B to A's BPM ---
        stretch_ratio = plan.tempo_shift_ratio
        log.info("Tempo matching: stretching B by %.3f×", stretch_ratio)
        track_b_stretched = _time_stretch(track_b, stretch_ratio)

        # --- Compute transition sample boundaries ---
        exit_sample_a = int(plan.exit_time_a * sr)
        trans_samples = int(plan.transition_seconds * sr)
        mid           = trans_samples // 2   # the "0 mark" — where Song B enters

        # entry_sample_b must index into *stretched* B, not original B.
        # time_stretch(rate=r) speeds up by r×, so t → t/r in stretched time.
        safe_ratio     = _safe_ratio(plan.tempo_shift_ratio)
        entry_sample_b = int(plan.entry_time_b * sr / safe_ratio)
        entry_sample_b = max(0, min(entry_sample_b, len(track_b_stretched) - 1))
        exit_sample_a  = max(0, min(exit_sample_a,  len(track_a) - 1))

        # --- Beat-grid lock ---
        #
        # Both exit_sample_a and entry_sample_b are bar boundaries by design
        # (plan_transition snaps both to phrase boundaries).  After time-stretch
        # they share the same BPM, so their bar grids should already align.
        #
        # In practice, two sources of sub-bar drift remain:
        #   1. int() rounding of sample positions (< 1 sample, negligible)
        #   2. librosa time_stretch phase-vocoder drift — the output bar period
        #      may differ from the ideal by up to a few percent, compounding
        #      over the distance from entry_bar_b to the 0 mark.
        #
        # Fix: align both songs to the SAME absolute bar grid.
        # Since exit_sample_a is a bar boundary, A's bar grid phase at sample n
        # is  (n - exit_sample_a) % bar_samples.
        # We want B's entry to have the same absolute bar phase as A's exit
        # (both on beat 1 of a bar simultaneously) so that at the 0 mark
        # (mid samples later, same offset for both) they're locked.
        #
        bar_samples  = max(1, int(round(sr * 60.0 / plan.bpm_a * 4)))
        half_bar     = bar_samples // 2

        # A's absolute bar-grid phase at exit (anchors the shared grid)
        phase_a = exit_sample_a % bar_samples

        # B's absolute bar-grid phase at its computed entry point
        phase_b = entry_sample_b % bar_samples

        # Correction: shift B so phase_b == phase_a
        correction = phase_a - phase_b
        if   correction >  half_bar: correction -= bar_samples
        elif correction < -half_bar: correction += bar_samples

        if correction:
            log.info(
                "Beat-grid lock: %+d samples (%+.1f ms) — bars stay locked at 0 mark",
                correction, correction * 1000.0 / sr,
            )

        b_start = max(0, entry_sample_b + correction)

        # --- Extract the overlapping windows ---
        a_trans = track_a[exit_sample_a : exit_sample_a + trans_samples]
        b_trans = track_b_stretched[b_start : b_start + trans_samples]

        # Pad if shorter than expected
        a_trans = _pad_or_trim(a_trans, trans_samples)
        b_trans = _pad_or_trim(b_trans, trans_samples)

        # --- Smart transition effect on Song A's exit window ---
        _eff = transition_effect
        if _eff == "auto":
            if plan.harmonic_score >= 0.0 and plan.harmonic_score < 0.35:
                _eff = "filter"   # clashing keys → close with LP sweep
            else:
                _eff = "echo"     # compatible or unknown → echo throw

        if _eff == "echo":
            a_trans = _apply_echo_out(a_trans, sr, plan.bpm_a)
        elif _eff == "filter":
            a_trans = _apply_filter_sweep(a_trans, sr, direction="out")
        elif _eff == "reverb":
            a_trans = _apply_reverb_tail(a_trans, sr)
        # "none" → no processing

        # --- Apply HP ramp to B (eliminate bass clash in first half) ---
        hp_ramp_samples = int(plan.transition_seconds / 2 * sr)
        b_hp_section = b_trans[:hp_ramp_samples]
        b_hp_ramp = _make_hp_ramp(
            len(b_hp_section), sr,
            plan.eq.hp_start_hz,
            plan.eq.hp_end_hz,
            b_hp_section,
        )
        b_trans_processed = np.concatenate([b_hp_ramp, b_trans[hp_ramp_samples:]])

        # --- Bass swap ---
        swap_sample = int(plan.eq.bass_swap_bar *
                          (plan.transition_seconds / plan.transition_bars) * sr)
        swap_sample = min(swap_sample, trans_samples)

        b_lp, a_lp_coeffs = _butter_lowpass(plan.eq.bass_crossover_hz, sr), None
        b_lp_b, b_lp_a = _butter_lowpass(plan.eq.bass_crossover_hz, sr)

        # Before swap: A keeps bass, B has no bass (already HP'd)
        # After swap: A loses bass, B gets full bass
        a_before = a_trans[:swap_sample]
        a_after  = a_trans[swap_sample:]
        b_before = b_trans_processed[:swap_sample]
        b_after  = b_trans_processed[swap_sample:]

        # Cut bass from A after swap point
        a_hp_b, a_hp_a = _butter_highpass(plan.eq.bass_crossover_hz, sr)
        a_after_no_bass = _apply_filter(a_after, a_hp_b, a_hp_a)

        a_trans_final = np.concatenate([a_before, a_after_no_bass])
        b_trans_final = b_trans_processed  # B got bass at HP ramp end

        # --- Sequential handoff crossfade ---
        #
        # Timeline (mid = halfway point = the "0 mark"):
        #
        #   Song A  |══FULL══|══FADE OUT══|░░░░░░░░░░|
        #   Song B  |░░░░░░░░|░░░░░░░░░░░░|══RISE IN══|
        #                    ^            ^
        #              fade starts     0 mark / B enters
        #
        # First half  → Song A plays alone at 1.0. Song B is completely silent.
        # Second half → Song A smoothly fades 1 → 0 (cosine curve, sounds natural).
        #               Song B smoothly rises 0 → 1 (cosine curve).
        #
        # The two tracks only overlap during the second half, and never both
        # at high volume simultaneously → no clash, clean handoff.
        #
        # (mid is computed earlier in the beat-grid lock block)
        tail  = trans_samples - mid          # length of the actual crossfade region

        # ── Dynamic EQ crossfade ──────────────────────────────────────────
        # Instead of flat cosine, use frequency-selective fading:
        # Song A: highs fade first, bass lingers (natural exit)
        # Song B: bass enters first, highs bloom in (natural entry)
        a_faded = self._apply_dynamic_eq_fade(a_trans_final, tail, direction="out")
        b_faded = self._apply_dynamic_eq_fade(b_trans_final, tail, direction="in")

        mixed_transition = a_faded + b_faded

        # --- Bridge beat layer (optional) ---
        #
        # The bridge beat sits on top of the A→B crossfade.
        # Its envelope is a full sine arch — sin(t) for t ∈ [0, π] —
        # which means it rises from 0, peaks at the midpoint (where both
        # songs are at equal power), then falls back to 0. This gives the
        # mix a "DJ drops a loop" feel without competing with either track.
        #
        if bridge_beat is not None and len(bridge_beat) > 0:
            # Tile the beat loop to fill the transition window
            reps         = trans_samples // len(bridge_beat) + 1
            beat_tiled   = np.tile(bridge_beat, reps)[:trans_samples]
            # Bell-curve (sine arch) envelope: 0 → 1 → 0
            t_bell       = np.linspace(0.0, np.pi, trans_samples, dtype=np.float32)
            beat_env     = np.sin(t_bell)
            mixed_transition = mixed_transition + beat_tiled * beat_env * bridge_gain

        # --- Assemble full output ---
        if full_output:
            pre_a    = track_a[:exit_sample_a]
            post_b   = track_b_stretched[b_start + trans_samples:]
            result   = np.concatenate([pre_a, mixed_transition, post_b])
        else:
            result = mixed_transition

        # --- Peak normalise to -1 dBFS (headroom without distortion) ---
        # Prefer this over tanh soft-limiting which bends the waveform shape.
        peak = float(np.abs(result).max())
        if peak > 0.0:
            result = result / peak * 0.891  # 0.891 ≈ -1 dBFS

        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Multi-song chain render
    # ------------------------------------------------------------------

    def render_chain(
        self,
        tracks: list,
        plans: list,
        bridge_beats: Optional[list] = None,
        bridge_gain: float = 0.38,
        transition_effect: str = "auto",
        stems_dirs: Optional[list] = None,
    ) -> np.ndarray:
        """
        Render a continuous N-song DJ mix chain.

        All songs are pre-warped to Song 1's BPM for temporal consistency,
        then stitched:

          [pre_A] → [trans AB] → [mid_B] → [trans BC] → [mid_C] → … → [post_N]

        Parameters
        ----------
        tracks : list of N mono float32 arrays (same sr as self.sr)
        plans  : list of N-1 TransitionPlan objects
                 plans[i] describes the transition from tracks[i] → tracks[i+1]
        bridge_beats : list of N-1 optional beat arrays (or None to skip beats)
        bridge_gain  : peak gain of each bridge beat layer
        stems_dirs   : optional list of N Path-like objects (one per track).
                       When both stems_dirs[i] and stems_dirs[i+1] are non-None
                       and the directories exist, the transition window for pair
                       (i, i+1) is rendered via render_stem_blend() which applies
                       per-stem LUFS normalization and the bass-swap envelope
                       (Gaps 1A and 1B).

                       Note: render_stem_blend() operates on the *original* (un-
                       pre-stretched) tracks so it can time-stretch stems to match
                       plan.tempo_shift_ratio.  For N>2 chains where bpm_a ≠ bpm_1
                       the resulting transition window is resampled to fit the
                       chain's pre-stretched timeline — a minor quality trade-off
                       that is inaudible over a short (8–32 bar) transition.

        Returns
        -------
        np.ndarray — mono float32 mix at self.sr
        """
        sr = self.sr
        n = len(tracks)
        if n < 2:
            raise ValueError("render_chain needs at least 2 tracks")
        if len(plans) != n - 1:
            raise ValueError(f"Expected {n-1} plans for {n} tracks, got {len(plans)}")

        for i, track in enumerate(tracks):
            if len(track) == 0:
                raise ValueError(f"Empty audio for track {i + 1}")

        if bridge_beats is None:
            bridge_beats = [None] * (n - 1)

        # ── Pre-stretch: warp all songs to Song 1's BPM ──────────────────
        # tracks[0] is the reference; tracks[i>0] are stretched so their
        # time-stretch ratio = original_bpm_i / bpm_song1.
        bpm1 = _safe_bpm(plans[0].bpm_a)
        stretch_ratios: list = [1.0]
        stretched: list = [tracks[0]]
        for i in range(1, n):
            orig_bpm = plans[i - 1].bpm_b
            ratio    = _safe_ratio(_safe_bpm(orig_bpm) / bpm1)
            stretch_ratios.append(ratio)
            stretched.append(_time_stretch(tracks[i], ratio))
            log.info(
                "Chain: stretched song %d (%.1f BPM → %.1f BPM, ratio=%.3f)",
                i + 1, orig_bpm, bpm1, ratio,
            )

        def _orig_to_stretched(t_sec: float, idx: int) -> int:
            """Convert an original-time second offset to a sample index in stretched[idx]."""
            r = stretch_ratios[idx]
            # time_stretch(rate=r) maps t_orig → t_stretched = t_orig / r
            # r is always ≥ _MIN_STRETCH (0.5) after _safe_ratio — no max guard needed
            return int(t_sec * sr / r)

        # Transition length in samples (always relative to bpm1 after pre-stretch)
        def _trans_samples(plan: "TransitionPlan") -> int:
            return int(plan.transition_bars * 4 * (60.0 / bpm1) * sr)

        # Shared bar-grid parameters (all audio is now at bpm1)
        bar_samples = max(1, int(round(sr * 60.0 / bpm1 * 4)))
        half_bar    = bar_samples // 2

        # ── Build chain segments ─────────────────────────────────────────
        segments: list = []

        for i, plan in enumerate(plans):
            audio_a = stretched[i]
            audio_b = stretched[i + 1]

            exit_sample_a  = _orig_to_stretched(plan.exit_time_a, i)
            entry_sample_b = _orig_to_stretched(plan.entry_time_b, i + 1)
            exit_sample_a  = max(0, min(exit_sample_a,  len(audio_a) - 1))
            entry_sample_b = max(0, min(entry_sample_b, len(audio_b) - 1))
            trans_s        = _trans_samples(plan)
            mid            = trans_s // 2

            # Beat-grid lock
            phase_a    = exit_sample_a % bar_samples
            phase_b    = entry_sample_b % bar_samples
            correction = phase_a - phase_b
            if   correction >  half_bar: correction -= bar_samples
            elif correction < -half_bar: correction += bar_samples
            if correction:
                log.info(
                    "Chain beat-grid lock [%d→%d]: %+d samples (%+.1f ms)",
                    i + 1, i + 2, correction, correction * 1000.0 / sr,
                )
            b_start = max(0, entry_sample_b + correction)

            # Extract windows
            a_trans = _pad_or_trim(audio_a[exit_sample_a : exit_sample_a + trans_s], trans_s)
            b_trans = _pad_or_trim(audio_b[b_start : b_start + trans_s], trans_s)

            # Smart transition effect on Song A's exit window
            _eff = transition_effect
            if _eff == "auto":
                if plan.harmonic_score >= 0.0 and plan.harmonic_score < 0.35:
                    _eff = "filter"
                else:
                    _eff = "echo"
            if _eff == "echo":
                a_trans = _apply_echo_out(a_trans, sr, plan.bpm_a)
            elif _eff == "filter":
                a_trans = _apply_filter_sweep(a_trans, sr, direction="out")
            elif _eff == "reverb":
                a_trans = _apply_reverb_tail(a_trans, sr)

            # HP ramp on B (remove bass during intro)
            hp_half        = trans_s // 2
            b_hp_ramp      = _make_hp_ramp(
                hp_half, sr,
                plan.eq.hp_start_hz, plan.eq.hp_end_hz,
                b_trans[:hp_half],
            )
            b_trans_proc   = np.concatenate([b_hp_ramp, b_trans[hp_half:]])

            # Bass swap at midpoint
            swap_s         = min(
                int(plan.eq.bass_swap_bar * (trans_s / max(plan.transition_bars, 1))),
                trans_s,
            )
            a_hp_b, a_hp_a = _butter_highpass(plan.eq.bass_crossover_hz, sr)
            a_trans_final  = np.concatenate([
                a_trans[:swap_s],
                _apply_filter(a_trans[swap_s:], a_hp_b, a_hp_a),
            ])

            # ── Transition: stem-blend (Gap 1C) or sequential crossfade ────
            _use_stems = (
                stems_dirs is not None
                and stems_dirs[i] is not None
                and stems_dirs[i + 1] is not None
                and Path(stems_dirs[i]).is_dir()
                and Path(stems_dirs[i + 1]).is_dir()
            )
            if _use_stems:
                # render_stem_blend operates on the original tracks so it can
                # time-stretch stems via plan.tempo_shift_ratio.
                _blend = self.render_stem_blend(
                    tracks[i], tracks[i + 1], plan,
                    stems_dir_a=stems_dirs[i],
                    stems_dir_b=stems_dirs[i + 1],
                    full_output=False,
                )
                # Fit to chain's pre-stretched transition window length.
                # For 2-song chains and same-BPM songs this is a no-op.
                if len(_blend) != trans_s:
                    ratio_fit = len(_blend) / max(trans_s, 1)
                    _blend = _time_stretch(_blend, ratio_fit)
                mixed = _pad_or_trim(_blend, trans_s).astype(np.float32)
                log.info(
                    "Chain [%d→%d]: stem-blend transition (Gap 1C, %d samples)",
                    i + 1, i + 2, trans_s,
                )
            else:
                # Sequential crossfade (original path)
                tail   = trans_s - mid
                a_env  = np.ones(trans_s, dtype=np.float32)
                a_env[mid:] = np.cos(np.linspace(0.0, np.pi / 2.0, tail, dtype=np.float32))
                b_env  = np.zeros(trans_s, dtype=np.float32)
                b_env[mid:] = np.sin(np.linspace(0.0, np.pi / 2.0, tail, dtype=np.float32))
                mixed = a_trans_final * a_env + b_trans_proc * b_env

            # Bridge beat (optional)
            bb = bridge_beats[i]
            if bb is not None and len(bb) > 0:
                reps       = trans_s // len(bb) + 1
                beat_tiled = np.tile(bb, reps)[:trans_s]
                t_bell     = np.linspace(0.0, np.pi, trans_s, dtype=np.float32)
                mixed      = mixed + beat_tiled * np.sin(t_bell) * bridge_gain

            # ── Pre-segment (first song only) ──────────────────────────
            if i == 0:
                segments.append(audio_a[:exit_sample_a])

            segments.append(mixed)

            # ── Post-transition body ────────────────────────────────────
            if i < len(plans) - 1:
                # Middle song: B's body runs from after its entry transition
                # to where it needs to exit for the NEXT transition.
                next_exit_b = _orig_to_stretched(plans[i + 1].exit_time_a, i + 1)
                body_start  = b_start + trans_s
                body_end    = min(next_exit_b, len(audio_b))
                if body_end > body_start:
                    segments.append(audio_b[body_start : body_end])
            else:
                # Last song: take everything after the entry transition
                tail_b = audio_b[b_start + trans_s :]
                if len(tail_b) > 0:
                    segments.append(tail_b)

        # ── Concatenate and peak-normalise ───────────────────────────────
        result = np.concatenate([s for s in segments if len(s) > 0])
        peak   = float(np.abs(result).max())
        if peak > 0.0:
            result = result / peak * 0.891  # −1 dBFS headroom

        log.info(
            "Chain render complete: %d songs, %.1f s total",
            n, len(result) / sr,
        )
        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Convenience: render from two stem paths
    # ------------------------------------------------------------------

    def render_from_stems(
        self,
        vocals_path,         # Path or str
        instrumentals_path,  # Path or str
        plan: TransitionPlan,
    ) -> np.ndarray:
        """Load two stem files and render a DJ transition between them."""
        try:
            import soundfile as sf
            vocals, _ = sf.read(str(vocals_path), dtype="float32", always_2d=False)
            instrumentals, _ = sf.read(str(instrumentals_path), dtype="float32", always_2d=False)
            if vocals.ndim > 1:
                vocals = vocals.mean(axis=1)
            if instrumentals.ndim > 1:
                instrumentals = instrumentals.mean(axis=1)
        except Exception as e:
            log.error("Could not load stem files: %s", e)
            raise

        return self.render(vocals, instrumentals, plan)

    # ------------------------------------------------------------------
    # Stem-aware intelligent mixing
    # ------------------------------------------------------------------

    def render_stem_blend(
        self,
        track_a: np.ndarray,
        track_b: np.ndarray,
        plan: "TransitionPlan",
        stems_dir_a: "Optional[Path]" = None,
        stems_dir_b: "Optional[Path]" = None,
        full_output: bool = True,
        bridge_beat: "Optional[np.ndarray]" = None,
        bridge_gain: float = 0.38,
        transition_effect: str = "auto",
    ) -> np.ndarray:
        """
        Render a DJ mix with per-stem intelligent blending.

        When stems are available for both songs, each stem pair (vocals,
        drums, bass, other) is mixed independently with a crossfade curve
        tuned to their acoustic similarity:

          High similarity  → extended blend + loop extension:
                             A's stem keeps playing (looped from its last
                             2 bars) while B's stem fades in gradually.
                             Gives the "instruments continue naturally"  feel.

          Medium similarity → standard cosine crossfade.

          Low similarity   → sharp handoff: A exits first, B enters after.
                             Maximises contrast between the two sounds.

        Falls back to the standard ``render()`` method if stems are missing.

        Parameters
        ----------
        stems_dir_a / stems_dir_b : Path, optional
            Directory containing vocals.flac/wav, drums.flac/wav,
            bass.flac/wav, other.flac/wav for each song.
        """
        sr = self.sr

        # ── Try to load stems ──────────────────────────────────────────────
        stems_a: Dict[str, Optional[np.ndarray]] = {}
        stems_b: Dict[str, Optional[np.ndarray]] = {}

        if stems_dir_a and stems_dir_b:
            for s in _STEM_NAMES:
                stems_a[s] = _load_stem(stems_dir_a, s, sr)
                stems_b[s] = _load_stem(stems_dir_b, s, sr)

        have_stems = any(v is not None for v in stems_a.values()) and \
                     any(v is not None for v in stems_b.values())

        # ── Fallback: no stems available ──────────────────────────────────
        if not have_stems:
            log.info("render_stem_blend: no stems found — falling back to render()")
            return self.render(
                track_a, track_b, plan,
                full_output=full_output,
                bridge_beat=bridge_beat,
                bridge_gain=bridge_gain,
                transition_effect=transition_effect,
            )

        # ── Tempo match track_b ────────────────────────────────────────────
        stretch_ratio = plan.tempo_shift_ratio
        track_b_stretched = _time_stretch(track_b, stretch_ratio)

        # Stretch each stem_b by the same ratio
        stretched_b: Dict[str, Optional[np.ndarray]] = {}
        for s, stem in stems_b.items():
            if stem is not None:
                stretched_b[s] = _time_stretch(stem, stretch_ratio)
            else:
                stretched_b[s] = None

        # ── Pitch shift Song B to align keys (Gap 1B) ─────────────────────
        ps = plan.suggested_pitch_shift
        if ps != 0.0 and abs(ps) <= 3:
            log.info("Applying pitch shift of %+.1f semitones to Song B", ps)
            track_b_stretched = pitch_shift_audio(track_b_stretched, sr, ps)
            for s in stretched_b:
                if stretched_b[s] is not None:
                    stretched_b[s] = pitch_shift_audio(stretched_b[s], sr, ps)

        # ── Per-stem LUFS normalization (Gap 1A) ──────────────────────────
        #
        # Normalize every available stem to −20 LUFS before the crossfade loop.
        # −20 LUFS leaves 6 dB of headroom for summing 4 stems (vocals + drums
        # + bass + other) before the final master_mix() brings the whole mix to
        # −14 LUFS. Without this step, a loud bass stem from Song B would crush
        # Song A's softer stems — neither Spotify nor Apple do this per-stem.
        #
        # normalize_stems_to_target() expects {stem_name: (audio, sr)} dicts.
        stems_a_input = {
            s: (stems_a[s], sr)
            for s in _STEM_NAMES if stems_a.get(s) is not None
        }
        stems_b_input = {
            s: (stretched_b[s], sr)
            for s in _STEM_NAMES if stretched_b.get(s) is not None
        }

        if stems_a_input:
            normed_a = normalize_stems_to_target(stems_a_input, target_lufs=-20.0)
            for s in normed_a:
                stems_a[s] = normed_a[s]
            log.info("Per-stem LUFS normalization applied to Song A stems (target −20 LUFS)")

        if stems_b_input:
            normed_b = normalize_stems_to_target(stems_b_input, target_lufs=-20.0)
            for s in normed_b:
                stretched_b[s] = normed_b[s]
            log.info("Per-stem LUFS normalization applied to Song B stems (target −20 LUFS)")

        # ── Compute transition boundaries ─────────────────────────────────
        exit_sample_a = int(plan.exit_time_a * sr)
        trans_samples = int(plan.transition_seconds * sr)
        safe_ratio    = _safe_ratio(plan.tempo_shift_ratio)
        entry_sample_b = int(plan.entry_time_b * sr / safe_ratio)
        entry_sample_b = max(0, min(entry_sample_b, len(track_b_stretched) - 1))
        exit_sample_a  = max(0, min(exit_sample_a,  len(track_a) - 1))

        # Beat-grid correction
        bar_samples = max(1, int(round(sr * 60.0 / plan.bpm_a * 4)))
        half_bar    = bar_samples // 2
        phase_a     = exit_sample_a % bar_samples
        phase_b     = entry_sample_b % bar_samples
        correction  = phase_a - phase_b
        if   correction >  half_bar: correction -= bar_samples
        elif correction < -half_bar: correction += bar_samples
        entry_sample_b = max(0, entry_sample_b + correction)

        # ── Per-stem similarity + crossfade curves ─────────────────────────
        similarities: Dict[str, float] = {}
        for s in _STEM_NAMES:
            similarities[s] = _stem_similarity(stems_a.get(s), stems_b.get(s), sr)
            log.info("Stem similarity  %-8s  %.2f  → %s",
                     s, similarities[s],
                     "extended blend" if similarities[s] >= 0.75 else
                     "crossfade" if similarities[s] >= 0.40 else "sharp handoff")

        # ── Build stem-blended transition window ──────────────────────────
        transition_mix = np.zeros(trans_samples, dtype=np.float32)

        for s in _STEM_NAMES:
            stem_a_full = stems_a.get(s)
            stem_b_full = stretched_b.get(s)
            sim = similarities[s]

            # ── Bass stem: explicit swap-point handoff (Gap 1B) ───────────
            #
            # The bass stem gets special treatment instead of the generic
            # similarity-based crossfade. Song A's bass fades to zero by the
            # swap point; Song B's bass enters clean from zero at that same
            # point. This prevents bass clash — the most common artifact in
            # automated DJ mixes — and mirrors what a human DJ does manually
            # with a 3-band mixer (kill A's bass, release B's bass at bar 8).
            #
            # swap_sample maps plan.eq.bass_swap_bar to a sample offset inside
            # the transition window using the same formula as render().
            if s == "bass":
                swap_sample = int(
                    plan.eq.bass_swap_bar
                    * (trans_samples / max(plan.transition_bars, 1))
                )
                swap_sample = max(1, min(swap_sample, trans_samples - 1))

                fade_out = np.concatenate([
                    np.linspace(1.0, 0.0, swap_sample, dtype=np.float32),
                    np.zeros(trans_samples - swap_sample, dtype=np.float32),
                ])
                fade_in = np.concatenate([
                    np.zeros(swap_sample, dtype=np.float32),
                    np.linspace(0.0, 1.0, trans_samples - swap_sample, dtype=np.float32),
                ])
                log.info(
                    "Bass swap: A exits at sample %d (%.1f s), B enters clean",
                    swap_sample, swap_sample / sr,
                )
            else:
                fade_out, fade_in = _stem_crossfade_curves(trans_samples, sim)

            # ── Stem A segment: exit window ───────────────────────────────
            if stem_a_full is not None:
                a_start = exit_sample_a
                a_end   = min(a_start + trans_samples, len(stem_a_full))
                a_slice = stem_a_full[a_start:a_end]

                if sim >= 0.75:
                    # Loop extension: extract last 2 bars, tile to fill window
                    loop = _extract_rhythmic_loop(stem_a_full[:exit_sample_a],
                                                  plan.bpm_a, sr, bars=2)
                    # Tile the loop to fill trans_samples
                    reps  = int(np.ceil(trans_samples / max(len(loop), 1)))
                    tiled = np.tile(loop, reps)[:trans_samples]
                    a_contrib = tiled * fade_out
                elif len(a_slice) > 0:
                    padded    = _pad_or_trim(a_slice, trans_samples)
                    a_contrib = padded * fade_out
                else:
                    a_contrib = np.zeros(trans_samples, dtype=np.float32)

                transition_mix += a_contrib

            # ── Stem B segment: entry window ──────────────────────────────
            if stem_b_full is not None:
                b_start = entry_sample_b
                b_end   = min(b_start + trans_samples, len(stem_b_full))
                b_slice = stem_b_full[b_start:b_end]
                if len(b_slice) > 0:
                    padded    = _pad_or_trim(b_slice, trans_samples)
                    b_contrib = padded * fade_in
                    transition_mix += b_contrib

        # ── Normalise transition window (sum of 4 stems can clip) ─────────
        peak = float(np.abs(transition_mix).max())
        if peak > 1.0:
            transition_mix /= peak

        # ── Bridge beat layer ─────────────────────────────────────────────
        if bridge_beat is not None and len(bridge_beat) > 0:
            reps = int(np.ceil(trans_samples / len(bridge_beat)))
            tiled_beat = np.tile(bridge_beat, reps)[:trans_samples]
            bell = np.sin(np.pi * np.linspace(0, 1, trans_samples, dtype=np.float32))
            transition_mix += tiled_beat * bell * bridge_gain

        # ── Assemble full output ───────────────────────────────────────────
        if not full_output:
            return transition_mix

        # Pre-transition: Song A from start to exit point
        pre_a = track_a[:exit_sample_a] if len(track_a) > exit_sample_a else track_a

        # Post-transition: Song B from (entry + transition) onward
        post_b_start = entry_sample_b + trans_samples
        post_b = track_b_stretched[post_b_start:] if len(track_b_stretched) > post_b_start \
                 else np.zeros(0, dtype=np.float32)

        mix = np.concatenate([pre_a, transition_mix, post_b]).astype(np.float32)

        # Final normalise
        peak = float(np.abs(mix).max())
        if peak > 1.0:
            mix /= peak

        log.info(
            "render_stem_blend complete: %.1f s  (pre=%.1fs trans=%.1fs post=%.1fs)  "
            "stems=[%s]",
            len(mix) / sr,
            len(pre_a) / sr,
            len(transition_mix) / sr,
            len(post_b) / sr,
            " ".join(f"{s}:{similarities[s]:.2f}" for s in _STEM_NAMES),
        )
        return mix


# ---------------------------------------------------------------------------
# DSP helpers (continued)
# ---------------------------------------------------------------------------

def _pad_or_trim(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad with zeros or trim to exactly target_samples."""
    n = len(audio)
    if n == target_samples:
        return audio
    if n < target_samples:
        return np.concatenate([audio, np.zeros(target_samples - n, dtype=audio.dtype)])
    return audio[:target_samples]


def _fade_envelope(
    total_samples: int,
    direction: str,       # "in" or "out"
    start_bar: int,
    end_bar: int,
    total_bars: int,
) -> np.ndarray:
    """
    Build a linear fade envelope.

    direction="in":  samples before start_bar = 0, samples after end_bar = 1
    direction="out": samples before start_bar = 1, samples after end_bar = 0
    """
    env = np.ones(total_samples, dtype=np.float32)

    start_sample = int(start_bar / total_bars * total_samples)
    end_sample   = int(end_bar   / total_bars * total_samples)
    end_sample   = max(end_sample, start_sample + 1)

    ramp = np.linspace(0.0, 1.0, end_sample - start_sample)

    if direction == "in":
        env[:start_sample]       = 0.0
        env[start_sample:end_sample] = ramp
        env[end_sample:]         = 1.0
    else:  # "out"
        env[:start_sample]       = 1.0
        env[start_sample:end_sample] = 1.0 - ramp
        env[end_sample:]         = 0.0

    return env
