"""
scripts/core/vocal_analyzer.py - Vocal F0 and performance analysis.

Stage 4A module for analysing Demucs vocal stems.  CREPE is used when it is
installed; otherwise the module falls back to a deterministic autocorrelation
F0 tracker so tests and local analysis do not require model downloads.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_EPS = 1e-10


@dataclass
class F0Curve:
    """Frame-wise fundamental-frequency estimate."""

    times_s: np.ndarray
    frequency_hz: np.ndarray
    confidence: np.ndarray
    backend: str


@dataclass
class VocalReport:
    """Summary metrics for a vocal stem."""

    mean_pitch_hz: float = 0.0
    pitch_std_cents: float = 0.0
    vibrato_rate_hz: Optional[float] = None
    vibrato_extent_cents: Optional[float] = None
    phrase_count: int = 0
    avg_phrase_duration_s: float = 0.0
    energy_dynamics_db: float = 0.0
    voiced_fraction: float = 0.0
    duration_s: float = 0.0
    backend: str = "none"

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Return mono float32 audio from common channel layouts."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            return arr.mean(axis=0).astype(np.float32)
        if arr.shape[1] <= 8:
            return arr.mean(axis=1).astype(np.float32)
        return arr.mean(axis=1).astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def _default_hop_length(sr: int) -> int:
    """Use a 20 ms F0 frame step, enough for 5-7 Hz vibrato detection."""
    return max(1, int(round(sr * 0.020)))


def _default_frame_length(sr: int, fmin: float) -> int:
    """Choose a power-of-two window long enough for low vocal notes."""
    periods = max(1.0, 4.0 * sr / max(fmin, 1.0))
    length = 2 ** int(np.ceil(np.log2(max(1024.0, periods))))
    return int(min(max(length, 1024), 8192))


def _empty_curve(backend: str) -> F0Curve:
    return F0Curve(
        times_s=np.zeros(0, dtype=np.float32),
        frequency_hz=np.zeros(0, dtype=np.float32),
        confidence=np.zeros(0, dtype=np.float32),
        backend=backend,
    )


def _estimate_f0_crepe(
    audio: np.ndarray,
    sr: int,
    *,
    hop_length: int,
    model_capacity: str,
    viterbi: bool,
) -> F0Curve:
    """Estimate F0 with the optional CREPE package."""
    try:
        import crepe  # type: ignore
    except ImportError as exc:
        raise ImportError("crepe is not installed") from exc

    step_size_ms = max(1, int(round(1000.0 * hop_length / sr)))
    time, frequency, confidence, _activation = crepe.predict(
        audio.astype(np.float32),
        sr,
        step_size=step_size_ms,
        model_capacity=model_capacity,
        viterbi=viterbi,
        verbose=0,
    )
    return F0Curve(
        times_s=np.asarray(time, dtype=np.float32),
        frequency_hz=np.asarray(frequency, dtype=np.float32),
        confidence=np.asarray(confidence, dtype=np.float32),
        backend="crepe",
    )


def _parabolic_peak_offset(values: np.ndarray, idx: int) -> float:
    """Return sub-sample peak offset in [-0.5, 0.5] around idx."""
    if idx <= 0 or idx >= len(values) - 1:
        return 0.0
    y0, y1, y2 = float(values[idx - 1]), float(values[idx]), float(values[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < _EPS:
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))


def _estimate_f0_autocorrelation(
    audio: np.ndarray,
    sr: int,
    *,
    fmin: float,
    fmax: float,
    hop_length: int,
    frame_length: int,
) -> F0Curve:
    """Estimate F0 with a simple normalized autocorrelation tracker."""
    if audio.size == 0 or sr <= 0:
        return _empty_curve("autocorrelation")

    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))

    min_lag = max(1, int(np.floor(sr / max(fmax, 1.0))))
    max_lag = min(frame_length - 2, int(np.ceil(sr / max(fmin, 1.0))))
    if min_lag >= max_lag:
        return _empty_curve("autocorrelation")

    starts = np.arange(0, max(1, audio.size - frame_length + 1), hop_length)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int64)

    times = (starts + frame_length * 0.5) / float(sr)
    freqs = np.full(starts.shape, np.nan, dtype=np.float32)
    confidence = np.zeros(starts.shape, dtype=np.float32)
    window = np.hanning(frame_length).astype(np.float32)

    rms_values = np.empty(starts.shape, dtype=np.float32)
    for i, start in enumerate(starts):
        frame = audio[start : start + frame_length]
        rms_values[i] = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))

    rms_floor = max(1e-6, float(np.percentile(rms_values, 95)) * 0.03)

    for i, start in enumerate(starts):
        if rms_values[i] <= rms_floor:
            continue
        frame = audio[start : start + frame_length].astype(np.float32)
        frame = (frame - float(np.mean(frame))) * window
        corr = np.correlate(frame, frame, mode="full")[frame_length - 1 :]
        if corr[0] <= _EPS:
            continue

        search = corr[min_lag : max_lag + 1]
        if search.size == 0:
            continue
        lag_idx = int(np.argmax(search)) + min_lag
        peak_conf = float(corr[lag_idx] / (corr[0] + _EPS))
        if peak_conf <= 0.05:
            continue

        lag = lag_idx + _parabolic_peak_offset(corr, lag_idx)
        if lag <= 0:
            continue
        freqs[i] = float(sr / lag)
        confidence[i] = float(np.clip(peak_conf, 0.0, 1.0))

    return F0Curve(
        times_s=times.astype(np.float32),
        frequency_hz=freqs,
        confidence=confidence,
        backend="autocorrelation",
    )


def estimate_f0(
    audio: np.ndarray,
    sr: int,
    *,
    backend: str = "auto",
    fmin: float = 65.0,
    fmax: float = 1100.0,
    hop_length: Optional[int] = None,
    frame_length: Optional[int] = None,
    crepe_model_capacity: str = "full",
    crepe_viterbi: bool = True,
) -> F0Curve:
    """
    Estimate a vocal F0 curve.

    Parameters
    ----------
    backend:
        "auto" tries CREPE first and falls back to autocorrelation.  "crepe"
        requests CREPE but still falls back gracefully if it is unavailable.
        "autocorrelation" uses the built-in fallback directly.
    """
    if sr <= 0:
        raise ValueError("sr must be a positive sample rate")
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("fmin must be positive and lower than fmax")

    mono = _to_mono(audio)
    if mono.size == 0:
        return _empty_curve("none")

    hop = hop_length or _default_hop_length(sr)
    frame = frame_length or _default_frame_length(sr, fmin)
    backend_norm = backend.lower().replace("-", "_")
    if backend_norm not in {"auto", "crepe", "autocorrelation"}:
        raise ValueError("backend must be 'auto', 'crepe', or 'autocorrelation'")

    if backend_norm in {"auto", "crepe"}:
        try:
            return _estimate_f0_crepe(
                mono,
                sr,
                hop_length=hop,
                model_capacity=crepe_model_capacity,
                viterbi=crepe_viterbi,
            )
        except Exception as exc:
            if backend_norm == "crepe":
                log.warning("[vocal_analyzer] CREPE unavailable; using autocorrelation: %s", exc)
            else:
                log.debug("[vocal_analyzer] CREPE unavailable; using autocorrelation: %s", exc)

    return _estimate_f0_autocorrelation(
        mono,
        sr,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop,
        frame_length=frame,
    )


def _rms_at_times(audio: np.ndarray, sr: int, times: np.ndarray, window_s: float = 0.050) -> np.ndarray:
    """Measure RMS around each F0 frame timestamp."""
    if times.size == 0:
        return np.zeros(0, dtype=np.float32)
    half = max(1, int(round(sr * window_s * 0.5)))
    rms = np.zeros(times.shape, dtype=np.float32)
    for i, t in enumerate(times):
        center = int(round(float(t) * sr))
        start = max(0, center - half)
        end = min(audio.size, center + half)
        if end <= start:
            continue
        frame = audio[start:end].astype(np.float64)
        rms[i] = float(np.sqrt(np.mean(frame * frame)))
    return rms


def _fill_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Fill inactive runs shorter than max_gap_frames inside an active mask."""
    if mask.size == 0 or max_gap_frames <= 0:
        return mask.copy()
    out = mask.astype(bool).copy()
    i = 0
    while i < out.size:
        if out[i]:
            i += 1
            continue
        start = i
        while i < out.size and not out[i]:
            i += 1
        end = i
        bounded = start > 0 and end < out.size and out[start - 1] and out[end]
        if bounded and (end - start) <= max_gap_frames:
            out[start:end] = True
    return out


def _phrase_segments(
    times: np.ndarray,
    active: np.ndarray,
    *,
    min_phrase_s: float,
    max_gap_s: float,
) -> list[tuple[int, int, float]]:
    """Return inclusive frame ranges for vocal phrases."""
    if times.size == 0 or active.size == 0:
        return []
    frame_step = float(np.median(np.diff(times))) if times.size > 1 else 0.02
    frame_step = max(frame_step, 1e-3)
    merged = _fill_short_gaps(active, int(round(max_gap_s / frame_step)))

    segments: list[tuple[int, int, float]] = []
    i = 0
    while i < merged.size:
        if not merged[i]:
            i += 1
            continue
        start = i
        while i < merged.size and merged[i]:
            i += 1
        end = i - 1
        duration = float(times[end] - times[start] + frame_step)
        if duration >= min_phrase_s:
            segments.append((start, end, duration))
    return segments


def _cents_from_hz(frequency_hz: np.ndarray, reference_hz: float) -> np.ndarray:
    safe = np.maximum(np.asarray(frequency_hz, dtype=np.float64), _EPS)
    return 1200.0 * np.log2(safe / max(reference_hz, _EPS))


def _detect_vibrato(
    times: np.ndarray,
    frequency_hz: np.ndarray,
    voiced: np.ndarray,
    phrases: list[tuple[int, int, float]],
    *,
    min_duration_s: float = 0.8,
) -> tuple[Optional[float], Optional[float]]:
    """Detect the strongest 4-8 Hz modulation in the voiced F0 contour."""
    best: Optional[tuple[float, float, float]] = None
    for start, end, duration in phrases:
        if duration < min_duration_s:
            continue
        idx = np.arange(start, end + 1)
        valid = idx[voiced[idx] & np.isfinite(frequency_hz[idx]) & (frequency_hz[idx] > 0)]
        if valid.size < 16:
            continue

        segment_times = times[valid].astype(np.float64)
        segment_hz = frequency_hz[valid].astype(np.float64)
        frame_step = float(np.median(np.diff(segment_times))) if valid.size > 1 else 0.02
        if frame_step <= 0:
            continue

        median_hz = float(np.median(segment_hz))
        cents = _cents_from_hz(segment_hz, median_hz)
        x = np.arange(cents.size, dtype=np.float64) * frame_step
        if cents.size >= 3:
            trend = np.polyval(np.polyfit(x, cents, 1), x)
            cents = cents - trend
        cents = cents - float(np.mean(cents))
        if float(np.std(cents)) < 2.0:
            continue

        window = np.hanning(cents.size)
        spectrum = np.fft.rfft(cents * window)
        freqs = np.fft.rfftfreq(cents.size, d=frame_step)
        band = (freqs >= 4.0) & (freqs <= 8.0)
        if not np.any(band):
            continue

        amp = 2.0 * np.abs(spectrum) / (float(np.sum(window)) + _EPS)
        band_idx = np.flatnonzero(band)
        peak_idx = int(band_idx[np.argmax(amp[band])])
        extent = float(amp[peak_idx])

        broad = (freqs >= 2.0) & (freqs <= 12.0)
        broad_power = float(np.sum(amp[broad] ** 2)) + _EPS
        dominance = float((amp[peak_idx] ** 2) / broad_power)
        if extent < 5.0 or dominance < 0.12:
            continue

        score = extent * dominance
        candidate = (score, float(freqs[peak_idx]), extent)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return None, None
    return round(best[1], 2), round(best[2], 2)


def analyze_vocal_performance(
    audio: np.ndarray,
    sr: int,
    *,
    backend: str = "auto",
    fmin: float = 65.0,
    fmax: float = 1100.0,
    confidence_threshold: Optional[float] = None,
    min_phrase_s: float = 0.25,
    max_phrase_gap_s: float = 0.20,
) -> VocalReport:
    """
    Analyse a vocal stem and return aggregate pitch/performance metrics.

    The input is expected to be an isolated vocal stem, but the function is
    defensive about silence and stereo layouts.
    """
    if sr <= 0:
        raise ValueError("sr must be a positive sample rate")

    mono = _to_mono(audio)
    duration_s = float(mono.size / sr) if sr > 0 else 0.0
    if mono.size == 0 or float(np.max(np.abs(mono))) <= _EPS:
        return VocalReport(duration_s=round(duration_s, 3), backend="none")

    curve = estimate_f0(mono, sr, backend=backend, fmin=fmin, fmax=fmax)
    if curve.times_s.size == 0:
        return VocalReport(duration_s=round(duration_s, 3), backend=curve.backend)

    threshold = confidence_threshold
    if threshold is None:
        threshold = 0.50 if curve.backend == "crepe" else 0.30

    hz = curve.frequency_hz.astype(np.float64)
    voiced = (
        np.isfinite(hz)
        & (hz >= fmin)
        & (hz <= fmax)
        & (curve.confidence.astype(np.float64) >= float(threshold))
    )

    rms = _rms_at_times(mono, sr, curve.times_s)
    rms_floor = max(1e-8, float(np.percentile(rms, 95)) * 0.03) if rms.size else 1e-8
    active = voiced & (rms > rms_floor)
    phrases = _phrase_segments(
        curve.times_s.astype(np.float64),
        active,
        min_phrase_s=min_phrase_s,
        max_gap_s=max_phrase_gap_s,
    )

    if np.any(voiced):
        voiced_hz = hz[voiced]
        mean_pitch_hz = float(np.mean(voiced_hz))
        reference_hz = float(np.median(voiced_hz))
        pitch_std_cents = float(np.std(_cents_from_hz(voiced_hz, reference_hz)))
    else:
        mean_pitch_hz = 0.0
        pitch_std_cents = 0.0

    vib_rate, vib_extent = _detect_vibrato(
        curve.times_s.astype(np.float64),
        hz,
        voiced,
        phrases,
    )

    if phrases:
        phrase_durations = [duration for _start, _end, duration in phrases]
        phrase_count = len(phrases)
        avg_phrase_duration_s = float(np.mean(phrase_durations))
        phrase_mask = np.zeros(active.shape, dtype=bool)
        for start, end, _duration in phrases:
            phrase_mask[start : end + 1] = True
        energy_rms = rms[phrase_mask & (rms > rms_floor)]
    else:
        phrase_count = 0
        avg_phrase_duration_s = 0.0
        energy_rms = rms[active]

    if energy_rms.size >= 2:
        loudness_db = 20.0 * np.log10(np.maximum(energy_rms.astype(np.float64), _EPS))
        energy_dynamics_db = float(np.percentile(loudness_db, 95) - np.percentile(loudness_db, 5))
    else:
        energy_dynamics_db = 0.0

    return VocalReport(
        mean_pitch_hz=round(mean_pitch_hz, 2),
        pitch_std_cents=round(max(0.0, pitch_std_cents), 2),
        vibrato_rate_hz=vib_rate,
        vibrato_extent_cents=vib_extent,
        phrase_count=phrase_count,
        avg_phrase_duration_s=round(avg_phrase_duration_s, 3),
        energy_dynamics_db=round(max(0.0, energy_dynamics_db), 2),
        voiced_fraction=round(float(np.mean(voiced)), 4) if voiced.size else 0.0,
        duration_s=round(duration_s, 3),
        backend=curve.backend,
    )


def analyze_vocal_file(
    path: str | Path,
    *,
    target_sr: Optional[int] = None,
    duration: Optional[float] = None,
    **kwargs,
) -> VocalReport:
    """Load a vocal stem from disk and run analyze_vocal_performance()."""
    wav_path = Path(path)
    try:
        import soundfile as sf

        audio, sr = sf.read(str(wav_path), always_2d=False)
        audio = _to_mono(audio)
        if duration is not None:
            audio = audio[: int(float(duration) * sr)]
    except Exception:
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError(f"Cannot load {wav_path}: soundfile/librosa unavailable") from exc
        audio, sr = librosa.load(
            str(wav_path),
            sr=target_sr,
            mono=True,
            duration=duration,
        )
        return analyze_vocal_performance(audio, int(sr), **kwargs)

    if target_sr is not None and int(sr) != int(target_sr):
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("Resampling requires librosa") from exc
        audio = librosa.resample(audio.astype(np.float32), orig_sr=int(sr), target_sr=int(target_sr))
        sr = int(target_sr)

    return analyze_vocal_performance(audio, int(sr), **kwargs)
