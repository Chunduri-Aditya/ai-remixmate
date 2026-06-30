"""
scripts/core/beat_tracker.py — Swappable beat tracking backend.

Provides a unified BeatResult / BeatTracker interface so the rest of the
codebase can switch between beat-tracking backends without touching analysis
logic.

Backends
--------
LibrosaBeatTracker (default)
    Wraps librosa.beat.beat_track().  Reliable, always available, but
    produces beat times only — no downbeats.  Downbeats are estimated by
    assuming the first beat in each bar of 4 is a downbeat (bar-aligned
    heuristic).  Accurate to ~1 bar but not sample-accurate.

BeatThisTracker
    Wraps the Beat This! model (Böck & Krebs, ISMIR 2024) — a Beat Transformer
    that achieves state-of-the-art downbeat accuracy on EDM and pop.
    Install: pip install beat-this
    Falls back gracefully to LibrosaBeatTracker if not installed.

Usage
-----
    from scripts.core.beat_tracker import get_tracker, BeatResult

    result = get_tracker("librosa").track(audio, sr)
    # or: get_tracker("beat_this").track(audio, sr)
    # or: get_tracker("auto").track(audio, sr)   # uses beat_this if available

    print(result.bpm, result.beat_times, result.downbeat_times)

Config integration
------------------
    The backend is read from config.yaml: analysis.beat_backend ("librosa" |
    "beat_this" | "auto").  Default is "auto".

    Call get_tracker() without arguments to use the configured backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class BeatResult:
    """
    Output of a beat-tracking pass.

    Attributes
    ----------
    beat_times : np.ndarray
        Times (seconds) of every detected beat, shape (N,).
    beat_frames : np.ndarray
        Same beats as STFT frame indices (hop_length=512 at SR), shape (N,).
        Used for librosa chroma sync and SSM construction.
    downbeat_times : np.ndarray
        Times (seconds) of beat 1 of each bar (downbeats), shape (M,).
        M ≈ N/4 for 4/4 time.  For LibrosaBeatTracker this is estimated
        from bar alignment; for BeatThisTracker it is model-predicted.
    bpm : float
        Estimated global tempo in BPM.
    sr : int
        Sample rate used during tracking.
    backend : str
        Which backend produced this result ("librosa" | "beat_this").
    """
    beat_times: np.ndarray
    beat_frames: np.ndarray
    downbeat_times: np.ndarray
    bpm: float
    sr: int = 44100
    backend: str = "librosa"

    def bar_duration(self) -> float:
        """Seconds per bar (assumes 4/4). Falls back to 120 BPM default when bpm is 0."""
        bpm = self.bpm if self.bpm > 0.0 else 120.0
        return 60.0 / bpm * 4.0

    def nearest_downbeat(self, time_s: float) -> float:
        """Return the downbeat time nearest to `time_s`."""
        if len(self.downbeat_times) == 0:
            # fallback: align to bar grid
            bar_dur = self.bar_duration()
            return round(time_s / bar_dur) * bar_dur
        diffs = np.abs(self.downbeat_times - time_s)
        return float(self.downbeat_times[int(np.argmin(diffs))])

    def nearest_downbeat_at_or_after(self, time_s: float) -> float:
        """Return the first downbeat at or after `time_s`."""
        if len(self.downbeat_times) == 0:
            bar_dur = self.bar_duration()
            n = int(np.ceil(time_s / bar_dur))
            return n * bar_dur
        future = self.downbeat_times[self.downbeat_times >= time_s - 0.05]
        if len(future) == 0:
            return float(self.downbeat_times[-1])
        return float(future[0])


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

class LibrosaBeatTracker:
    """
    Beat tracker using librosa.beat.beat_track().

    Downbeats are estimated: beat index mod 4 == 0 → assumed downbeat.
    This is accurate within 1 bar for well-produced 4/4 music but will
    be off-by-a-bar if the model mis-identifies the downbeat phase.
    """

    def track(self, audio: np.ndarray, sr: int, hop_length: int = 512) -> BeatResult:
        try:
            import librosa
        except ImportError as e:
            raise ImportError("librosa is required for LibrosaBeatTracker") from e

        tempo, beat_frames = librosa.beat.beat_track(
            y=audio, sr=sr, hop_length=hop_length, trim=False
        )
        bpm = float(np.atleast_1d(tempo)[0])
        beat_frames = np.asarray(beat_frames, dtype=np.int64)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

        # Estimate downbeats: every 4th beat starting at index 0
        downbeat_indices = beat_frames[::4]
        downbeat_times = librosa.frames_to_time(downbeat_indices, sr=sr, hop_length=hop_length)

        log.debug(
            "[beat_tracker:librosa] %.1f BPM | %d beats | %d downbeats (estimated)",
            bpm, len(beat_times), len(downbeat_times),
        )
        return BeatResult(
            beat_times=beat_times,
            beat_frames=beat_frames,
            downbeat_times=downbeat_times,
            bpm=bpm,
            sr=sr,
            backend="librosa",
        )


class BeatThisTracker:
    """
    Beat tracker using Beat This! (Böck & Krebs, ISMIR 2024).

    Produces *model-predicted* downbeats — not just bar-aligned heuristics.
    Requires: pip install beat-this (~50 MB model, ~1s inference per 3min clip)

    Falls back to LibrosaBeatTracker if beat_this is not installed.
    """

    def track(self, audio: np.ndarray, sr: int, **_) -> BeatResult:
        try:
            from beat_this.inference import File2Beats  # type: ignore
        except ImportError:
            log.warning(
                "[beat_tracker] beat-this not installed — falling back to librosa. "
                "Install with: pip install beat-this"
            )
            return LibrosaBeatTracker().track(audio, sr)

        try:
            import tempfile, soundfile as _sf, librosa as _librosa
            import os

            # Beat This! works on file paths, not raw arrays.
            # Write to a temp WAV, run inference, clean up.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            audio_16k = _librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            _sf.write(tmp_path, audio_16k, 16000)

            predictor = File2Beats(device="cpu", dbn=True)
            beats_arr, downbeats_arr = predictor(tmp_path)

            os.unlink(tmp_path)

            beat_times = np.asarray(beats_arr, dtype=np.float64)
            downbeat_times = np.asarray(downbeats_arr, dtype=np.float64)

            # Compute BPM from median beat interval
            if len(beat_times) > 1:
                intervals = np.diff(beat_times)
                bpm = float(60.0 / np.median(intervals))
            else:
                bpm = 120.0

            # Convert beat times to frames (hop_length=512) for chroma sync compat
            hop_length = 512
            beat_frames = np.round(beat_times * sr / hop_length).astype(np.int64)

            log.info(
                "[beat_tracker:beat_this] %.1f BPM | %d beats | %d downbeats (model)",
                bpm, len(beat_times), len(downbeat_times),
            )
            return BeatResult(
                beat_times=beat_times,
                beat_frames=beat_frames,
                downbeat_times=downbeat_times,
                bpm=bpm,
                sr=sr,
                backend="beat_this",
            )
        except Exception as exc:
            log.warning(
                "[beat_tracker] beat-this inference failed (%s) — falling back to librosa",
                exc,
            )
            return LibrosaBeatTracker().track(audio, sr)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_tracker(backend: str = "auto") -> LibrosaBeatTracker | BeatThisTracker:
    """
    Return a beat tracker for the given backend name.

    Parameters
    ----------
    backend : str
        "librosa"  — always use LibrosaBeatTracker
        "beat_this" — always use BeatThisTracker (falls back if not installed)
        "auto"     — use BeatThisTracker if available, else librosa

    Returns
    -------
    A tracker with a .track(audio, sr) → BeatResult interface.
    """
    if backend == "librosa":
        return LibrosaBeatTracker()
    if backend == "beat_this":
        return BeatThisTracker()
    if backend == "auto":
        try:
            import beat_this  # noqa: F401
            log.debug("[beat_tracker] Auto-selected: beat_this")
            return BeatThisTracker()
        except ImportError:
            log.debug("[beat_tracker] Auto-selected: librosa (beat-this not installed)")
            return LibrosaBeatTracker()
    raise ValueError(f"Unknown beat_tracker backend: {backend!r}. Use 'librosa', 'beat_this', or 'auto'.")


def get_configured_tracker() -> LibrosaBeatTracker | BeatThisTracker:
    """
    Return the tracker configured in config.yaml (analysis.beat_backend).
    Falls back to "auto" if config is unavailable.
    """
    try:
        from scripts.core.config import cfg
        backend = getattr(getattr(cfg, "analysis", None), "beat_backend", "auto")
    except Exception:
        backend = "auto"
    return get_tracker(backend)
