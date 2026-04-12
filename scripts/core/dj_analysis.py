"""
scripts/core/dj_analysis.py — Music structure analysis for AI RemixMate.

Extracted from dj_engine.py to separate concerns:
  - Data structures (SongStructure, Section, Beat, etc.)
  - Audio analysis (beat tracking, section labeling, BPM detection)
  - Transition planning (compute mix points, EQ plans, harmonic scoring)

These are decoupled from the audio rendering engine.

Architecture note:
  analyze_structure() and plan_transition() work on raw audio (with librosa)
  to produce metadata SongStructure and TransitionPlan objects. The rendering
  engine (dj_engine.DJEngine) then applies these plans to the audio samples.

Usage:
  from scripts.core.dj_analysis import (
      analyze_structure, plan_transition,
      SongStructure, Section, TransitionPlan
  )

  structure_a = analyze_structure(vocals_a, sr)
  structure_b = analyze_structure(instrumentals_b, sr)
  plan = plan_transition(structure_a, structure_b)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

try:
    from scripts.core.config import cfg as _cfg
    _TRANSITION_BARS: int   = getattr(getattr(_cfg, "dj", None), "default_transition_bars", 16)
    _HP_START_HZ: float     = getattr(getattr(_cfg, "dj", None), "hp_filter_start_hz", 400.0)
    _HP_END_HZ: float       = getattr(getattr(_cfg, "dj", None), "hp_filter_end_hz", 80.0)
    _BASS_CROSSOVER: float  = getattr(getattr(_cfg, "dj", None), "bass_crossover_hz", 150.0)
except Exception:
    _TRANSITION_BARS = 16
    _HP_START_HZ     = 400.0
    _HP_END_HZ       = 80.0
    _BASS_CROSSOVER  = 150.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Beat:
    """A single beat in a track."""
    index: int              # 0-based beat number
    time: float             # seconds from start
    bar: int                # which bar this beat falls in (0-based)
    beat_in_bar: int        # 1–4 (position within the bar)


@dataclass
class Section:
    """A musical section of a track."""
    type: str               # "intro" | "verse" | "chorus" | "drop" | "break" | "outro" | "build"
    start_bar: int
    end_bar: int            # exclusive
    start_time: float
    end_time: float
    avg_energy: float       # 0–1 average RMS energy in this section
    avg_spectral: float     # 0–1 normalised spectral centroid (brightness)

    @property
    def length_bars(self) -> int:
        return self.end_bar - self.start_bar

    @property
    def is_mixable_exit(self) -> bool:
        """True if this section is a good place for Song A to exit."""
        return self.type in ("outro", "break", "verse")

    @property
    def is_mixable_entry(self) -> bool:
        """True if this section is a good place for Song B to enter."""
        return self.type in ("intro", "break", "verse")


@dataclass
class SongStructure:
    """Analysed structure of a single track."""
    bpm: float
    duration: float         # seconds
    beats: List[Beat]       = field(default_factory=list)
    bars: List[Tuple[float, float]] = field(default_factory=list)   # (start_s, end_s)
    sections: List[Section] = field(default_factory=list)

    # ── Music intelligence fields (populated by compute_track_vector) ─────────
    key_name:            str   = ""
    mode:                str   = ""
    camelot:             str   = ""
    key_confidence:      float = 0.0
    energy_mean:         float = 0.0
    energy_std:          float = 0.0
    drop_position:       Optional[float] = None
    danceability:        float = 0.5
    vocal_density:       float = 0.5
    spectral_centroid_hz: float = 0.0
    chord_sequence:      List[str] = field(default_factory=list)
    # numpy array stored with compare=False to avoid element-wise equality issues
    energy_curve: Optional[np.ndarray] = field(
        default=None, compare=False, repr=False,
    )

    @property
    def total_bars(self) -> int:
        return len(self.bars)

    def bar_start_time(self, bar: int) -> float:
        """Return the start time (seconds) of bar number `bar` (0-based)."""
        if 0 <= bar < len(self.bars):
            return self.bars[bar][0]
        return float(bar) * (60.0 / self.bpm) * 4

    def phrase_boundary(self, from_bar: int, phrase_size: int = 8) -> int:
        """Return the next phrase boundary at or after from_bar."""
        if from_bar % phrase_size == 0:
            return from_bar
        return from_bar + (phrase_size - from_bar % phrase_size)

    def best_exit_bar(self, phrase_size: int = 8) -> int:
        """
        Return the best bar for Song A to exit (start fading out).
        Prefers the last outro/break section aligned to a phrase boundary.
        """
        # Look for outro / break sections near the end
        for sec in reversed(self.sections):
            if sec.is_mixable_exit:
                return self.phrase_boundary(sec.start_bar, phrase_size)
        # Fallback: last 25% of track, phrase-aligned
        fallback = int(self.total_bars * 0.75)
        return self.phrase_boundary(fallback, phrase_size)

    def best_entry_bar(self, phrase_size: int = 8) -> int:
        """
        Return the best bar for Song B to enter.
        Prefers bar 0 (intro) if an intro section exists.
        """
        for sec in self.sections:
            if sec.is_mixable_entry:
                return self.phrase_boundary(sec.start_bar, phrase_size)
        return 0  # default: start of track


@dataclass
class EQPlan:
    """
    EQ automation schedule for a DJ transition.

    All times are relative to the start of the transition window (0 = mix start).
    """
    # High-pass filter on Song B (incoming)
    hp_start_hz: float      # starting cutoff (blocks bass initially)
    hp_end_hz: float        # ending cutoff (after full open)
    hp_ramp_bars: int       # number of bars to ramp down from hp_start → hp_end

    # Bass swap
    bass_swap_bar: int      # transition bar where bass ownership changes
    bass_crossover_hz: float

    # Song A fade out
    a_fade_start_bar: int   # Song A starts fading out here
    a_fade_end_bar: int     # Song A is fully out by this bar

    # Song B fade in
    b_fade_start_bar: int   # Song B volume starts rising here
    b_fade_end_bar: int     # Song B is fully in here


@dataclass
class TransitionPlan:
    """Complete plan for transitioning from Song A to Song B."""

    # Exit point in Song A
    exit_bar_a: int
    exit_time_a: float      # seconds

    # Entry point in Song B
    entry_bar_b: int
    entry_time_b: float     # seconds

    # Overlap window
    transition_bars: int
    transition_seconds: float

    # BPM delta to bridge
    bpm_a: float
    bpm_b: float
    tempo_shift_ratio: float  # bpm_b / bpm_a

    # EQ plan
    eq: EQPlan

    # Harmonic compatibility (0.0–1.0 from Camelot wheel; -1.0 = unknown)
    harmonic_score: float = -1.0

    def __str__(self) -> str:
        return (
            f"TransitionPlan: "
            f"A exits at bar {self.exit_bar_a} ({self.exit_time_a:.1f}s) | "
            f"B enters at bar {self.entry_bar_b} | "
            f"{self.transition_bars}-bar overlap | "
            f"bass swap at bar {self.eq.bass_swap_bar}"
        )


# ---------------------------------------------------------------------------
# Structure analyser
# ---------------------------------------------------------------------------

def analyze_structure(audio: np.ndarray, sr: int = 44100) -> SongStructure:
    """
    Analyse the musical structure of an audio clip.

    Returns a SongStructure with beats, bars, and labelled sections.
    Runs in about 2–5 s for a typical 5-minute track.
    """
    try:
        import librosa
    except ImportError:
        log.warning("librosa not available — returning minimal SongStructure")
        return _minimal_structure(audio, sr)

    duration = len(audio) / sr
    log.info("Analysing structure (%.1f s)…", duration)

    # ── 1. Beat tracking ──────────────────────────────────────────────
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, trim=False)
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # ── 2. Build bar grid (4 beats per bar) ───────────────────────────
    beats: List[Beat] = []
    bars: List[Tuple[float, float]] = []

    for i, t in enumerate(beat_times):
        bar_index = i // 4
        beat_in_bar = (i % 4) + 1
        beats.append(Beat(index=i, time=float(t),
                          bar=bar_index, beat_in_bar=beat_in_bar))

    for bar_idx in range(len(beat_times) // 4):
        b_start = beat_times[bar_idx * 4]
        b_end   = beat_times[min(bar_idx * 4 + 4, len(beat_times) - 1)]
        bars.append((float(b_start), float(b_end)))

    total_bars = len(bars)

    # ── 3. Per-bar energy + spectral features ─────────────────────────
    bar_energies:  List[float] = []
    bar_centroids: List[float] = []

    for bar_start, bar_end in bars:
        start_sample = int(bar_start * sr)
        end_sample   = int(bar_end   * sr)
        chunk = audio[start_sample:end_sample]

        if len(chunk) < 512:
            bar_energies.append(0.0)
            bar_centroids.append(0.0)
            continue

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        bar_energies.append(rms)

        try:
            from scripts.core.gpu import gpu_stft
            stft = np.abs(gpu_stft(chunk, n_fft=512, hop_length=128))
        except (ImportError, Exception):
            stft = np.abs(librosa.stft(chunk, n_fft=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        total_e = stft.sum()
        centroid = float((freqs[:, None] * stft).sum() / (total_e + 1e-10))
        bar_centroids.append(centroid)

    # Normalise energy + centroid to 0–1
    max_e = max(bar_energies) or 1.0
    max_c = max(bar_centroids) or 1.0
    bar_energies  = [e / max_e for e in bar_energies]
    bar_centroids = [c / max_c for c in bar_centroids]

    # ── 4. Section labelling ──────────────────────────────────────────
    sections = _label_sections(bar_energies, bar_centroids, bars)

    log.info(
        "Structure: %.1f BPM | %d bars | sections: %s",
        bpm, total_bars,
        " → ".join(s.type for s in sections),
    )

    struct = SongStructure(
        bpm=bpm,
        duration=duration,
        beats=beats,
        bars=bars,
        sections=sections,
    )

    # ── Enrich with music intelligence ──────────────────────────────
    try:
        from scripts.core.music_intelligence import compute_track_vector
        vec = compute_track_vector(audio, sr, bpm=bpm)
        struct.key_name             = vec.key_name
        struct.mode                 = vec.mode
        struct.camelot              = vec.camelot
        struct.key_confidence       = float(vec.key_confidence)
        struct.energy_curve         = vec.energy_curve
        struct.energy_mean          = float(vec.energy_mean)
        struct.energy_std           = float(vec.energy_std)
        struct.drop_position        = vec.drop_position
        struct.danceability         = float(vec.danceability)
        struct.vocal_density        = float(vec.vocal_density)
        struct.spectral_centroid_hz = float(vec.spectral_centroid)
        struct.chord_sequence       = list(vec.chord_sequence)
        log.info(
            "Music intel: key=%s %s (%s) conf=%.2f  dance=%.2f  vocal=%.2f  drop=%s",
            vec.key_name, vec.mode, vec.camelot, vec.key_confidence,
            vec.danceability, vec.vocal_density,
            f"{vec.drop_position:.1f}s" if vec.drop_position is not None else "none",
        )
    except Exception as exc:
        log.warning("Music intelligence enrichment failed (non-critical): %s", exc)

    return struct


def _minimal_structure(audio: np.ndarray, sr: int) -> SongStructure:
    """Fallback structure when librosa is unavailable."""
    duration = len(audio) / sr
    bpm = 120.0
    bar_seconds = 60.0 / bpm * 4
    total_bars = int(duration / bar_seconds)
    bars = [
        (i * bar_seconds, (i + 1) * bar_seconds)
        for i in range(total_bars)
    ]
    sections = [
        Section("intro", 0, 8, 0.0, bars[7][1] if len(bars) > 7 else duration,
                0.3, 0.3),
        Section("outro", max(0, total_bars - 8), total_bars,
                bars[-8][0] if len(bars) > 8 else 0.0, duration,
                0.3, 0.3),
    ]
    return SongStructure(bpm=bpm, duration=duration, bars=bars, sections=sections)


def _label_sections(
    energies: List[float],
    centroids: List[float],
    bars: List[Tuple[float, float]],
    min_section_bars: int = 4,
    phrase_size: int = 8,
) -> List[Section]:
    """
    Heuristic section labelling based on energy and spectral profiles.

    Rules:
      • First 8–16 bars with low energy → intro
      • Last 8–16 bars with low/falling energy → outro
      • Sustained high energy peaks → chorus / drop
      • Sustained low energy in middle → break
      • Transitions rising to high energy → build
      • Everything else → verse
    """
    n = len(energies)
    if n == 0:
        return []

    # Smooth energy curve with a small window to reduce bar-by-bar noise
    window = min(4, n)
    smoothed = np.convolve(energies, np.ones(window) / window, mode="same").tolist()

    mean_e    = float(np.mean(smoothed))
    high_thr  = mean_e + 0.2
    low_thr   = mean_e - 0.15

    # Assign a raw label to each bar
    raw: List[str] = []
    for i, e in enumerate(smoothed):
        position = i / max(n - 1, 1)  # 0 = start, 1 = end
        if e < low_thr:
            if position < 0.2:
                raw.append("intro")
            elif position > 0.8:
                raw.append("outro")
            else:
                raw.append("break")
        elif e > high_thr:
            # Check if centroid is also high (indicates "drop" vs "chorus")
            if centroids[i] > 0.65:
                raw.append("drop")
            else:
                raw.append("chorus")
        else:
            # Check for build: moderate energy but rising
            if i > 0 and smoothed[i] - smoothed[i - 1] > 0.08:
                raw.append("build")
            else:
                raw.append("verse")

    # Force intro/outro overrides based on position
    intro_end = min(16, int(n * 0.15))
    for i in range(min(8, intro_end)):
        if smoothed[i] < mean_e + 0.1:
            raw[i] = "intro"

    outro_start = max(n - 16, int(n * 0.85))
    for i in range(max(outro_start, n - 8), n):
        if smoothed[i] < mean_e + 0.1:
            raw[i] = "outro"

    # Merge consecutive same-label bars into sections
    sections: List[Section] = []
    if not raw:
        return sections

    current_type  = raw[0]
    current_start = 0

    for i in range(1, len(raw) + 1):
        label = raw[i] if i < len(raw) else None
        if label != current_type or i == len(raw):
            length = i - current_start
            if length >= min_section_bars:
                s_time = bars[current_start][0] if current_start < len(bars) else 0.0
                e_time = bars[min(i, len(bars)) - 1][1] if i <= len(bars) else bars[-1][1]
                avg_e  = float(np.mean(energies[current_start:i]))
                avg_c  = float(np.mean(centroids[current_start:i]))
                sections.append(Section(
                    type=current_type,
                    start_bar=current_start,
                    end_bar=i,
                    start_time=s_time,
                    end_time=e_time,
                    avg_energy=avg_e,
                    avg_spectral=avg_c,
                ))
            current_type  = label or current_type
            current_start = i

    return sections


# ---------------------------------------------------------------------------
# Transition planner
# ---------------------------------------------------------------------------

def plan_transition(
    song_a: SongStructure,
    song_b: SongStructure,
    transition_bars: Optional[int] = None,
    phrase_size: int = 8,
) -> TransitionPlan:
    """
    Plan a DJ transition from Song A (outgoing) to Song B (incoming).

    Parameters
    ----------
    song_a : SongStructure
        Analysed structure of the outgoing track.
    song_b : SongStructure
        Analysed structure of the incoming track.
    transition_bars : int, optional
        How many bars the overlap should be.  If None, auto-selects based on
        section lengths.  Always rounded to the nearest phrase_size.
    phrase_size : int
        Phrase boundary size (default 8 bars).

    Returns
    -------
    TransitionPlan
    """
    # --- Choose transition length ---
    if transition_bars is None:
        # Use 16 bars unless BPMs differ a lot (use 8 for safety)
        ratio = song_b.bpm / song_a.bpm if song_a.bpm > 0 else 1.0
        if abs(1.0 - ratio) > 0.08:
            transition_bars = 8   # fast-and-safe for big BPM differences
        else:
            transition_bars = _TRANSITION_BARS   # default 16
    # Snap to phrase boundary
    transition_bars = ((transition_bars + phrase_size - 1) // phrase_size) * phrase_size

    # --- Find exit point in Song A ---
    exit_bar = song_a.best_exit_bar(phrase_size)
    # Make sure there are enough bars left in A to cover the transition
    max_exit = max(0, song_a.total_bars - transition_bars)
    exit_bar = min(exit_bar, max_exit)
    exit_time = song_a.bar_start_time(exit_bar)

    # --- Find entry point in Song B ---
    entry_bar = song_b.best_entry_bar(phrase_size)
    entry_time = song_b.bar_start_time(entry_bar)

    # --- BPM bridge ---
    bpm_a = song_a.bpm
    bpm_b = song_b.bpm
    ratio = bpm_b / bpm_a if bpm_a > 0 else 1.0
    transition_seconds = transition_bars * (60.0 / bpm_a) * 4

    # --- EQ plan ---
    # Classic bass swap at the midpoint of the transition.
    # Volume fade bars span the FULL window (0 → transition_bars) so the
    # equal-power crossfade in DJEngine.render() controls amplitude — these
    # fields are kept for metadata/logging but are no longer used to drive
    # the actual crossfade curve (render() uses cos/sin directly).
    bass_swap_bar = transition_bars // 2

    eq = EQPlan(
        hp_start_hz=_HP_START_HZ,
        hp_end_hz=_HP_END_HZ,
        hp_ramp_bars=transition_bars // 2,      # HP ramp over first half
        bass_swap_bar=bass_swap_bar,
        bass_crossover_hz=_BASS_CROSSOVER,
        a_fade_start_bar=0,                      # full-window fade (metadata only)
        a_fade_end_bar=transition_bars,
        b_fade_start_bar=0,                      # full-window fade (metadata only)
        b_fade_end_bar=transition_bars,
    )

    # ── Harmonic compatibility (Camelot wheel) ────────────────────────────────
    harmonic_score = -1.0
    if song_a.camelot and song_b.camelot:
        try:
            from scripts.core.music_intelligence import camelot_harmonic_score
            harmonic_score = float(camelot_harmonic_score(song_a.camelot, song_b.camelot))
            # If tracks are harmonically incompatible (< 0.5), prefer a shorter
            # overlap so the clash window is minimised.
            if harmonic_score < 0.35 and transition_bars >= 16:
                old_bars = transition_bars
                transition_bars = ((8 + phrase_size - 1) // phrase_size) * phrase_size
                transition_seconds = transition_bars * (60.0 / bpm_a) * 4
                log.info(
                    "Harmonic clash (score=%.2f, %s→%s): shortened transition %d→%d bars",
                    harmonic_score, song_a.camelot, song_b.camelot, old_bars, transition_bars,
                )
            else:
                log.info(
                    "Harmonic score: %.2f (%s→%s)",
                    harmonic_score, song_a.camelot, song_b.camelot,
                )
        except Exception as exc:
            log.debug("Harmonic score computation failed: %s", exc)

    plan = TransitionPlan(
        exit_bar_a=exit_bar,
        exit_time_a=exit_time,
        entry_bar_b=entry_bar,
        entry_time_b=entry_time,
        transition_bars=transition_bars,
        transition_seconds=transition_seconds,
        bpm_a=bpm_a,
        bpm_b=bpm_b,
        tempo_shift_ratio=ratio,
        eq=eq,
        harmonic_score=harmonic_score,
    )

    log.info(str(plan))
    return plan
