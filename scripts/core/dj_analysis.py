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

    # SSM-novelty phrase boundary times (seconds); populated by analyze_structure
    phrase_boundaries: list = field(default_factory=list)

    # Downbeat times (seconds) — beat 1 of each bar.  Estimated from bar
    # alignment when using LibrosaBeatTracker; model-predicted when using
    # BeatThisTracker.  Used for bar-grid cue point snapping (Stage 2B).
    downbeat_times: list = field(default_factory=list)

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

    # TIS (Tonal Interval Space) harmonic score — continuous [0, 1].
    # Computed from chroma vectors when available; None when chroma unavailable.
    # Captures continuous harmonic content, not just Camelot key label.
    # Reference: Bernardes et al. DAFx/JNMR 2016.
    tiv_compatibility: Optional[float] = None

    # Semitones to shift Song B to align with Song A's key (0.0 = no shift)
    suggested_pitch_shift: float = 0.0

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

def analyze_structure(
    audio: np.ndarray,
    sr: int = 44100,
    key_profile: str = "auto",
) -> SongStructure:
    """
    Analyse the musical structure of an audio clip.

    Returns a SongStructure with beats, bars, and labelled sections.
    Runs in about 2–5 s for a typical 5-minute track.

    Parameters
    ----------
    key_profile : str
        Tonal profile for key detection: 'ks' (Krumhansl-Schmuckler),
        'edma' (EDM major), 'edmm' (EDM major+minor), or 'auto' (choose
        based on spectral centroid). Defaults to 'auto'.
    """
    try:
        import librosa
    except ImportError:
        log.warning("librosa not available — returning minimal SongStructure")
        return _minimal_structure(audio, sr)

    duration = len(audio) / sr
    log.info("Analysing structure (%.1f s)…", duration)

    # ── 1. Beat tracking (swappable backend) ────────────────────────
    from scripts.core.beat_tracker import get_configured_tracker
    _tracker = get_configured_tracker()
    _beat_result = _tracker.track(audio, sr)

    bpm        = _beat_result.bpm
    beat_frames = _beat_result.beat_frames
    beat_times  = _beat_result.beat_times

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

    # ── Store downbeat times for bar-grid cue snapping (Stage 2B) ───
    struct.downbeat_times = list(_beat_result.downbeat_times)

    # ── Phrase boundaries (SSM novelty + bar-grid snap) ─────────────
    # Pass beat_frames and downbeat_times from the BeatResult so that:
    #  1. The redundant beat_track() call inside _detect_phrase_boundaries is skipped.
    #  2. Raw SSM times are snapped to the nearest 8/16/32-bar downbeat (Stage 2B).
    struct.phrase_boundaries = _detect_phrase_boundaries(
        audio, sr, bpm,
        beat_frames=beat_frames,
        downbeat_times=_beat_result.downbeat_times,
    )
    if struct.phrase_boundaries:
        log.info("Phrase boundaries: %d detected (bar-grid snapped)", len(struct.phrase_boundaries))

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

    # ── Key detection with requested profile (authoritative key result) ────────
    try:
        from scripts.core.key_detection import detect_key as _detect_key
        kr = _detect_key(audio, sr, profile=key_profile)
        struct.key_name       = kr.key_name
        struct.mode           = kr.mode
        struct.camelot        = kr.camelot
        struct.key_confidence = float(kr.confidence)
        log.info(
            "Key (profile=%s): %s %s (%s) conf=%.2f",
            key_profile, kr.key_name, kr.mode, kr.camelot, kr.confidence,
        )
    except Exception as exc:
        log.debug("Key detection (profile=%s) failed: %s", key_profile, exc)

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
# Bar-grid cue point snapping (Stage 2B)
# ---------------------------------------------------------------------------

def _snap_to_bar_grid(
    boundary_times: list,
    downbeat_times: np.ndarray,
    preferred_lengths_bars: list = None,
    bar_duration_s: float = 0.0,
) -> list:
    """
    Snap raw SSM phrase-boundary times to the nearest downbeat that is also
    a multiple of a preferred phrase length (8 or 16 bars) from the song start.

    A DJ never drops a track mid-bar.  SSM novelty catches structurally
    interesting moments but their timestamps drift off the bar grid by
    several beats.  This function corrects that.

    Algorithm:
      For each raw boundary time:
        1. Find all downbeats within ±2 bars of the raw time.
        2. Among those candidates, prefer ones that fall on a multiple of
           preferred_lengths_bars bars from song start (8, 16, 32…).
        3. If no candidate aligns to a preferred length, take the nearest
           downbeat.

    Parameters
    ----------
    boundary_times : list[float]
        Raw boundary times in seconds.
    downbeat_times : np.ndarray
        Downbeat times from BeatResult (beat 1 of each bar), shape (M,).
    preferred_lengths_bars : list[int]
        Bar lengths to prefer for snapping.  Default [8, 16, 32].
    bar_duration_s : float
        Duration of one bar in seconds.  Used to compute the ±2-bar window.
        If 0.0, computed from median interval between downbeats.

    Returns
    -------
    list[float]
        Bar-grid-aligned boundary times, deduplicated, sorted.
    """
    if preferred_lengths_bars is None:
        preferred_lengths_bars = [8, 16, 32]

    if len(downbeat_times) == 0 or len(boundary_times) == 0:
        return list(boundary_times)

    downbeat_times = np.asarray(downbeat_times, dtype=np.float64)

    # Estimate bar duration from downbeat spacing
    if bar_duration_s <= 0.0:
        if len(downbeat_times) >= 2:
            bar_duration_s = float(np.median(np.diff(downbeat_times)))
        else:
            bar_duration_s = 2.0   # 120 BPM fallback

    snap_window = 2.0 * bar_duration_s   # ±2 bars search radius
    snapped: list[float] = []

    for raw_t in boundary_times:
        # Candidates: downbeats within ±2 bars
        diffs = np.abs(downbeat_times - raw_t)
        candidate_mask = diffs <= snap_window
        candidates = downbeat_times[candidate_mask]

        if len(candidates) == 0:
            # No nearby downbeat — keep raw (won't happen in well-formed tracks)
            snapped.append(raw_t)
            continue

        # Prefer candidates on multiples of preferred_lengths_bars from downbeat[0]
        origin = float(downbeat_times[0])
        best = None
        best_score = float("inf")

        for cand in candidates:
            bar_offset = (cand - origin) / bar_duration_s
            bar_index = int(round(bar_offset))
            dist = abs(cand - raw_t)

            # Score: prefer phrase-length multiples; among ties, nearest wins
            on_preferred = any(bar_index % n == 0 for n in preferred_lengths_bars)
            score = dist if on_preferred else dist + snap_window
            if score < best_score:
                best_score = score
                best = float(cand)

        snapped.append(best if best is not None else float(candidates[np.argmin(diffs[candidate_mask])]))

    # Deduplicate and sort
    seen: set[float] = set()
    result: list[float] = []
    for t in sorted(snapped):
        # Round to 3 decimal places to collapse values that differ by < 1ms
        rounded = round(t, 3)
        if rounded not in seen:
            seen.add(rounded)
            result.append(rounded)

    return result


# ---------------------------------------------------------------------------
# SSM-novelty phrase boundary detector
# ---------------------------------------------------------------------------

def _detect_phrase_boundaries(
    audio: np.ndarray,
    sr: int,
    bpm: float,
    beat_frames: Optional[np.ndarray] = None,
    downbeat_times: Optional[np.ndarray] = None,
    hop_length: int = 512,
) -> list:
    """
    Detect phrase boundaries via self-similarity matrix (SSM) novelty,
    then snap them to the bar grid.

    Algorithm:
      1. Beat-track to get beat frames/times (re-used from caller when available).
      2. Compute beat-synchronised chroma (chroma_cqt), normalise columns.
      3. Build SSM: sim = chroma_norm.T @ chroma_norm.
      4. Apply 8×8 checkerboard kernel along the diagonal → novelty curve.
      5. Peak-pick with minimum 8-beat gap (= 2 bars).
      6. Convert peak beat indices → seconds.
      7. Snap each raw boundary to the nearest 8/16/32-bar downbeat (Stage 2B).

    Args:
        beat_frames: Pre-computed beat frame indices from the outer analysis.
                     When supplied, the redundant librosa.beat.beat_track() call
                     is skipped — saves ~0.5 s per song on CPU.
        downbeat_times: Downbeat times (beat 1 of each bar) from BeatResult.
                        When supplied, raw SSM boundaries are snapped to the bar
                        grid.  When None, raw SSM times are returned unchanged.

    Returns [] on any exception — never crashes the caller.
    """
    try:
        import librosa
        from scipy.signal import find_peaks

        # Re-use caller's beat frames when available to avoid a redundant
        # beat_track() call (beat tracking is the most expensive step).
        if beat_frames is None or len(beat_frames) == 0:
            _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, trim=False)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        if len(beat_times) < 16:
            return []

        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
        chroma_beat = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        # shape: (12, n_beats)

        col_norms = np.linalg.norm(chroma_beat, axis=0, keepdims=True)
        col_norms = np.maximum(col_norms, 1e-10)
        chroma_norm = chroma_beat / col_norms

        ssm = chroma_norm.T @ chroma_norm  # (n_beats, n_beats)

        n = ssm.shape[0]
        k = 8  # checkerboard kernel size
        kernel = np.zeros((k, k))
        kernel[:k // 2, :k // 2] =  1   # top-left
        kernel[k // 2:, k // 2:] =  1   # bottom-right
        kernel[:k // 2, k // 2:] = -1   # top-right
        kernel[k // 2:, :k // 2] = -1   # bottom-left

        half_k = k // 2
        novelty = np.zeros(n)
        for i in range(half_k, n - half_k):
            patch = ssm[i - half_k:i + half_k, i - half_k:i + half_k]
            novelty[i] = float(np.sum(patch * kernel))

        peaks, _ = find_peaks(novelty, distance=8)
        raw_boundaries = [float(beat_times[p]) for p in peaks if p < len(beat_times)]

        # ── Stage 2B: snap raw SSM times to the bar grid ─────────────
        if downbeat_times is not None and len(downbeat_times) >= 2:
            bar_duration_s = float(60.0 / bpm * 4.0)
            raw_boundaries = _snap_to_bar_grid(
                raw_boundaries,
                np.asarray(downbeat_times, dtype=np.float64),
                preferred_lengths_bars=[8, 16, 32],
                bar_duration_s=bar_duration_s,
            )

        return raw_boundaries

    except Exception as exc:
        log.debug("Phrase boundary detection failed (non-critical): %s", exc)
        return []


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

    # --- Refine with SSM phrase boundaries ---
    if song_a.phrase_boundaries and song_a.bars:
        threshold_a = song_a.duration * 0.85
        candidates_a = [b for b in song_a.phrase_boundaries if b < threshold_a]
        if candidates_a:
            last_boundary = candidates_a[-1]
            nearest = min(
                range(len(song_a.bars)),
                key=lambda i: abs(song_a.bars[i][0] - last_boundary),
            )
            nearest = song_a.phrase_boundary(nearest, phrase_size)
            nearest = min(nearest, max_exit)
            exit_bar = nearest
            exit_time = song_a.bar_start_time(exit_bar)

    if song_b.phrase_boundaries and song_b.bars:
        candidates_b = [b for b in song_b.phrase_boundaries if b > 0]
        if candidates_b:
            first_boundary = candidates_b[0]
            nearest = min(
                range(len(song_b.bars)),
                key=lambda i: abs(song_b.bars[i][0] - first_boundary),
            )
            nearest = song_b.phrase_boundary(nearest, phrase_size)
            entry_bar = nearest
            entry_time = song_b.bar_start_time(entry_bar)

    # --- BPM bridge ---
    bpm_a = song_a.bpm
    bpm_b = song_b.bpm
    ratio = bpm_b / bpm_a if bpm_a > 0 else 1.0
    transition_seconds = transition_bars * (60.0 / bpm_a) * 4

    # ── Psychoacoustic consonance — computed FIRST so shortening applies
    #    before EQPlan is built (otherwise bass_swap_bar lands at the end of
    #    the shortened window instead of the midpoint, disabling bass-swap
    #    exactly when keys clash).
    harmonic_score: float = 0.5
    try:
        from scripts.core.key_detection import psychoacoustic_consonance
        harmonic_score = psychoacoustic_consonance(
            song_a.key_name, song_a.mode,
            song_b.key_name, song_b.mode,
        )
        # Shorten the transition window when consonance is low to reduce clash
        if harmonic_score < 0.35 and transition_bars >= 16:
            old_bars = transition_bars
            transition_bars = ((8 + phrase_size - 1) // phrase_size) * phrase_size
            transition_seconds = transition_bars * (60.0 / bpm_a) * 4
            log.info(
                "Harmonic clash (consonance=%.2f, %s %s→%s %s): "
                "shortened transition %d→%d bars",
                harmonic_score,
                song_a.key_name, song_a.mode,
                song_b.key_name, song_b.mode,
                old_bars, transition_bars,
            )
        else:
            log.info(
                "Psychoacoustic consonance: %.2f (%s %s → %s %s)",
                harmonic_score,
                song_a.key_name, song_a.mode,
                song_b.key_name, song_b.mode,
            )
    except Exception as exc:
        log.debug("Consonance computation failed: %s", exc)

    # --- EQ plan (built AFTER consonance shortening so bass_swap_bar
    #     always lands at the midpoint of the final transition window) ---
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

    # ── TIV (Tonal Interval Space) harmonic score ───────────────────────────
    # Computed from mean chroma vectors when available.  Complements the
    # binary Camelot adjacency check with a continuous score over actual
    # harmonic content.  Reference: Bernardes et al. DAFx/JNMR 2016.
    tiv_compatibility: Optional[float] = None
    chroma_a = getattr(song_a, "mean_chroma", None)
    chroma_b = getattr(song_b, "mean_chroma", None)
    if chroma_a is not None and chroma_b is not None:
        try:
            from scripts.core.tiv_scoring import tiv_harmonic_score
            tiv_compatibility = tiv_harmonic_score(
                np.asarray(chroma_a, dtype=np.float64),
                np.asarray(chroma_b, dtype=np.float64),
            )
            log.info(
                "TIV harmonic score: %.3f (%s %s → %s %s)",
                tiv_compatibility,
                song_a.key_name, song_a.mode,
                song_b.key_name, song_b.mode,
            )
        except Exception as exc:
            log.debug("TIV scoring failed (chroma unavailable?): %s", exc)

    # ── Pitch shift suggestion (Camelot-based) ──────────────────────────────
    suggested_pitch_shift = 0.0
    if song_a.camelot and song_b.camelot:
        try:
            from scripts.core.key_detection import pitch_shift_for_camelot
            suggested_pitch_shift = float(
                pitch_shift_for_camelot(song_b.camelot, song_a.camelot)
            )
            log.info(
                "Suggested pitch shift: %+.1f semitones (%s→%s)",
                suggested_pitch_shift, song_b.camelot, song_a.camelot,
            )
        except Exception as exc:
            log.debug("Pitch shift suggestion failed: %s", exc)

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
        tiv_compatibility=tiv_compatibility,
        suggested_pitch_shift=suggested_pitch_shift,
    )

    log.info(str(plan))
    return plan
