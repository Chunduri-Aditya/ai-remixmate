"""
Structure detection: beat grid, phrases, sections (intro, verse, build, drop, break, outro).

This module provides:
- Beat grid detection and grouping (beats → bars → phrases)
- Section boundary detection (spectral novelty analysis)
- Energy analysis per bar
- Vocal section detection
- Complete song structure detection
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library
import logging
from dataclasses import dataclass
from typing import List, Tuple

# Third-party
import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Section:
    """Represents a section of a song."""
    track: str          # "A" or "B"
    label: str          # "intro", "build", "drop", "break", "outro", "verse"
    start_time: float
    end_time: float
    energy: float       # 0.0-1.0
    has_vocals: bool
    bars: int           # Number of bars in this section

@dataclass
class Phrase:
    """Represents a musical phrase (typically 8 or 16 bars)."""
    start_time: float
    end_time: float
    bar_start: int
    bar_end: int

# ============================================================================
# BEAT GRID DETECTION
# ============================================================================

def detect_beat_grid(audio, sr=44100):
    """
    Detect beat grid and group into bars and phrases.
    
    Process:
    1. Detect beats using librosa
    2. Group beats into bars (4 beats per bar)
    3. Group bars into phrases (8 bars per phrase)
    
    Args:
        audio: Audio array
        sr: Sample rate (default: 44100)
    
    Returns:
        Tuple of (beats, bars, phrases):
        - beats: Array of beat times in seconds
        - bars: List of (start_time, end_time) tuples
        - phrases: List of Phrase objects
    """
    try:
        # Check minimum audio length
        min_length = sr * 1.0  # At least 1 second
        if len(audio) < min_length:
            logger.warning(f"Audio too short for beat detection: {len(audio)/sr:.2f}s")
            return np.array([]), [], []
        
        # Detect beats
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        if len(beat_times) < 4:
            logger.warning("Not enough beats detected")
            return beat_times, [], []
        
        # Group into bars (4 beats per bar)
        bars = []
        for i in range(0, len(beat_times) - 3, 4):
            bars.append((
                beat_times[i],
                beat_times[min(i + 3, len(beat_times) - 1)]
            ))
        
        # Group into phrases (8 bars per phrase)
        phrases = []
        for i in range(0, len(bars) - 7, 8):
            phrases.append(Phrase(
                start_time=bars[i][0],
                end_time=bars[min(i + 7, len(bars) - 1)][1],
                bar_start=i,
                bar_end=min(i + 7, len(bars) - 1)
            ))
        
        return beat_times, bars, phrases
        
    except Exception as e:
        logger.error(f"Beat grid detection failed: {e}")
        return np.array([]), [], []

# ============================================================================
# SPECTRAL ANALYSIS
# ============================================================================

def calculate_spectral_novelty(audio, sr=44100, hop_length=512):
    """
    Calculate spectral novelty curve for section boundary detection.
    
    Uses spectral flux (difference in magnitude spectrum) to detect
    points where the audio content changes significantly.
    
    Args:
        audio: Audio array
        sr: Sample rate (default: 44100)
        hop_length: STFT hop length (default: 512)
    
    Returns:
        Tuple of (times, novelty):
        - times: Array of time points
        - novelty: Array of novelty values (higher = more change)
    """
    try:
        # Check minimum length
        min_length = hop_length * 2  # At least 2 hop windows
        if len(audio) < min_length:
            logger.warning(f"Audio too short for spectral analysis: {len(audio)/sr:.2f}s")
            return np.array([]), np.array([])
        
        # Use adaptive n_fft based on audio length
        n_fft = min(2048, len(audio) // 2)
        if n_fft < 64:
            n_fft = 64
        
        # Compute STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Spectral flux (novelty)
        novelty = np.diff(magnitude, axis=1)
        novelty = np.sum(np.maximum(novelty, 0), axis=0)
        
        # Smooth
        if len(novelty) > 1:
            novelty = gaussian_filter1d(
                novelty,
                sigma=min(3, len(novelty) // 10)
            )
        
        # Convert to time
        times = librosa.frames_to_time(
            np.arange(len(novelty)),
            sr=sr,
            hop_length=hop_length
        )
        
        return times, novelty
        
    except Exception as e:
        logger.error(f"Spectral novelty calculation failed: {e}")
        return np.array([]), np.array([])

def calculate_energy_per_bar(audio, bars, sr=44100):
    """
    Calculate RMS energy for each bar.
    
    Args:
        audio: Audio array
        bars: List of (start_time, end_time) tuples
        sr: Sample rate (default: 44100)
    
    Returns:
        Array of energy values (one per bar)
    """
    energies = []
    
    for bar_start, bar_end in bars:
        start_sample = int(bar_start * sr)
        end_sample = int(bar_end * sr)
        end_sample = min(end_sample, len(audio))
        
        if start_sample < end_sample:
            bar_audio = audio[start_sample:end_sample]
            rms = np.sqrt(np.mean(bar_audio ** 2))
            energies.append(rms)
        else:
            energies.append(0.0)
    
    return np.array(energies)

# ============================================================================
# VOCAL DETECTION
# ============================================================================

def detect_vocal_sections(audio, sr=44100):
    """
    Detect sections with vocals using spectral analysis.
    
    Uses frequency analysis in the vocal range (200-4000 Hz) combined
    with spectral centroid to identify vocal presence.
    
    Args:
        audio: Audio array
        sr: Sample rate (default: 44100)
    
    Returns:
        List of (start_time, end_time) tuples for vocal sections
    """
    try:
        # Ensure minimum length
        if len(audio) < sr * 2:  # At least 2 seconds
            return []
        
        # Use adaptive n_fft based on audio length
        n_fft = min(2048, len(audio) // 2)
        if n_fft < 64:
            n_fft = 64
        hop_length = 512
        
        # Ensure audio is long enough
        if len(audio) < n_fft:
            logger.warning(f"Audio too short for STFT: {len(audio)/sr:.2f}s")
            return []
        
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Vocal range: 200-4000 Hz
        vocal_mask = (freqs >= 200) & (freqs <= 4000)
        
        # Check if we have the right dimensions
        if len(vocal_mask) != magnitude.shape[0]:
            # Fallback: use indices
            vocal_range_start = int(200 * n_fft / sr)
            vocal_range_end = int(4000 * n_fft / sr)
            vocal_range_end = min(vocal_range_end, magnitude.shape[0])
            vocal_energy = np.mean(
                magnitude[vocal_range_start:vocal_range_end, :],
                axis=0
            )
        else:
            vocal_energy = np.mean(magnitude[vocal_mask, :], axis=0)
        
        # Spectral centroid (brightness indicator)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=hop_length
        )[0]
        
        # Combine indicators
        vocal_indicator = (
            (vocal_energy / np.max(vocal_energy)) *
            (spectral_centroid / np.max(spectral_centroid))
        )
        
        # Threshold
        threshold = np.percentile(vocal_indicator, 40)
        vocal_mask_time = vocal_indicator > threshold
        
        # Convert to time segments
        times = librosa.frames_to_time(
            np.arange(len(vocal_mask_time)),
            sr=sr,
            hop_length=hop_length
        )
        
        vocal_sections = []
        in_vocal = False
        start_time = 0.0
        
        for i, is_vocal in enumerate(vocal_mask_time):
            if is_vocal and not in_vocal:
                start_time = times[i]
                in_vocal = True
            elif not is_vocal and in_vocal:
                end_time = times[i]
                if end_time - start_time > 2.0:  # At least 2 seconds
                    vocal_sections.append((start_time, end_time))
                in_vocal = False
        
        if in_vocal:
            vocal_sections.append((start_time, times[-1]))
        
        return vocal_sections
        
    except Exception as e:
        logger.error(f"Vocal detection failed: {e}")
        return []

# ============================================================================
# SECTION DETECTION
# ============================================================================

def detect_sections(audio, sr=44100, track_label="A", vocals_audio=None):
    """
    Detect song structure: intro, verse, build, drop, break, outro.
    
    Uses a combination of:
    - Beat grid for timing
    - Energy analysis for intensity
    - Spectral novelty for boundaries
    - Vocal detection for content
    
    Args:
        audio: Audio array
        sr: Sample rate (default: 44100)
        track_label: Label for this track ("A" or "B")
        vocals_audio: Optional separate vocals audio for better detection
    
    Returns:
        List of Section objects
    """
    try:
        # Check minimum audio length
        min_duration = 5.0  # At least 5 seconds
        duration = len(audio) / sr
        
        if duration < min_duration:
            logger.warning(f"Audio too short for section detection: {duration:.2f}s")
            # Return simple fallback sections
            return [
                Section(track_label, "intro", 0.0, duration * 0.3, 0.3, False, 0),
                Section(track_label, "verse", duration * 0.3, duration * 0.7, 0.5, True, 0),
                Section(track_label, "outro", duration * 0.7, duration, 0.3, False, 0)
            ]
        
        # Get beat grid
        beats, bars, phrases = detect_beat_grid(audio, sr)
        
        if len(bars) == 0:
            # Fallback: simple time-based sections
            return [
                Section(track_label, "intro", 0.0, duration * 0.15, 0.3, False, 0),
                Section(track_label, "verse", duration * 0.15, duration * 0.5, 0.5, True, 0),
                Section(track_label, "drop", duration * 0.5, duration * 0.85, 0.8, False, 0),
                Section(track_label, "outro", duration * 0.85, duration, 0.3, False, 0)
            ]
        
        # Calculate energy per bar
        energies = calculate_energy_per_bar(audio, bars, sr)
        
        # Normalize energy
        if len(energies) > 0:
            max_energy = np.max(energies)
            if max_energy > 0:
                energies = energies / max_energy
        
        # Detect vocal sections
        if vocals_audio is not None:
            vocal_sections = detect_vocal_sections(vocals_audio, sr)
        else:
            vocal_sections = []
        
        # Spectral novelty for boundaries
        times_novelty, novelty = calculate_spectral_novelty(audio, sr)
        
        # Find section boundaries (peaks in novelty)
        if len(novelty) > 0:
            peaks, _ = find_peaks(
                novelty,
                height=np.percentile(novelty, 75),
                distance=int(5 * sr / 512)
            )
            boundary_times = times_novelty[peaks] if len(peaks) > 0 else []
        else:
            boundary_times = []
        
        # Label sections based on energy, position, and vocals
        sections = []
        
        # Intro: First 15%, low energy
        intro_end = min(
            duration * 0.15,
            bars[4][1] if len(bars) > 4 else duration * 0.15
        )
        intro_energy = (
            np.mean(energies[:min(4, len(energies))])
            if len(energies) > 0
            else 0.3
        )
        sections.append(Section(
            track_label, "intro", 0.0, intro_end,
            float(intro_energy), False, min(4, len(bars))
        ))
        
        # Main sections
        current_time = intro_end
        section_idx = 1
        
        while current_time < duration * 0.85:
            # Find next boundary or use default
            next_boundary = None
            for bt in boundary_times:
                if bt > current_time and bt < duration * 0.85:
                    next_boundary = bt
                    break
            
            if next_boundary is None:
                next_boundary = min(current_time + duration * 0.25, duration * 0.85)
            
            # Calculate energy for this section
            bar_start_idx = (
                int(current_time / (bars[0][1] - bars[0][0]))
                if len(bars) > 0
                else 0
            )
            bar_end_idx = (
                int(next_boundary / (bars[0][1] - bars[0][0]))
                if len(bars) > 0
                else len(bars)
            )
            bar_end_idx = min(bar_end_idx, len(energies))
            
            section_energy = (
                np.mean(energies[bar_start_idx:bar_end_idx])
                if bar_end_idx > bar_start_idx
                else 0.5
            )
            
            # Check for vocals
            has_vocals = any(
                vs[0] <= current_time <= vs[1] or vs[0] <= next_boundary <= vs[1]
                for vs in vocal_sections
            )
            
            # Label based on energy and position
            if section_energy > 0.7:
                label = "drop"
            elif section_energy < 0.3:
                label = "break"
            elif section_energy > 0.5 and section_idx % 2 == 0:
                label = "build"
            elif has_vocals:
                label = "verse"
            else:
                label = "verse"
            
            sections.append(Section(
                track_label, label, current_time, next_boundary,
                float(section_energy), has_vocals, bar_end_idx - bar_start_idx
            ))
            
            current_time = next_boundary
            section_idx += 1
        
        # Outro: Last 15%, decreasing energy
        outro_start = max(duration * 0.85, current_time)
        sections.append(Section(
            track_label, "outro", outro_start, duration,
            0.3, False, 0
        ))
        
        return sections
        
    except Exception as e:
        logger.error(f"Section detection failed: {e}")
        # Fallback
        duration = len(audio) / sr
        return [
            Section(track_label, "intro", 0.0, duration * 0.25, 0.3, False, 0),
            Section(track_label, "drop", duration * 0.25, duration * 0.75, 0.8, False, 0),
            Section(track_label, "outro", duration * 0.75, duration, 0.3, False, 0)
        ]
