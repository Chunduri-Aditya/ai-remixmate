"""
Remix generation utilities.

This module provides functions for creating remixes by combining vocals from one song
with instrumentals from another, with tempo-aware alignment.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import librosa
from pydub import AudioSegment
from pydub.effects import normalize

from .paths import SEPARATED, OUTPUT_DIR, vocals_path, other_path, ensure_directories


def load_audio_segment(file_path: Path) -> AudioSegment:
    """
    Load audio file as pydub AudioSegment.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        AudioSegment object
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    return AudioSegment.from_wav(str(file_path))


def estimate_tempo_from_audio(file_path: Path) -> float:
    """
    Estimate tempo from audio file using librosa.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Estimated tempo in BPM
    """
    y, sr = librosa.load(str(file_path), sr=22050)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


def time_stretch_audio(audio: AudioSegment, tempo_ratio: float) -> AudioSegment:
    """
    Time stretch audio to match tempo ratio.
    
    Args:
        audio: Input AudioSegment
        tempo_ratio: Ratio of target tempo to source tempo
    
    Returns:
        Time-stretched AudioSegment
    """
    if abs(tempo_ratio - 1.0) < 0.01:  # No significant tempo difference
        return audio
    
    # Use pydub's speedup method with different playback speeds
    if tempo_ratio > 1.0:
        # Speed up (higher pitch)
        return audio.speedup(playback_speed=tempo_ratio)
    else:
        # Slow down (lower pitch) - use speedup with fractional speed
        return audio.speedup(playback_speed=tempo_ratio)


def apply_fade(audio: AudioSegment, fade_in: float = 0.5, fade_out: float = 0.5) -> AudioSegment:
    """
    Apply fade in/out to audio.
    
    Args:
        audio: Input AudioSegment
        fade_in: Fade in duration in seconds
        fade_out: Fade out duration in seconds
    
    Returns:
        AudioSegment with fades applied
    """
    fade_in_ms = int(fade_in * 1000)
    fade_out_ms = int(fade_out * 1000)
    
    return audio.fade_in(fade_in_ms).fade_out(fade_out_ms)


def remix_songs(
    base_song: str,
    match_song: str,
    base_tempo: Optional[float] = None,
    match_tempo: Optional[float] = None,
    fade_sec: float = 0.5,
    output_name: Optional[str] = None
) -> Path:
    """
    Create a remix by combining vocals from base_song with instrumentals from match_song.
    
    Args:
        base_song: Name of the song to take vocals from
        match_song: Name of the song to take instrumentals from
        base_tempo: Tempo of base song (will be estimated if None)
        match_tempo: Tempo of match song (will be estimated if None)
        fade_sec: Fade in/out duration in seconds
        output_name: Custom output filename (auto-generated if None)
    
    Returns:
        Path to the generated remix file
    
    Raises:
        FileNotFoundError: If required audio files are missing
        ValueError: If audio processing fails
    """
    ensure_directories()
    
    # Get file paths
    vocals_file = vocals_path(base_song)
    instrumentals_file = other_path(match_song)
    
    if not vocals_file.exists():
        raise FileNotFoundError(f"Vocals file not found: {vocals_file}")
    if not instrumentals_file.exists():
        raise FileNotFoundError(f"Instrumentals file not found: {instrumentals_file}")
    
    # Load audio segments
    print(f"🎤 Loading vocals from: {base_song}")
    vocals = load_audio_segment(vocals_file)
    
    print(f"🎸 Loading instrumentals from: {match_song}")
    instrumentals = load_audio_segment(instrumentals_file)
    
    # Estimate tempos if not provided
    if base_tempo is None:
        print("🕺 Estimating base song tempo...")
        base_tempo = estimate_tempo_from_audio(vocals_file)
        print(f"   Base tempo: {base_tempo:.1f} BPM")
    
    if match_tempo is None:
        print("🕺 Estimating match song tempo...")
        match_tempo = estimate_tempo_from_audio(instrumentals_file)
        print(f"   Match tempo: {match_tempo:.1f} BPM")
    
    # Calculate tempo ratio for alignment
    tempo_ratio = base_tempo / match_tempo
    print(f"⚡ Tempo ratio: {tempo_ratio:.3f}")
    
    # Time stretch instrumentals to match vocals tempo
    if abs(tempo_ratio - 1.0) > 0.01:  # Significant tempo difference
        print("🔄 Time-stretching instrumentals...")
        instrumentals = time_stretch_audio(instrumentals, tempo_ratio)
    
    # Match lengths (use shorter duration)
    min_length = min(len(vocals), len(instrumentals))
    vocals = vocals[:min_length]
    instrumentals = instrumentals[:min_length]
    
    print(f"⏱️  Final duration: {min_length / 1000:.1f} seconds")
    
    # Mix vocals and instrumentals
    print("🎵 Mixing vocals and instrumentals...")
    remix = vocals.overlay(instrumentals)
    
    # Normalize and apply fades
    print("🎛️  Applying normalization and fades...")
    remix = normalize(remix)
    remix = apply_fade(remix, fade_sec, fade_sec)
    
    # Generate output filename
    if output_name is None:
        output_name = f"remix_{base_song}_{match_song}.wav"
    
    output_path = OUTPUT_DIR / output_name
    
    # Export remix
    print(f"💾 Exporting remix to: {output_path}")
    remix.export(str(output_path), format="wav")
    
    return output_path


def create_acapella(
    song_name: str,
    output_name: Optional[str] = None
) -> Path:
    """
    Create an acapella version by isolating vocals.
    
    Args:
        song_name: Name of the song
        output_name: Custom output filename (auto-generated if None)
    
    Returns:
        Path to the generated acapella file
    """
    ensure_directories()
    
    vocals_file = vocals_path(song_name)
    if not vocals_file.exists():
        raise FileNotFoundError(f"Vocals file not found: {vocals_file}")
    
    vocals = load_audio_segment(vocals_file)
    vocals = normalize(vocals)
    
    if output_name is None:
        output_name = f"acapella_{song_name}.wav"
    
    output_path = OUTPUT_DIR / output_name
    vocals.export(str(output_path), format="wav")
    
    return output_path


def create_instrumental(
    song_name: str,
    output_name: Optional[str] = None
) -> Path:
    """
    Create an instrumental version by isolating the 'other' track.
    
    Args:
        song_name: Name of the song
        output_name: Custom output filename (auto-generated if None)
    
    Returns:
        Path to the generated instrumental file
    """
    ensure_directories()
    
    instrumentals_file = other_path(song_name)
    if not instrumentals_file.exists():
        raise FileNotFoundError(f"Instrumentals file not found: {instrumentals_file}")
    
    instrumentals = load_audio_segment(instrumentals_file)
    instrumentals = normalize(instrumentals)
    
    if output_name is None:
        output_name = f"instrumental_{song_name}.wav"
    
    output_path = OUTPUT_DIR / output_name
    instrumentals.export(str(output_path), format="wav")
    
    return output_path
