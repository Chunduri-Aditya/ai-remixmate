"""
Main remix orchestrator: handles audio conversion, stem separation, mixing, and arrangement.

This module provides the core remixing functionality, including:
- Audio format conversion (multi-format support)
- Stem separation (Demucs-based)
- Classic mixing (stem-based)
- Arrangement mixing (structure-aware, timeline-based)
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library
import os
import logging
from pathlib import Path

# Third-party
import numpy as np
import librosa
import soundfile as sf

# Optional Demucs import (for stem separation)
try:
    import demucs.separate
    DEMUCS_AVAILABLE = True
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Demucs not available: {e}. Stem separation will be limited.")
    DEMUCS_AVAILABLE = False

# Local imports
from .config import (
    CONVERTED_DIR,
    AUDIO_OUTPUT_DIR,
    SEPARATED_DIR,
    SAMPLE_RATE,
    MAX_DURATION_SEC
)
from .dj_mixing import (
    balance_stem_volumes,
    detect_bpm,
    estimate_key,
    time_stretch_audio,
    create_crossfade_curve,
    apply_eq_filter,
    apply_reverb,
    apply_delay
)
from .structure_detection import detect_sections, detect_beat_grid
from .timeline_planner import TimelinePlanner
from .timeline_renderer import TimelineRenderer
from .recommendations import analyze_track_characteristics

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# AUDIO CONVERSION
# ============================================================================

def convert_audio_to_wav(input_path, output_path=None):
    """
    Convert any audio format to WAV.
    
    Supports: MP3, M4A, FLAC, OGG, AAC, WMA, AIFF, etc.
    Uses librosa (primary) and pydub (fallback).
    
    Args:
        input_path: Path to input audio file
        output_path: Optional output path (defaults to CONVERTED_DIR)
    
    Returns:
        Path to converted WAV file
    
    Raises:
        Exception: If conversion fails with both methods
    """
    try:
        # Try librosa first (faster, better quality)
        y, sr = librosa.load(input_path, sr=SAMPLE_RATE)
        
        # Ensure stereo
        if len(y.shape) == 1:
            y = np.column_stack([y, y])
        
        if output_path is None:
            os.makedirs(CONVERTED_DIR, exist_ok=True)
            filename = Path(input_path).stem + ".wav"
            output_path = os.path.join(CONVERTED_DIR, filename)
        
        sf.write(output_path, y, SAMPLE_RATE)
        logger.info(f"Converted {input_path} to {output_path}")
        return output_path
        
    except Exception as e:
        logger.warning(f"Librosa conversion failed: {e}, trying pydub")
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(2)
            
            if output_path is None:
                os.makedirs(CONVERTED_DIR, exist_ok=True)
                filename = Path(input_path).stem + ".wav"
                output_path = os.path.join(CONVERTED_DIR, filename)
            
            audio.export(output_path, format="wav")
            logger.info(f"Converted {input_path} to {output_path} using pydub")
            return output_path
            
        except Exception as e2:
            logger.error(f"Audio conversion failed: {e2}")
            raise

# ============================================================================
# STEM SEPARATION
# ============================================================================

def separate_stems(audio_path, output_dir=None):
    """
    Separate audio into stems using Demucs.
    
    Separates audio into: vocals, drums, bass, other.
    Falls back to original audio if Demucs unavailable.
    
    Args:
        audio_path: Path to audio file
        output_dir: Optional output directory (defaults to SEPARATED_DIR)
    
    Returns:
        Dict with keys 'vocals', 'drums', 'bass', 'other' mapping to file paths
    """
    if output_dir is None:
        output_dir = SEPARATED_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not DEMUCS_AVAILABLE:
        logger.warning("Demucs not available, using original audio for all stems")
        return {
            "vocals": audio_path,
            "drums": audio_path,
            "bass": audio_path,
            "other": audio_path
        }
    
    try:
        # Use Demucs to separate
        track_name = Path(audio_path).stem
        track_output_dir = os.path.join(output_dir, "htdemucs", track_name)
        
        # Run Demucs separation
        import demucs.separate
        demucs.separate.main([
            "--mp3", "--mp3-bitrate", "320",
            audio_path, "-o", output_dir
        ])
        
        # Find output files
        stems = {}
        for stem_name in ["vocals", "drums", "bass", "other"]:
            stem_path = os.path.join(track_output_dir, f"{stem_name}.wav")
            if os.path.exists(stem_path):
                stems[stem_name] = stem_path
            else:
                # Try MP3
                stem_path = os.path.join(track_output_dir, f"{stem_name}.mp3")
                if os.path.exists(stem_path):
                    stems[stem_name] = stem_path
        
        return stems
        
    except Exception as e:
        logger.error(f"Stem separation failed: {e}")
        # Fallback: return original file for all stems
        return {
            "vocals": audio_path,
            "drums": audio_path,
            "bass": audio_path,
            "other": audio_path
        }

def load_stems(stem_paths):
    """
    Load stem audio files into memory.
    
    Args:
        stem_paths: Dict mapping stem names to file paths
    
    Returns:
        Dict mapping stem names to audio arrays (numpy arrays)
    """
    stems = {}
    for name, path in stem_paths.items():
        if path and os.path.exists(path):
            try:
                y, sr = librosa.load(path, sr=SAMPLE_RATE)
                # Ensure stereo
                if len(y.shape) == 1:
                    y = np.column_stack([y, y])
                stems[name] = y
            except Exception as e:
                logger.error(f"Failed to load {name} stem: {e}")
                stems[name] = None
        else:
            stems[name] = None
    return stems

# ============================================================================
# MAIN REMIX FUNCTION
# ============================================================================

def remix_two_files(
    file1_path,
    file2_path,
    mode="mashup",
    use_arrangement_mixing=False,
    use_intelligent_mixing=False,
    crossfade_length=8.0,
    mixing_technique="crossfade",
    apply_beatmatching=True,
    apply_harmonic_mixing=False,
    bass_swap=False,
    eq_low_cut=0.0,
    eq_high_cut=1.0,
    reverb_decay=0.0,
    delay_ms=0.0,
    aggressiveness=0.5,
    energy_shape="chill_to_peak",
    output_path=None
):
    """
    Main remix function - orchestrates the entire remixing pipeline.
    
    Args:
        file1_path: Path to first audio file
        file2_path: Path to second audio file
        mode: Remix mode ("mashup", "base_vocals_match_instr", "base_instr_match_vocals")
        use_arrangement_mixing: Enable arrangement-level mixing (structure-aware)
        use_intelligent_mixing: Enable intelligent mixing (adaptive strategy)
        crossfade_length: Crossfade duration in seconds
        mixing_technique: Mixing technique ("crossfade", "bass_swap", "quick_cut")
        apply_beatmatching: Whether to apply beatmatching
        apply_harmonic_mixing: Whether to apply harmonic mixing
        bass_swap: Enable bass swap technique
        eq_low_cut: Low frequency cut (0.0-1.0)
        eq_high_cut: High frequency cut (0.0-1.0)
        reverb_decay: Reverb decay time in seconds
        delay_ms: Delay time in milliseconds
        aggressiveness: Arrangement aggressiveness (0.0-1.0)
        energy_shape: Target energy curve shape
        output_path: Optional output file path
    
    Returns:
        Path to output remix file
    
    Raises:
        Exception: If remix fails
    """
    try:
        # Convert to WAV if needed
        file1_wav = (
            convert_audio_to_wav(file1_path)
            if not file1_path.lower().endswith('.wav')
            else file1_path
        )
        file2_wav = (
            convert_audio_to_wav(file2_path)
            if not file2_path.lower().endswith('.wav')
            else file2_path
        )
        
        # Separate stems
        logger.info("Separating stems for track 1...")
        stems1_paths = separate_stems(file1_wav)
        logger.info("Separating stems for track 2...")
        stems2_paths = separate_stems(file2_wav)
        
        # Load stems
        stems1 = load_stems(stems1_paths)
        stems2 = load_stems(stems2_paths)
        
        # Arrangement-level mixing
        if use_arrangement_mixing:
            return _render_arrangement_mix(
                stems1, stems2, file1_wav, file2_wav,
                mode, aggressiveness, energy_shape, output_path
            )
        
        # Classic mixing
        return _render_classic_mix(
            stems1, stems2, mode, crossfade_length, mixing_technique,
            apply_beatmatching, bass_swap, eq_low_cut, eq_high_cut,
            reverb_decay, delay_ms, output_path
        )
        
    except Exception as e:
        logger.error(f"Remix failed: {e}")
        raise

# ============================================================================
# ARRANGEMENT MIXING
# ============================================================================

def _render_arrangement_mix(
    stems1, stems2, file1_path, file2_path,
    mode, aggressiveness, energy_shape, output_path
):
    """
    Render arrangement-level mix (structure-aware, timeline-based).
    
    Args:
        stems1: Dict of stems from track 1
        stems2: Dict of stems from track 2
        file1_path: Path to track 1
        file2_path: Path to track 2
        mode: Remix mode
        aggressiveness: Arrangement aggressiveness (0.0-1.0)
        energy_shape: Target energy curve shape
        output_path: Optional output path
    
    Returns:
        Path to output file
    """
    try:
        # Load full audio for structure detection
        audio1, sr1 = librosa.load(file1_path, sr=SAMPLE_RATE)
        audio2, sr2 = librosa.load(file2_path, sr=SAMPLE_RATE)
        
        # Detect structure
        logger.info("Detecting structure for track 1...")
        sections1 = detect_sections(audio1, sr1, "A", stems1.get("vocals"))
        logger.info("Detecting structure for track 2...")
        sections2 = detect_sections(audio2, sr2, "B", stems2.get("vocals"))
        
        # Get beat grids
        beats1, bars1, phrases1 = detect_beat_grid(audio1, sr1)
        beats2, bars2, phrases2 = detect_beat_grid(audio2, sr2)
        
        # Analyze genre
        track1_features = analyze_track_characteristics(file1_path)
        genre = track1_features.get("genre", "edm")
        
        # Plan arrangement
        planner = TimelinePlanner()
        plan = planner.plan_arrangement(
            sections1, sections2, beats1, beats2,
            genre=genre, mode=mode,
            aggressiveness=aggressiveness, energy_shape=energy_shape
        )
        
        logger.info(f"Arrangement plan: {plan.explanation}")
        
        # Detect BPMs
        bpm1 = detect_bpm(audio1, sr1)
        bpm2 = detect_bpm(audio2, sr2)
        target_bpm = bpm1
        
        # Render timeline
        renderer = TimelineRenderer(sr=SAMPLE_RATE)
        final_mix = renderer.render_timeline(
            plan, stems1, stems2, bpm1, bpm2, target_bpm
        )
        
        # Save output
        if output_path is None:
            os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(AUDIO_OUTPUT_DIR, "remix_arrangement.wav")
        
        sf.write(output_path, final_mix, SAMPLE_RATE)
        logger.info(f"Arrangement mix saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Arrangement mixing failed: {e}")
        raise

# ============================================================================
# CLASSIC MIXING
# ============================================================================

def _render_classic_mix(
    stems1, stems2, mode, crossfade_length, mixing_technique,
    apply_beatmatching, bass_swap, eq_low_cut, eq_high_cut,
    reverb_decay, delay_ms, output_path
):
    """
    Render classic stem-based mix.
    
    Args:
        stems1: Dict of stems from track 1
        stems2: Dict of stems from track 2
        mode: Remix mode
        crossfade_length: Crossfade duration
        mixing_technique: Mixing technique
        apply_beatmatching: Whether to apply beatmatching
        bass_swap: Enable bass swap
        eq_low_cut: Low frequency cut
        eq_high_cut: High frequency cut
        reverb_decay: Reverb decay
        delay_ms: Delay time
        output_path: Optional output path
    
    Returns:
        Path to output file
    """
    try:
        # Select stems based on mode
        if mode == "mashup":
            vocals = _mix_stems(stems1.get("vocals"), stems2.get("vocals"))
            drums = _mix_stems(stems1.get("drums"), stems2.get("drums"))
            bass = _mix_stems(stems1.get("bass"), stems2.get("bass"))
            other = _mix_stems(stems1.get("other"), stems2.get("other"))
        elif mode == "base_vocals_match_instr":
            vocals = stems1.get("vocals")
            drums = stems2.get("drums")
            bass = stems2.get("bass")
            other = stems2.get("other")
        else:  # base_instr_match_vocals
            vocals = stems2.get("vocals")
            drums = stems1.get("drums")
            bass = stems1.get("bass")
            other = stems1.get("other")
        
        # Balance volumes
        all_stems = {
            "vocals": vocals,
            "drums": drums,
            "bass": bass,
            "other": other
        }
        balanced = balance_stem_volumes(all_stems)
        
        # Apply mixing technique
        if mixing_technique == "crossfade":
            final = _apply_crossfade(balanced, crossfade_length)
        elif mixing_technique == "bass_swap":
            final = _apply_bass_swap(balanced, crossfade_length, bass_swap)
        else:  # quick_cut
            final = _apply_quick_cut(balanced)
        
        # Apply effects
        if eq_low_cut > 0.0 or eq_high_cut < 1.0:
            final = apply_eq_filter(final, eq_low_cut, eq_high_cut, SAMPLE_RATE)
        
        if reverb_decay > 0.0:
            final = apply_reverb(final, reverb_decay, SAMPLE_RATE)
        
        if delay_ms > 0.0:
            final = apply_delay(final, delay_ms, 0.3, SAMPLE_RATE)
        
        # Normalize
        max_val = np.max(np.abs(final))
        if max_val > 1.0:
            final = final / max_val
        
        # Save output
        if output_path is None:
            os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(AUDIO_OUTPUT_DIR, "remix.wav")
        
        sf.write(output_path, final, SAMPLE_RATE)
        logger.info(f"Classic mix saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Classic mixing failed: {e}")
        raise

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _mix_stems(stem1, stem2):
    """
    Mix two stems together.
    
    Args:
        stem1: First stem (numpy array or None)
        stem2: Second stem (numpy array or None)
    
    Returns:
        Mixed stem (numpy array) or None
    """
    if stem1 is None and stem2 is None:
        return None
    if stem1 is None:
        return stem2
    if stem2 is None:
        return stem1
    
    # Align to shortest
    min_len = min(len(stem1), len(stem2))
    return (stem1[:min_len] + stem2[:min_len]) / 2.0

def _align_stems_to_same_length(stems):
    """
    Align all stems to the same length (minimum length).
    
    Handles both mono and stereo audio. Replaces None stems with zeros.
    
    Args:
        stems: Dict with keys "vocals", "drums", "bass", "other"
    
    Returns:
        Dict with aligned stems (all same length)
    """
    # Find minimum length (first dimension)
    lengths = []
    for key, stem in stems.items():
        if stem is not None:
            lengths.append(stem.shape[0])
    
    if not lengths:
        logger.warning("All stems are None in _align_stems_to_same_length")
        return stems
    
    min_len = min(lengths)
    
    # Get reference shape from first non-None stem
    reference_stem = None
    for stem in stems.values():
        if stem is not None:
            reference_stem = stem
            break
    
    if reference_stem is None:
        logger.warning("No reference stem found")
        return stems
    
    # Align all stems to min_len
    aligned = {}
    for key, stem in stems.items():
        if stem is not None:
            # Trim to minimum length
            aligned[key] = stem[:min_len]
        else:
            # Create zeros with same shape as reference
            if reference_stem.ndim == 1:
                aligned[key] = np.zeros(min_len, dtype=reference_stem.dtype)
            else:
                aligned[key] = np.zeros(
                    (min_len, reference_stem.shape[1]),
                    dtype=reference_stem.dtype
                )
    
    return aligned

def _apply_crossfade(stems, fade_length):
    """
    Apply crossfade mixing.
    
    Args:
        stems: Dict of stems
        fade_length: Fade length in seconds
    
    Returns:
        Mixed audio array
    """
    # Align all stems to same length
    aligned_stems = _align_stems_to_same_length(stems)
    
    # Combine all stems
    combined = (
        aligned_stems["vocals"] +
        aligned_stems["drums"] +
        aligned_stems["bass"] +
        aligned_stems["other"]
    )
    
    # Simple crossfade (fade out at end)
    fade_samples = int(fade_length * SAMPLE_RATE)
    fade_curve = create_crossfade_curve(fade_samples, "s-curve")
    
    if len(combined) >= fade_samples:
        combined[-fade_samples:] *= (1.0 - fade_curve)
    
    return combined

def _apply_bass_swap(stems, fade_length, enable_bass_swap):
    """
    Apply bass swap technique.
    
    Args:
        stems: Dict of stems
        fade_length: Fade length in seconds
        enable_bass_swap: Whether to enable bass swap
    
    Returns:
        Mixed audio array
    """
    # Align all stems to same length
    aligned_stems = _align_stems_to_same_length(stems)
    
    # Combine all stems
    combined = (
        aligned_stems["vocals"] +
        aligned_stems["drums"] +
        aligned_stems["bass"] +
        aligned_stems["other"]
    )
    return combined  # Simplified implementation

def _apply_quick_cut(stems):
    """
    Apply quick cut (instant switch).
    
    Args:
        stems: Dict of stems
    
    Returns:
        Mixed audio array
    """
    # Align all stems to same length
    aligned_stems = _align_stems_to_same_length(stems)
    
    # Combine all stems
    combined = (
        aligned_stems["vocals"] +
        aligned_stems["drums"] +
        aligned_stems["bass"] +
        aligned_stems["other"]
    )
    return combined  # Simplified implementation
