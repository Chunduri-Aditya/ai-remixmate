"""
Auto mode selector: intelligently chooses best remix mode.
"""
import logging
import numpy as np
import librosa
from .recommendations import analyze_track_characteristics

logger = logging.getLogger(__name__)

def analyze_vocal_presence(audio_path, threshold=0.05):
    """
    Analyze if a track has significant vocals.
    
    Args:
        audio_path: Path to audio file
        threshold: RMS threshold for vocal detection
    
    Returns:
        (has_vocals, vocal_confidence) tuple
    """
    try:
        y, sr = librosa.load(audio_path, sr=44100, duration=30)  # First 30 seconds
        
        # Use spectral centroid and high-frequency energy as indicators
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        
        # High-frequency energy (vocals typically in 2-4kHz range)
        stft = librosa.stft(y)
        freqs = librosa.fft_frequencies(sr=sr)
        vocal_range_mask = (freqs >= 2000) & (freqs <= 4000)
        vocal_energy = np.mean(np.abs(stft[vocal_range_mask, :]))
        
        # Overall RMS
        rms = np.mean(librosa.feature.rms(y=y))
        
        # Heuristic: vocals present if:
        # - High spectral centroid (brightness)
        # - Significant energy in vocal range
        # - Overall RMS above threshold
        has_vocals = (
            spectral_centroid_mean > 2000 and
            vocal_energy > threshold and
            rms > threshold
        )
        
        confidence = min(1.0, (spectral_centroid_mean / 3000.0) * (vocal_energy / threshold))
        
        return (has_vocals, float(confidence))
    except Exception as e:
        logger.error(f"Vocal analysis failed: {e}")
        return (False, 0.0)

def select_auto_mode(track1_path, track2_path):
    """
    Automatically select best remix mode.
    
    Returns:
        Mode string: "mashup", "base_vocals_match_instr", "base_instr_match_vocals"
    """
    try:
        # Analyze both tracks
        track1_features = analyze_track_characteristics(track1_path)
        track2_features = analyze_track_characteristics(track2_path)
        
        track1_vocals, track1_vocal_conf = analyze_vocal_presence(track1_path)
        track2_vocals, track2_vocal_conf = analyze_vocal_presence(track2_path)
        
        # Decision logic
        # If both have vocals: mashup
        if track1_vocals and track2_vocals:
            return "mashup"
        
        # If track1 has vocals, track2 doesn't: track1 vocals + track2 instruments
        if track1_vocals and not track2_vocals:
            return "base_vocals_match_instr"
        
        # If track2 has vocals, track1 doesn't: track1 instruments + track2 vocals
        if track2_vocals and not track1_vocals:
            return "base_instr_match_vocals"
        
        # If neither has clear vocals, use mashup
        return "mashup"
    except Exception as e:
        logger.error(f"Auto mode selection failed: {e}")
        return "mashup"  # Default fallback

