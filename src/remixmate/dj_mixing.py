"""
Professional DJ mixing techniques: volume balancing, crossfade curves, harmonic mixing.

This module provides:
- Volume balancing and normalization
- Dynamic crossfade curves
- Harmonic mixing (Camelot Wheel)
- Audio analysis (BPM, key detection)
- Audio processing (pitch shift, time stretch)
- Effects (EQ, reverb, delay)
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library
import logging

# Third-party
import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# VOLUME BALANCING
# ============================================================================

def calculate_rms_volume(audio):
    """
    Calculate RMS (Root Mean Square) volume.
    
    Args:
        audio: Audio array (mono or stereo)
    
    Returns:
        RMS volume value
    """
    if len(audio.shape) == 1:
        return np.sqrt(np.mean(audio ** 2))
    else:
        # Stereo: average of both channels
        return np.sqrt(
            np.mean(audio[:, 0] ** 2) + np.mean(audio[:, 1] ** 2)
        ) / 2.0

def normalize_to_target_volume(audio, target_rms):
    """
    Normalize audio to target RMS volume.
    
    Args:
        audio: Audio array
        target_rms: Target RMS value
    
    Returns:
        Normalized audio array
    """
    current_rms = calculate_rms_volume(audio)
    if current_rms == 0:
        return audio
    
    scale_factor = target_rms / current_rms
    normalized = audio * scale_factor
    
    # Prevent clipping
    max_val = np.max(np.abs(normalized))
    if max_val > 1.0:
        normalized = normalized / max_val
    
    return normalized

def enhance_vocals(vocals_audio, boost_factor=1.5):
    """
    Boost vocals by specified factor.
    
    Args:
        vocals_audio: Vocals audio array
        boost_factor: Boost factor (default 1.5 = 50% boost)
    
    Returns:
        Boosted audio array
    """
    return vocals_audio * boost_factor

def balance_stem_volumes(stems_dict):
    """
    Intelligent volume balancing.
    
    Process:
    1. Calculate average volume across all stems
    2. Normalize each stem to average
    3. Boost vocals by 50%
    4. Reduce instrumentals by 30%
    
    Args:
        stems_dict: Dict with keys like 'vocals', 'drums', 'bass', 'other'
    
    Returns:
        Balanced stems dict
    """
    # Collect all stems
    all_stems = []
    stem_names = []
    
    for key, value in stems_dict.items():
        if value is not None and len(value) > 0:
            all_stems.append(value)
            stem_names.append(key)
    
    if not all_stems:
        return stems_dict
    
    # Calculate RMS for all stems
    rms_values = [calculate_rms_volume(stem) for stem in all_stems]
    
    # Calculate average
    avg_rms = np.mean(rms_values)
    logger.info(f"Average RMS volume: {avg_rms:.4f}")
    
    # Normalize each stem to average
    balanced_stems = {}
    for i, (key, stem) in enumerate(zip(stem_names, all_stems)):
        # Normalize to average
        normalized = normalize_to_target_volume(stem, avg_rms)
        
        # Special handling for vocals and instrumentals
        if 'vocals' in key.lower():
            # Boost vocals by 50%
            normalized = enhance_vocals(normalized, boost_factor=1.5)
            logger.info(f"Boosted {key} by 50%")
        else:
            # Reduce instrumentals by 30% to give vocals space
            normalized = normalized * 0.7
            logger.info(f"Reduced {key} by 30%")
        
        # Final clipping prevention
        max_val = np.max(np.abs(normalized))
        if max_val > 1.0:
            normalized = normalized / max_val
        
        balanced_stems[key] = normalized
    
    return balanced_stems

# ============================================================================
# CROSSFADE CURVES
# ============================================================================

def create_crossfade_curve(length_samples, curve_type="linear", adaptive=True):
    """
    Create crossfade curve with various types.
    
    Supported types:
    - linear: Linear fade
    - logarithmic: Logarithmic fade (slow start, fast end)
    - exponential: Exponential fade (fast start, slow end)
    - s-curve: S-curve (smooth start and end)
    - inverse-exponential: Inverse exponential
    - cosine: Cosine fade
    - sigmoid: Sigmoid fade
    
    Args:
        length_samples: Length in samples
        curve_type: Curve type string
        adaptive: Apply smoothing for natural sound
    
    Returns:
        Array of fade values (0.0 to 1.0)
    """
    t = np.linspace(0, 1, length_samples)
    
    if curve_type == "linear":
        curve = t
    elif curve_type == "logarithmic":
        curve = np.log10(1 + 9 * t) / np.log10(10)
    elif curve_type == "exponential":
        curve = (np.exp(t * 5) - 1) / (np.exp(5) - 1)
    elif curve_type == "s-curve":
        # Hermite interpolation (smooth start and end)
        curve = t * t * (3 - 2 * t)
    elif curve_type == "inverse-exponential":
        curve = 1 - (np.exp((1 - t) * 5) - 1) / (np.exp(5) - 1)
    elif curve_type == "cosine":
        curve = (1 - np.cos(t * np.pi)) / 2
    elif curve_type == "sigmoid":
        curve = 1 / (1 + np.exp(-10 * (t - 0.5)))
    else:
        curve = t  # Default to linear
    
    # Adaptive smoothing
    if adaptive and length_samples > 100:
        # Apply gentle smoothing to reduce discontinuities
        curve = gaussian_filter1d(curve, sigma=length_samples / 1000.0)
    
    # Ensure bounds
    curve = np.clip(curve, 0.0, 1.0)
    
    return curve

# ============================================================================
# HARMONIC MIXING
# ============================================================================

# Camelot Wheel mapping
CAMELOT_WHEEL = {
    "1A": "Abm", "1B": "B",
    "2A": "Ebm", "2B": "Gb",
    "3A": "Bbm", "3B": "Db",
    "4A": "Fm", "4B": "Ab",
    "5A": "Cm", "5B": "Eb",
    "6A": "Gm", "6B": "Bb",
    "7A": "Dm", "7B": "F",
    "8A": "Am", "8B": "C",
    "9A": "Em", "9B": "G",
    "10A": "Bm", "10B": "D",
    "11A": "F#m", "11B": "A",
    "12A": "C#m", "12B": "E"
}

def find_compatible_keys(key_camelot, max_semitones=2):
    """
    Find compatible keys using Camelot Wheel.
    
    Args:
        key_camelot: Key in Camelot notation (e.g., "8A")
        max_semitones: Maximum semitones to shift
    
    Returns:
        Dict with 'compatible' (list of compatible keys) and
        'shift_options' (list of (target_key, semitones) tuples)
    """
    if key_camelot not in CAMELOT_WHEEL:
        return {"compatible": [], "shift_options": []}
    
    # Extract number and letter
    num = int(key_camelot[:-1])
    letter = key_camelot[-1]
    
    # Compatible keys (same number, adjacent numbers, opposite letter)
    compatible = [
        key_camelot,  # Same key
        f"{num}B" if letter == "A" else f"{num}A",  # Opposite letter
    ]
    
    # Adjacent numbers (wrap around)
    for offset in [-1, 1]:
        adj_num = ((num - 1 + offset) % 12) + 1
        compatible.append(f"{adj_num}{letter}")
        compatible.append(f"{adj_num}{'B' if letter == 'A' else 'A'}")
    
    # Remove duplicates
    compatible = list(set(compatible))
    
    # Shift options
    shift_options = []
    for semitones in range(-max_semitones, max_semitones + 1):
        if semitones == 0:
            continue
        # Calculate target key (simplified)
        target_num = ((num - 1 + semitones) % 12) + 1
        target_key = f"{target_num}{letter}"
        shift_options.append((target_key, semitones))
    
    return {
        "compatible": compatible,
        "shift_options": shift_options
    }

def are_keys_compatible(key1, key2):
    """
    Check if two Camelot keys are compatible.
    
    Args:
        key1: First key in Camelot notation
        key2: Second key in Camelot notation
    
    Returns:
        True if keys are compatible, False otherwise
    """
    compat_info = find_compatible_keys(key1)
    return key2 in compat_info["compatible"]

def pitch_shift_audio(audio, semitones, sr=44100):
    """
    Pitch-shift audio by specified semitones.
    
    Args:
        audio: Audio array
        semitones: Number of semitones to shift (positive = higher, negative = lower)
        sr: Sample rate
    
    Returns:
        Pitch-shifted audio
    """
    if semitones == 0:
        return audio
    
    try:
        shifted = librosa.effects.pitch_shift(
            audio, sr=sr, n_steps=semitones
        )
        return shifted
    except Exception as e:
        logger.error(f"Pitch shifting failed: {e}")
        return audio

# ============================================================================
# AUDIO ANALYSIS
# ============================================================================

def detect_bpm(audio, sr=44100):
    """
    Detect BPM using librosa.
    
    Args:
        audio: Audio array
        sr: Sample rate
    
    Returns:
        BPM value (float)
    """
    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        return float(tempo)
    except Exception as e:
        logger.error(f"BPM detection failed: {e}")
        return 128.0  # Default

def estimate_key(audio, sr=44100):
    """
    Estimate musical key using chroma features.
    
    Note: This is a simplified implementation. For production use,
    consider using a dedicated key detection library.
    
    Args:
        audio: Audio array
        sr: Sample rate
    
    Returns:
        Key in Camelot notation (e.g., "8A")
    """
    try:
        # Get chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Simple key estimation
        # Map chroma to Camelot notation (simplified)
        key_idx = np.argmax(chroma_mean)
        
        # Rough mapping
        camelot_keys = [
            "1A", "2A", "3A", "4A", "5A", "6A",
            "7A", "8A", "9A", "10A", "11A", "12A"
        ]
        return camelot_keys[key_idx % 12]
    except Exception as e:
        logger.error(f"Key estimation failed: {e}")
        return "8A"  # Default

# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def time_stretch_audio(audio, rate, sr=44100):
    """
    Time-stretch audio without changing pitch.
    
    Args:
        audio: Audio array
        rate: Stretch rate (1.0 = no change, >1.0 = faster, <1.0 = slower)
        sr: Sample rate
    
    Returns:
        Time-stretched audio
    """
    if rate == 1.0:
        return audio
    
    try:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return stretched
    except Exception as e:
        logger.error(f"Time stretching failed: {e}")
        return audio

# ============================================================================
# EFFECTS
# ============================================================================

def apply_eq_filter(audio, low_cut=0.0, high_cut=1.0, sr=44100):
    """
    Apply EQ filter (high-pass for low_cut, low-pass for high_cut).
    
    Args:
        audio: Audio array
        low_cut: Amount of low frequencies to cut (0.0-1.0)
        high_cut: Amount of high frequencies to keep (0.0-1.0)
        sr: Sample rate
    
    Returns:
        Filtered audio
    """
    if low_cut == 0.0 and high_cut == 1.0:
        return audio
    
    try:
        # High-pass filter (cut low frequencies)
        if low_cut > 0.0:
            cutoff = 200 + (low_cut * 300)  # 200-500 Hz
            b, a = signal.butter(4, cutoff / (sr / 2), 'high')
            audio = signal.filtfilt(b, a, audio)
        
        # Low-pass filter (cut high frequencies)
        if high_cut < 1.0:
            cutoff = 5000 + (high_cut * 5000)  # 5000-10000 Hz
            b, a = signal.butter(4, cutoff / (sr / 2), 'low')
            audio = signal.filtfilt(b, a, audio)
        
        return audio
    except Exception as e:
        logger.error(f"EQ filtering failed: {e}")
        return audio

def apply_reverb(audio, decay=0.3, sr=44100):
    """
    Apply simple reverb effect.
    
    Note: This is a simplified version. For production use,
    consider using a dedicated reverb library.
    
    Args:
        audio: Audio array
        decay: Reverb decay (0.0-1.0)
        sr: Sample rate
    
    Returns:
        Audio with reverb applied
    """
    if decay == 0.0:
        return audio
    
    try:
        # Simple reverb using impulse response
        delay_samples = int(0.03 * sr)  # 30ms delay
        feedback = decay
        
        reverb = audio.copy()
        for i in range(delay_samples, len(audio)):
            reverb[i] += audio[i - delay_samples] * feedback
        
        # Normalize
        max_val = np.max(np.abs(reverb))
        if max_val > 1.0:
            reverb = reverb / max_val
        
        return reverb
    except Exception as e:
        logger.error(f"Reverb failed: {e}")
        return audio

def apply_delay(audio, delay_ms=200, feedback=0.3, sr=44100):
    """
    Apply delay/echo effect.
    
    Args:
        audio: Audio array
        delay_ms: Delay time in milliseconds
        feedback: Feedback amount (0.0-1.0)
        sr: Sample rate
    
    Returns:
        Audio with delay applied
    """
    if delay_ms == 0:
        return audio
    
    try:
        delay_samples = int(delay_ms * sr / 1000.0)
        delayed = audio.copy()
        
        for i in range(delay_samples, len(audio)):
            delayed[i] += audio[i - delay_samples] * feedback
        
        # Normalize
        max_val = np.max(np.abs(delayed))
        if max_val > 1.0:
            delayed = delayed / max_val
        
        return delayed
    except Exception as e:
        logger.error(f"Delay failed: {e}")
        return audio
