"""
Advanced arrangement features: phrase precision, energy curves, sidechain, bus processing, FX.
"""
import numpy as np
import logging
from scipy import signal
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

class PhraseBoundaryEnforcer:
    """Enforces phrase boundaries (no mid-phrase cuts)."""
    
    def __init__(self, phrases):
        self.phrases = phrases
    
    def snap_to_phrase_start(self, time):
        """Snap time to nearest phrase start."""
        if not self.phrases:
            return time
        
        best_phrase = None
        best_diff = float('inf')
        
        for phrase in self.phrases:
            diff = abs(time - phrase.start_time)
            if diff < best_diff:
                best_diff = diff
                best_phrase = phrase
        
        return best_phrase.start_time if best_phrase else time
    
    def extend_to_phrase_end(self, time):
        """Extend time to end of current phrase."""
        if not self.phrases:
            return time
        
        for phrase in self.phrases:
            if phrase.start_time <= time <= phrase.end_time:
                return phrase.end_time
        
        return time

class EnergyCurvePlanner:
    """Plans target energy curves for DJ sets."""
    
    def generate_curve(self, duration, shape="chill_to_peak", num_points=100):
        """
        Generate target energy curve.
        
        Shapes:
        - chill_to_peak: Start low, build to peak
        - rollercoaster: Multiple peaks and valleys
        - slow_build: Gradual build
        - double_peak: Two peaks
        """
        times = np.linspace(0, duration, num_points)
        
        if shape == "chill_to_peak":
            curve = 0.3 + 0.7 * (times / duration) ** 2
        elif shape == "rollercoaster":
            curve = 0.4 + 0.3 * np.sin(times * 2 * np.pi / (duration / 3)) + 0.3 * (times / duration)
        elif shape == "slow_build":
            curve = 0.2 + 0.8 * (times / duration) ** 1.5
        elif shape == "double_peak":
            curve = 0.3 + 0.4 * np.sin(times * 2 * np.pi / duration) + 0.3 * (times / duration)
        else:
            curve = 0.5 * np.ones_like(times)
        
        return np.clip(curve, 0.0, 1.0)

class CompatibilityChecker:
    """Checks BPM/key compatibility and enforces rules."""
    
    def check_bpm_stretch(self, bpm1, bpm2, max_stretch=0.12):
        """Check if BPM difference is within acceptable stretch range."""
        ratio = abs(bpm1 - bpm2) / max(bpm1, bpm2)
        return ratio <= max_stretch
    
    def check_key_compatibility(self, key1, key2):
        """Check if keys are compatible."""
        from .dj_mixing import are_keys_compatible
        return are_keys_compatible(key1, key2)
    
    def prevent_back_to_back_drops(self, sections):
        """Ensure no two drops are back-to-back unless compatible."""
        # Simplified: just check if consecutive sections are both drops
        for i in range(len(sections) - 1):
            if sections[i].label == "drop" and sections[i+1].label == "drop":
                # Check if energies/genres match
                if abs(sections[i].energy - sections[i+1].energy) > 0.3:
                    return False
        return True

class GenreAwarePlanner:
    """Selects transition styles based on genre."""
    
    def get_transition_style(self, genre1, genre2):
        """Get appropriate transition style for genre pair."""
        if genre1 in ["edm", "electronic"] or genre2 in ["edm", "electronic"]:
            return "bass_swap"
        elif genre1 in ["hip-hop", "rap"] or genre2 in ["hip-hop", "rap"]:
            return "quick_cut"
        else:
            return "crossfade"

class SidechainDucker:
    """Applies sidechain ducking (kick vs everything, vocals vs instruments)."""
    
    def __init__(self, sr=44100):
        self.sr = sr
    
    def detect_kick_transients(self, drums_audio, threshold=0.3):
        """Detect kick drum transients."""
        # Simple envelope follower
        envelope = np.abs(drums_audio)
        if len(envelope.shape) > 1:
            envelope = np.mean(envelope, axis=1)
        
        # Detect peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(envelope, height=threshold * np.max(envelope), distance=int(0.1 * self.sr))
        
        return peaks / self.sr  # Convert to times
    
    def apply_ducking(self, audio, trigger_times, attack=0.01, release=0.1, depth=0.3):
        """Apply ducking based on trigger times."""
        result = audio.copy()
        samples = len(audio)
        
        # Create ducking envelope
        envelope = np.ones(samples)
        
        attack_samples = int(attack * self.sr)
        release_samples = int(release * self.sr)
        
        for trigger_time in trigger_times:
            trigger_sample = int(trigger_time * self.sr)
            if 0 <= trigger_sample < samples:
                # Attack (quick duck)
                start = max(0, trigger_sample - attack_samples)
                for i in range(start, trigger_sample):
                    t = (i - start) / attack_samples
                    envelope[i] = 1.0 - depth * t
                
                # Release (gradual recovery)
                end = min(samples, trigger_sample + release_samples)
                for i in range(trigger_sample, end):
                    t = (i - trigger_sample) / release_samples
                    envelope[i] = 1.0 - depth * (1.0 - t)
        
        # Apply envelope
        if len(result.shape) == 1:
            result = result * envelope
        else:
            result = result * envelope[:, np.newaxis]
        
        return result

class BusProcessor:
    """Applies bus processing (compression, soft clipping, stereo width)."""
    
    def __init__(self, sr=44100):
        self.sr = sr
    
    def soft_clip(self, audio, threshold=0.8):
        """Apply soft clipping for warmth."""
        # Tanh-based soft clipping
        clipped = np.tanh(audio / threshold) * threshold
        return clipped
    
    def compress(self, audio, ratio=2.0, threshold=0.7, attack=0.003, release=0.1):
        """Simple bus compressor."""
        # Simplified compressor (RMS-based)
        envelope = np.abs(audio)
        if len(envelope.shape) > 1:
            envelope = np.mean(envelope, axis=1)
        
        # Smooth envelope
        envelope = gaussian_filter1d(envelope, sigma=int(0.01 * self.sr))
        
        # Apply compression
        compressed = audio.copy()
        for i in range(len(envelope)):
            if envelope[i] > threshold:
                excess = envelope[i] - threshold
                gain_reduction = excess / ratio
                gain = 1.0 - (gain_reduction / envelope[i]) if envelope[i] > 0 else 1.0
                
                if len(compressed.shape) == 1:
                    compressed[i] *= gain
                else:
                    compressed[i] *= gain
        
        return compressed
    
    def increase_stereo_width(self, audio, width=1.2):
        """Increase stereo width (except sub-bass)."""
        if len(audio.shape) == 1:
            return audio
        
        # Mid-side processing
        mid = (audio[:, 0] + audio[:, 1]) / 2.0
        side = (audio[:, 0] - audio[:, 1]) / 2.0
        
        # Increase side
        side = side * width
        
        # Convert back to L/R
        left = mid + side
        right = mid - side
        
        return np.column_stack([left, right])

class TransitionFXLibrary:
    """Library of transition effects."""
    
    def __init__(self, sr=44100):
        self.sr = sr
    
    def noise_sweep(self, duration, direction="up", filter_type="lp"):
        """Generate noise sweep effect."""
        samples = int(duration * self.sr)
        noise = np.random.randn(samples, 2) * 0.3
        
        # Apply filter sweep
        if filter_type == "lp":
            # Low-pass filter opening up
            for i in range(samples):
                cutoff = 200 + (i / samples) * 8000 if direction == "up" else 8000 - (i / samples) * 7800
                b, a = signal.butter(2, cutoff / (self.sr / 2), 'low')
                noise[i] = signal.lfilter(b, a, noise[i])
        
        return noise
    
    def reverse_swell(self, audio, swell_duration=1.0):
        """Create reverse swell effect."""
        # Reverse, apply fade, reverse again
        reversed_audio = audio[::-1]
        
        fade_samples = int(swell_duration * self.sr)
        fade_curve = np.linspace(0, 1, fade_samples)
        
        if len(reversed_audio) >= fade_samples:
            reversed_audio[:fade_samples] *= fade_curve
        
        return reversed_audio[::-1]
    
    def filter_sweep(self, audio, start_cutoff=200, end_cutoff=8000, duration=None):
        """Apply filter sweep."""
        if duration is None:
            duration = len(audio) / self.sr
        
        samples = len(audio)
        result = audio.copy()
        
        for i in range(samples):
            t = i / samples
            cutoff = start_cutoff + (end_cutoff - start_cutoff) * t
            b, a = signal.butter(2, cutoff / (self.sr / 2), 'low')
            result[i] = signal.lfilter(b, a, result[i])
        
        return result

