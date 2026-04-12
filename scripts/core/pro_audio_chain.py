"""
Professional Audio Processing Chain

This module implements a lean but professional audio processing chain:
- High-pass filtering for vocals
- Sidechain compression for intelligibility
- Subtle reverb with pre-delay
- True-peak limiting
- High-quality export (44.1kHz/24-bit)
"""

from __future__ import annotations
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import time
from scipy import signal
from scipy.signal import butter, filtfilt

# Audio processing constants
SAMPLE_RATE = 44100
BIT_DEPTH = 24
TARGET_LUFS = -14.0
TRUE_PEAK_LIMIT = -1.0  # dB
VOCAL_HIGH_PASS_FREQ = 100.0  # Hz
REVERB_PREDELAY_MS = 20.0  # ms
REVERB_DECAY_TIME = 1.5  # seconds


@dataclass
class AudioChainParams:
    """Parameters for the professional audio chain."""
    vocal_high_pass_freq: float = VOCAL_HIGH_PASS_FREQ
    sidechain_threshold: float = -12.0  # dB
    sidechain_ratio: float = 3.0
    sidechain_attack: float = 5.0  # ms
    sidechain_release: float = 100.0  # ms
    reverb_amount: float = 0.15
    reverb_predelay: float = REVERB_PREDELAY_MS
    reverb_decay: float = REVERB_DECAY_TIME
    limiter_threshold: float = TRUE_PEAK_LIMIT
    target_lufs: float = TARGET_LUFS


class ProfessionalAudioChain:
    """Professional audio processing chain for remix generation."""
    
    def __init__(self, params: Optional[AudioChainParams] = None):
        self.params = params or AudioChainParams()
        self.sr = SAMPLE_RATE
    
    def high_pass_filter(self, audio: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """
        Apply high-pass filter to remove low frequencies.
        
        Args:
            audio: Input audio
            cutoff_freq: Cutoff frequency in Hz
            
        Returns:
            High-pass filtered audio
        """
        if cutoff_freq <= 0:
            return audio
        
        try:
            # Design Butterworth high-pass filter
            nyquist = self.sr / 2
            normalized_cutoff = cutoff_freq / nyquist
            
            # Ensure cutoff is within valid range
            normalized_cutoff = min(normalized_cutoff, 0.99)
            
            b, a = butter(4, normalized_cutoff, btype='high', analog=False)
            filtered = filtfilt(b, a, audio)
            
            return filtered
            
        except Exception as e:
            print(f"⚠️ High-pass filter failed: {e}")
            return audio
    
    def sidechain_compression(self, vocals: np.ndarray, instrumentals: np.ndarray, 
                            threshold: float, ratio: float, attack: float, release: float) -> np.ndarray:
        """
        Apply sidechain compression to instrumentals based on vocal level.
        
        Args:
            vocals: Vocal audio (sidechain source)
            instrumentals: Instrumental audio (compressed)
            threshold: Compression threshold in dB
            ratio: Compression ratio
            attack: Attack time in ms
            release: Release time in ms
            
        Returns:
            Sidechain compressed instrumentals
        """
        try:
            # Ensure both arrays have the same length
            min_length = min(len(vocals), len(instrumentals))
            vocals = vocals[:min_length]
            instrumentals = instrumentals[:min_length]
            
            # Convert dB to linear
            threshold_linear = 10**(threshold / 20.0)
            
            # Calculate vocal envelope (RMS with smoothing)
            window_size = int(self.sr * 0.01)  # 10ms window
            vocal_envelope = np.zeros_like(vocals)
            
            for i in range(len(vocals)):
                start = max(0, i - window_size // 2)
                end = min(len(vocals), i + window_size // 2)
                vocal_envelope[i] = np.sqrt(np.mean(vocals[start:end]**2))
            
            # Apply attack and release smoothing
            attack_samples = int(attack * self.sr / 1000)
            release_samples = int(release * self.sr / 1000)
            
            smoothed_envelope = np.zeros_like(vocal_envelope)
            smoothed_envelope[0] = vocal_envelope[0]
            
            for i in range(1, len(vocal_envelope)):
                if vocal_envelope[i] > smoothed_envelope[i-1]:
                    # Attack
                    alpha = 1.0 / attack_samples if attack_samples > 0 else 1.0
                else:
                    # Release
                    alpha = 1.0 / release_samples if release_samples > 0 else 1.0
                
                smoothed_envelope[i] = alpha * vocal_envelope[i] + (1 - alpha) * smoothed_envelope[i-1]
            
            # Calculate compression gain reduction
            gain_reduction = np.ones_like(smoothed_envelope)
            
            for i in range(len(smoothed_envelope)):
                if smoothed_envelope[i] > threshold_linear:
                    # Calculate compression
                    over_threshold = smoothed_envelope[i] / threshold_linear
                    compressed_level = threshold_linear * (1 + (over_threshold - 1) / ratio)
                    gain_reduction[i] = compressed_level / smoothed_envelope[i]
            
            # Apply gain reduction to instrumentals
            compressed_instrumentals = instrumentals * gain_reduction
            
            return compressed_instrumentals
            
        except Exception as e:
            print(f"⚠️ Sidechain compression failed: {e}")
            return instrumentals
    
    def apply_reverb(self, audio: np.ndarray, amount: float, predelay: float, 
                    decay_time: float) -> np.ndarray:
        """
        Apply subtle reverb with pre-delay.
        
        Args:
            audio: Input audio
            amount: Reverb amount (0-1)
            predelay: Pre-delay in ms
            decay_time: Decay time in seconds
            
        Returns:
            Audio with reverb applied
        """
        if amount <= 0:
            return audio
        
        try:
            # Calculate delay parameters
            predelay_samples = int(predelay * self.sr / 1000)
            decay_samples = int(decay_time * self.sr)
            
            # Create reverb buffer
            reverb_buffer = np.zeros(len(audio) + decay_samples)
            
            # Apply reverb (simplified algorithm)
            for i in range(len(audio)):
                # Direct signal
                reverb_buffer[i] += audio[i] * (1 - amount)
                
                # Reverb signal
                if i >= predelay_samples:
                    # Multiple delay taps for reverb effect
                    for tap in [predelay_samples, predelay_samples * 2, predelay_samples * 3]:
                        if i >= tap:
                            delay_gain = amount * 0.3 * np.exp(-tap / (decay_samples * 0.3))
                            reverb_buffer[i] += audio[i - tap] * delay_gain
            
            # Apply decay envelope
            decay_envelope = np.exp(-np.arange(len(reverb_buffer)) / decay_samples)
            reverb_buffer *= decay_envelope
            
            # Return original length
            return reverb_buffer[:len(audio)]
            
        except Exception as e:
            print(f"⚠️ Reverb application failed: {e}")
            return audio
    
    def true_peak_limiter(self, audio: np.ndarray, threshold_db: float) -> np.ndarray:
        """
        Apply true-peak limiting to prevent clipping.
        
        Args:
            audio: Input audio
            threshold_db: Limiter threshold in dB
            
        Returns:
            Limited audio
        """
        try:
            # Convert threshold to linear
            threshold_linear = 10**(threshold_db / 20.0)
            
            # Find peaks above threshold
            peak_indices = np.where(np.abs(audio) > threshold_linear)[0]
            
            if len(peak_indices) == 0:
                return audio  # No limiting needed
            
            # Apply soft limiting
            limited = audio.copy()
            
            for idx in peak_indices:
                sample = audio[idx]
                if abs(sample) > threshold_linear:
                    # Soft limiting using tanh
                    limited[idx] = np.sign(sample) * threshold_linear * np.tanh(
                        abs(sample) / threshold_linear
                    )
            
            return limited
            
        except Exception as e:
            print(f"⚠️ True-peak limiting failed: {e}")
            return audio
    
    def loudness_normalize(self, audio: np.ndarray, target_lufs: float) -> np.ndarray:
        """
        Normalize audio to target LUFS level.
        
        Args:
            audio: Input audio
            target_lufs: Target LUFS level
            
        Returns:
            Loudness-normalized audio
        """
        try:
            # Calculate current RMS level
            rms_current = np.sqrt(np.mean(audio**2))
            
            if rms_current <= 0:
                return audio
            
            # Calculate target RMS (rough LUFS approximation)
            target_rms = 10**(target_lufs / 20.0)
            
            # Calculate gain adjustment
            gain = target_rms / rms_current
            
            # Limit gain to prevent excessive amplification
            max_gain = 10.0  # 20dB max gain
            gain = min(gain, max_gain)
            
            # Apply gain
            normalized = audio * gain
            
            return normalized
            
        except Exception as e:
            print(f"⚠️ Loudness normalization failed: {e}")
            return audio
    
    def process_vocals(self, vocals: np.ndarray) -> np.ndarray:
        """
        Process vocals through the professional chain.
        
        Args:
            vocals: Raw vocal audio
            
        Returns:
            Processed vocals
        """
        print("🎤 Processing vocals...")
        
        # High-pass filter to remove low-frequency noise
        processed = self.high_pass_filter(vocals, self.params.vocal_high_pass_freq)
        
        # Apply subtle reverb
        processed = self.apply_reverb(
            processed, 
            self.params.reverb_amount,
            self.params.reverb_predelay,
            self.params.reverb_decay
        )
        
        return processed
    
    def process_instrumentals(self, instrumentals: np.ndarray, vocal_sidechain: np.ndarray) -> np.ndarray:
        """
        Process instrumentals with sidechain compression.
        
        Args:
            instrumentals: Raw instrumental audio
            vocal_sidechain: Vocal audio for sidechain source
            
        Returns:
            Processed instrumentals
        """
        print("🎸 Processing instrumentals...")
        
        # Apply sidechain compression
        processed = self.sidechain_compression(
            vocal_sidechain,
            instrumentals,
            self.params.sidechain_threshold,
            self.params.sidechain_ratio,
            self.params.sidechain_attack,
            self.params.sidechain_release
        )
        
        return processed
    
    def finalize_mix(self, vocals: np.ndarray, instrumentals: np.ndarray) -> np.ndarray:
        """
        Finalize the mix with limiting and normalization.
        
        Args:
            vocals: Processed vocals
            instrumentals: Processed instrumentals
            
        Returns:
            Final mixed audio
        """
        print("🎛️ Finalizing mix...")
        
        # Align lengths
        min_length = min(len(vocals), len(instrumentals))
        vocals_aligned = vocals[:min_length]
        instrumentals_aligned = instrumentals[:min_length]
        
        # Mix vocals and instrumentals
        mixed = vocals_aligned + instrumentals_aligned
        
        # Apply true-peak limiting
        mixed = self.true_peak_limiter(mixed, self.params.limiter_threshold)
        
        # Loudness normalization
        mixed = self.loudness_normalize(mixed, self.params.target_lufs)
        
        # Final limiting to ensure no clipping
        mixed = self.true_peak_limiter(mixed, -0.1)  # Slightly below 0dB
        
        return mixed
    
    def process_remix(self, vocals: np.ndarray, instrumentals: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a complete remix through the professional chain.
        
        Args:
            vocals: Raw vocal audio
            instrumentals: Raw instrumental audio
            
        Returns:
            Tuple of (processed_audio, processing_info)
        """
        start_time = time.time()
        
        print("🎵 Processing remix through professional audio chain...")
        
        # Process vocals
        processed_vocals = self.process_vocals(vocals)
        
        # Process instrumentals with sidechain
        processed_instrumentals = self.process_instrumentals(instrumentals, vocals)
        
        # Finalize mix
        final_mix = self.finalize_mix(processed_vocals, processed_instrumentals)
        
        processing_time = time.time() - start_time
        
        # Calculate processing info
        processing_info = {
            'processing_time': processing_time,
            'sample_rate': self.sr,
            'bit_depth': BIT_DEPTH,
            'final_length': len(final_mix),
            'final_duration': len(final_mix) / self.sr,
            'parameters_used': {
                'vocal_high_pass_freq': self.params.vocal_high_pass_freq,
                'sidechain_threshold': self.params.sidechain_threshold,
                'sidechain_ratio': self.params.sidechain_ratio,
                'reverb_amount': self.params.reverb_amount,
                'limiter_threshold': self.params.limiter_threshold,
                'target_lufs': self.params.target_lufs
            }
        }
        
        print(f"✅ Professional processing completed in {processing_time:.2f}s")
        
        return final_mix, processing_info
    
    def export_high_quality(self, audio: np.ndarray, output_path: Path) -> None:
        """
        Export audio in high quality format (44.1kHz/24-bit).
        
        Args:
            audio: Audio to export
            output_path: Output file path
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to 24-bit (scale to 24-bit range)
            audio_24bit = (audio * 8388607).astype(np.int32)  # 2^23 - 1
            
            # Save as WAV file
            import soundfile as sf
            sf.write(
                str(output_path),
                audio_24bit,
                self.sr,
                subtype='PCM_24',
                format='WAV'
            )
            
            print(f"💾 High-quality audio exported to: {output_path}")
            print(f"   Sample rate: {self.sr} Hz")
            print(f"   Bit depth: {BIT_DEPTH} bits")
            print(f"   Duration: {len(audio) / self.sr:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            # Fallback to basic export
            import soundfile as sf
            sf.write(str(output_path), audio, self.sr)
            print(f"💾 Basic audio exported to: {output_path}")
    
    def print_processing_summary(self, processing_info: Dict) -> None:
        """Print processing summary."""
        print("\n" + "="*60)
        print("🎛️ PROFESSIONAL AUDIO PROCESSING SUMMARY")
        print("="*60)
        
        print(f"\n⏱️  PROCESSING")
        print(f"   Processing time: {processing_info['processing_time']:.2f}s")
        print(f"   Final duration: {processing_info['final_duration']:.2f}s")
        print(f"   Sample rate: {processing_info['sample_rate']} Hz")
        print(f"   Bit depth: {processing_info['bit_depth']} bits")
        
        params = processing_info['parameters_used']
        print(f"\n🎛️ PARAMETERS")
        print(f"   Vocal high-pass: {params['vocal_high_pass_freq']:.0f} Hz")
        print(f"   Sidechain threshold: {params['sidechain_threshold']:.1f} dB")
        print(f"   Sidechain ratio: {params['sidechain_ratio']:.1f}:1")
        print(f"   Reverb amount: {params['reverb_amount']:.2f}")
        print(f"   Limiter threshold: {params['limiter_threshold']:.1f} dB")
        print(f"   Target LUFS: {params['target_lufs']:.1f}")
        
        print("\n" + "="*60)
