#!/usr/bin/env python3
"""
Pure Python Audio Mixing Engine

This module provides professional-grade audio mixing capabilities using only Python libraries.
No external DAW dependencies - everything is handled in Python.
"""

from __future__ import annotations
import numpy as np
import soundfile as sf
import librosa
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import time

# Try to import pyloudnorm for professional loudness measurement
try:
    import pyloudnorm as pyln
    PYLN_AVAILABLE = True
except ImportError:
    PYLN_AVAILABLE = False
    print("⚠️ pyloudnorm not available, using basic LUFS approximation")


@dataclass
class MixParameters:
    """Parameters for audio mixing."""
    vocal_gain_db: float = 0.0
    instrumental_gain_db: float = 0.0
    sidechain_amount: float = 0.0
    hp_filter_freq: float = 100.0
    reverb_send: float = 0.0
    master_limiter_threshold: float = -1.0
    master_limiter_ratio: float = 10.0
    eq_low_cut: float = 80.0
    eq_high_cut: float = 18000.0
    compression_ratio: float = 2.0
    compression_threshold: float = -12.0


@dataclass
class MixResult:
    """Result of audio mixing operation."""
    mixed_audio: np.ndarray
    sample_rate: int
    processing_time: float
    parameters_used: MixParameters
    quality_metrics: Dict[str, float]
    success: bool
    error_message: str = ""


class PythonMixer:
    """Professional Python-only audio mixer."""
    
    def __init__(self, sample_rate: int = 44100, bit_depth: int = 24):
        self.sr = sample_rate
        self.bit_depth = bit_depth
        self.max_amplitude = 2**(bit_depth - 1) - 1
        
        # Initialize loudness meter if available
        if PYLN_AVAILABLE:
            self.loudness_meter = pyln.Meter(self.sr)
        else:
            self.loudness_meter = None
    
    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file and ensure proper format."""
        audio, sr = sf.read(str(file_path))
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != self.sr:
            try:
                from scripts.core.gpu import gpu_resample
                audio = gpu_resample(audio, orig_sr=sr, target_sr=self.sr)
            except (ImportError, Exception):
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        
        return audio, self.sr
    
    def high_pass_filter(self, audio: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """Apply high-pass filter using scipy."""
        from scipy import signal
        
        # Design Butterworth high-pass filter
        nyquist = self.sr / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Ensure cutoff is valid
        if normalized_cutoff >= 1.0:
            return audio
        
        b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def low_pass_filter(self, audio: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """Apply low-pass filter using scipy."""
        from scipy import signal
        
        # Design Butterworth low-pass filter
        nyquist = self.sr / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Ensure cutoff is valid
        if normalized_cutoff >= 1.0:
            return audio
        
        b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def apply_compression(self, audio: np.ndarray, ratio: float, threshold: float, 
                         attack: float = 0.003, release: float = 0.1) -> np.ndarray:
        """Apply dynamic range compression."""
        from scipy import signal
        
        # Convert threshold from dB to linear
        threshold_linear = 10**(threshold / 20)
        
        # Calculate gain reduction
        envelope = np.abs(audio)
        
        # Smooth envelope with attack/release
        alpha_attack = np.exp(-1 / (attack * self.sr))
        alpha_release = np.exp(-1 / (release * self.sr))
        
        smoothed_envelope = np.zeros_like(envelope)
        for i in range(1, len(envelope)):
            if envelope[i] > smoothed_envelope[i-1]:
                alpha = alpha_attack
            else:
                alpha = alpha_release
            smoothed_envelope[i] = alpha * smoothed_envelope[i-1] + (1 - alpha) * envelope[i]
        
        # Calculate gain reduction
        gain_reduction = np.ones_like(smoothed_envelope)
        mask = smoothed_envelope > threshold_linear
        gain_reduction[mask] = threshold_linear + (smoothed_envelope[mask] - threshold_linear) / ratio
        gain_reduction[mask] = gain_reduction[mask] / smoothed_envelope[mask]
        
        # Apply gain reduction
        compressed_audio = audio * gain_reduction
        
        return compressed_audio
    
    def apply_sidechain_compression(self, vocals: np.ndarray, instrumentals: np.ndarray, 
                                   amount: float) -> np.ndarray:
        """Apply sidechain compression to instrumentals based on vocal level."""
        if amount <= 0:
            return instrumentals
        
        # Ensure both arrays have the same length
        min_length = min(len(vocals), len(instrumentals))
        vocals = vocals[:min_length]
        instrumentals = instrumentals[:min_length]
        
        # Calculate vocal envelope
        vocal_envelope = np.abs(vocals)
        
        # Smooth the envelope
        from scipy import signal
        b, a = signal.butter(2, 0.1, btype='low')
        vocal_envelope = signal.filtfilt(b, a, vocal_envelope)
        
        # Calculate sidechain gain reduction
        sidechain_gain = 1.0 - (vocal_envelope * amount)
        sidechain_gain = np.clip(sidechain_gain, 0.1, 1.0)  # Prevent complete silence
        
        # Apply sidechain compression
        sidechained_instrumentals = instrumentals * sidechain_gain
        
        return sidechained_instrumentals
    
    def apply_reverb(self, audio: np.ndarray, amount: float, decay_time: float = 1.0) -> np.ndarray:
        """Apply simple reverb using convolution."""
        if amount <= 0:
            return audio
        
        # Create simple impulse response for reverb
        reverb_length = int(decay_time * self.sr)
        impulse_response = np.random.randn(reverb_length) * np.exp(-np.arange(reverb_length) / (decay_time * self.sr))
        impulse_response = impulse_response / np.max(np.abs(impulse_response))
        
        # Apply convolution
        from scipy import signal
        reverb_audio = signal.convolve(audio, impulse_response, mode='same')
        
        # Mix dry and wet signals
        mixed_audio = audio * (1 - amount) + reverb_audio * amount
        
        return mixed_audio
    
    def apply_limiter(self, audio: np.ndarray, threshold: float, ratio: float = 10.0) -> np.ndarray:
        """Apply brickwall limiter."""
        # Convert threshold from dB to linear
        threshold_linear = 10**(threshold / 20)
        
        # Apply limiting
        limited_audio = np.clip(audio, -threshold_linear, threshold_linear)
        
        return limited_audio
    
    def normalize_loudness(self, audio: np.ndarray, target_lufs: float = -14.0) -> np.ndarray:
        """Normalize audio to target LUFS."""
        if self.loudness_meter is not None:
            # Use pyloudnorm for accurate loudness measurement
            current_lufs = self.loudness_meter.integrated_loudness(audio)
            gain_db = target_lufs - current_lufs
            gain_linear = 10**(gain_db / 20)
            normalized_audio = audio * gain_linear
        else:
            # Fallback to RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            target_rms = 10**(target_lufs / 20)
            gain = target_rms / (rms + 1e-10)
            normalized_audio = audio * gain
        
        return normalized_audio
    
    def mix_audio(self, vocals: np.ndarray, instrumentals: np.ndarray, 
                  params: MixParameters) -> MixResult:
        """Mix vocals and instrumentals with professional processing."""
        start_time = time.time()
        
        try:
            # Ensure both tracks have the same length
            min_length = min(len(vocals), len(instrumentals))
            vocals = vocals[:min_length]
            instrumentals = instrumentals[:min_length]
            
            # Apply high-pass filter to vocals
            vocals_filtered = self.high_pass_filter(vocals, params.hp_filter_freq)
            
            # Apply EQ (high-pass and low-pass)
            vocals_eq = self.low_pass_filter(vocals_filtered, params.eq_high_cut)
            instrumentals_eq = self.high_pass_filter(instrumentals, params.eq_low_cut)
            instrumentals_eq = self.low_pass_filter(instrumentals_eq, params.eq_high_cut)
            
            # Apply compression to vocals
            vocals_compressed = self.apply_compression(
                vocals_eq, params.compression_ratio, params.compression_threshold
            )
            
            # Apply sidechain compression to instrumentals
            instrumentals_sidechained = self.apply_sidechain_compression(
                vocals_compressed, instrumentals_eq, params.sidechain_amount
            )
            
            # Apply reverb to vocals
            vocals_reverb = self.apply_reverb(vocals_compressed, params.reverb_send)
            
            # Apply gain adjustments
            vocals_gained = vocals_reverb * (10**(params.vocal_gain_db / 20))
            instrumentals_gained = instrumentals_sidechained * (10**(params.instrumental_gain_db / 20))
            
            # Mix the tracks
            mixed_audio = vocals_gained + instrumentals_gained
            
            # Apply master limiter
            mixed_limited = self.apply_limiter(
                mixed_audio, params.master_limiter_threshold, params.master_limiter_ratio
            )
            
            # Normalize to target loudness
            mixed_final = self.normalize_loudness(mixed_limited, -14.0)
            
            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(vocals, instrumentals, mixed_final)
            
            processing_time = time.time() - start_time
            
            return MixResult(
                mixed_audio=mixed_final,
                sample_rate=self.sr,
                processing_time=processing_time,
                parameters_used=params,
                quality_metrics=quality_metrics,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return MixResult(
                mixed_audio=np.array([]),
                sample_rate=self.sr,
                processing_time=processing_time,
                parameters_used=params,
                quality_metrics={},
                success=False,
                error_message=str(e)
            )
    
    def calculate_quality_metrics(self, vocals: np.ndarray, instrumentals: np.ndarray, 
                                 mixed: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for the mix."""
        metrics = {}
        
        # LUFS measurement
        if self.loudness_meter is not None:
            metrics['lufs_integrated'] = self.loudness_meter.integrated_loudness(mixed)
        else:
            rms = np.sqrt(np.mean(mixed**2))
            metrics['lufs_integrated'] = 20 * np.log10(rms + 1e-10)
        
        # True peak
        metrics['true_peak_db'] = 20 * np.log10(np.max(np.abs(mixed)) + 1e-10)
        
        # Clipping percentage
        clipping_threshold = 0.999
        clipping_samples = np.sum(np.abs(mixed) > clipping_threshold)
        metrics['clipping_percentage'] = (clipping_samples / len(mixed)) * 100
        
        # RMS balance
        vocal_rms = np.sqrt(np.mean(vocals**2))
        instrumental_rms = np.sqrt(np.mean(instrumentals**2))
        metrics['rms_balance'] = vocal_rms / (instrumental_rms + 1e-10)
        
        # Spectral centroid (brightness)
        metrics['spectral_centroid'] = librosa.feature.spectral_centroid(y=mixed, sr=self.sr).mean()
        
        # Zero crossing rate (noise/transient content)
        metrics['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(mixed).mean()
        
        return metrics
    
    def save_mix(self, mix_result: MixResult, output_path: Path) -> bool:
        """Save the mixed audio to file."""
        try:
            # Convert to appropriate bit depth
            if self.bit_depth == 16:
                audio_int16 = (mix_result.mixed_audio * 32767).astype(np.int16)
                sf.write(str(output_path), audio_int16, mix_result.sample_rate, subtype='PCM_16')
            elif self.bit_depth == 24:
                # 24-bit is typically stored as 32-bit with 24-bit data
                audio_int32 = (mix_result.mixed_audio * 8388607).astype(np.int32)
                sf.write(str(output_path), audio_int32, mix_result.sample_rate, subtype='PCM_24')
            else:
                # 32-bit float
                sf.write(str(output_path), mix_result.mixed_audio, mix_result.sample_rate, subtype='FLOAT')
            
            return True
        except Exception as e:
            print(f"❌ Failed to save mix: {e}")
            return False


def main():
    """Example usage of the Python mixer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pure Python Audio Mixer")
    parser.add_argument("vocals", help="Path to vocal track")
    parser.add_argument("instrumentals", help="Path to instrumental track")
    parser.add_argument("output", help="Output path for mixed audio")
    parser.add_argument("--vocal-gain", type=float, default=0.0, help="Vocal gain in dB")
    parser.add_argument("--inst-gain", type=float, default=0.0, help="Instrumental gain in dB")
    parser.add_argument("--sidechain", type=float, default=0.1, help="Sidechain amount (0-1)")
    parser.add_argument("--reverb", type=float, default=0.05, help="Reverb send amount (0-1)")
    
    args = parser.parse_args()
    
    # Create mixer
    mixer = PythonMixer()
    
    # Load audio files
    vocals, sr = mixer.load_audio(Path(args.vocals))
    instrumentals, _ = mixer.load_audio(Path(args.instrumentals))
    
    # Set up mix parameters
    params = MixParameters(
        vocal_gain_db=args.vocal_gain,
        instrumental_gain_db=args.inst_gain,
        sidechain_amount=args.sidechain,
        reverb_send=args.reverb
    )
    
    # Mix the audio
    print("🎛️ Mixing audio...")
    result = mixer.mix_audio(vocals, instrumentals, params)
    
    if result.success:
        # Save the mix
        if mixer.save_mix(result, Path(args.output)):
            print(f"✅ Mix saved to: {args.output}")
            print(f"⏱️ Processing time: {result.processing_time:.2f}s")
            print(f"📊 Quality metrics:")
            for metric, value in result.quality_metrics.items():
                print(f"   {metric}: {value:.2f}")
        else:
            print("❌ Failed to save mix")
            return 1
    else:
        print(f"❌ Mixing failed: {result.error_message}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
