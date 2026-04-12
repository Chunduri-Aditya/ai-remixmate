"""
Advanced Remix Engine with Optimization Proxy

This module implements high-quality remix generation using the optimization proxy concept:
- ML prediction for optimal parameters
- Constraint enforcement for audio quality
- Quality certification with performance bounds
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import librosa
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from pydub.generators import Sine, WhiteNoise
import time

from .optimization_proxy import OptimizationProxy, RemixParameters, QualityMetrics
from .features import extract_features_from_wav
from .paths import OUTPUT_DIR, vocals_path, other_path, ensure_directories


class AdvancedRemixEngine:
    """
    Advanced remix engine using optimization proxy methodology.
    
    Delivers reliable, near-optimal, real-time remix decisions with
    guaranteed constraints and provable performance bounds.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.proxy = OptimizationProxy(model_path)
        self.sample_rate = 44100  # Higher quality sample rate
        self.bit_depth = 24  # Higher bit depth for better quality
    
    def create_high_quality_remix(self, base_song: str, match_song: str, 
                                output_name: Optional[str] = None,
                                auto_optimize: bool = True) -> Tuple[Path, Dict]:
        """
        Create a high-quality remix using optimization proxy.
        
        Args:
            base_song: Name of the song to take vocals from
            match_song: Name of the song to take instrumentals from
            output_name: Custom output filename
            auto_optimize: Whether to use ML optimization
            
        Returns:
            Tuple of (output_path, performance_metrics)
        """
        start_time = time.time()
        
        print(f"🎛️ Advanced Remix Engine: {base_song} + {match_song}")
        
        # Get file paths
        vocals_file = vocals_path(base_song)
        instrumentals_file = other_path(match_song)
        
        if not vocals_file.exists() or not instrumentals_file.exists():
            raise FileNotFoundError("Required audio files not found")
        
        # Load high-quality audio
        print("🎧 Loading high-quality audio...")
        vocals = self._load_high_quality_audio(vocals_file)
        instrumentals = self._load_high_quality_audio(instrumentals_file)
        
        # Extract features for optimization
        print("🔍 Extracting audio features...")
        base_features = extract_features_from_wav(vocals_file)
        match_features = extract_features_from_wav(instrumentals_file)
        
        # Optimize parameters using proxy
        if auto_optimize:
            print("🧠 Optimizing parameters with ML proxy...")
            optimized_params, proxy_metrics = self.proxy.optimize_remix_parameters(
                base_features, match_features
            )
            print(f"   Quality bounds: {proxy_metrics['final_quality_bounds']}")
            print(f"   Optimization time: {proxy_metrics['optimization_time_ms']:.1f}ms")
        else:
            optimized_params = RemixParameters()
            proxy_metrics = {}
        
        # Apply advanced audio processing
        print("🎵 Applying advanced audio processing...")
        processed_vocals = self._apply_advanced_processing(vocals, optimized_params, "vocals")
        processed_instrumentals = self._apply_advanced_processing(instrumentals, optimized_params, "instrumentals")
        
        # Time alignment and tempo matching
        print("⏱️ Time alignment and tempo matching...")
        aligned_vocals, aligned_instrumentals = self._advanced_time_alignment(
            processed_vocals, processed_instrumentals, optimized_params
        )
        
        # Advanced mixing with EQ and effects
        print("🎛️ Advanced mixing with EQ and effects...")
        final_mix = self._advanced_mixing(aligned_vocals, aligned_instrumentals, optimized_params)
        
        # Quality enhancement
        print("✨ Quality enhancement...")
        enhanced_mix = self._enhance_audio_quality(final_mix)
        
        # Generate output filename
        if output_name is None:
            output_name = f"advanced_remix_{base_song}_{match_song}.wav"
        
        output_path = OUTPUT_DIR / output_name
        
        # Export high-quality audio
        print(f"💾 Exporting high-quality audio to: {output_path}")
        enhanced_mix.export(
            str(output_path),
            format="wav",
            parameters=["-acodec", "pcm_s24le", "-ar", str(self.sample_rate)]
        )
        
        # Calculate final performance metrics
        total_time = (time.time() - start_time) * 1000
        final_metrics = {
            "total_processing_time_ms": total_time,
            "proxy_metrics": proxy_metrics,
            "output_file_size_mb": output_path.stat().st_size / (1024 * 1024),
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "optimization_used": auto_optimize
        }
        
        print(f"✅ Advanced remix completed in {total_time:.1f}ms")
        return output_path, final_metrics
    
    def _load_high_quality_audio(self, file_path: Path) -> AudioSegment:
        """Load audio with high quality settings."""
        audio = AudioSegment.from_wav(str(file_path))
        
        # Resample to higher quality if needed
        if audio.frame_rate < self.sample_rate:
            audio = audio.set_frame_rate(self.sample_rate)
        
        # Convert to higher bit depth
        audio = audio.set_sample_width(3)  # 24-bit
        
        return audio
    
    def _apply_advanced_processing(self, audio: AudioSegment, params: RemixParameters, 
                                 track_type: str) -> AudioSegment:
        """Apply advanced audio processing based on optimized parameters."""
        processed = audio
        
        # Apply EQ
        if any([params.eq_low, params.eq_mid, params.eq_high]):
            processed = self._apply_eq(processed, params)
        
        # Apply compression
        if params.compression_ratio > 1.0:
            processed = self._apply_compression(processed, params.compression_ratio)
        
        # Apply reverb
        if params.reverb_amount > 0.0:
            processed = self._apply_reverb(processed, params.reverb_amount)
        
        # Normalize
        processed = normalize(processed)
        
        return processed
    
    def _apply_eq(self, audio: AudioSegment, params: RemixParameters) -> AudioSegment:
        """Apply equalization based on parameters."""
        # Simple EQ implementation using frequency filtering
        # In a production system, this would use more sophisticated EQ
        
        # Apply low frequency adjustment
        if params.eq_low != 0.0:
            # Convert to numpy for processing
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            
            # Simple low-pass filter for low EQ
            if params.eq_low > 0:
                # Boost low frequencies
                samples = self._apply_low_pass_boost(samples, params.eq_low)
            else:
                # Cut low frequencies
                samples = self._apply_low_pass_cut(samples, abs(params.eq_low))
            
            # Convert back to AudioSegment
            if audio.channels == 2:
                samples = samples.flatten()
            audio = AudioSegment(
                samples.tobytes(),
                frame_rate=audio.frame_rate,
                sample_width=audio.sample_width,
                channels=audio.channels
            )
        
        return audio
    
    def _apply_low_pass_boost(self, samples: np.ndarray, gain_db: float) -> np.ndarray:
        """Apply low-pass boost filter."""
        # Simple implementation - in production would use proper filters
        gain_linear = 10 ** (gain_db / 20)
        return samples * (1 + gain_linear * 0.1)
    
    def _apply_low_pass_cut(self, samples: np.ndarray, cut_db: float) -> np.ndarray:
        """Apply low-pass cut filter."""
        # Simple implementation - in production would use proper filters
        cut_linear = 10 ** (-cut_db / 20)
        return samples * (1 - cut_linear * 0.1)
    
    def _apply_compression(self, audio: AudioSegment, ratio: float) -> AudioSegment:
        """Apply dynamic range compression."""
        return compress_dynamic_range(audio, threshold=-20.0, ratio=ratio)
    
    def _apply_reverb(self, audio: AudioSegment, amount: float) -> AudioSegment:
        """Apply reverb effect."""
        # Simple reverb implementation using delay and feedback
        # In production, would use more sophisticated reverb algorithms
        
        delay_ms = int(100 * amount)  # Delay in milliseconds
        feedback = 0.3 * amount  # Feedback amount
        
        # Create delayed version
        delayed = audio + AudioSegment.silent(duration=delay_ms)
        
        # Mix with original
        reverb_audio = audio.overlay(delayed, gain_during_overlay=-20)
        
        return reverb_audio
    
    def _advanced_time_alignment(self, vocals: AudioSegment, instrumentals: AudioSegment,
                               params: RemixParameters) -> Tuple[AudioSegment, AudioSegment]:
        """Advanced time alignment and tempo matching."""
        # Apply tempo ratio
        if abs(params.tempo_ratio - 1.0) > 0.01:
            instrumentals = instrumentals.speedup(playback_speed=params.tempo_ratio)
        
        # Match lengths
        min_length = min(len(vocals), len(instrumentals))
        vocals = vocals[:min_length]
        instrumentals = instrumentals[:min_length]
        
        return vocals, instrumentals
    
    def _advanced_mixing(self, vocals: AudioSegment, instrumentals: AudioSegment,
                        params: RemixParameters) -> AudioSegment:
        """Advanced mixing with volume balance and effects."""
        # Apply volume balance
        vocals_volume = (1.0 - params.volume_balance) * 100  # Convert to percentage
        instrumentals_volume = params.volume_balance * 100
        
        vocals = vocals + vocals_volume
        instrumentals = instrumentals + instrumentals_volume
        
        # Apply fades
        if params.fade_in_duration > 0:
            vocals = vocals.fade_in(int(params.fade_in_duration * 1000))
            instrumentals = instrumentals.fade_in(int(params.fade_in_duration * 1000))
        
        if params.fade_out_duration > 0:
            vocals = vocals.fade_out(int(params.fade_out_duration * 1000))
            instrumentals = instrumentals.fade_out(int(params.fade_out_duration * 1000))
        
        # Mix the tracks
        mix = vocals.overlay(instrumentals)
        
        return mix
    
    def _enhance_audio_quality(self, audio: AudioSegment) -> AudioSegment:
        """Enhance audio quality with final processing."""
        # Normalize to optimal levels
        audio = normalize(audio)
        
        # Apply gentle compression for consistency
        audio = compress_dynamic_range(audio, threshold=-12.0, ratio=2.0)
        
        # Final normalization
        audio = normalize(audio)
        
        return audio
    
    def evaluate_remix_quality(self, remix_path: Path) -> QualityMetrics:
        """Evaluate the quality of a generated remix."""
        return self.proxy.extract_quality_metrics(remix_path)
    
    def generate_quality_report(self, remix_path: Path, metrics: Dict) -> str:
        """Generate a detailed quality report."""
        quality_metrics = self.evaluate_remix_quality(remix_path)
        
        report = f"""
🎵 Advanced Remix Quality Report
================================

📁 File: {remix_path.name}
📊 Size: {metrics.get('output_file_size_mb', 0):.1f} MB
⏱️ Processing Time: {metrics.get('total_processing_time_ms', 0):.1f} ms
🎛️ Sample Rate: {metrics.get('sample_rate', 0)} Hz
🔢 Bit Depth: {metrics.get('bit_depth', 0)} bit

🎧 Audio Quality Metrics:
   RMS Energy: {quality_metrics.rms_energy:.4f}
   Dynamic Range: {quality_metrics.dynamic_range:.4f}
   Spectral Centroid: {quality_metrics.spectral_centroid:.1f} Hz
   Zero Crossing Rate: {quality_metrics.zero_crossing_rate:.4f}
   Harmonic Ratio: {quality_metrics.harmonic_ratio:.4f}

🧠 ML Optimization:
   Used: {metrics.get('optimization_used', False)}
   Quality Bounds: {metrics.get('proxy_metrics', {}).get('final_quality_bounds', 'N/A')}
   Constraint Violations: {metrics.get('proxy_metrics', {}).get('constraint_violations', 'N/A')}

✨ Overall Quality: {'Excellent' if quality_metrics.rms_energy > 0.1 else 'Good' if quality_metrics.rms_energy > 0.05 else 'Needs Improvement'}
"""
        return report
