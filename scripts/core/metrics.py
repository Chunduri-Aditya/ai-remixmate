"""
Objective Audio Quality Metrics for Remix Evaluation

This module provides comprehensive metrics for evaluating remix quality:
- Tempo error (BPM accuracy)
- Key match (Camelot wheel compatibility)
- LUFS loudness measurement
- Clipping detection
- RMS balance analysis
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
"""

from __future__ import annotations
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import json
import time

# Audio processing constants — pulled from central config, with fallback defaults
# so this module stays importable even before config is fully initialised.
try:
    from scripts.core.config import cfg as _cfg
    TARGET_LUFS = _cfg.audio.target_lufs
    SAMPLE_RATE = _cfg.audio.sample_rate
except Exception:
    TARGET_LUFS = -14.0
    SAMPLE_RATE = 44100

CLIPPING_THRESHOLD = 0.99  # Clipping detection threshold


@dataclass
class TempoMetrics:
    """Tempo-related quality metrics."""
    base_tempo: float
    match_tempo: float
    tempo_error_bpm: float
    tempo_ratio: float
    beat_alignment_ms: float  # Beat alignment accuracy in milliseconds


@dataclass
class KeyMetrics:
    """Key and harmonic compatibility metrics."""
    base_key: str
    match_key: str
    key_compatibility: float  # 0-1 score
    camelot_distance: int  # Steps on Camelot wheel
    pitch_shift_applied: float  # Semitones shifted


@dataclass
class LoudnessMetrics:
    """Loudness and dynamic range metrics."""
    lufs_integrated: float
    lufs_target: float
    lufs_error: float
    true_peak_db: float
    clipping_percentage: float
    dynamic_range_db: float


@dataclass
class BalanceMetrics:
    """Audio balance and mixing metrics."""
    vocal_rms: float
    instrumental_rms: float
    rms_balance_ratio: float
    spectral_centroid_vocals: float
    spectral_centroid_instrumentals: float
    spectral_balance: float


@dataclass
class QualityMetrics:
    """Comprehensive quality assessment."""
    si_sdr: float  # Scale-Invariant Signal-to-Distortion Ratio
    perceptual_quality: float  # Overall quality score 0-1
    intelligibility_score: float  # Vocal intelligibility proxy
    musical_coherence: float  # Musical compatibility score
    technical_quality: float  # Technical audio quality score


class AudioMetrics:
    """Comprehensive audio quality metrics calculator."""
    
    def __init__(self):
        self.camelot_wheel = self._init_camelot_wheel()
    
    def _init_camelot_wheel(self) -> Dict[str, int]:
        """Initialize Camelot wheel mapping for key compatibility."""
        return {
            # Major keys
            'C': 8, 'G': 9, 'D': 10, 'A': 11, 'E': 12, 'B': 1, 'F#': 2, 'C#': 3,
            'G#': 4, 'D#': 5, 'A#': 6, 'F': 7,
            # Minor keys  
            'Am': 5, 'Em': 6, 'Bm': 7, 'F#m': 8, 'C#m': 9, 'G#m': 10, 'D#m': 11,
            'A#m': 12, 'Fm': 1, 'Cm': 2, 'Gm': 3, 'Dm': 4
        }
    
    def detect_key(self, audio: np.ndarray, sr: int) -> str:
        """Detect musical key using chroma analysis."""
        try:
            # Extract chroma features
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            
            # Get key profile correlation
            key_profiles = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=12)
            key_correlations = np.zeros(24)  # 12 major + 12 minor keys
            
            # Major key profiles (Krumhansl-Schmuckler)
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            # Calculate correlations for all keys
            for i in range(12):
                # Major keys
                major_corr = np.corrcoef(chroma.mean(axis=1), np.roll(major_profile, i))[0, 1]
                key_correlations[i] = major_corr if not np.isnan(major_corr) else 0
                
                # Minor keys
                minor_corr = np.corrcoef(chroma.mean(axis=1), np.roll(minor_profile, i))[0, 1]
                key_correlations[i + 12] = minor_corr if not np.isnan(minor_corr) else 0
            
            # Find best key
            best_key_idx = np.argmax(key_correlations)
            
            if best_key_idx < 12:  # Major key
                key_names = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
                return key_names[best_key_idx]
            else:  # Minor key
                key_names = ['Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m', 'Fm', 'Cm', 'Gm', 'Dm']
                return key_names[best_key_idx - 12]
                
        except Exception:
            return 'C'  # Default fallback
    
    def calculate_tempo_metrics(self, base_audio: np.ndarray, match_audio: np.ndarray, 
                              sr: int) -> TempoMetrics:
        """Calculate tempo-related metrics."""
        # Extract tempos
        base_tempo, _ = librosa.beat.beat_track(y=base_audio, sr=sr)
        match_tempo, _ = librosa.beat.beat_track(y=match_audio, sr=sr)
        
        # Calculate tempo error
        tempo_error = abs(base_tempo - match_tempo)
        tempo_ratio = base_tempo / match_tempo if match_tempo > 0 else 1.0
        
        # Estimate beat alignment (simplified)
        beat_alignment_ms = min(tempo_error * 10, 100)  # Rough estimate
        
        return TempoMetrics(
            base_tempo=float(base_tempo),
            match_tempo=float(match_tempo),
            tempo_error_bpm=float(tempo_error),
            tempo_ratio=float(tempo_ratio),
            beat_alignment_ms=float(beat_alignment_ms)
        )
    
    def calculate_key_metrics(self, base_audio: np.ndarray, match_audio: np.ndarray, 
                            sr: int) -> KeyMetrics:
        """Calculate key compatibility metrics."""
        base_key = self.detect_key(base_audio, sr)
        match_key = self.detect_key(match_audio, sr)
        
        # Calculate Camelot wheel distance
        base_camelot = self.camelot_wheel.get(base_key, 8)
        match_camelot = self.camelot_wheel.get(match_key, 8)
        
        # Calculate distance (circular)
        distance = min(abs(base_camelot - match_camelot), 
                      12 - abs(base_camelot - match_camelot))
        
        # Key compatibility score (0-1, higher is better)
        compatibility = max(0, 1 - distance / 6)  # Perfect match = 1, 6+ steps = 0
        
        # Determine if pitch shift is needed
        pitch_shift = 0.0
        if distance > 2:  # If keys are too far apart
            # Calculate optimal pitch shift (simplified)
            if distance <= 6:
                pitch_shift = distance * 0.5  # Rough semitone mapping
        
        return KeyMetrics(
            base_key=base_key,
            match_key=match_key,
            key_compatibility=float(compatibility),
            camelot_distance=int(distance),
            pitch_shift_applied=float(pitch_shift)
        )
    
    def calculate_loudness_metrics(self, audio: np.ndarray, sr: int) -> LoudnessMetrics:
        """Calculate loudness and dynamic range metrics."""
        # Calculate RMS for LUFS approximation
        rms = np.sqrt(np.mean(audio**2))
        lufs_approx = 20 * np.log10(rms + 1e-10)  # Rough LUFS approximation
        
        # Calculate true peak
        true_peak = np.max(np.abs(audio))
        true_peak_db = 20 * np.log10(true_peak + 1e-10)
        
        # Calculate clipping percentage
        clipped_samples = np.sum(np.abs(audio) > CLIPPING_THRESHOLD)
        clipping_percentage = (clipped_samples / len(audio)) * 100
        
        # Calculate dynamic range
        dynamic_range_db = 20 * np.log10(np.max(audio) / (np.mean(audio) + 1e-10))
        
        return LoudnessMetrics(
            lufs_integrated=float(lufs_approx),
            lufs_target=TARGET_LUFS,
            lufs_error=float(abs(lufs_approx - TARGET_LUFS)),
            true_peak_db=float(true_peak_db),
            clipping_percentage=float(clipping_percentage),
            dynamic_range_db=float(dynamic_range_db)
        )
    
    def calculate_balance_metrics(self, vocals: np.ndarray, instrumentals: np.ndarray, 
                                sr: int) -> BalanceMetrics:
        """Calculate audio balance and mixing metrics."""
        # Calculate RMS levels
        vocal_rms = np.sqrt(np.mean(vocals**2))
        instrumental_rms = np.sqrt(np.mean(instrumentals**2))
        
        # Calculate balance ratio
        rms_balance_ratio = vocal_rms / (instrumental_rms + 1e-10)
        
        # Calculate spectral centroids
        vocal_centroid = librosa.feature.spectral_centroid(y=vocals, sr=sr).mean()
        instrumental_centroid = librosa.feature.spectral_centroid(y=instrumentals, sr=sr).mean()
        
        # Calculate spectral balance
        spectral_balance = abs(vocal_centroid - instrumental_centroid) / (vocal_centroid + 1e-10)
        
        return BalanceMetrics(
            vocal_rms=float(vocal_rms),
            instrumental_rms=float(instrumental_rms),
            rms_balance_ratio=float(rms_balance_ratio),
            spectral_centroid_vocals=float(vocal_centroid),
            spectral_centroid_instrumentals=float(instrumental_centroid),
            spectral_balance=float(spectral_balance)
        )
    
    def calculate_si_sdr(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        """Calculate Scale-Invariant Signal-to-Distortion Ratio."""
        try:
            # Ensure same length
            min_len = min(len(reference), len(estimate))
            reference = reference[:min_len]
            estimate = estimate[:min_len]
            
            # Calculate optimal scaling factor
            alpha = np.dot(reference, estimate) / (np.dot(estimate, estimate) + 1e-10)
            
            # Calculate SI-SDR
            scaled_estimate = alpha * estimate
            error = reference - scaled_estimate
            
            signal_power = np.dot(reference, reference)
            error_power = np.dot(error, error)
            
            si_sdr = 10 * np.log10(signal_power / (error_power + 1e-10))
            return float(si_sdr)
            
        except Exception:
            return 0.0
    
    def calculate_quality_metrics(self, vocals: np.ndarray, instrumentals: np.ndarray, 
                                mixed: np.ndarray, sr: int) -> QualityMetrics:
        """Calculate overall quality metrics."""
        # Calculate SI-SDR (using vocals as reference for vocal intelligibility)
        si_sdr = self.calculate_si_sdr(vocals, mixed)
        
        # Calculate intelligibility proxy (spectral centroid preservation)
        vocal_centroid = librosa.feature.spectral_centroid(y=vocals, sr=sr).mean()
        mixed_centroid = librosa.feature.spectral_centroid(y=mixed, sr=sr).mean()
        intelligibility = max(0, 1 - abs(vocal_centroid - mixed_centroid) / (vocal_centroid + 1e-10))
        
        # Calculate musical coherence (simplified)
        musical_coherence = 0.8  # Placeholder - would need more complex analysis
        
        # Calculate technical quality
        clipping = np.sum(np.abs(mixed) > CLIPPING_THRESHOLD) / len(mixed)
        technical_quality = max(0, 1 - clipping * 10)  # Penalize clipping heavily
        
        # Overall perceptual quality
        perceptual_quality = (intelligibility * 0.4 + technical_quality * 0.4 + 
                            musical_coherence * 0.2)
        
        return QualityMetrics(
            si_sdr=float(si_sdr),
            perceptual_quality=float(perceptual_quality),
            intelligibility_score=float(intelligibility),
            musical_coherence=float(musical_coherence),
            technical_quality=float(technical_quality)
        )
    
    def evaluate_remix(self, vocals_path: Path, instrumentals_path: Path, 
                      remix_path: Path, sr: int = SAMPLE_RATE) -> Dict:
        """Comprehensive remix evaluation."""
        print(f"🔍 Evaluating remix quality: {remix_path.name}")
        
        # Load audio files
        vocals, _ = librosa.load(str(vocals_path), sr=sr, mono=True)
        instrumentals, _ = librosa.load(str(instrumentals_path), sr=sr, mono=True)
        remix, _ = librosa.load(str(remix_path), sr=sr, mono=True)
        
        # Calculate all metrics
        tempo_metrics = self.calculate_tempo_metrics(vocals, instrumentals, sr)
        key_metrics = self.calculate_key_metrics(vocals, instrumentals, sr)
        loudness_metrics = self.calculate_loudness_metrics(remix, sr)
        balance_metrics = self.calculate_balance_metrics(vocals, instrumentals, sr)
        quality_metrics = self.calculate_quality_metrics(vocals, instrumentals, remix, sr)
        
        # Compile results
        results = {
            'timestamp': time.time(),
            'remix_file': str(remix_path),
            'tempo': {
                'base_tempo': tempo_metrics.base_tempo,
                'match_tempo': tempo_metrics.match_tempo,
                'tempo_error_bpm': tempo_metrics.tempo_error_bpm,
                'tempo_ratio': tempo_metrics.tempo_ratio,
                'beat_alignment_ms': tempo_metrics.beat_alignment_ms
            },
            'key': {
                'base_key': key_metrics.base_key,
                'match_key': key_metrics.match_key,
                'key_compatibility': key_metrics.key_compatibility,
                'camelot_distance': key_metrics.camelot_distance,
                'pitch_shift_applied': key_metrics.pitch_shift_applied
            },
            'loudness': {
                'lufs_integrated': loudness_metrics.lufs_integrated,
                'lufs_target': loudness_metrics.lufs_target,
                'lufs_error': loudness_metrics.lufs_error,
                'true_peak_db': loudness_metrics.true_peak_db,
                'clipping_percentage': loudness_metrics.clipping_percentage,
                'dynamic_range_db': loudness_metrics.dynamic_range_db
            },
            'balance': {
                'vocal_rms': balance_metrics.vocal_rms,
                'instrumental_rms': balance_metrics.instrumental_rms,
                'rms_balance_ratio': balance_metrics.rms_balance_ratio,
                'spectral_centroid_vocals': balance_metrics.spectral_centroid_vocals,
                'spectral_centroid_instrumentals': balance_metrics.spectral_centroid_instrumentals,
                'spectral_balance': balance_metrics.spectral_balance
            },
            'quality': {
                'si_sdr': quality_metrics.si_sdr,
                'perceptual_quality': quality_metrics.perceptual_quality,
                'intelligibility_score': quality_metrics.intelligibility_score,
                'musical_coherence': quality_metrics.musical_coherence,
                'technical_quality': quality_metrics.technical_quality
            }
        }
        
        return results
    
    def save_metrics_report(self, results: Dict, output_path: Path) -> None:
        """Save metrics report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📊 Metrics report saved to: {output_path}")
    
    def print_metrics_summary(self, results: Dict) -> None:
        """Print a human-readable metrics summary."""
        print("\n" + "="*60)
        print("🎵 REMIX QUALITY METRICS SUMMARY")
        print("="*60)
        
        # Tempo metrics
        tempo = results['tempo']
        print(f"\n⏱️  TEMPO ANALYSIS")
        print(f"   Base tempo: {tempo['base_tempo']:.1f} BPM")
        print(f"   Match tempo: {tempo['match_tempo']:.1f} BPM")
        print(f"   Tempo error: {tempo['tempo_error_bpm']:.1f} BPM")
        print(f"   Beat alignment: {tempo['beat_alignment_ms']:.1f} ms")
        
        # Key metrics
        key = results['key']
        print(f"\n🎹 KEY COMPATIBILITY")
        print(f"   Base key: {key['base_key']}")
        print(f"   Match key: {key['match_key']}")
        print(f"   Compatibility: {key['key_compatibility']:.2f}")
        print(f"   Camelot distance: {key['camelot_distance']} steps")
        
        # Loudness metrics
        loudness = results['loudness']
        print(f"\n🔊 LOUDNESS ANALYSIS")
        print(f"   LUFS: {loudness['lufs_integrated']:.1f} (target: {loudness['lufs_target']:.1f})")
        print(f"   LUFS error: {loudness['lufs_error']:.1f}")
        print(f"   True peak: {loudness['true_peak_db']:.1f} dB")
        print(f"   Clipping: {loudness['clipping_percentage']:.2f}%")
        
        # Quality metrics
        quality = results['quality']
        print(f"\n⭐ OVERALL QUALITY")
        print(f"   Perceptual quality: {quality['perceptual_quality']:.2f}")
        print(f"   Intelligibility: {quality['intelligibility_score']:.2f}")
        print(f"   Technical quality: {quality['technical_quality']:.2f}")
        print(f"   SI-SDR: {quality['si_sdr']:.1f} dB")
        
        print("\n" + "="*60)
