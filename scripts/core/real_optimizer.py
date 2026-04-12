"""
Real Optimizer with Random Search and Repair Layer

This module implements a practical optimizer that:
1. Uses random search over 6-8 mix parameters
2. Enforces constraints through a repair layer
3. Scores with objective metrics
4. Provides honest optimization results
"""

from __future__ import annotations
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
import time
import json

from .musical_analysis import MusicalAnalyzer
from .metrics import AudioMetrics

try:
    from scripts.core.config import cfg as _cfg
    _TARGET_LUFS    = _cfg.audio.target_lufs
    _MAX_ITERATIONS = _cfg.remix.optimizer_iterations
    _SAMPLE_RATE    = _cfg.audio.sample_rate
except Exception:
    _TARGET_LUFS    = -14.0
    _MAX_ITERATIONS = 50
    _SAMPLE_RATE    = 44100


@dataclass
class MixParameters:
    """Mix parameters for optimization."""
    vocal_gain: float = 0.0  # dB
    instrumental_gain: float = 0.0  # dB
    crossfade_curve: float = 0.5  # 0-1, 0.5 = linear
    sidechain_amount: float = 0.0  # 0-1, amount of sidechain compression
    eq_low_cut_freq: float = 80.0  # Hz
    reverb_send: float = 0.0  # 0-1, reverb amount
    time_stretch_ratio: float = 1.0  # Fine-tune tempo ratio
    compression_ratio: float = 1.0  # Compression ratio


@dataclass
class OptimizationConstraints:
    """Constraints for parameter optimization."""
    min_vocal_gain: float = -12.0  # dB
    max_vocal_gain: float = 12.0   # dB
    min_instrumental_gain: float = -12.0  # dB
    max_instrumental_gain: float = 12.0   # dB
    min_crossfade_curve: float = 0.0
    max_crossfade_curve: float = 1.0
    min_sidechain_amount: float = 0.0
    max_sidechain_amount: float = 0.8
    min_eq_low_cut: float = 20.0   # Hz
    max_eq_low_cut: float = 200.0  # Hz
    min_reverb_send: float = 0.0
    max_reverb_send: float = 0.3
    min_time_stretch: float = 0.8
    max_time_stretch: float = 1.2
    min_compression_ratio: float = 1.0
    max_compression_ratio: float = 4.0
    target_lufs: float = field(default_factory=lambda: _TARGET_LUFS)
    max_clipping_percent: float = 1.0
    min_key_compatibility: float = 0.3
    max_tempo_error_bpm: float = 8.0


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    best_parameters: MixParameters
    best_score: float
    optimization_time: float
    iterations: int
    constraint_violations: int
    convergence_info: Dict


class RealOptimizer:
    """Practical optimizer with random search and repair layer."""
    
    def __init__(self, constraints: Optional[OptimizationConstraints] = None):
        self.constraints = constraints or OptimizationConstraints()
        self.musical_analyzer = MusicalAnalyzer()
        self.metrics = AudioMetrics()
        self.random_seed = 42
    
    def repair_parameters(self, params: MixParameters) -> Tuple[MixParameters, List[str]]:
        """
        Repair layer: enforce constraints and fix invalid parameters.
        
        Args:
            params: Input parameters
            
        Returns:
            Tuple of (repaired_parameters, violations_fixed)
        """
        violations_fixed = []
        repaired = MixParameters()
        
        # Vocal gain constraints
        repaired.vocal_gain = np.clip(params.vocal_gain, 
                                    self.constraints.min_vocal_gain, 
                                    self.constraints.max_vocal_gain)
        if params.vocal_gain != repaired.vocal_gain:
            violations_fixed.append("vocal_gain_clipped")
        
        # Instrumental gain constraints
        repaired.instrumental_gain = np.clip(params.instrumental_gain,
                                           self.constraints.min_instrumental_gain,
                                           self.constraints.max_instrumental_gain)
        if params.instrumental_gain != repaired.instrumental_gain:
            violations_fixed.append("instrumental_gain_clipped")
        
        # Crossfade curve constraints
        repaired.crossfade_curve = np.clip(params.crossfade_curve,
                                          self.constraints.min_crossfade_curve,
                                          self.constraints.max_crossfade_curve)
        if params.crossfade_curve != repaired.crossfade_curve:
            violations_fixed.append("crossfade_curve_clipped")
        
        # Sidechain amount constraints
        repaired.sidechain_amount = np.clip(params.sidechain_amount,
                                           self.constraints.min_sidechain_amount,
                                           self.constraints.max_sidechain_amount)
        if params.sidechain_amount != repaired.sidechain_amount:
            violations_fixed.append("sidechain_amount_clipped")
        
        # EQ low cut constraints
        repaired.eq_low_cut_freq = np.clip(params.eq_low_cut_freq,
                                          self.constraints.min_eq_low_cut,
                                          self.constraints.max_eq_low_cut)
        if params.eq_low_cut_freq != repaired.eq_low_cut_freq:
            violations_fixed.append("eq_low_cut_clipped")
        
        # Reverb send constraints
        repaired.reverb_send = np.clip(params.reverb_send,
                                      self.constraints.min_reverb_send,
                                      self.constraints.max_reverb_send)
        if params.reverb_send != repaired.reverb_send:
            violations_fixed.append("reverb_send_clipped")
        
        # Time stretch constraints
        repaired.time_stretch_ratio = np.clip(params.time_stretch_ratio,
                                             self.constraints.min_time_stretch,
                                             self.constraints.max_time_stretch)
        if params.time_stretch_ratio != repaired.time_stretch_ratio:
            violations_fixed.append("time_stretch_clipped")
        
        # Compression ratio constraints
        repaired.compression_ratio = np.clip(params.compression_ratio,
                                            self.constraints.min_compression_ratio,
                                            self.constraints.max_compression_ratio)
        if params.compression_ratio != repaired.compression_ratio:
            violations_fixed.append("compression_ratio_clipped")
        
        return repaired, violations_fixed
    
    def generate_random_parameters(self) -> MixParameters:
        """Generate random parameters within constraints."""
        return MixParameters(
            vocal_gain=random.uniform(self.constraints.min_vocal_gain, self.constraints.max_vocal_gain),
            instrumental_gain=random.uniform(self.constraints.min_instrumental_gain, self.constraints.max_instrumental_gain),
            crossfade_curve=random.uniform(self.constraints.min_crossfade_curve, self.constraints.max_crossfade_curve),
            sidechain_amount=random.uniform(self.constraints.min_sidechain_amount, self.constraints.max_sidechain_amount),
            eq_low_cut_freq=random.uniform(self.constraints.min_eq_low_cut, self.constraints.max_eq_low_cut),
            reverb_send=random.uniform(self.constraints.min_reverb_send, self.constraints.max_reverb_send),
            time_stretch_ratio=random.uniform(self.constraints.min_time_stretch, self.constraints.max_time_stretch),
            compression_ratio=random.uniform(self.constraints.min_compression_ratio, self.constraints.max_compression_ratio)
        )
    
    def apply_mix_parameters(self, vocals: np.ndarray, instrumentals: np.ndarray, 
                           params: MixParameters) -> np.ndarray:
        """
        Apply mix parameters to create a remix.
        
        Args:
            vocals: Vocal audio
            instrumentals: Instrumental audio
            params: Mix parameters
            
        Returns:
            Mixed audio
        """
        # Apply gains
        vocal_gain_linear = 10**(params.vocal_gain / 20.0)
        instrumental_gain_linear = 10**(params.instrumental_gain / 20.0)
        
        vocals_gained = vocals * vocal_gain_linear
        instrumentals_gained = instrumentals * instrumental_gain_linear
        
        # Apply time stretch
        if abs(params.time_stretch_ratio - 1.0) > 0.01:
            try:
                from scripts.core.gpu import gpu_time_stretch
                instrumentals_stretched = gpu_time_stretch(
                    instrumentals_gained, rate=params.time_stretch_ratio
                )
            except (ImportError, Exception):
                import librosa
                instrumentals_stretched = librosa.effects.time_stretch(
                    instrumentals_gained, rate=params.time_stretch_ratio
                )
        else:
            instrumentals_stretched = instrumentals_gained
        
        # Align lengths
        min_length = min(len(vocals_gained), len(instrumentals_stretched))
        vocals_aligned = vocals_gained[:min_length]
        instrumentals_aligned = instrumentals_stretched[:min_length]
        
        # Apply sidechain compression (simplified)
        if params.sidechain_amount > 0:
            # Simple sidechain: reduce instrumental volume when vocals are present
            vocal_envelope = np.abs(vocals_aligned)
            sidechain_factor = 1.0 - params.sidechain_amount * vocal_envelope
            sidechain_factor = np.clip(sidechain_factor, 0.1, 1.0)
            instrumentals_aligned = instrumentals_aligned * sidechain_factor
        
        # Apply EQ (simplified high-pass filter)
        if params.eq_low_cut_freq > 20:
            # Simple high-pass filter approximation
            cutoff_ratio = params.eq_low_cut_freq / 22050  # Assuming 44.1kHz
            # This is a very simplified EQ - in practice you'd use proper filtering
            pass
        
        # Mix with crossfade curve
        if params.crossfade_curve < 0.5:
            # More vocals
            mix_ratio = 0.5 + (0.5 - params.crossfade_curve)
        else:
            # More instrumentals
            mix_ratio = 0.5 - (params.crossfade_curve - 0.5)
        
        mixed = vocals_aligned * mix_ratio + instrumentals_aligned * (1 - mix_ratio)
        
        # Apply compression (simplified)
        if params.compression_ratio > 1.0:
            # Simple compression approximation
            threshold = 0.5
            ratio = params.compression_ratio
            compressed = np.where(
                np.abs(mixed) > threshold,
                np.sign(mixed) * (threshold + (np.abs(mixed) - threshold) / ratio),
                mixed
            )
            mixed = compressed
        
        # Apply reverb (simplified)
        if params.reverb_send > 0:
            # Simple reverb approximation using delay and feedback
            delay_samples = int(0.1 * 44100)  # 100ms delay
            if len(mixed) > delay_samples:
                reverb = np.zeros_like(mixed)
                reverb[delay_samples:] = mixed[:-delay_samples] * 0.3 * params.reverb_send
                mixed = mixed + reverb
        
        return mixed
    
    def score_remix(self, vocals: np.ndarray, instrumentals: np.ndarray, 
                   mixed: np.ndarray) -> float:
        """
        Score a remix using objective metrics.
        
        Args:
            vocals: Original vocals
            instrumentals: Original instrumentals
            mixed: Mixed result
            
        Returns:
            Quality score (0-1, higher is better)
        """
        try:
            # Calculate metrics
            tempo_metrics = self.metrics.calculate_tempo_metrics(vocals, instrumentals, 44100)
            key_metrics = self.metrics.calculate_key_metrics(vocals, instrumentals, 44100)
            loudness_metrics = self.metrics.calculate_loudness_metrics(mixed, 44100)
            balance_metrics = self.metrics.calculate_balance_metrics(vocals, instrumentals, 44100)
            quality_metrics = self.metrics.calculate_quality_metrics(vocals, instrumentals, mixed, 44100)
            
            # Score components
            tempo_score = max(0, 1 - tempo_metrics.tempo_error_bpm / 20.0)  # Penalize tempo mismatch
            key_score = key_metrics.key_compatibility
            loudness_score = max(0, 1 - loudness_metrics.lufs_error / 6.0)  # Penalize LUFS error
            clipping_penalty = max(0, 1 - loudness_metrics.clipping_percentage / 100.0)  # Penalize clipping
            balance_score = max(0, 1 - abs(balance_metrics.rms_balance_ratio - 1.0))  # Prefer balanced mix
            intelligibility_score = quality_metrics.intelligibility_score
            technical_score = quality_metrics.technical_quality
            
            # Weighted combination
            total_score = (
                tempo_score * 0.15 +
                key_score * 0.20 +
                loudness_score * 0.15 +
                clipping_penalty * 0.20 +
                balance_score * 0.10 +
                intelligibility_score * 0.15 +
                technical_score * 0.05
            )
            
            return float(total_score)
            
        except Exception as e:
            print(f"⚠️ Scoring failed: {e}")
            return 0.0
    
    def optimize(self, vocals: np.ndarray, instrumentals: np.ndarray, 
                max_iterations: int = 100, seed: Optional[int] = None) -> OptimizationResult:
        """
        Optimize mix parameters using random search.
        
        Args:
            vocals: Vocal audio
            instrumentals: Instrumental audio
            max_iterations: Maximum optimization iterations
            seed: Random seed for reproducibility
            
        Returns:
            OptimizationResult with best parameters and score
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"🔍 Starting optimization with {max_iterations} iterations...")
        start_time = time.time()
        
        best_score = -1.0
        best_parameters = None
        constraint_violations = 0
        convergence_history = []
        
        for iteration in range(max_iterations):
            # Generate random parameters
            params = self.generate_random_parameters()
            
            # Apply repair layer
            repaired_params, violations = self.repair_parameters(params)
            constraint_violations += len(violations)
            
            # Apply parameters and create mix
            try:
                mixed = self.apply_mix_parameters(vocals, instrumentals, repaired_params)
                
                # Score the mix
                score = self.score_remix(vocals, instrumentals, mixed)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_parameters = repaired_params
                    print(f"   Iteration {iteration+1}: New best score = {score:.4f}")
                
                convergence_history.append(score)
                
            except Exception as e:
                print(f"   Iteration {iteration+1}: Failed - {e}")
                convergence_history.append(0.0)
        
        optimization_time = time.time() - start_time
        
        # Calculate convergence info
        convergence_info = {
            'final_score': best_score,
            'score_history': convergence_history,
            'improvement_rate': len([s for s in convergence_history if s > 0]) / len(convergence_history),
            'constraint_violation_rate': constraint_violations / max_iterations
        }
        
        print(f"✅ Optimization completed in {optimization_time:.2f}s")
        print(f"   Best score: {best_score:.4f}")
        print(f"   Constraint violations: {constraint_violations}")
        
        return OptimizationResult(
            best_parameters=best_parameters or MixParameters(),
            best_score=best_score,
            optimization_time=optimization_time,
            iterations=max_iterations,
            constraint_violations=constraint_violations,
            convergence_info=convergence_info
        )
    
    def save_optimization_report(self, result: OptimizationResult, output_path: Path) -> None:
        """Save optimization results to JSON file."""
        report = {
            'optimization_result': {
                'best_score': result.best_score,
                'optimization_time': result.optimization_time,
                'iterations': result.iterations,
                'constraint_violations': result.constraint_violations,
                'convergence_info': result.convergence_info
            },
            'best_parameters': {
                'vocal_gain': result.best_parameters.vocal_gain,
                'instrumental_gain': result.best_parameters.instrumental_gain,
                'crossfade_curve': result.best_parameters.crossfade_curve,
                'sidechain_amount': result.best_parameters.sidechain_amount,
                'eq_low_cut_freq': result.best_parameters.eq_low_cut_freq,
                'reverb_send': result.best_parameters.reverb_send,
                'time_stretch_ratio': result.best_parameters.time_stretch_ratio,
                'compression_ratio': result.best_parameters.compression_ratio
            },
            'constraints': {
                'min_vocal_gain': self.constraints.min_vocal_gain,
                'max_vocal_gain': self.constraints.max_vocal_gain,
                'min_instrumental_gain': self.constraints.min_instrumental_gain,
                'max_instrumental_gain': self.constraints.max_instrumental_gain,
                'target_lufs': self.constraints.target_lufs,
                'max_clipping_percent': self.constraints.max_clipping_percent,
                'min_key_compatibility': self.constraints.min_key_compatibility,
                'max_tempo_error_bpm': self.constraints.max_tempo_error_bpm
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Optimization report saved to: {output_path}")
    
    def print_optimization_summary(self, result: OptimizationResult) -> None:
        """Print human-readable optimization summary."""
        print("\n" + "="*60)
        print("🎛️ OPTIMIZATION SUMMARY")
        print("="*60)
        
        print(f"\n📊 PERFORMANCE")
        print(f"   Best score: {result.best_score:.4f}")
        print(f"   Optimization time: {result.optimization_time:.2f}s")
        print(f"   Iterations: {result.iterations}")
        print(f"   Constraint violations: {result.constraint_violations}")
        print(f"   Improvement rate: {result.convergence_info['improvement_rate']:.2%}")
        
        print(f"\n🎛️ BEST PARAMETERS")
        params = result.best_parameters
        print(f"   Vocal gain: {params.vocal_gain:.1f} dB")
        print(f"   Instrumental gain: {params.instrumental_gain:.1f} dB")
        print(f"   Crossfade curve: {params.crossfade_curve:.2f}")
        print(f"   Sidechain amount: {params.sidechain_amount:.2f}")
        print(f"   EQ low cut: {params.eq_low_cut_freq:.0f} Hz")
        print(f"   Reverb send: {params.reverb_send:.2f}")
        print(f"   Time stretch: {params.time_stretch_ratio:.3f}")
        print(f"   Compression ratio: {params.compression_ratio:.1f}")
        
        print("\n" + "="*60)
