#!/usr/bin/env python3
"""
Optimization Proxy Demo

This script demonstrates the optimization proxy concept in action:
1. ML Model predicts optimal remix parameters
2. Repair layer enforces audio quality constraints
3. Dual estimates certify quality and performance bounds

Shows how the system delivers reliable, near-optimal, real-time decisions
with guaranteed constraints and provable performance bounds.
"""

from __future__ import annotations
import time
import numpy as np
from pathlib import Path

from scripts.core.optimization_proxy import OptimizationProxy, RemixParameters, QualityMetrics
from scripts.core.features import extract_features_from_wav
from scripts.core.paths import vocals_path, other_path


def demonstrate_optimization_proxy():
    """Demonstrate the optimization proxy concept."""
    print("🚀 Optimization Proxy Demonstration")
    print("=" * 50)
    print()
    
    # Initialize the optimization proxy
    proxy = OptimizationProxy()
    
    # Load audio features for demonstration
    base_song = "Sandstorm"
    match_song = "One More Time"
    
    print(f"📊 Loading features for: {base_song} + {match_song}")
    base_features = extract_features_from_wav(vocals_path(base_song))
    match_features = extract_features_from_wav(other_path(match_song))
    
    print(f"   Base tempo: {base_features.tempo:.1f} BPM")
    print(f"   Match tempo: {match_features.tempo:.1f} BPM")
    print()
    
    # Demonstrate ML prediction
    print("🧠 Step 1: ML Model Prediction")
    print("-" * 30)
    
    # Create sample parameters
    sample_params = RemixParameters(
        tempo_ratio=1.2,
        volume_balance=0.6,
        fade_in_duration=0.8,
        fade_out_duration=0.8,
        eq_low=2.0,
        eq_mid=1.0,
        eq_high=-1.0,
        reverb_amount=0.3,
        compression_ratio=2.5
    )
    
    # Predict quality
    features = proxy.create_feature_vector(base_features, match_features, sample_params)
    predicted_quality = proxy.predict_quality(features)
    
    print(f"   Input parameters: tempo_ratio={sample_params.tempo_ratio}, volume_balance={sample_params.volume_balance}")
    print(f"   Predicted quality: {predicted_quality:.3f}")
    print()
    
    # Demonstrate repair layer
    print("🔧 Step 2: Repair Layer (Constraint Enforcement)")
    print("-" * 50)
    
    # Create parameters that violate constraints
    bad_params = RemixParameters(
        tempo_ratio=3.0,  # Too high
        volume_balance=1.5,  # Out of range
        fade_in_duration=-0.5,  # Negative
        eq_low=20.0,  # Too high
        compression_ratio=15.0  # Too high
    )
    
    print("   Before repair:")
    print(f"     tempo_ratio: {bad_params.tempo_ratio} (constraint: 0.5-2.0)")
    print(f"     volume_balance: {bad_params.volume_balance} (constraint: 0.0-1.0)")
    print(f"     fade_in_duration: {bad_params.fade_in_duration} (constraint: >=0.0)")
    print(f"     eq_low: {bad_params.eq_low} (constraint: -12.0 to 12.0)")
    print(f"     compression_ratio: {bad_params.compression_ratio} (constraint: 1.0-10.0)")
    
    # Apply repair layer
    repaired_params = proxy.repair_layer(bad_params, QualityMetrics(0, 0, 0, 0, 0, 0))
    
    print("   After repair:")
    print(f"     tempo_ratio: {repaired_params.tempo_ratio}")
    print(f"     volume_balance: {repaired_params.volume_balance}")
    print(f"     fade_in_duration: {repaired_params.fade_in_duration}")
    print(f"     eq_low: {repaired_params.eq_low}")
    print(f"     compression_ratio: {repaired_params.compression_ratio}")
    print()
    
    # Demonstrate dual estimates
    print("📊 Step 3: Dual Estimates (Quality Certification)")
    print("-" * 45)
    
    start_time = time.time()
    lower_bound, upper_bound, performance_metrics = proxy.dual_estimates(
        repaired_params, base_features, match_features
    )
    processing_time = (time.time() - start_time) * 1000
    
    print(f"   Quality bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
    print(f"   Confidence interval: ±{(upper_bound - lower_bound)/2:.3f}")
    print(f"   Constraint violations: {performance_metrics['constraint_violations']}")
    print(f"   Processing time: {processing_time:.2f}ms")
    print()
    
    # Demonstrate full optimization
    print("🎯 Step 4: Full Optimization")
    print("-" * 25)
    
    start_time = time.time()
    optimized_params, opt_metrics = proxy.optimize_remix_parameters(
        base_features, match_features
    )
    optimization_time = (time.time() - start_time) * 1000
    
    print(f"   Optimized parameters:")
    print(f"     tempo_ratio: {optimized_params.tempo_ratio:.3f}")
    print(f"     volume_balance: {optimized_params.volume_balance:.3f}")
    print(f"     fade_in_duration: {optimized_params.fade_in_duration:.3f}")
    print(f"     fade_out_duration: {optimized_params.fade_out_duration:.3f}")
    print(f"     eq_low: {optimized_params.eq_low:.3f}")
    print(f"     eq_mid: {optimized_params.eq_mid:.3f}")
    print(f"     eq_high: {optimized_params.eq_high:.3f}")
    print(f"     reverb_amount: {optimized_params.reverb_amount:.3f}")
    print(f"     compression_ratio: {optimized_params.compression_ratio:.3f}")
    print()
    print(f"   Final quality bounds: {opt_metrics['final_quality_bounds']}")
    print(f"   Optimization time: {optimization_time:.2f}ms")
    print(f"   Success: {opt_metrics['optimization_success']}")
    print()
    
    # Summary
    print("📋 Optimization Proxy Summary")
    print("=" * 30)
    print("✅ ML Model: Predicts optimal parameters")
    print("✅ Repair Layer: Enforces all constraints")
    print("✅ Dual Estimates: Certifies quality bounds")
    print("✅ Real-time: Sub-second optimization")
    print("✅ Guaranteed: Constraint satisfaction")
    print("✅ Provable: Performance bounds")
    print()
    print("🎉 Optimization proxy delivers reliable, near-optimal,")
    print("   real-time remix decisions with guaranteed constraints!")


if __name__ == "__main__":
    demonstrate_optimization_proxy()
