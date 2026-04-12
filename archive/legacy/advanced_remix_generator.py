#!/usr/bin/env python3
"""
Advanced Remix Generator with Optimization Proxy

This script uses the optimization proxy concept to create high-quality remixes:
1. ML Model predicts optimal remix parameters
2. Repair layer enforces audio quality constraints  
3. Dual estimates certify quality and performance bounds

Delivers reliable, near-optimal, real-time remix decisions with guaranteed
constraints and provable performance bounds.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

from scripts.core.advanced_remix import AdvancedRemixEngine
from scripts.core.optimization_proxy import OptimizationProxy, RemixParameters, QualityMetrics
from scripts.core.features import extract_features_from_wav
from scripts.core.paths import vocals_path, other_path


def create_advanced_remix(base_song: str, match_song: str, 
                         output_name: Optional[str] = None,
                         auto_optimize: bool = True,
                         quality_report: bool = True) -> Path:
    """
    Create an advanced remix using optimization proxy methodology.
    
    Args:
        base_song: Name of the song to take vocals from
        match_song: Name of the song to take instrumentals from
        output_name: Custom output filename
        auto_optimize: Whether to use ML optimization
        quality_report: Whether to generate quality report
        
    Returns:
        Path to the generated remix
    """
    print("🚀 Advanced Remix Generator with Optimization Proxy")
    print("=" * 60)
    
    # Initialize advanced remix engine
    engine = AdvancedRemixEngine()
    
    # Create the remix
    output_path, metrics = engine.create_high_quality_remix(
        base_song=base_song,
        match_song=match_song,
        output_name=output_name,
        auto_optimize=auto_optimize
    )
    
    # Generate quality report
    if quality_report:
        report = engine.generate_quality_report(output_path, metrics)
        print(report)
        
        # Save report to file
        report_path = output_path.with_suffix('.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"📋 Quality report saved to: {report_path}")
    
    return output_path


def train_optimization_model(training_songs: list[str]):
    """
    Train the optimization proxy model on existing remixes.
    
    Args:
        training_songs: List of song names to use for training
    """
    print("🧠 Training Optimization Proxy Model")
    print("=" * 40)
    
    proxy = OptimizationProxy()
    training_data = []
    
    for song in training_songs:
        try:
            vocals_file = vocals_path(song)
            other_file = other_path(song)
            
            if vocals_file.exists() and other_file.exists():
                print(f"📊 Processing training data: {song}")
                
                # Extract features
                base_features = extract_features_from_wav(vocals_file)
                match_features = extract_features_from_wav(other_file)
                
                # Create sample parameters (in practice, these would come from user feedback)
                params = RemixParameters(
                    tempo_ratio=1.0,
                    volume_balance=0.5,
                    fade_in_duration=0.5,
                    fade_out_duration=0.5
                )
                
                # Create quality metrics (in practice, these would be human-rated)
                quality_metrics = QualityMetrics(
                    rms_energy=0.1,
                    dynamic_range=0.8,
                    spectral_centroid=2000.0,
                    zero_crossing_rate=0.05,
                    harmonic_ratio=0.7,
                    perceptual_quality=0.8  # Simulated quality score
                )
                
                training_data.append((base_features, match_features, params, quality_metrics))
                
        except Exception as e:
            print(f"⚠️ Skipping {song}: {e}")
    
    if training_data:
        proxy.train_model(training_data)
        print(f"✅ Model trained on {len(training_data)} samples")
    else:
        print("❌ No training data available")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Remix Generator with Optimization Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create optimized remix
  python scripts/advanced_remix_generator.py --base "Sandstorm" --match "One More Time"
  
  # Create remix without optimization
  python scripts/advanced_remix_generator.py --base "Sandstorm" --match "One More Time" --no-optimize
  
  # Train the model
  python scripts/advanced_remix_generator.py --train --songs "Sandstorm" "One More Time"
        """
    )
    
    parser.add_argument("--base", help="Base song name (vocals source)")
    parser.add_argument("--match", help="Match song name (instrumentals source)")
    parser.add_argument("--output", help="Custom output filename")
    parser.add_argument("--no-optimize", action="store_true", help="Disable ML optimization")
    parser.add_argument("--no-report", action="store_true", help="Disable quality report")
    parser.add_argument("--train", action="store_true", help="Train the optimization model")
    parser.add_argument("--songs", nargs="+", help="Songs to use for training")
    
    args = parser.parse_args()
    
    if args.train:
        if not args.songs:
            print("❌ Please provide songs for training with --songs")
            return
        train_optimization_model(args.songs)
    else:
        if not args.base or not args.match:
            print("❌ Please provide both --base and --match song names")
            return
        
        try:
            output_path = create_advanced_remix(
                base_song=args.base,
                match_song=args.match,
                output_name=args.output,
                auto_optimize=not args.no_optimize,
                quality_report=not args.no_report
            )
            print(f"\n🎉 Advanced remix created: {output_path}")
            
        except Exception as e:
            print(f"❌ Error creating remix: {e}")


if __name__ == "__main__":
    main()
