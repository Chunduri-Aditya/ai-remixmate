#!/usr/bin/env python3
"""
AI RemixMate - Professional Remix Generator

This script provides the main CLI interface for creating high-quality remixes:
- remix --base "SongA" --match "SongB" --optimize --report out/report.json
- recommend --base "SongA" --top 5 --filter key_compatible
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict
import time


from scripts.core.database import RemixMateDatabase
from scripts.core.musical_analysis import MusicalAnalyzer
from scripts.core.real_optimizer import RealOptimizer, MixParameters
from scripts.core.pro_audio_chain import ProfessionalAudioChain
from scripts.core.metrics import AudioMetrics
from scripts.core.features import extract_features_from_wav
from scripts.core.paths import vocals_path, other_path, OUTPUT_DIR, ensure_directories
import librosa
import numpy as np


def remix_command(args):
    """Create a professional remix with optimization."""
    print("🎵 AI RemixMate - Professional Remix Generator")
    print("=" * 60)
    
    # Initialize components
    db = RemixMateDatabase()
    musical_analyzer = MusicalAnalyzer()
    optimizer = RealOptimizer()
    audio_chain = ProfessionalAudioChain()
    metrics = AudioMetrics()
    
    # Ensure output directory exists
    ensure_directories()
    
    # Load audio files
    print(f"🎧 Loading audio files...")
    vocals_file = vocals_path(args.base)
    instrumentals_file = other_path(args.match)
    
    if not vocals_file.exists() or not instrumentals_file.exists():
        print(f"❌ Required audio files not found:")
        print(f"   Vocals: {vocals_file}")
        print(f"   Instrumentals: {instrumentals_file}")
        return 1
    
    # Load audio
    vocals, sr = librosa.load(str(vocals_file), sr=44100, mono=True)
    instrumentals, sr = librosa.load(str(instrumentals_file), sr=44100, mono=True)
    
    print(f"   Vocals: {len(vocals)/sr:.1f}s")
    print(f"   Instrumentals: {len(instrumentals)/sr:.1f}s")
    
    # Musical analysis and correction
    print(f"\n🎵 Analyzing musical compatibility...")
    vocals_corrected, instrumentals_corrected, correction_info = musical_analyzer.apply_musical_corrections(
        vocals, instrumentals
    )
    
    print(f"   Key compatibility: {correction_info['key_compatibility']:.2f}")
    print(f"   Tempo ratio: {correction_info['tempo_ratio']:.3f}")
    print(f"   Corrections applied: {', '.join(correction_info['corrections_applied'])}")
    
    # Optimization
    if args.optimize:
        print(f"\n🧠 Optimizing mix parameters...")
        optimization_result = optimizer.optimize(
            vocals_corrected, instrumentals_corrected, 
            max_iterations=args.optimize_iterations,
            seed=args.seed
        )
        
        print(f"   Best score: {optimization_result.best_score:.4f}")
        print(f"   Optimization time: {optimization_result.optimization_time:.2f}s")
        
        # Apply optimized parameters
        mixed = optimizer.apply_mix_parameters(
            vocals_corrected, instrumentals_corrected, 
            optimization_result.best_parameters
        )
    else:
        print(f"\n🎛️ Using default parameters...")
        default_params = MixParameters()
        mixed = optimizer.apply_mix_parameters(
            vocals_corrected, instrumentals_corrected, default_params
        )
    
    # Professional audio processing
    print(f"\n🎛️ Applying professional audio processing...")
    final_mix, processing_info = audio_chain.process_remix(vocals_corrected, instrumentals_corrected)
    
    # Generate output filename
    if args.output:
        output_name = args.output
    else:
        timestamp = int(time.time())
        output_name = f"remix_{args.base}_{args.match}_{timestamp}.wav"
    
    output_path = OUTPUT_DIR / output_name
    
    # Export high-quality audio
    print(f"\n💾 Exporting high-quality audio...")
    audio_chain.export_high_quality(final_mix, output_path)
    
    # Quality evaluation
    print(f"\n📊 Evaluating remix quality...")
    quality_results = metrics.evaluate_remix(
        vocals_file, instrumentals_file, output_path
    )
    
    # Generate report
    report_data = {
        'remix_info': {
            'base_song': args.base,
            'match_song': args.match,
            'output_file': str(output_path),
            'created_at': time.time()
        },
        'musical_analysis': correction_info,
        'optimization': {
            'enabled': args.optimize,
            'best_score': optimization_result.best_score if args.optimize else 0.0,
            'optimization_time': optimization_result.optimization_time if args.optimize else 0.0,
            'iterations': optimization_result.iterations if args.optimize else 0
        } if args.optimize else {'enabled': False},
        'audio_processing': processing_info,
        'quality_metrics': quality_results
    }
    
    # Save report
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.save_metrics_report(report_data, report_path)
        print(f"📋 Report saved to: {report_path}")
    
    # Print summary
    print(f"\n✅ Remix completed successfully!")
    print(f"   Output: {output_path}")
    print(f"   Quality score: {quality_results['quality']['perceptual_quality']:.3f}")
    print(f"   LUFS: {quality_results['loudness']['lufs_integrated']:.1f}")
    print(f"   Clipping: {quality_results['loudness']['clipping_percentage']:.2f}%")
    
    return 0


def recommend_command(args):
    """Recommend compatible songs for remixing."""
    print("🔍 AI RemixMate - Song Recommendation")
    print("=" * 50)
    
    # Initialize components
    db = RemixMateDatabase()
    musical_analyzer = MusicalAnalyzer()
    
    # Get base song features
    base_song = db.get_song(args.base)
    if not base_song:
        print(f"❌ Song '{args.base}' not found in database")
        print(f"   Add it first with: python scripts/song_database.py --wav path/to/song.wav --name '{args.base}'")
        return 1
    
    print(f"🎵 Base song: {args.base}")
    print(f"   BPM: {base_song.bpm:.1f}")
    print(f"   Key: {base_song.key}")
    print(f"   Camelot: {base_song.camelot_number}")
    
    # Get recommendations
    recommendations = []
    
    if args.filter == 'key_compatible':
        print(f"\n🎹 Finding key-compatible songs...")
        compatible = db.find_songs_by_camelot_compatibility(
            base_song.camelot_number, max_distance=2
        )
        recommendations = [(name, 1.0 - distance/6.0) for name, distance in compatible]
    
    elif args.filter == 'tempo_compatible':
        print(f"\n⏱️ Finding tempo-compatible songs...")
        tempo_range = 10  # ±10 BPM
        compatible = db.find_songs_by_tempo_range(
            base_song.bpm - tempo_range, base_song.bpm + tempo_range
        )
        recommendations = [(name, 0.8) for name in compatible]
    
    else:  # similarity
        print(f"\n🔍 Finding similar songs...")
        base_features = db.get_song_features(args.base)
        if base_features:
            recommendations = db.find_similar_songs(base_features, top_k=args.top)
    
    # Filter out base song
    recommendations = [(name, score) for name, score in recommendations if name != args.base]
    
    # Display results
    print(f"\n🎯 Top {min(args.top, len(recommendations))} recommendations:")
    print("-" * 60)
    
    for i, (name, score) in enumerate(recommendations[:args.top], 1):
        song = db.get_song(name)
        if song:
            print(f"{i:2d}. {name:<30} | Score: {score:.3f} | BPM: {song.bpm:5.1f} | Key: {song.key}")
    
    if not recommendations:
        print("   No compatible songs found. Try adding more songs to the database.")
    
    return 0


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="AI RemixMate - Professional Remix Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create optimized remix
  python scripts/remix.py --base "SongA" --match "SongB" --optimize --report report.json
  
  # Create basic remix
  python scripts/remix.py --base "SongA" --match "SongB" --output "my_remix.wav"
  
  # Find key-compatible songs
  python scripts/remix.py recommend --base "SongA" --top 5 --filter key_compatible
  
  # Find similar songs
  python scripts/remix.py recommend --base "SongA" --top 10 --filter similarity
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Remix command
    remix_parser = subparsers.add_parser('remix', help='Create a remix')
    remix_parser.add_argument('--base', required=True, help='Base song name (vocals source)')
    remix_parser.add_argument('--match', required=True, help='Match song name (instrumentals source)')
    remix_parser.add_argument('--output', help='Output filename (auto-generated if not specified)')
    remix_parser.add_argument('--optimize', action='store_true', help='Use ML optimization')
    remix_parser.add_argument('--optimize-iterations', type=int, default=100, help='Optimization iterations')
    remix_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    remix_parser.add_argument('--report', help='Save quality report to JSON file')
    
    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Recommend compatible songs')
    recommend_parser.add_argument('--base', required=True, help='Base song name')
    recommend_parser.add_argument('--top', type=int, default=5, help='Number of recommendations')
    recommend_parser.add_argument('--filter', choices=['similarity', 'key_compatible', 'tempo_compatible'], 
                                default='similarity', help='Filtering method')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'remix':
            return remix_command(args)
        elif args.command == 'recommend':
            return recommend_command(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
