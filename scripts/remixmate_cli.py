#!/usr/bin/env python3
"""
AI RemixMate CLI with Bridge Integration

This module provides the main CLI interface for AI RemixMate including:
- remixmate bridge --base "SongA" --match "SongB" --project ~/Music/Logic/AI_RemixMate.logicx
- remixmate recommend --base "SongA" --top 5 --filter key_compatible
"""

from __future__ import annotations
import argparse
import json
import datetime
import sys
from pathlib import Path
from typing import Dict, Any, Optional


from scripts.core.python_bridge import PythonBridge
from scripts.core.database import RemixMateDatabase
from scripts.core.musical_analysis import MusicalAnalyzer
from scripts.core.real_optimizer import RealOptimizer
from scripts.core.pro_audio_chain import ProfessionalAudioChain
from scripts.core.features import extract_features_from_wav
from scripts.core.paths import vocals_path, other_path, OUTPUT_DIR, ensure_directories
import librosa
import numpy as np


def generate_stems_and_params(base_song: str, match_song: str, out_dir: str) -> Dict[str, Any]:
    """
    Generate stems and suggested mix parameters for bridge integration.
    
    Args:
        base_song: Base song name (vocals source)
        match_song: Match song name (instrumentals source)
        out_dir: Output directory
        
    Returns:
        Dictionary with stem paths and parameters
    """
    print(f"🎵 Generating stems and parameters for: {base_song} + {match_song}")
    
    # Get file paths
    vocals_file = vocals_path(base_song)
    instrumentals_file = other_path(match_song)
    
    if not vocals_file.exists() or not instrumentals_file.exists():
        raise FileNotFoundError("Required audio files not found")
    
    # Load audio
    vocals, sr = librosa.load(str(vocals_file), sr=44100, mono=True)
    instrumentals, sr = librosa.load(str(instrumentals_file), sr=44100, mono=True)
    
    # Analyze musical compatibility
    musical_analyzer = MusicalAnalyzer()
    vocals_corrected, instrumentals_corrected, correction_info = musical_analyzer.apply_musical_corrections(
        vocals, instrumentals
    )
    
    # Optimize parameters
    optimizer = RealOptimizer()
    optimization_result = optimizer.optimize(vocals_corrected, instrumentals_corrected, max_iterations=50)
    
    # Extract features for BPM/key detection
    base_features = extract_features_from_wav(vocals_file)
    match_features = extract_features_from_wav(instrumentals_file)
    
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save processed stems
    vocals_output = out_path / "vocals.wav"
    instrumentals_output = out_path / "instrumental.wav"
    
    import soundfile as sf
    sf.write(str(vocals_output), vocals_corrected, sr)
    sf.write(str(instrumentals_output), instrumentals_corrected, sr)
    
    # Build stems dictionary
    stems = {
        "vocals_path": str(vocals_output),
        "instrumental_path": str(instrumentals_output),
        "vocals_bpm": base_features.tempo,
        "instrumental_bpm": match_features.tempo,
        "vocals_key": "8A",  # Default, would be detected from features
        "instrumental_key": "8A",  # Default, would be detected from features
        "project_bpm": (base_features.tempo + match_features.tempo) / 2,
        "mix_params": {
            "vocal_gain_db": optimization_result.best_parameters.vocal_gain,
            "inst_gain_db": optimization_result.best_parameters.instrumental_gain,
            "sidechain_amount": optimization_result.best_parameters.sidechain_amount,
            "hp_filter_hz": optimization_result.best_parameters.eq_low_cut_freq,
            "reverb_send": optimization_result.best_parameters.reverb_send
        }
    }
    
    return stems


def cmd_bridge(args) -> int:
    """Bridge command: Create remix using Python-only processing."""
    print("🐍 AI RemixMate Bridge - Python-Only Processing")
    print("=" * 60)
    
    try:
        # Create Python bridge
        bridge = PythonBridge()
        
        # Create remix
        result = bridge.create_remix(
            base_song=args.base,
            match_song=args.match,
            output_dir=args.out_dir,
            preset=args.preset,
            optimize=args.optimize,
            online=args.online
        )
        
        if result["success"]:
            print(f"\n✅ Python bridge completed successfully!")
            print(f"   Session ID: {result['session_id']}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Mix output: {result['files']['mix_output']}")
            
            # Print quality metrics
            if "quality_metrics" in result:
                print(f"\n📊 Quality Metrics:")
                for metric, value in result["quality_metrics"].items():
                    if isinstance(value, dict):
                        print(f"   {metric}:")
                        for sub_metric, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                print(f"     {sub_metric}: {sub_value:.2f}")
                            else:
                                print(f"     {sub_metric}: {sub_value}")
                    else:
                        if isinstance(value, (int, float)):
                            print(f"   {metric}: {value:.2f}")
                        else:
                            print(f"   {metric}: {value}")
            
            return 0
        else:
            print(f"❌ Python bridge failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"❌ Bridge command failed: {e}")
        return 1


def cmd_recommend(args) -> int:
    """Recommend command: Find compatible songs for remixing."""
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


def build_parser() -> argparse.ArgumentParser:
    """Build the main argument parser."""
    parser = argparse.ArgumentParser(
        description="AI RemixMate - Professional Remix Generator with Python-Only Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create remix using Python-only processing
  python scripts/remixmate_cli.py bridge --base "SongA" --match "SongB" --out-dir runs/session1
  
  # Create club-style remix with optimization
  python scripts/remixmate_cli.py bridge --base "SongA" --match "SongB" --out-dir runs/session1 --preset club --optimize
  
  # Find key-compatible songs
  python scripts/remixmate_cli.py recommend --base "SongA" --top 5 --filter key_compatible
  
  # Find similar songs
  python scripts/remixmate_cli.py recommend --base "SongA" --top 10 --filter similarity
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Bridge command
    bridge_parser = subparsers.add_parser('bridge', help='Create remix using Python-only processing')
    bridge_parser.add_argument('--base', required=True, help='Base song name (vocals source)')
    bridge_parser.add_argument('--match', required=True, help='Match song name (instrumentals source)')
    bridge_parser.add_argument('--out-dir', required=True, help='Output directory for session')
    bridge_parser.add_argument('--preset', choices=['radio', 'club', 'ambient'], default='radio', help='Remix preset style')
    bridge_parser.add_argument('--optimize', action='store_true', help='Use iterative optimization')
    bridge_parser.add_argument('--online', action='store_true', help='Fetch missing songs via YouTube Music')
    bridge_parser.set_defaults(func=cmd_bridge)
    
    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Recommend compatible songs')
    recommend_parser.add_argument('--base', required=True, help='Base song name')
    recommend_parser.add_argument('--top', type=int, default=5, help='Number of recommendations')
    recommend_parser.add_argument('--filter', 
                                choices=['similarity', 'key_compatible', 'tempo_compatible'], 
                                default='similarity', help='Filtering method')
    recommend_parser.set_defaults(func=cmd_recommend)
    
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
