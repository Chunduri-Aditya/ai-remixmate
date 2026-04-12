#!/usr/bin/env python3
"""
YouTube Music Remix Demo

This script demonstrates the complete workflow:
1. Download high-quality songs from YouTube Music
2. Separate audio using demucs
3. Create optimized remix using the optimization proxy
4. Compare quality with previous versions
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time

from scripts.core.youtube_music import download_from_youtube_music
from scripts.core.advanced_remix import AdvancedRemixEngine
from scripts.core.paths import AUDIO_IN, SEPARATED


def download_high_quality_songs():
    """Download high-quality techno songs from YouTube Music."""
    print("🎵 Downloading High-Quality Songs from YouTube Music")
    print("=" * 55)
    
    songs_to_download = [
        ("Eric Prydz Opus", "Opus_YTMusic"),
        ("Deadmau5 Strobe", "Strobe_YTMusic"),
        ("Avicii Levels", "Levels_YTMusic"),
    ]
    
    downloaded_songs = []
    
    for query, filename in songs_to_download:
        print(f"\n🎧 Downloading: {query}")
        success, path = download_from_youtube_music(query, filename, prefer_music=True)
        
        if success:
            downloaded_songs.append((query, filename, path))
            print(f"✅ Success: {path.name}")
            
            # Show file info
            file_size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"   Size: {file_size:.1f} MB")
        else:
            print(f"❌ Failed: {query}")
    
    return downloaded_songs


def separate_audio_files(song_names: list[str]):
    """Separate audio files using demucs."""
    print(f"\n🎛️ Separating Audio Files")
    print("=" * 30)
    
    from scripts.batch_demucs import separate_all
    
    for song_name in song_names:
        print(f"🎧 Separating: {song_name}")
        separate_all(pattern=f"{song_name}.wav")


def create_optimized_remix(base_song: str, match_song: str, output_name: str):
    """Create an optimized remix using the advanced engine."""
    print(f"\n🚀 Creating Optimized Remix")
    print("=" * 35)
    print(f"Base: {base_song}")
    print(f"Match: {match_song}")
    
    engine = AdvancedRemixEngine()
    
    start_time = time.time()
    output_path, metrics = engine.create_high_quality_remix(
        base_song=base_song,
        match_song=match_song,
        output_name=output_name,
        auto_optimize=True
    )
    processing_time = time.time() - start_time
    
    print(f"\n📊 Remix Results:")
    print(f"   Output: {output_path.name}")
    print(f"   Size: {metrics['output_file_size_mb']:.1f} MB")
    print(f"   Processing Time: {processing_time:.1f} seconds")
    print(f"   Sample Rate: {metrics['sample_rate']} Hz")
    print(f"   Bit Depth: {metrics['bit_depth']} bit")
    
    return output_path, metrics


def compare_quality():
    """Compare quality between different sources."""
    print(f"\n📈 Quality Comparison")
    print("=" * 25)
    
    # Compare file sizes
    audio_files = list(AUDIO_IN.glob("*.wav"))
    ytmusic_files = [f for f in audio_files if "YTMusic" in f.name]
    regular_files = [f for f in audio_files if "YTMusic" not in f.name and f.name not in ["Test_YTMusic.wav"]]
    
    if ytmusic_files and regular_files:
        print("📊 File Size Comparison:")
        for yt_file in ytmusic_files[:2]:  # Show first 2
            yt_size = yt_file.stat().st_size / (1024 * 1024)
            print(f"   YouTube Music: {yt_file.name} - {yt_size:.1f} MB")
        
        for reg_file in regular_files[:2]:  # Show first 2
            reg_size = reg_file.stat().st_size / (1024 * 1024)
            print(f"   Regular YouTube: {reg_file.name} - {reg_size:.1f} MB")
    
    # Compare remix outputs
    output_files = list(Path("output").glob("*.wav"))
    optimized_files = [f for f in output_files if "optimized" in f.name]
    regular_remix_files = [f for f in output_files if "optimized" not in f.name and "techno_remix" in f.name]
    
    if optimized_files and regular_remix_files:
        print(f"\n🎵 Remix Quality Comparison:")
        for opt_file in optimized_files:
            opt_size = opt_file.stat().st_size / (1024 * 1024)
            print(f"   Optimized: {opt_file.name} - {opt_size:.1f} MB")
        
        for reg_file in regular_remix_files:
            reg_size = reg_file.stat().st_size / (1024 * 1024)
            print(f"   Regular: {reg_file.name} - {reg_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="YouTube Music Remix Demo")
    parser.add_argument("--download-only", action="store_true", help="Only download songs")
    parser.add_argument("--remix-only", action="store_true", help="Only create remix (skip download)")
    parser.add_argument("--base", default="Opus_YTMusic", help="Base song for remix")
    parser.add_argument("--match", default="Strobe_YTMusic", help="Match song for remix")
    parser.add_argument("--output", help="Custom output name for remix")
    
    args = parser.parse_args()
    
    print("🚀 YouTube Music Remix Demo")
    print("=" * 40)
    print("This demo showcases:")
    print("• High-quality downloads from YouTube Music")
    print("• Advanced audio separation")
    print("• ML-optimized remix generation")
    print("• Quality comparison and analysis")
    print()
    
    if not args.remix_only:
        # Download songs
        downloaded_songs = download_high_quality_songs()
        
        if not downloaded_songs:
            print("❌ No songs downloaded successfully")
            return
        
        # Separate audio
        song_names = [song[1] for song in downloaded_songs]
        separate_audio_files(song_names)
    
    if not args.download_only:
        # Create optimized remix
        output_name = args.output or f"youtube_music_remix_{args.base}_{args.match}.wav"
        
        try:
            output_path, metrics = create_optimized_remix(
                args.base, args.match, output_name
            )
            
            # Generate quality report
            engine = AdvancedRemixEngine()
            report = engine.generate_quality_report(output_path, metrics)
            print(report)
            
        except Exception as e:
            print(f"❌ Remix creation failed: {e}")
            print("💡 Make sure the songs are downloaded and separated first")
    
    # Compare quality
    compare_quality()
    
    print(f"\n🎉 YouTube Music Remix Demo Complete!")
    print("=" * 45)
    print("✅ High-quality downloads from YouTube Music")
    print("✅ Advanced audio processing")
    print("✅ ML-optimized remix generation")
    print("✅ Quality analysis and comparison")


if __name__ == "__main__":
    main()
