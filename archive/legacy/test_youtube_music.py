#!/usr/bin/env python3
"""
Test YouTube Music Download Functionality

This script tests the YouTube Music download capabilities and compares
quality with regular YouTube downloads.
"""

from __future__ import annotations
import argparse
from pathlib import Path

from scripts.core.youtube_music import (
    download_from_youtube_music, 
    test_youtube_music_connection,
    get_youtube_music_url
)


def test_connection():
    """Test YouTube Music connectivity."""
    print("🔍 Testing YouTube Music Connection...")
    success = test_youtube_music_connection()
    if success:
        print("✅ YouTube Music connection successful!")
    else:
        print("❌ YouTube Music connection failed!")
    return success


def download_test_songs():
    """Download test songs from YouTube Music."""
    print("🎵 Downloading Test Songs from YouTube Music...")
    print("=" * 50)
    
    # Test songs
    test_songs = [
        ("Darude Sandstorm", "Sandstorm_YTMusic"),
        ("Daft Punk One More Time", "One_More_Time_YTMusic"),
        ("Deadmau5 Strobe", "Strobe_YTMusic"),
    ]
    
    results = []
    for query, filename in test_songs:
        print(f"\n🎧 Downloading: {query}")
        success, path = download_from_youtube_music(query, filename, prefer_music=True)
        results.append((query, success, path))
        
        if success:
            print(f"✅ Success: {path.name}")
        else:
            print(f"❌ Failed: {query}")
    
    return results


def compare_quality():
    """Compare quality between YouTube Music and regular YouTube."""
    print("\n📊 Quality Comparison")
    print("=" * 30)
    
    # Download same song from both sources
    song_query = "Darude Sandstorm"
    
    print(f"🎵 Downloading '{song_query}' from YouTube Music...")
    music_success, music_path = download_from_youtube_music(
        song_query, "Sandstorm_YTMusic", prefer_music=True
    )
    
    print(f"📺 Downloading '{song_query}' from regular YouTube...")
    youtube_success, youtube_path = download_from_youtube_music(
        song_query, "Sandstorm_YouTube", prefer_music=False
    )
    
    if music_success and youtube_success:
        print("\n📈 File Size Comparison:")
        music_size = music_path.stat().st_size / (1024 * 1024)  # MB
        youtube_size = youtube_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"   YouTube Music: {music_size:.1f} MB")
        print(f"   Regular YouTube: {youtube_size:.1f} MB")
        print(f"   Difference: {((music_size - youtube_size) / youtube_size * 100):+.1f}%")
        
        if music_size > youtube_size:
            print("✅ YouTube Music provides higher quality!")
        else:
            print("⚠️ Similar quality between sources")
    
    return music_success, youtube_success


def generate_music_urls():
    """Generate YouTube Music URLs for popular songs."""
    print("\n🔗 YouTube Music URLs")
    print("=" * 25)
    
    songs = [
        ("Sandstorm", "Darude"),
        ("One More Time", "Daft Punk"),
        ("Strobe", "Deadmau5"),
        ("Opus", "Eric Prydz"),
        ("Levels", "Avicii"),
    ]
    
    for title, artist in songs:
        url = get_youtube_music_url(title, artist)
        print(f"🎵 {artist} - {title}")
        print(f"   {url}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Test YouTube Music download functionality")
    parser.add_argument("--test-connection", action="store_true", help="Test YouTube Music connectivity")
    parser.add_argument("--download-songs", action="store_true", help="Download test songs")
    parser.add_argument("--compare-quality", action="store_true", help="Compare quality with regular YouTube")
    parser.add_argument("--generate-urls", action="store_true", help="Generate YouTube Music URLs")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.all or not any([args.test_connection, args.download_songs, args.compare_quality, args.generate_urls]):
        # Run all tests by default
        args.test_connection = True
        args.download_songs = True
        args.compare_quality = True
        args.generate_urls = True
    
    print("🚀 YouTube Music Test Suite")
    print("=" * 40)
    
    if args.test_connection:
        test_connection()
        print()
    
    if args.download_songs:
        download_test_songs()
        print()
    
    if args.compare_quality:
        compare_quality()
        print()
    
    if args.generate_urls:
        generate_music_urls()
    
    print("🎉 YouTube Music testing complete!")


if __name__ == "__main__":
    main()
