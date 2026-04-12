#!/usr/bin/env python3
# scripts/download_song.py
"""
Download a single song from YouTube Music into audio_input/ as WAV.
Requires: yt-dlp and ffmpeg
"""

from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path
import yt_dlp

from scripts.core.paths import AUDIO_IN


def sanitize_title(title: str) -> str:
    """Remove illegal filename characters and tidy up spacing."""
    name = re.sub(r"[\\/:*?\"<>|]+", "_", title)
    name = re.sub(r"\s+", " ", name)
    return name.strip("._ ")


def download_song(url: str, title: str | None = None) -> None:
    """Download a single YouTube Music link as WAV into AUDIO_IN."""
    if shutil.which("ffmpeg") is None:
        print("❌ Missing dependency: ffmpeg. Please install it before running.")
        return

    AUDIO_IN.mkdir(parents=True, exist_ok=True)

    safe_title = sanitize_title(title) if title else None
    out_path = AUDIO_IN / (f"{safe_title}.wav" if safe_title else "%(title)s.%(ext)s")

    if safe_title and out_path.exists():
        print(f"⏩ Skipping (already exists): {out_path.name}")
        return

    # YouTube Music optimized options
    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(out_path if safe_title else AUDIO_IN / "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
        # YouTube Music specific options
        "extract_flat": False,
        "writethumbnail": False,
        "writeinfojson": False,
        # Prefer YouTube Music over regular YouTube
        "prefer_insecure": False,
        "nocheckcertificate": False,
        # Audio quality preferences
        "audioformat": "wav",
        "audioquality": "0",  # Best quality
    }

    # Convert YouTube Music URLs if needed
    if "music.youtube.com" not in url and "youtube.com" in url:
        # Convert regular YouTube URL to YouTube Music search
        if "watch?v=" in url:
            video_id = url.split("watch?v=")[1].split("&")[0]
            # Try to find the same song on YouTube Music
            search_url = f"ytsearch1:music.youtube.com {video_id}"
            print(f"🔄 Converting to YouTube Music search: {search_url}")
        else:
            search_url = url
    else:
        search_url = url

    print(f"🎵 Downloading from YouTube Music: {search_url}")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([search_url])
        print(f"✅ Saved to: {AUDIO_IN}")
    except Exception as e:
        print(f"❌ Failed to download {search_url}: {e}")
        # Fallback to regular YouTube if YouTube Music fails
        if "music.youtube.com" in search_url or "ytsearch" in search_url:
            print("🔄 Falling back to regular YouTube...")
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])
                print(f"✅ Fallback successful, saved to: {AUDIO_IN}")
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")


def main():
    ap = argparse.ArgumentParser(description="Download a single YouTube Music link as WAV into audio_input/")
    ap.add_argument("url", help="YouTube Music link or regular YouTube link (will be converted)")
    ap.add_argument("--title", help="Optional custom title for the output file")
    args = ap.parse_args()

    download_song(args.url, args.title)


if __name__ == "__main__":
    main()