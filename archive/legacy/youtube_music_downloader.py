#!/usr/bin/env python3
"""
Simple YouTube Music downloader + stem splitter.

Usage examples:

  # Download full song from YouTube Music (search by text)
  python scripts/youtube_music_downloader.py --query "Eric Prydz Opus"

  # Download by URL and force stem splitting with Demucs
  python scripts/youtube_music_downloader.py --query "https://music.youtube.com/watch?v=XXXX" --name "Opus_YTMusic" --split

  # Download only (no stems)
  python scripts/youtube_music_downloader.py --query "Deadmau5 Strobe" --no-split

By default (no --split/--no-split), the script will interactively ask whether
to split the downloaded WAV into stems using Demucs so you can load them into
FL Studio, Logic Pro, etc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.core.youtube_music import fetch_song_to_library
from scripts.core.paths import AUDIO_IN, SEPARATED, ensure_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download songs from YouTube Music and optionally split into stems for DAW use."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Song search query, YouTube Music URL, or YouTube video ID.",
    )
    parser.add_argument(
        "--name",
        help="Optional output name (base file name without extension). If omitted, uses the track title.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--split",
        action="store_true",
        help="Automatically split into stems using Demucs (no prompt).",
    )
    group.add_argument(
        "--no-split",
        action="store_true",
        help="Download full song only, do not split into stems.",
    )
    return parser.parse_args()


def should_split_interactively() -> bool:
    """Ask the user if we should run stem splitting (when in a TTY and flags not set)."""
    if not sys.stdin.isatty():
        # Non-interactive environment: default to no extra processing
        return False

    while True:
        answer = input(
            "Do you want to split this song into stems using the Demucs Python library now? [y/N]: "
        ).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no", ""):
            return False
        print("Please answer 'y' or 'n'.")


def main() -> int:
    args = parse_args()

    ensure_directories()

    print("🚀 AI RemixMate - YouTube Music Downloader")
    print("=" * 55)
    print(f"🎵 Query: {args.query}")
    if args.name:
        print(f"💾 Output name: {args.name}")
    print(f"📂 Audio input folder: {AUDIO_IN}")
    print(f"📂 Stems folder (for Demucs outputs): {SEPARATED}")
    print()

    if args.split:
        separate = True
    elif args.no_split:
        separate = False
    else:
        separate = should_split_interactively()

    if separate:
        print("🎛️ Will run Demucs stem separation after download.")
    else:
        print("🎧 Will download full song only (no stem splitting).")

    print()
    print("⬇️  Downloading from YouTube Music (via search / yt-dlp)...")

    result = fetch_song_to_library(
        query=args.query,
        out_name=args.name,
        separate=separate,
    )

    if not result or "wav" not in result:
        print("❌ Failed to download audio from YouTube/YouTube Music.")
        print("   Check your network connection and that yt-dlp/ffmpeg are installed.")
        return 1

    wav_path: Path = result["wav"]
    print(f"✅ Downloaded WAV: {wav_path}")

    if separate:
        # Show any stems we produced
        stem_keys = [k for k in result.keys() if k != "wav"]
        if stem_keys:
            print("\n🎚️ Generated stems using Demucs:")
            for key in stem_keys:
                print(f"   - {key}: {result[key]}")
            print("\nYou can now drag these stems (vocals, drums, bass, other) into FL Studio or Logic Pro.")
        else:
            print("\n⚠️ Demucs stem separation did not produce individual stems.")
            print("   A fallback 'other.wav' may have been created, or Demucs is not installed on PATH.")
    else:
        print("\n💡 Tip: Re-run with --split to generate stems using Demucs.")

    print("\n🎉 Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

