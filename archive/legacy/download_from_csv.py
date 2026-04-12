#!/usr/bin/env python3
# scripts/download_from_csv.py

from __future__ import annotations
import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

from scripts.core.paths import AUDIO_IN

def sanitize(name: str) -> str:
    # Keep it filesystem-friendly but readable
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip("._ ")

def download_song(url_or_search: str, title: str) -> None:
    safe_title = sanitize(title)
    out_wav = AUDIO_IN / f"{safe_title}.wav"

    if out_wav.exists():
        print(f"⏩ Skipping (already downloaded): {safe_title}")
        return

    if shutil.which("yt-dlp") is None:
        print("❌ 'yt-dlp' not found on PATH. Install with: pip install -U yt-dlp")
        return

    AUDIO_IN.mkdir(parents=True, exist_ok=True)

    print(f"🎧 Downloading from YouTube Music: {safe_title}")
    
    # Convert to YouTube Music search if it's a regular YouTube URL
    if url_or_search.startswith("http") and "music.youtube.com" not in url_or_search:
        # Convert regular YouTube URL to YouTube Music search
        if "watch?v=" in url_or_search:
            video_id = url_or_search.split("watch?v=")[1].split("&")[0]
            search_query = f"ytsearch1:music.youtube.com {video_id}"
        else:
            search_query = f"ytsearch1:music.youtube.com {title}"
    else:
        search_query = url_or_search
    
    # YouTube Music optimized yt-dlp command
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",  # Best quality
        "--format", "bestaudio[ext=m4a]/bestaudio/best",
        "--prefer-ffmpeg",
        "-o", str(out_wav.with_suffix(".%(ext)s")),  # resolves to *.wav
        search_query,
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Successfully downloaded: {safe_title}")
    except subprocess.CalledProcessError as e:
        print(f"❌ YouTube Music download failed for '{safe_title}': {e}")
        # Fallback to regular YouTube search
        if "music.youtube.com" in search_query or "ytsearch" in search_query:
            print(f"🔄 Falling back to regular YouTube for: {safe_title}")
            fallback_cmd = [
                "yt-dlp",
                "-x",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "-o", str(out_wav.with_suffix(".%(ext)s")),
                f"ytsearch1:{title}",
            ]
            try:
                subprocess.run(fallback_cmd, check=True)
                print(f"✅ Fallback successful: {safe_title}")
            except subprocess.CalledProcessError as e2:
                print(f"❌ Fallback also failed for '{safe_title}': {e2}")

def process_csv(csv_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("⚠️ CSV appears empty.")
        return

    # Try to detect header: if first row has 'title' in col0, treat as header
    start_idx = 1 if rows and rows[0] and rows[0][0].strip().lower() in {"title", "song", "name"} else 0

    for row in rows[start_idx:]:
        if not row:
            continue
        title = (row[0] or "").strip()
        if not title:
            continue

        url = row[1].strip() if len(row) > 1 else ""
        if url.startswith("http"):
            download_song(url, title)
        else:
            # Fallback to YouTube Music search
            download_song(f"ytsearch1:music.youtube.com {title}", title)

def main():
    ap = argparse.ArgumentParser(description="Download WAVs from YouTube Music using CSV of 'title,url' (url optional).")
    ap.add_argument("--csv", required=True, help="Path to CSV file (columns: title,url). Header optional.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return

    process_csv(csv_path)

if __name__ == "__main__":
    main()