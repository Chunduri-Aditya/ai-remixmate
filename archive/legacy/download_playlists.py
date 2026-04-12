#!/usr/bin/env python3
"""
scripts/download_playlists.py — Batch download all songs from playlist JSON files.

Reads every playlist JSON in the playlist_datasets folder, deduplicates by
YouTube video ID, then downloads each track using the existing download_track()
infrastructure. Skips songs that are already on disk.

Usage:
    # Download all playlists (no stem separation)
    python -m scripts.download_playlists

    # Download all playlists AND run Demucs stem separation
    python -m scripts.download_playlists --split

    # Dry run — show what would be downloaded without downloading
    python -m scripts.download_playlists --dry-run

    # Limit to N songs (useful for testing)
    python -m scripts.download_playlists --limit 5

    # Specific playlist only
    python -m scripts.download_playlists --playlist "Bad_playlist"

    # Start from a specific index (resume interrupted run)
    python -m scripts.download_playlists --start 50
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
PLAYLIST_DIR = PROJECT_ROOT / "github_files" / "ai-remixmate" / "models" / "playlist_datasets"
LOG_DIR = PROJECT_ROOT / "runs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "playlist_download.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlaylistSong:
    video_id: str
    title: str
    uploader: str
    url: str
    duration: int          # seconds
    playlist_name: str
    safe_name: str = ""    # filesystem-safe name, filled on load

    def __post_init__(self):
        if not self.safe_name:
            self.safe_name = _safe_filename(f"{self.uploader} - {self.title}")


@dataclass
class BatchResult:
    total: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    failed_songs: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(s: str, max_len: int = 80) -> str:
    """Convert a song title + artist into a safe filesystem name."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s[:max_len].rstrip(". ")
    return s or "unknown"


def _load_playlist(path: Path) -> List[PlaylistSong]:
    """Load songs from a single playlist JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    playlist_name = data.get("title", path.stem)
    songs = []
    for item in data.get("songs", []):
        vid_id = item.get("id", "")
        title  = item.get("title", "Unknown")
        uploader = item.get("uploader", "Unknown")
        url    = item.get("url", f"https://music.youtube.com/watch?v={vid_id}")
        dur    = item.get("duration", 0)
        if not vid_id:
            continue
        songs.append(PlaylistSong(
            video_id=vid_id,
            title=title,
            uploader=uploader,
            url=url,
            duration=dur,
            playlist_name=playlist_name,
        ))
    return songs


def _load_all_playlists(playlist_filter: Optional[str] = None) -> List[PlaylistSong]:
    """Load and deduplicate all playlist files. Returns songs in playlist order."""
    playlist_files = sorted(PLAYLIST_DIR.iterdir())
    seen_ids: Set[str] = set()
    all_songs: List[PlaylistSong] = []

    for pf in playlist_files:
        # Accept files with or without .json extension
        try:
            songs = _load_playlist(pf)
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            log.warning("Skipping non-playlist file: %s", pf.name)
            continue

        if playlist_filter and playlist_filter.lower() not in pf.name.lower():
            continue

        before = len(all_songs)
        for song in songs:
            if song.video_id not in seen_ids:
                seen_ids.add(song.video_id)
                all_songs.append(song)

        added = len(all_songs) - before
        dupes = len(songs) - added
        log.info("Loaded %-35s  %3d songs  (%d dupes skipped)",
                 f"'{pf.name}'", added, dupes)

    return all_songs


def _is_already_downloaded(safe_name: str) -> bool:
    """Check if a song already exists in the library (WAV or stems)."""
    from scripts.core.paths import song_dir
    d = song_dir(safe_name)
    if not d.exists():
        return False
    # Has full wav
    if (d / "full.wav").exists():
        return True
    # Has at least one stem
    for stem in ("vocals.wav", "drums.wav", "bass.wav", "other.wav"):
        if (d / stem).exists():
            return True
    return False


def _fmt_duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Core download loop
# ---------------------------------------------------------------------------

def download_all(
    songs: List[PlaylistSong],
    separate: bool = False,
    dry_run: bool = False,
    start: int = 0,
) -> BatchResult:
    from scripts.download import download_track, TrackSpec

    result = BatchResult(total=len(songs))
    songs = songs[start:]

    for i, song in enumerate(songs, start=start + 1):
        tag = f"[{i}/{result.total}]"

        # --- Skip check ---
        if _is_already_downloaded(song.safe_name):
            log.info("%s SKIP  %-60s  (already on disk)", tag, song.safe_name[:60])
            result.skipped += 1
            continue

        if dry_run:
            log.info("%s DRY   %-60s  [%s]  %s",
                     tag, song.safe_name[:60], song.playlist_name, _fmt_duration(song.duration))
            result.downloaded += 1
            continue

        # --- Download ---
        log.info("%s ⬇️   %-60s  [%s]  %s",
                 tag, song.safe_name[:60], song.playlist_name, _fmt_duration(song.duration))
        t0 = time.time()
        try:
            spec = TrackSpec(
                query=song.url,
                name=song.safe_name,
                separate=separate,
            )
            dl = download_track(spec)
            elapsed = time.time() - t0

            if dl.success:
                size_mb = dl.wav.stat().st_size / 1_048_576 if dl.wav and dl.wav.exists() else 0
                log.info("%s ✅  %-60s  %.1fs  %.1fMB", tag, song.safe_name[:60], elapsed, size_mb)
                if dl.license_warning:
                    log.warning("   ⚠️  %s", dl.license_warning)
                result.downloaded += 1
            else:
                log.error("%s ❌  %-60s  %s", tag, song.safe_name[:60], dl.error)
                result.failed += 1
                result.failed_songs.append(f"{song.safe_name}  ({dl.error})")

        except KeyboardInterrupt:
            log.warning("\n⏸  Interrupted at song %d. Re-run with --start %d to resume.", i, i - 1)
            sys.exit(0)
        except Exception as exc:
            log.exception("%s 💥  Unexpected error: %s", tag, exc)
            result.failed += 1
            result.failed_songs.append(f"{song.safe_name}  ({exc})")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_summary(songs: List[PlaylistSong]) -> None:
    """Print a table of what will be downloaded, grouped by playlist."""
    by_playlist: Dict[str, int] = {}
    total_dur = 0
    for s in songs:
        by_playlist[s.playlist_name] = by_playlist.get(s.playlist_name, 0) + 1
        total_dur += s.duration

    print("\n📋  Playlists to download:")
    print(f"  {'Playlist':<40}  Songs")
    print(f"  {'-'*40}  -----")
    for name, count in sorted(by_playlist.items()):
        print(f"  {name:<40}  {count}")
    hrs = total_dur // 3600
    mins = (total_dur % 3600) // 60
    print(f"\n  Total: {len(songs)} unique songs  (~{hrs}h {mins}m of audio)\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch download all playlist JSON files into the RemixMate library."
    )
    parser.add_argument("--split", action="store_true",
                        help="Run Demucs stem separation after each download.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without actually downloading.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only download the first N songs (for testing).")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip the first N songs (resume an interrupted run).")
    parser.add_argument("--playlist", type=str, default="",
                        help="Only download songs from playlists whose name contains this string.")
    args = parser.parse_args()

    # Load & deduplicate
    log.info("=" * 60)
    log.info("AI RemixMate — Playlist Batch Downloader")
    log.info("Playlist folder: %s", PLAYLIST_DIR)
    log.info("=" * 60)

    songs = _load_all_playlists(playlist_filter=args.playlist or None)

    if not songs:
        log.error("No songs found. Check that PLAYLIST_DIR exists: %s", PLAYLIST_DIR)
        sys.exit(1)

    if args.limit > 0:
        songs = songs[:args.limit]
        log.info("--limit %d applied — capped at %d songs.", args.limit, len(songs))

    _print_summary(songs)

    if args.dry_run:
        log.info("DRY RUN — no files will be downloaded.")

    # Run
    result = download_all(
        songs=songs,
        separate=args.split,
        dry_run=args.dry_run,
        start=args.start,
    )

    # Summary
    log.info("=" * 60)
    log.info("DONE  |  total=%d  downloaded=%d  skipped=%d  failed=%d",
             result.total, result.downloaded, result.skipped, result.failed)
    if result.failed_songs:
        log.warning("Failed songs:")
        for s in result.failed_songs:
            log.warning("  ❌  %s", s)
    log.info("Log saved to: %s", LOG_DIR / "playlist_download.log")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
