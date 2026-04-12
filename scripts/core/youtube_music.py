#!/usr/bin/env python3
"""
YouTube Music helper utilities.

Features:
- Search for tracks (via ytmusicapi if available, else yt-dlp text search)
- Download best audio with yt-dlp
- Convert to WAV with ffmpeg
- Optional: trigger local stem separation

Notes:
- Network access and external binaries (ffmpeg) are required at runtime.
- All functions are defensive: they fail gracefully if deps are missing.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import subprocess
import shlex
import json
import re
import shutil

# Local paths
from scripts.core.paths import AUDIO_IN, SEPARATED, ensure_directories


def _have_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(shlex.split(cmd), capture_output=True, text=True)


def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^\w\-\s\(\)\[\]\&]+", "", name)
    return name[:120]


@dataclass
class SearchResult:
    title: str
    artist: Optional[str]
    url: str
    video_id: Optional[str]
    duration: Optional[int]


def search_youtube_music(query: str, limit: int = 5) -> List[SearchResult]:
    """Search YouTube Music for a query.
    Prefers ytmusicapi if available. Falls back to yt-dlp text search.
    """
    results: List[SearchResult] = []
    if _have_module("ytmusicapi"):
        try:
            from ytmusicapi import YTMusic
            ytm = YTMusic()  # uses anonymous headers
            hits = ytm.search(query, filter="songs", limit=limit)
            for h in hits:
                video_id = h.get("videoId")
                title = h.get("title")
                artists = h.get("artists") or []
                artist = artists[0]["name"] if artists else None
                url = f"https://music.youtube.com/watch?v={video_id}" if video_id else ""
                duration = None
                results.append(SearchResult(title=title or query, artist=artist, url=url, video_id=video_id, duration=duration))
            return results
        except Exception:
            # fall back to yt-dlp
            pass

    # Fallback: yt-dlp text search
    try:
        # Use ytsearch to get a JSON dump
        cmd = f"yt-dlp --dump-json \"ytsearch{limit}:{query}\""
        p = _run(cmd)
        if p.returncode == 0 and p.stdout:
            for line in p.stdout.splitlines():
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                title = obj.get("title")
                url = obj.get("webpage_url") or ""
                duration = obj.get("duration")
                uploader = obj.get("uploader")
                video_id = obj.get("id")
                results.append(SearchResult(title=title or query, artist=uploader, url=url, video_id=video_id, duration=duration))
    except Exception:
        # no results
        pass

    return results[:limit]


def download_best_audio(url_or_id: str, out_name: str) -> Optional[Path]:
    """Download best audio to AUDIO_IN/out_name.(m4a|webm) and convert to WAV.
    Returns path to WAV or None on failure.
    """
    ensure_directories()
    AUDIO_IN.mkdir(parents=True, exist_ok=True)

    # Determine URL
    if re.match(r"^[\w-]{11}$", url_or_id):
        url = f"https://www.youtube.com/watch?v={url_or_id}"
    else:
        url = url_or_id

    safe_name = sanitize_name(out_name)
    base = AUDIO_IN / safe_name

    # Download with yt-dlp
    # Prefer m4a for AAC
    dl_cmd = (
        f"yt-dlp -f 'bestaudio[ext=m4a]/bestaudio/best' --no-playlist "
        f"-o '{base}.%(ext)s' {shlex.quote(url)}"
    )
    p = _run(dl_cmd)
    if p.returncode != 0:
        return None

    # Find the downloaded file (m4a/webm)
    src_candidates = list(base.parent.glob(base.name + ".m4a")) + list(base.parent.glob(base.name + ".webm"))
    if not src_candidates:
        return None
    src = src_candidates[0]

    # Convert to WAV with ffmpeg
    wav_path = AUDIO_IN / f"{safe_name}.wav"
    ffmpeg_cmd = f"ffmpeg -y -i {shlex.quote(str(src))} -ac 1 -ar 44100 {shlex.quote(str(wav_path))}"
    p2 = _run(ffmpeg_cmd)
    if p2.returncode != 0:
        return None

    return wav_path


def separate_to_stems(song_name: str, wav_path: Path) -> Dict[str, Path]:
    """Separate to stems using Demucs when available, with graceful fallback.

    Returns dict with available stem paths (e.g. wav, vocals, drums, bass, other).
    """
    stems: Dict[str, Path] = {}

    # Prefer the same demucs pipeline as batch_demucs.py
    # demucs -n htdemucs -o <separated_parent> <wav>
    separated_parent = SEPARATED.parent
    song_stem = song_name

    if shutil.which("demucs") is not None:
        separated_parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "demucs",
            "-n",
            "htdemucs",
            "-o",
            str(separated_parent),
            str(wav_path),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            # If demucs fails, fall back to simple copy below
            pass

    target_dir = SEPARATED / song_stem
    target_dir.mkdir(parents=True, exist_ok=True)

    # Collect any stems produced by demucs if they exist
    for stem_name in ("vocals", "drums", "bass", "other"):
        stem_path = target_dir / f"{stem_name}.wav"
        if stem_path.exists():
            stems[stem_name] = stem_path

    if stems:
        return stems

    # Fallback: copy original WAV as 'other.wav' so downstream code still works
    fallback_other = target_dir / "other.wav"
    try:
        shutil.copy2(str(wav_path), str(fallback_other))
        stems["other"] = fallback_other
        return stems
    except Exception:
        return {}


def fetch_song_to_library(query: str, out_name: Optional[str] = None, separate: bool = True) -> Dict[str, Path]:
    """High-level helper: search, download, convert, and optionally separate.
    Returns dict with available paths: wav, vocals, other.
    """
    out: Dict[str, Path] = {}

    # If it's a URL or video id, skip search
    url_or_id = query
    if not (query.startswith("http://") or query.startswith("https://") or re.match(r"^[\w-]{11}$", query)):
        hits = search_youtube_music(query, limit=1)
        if not hits:
            return out
        url_or_id = hits[0].url or (hits[0].video_id or query)
        if out_name is None:
            base = f"{hits[0].title}"
        else:
            base = out_name
    else:
        base = out_name or "downloaded_track"

    wav = download_best_audio(url_or_id, base)
    if not wav:
        return out
    out["wav"] = wav

    if separate:
        song_name = sanitize_name(base)
        stems = separate_to_stems(song_name, wav)
        out.update(stems)

    return out