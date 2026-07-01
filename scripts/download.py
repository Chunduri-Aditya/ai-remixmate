#!/usr/bin/env python3
"""
scripts/download.py — Unified download entry point for AI RemixMate.

Supports:
  • Single tracks — YouTube Music, YouTube, SoundCloud, or any URL yt-dlp handles
  • Full playlists — any yt-dlp-compatible playlist URL
  • Search queries — plain text, resolved via YouTube Music (ytmusicapi) or yt-dlp search
  • Optional stem separation via Demucs after download
  • Real-time progress bars (tqdm)
  • Skip-if-exists deduplication

Usage:
  # Single track by search query
  python -m scripts.download "Eric Prydz Opus" --split

  # Single track from URL
  python -m scripts.download "https://music.youtube.com/watch?v=XXX" --name "Opus" --split

  # Full playlist
  python -m scripts.download "https://www.youtube.com/playlist?list=PL..." --playlist

  # From a CSV file (title, optional_url)
  python -m scripts.download --csv songs.csv --split

Run `python -m scripts.download --help` for all options.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shlex
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Dict

# Progress callback signature: (fraction_0_to_1, human_readable_message)
ProgressCb = Callable[[float, str], None]

from scripts.core.paths import (
    LIBRARY_DIR,
    ensure_directories,
    song_dir,
    full_wav_path,
    vocals_path,
    other_path,
)
from scripts.core.license import (
    classify_license,
    license_from_ytdlp_info,
    license_warning,
    save_license,
    LicenseInfo,
    LicenseType,
)
from scripts.core.library import get_library_manager

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _have_bin(name: str) -> bool:
    return shutil.which(name) is not None


def _have_mod(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _check_deps() -> bool:
    ok = True
    if not (_have_mod("yt_dlp") or _have_bin("yt-dlp")):
        print("❌  yt-dlp is required. Install: pip install -U yt-dlp")
        ok = False
    if not _have_bin("ffmpeg"):
        print("❌  ffmpeg is required. Install: brew install ffmpeg  (macOS)")
        ok = False
    if not _have_mod("demucs"):
        print("⚠️   demucs not found — stem splitting will be skipped.")
        print("     Install: pip install demucs")
    if not _have_mod("ytmusicapi"):
        print("ℹ️   ytmusicapi not found — using yt-dlp text search instead.")
        print("     Install for better results: pip install ytmusicapi")
    return ok


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TrackSpec:
    """Describes one track to download."""
    query: str                        # search query or URL
    name: Optional[str] = None        # desired output name (no extension)
    separate: bool = False            # run Demucs after download?
    auto_analyze: bool = True         # run BPM/key/structure analysis after stems?


@dataclass
class DownloadResult:
    name: str
    success: bool
    wav: Optional[Path] = None
    stems: Dict[str, Path] = field(default_factory=dict)
    error: Optional[str] = None
    license: Optional[LicenseInfo] = None       # licence metadata for the track
    license_warning: Optional[str] = None       # pre-formatted human-readable warning
    analyzed: bool = False                      # BPM/key/structure analysis completed?
    evicted: List[str] = field(default_factory=list)  # other songs auto-evicted (over cap)


# ---------------------------------------------------------------------------
# Legal free-music sources
# ---------------------------------------------------------------------------

JAMENDO_API = "https://api.jamendo.com/v3.0"

# Jamendo client ID — pulled from config; override via:
#   config.local.yaml: download.jamendo_client_id: your_key
#   env var:           REMIXMATE_DOWNLOAD_JAMENDO_CLIENT_ID=your_key
# Register free at https://devportal.jamendo.com
try:
    from scripts.core.config import cfg as _cfg
    JAMENDO_CLIENT_ID: str = _cfg.download.jamendo_client_id
except Exception:
    import os as _os
    JAMENDO_CLIENT_ID = _os.environ.get("JAMENDO_CLIENT_ID", "")  # set via env var


def search_jamendo(query: str, limit: int = 5) -> List[Dict]:
    """
    Search Jamendo — fully legal, Creative Commons licensed music.
    Requires an internet connection but NO API key for basic search
    (uses the public demo client ID above).

    Returns tracks with a direct MP3 stream URL suitable for yt-dlp or requests.
    """
    try:
        import requests as req
        params = {
            "client_id": JAMENDO_CLIENT_ID,
            "format": "json",
            "limit": limit,
            "search": query,
            "audioformat": "mp32",   # 320 kbps MP3
            "include": "musicinfo",
        }
        r = req.get(f"{JAMENDO_API}/tracks/", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = []
        for t in data.get("results", []):
            results.append({
                "title": t.get("name", query),
                "artist": t.get("artist_name"),
                "url": t.get("audio"),       # direct MP3 URL
                "duration": t.get("duration"),
                "license": t.get("license_ccurl"),
                "source": "jamendo",
            })
        return results
    except Exception:
        return []


def download_jamendo_track(url: str, name: str) -> Optional[Path]:
    """
    Download a Jamendo MP3 stream and convert to WAV.
    Jamendo allows downloading for non-commercial use under CC licences.
    """
    from scripts.core.paths import song_dir, ensure_directories
    ensure_directories()
    dest_dir = song_dir(name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    wav_dest = dest_dir / "full.wav"

    if wav_dest.exists():
        return wav_dest

    # Use yt-dlp (handles direct MP3 URLs cleanly) or fall back to requests
    try:
        import yt_dlp  # type: ignore
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(dest_dir / "full.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio",
                                 "preferredcodec": "wav", "preferredquality": "0"}],
            "quiet": False,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        if wav_dest.exists():
            return wav_dest
    except Exception:
        pass

    # Fallback: requests streaming download → ffmpeg convert
    try:
        import requests as req
        mp3_path = dest_dir / "full.mp3"
        with req.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(mp3_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_path),
             "-ac", "1", "-ar", "44100", str(wav_dest)],
            check=True, capture_output=True,
        )
        mp3_path.unlink(missing_ok=True)
        return wav_dest
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_track(query: str, limit: int = 5, source: str = "auto") -> List[Dict]:
    """
    Return a ranked list of candidate tracks for a free-text query.

    source options:
      "auto"     — ytmusicapi → yt-dlp search (YouTube; personal use)
      "youtube"  — yt-dlp text search only
      "jamendo"  — Jamendo only (CC licensed, fully legal for commercial use)
      "all"      — Jamendo first, then YouTube results appended

    Tries ytmusicapi first (richer metadata), falls back to yt-dlp search.
    """
    results: List[Dict] = []

    # --- Jamendo (legal CC music) ---
    if source in ("jamendo", "all"):
        results.extend(search_jamendo(query, limit=limit))
        if source == "jamendo":
            return results[:limit]

    # --- YouTube / YouTube Music (personal use) ---
    if source in ("auto", "youtube", "all"):
        if _have_mod("ytmusicapi") and source != "youtube":
            try:
                from ytmusicapi import YTMusic  # type: ignore
                ytm = YTMusic()
                hits = ytm.search(query, filter="songs", limit=limit)
                for h in hits:
                    vid = h.get("videoId")
                    artists = h.get("artists") or []
                    results.append({
                        "title": h.get("title", query),
                        "artist": artists[0]["name"] if artists else None,
                        "url": f"https://music.youtube.com/watch?v={vid}" if vid else "",
                        "duration": None,
                        "source": "ytmusicapi",
                    })
                if results:
                    return results[:limit]
            except Exception:
                pass  # fall through to yt-dlp

    # yt-dlp text search fallback
    if source in ("auto", "youtube", "all"):
        try:
            import yt_dlp  # type: ignore
            opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
                for entry in (info.get("entries") or []):
                    results.append({
                        "title": entry.get("title", query),
                        "artist": entry.get("uploader"),
                    "url": entry.get("webpage_url") or entry.get("url", ""),
                    "duration": entry.get("duration"),
                    "source": "yt-dlp",
                })
        except Exception:
            pass

    return results[:limit]


# ---------------------------------------------------------------------------
# Core download
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """
    Return a filesystem-safe version of a track name.

    Delegates to scripts.core.paths.sanitize_song_name() — the single
    source of truth shared with _SAFE_SONG_NAME_RE (routers/_helpers.py).
    This used to be its own independent blocklist (only stripping
    Windows-reserved chars), which let punctuation through that the API's
    stricter allowlist then permanently rejected on every subsequent
    request for that song. Keeping one shared definition means a name
    written to disk here can never become unreadable later.
    """
    from scripts.core.paths import sanitize_song_name
    return sanitize_song_name(name)


def _resolve_url(spec: TrackSpec) -> tuple[str, str]:
    """
    Return (url, resolved_name) for a TrackSpec.
    If spec.query is a URL/ID it is used directly; otherwise a search is performed.
    """
    query = spec.query
    is_url = query.startswith("http://") or query.startswith("https://")
    is_id  = bool(re.match(r"^[\w-]{11}$", query))

    if is_url or is_id:
        url  = query if is_url else f"https://www.youtube.com/watch?v={query}"
        name = spec.name or _sanitize(query)
        return url, name

    # Text search
    hits = search_track(query, limit=1)
    if hits:
        url  = hits[0]["url"]
        name = spec.name or _sanitize(hits[0]["title"])
        return url, name

    # Last resort: pass query directly to yt-dlp as a search
    return f"ytsearch1:{query}", spec.name or _sanitize(query)


def download_track(spec: TrackSpec, progress_cb: Optional[ProgressCb] = None) -> DownloadResult:
    """
    Download a single track and (optionally) split into stems.
    Output lives in library/<name>/ .

    progress_cb, if given, receives (fraction 0.0–1.0, message) covering the
    whole pipeline: resolve → yt-dlp download → WAV convert → stems → register.
    """
    ensure_directories()

    def _emit(frac: float, msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(min(max(frac, 0.0), 1.0), msg)
        except Exception:
            pass

    _emit(0.01, "Resolving track…")
    url, name = _resolve_url(spec)
    dest_dir  = song_dir(name)
    wav_dest  = dest_dir / "full.wav"

    # --- Skip if already downloaded ---
    if wav_dest.exists():
        print(f"⏩  Skipping (already in library): {name}")
        _emit(1.0, f"Already in library: {name}")
        result = DownloadResult(name=name, success=True, wav=wav_dest)
        if spec.separate:
            result.stems = _collect_stems(name)
        # Attach existing licence info if present
        try:
            from scripts.core.license import load_license as _ll
            existing_lic = _ll(dest_dir)
            if existing_lic:
                result.license = existing_lic
                warn = license_warning(existing_lic, name)
                if warn:
                    result.license_warning = warn
                    print(warn)
        except Exception:
            pass
        return result

    # --- Duplicate detection (same audio under a different name) ---
    # We skip this pre-download since we don't have the WAV yet;
    # it's done post-download in the fingerprint step below.

    dest_dir.mkdir(parents=True, exist_ok=True)

    # --- Download via yt-dlp ---
    print(f"⬇️   Downloading: {name}")
    _emit(0.03, f"Downloading: {name}")

    def _ydl_hook(d: Dict) -> None:
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes") or 0
            if total:
                frac = done / total
                # yt-dlp download occupies the 3%–40% window of the pipeline
                _emit(0.03 + 0.37 * frac, f"Downloading audio… {int(frac * 100)}%")
        elif status == "finished":
            _emit(0.40, "Converting to WAV…")

    try:
        import yt_dlp  # type: ignore
        tmp_tmpl = str(dest_dir / "full.%(ext)s")
        ydl_opts = {
            "progress_hooks": [_ydl_hook],
            "format": "bestaudio[acodec=opus]/bestaudio[acodec=aac][abr>128]/bestaudio/best",
            "outtmpl": tmp_tmpl,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }],
            "quiet": False,
            "no_warnings": False,
            "no_playlist": True,       # single track only in this function
            "prefer_free_formats": True,
            "writethumbnail": False,
            "writeinfojson": True,     # save metadata alongside WAV
            "outtmpl_na_placeholder": "",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        return DownloadResult(name=name, success=False, error=str(e))

    if not wav_dest.exists():
        return DownloadResult(name=name, success=False,
                              error="WAV not produced after download.")

    print(f"✅  Saved: {wav_dest}")
    _emit(0.45, "Saving licence metadata…")

    # --- Classify and save licence ---
    lic_info = _extract_and_save_license(name, dest_dir, url)

    # --- Duplicate detection (fingerprint) ---
    try:
        mgr = get_library_manager()
        dup = mgr.find_duplicate(wav_dest)
        if dup and dup != name:
            print(
                f"⚠️  Possible duplicate detected! '{name}' looks identical to "
                f"'{dup}' already in the library."
            )
    except Exception:
        pass

    # --- Optional stem separation ---
    stems: Dict[str, Path] = {}
    if spec.separate:
        # Demucs occupies the 50%–95% window of the pipeline
        stems = _separate_stems(
            name, wav_dest,
            progress_cb=(lambda p, m: _emit(0.50 + 0.45 * p, m)) if progress_cb else None,
        )

    # --- Auto-analyze (BPM/key/structure) — MUST run before the prune step
    # below. prune_on_download defaults to True and deletes full.wav right
    # after stems are produced; analysis needs full.wav, so it has to run
    # first or it would fail for every single download.
    analyzed = False
    if stems and spec.auto_analyze:
        _emit(0.95, "Analyzing…")
        try:
            from scripts.core.analysis_pipeline import run_song_analysis
            run_song_analysis(name)
            analyzed = True
        except Exception as exc:
            print(f"⚠️  Auto-analyze failed for '{name}' (non-critical): {exc}")

    # --- Smart storage: prune raw WAV if stems were produced ---
    if stems and _PRUNE_ON_DL:
        try:
            mgr = get_library_manager()
            mgr.prune_raw(name)
        except Exception:
            pass

    # --- Register in library index + store fingerprint ---
    _emit(0.96, "Registering in library…")
    evicted: List[str] = []
    try:
        mgr = get_library_manager()
        mgr.register(name, source=_source_from_url(url))
        mgr.store_fingerprint(name, wav_dest if wav_dest.exists() else dest_dir / "vocals.wav")
        # Evict LRU songs if library is over cap — this can delete OTHER,
        # older songs' full.wav (or whole song dirs) without the caller
        # asking for it. Surfaced via DownloadResult.evicted so the API/UI
        # can tell the user what happened instead of it being invisible.
        if _AUTO_EVICT:
            evicted = mgr.evict_lru() or []
    except Exception:
        pass

    # --- Licence warning for user ---
    warn_str: Optional[str] = None
    if lic_info:
        warn_str = license_warning(lic_info, name)
        if warn_str:
            print(warn_str)

    _emit(1.0, f"Done: {name}")
    return DownloadResult(
        name=name,
        success=True,
        wav=wav_dest if wav_dest.exists() else None,
        stems=stems,
        license=lic_info,
        license_warning=warn_str,
        analyzed=analyzed,
        evicted=evicted,
    )


# ---------------------------------------------------------------------------
# Playlist download
# ---------------------------------------------------------------------------

def download_playlist(url: str, separate: bool = False) -> List[DownloadResult]:
    """
    Download all tracks from a yt-dlp-compatible playlist URL.
    Each track lands in library/<track_name>/ .
    """
    ensure_directories()

    print(f"📋  Fetching playlist info: {url}")
    try:
        import yt_dlp  # type: ignore
        info_opts = {"quiet": True, "extract_flat": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"❌  Could not fetch playlist: {e}")
        return []

    entries = info.get("entries") or []
    if not entries:
        print("⚠️  Playlist appears to be empty.")
        return []

    total = len(entries)
    print(f"🎵  Found {total} tracks in playlist: {info.get('title', url)}")

    results: List[DownloadResult] = []
    for i, entry in enumerate(entries, 1):
        vid  = entry.get("id") or entry.get("url", "")
        title = _sanitize(entry.get("title") or f"track_{i:03d}")
        track_url = entry.get("webpage_url") or entry.get("url") or f"https://www.youtube.com/watch?v={vid}"
        print(f"\n[{i}/{total}] {title}")
        spec = TrackSpec(query=track_url, name=title, separate=separate)
        results.append(download_track(spec))

    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# CSV batch download
# ---------------------------------------------------------------------------

def download_from_csv(csv_path: Path, separate: bool = False) -> List[DownloadResult]:
    """
    Read a CSV with columns  title[, url]  and download each row.
    The URL column is optional; if absent a search query is used.
    """
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        print("⚠️  CSV is empty.")
        return []

    # Strip optional header
    if rows[0] and rows[0][0].strip().lower() in {"title", "song", "name", "track"}:
        rows = rows[1:]

    specs: List[TrackSpec] = []
    for row in rows:
        if not row:
            continue
        title = row[0].strip()
        url   = row[1].strip() if len(row) > 1 else ""
        query = url if url.startswith("http") else title
        specs.append(TrackSpec(query=query, name=_sanitize(title), separate=separate))

    print(f"📋  Processing {len(specs)} tracks from {csv_path.name}...")
    results = [download_track(s) for s in specs]
    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Licence helpers
# ---------------------------------------------------------------------------

# Read pruning preference from config (same default as library.py)
try:
    from scripts.core.config import cfg as _cfg_dl
    _PRUNE_ON_DL: bool   = getattr(getattr(_cfg_dl, "library", None), "prune_on_download", True)
    _AUTO_EVICT: bool    = getattr(getattr(_cfg_dl, "library", None), "auto_evict_on_download", True)
except Exception:
    _PRUNE_ON_DL = True
    _AUTO_EVICT  = True


def _source_from_url(url: str) -> str:
    """Guess source name from a URL string."""
    url_lower = url.lower()
    if "jamendo" in url_lower:
        return "jamendo"
    if "soundcloud" in url_lower:
        return "soundcloud"
    if "youtube" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    return "yt-dlp"


def _extract_and_save_license(
    name: str, dest_dir: Path, url: str
) -> Optional[LicenseInfo]:
    """
    Try to extract licence info from yt-dlp's info.json, classify it,
    save it to  library/<name>/license.json , and return the LicenseInfo.
    """
    try:
        source = _source_from_url(url)

        # yt-dlp writes <song>/full.info.json
        info_json = dest_dir / "full.info.json"
        if info_json.exists():
            lic = license_from_ytdlp_info(info_json, source=source)
        else:
            lic = classify_license(source)

        save_license(dest_dir, lic)
        return lic
    except Exception as e:
        log.debug("Could not extract/save licence for '%s': %s", name, e)
        return None


# ---------------------------------------------------------------------------
# Stem separation
# ---------------------------------------------------------------------------

def _separate_stems(
    song_name: str,
    wav_path: Path,
    progress_cb: Optional[ProgressCb] = None,
) -> Dict[str, Path]:
    """
    Run Demucs on wav_path; place stems directly in library/<song_name>/.

    Delegates to scripts.core.stems.separate_song_stems() so the correct
    venv Python is always used and logic is never duplicated.
    """
    print(f"🎛️   Separating stems for: {song_name}")

    def _report(p: float, m: str) -> None:
        print(f"   [{int(p*100):3d}%] {m}")
        if progress_cb:
            try:
                progress_cb(p, m)
            except Exception:
                pass

    try:
        from scripts.core.stems import separate_song_stems
        result = separate_song_stems(
            song_name,
            enhance=False,       # download pipeline: skip enhance, just split
            progress_cb=_report,
        )
        if result.success:
            for stem in result.stems:
                print(f"   ✅ {stem}.wav")
            return result.stems
        else:
            print(f"❌  Stem separation failed: {result.error}")
            return {}
    except Exception as e:
        print(f"❌  Stem separation error: {e}")
        return {}


def _collect_stems(song_name: str) -> Dict[str, Path]:
    """Return any stems already present in library/<song_name>/."""
    d = song_dir(song_name)
    stems = {}
    for stem in ("vocals", "drums", "bass", "other"):
        p = d / f"{stem}.wav"
        if p.exists():
            stems[stem] = p
    return stems


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_summary(results: List[DownloadResult]) -> None:
    passed = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    print(f"\n{'='*55}")
    print(f"✅  {len(passed)}/{len(results)} tracks downloaded successfully")
    if failed:
        print(f"❌  {len(failed)} failed:")
        for r in failed:
            print(f"     • {r.name}: {r.error}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.download",
        description="AI RemixMate — download songs from YouTube Music, YouTube, or any URL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search and download a single track
  python -m scripts.download "Eric Prydz Opus" --split

  # Download from a URL with a custom name
  python -m scripts.download "https://music.youtube.com/watch?v=XXX" --name "Opus"

  # Download an entire playlist
  python -m scripts.download "https://www.youtube.com/playlist?list=PL..." --playlist

  # Batch download from a CSV file
  python -m scripts.download --csv songs.csv --split

  # Search and show top results (no download)
  python -m scripts.download "Deadmau5 Strobe" --search-only
""",
    )
    p.add_argument("query", nargs="?", help="Search query, URL, or YouTube video ID")
    p.add_argument("--name", help="Custom output name (no extension)")
    p.add_argument("--split", action="store_true", help="Separate into stems with Demucs")
    p.add_argument("--playlist", action="store_true",
                   help="Treat query as a playlist URL and download all tracks")
    p.add_argument("--csv", metavar="FILE",
                   help="CSV file with columns: title[, url]")
    p.add_argument("--search-only", action="store_true",
                   help="Show top search results without downloading")
    p.add_argument("--limit", type=int, default=5,
                   help="Number of search results to show with --search-only (default: 5)")
    p.add_argument(
        "--source",
        choices=["auto", "youtube", "jamendo", "all"],
        default="auto",
        help=(
            "Music source:  auto (YouTube Music → YouTube, default) | "
            "youtube (YouTube only) | "
            "jamendo (CC-licensed music, fully legal) | "
            "all (Jamendo + YouTube combined)"
        ),
    )
    return p


def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    if not _check_deps():
        return 1

    # ── Search-only mode ──────────────────────────────────────────────────
    if args.search_only:
        if not args.query:
            print("❌  Provide a query with --search-only.")
            return 1
        hits = search_track(args.query, limit=args.limit, source=args.source)
        if not hits:
            print("No results found.")
            return 0
        print(f"\n🔍  Top {len(hits)} results for: {args.query}  [source: {args.source}]\n")
        for i, h in enumerate(hits, 1):
            dur_raw = h.get("duration")
            if dur_raw and isinstance(dur_raw, (int, float)):
                dur = f"{int(dur_raw)//60}:{int(dur_raw)%60:02d}"
            else:
                dur = "?:??"
            artist  = h.get("artist") or "Unknown"
            src_tag = f"[{h.get('source', '?')}]"
            license_ = f"  📜 {h['license']}" if h.get("license") else ""
            print(f"  {i}. {h['title']} — {artist}  [{dur}]  {src_tag}{license_}")
            print(f"     {h['url']}")
        return 0

    # ── CSV batch mode ────────────────────────────────────────────────────
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"❌  CSV not found: {csv_path}")
            return 1
        download_from_csv(csv_path, separate=args.split)
        return 0

    # ── Require a query for everything else ───────────────────────────────
    if not args.query:
        parser.print_help()
        return 1

    # ── Playlist mode ─────────────────────────────────────────────────────
    if args.playlist:
        download_playlist(args.query, separate=args.split)
        return 0

    # ── Single track ──────────────────────────────────────────────────────
    spec   = TrackSpec(query=args.query, name=args.name, separate=args.split)
    result = download_track(spec)

    if result.success:
        print(f"\n🎉  Done!  Library: library/{result.name}/")
        if result.stems:
            print("   Stems:", ", ".join(result.stems.keys()))
    else:
        print(f"\n❌  Download failed: {result.error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
