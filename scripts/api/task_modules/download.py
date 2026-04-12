"""
scripts/api/task_modules/download.py — Track download task functions.

task_download          — download a single track with automatic Demucs
task_playlist_download — download all tracks from a playlist
"""

from pathlib import Path
from typing import Any, Dict, Optional

from scripts.api.jobs import update_job
from scripts.core.audit import log_audit
from scripts.core.logging_utils import get_logger

_log = get_logger(__name__)


def _index_upsert(song_name: str) -> None:
    """Non-blocking helper: upsert a song into the RAG music index."""
    try:
        from scripts.core.music_index import get_index
        get_index().upsert_song(song_name)
    except Exception as exc:
        _log.warning(f"music_index.upsert_song('{song_name}') failed (non-critical): {exc}")


def task_download(
    job_id: str,
    query: str,
    name: Optional[str] = None,
    separate: bool = True,   # always True now — Demucs is automatic
) -> Dict[str, Any]:
    from scripts.download import download_track, TrackSpec

    log_audit("download_start", resource=query, job_id=job_id, metadata={"separate": separate})
    update_job(job_id, progress=0.05, message="Starting download…")

    # Force separate=True — every download gets Demucs automatically
    spec = TrackSpec(query=query, name=name, separate=True)
    result = download_track(spec)

    if not result.success:
        log_audit("download_failed", resource=query, job_id=job_id, metadata={"error": result.error})
        raise RuntimeError(result.error or "Download failed")

    update_job(job_id, progress=0.75, message="Stems ready — indexing song…")

    # ── Upsert into music index (non-blocking) ─────────────────────────────
    _index_upsert(result.name)

    update_job(job_id, progress=0.95, message="Finalising…")

    out: Dict[str, Any] = {
        "name":    result.name,
        "wav":     str(result.wav) if result.wav else None,
        "stems":   {k: str(v) for k, v in result.stems.items()},
        "indexed": True,
        "success": True,
    }
    if result.license_warning:
        out["license_warning"] = result.license_warning

    log_audit("download_complete", resource=result.name, job_id=job_id,
              metadata={"query": query, "stems": list(result.stems.keys())})
    return out


def task_playlist_download(
    job_id: str,
    url: str,
    separate: bool,
    limit: Optional[int],
) -> Dict[str, Any]:
    """
    Download all tracks from a playlist URL.
    Reports per-track progress back to the job store.
    """
    from scripts.download import _sanitize, TrackSpec, download_track, DownloadResult

    update_job(job_id, progress=0.02, message="Fetching playlist info…")

    # ── Extract playlist entries via yt-dlp (no download yet) ────────────
    try:
        import yt_dlp  # type: ignore
        info_opts = {
            "quiet": True,
            "extract_flat": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise RuntimeError(f"Could not fetch playlist: {e}")

    entries = info.get("entries") or []
    if not entries:
        raise RuntimeError("Playlist appears to be empty or is not accessible.")

    playlist_title = info.get("title") or "Playlist"

    # Apply optional track limit
    if limit:
        entries = entries[:limit]

    total = len(entries)
    update_job(job_id, progress=0.05, message=f"Found {total} tracks in '{playlist_title}'")

    results = []
    failed  = []

    for i, entry in enumerate(entries):
        vid       = entry.get("id") or entry.get("url", "")
        title     = _sanitize(entry.get("title") or f"track_{i+1:03d}")
        track_url = (
            entry.get("webpage_url")
            or entry.get("url")
            or f"https://www.youtube.com/watch?v={vid}"
        )

        progress = 0.05 + 0.93 * (i / total)
        update_job(
            job_id,
            progress=round(progress, 3),
            message=f"[{i+1}/{total}] Downloading: {title[:40]}…",
        )

        spec   = TrackSpec(query=track_url, name=title, separate=separate)
        result = download_track(spec)

        if result.success:
            results.append({
                "name":    result.name,
                "wav":     str(result.wav) if result.wav else None,
                "stems":   {k: str(v) for k, v in result.stems.items()},
                "success": True,
            })
        else:
            failed.append({"name": title, "error": result.error})

    update_job(job_id, progress=0.99, message=f"Done — {len(results)}/{total} tracks saved")

    return {
        "playlist_title": playlist_title,
        "total":          total,
        "downloaded":     len(results),
        "failed":         len(failed),
        "tracks":         results,
        "errors":         failed,
        "success":        True,
    }
