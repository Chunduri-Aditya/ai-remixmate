#!/usr/bin/env python3
"""
scripts/download.py — Operator catalogue-ingest tool for AI RemixMate.

NOT a user-facing endpoint. The hosted product serves one shared, operator-curated
library of Creative-Commons / public-domain music. This module fetches a track
from a permitted source, verifies its licence, separates stems, analyses it, and
registers it in library/.

Permitted sources (both the licence *and* the platform's terms allow download +
derivatives):
  • ccMixter          — remix-first CC community            (implemented)
  • Internet Archive   — netlabel / live-music collections   (implemented)
  • Free Music Archive — CC catalogue                        (stub — API unverified)
  • Musopen            — public-domain classical             (stub — API unverified)

Every ingest runs classify_license -> is_publishable gate -> Demucs -> analysis.
The gate rejects non-CC/PD tracks only when REMIXMATE_LICENSE_ENFORCE=1 (set it
for real ingest; off by default so a local dev library still loads).

Usage:
  python -m scripts.download --source ccmixter "glitch hop" --search-only
  REMIXMATE_LICENSE_ENFORCE=1 python -m scripts.download --source ccmixter "glitch hop" --pick 2
  python -m scripts.download --source archive --url <download_url> --name "Artist - Title"
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from scripts.core.library import get_library_manager
from scripts.core.license import (
    LicenseInfo,
    classify_license,
    is_publishable,
    license_enforce_enabled,
    license_warning,
    load_license,
    save_license,
)
from scripts.core.paths import ensure_directories, sanitize_song_name, song_dir

log = logging.getLogger(__name__)

# Progress callback signature: (fraction_0_to_1, human_readable_message)
ProgressCb = Callable[[float, str], None]

_HTTP_TIMEOUT = 30  # seconds — catalogue API calls and audio downloads


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CatalogCandidate:
    """One track offered by a source fetcher."""

    title: str
    artist: Optional[str]
    audio_url: str
    source: str                        # "ccmixter" | "archive" | "fma" | "musopen"
    license_url: Optional[str] = None
    license_str: Optional[str] = None
    source_url: Optional[str] = None   # the track's page on its origin site
    duration: Optional[float] = None

    def suggested_name(self) -> str:
        base = f"{self.artist} - {self.title}" if self.artist else self.title
        return sanitize_song_name(base)


@dataclass
class IngestResult:
    name: str
    success: bool
    wav: Optional[Path] = None
    stems: Dict[str, Path] = field(default_factory=dict)
    license: Optional[LicenseInfo] = None
    error: Optional[str] = None
    analyzed: bool = False
    rejected_license: Optional[str] = None   # licence_type string when the gate blocked it


# ---------------------------------------------------------------------------
# Source fetchers  ->  list[CatalogCandidate]
# ---------------------------------------------------------------------------

def _requests():
    import requests  # already used across the repo; kept a lazy import here
    return requests


def fetch_ccmixter(query: str, limit: int = 10) -> List[CatalogCandidate]:
    """
    ccMixter query API (https://ccmixter.org/query-api). Returns uploads matching
    `query`, highest rank first. Verify field names against the live API when
    building the catalogue — ccMixter's JSON has drifted historically.
    """
    req = _requests()
    r = req.get(
        "https://ccmixter.org/api/query",
        params={"f": "json", "search": query, "search_type": "all", "limit": limit, "sort": "rank"},
        timeout=_HTTP_TIMEOUT,
    )
    r.raise_for_status()
    out: List[CatalogCandidate] = []
    for up in r.json() or []:
        files = up.get("files") or []
        audio = next(
            (
                f for f in files
                if str(f.get("file_format_info", {}).get("mime", "")).startswith("audio")
                or str(f.get("download_url", "")).lower().endswith((".mp3", ".flac", ".ogg", ".wav"))
            ),
            None,
        )
        if not audio:
            continue
        out.append(CatalogCandidate(
            title=up.get("upload_name") or query,
            artist=up.get("user_real_name") or up.get("user_name"),
            audio_url=audio["download_url"],
            source="ccmixter",
            license_url=up.get("license_url"),
            license_str=up.get("license_name"),
            source_url=up.get("file_page"),
        ))
    return out


_IA_AUDIO_FORMATS = {"VBR MP3", "MP3", "128Kbps MP3", "FLAC", "Ogg Vorbis", "AIFF"}


def fetch_internet_archive(query: str, limit: int = 10) -> List[CatalogCandidate]:
    """
    Internet Archive: advancedsearch for audio items, then per-item metadata to
    resolve a concrete audio file. `query` may be a plain search or a bare
    item identifier.
    """
    req = _requests()

    if " " not in query and re.fullmatch(r"[A-Za-z0-9._-]{3,120}", query):
        docs: List[dict] = [{"identifier": query}]
    else:
        r = req.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": f"({query}) AND mediatype:audio",
                "fl[]": ["identifier", "title", "creator", "licenseurl"],
                "rows": limit,
                "output": "json",
            },
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        docs = r.json().get("response", {}).get("docs", [])

    out: List[CatalogCandidate] = []
    for doc in docs[:limit]:
        ident = doc["identifier"]
        m = req.get(f"https://archive.org/metadata/{ident}", timeout=_HTTP_TIMEOUT)
        m.raise_for_status()
        meta = m.json()
        md = meta.get("metadata", {})
        file_obj = next(
            (f for f in meta.get("files", []) if f.get("format") in _IA_AUDIO_FORMATS), None
        )
        if not file_obj:
            continue
        creator = md.get("creator") or doc.get("creator")
        if isinstance(creator, list):
            creator = creator[0] if creator else None
        out.append(CatalogCandidate(
            title=md.get("title") or doc.get("title") or ident,
            artist=creator,
            audio_url=f"https://archive.org/download/{ident}/{file_obj['name']}",
            source="archive",
            license_url=md.get("licenseurl") or doc.get("licenseurl"),
            source_url=f"https://archive.org/details/{ident}",
        ))
    return out


def fetch_fma(query: str, limit: int = 10) -> List[CatalogCandidate]:
    raise NotImplementedError(
        "Free Music Archive fetcher not implemented — verify the current FMA API "
        "contract during catalogue build, then wire it here."
    )


def fetch_musopen(query: str, limit: int = 10) -> List[CatalogCandidate]:
    raise NotImplementedError(
        "Musopen fetcher not implemented — verify the Musopen API / download "
        "contract during catalogue build, then wire it here."
    )


_FETCHERS: Dict[str, Callable[[str, int], List[CatalogCandidate]]] = {
    "ccmixter": fetch_ccmixter,
    "archive": fetch_internet_archive,
    "internetarchive": fetch_internet_archive,
    "fma": fetch_fma,
    "musopen": fetch_musopen,
}


def search_catalog(source: str, query: str, limit: int = 10) -> List[CatalogCandidate]:
    try:
        fetcher = _FETCHERS[source.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown source {source!r}. Options: {', '.join(sorted(_FETCHERS))}"
        )
    return fetcher(query, limit)


# ---------------------------------------------------------------------------
# Audio download  (requests stream -> ffmpeg -> full.wav)
# ---------------------------------------------------------------------------

def _download_audio(url: str, dest_dir: Path, progress_cb: Optional[ProgressCb] = None) -> Path:
    """Stream `url` to a temp file and transcode to dest_dir/full.wav (mono 44.1k)."""
    req = _requests()
    dest_dir.mkdir(parents=True, exist_ok=True)
    wav_path = dest_dir / "full.wav"
    suffix = Path(url.split("?", 1)[0]).suffix.lower() or ".audio"
    tmp = dest_dir / f"_download{suffix}"
    with req.get(url, stream=True, timeout=_HTTP_TIMEOUT) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                done += len(chunk)
                if progress_cb and total:
                    progress_cb(0.05 + 0.25 * done / total, f"Downloading… {done * 100 // total}%")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp), "-ac", "1", "-ar", "44100", str(wav_path)],
        check=True, capture_output=True,
    )
    tmp.unlink(missing_ok=True)
    return wav_path


# ---------------------------------------------------------------------------
# Stem separation  (delegates to scripts.core.stems for venv / logic parity)
# ---------------------------------------------------------------------------

try:
    from scripts.core.config import cfg as _cfg_dl
    _PRUNE_ON_DL: bool = getattr(getattr(_cfg_dl, "library", None), "prune_on_download", True)
except Exception:
    _PRUNE_ON_DL = True


def _separate_stems(
    song_name: str, wav_path: Path, progress_cb: Optional[ProgressCb] = None
) -> Dict[str, Path]:
    def _report(p: float, m: str) -> None:
        if progress_cb:
            try:
                progress_cb(p, m)
            except Exception:
                pass

    try:
        from scripts.core.stems import separate_song_stems
        result = separate_song_stems(song_name, enhance=False, progress_cb=_report)
        if result.success:
            return result.stems
        log.warning("Stem separation failed for %r: %s", song_name, result.error)
        return {}
    except Exception as exc:
        log.warning("Stem separation error for %r: %s", song_name, exc)
        return {}


def _collect_stems(song_name: str) -> Dict[str, Path]:
    d = song_dir(song_name)
    return {s: d / f"{s}.wav" for s in ("vocals", "drums", "bass", "other") if (d / f"{s}.wav").exists()}


def _has_stems(song_dir_path: Path) -> bool:
    return any(
        (song_dir_path / f"{s}.{ext}").exists()
        for s in ("vocals", "drums", "bass", "other")
        for ext in ("flac", "wav")
    )


# ---------------------------------------------------------------------------
# Ingest pipeline
# ---------------------------------------------------------------------------

def ingest_track(
    candidate: CatalogCandidate,
    name: Optional[str] = None,
    separate: bool = True,
    auto_analyze: bool = True,
    progress_cb: Optional[ProgressCb] = None,
) -> IngestResult:
    """
    Fetch -> licence-classify -> publish gate -> stems -> analyse -> register.

    The gate (reject anything is_publishable() == False) is enforced only when
    license_enforce_enabled(). On rejection nothing is downloaded and
    IngestResult.success is False with rejected_license set.
    """
    ensure_directories()
    name = name or candidate.suggested_name()
    dest_dir = song_dir(name)

    def _emit(frac: float, msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(min(max(frac, 0.0), 1.0), msg)
            except Exception:
                pass

    if (dest_dir / "full.wav").exists() or _has_stems(dest_dir):
        _emit(1.0, f"Already in library: {name}")
        return IngestResult(
            name=name, success=True, wav=dest_dir / "full.wav",
            stems=_collect_stems(name), license=load_license(dest_dir),
        )

    # 1. classify the licence first so a rejection costs no download
    info = classify_license(
        candidate.source, license_url=candidate.license_url, license_str=candidate.license_str
    )
    info.artist_name = candidate.artist
    info.source_url = candidate.source_url

    if license_enforce_enabled() and not is_publishable(info):
        log.warning("Rejected %r — licence %s is not publishable", name, info.license_type.value)
        return IngestResult(
            name=name, success=False, rejected_license=info.license_type.value,
            error=f"licence {info.license_type.value} not permitted in the catalogue",
        )

    # 2. download audio
    _emit(0.05, f"Fetching {name}…")
    try:
        wav = _download_audio(candidate.audio_url, dest_dir, progress_cb=_emit)
    except Exception as exc:
        shutil.rmtree(dest_dir, ignore_errors=True)
        return IngestResult(name=name, success=False, error=f"download failed: {exc}")

    # 3. persist licence
    save_license(dest_dir, info)

    # 4. stems
    stems: Dict[str, Path] = {}
    if separate:
        _emit(0.35, "Separating stems…")
        stems = _separate_stems(name, wav, progress_cb=lambda p, m: _emit(0.35 + 0.5 * p, m))

    # 5. analysis — needs full.wav, so before any prune
    analyzed = False
    if stems and auto_analyze:
        _emit(0.88, "Analysing…")
        try:
            from scripts.core.analysis_pipeline import run_song_analysis
            run_song_analysis(name)
            analyzed = True
        except Exception as exc:
            log.warning("Analysis failed for %r (non-fatal): %s", name, exc)

    # 6. prune raw/enhanced WAV once stems exist, then register
    if stems and _PRUNE_ON_DL:
        try:
            mgr = get_library_manager()
            mgr.prune_raw(name)
            mgr.prune_enhanced(name)
        except Exception:
            pass

    _emit(0.96, "Registering…")
    try:
        mgr = get_library_manager()
        mgr.register(name, source=candidate.source)
        mgr.store_fingerprint(name, wav if wav.exists() else dest_dir / "vocals.wav")
    except Exception:
        pass

    warn = license_warning(info, name)
    if warn:
        log.info("%s", warn)
    _emit(1.0, f"Done: {name}")
    return IngestResult(
        name=name, success=True, wav=wav if wav.exists() else None,
        stems=stems, license=info, analyzed=analyzed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.download",
        description="Operator catalogue-ingest tool (CC / public-domain sources only)",
    )
    p.add_argument("query", nargs="?", help="search query, or a bare source id (Internet Archive)")
    p.add_argument("--source", required=True, choices=sorted(_FETCHERS), help="catalogue source")
    p.add_argument("--url", help="direct audio download URL (skip search; requires --name)")
    p.add_argument("--name", help="library name (default: 'Artist - Title')")
    p.add_argument("--no-split", action="store_true", help="skip Demucs stem separation")
    p.add_argument("--limit", type=int, default=10, help="max search results (default 10)")
    p.add_argument("--search-only", action="store_true", help="list candidates, do not ingest")
    p.add_argument("--pick", type=int, default=1, metavar="N",
                   help="ingest the Nth search result (default 1)")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg is required.", file=sys.stderr)
        return 1
    if not license_enforce_enabled():
        log.info("note: REMIXMATE_LICENSE_ENFORCE is not set — the publish gate is advisory only")

    if args.url:
        if not args.name:
            print("--url requires --name", file=sys.stderr)
            return 1
        cand = CatalogCandidate(title=args.name, artist=None, audio_url=args.url, source=args.source)
        res = ingest_track(cand, name=args.name, separate=not args.no_split)
        print(f"ingested: library/{res.name}/" if res.success else f"failed: {res.error}")
        return 0 if res.success else 1

    if not args.query:
        print("provide a search query or --url", file=sys.stderr)
        return 1

    cands = search_catalog(args.source, args.query, limit=args.limit)
    if not cands:
        print("no results")
        return 0

    if args.search_only:
        for i, c in enumerate(cands, 1):
            lic = c.license_url or c.license_str or "licence?"
            print(f"{i:2d}. {c.artist or '?'} — {c.title}  [{lic}]")
            print(f"    {c.audio_url}")
        return 0

    idx = max(1, args.pick) - 1
    if idx >= len(cands):
        print(f"--pick {args.pick} out of range (got {len(cands)} results)", file=sys.stderr)
        return 1
    res = ingest_track(cands[idx], name=args.name, separate=not args.no_split)
    if res.success:
        print(f"ingested: library/{res.name}/")
        return 0
    print(f"failed: {res.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
