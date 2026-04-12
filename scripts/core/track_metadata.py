"""
scripts/core/track_metadata.py — Metadata-first architecture for AI RemixMate.

This module implements the "query before you download" principle:
  1. Look up BPM, key, genre, energy from remote APIs (instant, no audio needed)
  2. Cache everything locally in SQLite (query once, reuse forever)
  3. Fall back to local librosa analysis if all APIs fail
  4. Return a clean TrackMetadata object — single source of truth for mixing decisions

Provider priority order:
  Cache → GetSongBPM → Last.fm (genres) → MusicBrainz (IDs) → Librosa (local fallback)

Usage:
  from scripts.core.track_metadata import MetadataClient, TrackMetadata

  client = MetadataClient()

  # Before downloading — get features from API
  meta = client.lookup("Bicep Glue")
  print(meta.bpm, meta.key, meta.genres)

  # After downloading — enrich with local analysis
  meta = client.enrich_from_file(meta, Path("library/Bicep - Glue/full.wav"))
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

try:
    from scripts.core.config import cfg as _cfg
    _GETSONGBPM_KEY: str = getattr(getattr(_cfg, "metadata", None), "getsongbpm_api_key", "")
    _LASTFM_KEY: str     = getattr(getattr(_cfg, "metadata", None), "lastfm_api_key", "")
    _CACHE_TTL: int      = getattr(getattr(_cfg, "metadata", None), "cache_ttl_days", 30) * 86400
    _CACHE_PATH: str     = getattr(getattr(_cfg, "metadata", None), "cache_path", "data/metadata.db")
except Exception:
    _GETSONGBPM_KEY = ""
    _LASTFM_KEY     = ""
    _CACHE_TTL      = 30 * 86400
    _CACHE_PATH     = "data/metadata.db"

try:
    from scripts.core.paths import PROJECT_ROOT
    _DB_PATH = PROJECT_ROOT / _CACHE_PATH
except Exception:
    _DB_PATH = Path(_CACHE_PATH)


# ---------------------------------------------------------------------------
# TrackMetadata — the single source of truth
# ---------------------------------------------------------------------------

@dataclass
class TrackMetadata:
    """
    Normalised metadata for one track.  Every field has a sensible default
    so callers never need to check for None on the hot path.
    """
    # Identity
    title: str                        = ""
    artist: str                       = ""
    album: str                        = ""
    year: Optional[int]               = None
    isrc: Optional[str]               = None
    mb_id: Optional[str]              = None        # MusicBrainz recording ID

    # Audio features — the values the mixer actually uses
    bpm: float                        = 0.0
    bpm_confidence: float             = 0.0         # 0–1
    key: str                          = ""          # e.g. "C"
    mode: str                         = ""          # "major" | "minor"
    camelot: Optional[str]            = None        # e.g. "8A", "5B"
    energy: float                     = 0.5         # 0–1 loudness/intensity
    danceability: float               = 0.5         # 0–1
    valence: float                    = 0.5         # 0–1 (happy↔sad)
    loudness_lufs: float              = -14.0       # integrated loudness
    duration_seconds: float           = 0.0

    # Genre
    genres: List[str]                 = field(default_factory=list)

    # Provenance
    source: str                       = "unknown"   # which provider filled this
    last_updated: float               = 0.0         # Unix timestamp

    # ── Derived helpers ──────────────────────────────────────────────────

    @property
    def key_full(self) -> str:
        """Return e.g. 'C major' or 'A minor'."""
        parts = [p for p in (self.key, self.mode) if p]
        return " ".join(parts)

    @property
    def is_complete(self) -> bool:
        """True if both BPM and key are known."""
        return self.bpm > 0 and bool(self.key)

    def merge(self, other: "TrackMetadata") -> "TrackMetadata":
        """
        Return a new TrackMetadata that prefers self's values but fills in
        missing fields from other.  Used to merge multiple provider results.
        """
        result = TrackMetadata(**asdict(self))
        for f_name, f_val in asdict(other).items():
            current = getattr(result, f_name)
            # Fill only if current is empty/zero/unknown
            if f_name in ("bpm", "energy", "danceability", "valence", "duration_seconds"):
                if current == 0.0 and f_val != 0.0:
                    setattr(result, f_name, f_val)
            elif f_name in ("key", "mode", "camelot", "isrc", "mb_id",
                            "title", "artist", "album"):
                if not current and f_val:
                    setattr(result, f_name, f_val)
            elif f_name == "genres":
                if not current and f_val:
                    setattr(result, f_name, f_val)
        return result


# ---------------------------------------------------------------------------
# Camelot wheel helper
# ---------------------------------------------------------------------------

# Maps (key, mode) → Camelot position
# key is a pitch class string ("C", "C#", "Db", "D" …)
# mode is "major" or "minor"
_CAMELOT: Dict[Tuple[str, str], str] = {
    ("B",  "major"): "1B",  ("Ab", "minor"): "1A",
    ("F#", "major"): "2B",  ("Eb", "minor"): "2A",
    ("Db", "major"): "3B",  ("Bb", "minor"): "3A",
    ("Ab", "major"): "4B",  ("F",  "minor"): "4A",
    ("Eb", "major"): "5B",  ("C",  "minor"): "5A",
    ("Bb", "major"): "6B",  ("G",  "minor"): "6A",
    ("F",  "major"): "7B",  ("D",  "minor"): "7A",
    ("C",  "major"): "8B",  ("A",  "minor"): "8A",
    ("G",  "major"): "9B",  ("E",  "minor"): "9A",
    ("D",  "major"): "10B", ("B",  "minor"): "10A",
    ("A",  "major"): "11B", ("F#", "minor"): "11A",
    ("E",  "major"): "12B", ("Db", "minor"): "12A",
    # Enharmonic aliases
    ("C#", "major"): "3B",  ("G#", "minor"): "1A",
    ("G#", "major"): "4B",  ("D#", "minor"): "2A",
    ("D#", "major"): "5B",  ("A#", "minor"): "3A",
    ("A#", "major"): "6B",
    ("Gb", "major"): "2B",  ("Cb", "minor"): "12A",
    ("Cb", "major"): "1B",
}


def key_to_camelot(key: str, mode: str) -> Optional[str]:
    """Convert a standard key + mode to a Camelot wheel position."""
    return _CAMELOT.get((key.strip().title(), mode.strip().lower()))


def camelot_distance(a: str, b: str) -> int:
    """
    Return the harmonic distance between two Camelot positions (0 = same key).
    0 = perfect match, 1 = adjacent (one semitone shift), 6 = tritone (worst).
    """
    if a == b:
        return 0
    def _parse(c: str) -> Tuple[int, str]:
        m = re.match(r"(\d{1,2})([AB])", c.upper())
        if not m:
            return 0, "B"
        return int(m.group(1)), m.group(2)

    num_a, side_a = _parse(a)
    num_b, side_b = _parse(b)

    if side_a != side_b:
        return 999  # different modes — very incompatible
    diff = abs(num_a - num_b)
    return min(diff, 12 - diff)  # circular distance on 12-position wheel


# ---------------------------------------------------------------------------
# Metadata Cache (SQLite)
# ---------------------------------------------------------------------------

class MetadataCache:
    """SQLite-backed cache for TrackMetadata.  TTL defaults to 30 days."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS metadata (
        cache_key    TEXT PRIMARY KEY,
        data         TEXT NOT NULL,
        updated_at   REAL NOT NULL
    );
    """

    def __init__(self, db_path: Path = _DB_PATH, ttl: int = _CACHE_TTL) -> None:
        self.db_path = db_path
        self.ttl = ttl
        self._ensure_db()

    def _ensure_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(self.SCHEMA)

    def _key(self, query: str) -> str:
        return query.lower().strip()

    def get(self, query: str) -> Optional[TrackMetadata]:
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                row = conn.execute(
                    "SELECT data, updated_at FROM metadata WHERE cache_key = ?",
                    (self._key(query),),
                ).fetchone()
            if row is None:
                return None
            data_str, updated_at = row
            if time.time() - updated_at > self.ttl:
                return None  # stale
            data = json.loads(data_str)
            return TrackMetadata(**data)
        except Exception as e:
            log.debug("Cache read failed for '%s': %s", query, e)
            return None

    def put(self, query: str, meta: TrackMetadata) -> None:
        try:
            meta.last_updated = time.time()
            data_str = json.dumps(asdict(meta))
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (cache_key, data, updated_at) "
                    "VALUES (?, ?, ?)",
                    (self._key(query), data_str, time.time()),
                )
        except Exception as e:
            log.debug("Cache write failed for '%s': %s", query, e)

    def invalidate(self, query: str) -> None:
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("DELETE FROM metadata WHERE cache_key = ?",
                             (self._key(query),))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

_USER_AGENT = "AI-RemixMate/0.2 (portfolio project; github.com/ai-remixmate)"


def _get_json(url: str, timeout: int = 8) -> Optional[dict]:
    """Simple GET → JSON with a custom User-Agent.  Returns None on any error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.debug("HTTP GET failed [%s]: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# Provider: GetSongBPM
# ---------------------------------------------------------------------------

class GetSongBPMProvider:
    """
    Queries https://getsongbpm.com/api for BPM and musical key.

    Free tier: 100 requests/day.  Register at https://getsongbpm.com/api
    and add your key to config:  metadata.getsongbpm_api_key

    Without a key this provider gracefully returns None (other providers
    or local analysis will fill in the gap).
    """

    BASE = "https://api.getsongbpm.com"

    def __init__(self, api_key: str = _GETSONGBPM_KEY) -> None:
        self.api_key = api_key

    def _has_key(self) -> bool:
        return bool(self.api_key and self.api_key not in ("", "YOUR_KEY_HERE"))

    def lookup(self, title: str, artist: str = "") -> Optional[TrackMetadata]:
        if not self._has_key():
            log.debug("GetSongBPM: no API key configured — skipping")
            return None

        query = f"{title} {artist}".strip()
        encoded = urllib.parse.quote(query)
        url = f"{self.BASE}/search/?api_key={self.api_key}&type=song&lookup={encoded}"

        data = _get_json(url)
        if not data:
            return None

        songs = data.get("search", [])
        if not songs:
            log.debug("GetSongBPM: no results for '%s'", query)
            return None

        # Take the first result
        song = songs[0]
        raw_key  = song.get("key_of_song") or song.get("key") or ""
        raw_bpm  = song.get("tempo") or 0
        raw_art  = (song.get("artist") or {}).get("name") or artist
        raw_title = song.get("song") or title

        # Parse key — GetSongBPM returns e.g. "C Major", "A Minor", "F# Minor"
        key_str, mode_str = _parse_key_string(raw_key)

        meta = TrackMetadata(
            title=raw_title,
            artist=raw_art,
            bpm=float(raw_bpm),
            bpm_confidence=0.85,        # API data is generally reliable
            key=key_str,
            mode=mode_str,
            camelot=key_to_camelot(key_str, mode_str),
            source="getsongbpm",
        )
        log.info("GetSongBPM: %s — %s BPM, key %s %s",
                 raw_title, raw_bpm, key_str, mode_str)
        return meta


def _parse_key_string(raw: str) -> Tuple[str, str]:
    """
    Parse a key string like "C Major", "A Minor", "F# Minor" into
    (key, mode) tuple.  Returns ("", "") on failure.
    """
    if not raw:
        return "", ""
    parts = raw.strip().split()
    if len(parts) < 2:
        return parts[0].title() if parts else "", ""
    key  = parts[0].title()
    mode = parts[-1].lower()
    if mode not in ("major", "minor"):
        mode = "major"
    return key, mode


# ---------------------------------------------------------------------------
# Provider: Last.fm
# ---------------------------------------------------------------------------

class LastFmProvider:
    """
    Queries the Last.fm API for genre tags and basic metadata.

    Free API key: register at https://www.last.fm/api/account/create
    Set in config:  metadata.lastfm_api_key

    Returns genre tags, playcount, duration.  No BPM or key data.
    """

    BASE = "https://ws.audioscrobbler.com/2.0/"

    def __init__(self, api_key: str = _LASTFM_KEY) -> None:
        self.api_key = api_key

    def _has_key(self) -> bool:
        return bool(self.api_key and self.api_key not in ("", "YOUR_KEY_HERE"))

    def lookup(self, title: str, artist: str = "") -> Optional[TrackMetadata]:
        if not self._has_key():
            log.debug("Last.fm: no API key configured — skipping")
            return None
        if not artist:
            log.debug("Last.fm: artist required for lookup, skipping")
            return None

        params = {
            "method": "track.getInfo",
            "api_key": self.api_key,
            "artist": artist,
            "track": title,
            "format": "json",
            "autocorrect": "1",
        }
        url = self.BASE + "?" + urllib.parse.urlencode(params)
        data = _get_json(url)
        if not data or "error" in data:
            return None

        track = data.get("track", {})
        if not track:
            return None

        # Extract genre tags (top 3, skip generic "seen live" type tags)
        _skip_tags = {"seen live", "favorites", "favourite", "love", "awesome"}
        tags = [
            t["name"].lower()
            for t in (track.get("toptags") or {}).get("tag", [])
            if t.get("name", "").lower() not in _skip_tags
        ][:5]

        duration_ms = int(track.get("duration") or 0)

        meta = TrackMetadata(
            title=track.get("name") or title,
            artist=(track.get("artist") or {}).get("name") or artist,
            genres=tags,
            duration_seconds=duration_ms / 1000.0 if duration_ms else 0.0,
            source="lastfm",
        )
        log.info("Last.fm: %s — genres: %s", meta.title, meta.genres)
        return meta


# ---------------------------------------------------------------------------
# Provider: MusicBrainz
# ---------------------------------------------------------------------------

class MusicBrainzProvider:
    """
    Queries the MusicBrainz API for canonical song identity and ISRC codes.

    Fully free, no API key required.  Rate limited to ~1 req/s.
    Returns IDs and basic metadata.  No BPM or key — use with other providers.
    """

    BASE = "https://musicbrainz.org/ws/2"

    def lookup(self, title: str, artist: str = "") -> Optional[TrackMetadata]:
        query = f'recording:"{title}"'
        if artist:
            query += f' AND artist:"{artist}"'
        encoded = urllib.parse.quote(query)
        url = f"{self.BASE}/recording/?query={encoded}&fmt=json&limit=1"

        data = _get_json(url)
        if not data:
            return None

        recordings = data.get("recordings", [])
        if not recordings:
            return None

        rec = recordings[0]
        artist_credit = rec.get("artist-credit", [])
        artist_name = artist_credit[0]["artist"]["name"] if artist_credit else artist

        # Extract ISRC from relations if present
        isrc_list = rec.get("isrcs", [])
        isrc = isrc_list[0] if isrc_list else None

        meta = TrackMetadata(
            title=rec.get("title") or title,
            artist=artist_name,
            duration_seconds=(rec.get("length") or 0) / 1000.0,
            mb_id=rec.get("id"),
            isrc=isrc,
            source="musicbrainz",
        )
        log.info("MusicBrainz: %s — MB ID %s", meta.title, meta.mb_id)
        return meta


# ---------------------------------------------------------------------------
# Provider: Local librosa fallback
# ---------------------------------------------------------------------------

class LocalAnalysisProvider:
    """
    Extract BPM and key directly from a local WAV file using librosa.
    This is the fallback when all APIs fail or the song isn't in any database.

    Requires the audio to already be downloaded.
    Returns a complete TrackMetadata with all available acoustic features.
    """

    def __init__(self, sr: int = 44100, analysis_duration: float = 60.0) -> None:
        self.sr = sr
        self.analysis_duration = analysis_duration

    def from_file(self, wav_path: Path, title: str = "", artist: str = "") -> Optional[TrackMetadata]:
        """Analyse a local WAV file and return TrackMetadata."""
        try:
            import librosa
            import numpy as np
        except ImportError:
            log.warning("librosa not available — cannot do local analysis")
            return None

        if not wav_path.exists():
            return None

        log.info("Local analysis: %s (first %.0fs)…", wav_path.name, self.analysis_duration)
        try:
            y, sr = librosa.load(str(wav_path), sr=self.sr, mono=True,
                                 duration=self.analysis_duration)

            # BPM
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo)

            # Key
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = chroma.mean(axis=1)
            pitch_classes = ["C", "C#", "D", "D#", "E", "F",
                             "F#", "G", "G#", "A", "A#", "B"]
            key_idx = int(chroma_mean.argmax())
            key = pitch_classes[key_idx]

            # Rough major/minor via Krumhansl profiles
            major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,
                                       2.52,5.19,2.39,3.66,2.29,2.88])
            minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,
                                       2.54,4.75,3.98,2.69,3.34,3.17])
            # Rotate profiles to match the detected key
            rot = np.roll(chroma_mean, -key_idx)
            major_score = float(np.corrcoef(rot, major_profile)[0, 1])
            minor_score = float(np.corrcoef(rot, minor_profile)[0, 1])
            mode = "major" if major_score >= minor_score else "minor"

            # Energy (RMS normalised to 0-1)
            rms = librosa.feature.rms(y=y)
            energy = float(np.clip(np.mean(rms) * 20, 0.0, 1.0))

            # Loudness approximation
            lufs = -14.0  # default; replace with pyloudnorm if available
            try:
                import pyloudnorm as pyln
                meter = pyln.Meter(sr)
                lufs = float(meter.integrated_loudness(y))
            except Exception:
                pass

            camelot = key_to_camelot(key, mode)

            meta = TrackMetadata(
                title=title or wav_path.parent.name,
                artist=artist,
                bpm=bpm,
                bpm_confidence=0.70,    # local analysis less reliable than API
                key=key,
                mode=mode,
                camelot=camelot,
                energy=energy,
                loudness_lufs=lufs,
                duration_seconds=float(len(y) / sr),
                source="librosa",
            )
            log.info("Local analysis: %s BPM, key %s %s, energy %.2f",
                     bpm, key, mode, energy)
            return meta

        except Exception as e:
            log.warning("Local analysis failed for %s: %s", wav_path, e)
            return None


# ---------------------------------------------------------------------------
# MetadataClient — the main entry point
# ---------------------------------------------------------------------------

class MetadataClient:
    """
    Unified metadata client.  Queries providers in order, merges results,
    and caches everything in SQLite.

    Usage
    -----
    client = MetadataClient()

    # Fast lookup (API only, no audio needed)
    meta = client.lookup("Bicep Glue", artist="Bicep")

    # Enrich from local audio after downloading
    meta = client.enrich_from_file(meta, Path("library/Bicep - Glue/full.wav"))

    # Check if two tracks are compatible for mixing
    compat = client.compatibility_score(meta_a, meta_b)
    """

    def __init__(
        self,
        getsongbpm_key: str = _GETSONGBPM_KEY,
        lastfm_key: str     = _LASTFM_KEY,
        cache_path: Path    = _DB_PATH,
        cache_ttl: int      = _CACHE_TTL,
    ) -> None:
        self.cache = MetadataCache(cache_path, cache_ttl)

        self._getsongbpm  = GetSongBPMProvider(getsongbpm_key)
        self._lastfm      = LastFmProvider(lastfm_key)
        self._musicbrainz = MusicBrainzProvider()
        self._local       = LocalAnalysisProvider()

    # ------------------------------------------------------------------
    # Primary lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        title: str,
        artist: str = "",
        force_refresh: bool = False,
    ) -> TrackMetadata:
        """
        Return TrackMetadata for a song.

        Tries cache first, then remote APIs, merges all results.
        Never raises — returns a partial TrackMetadata on failure.

        Parameters
        ----------
        title : str
            Song title (can include artist: "Bicep - Glue")
        artist : str, optional
            Artist name for more accurate lookups.
        force_refresh : bool
            Bypass cache and re-query all providers.
        """
        cache_key = f"{artist} {title}".strip().lower() if artist else title.lower()

        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached and cached.is_complete:
                log.debug("Cache hit for '%s'", cache_key)
                return cached

        # --- Try parsing "Artist - Title" format ---
        if " - " in title and not artist:
            parts = title.split(" - ", 1)
            artist = parts[0].strip()
            title  = parts[1].strip()

        log.info("Fetching metadata for: %s — %s", artist or "?", title)

        result = TrackMetadata(title=title, artist=artist)
        results: List[TrackMetadata] = []

        # --- GetSongBPM (BPM + key) ---
        try:
            bpm_meta = self._getsongbpm.lookup(title, artist)
            if bpm_meta:
                results.append(bpm_meta)
        except Exception as e:
            log.debug("GetSongBPM lookup error: %s", e)

        # --- Last.fm (genres + duration) ---
        try:
            lfm_meta = self._lastfm.lookup(title, artist)
            if lfm_meta:
                results.append(lfm_meta)
        except Exception as e:
            log.debug("Last.fm lookup error: %s", e)

        # --- MusicBrainz (canonical IDs, ISRC) — slower, do last ---
        if not any(r.mb_id for r in results):
            try:
                mb_meta = self._musicbrainz.lookup(title, artist)
                if mb_meta:
                    results.append(mb_meta)
            except Exception as e:
                log.debug("MusicBrainz lookup error: %s", e)

        # --- Merge all results ---
        for r in results:
            result = result.merge(r)

        # Ensure camelot is populated if we have key + mode
        if result.key and result.mode and not result.camelot:
            result.camelot = key_to_camelot(result.key, result.mode)

        # Store in cache
        if result.title or result.bpm:
            self.cache.put(cache_key, result)

        return result

    # ------------------------------------------------------------------
    # Local enrichment
    # ------------------------------------------------------------------

    def enrich_from_file(
        self,
        meta: TrackMetadata,
        wav_path: Path,
        overwrite_bpm: bool = False,
    ) -> TrackMetadata:
        """
        Run local librosa analysis on a downloaded WAV and merge the results
        into an existing TrackMetadata.

        Parameters
        ----------
        meta : TrackMetadata
            Existing metadata (from API or cache).
        wav_path : Path
            Path to the local audio file.
        overwrite_bpm : bool
            If True, replace existing BPM with locally-computed value.
            Defaults to False (trust API data over local analysis).
        """
        local = self._local.from_file(wav_path, title=meta.title, artist=meta.artist)
        if local is None:
            return meta

        enriched = meta.merge(local)

        # Optionally override BPM with local value
        if overwrite_bpm and local.bpm > 0:
            enriched.bpm = local.bpm
            enriched.bpm_confidence = local.bpm_confidence

        # Cache the enriched version
        cache_key = f"{meta.artist} {meta.title}".strip().lower()
        self.cache.put(cache_key, enriched)
        return enriched

    # ------------------------------------------------------------------
    # Compatibility scoring
    # ------------------------------------------------------------------

    def compatibility_score(
        self,
        meta_a: TrackMetadata,
        meta_b: TrackMetadata,
    ) -> Dict[str, float]:
        """
        Compute a compatibility score between two tracks for DJ mixing.

        Returns a dict with individual scores (0–1, higher = more compatible)
        and a combined "overall" score.

        Dimensions:
          - key_score   : harmonic compatibility (Camelot wheel distance)
          - bpm_score   : tempo compatibility (ratio closeness to 1.0)
          - energy_score: energy difference
          - overall     : weighted combination
        """
        # Key compatibility (Camelot distance)
        key_score = 0.5  # default: unknown
        if meta_a.camelot and meta_b.camelot:
            dist = camelot_distance(meta_a.camelot, meta_b.camelot)
            if dist == 999:      # different modes
                key_score = 0.0
            elif dist == 0:
                key_score = 1.0
            elif dist == 1:
                key_score = 0.85
            elif dist == 2:
                key_score = 0.5
            else:
                key_score = max(0.0, 1.0 - dist / 6)

        # BPM compatibility
        bpm_score = 0.5
        if meta_a.bpm > 0 and meta_b.bpm > 0:
            ratio = meta_a.bpm / meta_b.bpm
            # Accept 0.8–1.25x ratio (within ±25% after double-time correction)
            if 0.95 <= ratio <= 1.05:
                bpm_score = 1.0
            elif 0.90 <= ratio <= 1.10:
                bpm_score = 0.75
            elif 0.80 <= ratio <= 1.25:
                bpm_score = 0.5   # mixable with time-stretch
            elif 0.47 <= ratio <= 0.53:  # half-time
                bpm_score = 0.65
            elif 1.87 <= ratio <= 2.13:  # double-time
                bpm_score = 0.65
            else:
                bpm_score = 0.0  # outside all mixable windows — genuinely incompatible

        # Energy difference
        energy_score = 1.0 - abs(meta_a.energy - meta_b.energy)

        # Overall (weighted)
        overall = (
            key_score    * 0.45 +
            bpm_score    * 0.40 +
            energy_score * 0.15
        )

        return {
            "key_score":    round(key_score, 3),
            "bpm_score":    round(bpm_score, 3),
            "energy_score": round(energy_score, 3),
            "overall":      round(overall, 3),
            # Human-readable summary
            "key_info":     f"{meta_a.key_full} → {meta_b.key_full}",
            "bpm_info":     f"{meta_a.bpm:.1f} → {meta_b.bpm:.1f} BPM",
            "compatible":   overall >= 0.55,
        }

    # ------------------------------------------------------------------
    # Batch lookup
    # ------------------------------------------------------------------

    def lookup_many(
        self, queries: List[str], delay_seconds: float = 1.1
    ) -> List[TrackMetadata]:
        """
        Look up multiple songs.  Respects MusicBrainz's 1 req/s rate limit
        by sleeping between requests.
        """
        results = []
        for i, q in enumerate(queries):
            if i > 0:
                time.sleep(delay_seconds)
            results.append(self.lookup(q))
        return results


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

_client: Optional[MetadataClient] = None


def get_metadata_client() -> MetadataClient:
    """Return the module-level MetadataClient singleton."""
    global _client
    if _client is None:
        _client = MetadataClient()
    return _client


def quick_lookup(title: str, artist: str = "") -> TrackMetadata:
    """One-liner convenience: look up a track's metadata."""
    return get_metadata_client().lookup(title, artist)
