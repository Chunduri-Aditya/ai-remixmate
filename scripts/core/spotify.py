"""
scripts/core/spotify.py — Spotify Web API client for AI RemixMate.

What this gives you (free Spotify developer account):
  • Search Spotify's catalogue by track / artist / album name
  • Pull audio features (BPM, key, energy, danceability) if your app was
    created before Spotify deprecated that endpoint (Nov 2024). Falls back
    gracefully to "analyse locally" if unavailable.
  • Browse your own playlists and queue whole playlists for download
  • Auto-convert Spotify's key integers → Camelot Wheel notation so the
    metadata plugs straight into music_intelligence.py

What this does NOT do:
  • Stream or download audio from Spotify (against their ToS)
  • That's what yt-dlp is for — Spotify just gives you the metadata + discovery

Auth modes:
  CLIENT_CREDENTIALS — automatic, no user login needed, read-only catalogue access
  AUTHORIZATION_CODE — user login, gives access to personal playlists & library

Setup (one-time):
  1. Go to https://developer.spotify.com/dashboard and create a free app
  2. Set Redirect URI to: http://localhost:8000/spotify/callback
  3. Copy Client ID and Client Secret into config.local.yaml:
       spotify:
         client_id: "YOUR_CLIENT_ID"
         client_secret: "YOUR_CLIENT_SECRET"

Usage:
    from scripts.core.spotify import get_client
    sp = get_client()
    results = sp.search("Anyma Voices In My Head", limit=5)
    features = sp.audio_features(results[0]["id"])  # may be None if restricted
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

try:
    from scripts.core.paths import DATA_DIR
except Exception:
    DATA_DIR = Path(__file__).parents[2] / "data"

TOKEN_FILE      = DATA_DIR / "spotify_token.json"
PKCE_STATE_FILE = DATA_DIR / "spotify_pkce_state.json"

# ---------------------------------------------------------------------------
# Camelot Wheel mapping
# ---------------------------------------------------------------------------
# Spotify encodes key as 0–11 (C=0 … B=11) and mode as 0 (minor) / 1 (major).
# This lookup converts to Camelot Wheel notation used throughout RemixMate.

_CAMELOT: Dict[Tuple[int, int], str] = {
    # (spotify_key, spotify_mode) → camelot
    # Major (mode=1) — outer B ring
    (0,  1): "8B",   # C major
    (1,  1): "3B",   # C#/Db major
    (2,  1): "10B",  # D major
    (3,  1): "5B",   # D#/Eb major
    (4,  1): "12B",  # E major
    (5,  1): "7B",   # F major
    (6,  1): "2B",   # F#/Gb major
    (7,  1): "9B",   # G major
    (8,  1): "4B",   # G#/Ab major
    (9,  1): "11B",  # A major
    (10, 1): "6B",   # A#/Bb major
    (11, 1): "1B",   # B major
    # Minor (mode=0) — inner A ring
    (0,  0): "5A",   # C minor
    (1,  0): "12A",  # C#/Db minor
    (2,  0): "7A",   # D minor
    (3,  0): "2A",   # D#/Eb minor
    (4,  0): "9A",   # E minor
    (5,  0): "4A",   # F minor
    (6,  0): "11A",  # F#/Gb minor
    (7,  0): "6A",   # G minor
    (8,  0): "1A",   # G#/Ab minor
    (9,  0): "8A",   # A minor
    (10, 0): "3A",   # A#/Bb minor
    (11, 0): "10A",  # B minor
}

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def spotify_key_to_camelot(key: int, mode: int) -> str:
    """Convert Spotify key integer (0–11) + mode (0/1) → Camelot notation."""
    return _CAMELOT.get((key, mode), "8B")


def spotify_key_to_name(key: int, mode: int) -> str:
    """Return human-readable key name, e.g. 'A minor' or 'C# major'."""
    if key < 0 or key > 11:
        return "Unknown"
    note = _NOTE_NAMES[key]
    mode_str = "major" if mode == 1 else "minor"
    return f"{note} {mode_str}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpotifyTrack:
    """Lightweight representation of a Spotify track."""
    id:          str
    name:        str
    artists:     List[str]
    album:       str
    duration_ms: int
    popularity:  int
    uri:         str
    preview_url: Optional[str] = None

    # Audio features — populated separately via audio_features()
    bpm:          Optional[float] = None
    key:          Optional[int]   = None
    mode:         Optional[int]   = None
    energy:       Optional[float] = None
    danceability: Optional[float] = None
    valence:      Optional[float] = None
    loudness_db:  Optional[float] = None
    camelot:      Optional[str]   = None
    key_name:     Optional[str]   = None

    @property
    def artist_str(self) -> str:
        return ", ".join(self.artists)

    @property
    def download_query(self) -> str:
        """Best search query to feed into yt-dlp for this track."""
        return f"{self.artist_str} - {self.name}"

    @property
    def duration_str(self) -> str:
        secs = self.duration_ms // 1000
        return f"{secs // 60}:{secs % 60:02d}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":           self.id,
            "name":         self.name,
            "artists":      self.artists,
            "artist_str":   self.artist_str,
            "album":        self.album,
            "duration_ms":  self.duration_ms,
            "duration_str": self.duration_str,
            "popularity":   self.popularity,
            "uri":          self.uri,
            "preview_url":  self.preview_url,
            "bpm":          self.bpm,
            "key":          self.key,
            "mode":         self.mode,
            "energy":       self.energy,
            "danceability": self.danceability,
            "valence":      self.valence,
            "loudness_db":  self.loudness_db,
            "camelot":      self.camelot,
            "key_name":     self.key_name,
            "download_query": self.download_query,
        }


@dataclass
class SpotifyPlaylist:
    """A user's Spotify playlist."""
    id:          str
    name:        str
    description: str
    track_count: int
    owner:       str
    public:      bool
    image_url:   Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":          self.id,
            "name":        self.name,
            "description": self.description,
            "track_count": self.track_count,
            "owner":       self.owner,
            "public":      self.public,
            "image_url":   self.image_url,
        }


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

class _TokenStore:
    """
    Persists Spotify tokens to disk so they survive Streamlit reruns.

    Stores two separate token records:
      "client"   — Client Credentials token (auto-refresh, no user)
      "user"     — Authorization Code token (requires user login)
    """

    def load(self) -> Dict[str, Any]:
        try:
            if TOKEN_FILE.exists():
                return json.loads(TOKEN_FILE.read_text())
        except Exception:
            pass
        return {}

    def save(self, data: Dict[str, Any]) -> None:
        try:
            TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            TOKEN_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.warning("Could not save Spotify token: %s", e)

    def get_client_token(self) -> Optional[str]:
        data = self.load()
        rec  = data.get("client", {})
        if rec.get("access_token") and rec.get("expires_at", 0) > time.time() + 30:
            return rec["access_token"]
        return None

    def set_client_token(self, token: str, expires_in: int) -> None:
        data = self.load()
        data["client"] = {
            "access_token": token,
            "expires_at":   time.time() + expires_in,
        }
        self.save(data)

    def get_user_token(self) -> Optional[str]:
        data = self.load()
        rec  = data.get("user", {})
        if rec.get("access_token") and rec.get("expires_at", 0) > time.time() + 30:
            return rec["access_token"]
        return None

    def get_user_refresh_token(self) -> Optional[str]:
        return self.load().get("user", {}).get("refresh_token")

    def set_user_token(
        self, access_token: str, refresh_token: Optional[str], expires_in: int
    ) -> None:
        data = self.load()
        existing_refresh = data.get("user", {}).get("refresh_token", "")
        data["user"] = {
            "access_token":  access_token,
            "refresh_token": refresh_token or existing_refresh,
            "expires_at":    time.time() + expires_in,
        }
        self.save(data)

    def clear_user_token(self) -> None:
        data = self.load()
        data.pop("user", None)
        self.save(data)

    def is_user_connected(self) -> bool:
        data = self.load()
        return bool(data.get("user", {}).get("refresh_token"))


_token_store = _TokenStore()


# ---------------------------------------------------------------------------
# SpotifyClient
# ---------------------------------------------------------------------------

class SpotifyClient:
    """
    Thin wrapper around the Spotify Web API using only the `requests` library.

    Handles both auth flows and graceful fallback when endpoints are restricted.
    No spotipy dependency — works with what's already in requirements.txt.
    """

    BASE_URL     = "https://api.spotify.com/v1"
    ACCOUNTS_URL = "https://accounts.spotify.com"

    def __init__(
        self,
        client_id:     str,
        client_secret: str,
        redirect_uri:  str = "http://127.0.0.1:8000/spotify/callback",
    ):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.redirect_uri  = redirect_uri
        self._store        = _token_store

    # ── Auth helpers ──────────────────────────────────────────────────────────

    def _basic_auth_header(self) -> str:
        creds = f"{self.client_id}:{self.client_secret}"
        return "Basic " + base64.b64encode(creds.encode()).decode()

    def _ensure_client_token(self) -> str:
        """Get (or refresh) a Client Credentials token. Raises on failure."""
        token = self._store.get_client_token()
        if token:
            return token

        resp = requests.post(
            f"{self.ACCOUNTS_URL}/api/token",
            headers={"Authorization": self._basic_auth_header()},
            data={"grant_type": "client_credentials"},
            timeout=10,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Spotify client credentials failed: {resp.status_code} {resp.text}"
            )
        body = resp.json()
        self._store.set_client_token(body["access_token"], body.get("expires_in", 3600))
        return body["access_token"]

    def _ensure_user_token(self) -> Optional[str]:
        """Return user token, refreshing via refresh_token if needed. None if not connected."""
        token = self._store.get_user_token()
        if token:
            return token

        refresh = self._store.get_user_refresh_token()
        if not refresh:
            return None

        resp = requests.post(
            f"{self.ACCOUNTS_URL}/api/token",
            headers={"Authorization": self._basic_auth_header()},
            data={
                "grant_type":    "refresh_token",
                "refresh_token": refresh,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            log.warning("Spotify token refresh failed: %s", resp.text)
            return None

        body = resp.json()
        self._store.set_user_token(
            body["access_token"],
            body.get("refresh_token"),
            body.get("expires_in", 3600),
        )
        return body["access_token"]

    def _headers(self, user: bool = False) -> Dict[str, str]:
        """Build auth headers. user=True tries user token first, falls back to client."""
        if user:
            tok = self._ensure_user_token()
            if tok:
                return {"Authorization": f"Bearer {tok}"}
        return {"Authorization": f"Bearer {self._ensure_client_token()}"}

    def _get(self, path: str, params: Dict = None, user: bool = False) -> Optional[Dict]:
        """GET wrapper with error logging and None-on-error."""
        # Guard against path manipulation — path must be a clean relative path.
        # Prevents partial SSRF if a caller ever passes untrusted input as path.
        if not path.startswith("/") or ".." in path or "://" in path:
            log.warning("Spotify _get: rejected unsafe path %r", path)
            return None
        try:
            resp = requests.get(
                f"{self.BASE_URL}{path}",
                headers=self._headers(user=user),
                params=params or {},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            log.debug("Spotify GET %s → %s: %s", path, resp.status_code, resp.text[:200])
            return None
        except Exception as e:
            log.warning("Spotify GET %s failed: %s", path, e)
            return None

    # ── OAuth helpers ─────────────────────────────────────────────────────────

    def get_auth_url(self, scopes: List[str] = None) -> Tuple[str, str]:
        """
        Build the Spotify Authorization URL for PKCE flow.

        Returns:
            (auth_url, state_token)

        The state_token is saved to disk so the callback can verify it.
        """
        scopes = scopes or [
            "playlist-read-private",
            "playlist-read-collaborative",
            "user-library-read",
            "user-top-read",
            "user-read-recently-played",
        ]

        state        = secrets.token_urlsafe(16)
        code_verifier = secrets.token_urlsafe(43)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip("=")

        # Persist PKCE state so the callback can exchange the code
        try:
            PKCE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            PKCE_STATE_FILE.write_text(json.dumps({
                "state":          state,
                "code_verifier":  code_verifier,
            }))
        except Exception as e:
            log.warning("Could not save PKCE state: %s", e)

        params = {
            "client_id":             self.client_id,
            "response_type":         "code",
            "redirect_uri":          self.redirect_uri,
            "scope":                 " ".join(scopes),
            "state":                 state,
            "code_challenge_method": "S256",
            "code_challenge":        code_challenge,
        }
        url = f"{self.ACCOUNTS_URL}/authorize?" + urllib.parse.urlencode(params)
        return url, state

    def exchange_code(self, code: str, state: str) -> bool:
        """Exchange authorization code for tokens. Returns True on success."""
        try:
            pkce = json.loads(PKCE_STATE_FILE.read_text()) if PKCE_STATE_FILE.exists() else {}
        except Exception:
            pkce = {}

        if pkce.get("state") != state:
            log.warning("Spotify OAuth state mismatch — possible CSRF")
            return False

        resp = requests.post(
            f"{self.ACCOUNTS_URL}/api/token",
            data={
                "grant_type":    "authorization_code",
                "code":          code,
                "redirect_uri":  self.redirect_uri,
                "client_id":     self.client_id,
                "code_verifier": pkce.get("code_verifier", ""),
            },
            timeout=10,
        )
        if resp.status_code != 200:
            log.warning("Spotify code exchange failed: %s", resp.text)
            return False

        body = resp.json()
        self._store.set_user_token(
            body["access_token"],
            body.get("refresh_token"),
            body.get("expires_in", 3600),
        )
        try:
            PKCE_STATE_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        log.info("Spotify user connected successfully")
        return True

    def disconnect(self) -> None:
        """Remove stored user token."""
        self._store.clear_user_token()

    def is_connected(self) -> bool:
        """True if user has connected their Spotify account."""
        return self._store.is_user_connected()

    def is_configured(self) -> bool:
        """True if client_id and client_secret are set."""
        return bool(self.client_id and self.client_secret)

    # ── Catalogue (no user auth needed) ──────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> List[SpotifyTrack]:
        """
        Search Spotify for tracks matching `query`.
        Returns a list of SpotifyTrack objects (no audio features yet).
        """
        data = self._get("/search", params={
            "q":     query,
            "type":  "track",
            "limit": min(limit, 50),
        })
        if not data:
            return []

        tracks = []
        for item in data.get("tracks", {}).get("items", []):
            try:
                tracks.append(SpotifyTrack(
                    id          = item["id"],
                    name        = item["name"],
                    artists     = [a["name"] for a in item.get("artists", [])],
                    album       = item.get("album", {}).get("name", ""),
                    duration_ms = item.get("duration_ms", 0),
                    popularity  = item.get("popularity", 0),
                    uri         = item.get("uri", ""),
                    preview_url = item.get("preview_url"),
                ))
            except Exception as e:
                log.debug("Failed to parse track: %s", e)
        return tracks

    def audio_features(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch audio features for a single track.

        Returns dict with: bpm, key, mode, energy, danceability, valence,
        loudness_db, camelot, key_name — or None if the endpoint is unavailable
        (Spotify restricted audio features for new apps after Nov 2024).
        """
        # Spotify track IDs are base-62 (alphanumeric + no special chars).
        # Validate before interpolating into the path.
        if not track_id or not track_id.replace("-", "").isalnum():
            log.warning("get_audio_features: invalid track_id %r", track_id)
            return None
        data = self._get(f"/audio-features/{track_id}")
        if not data:
            return None

        key  = data.get("key",  -1)
        mode = data.get("mode", -1)

        return {
            "bpm":          data.get("tempo"),
            "key":          key,
            "mode":         mode,
            "energy":       data.get("energy"),
            "danceability": data.get("danceability"),
            "valence":      data.get("valence"),
            "loudness_db":  data.get("loudness"),
            "camelot":      spotify_key_to_camelot(key, mode) if key >= 0 else None,
            "key_name":     spotify_key_to_name(key, mode) if key >= 0 else None,
        }

    def enrich_tracks(self, tracks: List[SpotifyTrack]) -> List[SpotifyTrack]:
        """
        Attempt to attach audio features to a list of tracks in-place.
        Uses batch endpoint (up to 100 IDs at once). Silently skips on failure.
        """
        if not tracks:
            return tracks

        ids = [t.id for t in tracks[:100]]
        data = self._get("/audio-features", params={"ids": ",".join(ids)})
        if not data:
            return tracks

        features_by_id = {}
        for f in data.get("audio_features", []) or []:
            if f and f.get("id"):
                features_by_id[f["id"]] = f

        for track in tracks:
            f = features_by_id.get(track.id)
            if not f:
                continue
            key  = f.get("key",  -1)
            mode = f.get("mode", -1)
            track.bpm          = f.get("tempo")
            track.key          = key
            track.mode         = mode
            track.energy       = f.get("energy")
            track.danceability = f.get("danceability")
            track.valence      = f.get("valence")
            track.loudness_db  = f.get("loudness")
            if key >= 0:
                track.camelot  = spotify_key_to_camelot(key, mode)
                track.key_name = spotify_key_to_name(key, mode)

        return tracks

    # ── User library (requires user auth) ─────────────────────────────────────

    def get_user_playlists(self, limit: int = 50) -> List[SpotifyPlaylist]:
        """Fetch the authenticated user's playlists."""
        data = self._get("/me/playlists", params={"limit": min(limit, 50)}, user=True)
        if not data:
            return []

        playlists = []
        for item in data.get("items", []):
            images = item.get("images", [])
            playlists.append(SpotifyPlaylist(
                id          = item["id"],
                name        = item.get("name", "Untitled"),
                description = item.get("description", ""),
                track_count = item.get("tracks", {}).get("total", 0),
                owner       = item.get("owner", {}).get("display_name", ""),
                public      = item.get("public", False),
                image_url   = images[0]["url"] if images else None,
            ))
        return playlists

    def get_playlist_tracks(
        self, playlist_id: str, limit: int = 50
    ) -> List[SpotifyTrack]:
        """Fetch tracks in a playlist (up to `limit`)."""
        data = self._get(
            f"/playlists/{playlist_id}/tracks",
            params={"limit": min(limit, 50), "fields": "items(track(id,name,artists,album,duration_ms,popularity,uri,preview_url))"},
            user=True,
        )
        if not data:
            return []

        tracks = []
        for item in data.get("items", []):
            t = item.get("track")
            if not t or not t.get("id"):
                continue
            try:
                tracks.append(SpotifyTrack(
                    id          = t["id"],
                    name        = t["name"],
                    artists     = [a["name"] for a in t.get("artists", [])],
                    album       = t.get("album", {}).get("name", ""),
                    duration_ms = t.get("duration_ms", 0),
                    popularity  = t.get("popularity", 0),
                    uri         = t.get("uri", ""),
                    preview_url = t.get("preview_url"),
                ))
            except Exception as e:
                log.debug("Failed to parse playlist track: %s", e)

        return tracks

    def get_user_top_tracks(self, limit: int = 20, time_range: str = "medium_term") -> List[SpotifyTrack]:
        """
        Fetch the user's top tracks.
        time_range: 'short_term' (4 weeks), 'medium_term' (6 months), 'long_term' (all time)
        """
        data = self._get(
            "/me/top/tracks",
            params={"limit": min(limit, 50), "time_range": time_range},
            user=True,
        )
        if not data:
            return []

        tracks = []
        for item in data.get("items", []):
            try:
                tracks.append(SpotifyTrack(
                    id          = item["id"],
                    name        = item["name"],
                    artists     = [a["name"] for a in item.get("artists", [])],
                    album       = item.get("album", {}).get("name", ""),
                    duration_ms = item.get("duration_ms", 0),
                    popularity  = item.get("popularity", 0),
                    uri         = item.get("uri", ""),
                    preview_url = item.get("preview_url"),
                ))
            except Exception as e:
                log.debug("Failed to parse top track: %s", e)
        return tracks


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def _load_config() -> Dict[str, Any]:
    """Load Spotify config from config.local.yaml or config.yaml or env vars."""
    config: Dict[str, Any] = {}

    # Try config files
    for cfg_name in ("config.local.yaml", "config.yaml"):
        cfg_path = Path(__file__).parents[2] / cfg_name
        if cfg_path.exists():
            try:
                import yaml
                raw = yaml.safe_load(cfg_path.read_text()) or {}
                spotify_section = raw.get("spotify", {})
                if spotify_section:
                    config.update(spotify_section)
                    break
            except Exception:
                pass

    # Environment variable overrides
    config["client_id"]     = os.environ.get("SPOTIFY_CLIENT_ID",     config.get("client_id", ""))
    config["client_secret"] = os.environ.get("SPOTIFY_CLIENT_SECRET", config.get("client_secret", ""))
    config["redirect_uri"]  = os.environ.get("SPOTIFY_REDIRECT_URI",  config.get("redirect_uri", "http://127.0.0.1:8000/spotify/callback"))

    return config


_client: Optional[SpotifyClient] = None


def get_client() -> SpotifyClient:
    """Return the module-level SpotifyClient singleton (lazy init)."""
    global _client
    if _client is None:
        cfg     = _load_config()
        _client = SpotifyClient(
            client_id     = cfg.get("client_id", ""),
            client_secret = cfg.get("client_secret", ""),
            redirect_uri  = cfg.get("redirect_uri", "http://localhost:8000/spotify/callback"),
        )
    return _client


def is_configured() -> bool:
    """Quick check — True if Spotify credentials are present in config."""
    return get_client().is_configured()
