"""
scripts/core/library.py — Smart library management for AI RemixMate.

Responsibilities:
  1. Inventory — fast O(1) lookups via a lightweight  .index.json
  2. Deduplication — audio fingerprint (SHA-256 of first-30s PCM) catches
     the same song downloaded under different names
  3. Pruning — delete  full.wav  after stem separation to save ~60% of space
     (stems are ~40% the size of the full mix; you rarely need both)
  4. LRU eviction — when library > max_size_gb, remove full.wav files for
     least-recently-accessed songs (stems are preserved so re-mixing works)
  5. Licence lookup — convenience wrapper around scripts.core.license

Usage:
  from scripts.core.library import LibraryManager
  mgr = LibraryManager()
  mgr.touch("Eric Prydz - Opus")           # mark as accessed
  print(mgr.get_size_gb())                 # 3.14
  mgr.prune_raw("Eric Prydz - Opus")       # delete full.wav if stems exist
  mgr.evict_lru()                          # free space if over cap
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Lazy imports — these may be absent in lightweight envs
try:
    import numpy as np
    import soundfile as sf
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config — pulled from centralised config; defaults are safe standalone
# ---------------------------------------------------------------------------

try:
    from scripts.core.config import cfg as _cfg
    _MAX_SIZE_GB: float = getattr(getattr(_cfg, "library", None), "max_size_gb", 20.0)
    _KEEP_RAW: bool     = getattr(getattr(_cfg, "library", None), "keep_raw_after_separation", False)
    _PRUNE_ON_DL: bool  = getattr(getattr(_cfg, "library", None), "prune_on_download", True)
except Exception:
    _MAX_SIZE_GB = 20.0
    _KEEP_RAW    = False
    _PRUNE_ON_DL = True

try:
    from scripts.core.paths import LIBRARY_DIR as _DEFAULT_LIBRARY_DIR, song_dir as _song_dir
except Exception:
    _DEFAULT_LIBRARY_DIR = Path("library")
    def _song_dir(name: str) -> Path:   # type: ignore[misc]
        return _DEFAULT_LIBRARY_DIR / name


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SongEntry:
    """One entry in the library index."""
    name: str
    path: str                           # str(song_dir)
    size_bytes: int = 0
    has_full_wav: bool = False
    stems_available: List[str] = field(default_factory=list)
    fingerprint: Optional[str] = None   # SHA-256 of first 30 s PCM
    last_accessed: float = field(default_factory=time.time)
    source: Optional[str] = None        # "youtube", "jamendo", etc.
    license_type: Optional[str] = None  # LicenseType.value string


# ---------------------------------------------------------------------------
# Library manager
# ---------------------------------------------------------------------------

class LibraryManager:
    """
    Manages the on-disk library of songs, stems, and metadata.

    Parameters
    ----------
    library_dir : Path, optional
        Root directory.  Defaults to paths.LIBRARY_DIR.
    max_size_gb : float, optional
        Maximum library size before LRU eviction kicks in.
        Defaults to config.library.max_size_gb (20 GB).
    """

    INDEX_FILE = ".index.json"
    STEM_NAMES = ("vocals", "drums", "bass", "other")

    def __init__(
        self,
        library_dir: Optional[Path] = None,
        max_size_gb: Optional[float] = None,
    ) -> None:
        self.library_dir = library_dir or _DEFAULT_LIBRARY_DIR
        self.max_size_gb = max_size_gb if max_size_gb is not None else _MAX_SIZE_GB
        self._index_path = self.library_dir / self.INDEX_FILE

    # ------------------------------------------------------------------
    # Index I/O
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict[str, dict]:
        if not self._index_path.exists():
            return {}
        try:
            with open(self._index_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning("Could not load library index: %s — rebuilding.", e)
            return self._rebuild_index()

    def _save_index(self, index: Dict[str, dict]) -> None:
        self.library_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
        except Exception as e:
            log.warning("Could not save library index: %s", e)

    def _rebuild_index(self) -> Dict[str, dict]:
        """Walk the library directory and reconstruct the index from disk."""
        index: Dict[str, dict] = {}
        if not self.library_dir.exists():
            return index

        for song_path in self.library_dir.iterdir():
            if not song_path.is_dir() or song_path.name.startswith("."):
                continue
            name = song_path.name
            entry = self._scan_song_dir(name, song_path)
            index[name] = asdict(entry)

        self._save_index(index)
        log.info("Library index rebuilt: %d songs", len(index))
        return index

    def _scan_song_dir(self, name: str, song_path: Path) -> SongEntry:
        """Scan a single song directory and return a SongEntry."""
        size = sum(f.stat().st_size for f in song_path.rglob("*") if f.is_file())
        has_full = (song_path / "full.wav").exists()
        stems = [s for s in self.STEM_NAMES if (song_path / f"{s}.wav").exists()]

        # Try to load existing license info
        source = None
        license_type = None
        try:
            from scripts.core.license import load_license
            info = load_license(song_path)
            if info:
                source = info.source
                license_type = info.license_type.value
        except Exception:
            pass

        return SongEntry(
            name=name,
            path=str(song_path),
            size_bytes=size,
            has_full_wav=has_full,
            stems_available=stems,
            last_accessed=time.time(),
            source=source,
            license_type=license_type,
        )

    # ------------------------------------------------------------------
    # Public API — queries
    # ------------------------------------------------------------------

    def _dir(self, name: str) -> Path:
        """Return the song directory using this instance's library_dir."""
        return self.library_dir / name

    def has_song(self, name: str) -> bool:
        """Return True if the song exists in the library (any file)."""
        return self._dir(name).exists()

    def has_full_wav(self, name: str) -> bool:
        """Return True if full.wav is present for this song."""
        return (self._dir(name) / "full.wav").exists()

    def has_stems(self, name: str) -> bool:
        """Return True if at least vocals.wav exists for this song."""
        return (_song_dir(name) / "vocals.wav").exists()

    def list_songs(self) -> List[SongEntry]:
        """
        Return all songs currently in the library.

        Self-heals stale index entries: anything whose directory was deleted
        outside of evict_lru() (e.g. the pre-fix DELETE /library/{name},
        which never touched the index) gets dropped here and the index
        rewritten, instead of permanently inflating total_songs / corrupting
        eviction's LRU ordering.
        """
        index = self._load_index()
        entries = []
        stale_names = []
        for name, data in index.items():
            try:
                entry = SongEntry(**data)
            except Exception:
                stale_names.append(name)
                continue
            if not Path(entry.path).exists():
                stale_names.append(name)
                continue
            entries.append(entry)

        if stale_names:
            log.info("list_songs: dropping %d stale index entr%s (directory no longer exists): %s",
                      len(stale_names), "y" if len(stale_names) == 1 else "ies",
                      ", ".join(stale_names[:10]) + (", …" if len(stale_names) > 10 else ""))
            for name in stale_names:
                index.pop(name, None)
            self._save_index(index)

        return sorted(entries, key=lambda e: e.last_accessed, reverse=True)

    def song_info(self, name: str) -> Optional[SongEntry]:
        """Return the SongEntry for a named song, or None if not found."""
        index = self._load_index()
        data = index.get(name)
        if not data:
            return None
        try:
            return SongEntry(**data)
        except Exception:
            return None

    def get_size_bytes(self) -> int:
        """Return total size of the library in bytes (walks disk)."""
        if not self.library_dir.exists():
            return 0
        return sum(
            f.stat().st_size
            for f in self.library_dir.rglob("*")
            if f.is_file() and not f.name.startswith(".")
        )

    def get_size_gb(self) -> float:
        """Return total library size in gigabytes."""
        return self.get_size_bytes() / (1024 ** 3)

    # ------------------------------------------------------------------
    # Public API — mutations
    # ------------------------------------------------------------------

    def register(self, name: str, source: Optional[str] = None) -> SongEntry:
        """
        Register or refresh a song in the index.
        Call this after a download or separation completes.
        """
        song_path = self._dir(name)
        entry = self._scan_song_dir(name, song_path)
        if source:
            entry.source = source

        index = self._load_index()
        # Preserve existing fingerprint if we already computed it
        if name in index and index[name].get("fingerprint"):
            entry.fingerprint = index[name]["fingerprint"]
        index[name] = asdict(entry)
        self._save_index(index)
        return entry

    def touch(self, name: str) -> None:
        """Update last_accessed timestamp for a song (call on every remix)."""
        index = self._load_index()
        if name in index:
            index[name]["last_accessed"] = time.time()
            self._save_index(index)

    def unregister(self, name: str) -> bool:
        """
        Remove a song's entry from the persisted index without touching disk.

        Call this whenever a song directory is deleted by something other
        than evict_lru() (which already pops its own index entries) — e.g.
        DELETE /library/{name}. Without this, the index accumulates phantom
        entries for songs that no longer exist on disk, which inflates
        storage_status()'s total_songs and corrupts evict_lru()'s LRU
        ordering (it sorts/evicts from this same index).

        Returns True if an entry was actually removed.
        """
        index = self._load_index()
        if name in index:
            index.pop(name, None)
            self._save_index(index)
            return True
        return False

    # ------------------------------------------------------------------
    # Deduplication via audio fingerprinting
    # ------------------------------------------------------------------

    def audio_fingerprint(self, wav_path: Path) -> Optional[str]:
        """
        Compute a fast audio fingerprint: SHA-256 of the first 30 seconds of
        raw PCM samples (mono, downsampled to 8 kHz for speed).

        Returns a hex digest string, or None if the file can't be read.
        """
        if not _AUDIO_AVAILABLE:
            log.debug("audio fingerprint unavailable — soundfile not installed")
            return None
        try:
            # Read first 30 s at native rate then resample to 8 kHz mono
            data, sr = sf.read(str(wav_path), frames=int(sr_or_default(wav_path) * 30),
                               always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            # Downsample to 8 kHz for a fast, compact fingerprint
            try:
                from scripts.core.gpu import gpu_resample
                data_8k = gpu_resample(data.astype("float32"), orig_sr=sr, target_sr=8000)
            except (ImportError, Exception):
                import librosa
                data_8k = librosa.resample(data.astype("float32"), orig_sr=sr, target_sr=8000)
            # Quantise to int16 for determinism across float precision differences
            pcm = (data_8k * 32767).astype("int16")
            digest = hashlib.sha256(pcm.tobytes()).hexdigest()
            return digest
        except Exception as e:
            log.debug("Fingerprint failed for %s: %s", wav_path, e)
            return None

    def find_duplicate(self, wav_path: Path) -> Optional[str]:
        """
        Check if wav_path is a duplicate of something already in the library.

        Returns the existing song name if a duplicate is found, else None.
        """
        fp = self.audio_fingerprint(wav_path)
        if fp is None:
            return None

        index = self._load_index()
        for name, data in index.items():
            if data.get("fingerprint") == fp:
                return name
        return None

    def store_fingerprint(self, name: str, wav_path: Path) -> Optional[str]:
        """
        Compute and persist the fingerprint for a song.
        Call this once after download; the fingerprint is stored in the index.
        """
        fp = self.audio_fingerprint(wav_path)
        if fp is None:
            return None
        index = self._load_index()
        if name in index:
            index[name]["fingerprint"] = fp
            self._save_index(index)
        return fp

    # ------------------------------------------------------------------
    # Pruning — delete full.wav after stem separation
    # ------------------------------------------------------------------

    def prune_raw(self, name: str, force: bool = False) -> bool:
        """
        Delete  full.wav  for a song if all four stems are present.
        This reclaims ~60% of storage (the full mix is redundant once stems exist).

        Parameters
        ----------
        name : str
            Song name.
        force : bool
            Delete full.wav even if not all stems are present.

        Returns
        -------
        bool : True if full.wav was deleted, False otherwise.
        """
        if _KEEP_RAW and not force:
            log.debug("prune_raw skipped for '%s': keep_raw_after_separation=true", name)
            return False

        song_path = self._dir(name)
        full_wav = song_path / "full.wav"
        if not full_wav.exists():
            return False

        stems_present = [s for s in self.STEM_NAMES if (song_path / f"{s}.wav").exists()]
        all_stems = len(stems_present) == len(self.STEM_NAMES)

        if not all_stems and not force:
            log.debug(
                "prune_raw skipped for '%s': only %d/%d stems present",
                name, len(stems_present), len(self.STEM_NAMES),
            )
            return False

        try:
            size_mb = full_wav.stat().st_size / (1024 * 1024)
            full_wav.unlink()
            log.info("🗑️  Pruned full.wav for '%s' (freed %.1f MB)", name, size_mb)
            # Refresh index
            index = self._load_index()
            if name in index:
                index[name]["has_full_wav"] = False
                index[name]["size_bytes"] = sum(
                    f.stat().st_size for f in song_path.rglob("*") if f.is_file()
                )
                self._save_index(index)
            return True
        except Exception as e:
            log.warning("Could not prune full.wav for '%s': %s", name, e)
            return False

    # ------------------------------------------------------------------
    # LRU eviction
    # ------------------------------------------------------------------

    def evict_lru(
        self,
        target_gb: Optional[float] = None,
        dry_run: bool = False,
    ) -> List[str]:
        """
        Evict least-recently-used songs until library fits within target_gb.

        Strategy:
          1. Only delete  full.wav  (keeps stems so re-mixing still works)
          2. If still over budget, delete entire song directories (LRU order)

        Parameters
        ----------
        target_gb : float, optional
            Evict until library is below this size.  Defaults to max_size_gb.
        dry_run : bool
            If True, report what would be deleted without deleting.

        Returns
        -------
        List[str] : Names of songs whose full.wav was removed.
        """
        target = (target_gb or self.max_size_gb) * 1024 ** 3  # bytes
        current = self.get_size_bytes()

        if current <= target:
            log.info(
                "Library size %.2f GB is within %.2f GB cap — no eviction needed.",
                current / 1024 ** 3, target / 1024 ** 3,
            )
            return []

        log.warning(
            "Library %.2f GB exceeds %.2f GB cap — starting LRU eviction.",
            current / 1024 ** 3, target / 1024 ** 3,
        )

        # Sort by last_accessed ascending (oldest first = first to evict)
        songs = sorted(self.list_songs(), key=lambda e: e.last_accessed)
        evicted: List[str] = []

        # Pass 1: delete only full.wav
        for entry in songs:
            if current <= target:
                break
            full_wav = Path(entry.path) / "full.wav"
            if full_wav.exists():
                size = full_wav.stat().st_size
                if dry_run:
                    log.info("[DRY RUN] Would prune full.wav: %s (%.1f MB)",
                             entry.name, size / 1e6)
                else:
                    if self.prune_raw(entry.name, force=True):
                        current -= size
                        evicted.append(entry.name)

        # Pass 2: if still over budget, delete entire song dirs (LRU)
        if current > target:
            for entry in songs:
                if current <= target:
                    break
                if entry.name in evicted:
                    continue  # already partially evicted
                song_path = Path(entry.path)
                if song_path.exists():
                    size = entry.size_bytes
                    if dry_run:
                        log.info("[DRY RUN] Would delete entire song dir: %s (%.1f MB)",
                                 entry.name, size / 1e6)
                    else:
                        import shutil
                        shutil.rmtree(str(song_path), ignore_errors=True)
                        current -= size
                        evicted.append(entry.name)
                        log.info("🗑️  Evicted song dir: %s (freed %.1f MB)",
                                 entry.name, size / 1e6)
                        # Remove from index
                        index = self._load_index()
                        index.pop(entry.name, None)
                        self._save_index(index)

        return evicted

    # ------------------------------------------------------------------
    # Convenience: print library summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted string summary of the library."""
        songs = self.list_songs()
        total_gb = self.get_size_gb()
        lines = [
            f"📚 Library Summary",
            f"   Songs: {len(songs)}",
            f"   Total size: {total_gb:.2f} GB / {self.max_size_gb:.0f} GB cap",
            f"   {'⚠️  Over cap — run evict_lru()' if total_gb > self.max_size_gb else '✅ Within cap'}",
        ]
        for entry in songs[:10]:  # show top 10 most recently used
            stems = f"  stems: {', '.join(entry.stems_available)}" if entry.stems_available else ""
            raw = "  📼 raw" if entry.has_full_wav else ""
            size = f"{entry.size_bytes / 1e6:.0f} MB"
            lines.append(f"   • {entry.name} [{size}]{stems}{raw}")
        if len(songs) > 10:
            lines.append(f"   … and {len(songs) - 10} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sr_or_default(wav_path: Path, default: int = 44100) -> int:
    """Read sample rate from a WAV without loading the full file."""
    try:
        info = sf.info(str(wav_path))
        return info.samplerate
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

_manager: Optional[LibraryManager] = None


def get_library_manager() -> LibraryManager:
    """Return the module-level LibraryManager singleton."""
    global _manager
    if _manager is None:
        _manager = LibraryManager()
    return _manager
