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
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Lazy imports — these may be absent in lightweight envs
try:
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
    from scripts.core.paths import LIBRARY_DIR as _DEFAULT_LIBRARY_DIR
    from scripts.core.paths import song_dir as _song_dir
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


def is_playlist_artifact(song_path: Path) -> bool:
    """
    Return True if this directory's yt-dlp metadata (full.info.json) marks it
    as a playlist download rather than a single song.

    Two forms exist in the wild:
      - a playlist manifest: ``_type == "playlist"`` or a top-level
        ``entries`` list;
      - per-entry metadata left behind by a concatenated playlist download:
        ``n_entries`` / ``playlist_count`` > 1 (e.g. library/Walking_playlist,
        whose full.wav held multiple concatenated tracks and dominated
        /recommend because its averaged features "matched" everything).

    Single-track downloads never carry playlist keys — download.py resolves
    playlist entries to direct video URLs and downloads each with
    ``no_playlist: True`` — so the presence of these keys means the audio is
    not one song and must not be indexed as one.
    """
    info_path = song_path / "full.info.json"
    if not info_path.exists():
        return False
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(info, dict):
        return False
    if info.get("_type") == "playlist" or isinstance(info.get("entries"), list):
        return True
    try:
        n_entries = int(info.get("n_entries") or info.get("playlist_count") or 0)
    except (TypeError, ValueError):
        n_entries = 0
    return n_entries > 1


def _stem_is_healthy(path: Path) -> bool:
    """
    True if `path` is a stem file worth trusting as "this stem really
    exists" before deleting the last full-length source over it.

    prune_raw()/prune_enhanced() previously counted a stem as present via
    `.exists()` alone. That's true even for a zero-byte or truncated file —
    exactly the shape left behind if the per-stem LUFS normalisation in
    separate_song_stems() dies mid-`sf.write()` (that exception is swallowed
    and the overall result still reports success). Reuses the same cheap
    probe check_song_health() already uses for the same reason: sf.info()
    is header-only (no full decode), and the tail-window read catches the
    common truncation mode (write killed partway through) that a
    frames-count check alone would miss, since a truncated file can still
    report a plausible non-zero frame count in its header.

    Falls back to existence-only when audio libs aren't installed
    (_AUDIO_AVAILABLE=False) — no worse than the pre-existing behaviour in
    that environment, not a regression.
    """
    if not path.exists():
        return False
    if not _AUDIO_AVAILABLE:
        return True
    try:
        info = sf.info(str(path))
        if info.frames <= 0:
            return False
        win = min(info.frames, 44100)
        sf.read(str(path), frames=win, start=info.frames - win, always_2d=False)
        return True
    except Exception as e:
        log.warning("Stem file failed health probe, treating as absent: %s (%s)", path, e)
        return False


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
        # Guards every _load_index() -> mutate -> _save_index() cycle below.
        # Without it, two threads (task_batch_stem_split runs each song's
        # separation inside a ThreadPoolExecutor worker, and evict_lru can
        # fire concurrently via auto_evict_on_download) can interleave a
        # read-modify-write pair: both load the same on-disk index, each
        # mutates its own in-memory copy, and whichever saves last silently
        # discards the other's write (e.g. a register() from song A racing a
        # prune_raw() index update for song B loses one of the two updates).
        # RLock, not Lock — evict_lru() calls list_songs(), prune_raw(), and
        # prune_enhanced() internally, each of which acquires this same lock;
        # a plain Lock would deadlock the moment a locked method called
        # another locked method from the same thread.
        self._index_lock = threading.RLock()

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
            if is_playlist_artifact(song_path):
                log.warning("_rebuild_index: skipping playlist artifact '%s' — "
                            "not a single song", song_path.name)
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
        stems = [
            s for s in self.STEM_NAMES
            if (song_path / f"{s}.flac").exists() or (song_path / f"{s}.wav").exists()
        ]

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

        Self-heals the index in both directions:
        - stale entries: anything whose directory was deleted outside of
          evict_lru() (e.g. the pre-fix DELETE /library/{name}, which never
          touched the index) gets dropped, instead of permanently inflating
          total_songs / corrupting eviction's LRU ordering.
        - orphans: on-disk song directories that never got registered
          (e.g. copied in manually, or downloaded by a code path that
          predates register()) get adopted, so storage counts match what
          GET /library reports from the filesystem.
        """
        with self._index_lock:
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

            # Adopt on-disk song dirs the index doesn't know about
            adopted = []
            skipped_playlists = []
            if self.library_dir.exists():
                for song_path in self.library_dir.iterdir():
                    if not song_path.is_dir() or song_path.name.startswith("."):
                        continue
                    if song_path.name in index:
                        continue
                    if is_playlist_artifact(song_path):
                        skipped_playlists.append(song_path.name)
                        continue
                    entry = self._scan_song_dir(song_path.name, song_path)
                    index[song_path.name] = asdict(entry)
                    entries.append(entry)
                    adopted.append(song_path.name)
            if adopted:
                log.info("list_songs: adopted %d unregistered on-disk song(s): %s",
                          len(adopted),
                          ", ".join(adopted[:10]) + (", …" if len(adopted) > 10 else ""))
            if skipped_playlists:
                log.warning("list_songs: refused to adopt %d playlist artifact(s) "
                            "(not single songs): %s",
                            len(skipped_playlists), ", ".join(skipped_playlists[:10]))

            if stale_names or adopted:
                self._save_index(index)

        return sorted(entries, key=lambda e: e.last_accessed, reverse=True)

    def song_info(self, name: str) -> Optional[SongEntry]:
        """Return the SongEntry for a named song, or None if not found."""
        with self._index_lock:
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

        with self._index_lock:
            index = self._load_index()
            # Preserve existing fingerprint if we already computed it
            if name in index and index[name].get("fingerprint"):
                entry.fingerprint = index[name]["fingerprint"]
            index[name] = asdict(entry)
            self._save_index(index)
        return entry

    def touch(self, name: str) -> None:
        """Update last_accessed timestamp for a song (call on every remix)."""
        with self._index_lock:
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
        with self._index_lock:
            index = self._load_index()
            if name in index:
                index.pop(name, None)
                self._save_index(index)
                return True
        return False

    # ------------------------------------------------------------------
    # Health check — "is this song actually usable?"
    # ------------------------------------------------------------------

    def check_song_health(self, name: str) -> List[str]:
        """
        Return a list of reasons this song is broken, or [] if it's healthy.

        A song counts as broken when:
          - its directory doesn't exist at all, or
          - it has no playable audio source — checked via
            scripts.core.audio_source.resolve_source_file()/stem_files(),
            the same resolution order (full.wav → full_enhanced.wav →
            Demucs stems) the render/analysis path actually uses. A naive
            check for just full.wav + stems would wrongly flag every song
            that kept only full_enhanced.wav after pruning — which, with
            prune_on_download enabled, is most of the library, or
          - the audio source that IS present fails to decode (corrupt or
            truncated file — soundfile raises, or reports zero frames), or
          - meta.json / analysis.json exists on disk but isn't valid JSON
            (a partial/corrupted write — e.g. a process killed mid-analysis).

        Deliberately does NOT flag a song merely for never having been
        analyzed (meta.json/analysis.json simply absent) — that's a normal
        pending state already surfaced by has_analysis()/"Analyze missing",
        not damage. Conflating "never analyzed" with "broken" would make
        every unanalyzed song in the library look like a deletion candidate.
        """
        from scripts.core.audio_source import resolve_source_file, stem_files

        reasons: List[str] = []
        song_path = self._dir(name)
        if not song_path.exists():
            return ["song directory does not exist"]

        source = resolve_source_file(name)
        if source is None:
            stems = stem_files(name)
            source = stems[0] if stems else None

        if source is None:
            reasons.append("no playable audio source (no full.wav, full_enhanced.wav, or stems)")
        elif _AUDIO_AVAILABLE:
            try:
                info = sf.info(str(source))
                if info.frames <= 0:
                    reasons.append(f"'{source.name}' has zero audio frames (empty/corrupt file)")
                else:
                    # Cheap decode probe — read a small window from the END of
                    # the file rather than the start: the common damage mode
                    # (download/write killed partway) truncates the tail while
                    # the header and opening seconds still decode fine.
                    win = min(info.frames, 44100)
                    sf.read(str(source), frames=win, start=info.frames - win, always_2d=False)
            except Exception as e:
                reasons.append(f"'{source.name}' failed to decode: {e}")

        for meta_name in ("meta.json", "analysis.json"):
            p = song_path / meta_name
            if p.exists():
                try:
                    json.loads(p.read_text())
                except Exception as e:
                    reasons.append(f"{meta_name} is corrupt/unparsable: {e}")

        return reasons

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

        with self._index_lock:
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
        with self._index_lock:
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

        # .flac too, not just .wav — post-compression (task_compress_stems)
        # converts stem WAVs to FLAC and deletes the WAV, so a WAV-only check
        # could never fire for any song whose stems have already been
        # compressed, even though full.wav is genuinely redundant by then.
        # Each candidate is also health-probed (_stem_is_healthy), not just
        # existence-checked — a zero-byte or truncated stem must not count
        # as "safe to delete the last full-length copy over".
        stems_present = [
            s for s in self.STEM_NAMES
            if _stem_is_healthy(song_path / f"{s}.flac") or _stem_is_healthy(song_path / f"{s}.wav")
        ]
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
            with self._index_lock:
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

    def prune_enhanced(self, name: str, force: bool = False) -> bool:
        """
        Delete  full_enhanced.wav  for a song if all four stems are present.

        full_enhanced.wav is the intermediate audio separate_song_stems()
        writes before handing it to Demucs — once real stems exist it's
        redundant (audio_source.py's resolve_source_file() only reaches for
        it when full.wav AND stems are both absent). Nothing previously
        cleaned it up, so every successfully-split song was silently left
        carrying a full duplicate of its own audio: on the live library this
        was ~4 GB across ~130 already-completed songs and climbing with every
        new one, on top of an already near-capacity library.

        Parameters
        ----------
        name : str
            Song name.
        force : bool
            Delete full_enhanced.wav even if not all stems are present.

        Returns
        -------
        bool : True if full_enhanced.wav was deleted, False otherwise.
        """
        if _KEEP_RAW and not force:
            log.debug("prune_enhanced skipped for '%s': keep_raw_after_separation=true", name)
            return False

        song_path = self._dir(name)
        enhanced_wav = song_path / "full_enhanced.wav"
        if not enhanced_wav.exists():
            return False

        # Health-probed, not just existence-checked — see _stem_is_healthy's
        # docstring: a zero-byte or truncated stem must not count as "safe
        # to delete the last full-length copy over".
        stems_present = [
            s for s in self.STEM_NAMES
            if _stem_is_healthy(song_path / f"{s}.wav") or _stem_is_healthy(song_path / f"{s}.flac")
        ]
        all_stems = len(stems_present) == len(self.STEM_NAMES)

        if not all_stems and not force:
            log.debug(
                "prune_enhanced skipped for '%s': only %d/%d stems present",
                name, len(stems_present), len(self.STEM_NAMES),
            )
            return False

        try:
            size_mb = enhanced_wav.stat().st_size / (1024 * 1024)
            enhanced_wav.unlink()
            log.info("🗑️  Pruned full_enhanced.wav for '%s' (freed %.1f MB)", name, size_mb)
            with self._index_lock:
                index = self._load_index()
                if name in index:
                    index[name]["size_bytes"] = sum(
                        f.stat().st_size for f in song_path.rglob("*") if f.is_file()
                    )
                    self._save_index(index)
            return True
        except Exception as e:
            log.warning("Could not prune full_enhanced.wav for '%s': %s", name, e)
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
          1. Only delete  full.wav  / full_enhanced.wav  for songs that have
             all 4 stems (keeps stems so re-mixing still works)
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

        # Pass 1: delete only full.wav / full_enhanced.wav, and only for
        # songs that have all 4 stems. force=True here would bypass that
        # check entirely — for a song with full.wav but zero stems (the
        # common in-progress-download shape), that deletes its only audio
        # outright, permanently, with no user action required to trigger it
        # (auto_evict_on_download can fire this mid-download). This is the
        # exact bug fixed for the manual /library/storage/prune endpoint
        # (see storage_prune() in routers/library.py) — force=False here
        # applies the same fix to the automatic eviction path.
        for entry in songs:
            if current <= target:
                break
            song_path = Path(entry.path)
            full_wav = song_path / "full.wav"
            enhanced_wav = song_path / "full_enhanced.wav"
            if dry_run:
                if full_wav.exists():
                    log.info("[DRY RUN] Would prune full.wav: %s (%.1f MB)",
                             entry.name, full_wav.stat().st_size / 1e6)
                if enhanced_wav.exists():
                    log.info("[DRY RUN] Would prune full_enhanced.wav: %s (%.1f MB)",
                             entry.name, enhanced_wav.stat().st_size / 1e6)
                continue
            if full_wav.exists():
                size = full_wav.stat().st_size
                if self.prune_raw(entry.name, force=False):
                    current -= size
                    if entry.name not in evicted:
                        evicted.append(entry.name)
            if current <= target:
                break
            if enhanced_wav.exists():
                size = enhanced_wav.stat().st_size
                if self.prune_enhanced(entry.name, force=False):
                    current -= size
                    if entry.name not in evicted:
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
                        with self._index_lock:
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
            "📚 Library Summary",
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
