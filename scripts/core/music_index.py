"""
scripts/core/music_index.py — Music Library RAG Vector Index

Concept (the "hash map" analogy):
  Hash map:    hash(key)    → O(1) bucket lookup
  This index:  embed(song)  → pre-computed 35-dim vector → cosine similarity matmul

Every song gets a normalised 35-dimensional feature vector stored on disk.
On startup it loads into a numpy matrix in memory.  Queries are a single
weighted cosine-similarity matmul — sub-millisecond on hundreds of songs.

Vector layout (35 dims):
  dim  0        bpm_norm          bpm / 200.0 (clipped [0,1])
  dims 1–12     key_onehot[12]    one-hot for root note C=0 … B=11
  dim  13       mode              1.0 = major, 0.0 = minor
  dim  14       energy_mean       [0,1]
  dim  15       energy_std        [0,1]
  dim  16       drop_norm         drop_position / duration  (0.6 if unknown)
  dim  17       danceability      [0,1]
  dim  18       beat_strength     [0,1]
  dim  19       tempo_stability   [0,1]
  dim  20       vocal_density     [0,1]
  dim  21       centroid_norm     centroid_hz / 8000.0
  dim  22       rolloff_norm      rolloff_hz  / 16000.0
  dims 23–34    chroma[12]        mean chroma, L2-normalised

Default segment weights (applied before cosine similarity):
  BPM          0.40   — tempo is the #1 DJ-mixing constraint
  Key one-hot  0.30   — harmonic compatibility (Camelot wheel)
  Mode         0.05   — major / minor feel
  Energy       0.10   — dynamics match
  Rhythm       0.08   — groove compatibility
  Vocal        0.03   — avoid vocal clash
  Spectral     0.02   — timbre similarity (tie-breaker)
  Chroma       0.02   — harmonic colour (fine-grained key match)

Public API:
  upsert_song(song_name)         → add / refresh one song
  remove_song(song_name)         → remove from index
  search(song_name, k, weights)  → top-k similar songs with breakdown
  rebuild(library_dir, cb)       → rebuild entire library
  get_stats()                    → index size, last rebuild time
  get_index()                    → module-level singleton

Index file: data/music_index.json  (relative to project root)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

try:
    from scripts.core.paths import LIBRARY_DIR, DATA_DIR
except Exception:
    _here = Path(__file__).parents[2]
    LIBRARY_DIR = _here / "library"
    DATA_DIR    = _here / "data"

INDEX_PATH = DATA_DIR / "music_index.json"

# ---------------------------------------------------------------------------
# Vector layout constants
# ---------------------------------------------------------------------------

VECTOR_DIMS = 35

_NOTE_IDX: Dict[str, int] = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7,
    'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}

# Default weights — one per vector dimension
_DEFAULT_WEIGHTS = np.array([
    0.40,                    # dim 0: bpm_norm
    *([0.025] * 12),         # dims 1-12: key_onehot  (0.025×12 = 0.30 total)
    0.05,                    # dim 13: mode
    0.04, 0.03, 0.03,        # dims 14-16: energy
    0.03, 0.03, 0.02,        # dims 17-19: rhythm
    0.03,                    # dim 20: vocal_density
    0.01, 0.01,              # dims 21-22: spectral
    *([0.0017] * 12),        # dims 23-34: chroma (0.0017×12 ≈ 0.02 total)
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Vector construction
# ---------------------------------------------------------------------------

def _build_vector(data: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Build a 35-dim float32 feature vector from a metadata dict.

    ``data`` can come from meta.json, a MusicVector.to_dict(), or any dict
    that has at least {"bpm": <float>}.  Missing fields get safe defaults so
    we never crash on partial data.
    """
    try:
        vec = np.zeros(VECTOR_DIMS, dtype=np.float32)

        # ── dim 0: BPM ────────────────────────────────────────────────────
        bpm = float(data.get("bpm") or 0.0)
        if bpm <= 0:
            return None   # can't build a useful vector without BPM
        vec[0] = float(np.clip(bpm / 200.0, 0.0, 1.0))

        # ── dims 1-12: key one-hot ────────────────────────────────────────
        key_name = str(data.get("key") or data.get("key_name") or "C")
        idx = _NOTE_IDX.get(key_name, 0)
        vec[1 + idx] = 1.0

        # ── dim 13: mode ──────────────────────────────────────────────────
        mode = str(data.get("mode") or "major").lower()
        vec[13] = 1.0 if mode == "major" else 0.0

        # ── dims 14-16: energy ────────────────────────────────────────────
        vec[14] = float(np.clip(data.get("energy_mean") or 0.5, 0, 1))
        vec[15] = float(np.clip(data.get("energy_std")  or 0.2, 0, 1))
        # drop position normalised to [0,1] relative to track duration
        dur = float(data.get("duration") or data.get("duration_sec") or 0.0)
        drop = float(data.get("drop_position") or data.get("drop_position_sec") or 0.0)
        vec[16] = float(np.clip(drop / dur, 0, 1)) if dur > 0 and drop > 0 else 0.6

        # ── dims 17-19: rhythm ────────────────────────────────────────────
        vec[17] = float(np.clip(data.get("danceability")    or 0.5, 0, 1))
        vec[18] = float(np.clip(data.get("beat_strength")   or 0.5, 0, 1))
        vec[19] = float(np.clip(data.get("tempo_stability") or 0.5, 0, 1))

        # ── dim 20: vocal density ─────────────────────────────────────────
        vec[20] = float(np.clip(data.get("vocal_density") or 0.5, 0, 1))

        # ── dims 21-22: spectral brightness ──────────────────────────────
        centroid = float(data.get("spectral_centroid_hz") or
                         data.get("spectral_centroid")    or 2000.0)
        rolloff  = float(data.get("spectral_rolloff_hz")  or
                         data.get("spectral_rolloff")     or 8000.0)
        vec[21] = float(np.clip(centroid / 8000.0,  0, 1))
        vec[22] = float(np.clip(rolloff  / 16000.0, 0, 1))

        # ── dims 23-34: chroma vector ─────────────────────────────────────
        chroma = data.get("chroma_vector") or []
        if chroma and len(chroma) >= 12:
            c = np.array(chroma[:12], dtype=np.float32)
            norm = float(np.linalg.norm(c))
            if norm > 1e-8:
                c = c / norm
            vec[23:35] = c

        return vec.astype(np.float32)

    except Exception as exc:
        log.debug("_build_vector failed: %s", exc)
        return None


def _quick_features(wav_path: Path, sr: int = 22050, duration: float = 30.0) -> Dict[str, Any]:
    """
    Quick 30-second audio analysis to populate index fields for un-analysed songs.
    Returns a partial metadata dict.  Falls back gracefully on any error.
    """
    data: Dict[str, Any] = {}
    try:
        import librosa
        audio, _ = librosa.load(str(wav_path), sr=sr, mono=True, duration=duration)

        # BPM
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        data["bpm"] = round(float(np.atleast_1d(tempo)[0]), 1)

        # Key (Krumhansl-Schmuckler on CQT chroma)
        try:
            from scripts.core.music_intelligence import detect_key
            key_name, mode, _, confidence = detect_key(audio, sr)
            data["key"]  = key_name
            data["mode"] = mode
        except Exception:
            data["key"]  = "C"
            data["mode"] = "major"

        # Energy
        rms = librosa.feature.rms(y=audio)[0]
        data["energy_mean"] = round(float(rms.mean()), 4)
        data["energy_std"]  = round(float(rms.std()),  4)

        # Rhythm
        if len(beats) >= 4:
            ibi = np.diff(beats.astype(float))
            data["tempo_stability"] = round(float(np.clip(
                1.0 - np.std(ibi) / (np.mean(ibi) + 1e-8), 0, 1
            )), 4)
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            data["beat_strength"] = round(float(np.clip(
                np.mean(onset_env[beats]) / (np.max(onset_env) + 1e-8), 0, 1
            )), 4)
        else:
            data["tempo_stability"] = 0.5
            data["beat_strength"]   = 0.5

        # Danceability (simplified)
        data["danceability"] = round(float(np.clip(
            0.5 * data.get("tempo_stability", 0.5) +
            0.5 * data.get("beat_strength",   0.5), 0, 1
        )), 4)

        # Spectral
        try:
            from scripts.core.gpu import gpu_stft
            S = np.abs(gpu_stft(audio))
        except (ImportError, Exception):
            S = np.abs(librosa.stft(audio))
        data["spectral_centroid_hz"] = round(
            float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))), 1)
        data["spectral_rolloff_hz"]  = round(
            float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))), 1)

        # Vocal density (spectral proxy)
        freqs = librosa.fft_frequencies(sr=sr)
        vi = (freqs >= 300) & (freqs <= 3400)
        bi = freqs <= 200
        ve = S[vi].mean(axis=0) if vi.any() else np.ones(S.shape[1])
        be = S[bi].mean(axis=0) if bi.any() else np.ones(S.shape[1]) * 1e-8
        data["vocal_density"] = round(float(((ve / (be + 1e-8)) > 3.0).mean()), 4)

        # Chroma
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        data["chroma_vector"] = chroma.mean(axis=1).tolist()

    except Exception as exc:
        log.debug("_quick_features failed for %s: %s", wav_path, exc)

    return data


# ---------------------------------------------------------------------------
# Index record
# ---------------------------------------------------------------------------

class _SongRecord:
    """One song's entry in the index."""
    __slots__ = ("name", "vector", "meta", "indexed_at")

    def __init__(self, name: str, vector: np.ndarray, meta: Dict[str, Any],
                 indexed_at: float) -> None:
        self.name       = name
        self.vector     = vector.astype(np.float32)
        self.meta       = meta
        self.indexed_at = indexed_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":       self.name,
            "vector":     self.vector.tolist(),
            "meta":       self.meta,
            "indexed_at": self.indexed_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_SongRecord":
        return cls(
            name       = d["name"],
            vector     = np.array(d["vector"], dtype=np.float32),
            meta       = d.get("meta", {}),
            indexed_at = float(d.get("indexed_at", 0.0)),
        )


# ---------------------------------------------------------------------------
# MusicIndex class
# ---------------------------------------------------------------------------

class MusicIndex:
    """
    In-memory music feature vector index with disk persistence.

    Thread-safe: uses a single RW lock via threading.RLock.
    All mutation operations call _save() immediately after writing.
    """

    def __init__(self) -> None:
        self._lock   = threading.RLock()
        self._songs: Dict[str, _SongRecord] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load index from disk. Silent no-op if file doesn't exist."""
        try:
            if INDEX_PATH.exists():
                raw = json.loads(INDEX_PATH.read_text())
                for entry in raw.get("songs", {}).values():
                    rec = _SongRecord.from_dict(entry)
                    self._songs[rec.name] = rec
                log.info("MusicIndex loaded: %d songs from %s", len(self._songs), INDEX_PATH)
        except Exception as exc:
            log.warning("MusicIndex._load failed (%s) — starting fresh", exc)
            self._songs = {}

    def _save(self) -> None:
        """Persist index to disk."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "version":    2,
                "saved_at":   time.time(),
                "songs":      {name: rec.to_dict() for name, rec in self._songs.items()},
            }
            INDEX_PATH.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            log.warning("MusicIndex._save failed: %s", exc)

    # ── Upsert ───────────────────────────────────────────────────────────────

    def upsert_song(self, song_name: str) -> bool:
        """
        Compute (or refresh) the feature vector for ``song_name`` and store it.

        Data sources (in priority order):
          1. meta.json in the song directory (written by task_analyze / recommend)
          2. Quick 30-second audio analysis from full.wav
          3. Quick analysis from vocals.wav (if full.wav absent)

        Returns True on success, False if no audio was found.
        """
        song_dir = LIBRARY_DIR / song_name

        # ── Load any existing cached metadata ─────────────────────────────
        meta: Dict[str, Any] = {}
        meta_path = song_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass

        # ── Try to build vector from meta alone first ─────────────────────
        vec = _build_vector(meta)

        # ── If vector is None or partial, do quick audio analysis ─────────
        if vec is None:
            wav = song_dir / "full.wav"
            if not wav.exists():
                wav = song_dir / "vocals.wav"
            if not wav.exists():
                log.debug("upsert_song: no audio for %s", song_name)
                return False

            fresh = _quick_features(wav)
            meta.update(fresh)

            # Persist enriched meta back to meta.json
            try:
                meta_path.write_text(json.dumps(meta))
            except Exception:
                pass

            vec = _build_vector(meta)

        if vec is None:
            log.debug("upsert_song: could not build vector for %s", song_name)
            return False

        # ── Store ─────────────────────────────────────────────────────────
        record_meta = {
            "bpm":          meta.get("bpm"),
            "key":          meta.get("key") or meta.get("key_name"),
            "mode":         meta.get("mode"),
            "camelot":      meta.get("camelot"),
            "genre":        meta.get("genre"),
            "danceability": meta.get("danceability"),
            "vocal_density":meta.get("vocal_density"),
        }
        with self._lock:
            self._songs[song_name] = _SongRecord(
                name       = song_name,
                vector     = vec,
                meta       = {k: v for k, v in record_meta.items() if v is not None},
                indexed_at = time.time(),
            )
            self._save()

        log.info("MusicIndex: upserted '%s' (bpm=%.1f)", song_name,
                 float(meta.get("bpm") or 0))
        return True

    # ── Remove ───────────────────────────────────────────────────────────────

    def remove_song(self, song_name: str) -> bool:
        """Remove a song from the index.  Returns True if it existed."""
        with self._lock:
            if song_name in self._songs:
                del self._songs[song_name]
                self._save()
                return True
        return False

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        song_name: str,
        k: int = 5,
        weights: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k most similar songs to ``song_name``.

        Algorithm
        ---------
        1. Look up the query song's pre-computed vector.
        2. Apply per-dimension weights to the entire matrix.
        3. Compute cosine similarity of weighted query vs weighted matrix.
        4. Sort, exclude self, return top-k with rich metadata.

        Parameters
        ----------
        song_name : str
            Must already be in the index (call upsert_song first).
        k : int
            Number of results to return.
        weights : np.ndarray, optional
            35-dim weight override.  Defaults to _DEFAULT_WEIGHTS.

        Returns
        -------
        List of dicts::
            [
              {
                "name": "Artist - Title",
                "score": 0.94,
                "bpm":  128.0,
                "key":  "A",
                "mode": "minor",
                "camelot": "8A",
                "genre": "techno",
                "breakdown": {
                    "bpm_sim":    0.97,
                    "key_sim":    0.88,
                    "energy_sim": 0.91,
                    "rhythm_sim": 0.85,
                    "timbre_sim": 0.72,
                },
              }, ...
            ]
        """
        with self._lock:
            if song_name not in self._songs:
                return []
            if len(self._songs) < 2:
                return []

            w    = weights if weights is not None else _DEFAULT_WEIGHTS
            names = list(self._songs.keys())
            mat   = np.stack([self._songs[n].vector for n in names])

            # Weighted cosine similarity (GPU-accelerated when available)
            q_raw = self._songs[song_name].vector
            q     = q_raw * w
            m     = mat  * w

            try:
                from scripts.core.gpu import gpu_cosine_similarity
                sims = gpu_cosine_similarity(q, m)
            except (ImportError, Exception):
                # CPU fallback
                q_norm = np.linalg.norm(q)
                m_norm = np.linalg.norm(m, axis=1)
                if q_norm < 1e-10:
                    return []
                sims = m.dot(q) / (m_norm * q_norm + 1e-10)

        results = []
        for i, name in enumerate(names):
            if name == song_name:
                continue
            rec  = self._songs[name]
            sim  = float(sims[i])
            v    = rec.vector

            # Per-segment similarity breakdown
            # Reconstruct sub-scores against query vector (not weighted)
            q_raw_vec = self._songs[song_name].vector

            def _cos(a_slice, b_slice):
                a, b = a_slice.astype(float), b_slice.astype(float)
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na < 1e-8 or nb < 1e-8:
                    return 0.5
                return float(np.clip(np.dot(a, b) / (na * nb), 0, 1))

            breakdown = {
                "bpm_sim":    round(1.0 - abs(float(v[0]) - float(q_raw_vec[0])), 4),
                "key_sim":    round(_cos(v[1:13], q_raw_vec[1:13]), 4),
                "energy_sim": round(_cos(v[14:17], q_raw_vec[14:17]), 4),
                "rhythm_sim": round(_cos(v[17:20], q_raw_vec[17:20]), 4),
                "timbre_sim": round(_cos(v[21:23], q_raw_vec[21:23]), 4),
            }

            results.append({
                "name":       name,
                "score":      round(sim, 4),
                "bpm":        rec.meta.get("bpm"),
                "key":        rec.meta.get("key"),
                "mode":       rec.meta.get("mode"),
                "camelot":    rec.meta.get("camelot"),
                "genre":      rec.meta.get("genre"),
                "danceability":    rec.meta.get("danceability"),
                "vocal_density":   rec.meta.get("vocal_density"),
                "breakdown":  breakdown,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    # ── Rebuild ──────────────────────────────────────────────────────────────

    def rebuild(
        self,
        library_dir: Optional[Path] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Reindex every song in the library directory.

        Parameters
        ----------
        library_dir : Path, optional
            Defaults to LIBRARY_DIR.
        progress_cb : callable, optional
            Called with (fraction_0_to_1, message_str).

        Returns
        -------
        dict with keys: total, indexed, failed, duration_sec
        """
        lib = library_dir or LIBRARY_DIR
        dirs = [d for d in lib.iterdir() if d.is_dir()]

        total   = len(dirs)
        indexed = 0
        failed  = []
        t0      = time.time()

        for i, d in enumerate(dirs):
            frac = i / max(total, 1)
            msg  = f"Indexing [{i+1}/{total}]: {d.name[:40]}"
            if progress_cb:
                progress_cb(frac, msg)
            ok = self.upsert_song(d.name)
            if ok:
                indexed += 1
            else:
                failed.append(d.name)

        if progress_cb:
            progress_cb(1.0, f"Index rebuild done — {indexed}/{total} songs indexed")

        return {
            "total":        total,
            "indexed":      indexed,
            "failed":       len(failed),
            "failed_names": failed[:20],
            "duration_sec": round(time.time() - t0, 1),
        }

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return info about the current index state."""
        with self._lock:
            n = len(self._songs)
            newest = max(
                (r.indexed_at for r in self._songs.values()), default=0.0
            )
        return {
            "indexed_songs": n,
            "index_path":    str(INDEX_PATH),
            "last_updated":  datetime.fromtimestamp(newest, tz=timezone.utc).isoformat()
                             if newest else None,
            "vector_dims":   VECTOR_DIMS,
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self._songs)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_index_instance: Optional[MusicIndex] = None
_index_lock = threading.Lock()


def get_index() -> MusicIndex:
    """Return the module-level MusicIndex singleton (thread-safe)."""
    global _index_instance
    if _index_instance is None:
        with _index_lock:
            if _index_instance is None:
                _index_instance = MusicIndex()
    return _index_instance
