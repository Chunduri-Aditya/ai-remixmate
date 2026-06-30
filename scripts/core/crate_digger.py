"""
scripts/core/crate_digger.py — CLAP 512-D Semantic Similarity Search

CrateDigger replaces the handcrafted 35-D embedding index with LAION-CLAP
512-dimensional embeddings, enabling text-to-audio similarity search
("find me something dark and hypnotic") in addition to audio-to-audio.

Backends
--------
CLAP (preferred)
    Requires: pip install laion_clap  (~300 MB model, ~1s per 3min clip)
    Model:    HTSAT-RoBERTa checkpoint (music_audioset_epoch_15_esc_90.14.pt)
    Encodes audio and/or natural-language text into the same 512-D latent space.
    Cosine similarity is computed between the query and all indexed songs.

music_index.py fallback
    Falls back to the existing 35-D handcrafted index when laion_clap is absent.
    Text queries fall back to a keyword match on song name.

Usage
-----
    from scripts.core.crate_digger import CrateDigger

    digger = CrateDigger()
    digger.index_library()                          # build from library dir
    results = digger.find_similar(
        query_audio=audio,                          # np.ndarray or None
        query_text="dark minimal techno",           # str or None
        k=10,
        camelot_filter="10A",                       # optional Camelot fence
        bpm_range=(125, 135),                       # optional BPM window
    )

Index file: data/clap_index.npy + data/clap_index_meta.json

Config
------
    config.yaml:
        models:
            clap_model: null   # null → auto-download; or path to .pt checkpoint
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.core.paths import DATA_DIR, LIBRARY_DIR

log = logging.getLogger(__name__)

_INDEX_EMB_PATH  = DATA_DIR / "clap_index.npy"
_INDEX_META_PATH = DATA_DIR / "clap_index_meta.json"

# CLAP embedding dimension
CLAP_DIM = 512


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CrateResult:
    """One search result from CrateDigger.find_similar()."""
    name: str
    score: float            # cosine similarity [0, 1]
    bpm: float = 0.0
    key: str = ""
    camelot: str = ""
    energy: float = 0.0
    backend: str = "clap"   # "clap" | "music_index"


# ---------------------------------------------------------------------------
# CLAP helpers
# ---------------------------------------------------------------------------

def _load_clap_model():
    """
    Load the LAION-CLAP model.

    Raises ImportError if laion_clap is not installed.
    Returns the model object.
    """
    try:
        import laion_clap  # type: ignore
    except ImportError as e:
        raise ImportError(
            "laion_clap is required for CrateDigger. "
            "Install with: pip install laion_clap"
        ) from e

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-RoBERTa")

    # Try to load from config path, otherwise auto-download
    model_path = None
    try:
        from scripts.core.config import cfg
        mp = getattr(getattr(cfg, "models", None), "clap_model", None)
        if mp:
            from scripts.core.paths import MODELS_DIR
            candidate = MODELS_DIR / mp
            if candidate.exists():
                model_path = str(candidate)
    except Exception:
        pass

    if model_path:
        model.load_ckpt(model_path)
        log.info("[crate_digger] Loaded CLAP from %s", model_path)
    else:
        model.load_ckpt()   # auto-download to ~/.cache
        log.info("[crate_digger] CLAP model loaded (auto-downloaded)")

    return model


def _embed_audio_clap(model, audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Encode audio into a 512-D CLAP embedding.

    CLAP expects float32 audio at 48 kHz mono.  Resamples if needed.
    Returns a normalised (unit-norm) (512,) float32 vector.
    """
    import librosa

    target_sr = 48000
    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    mono = audio.astype(np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)

    # CLAP expects a list of audio arrays
    emb = model.get_audio_embedding_from_data([mono], use_tensor=False)
    emb = np.asarray(emb, dtype=np.float32)[0]
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-10 else emb


def _embed_text_clap(model, text: str) -> np.ndarray:
    """Encode a text query into a normalised 512-D CLAP embedding."""
    emb = model.get_text_embedding([text], use_tensor=False)
    emb = np.asarray(emb, dtype=np.float32)[0]
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-10 else emb


# ---------------------------------------------------------------------------
# CrateDigger class
# ---------------------------------------------------------------------------

class CrateDigger:
    """
    Semantic similarity search over the RemixMate library using CLAP embeddings.

    Thread-safe: a single RLock guards index reads and writes.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Embedding matrix: (n_songs, CLAP_DIM)
        self._embeddings: Optional[np.ndarray] = None
        # Metadata list parallel to embedding rows
        self._meta: List[Dict] = []
        # Song name → row index for O(1) lookup
        self._name_to_idx: Dict[str, int] = {}
        self._built_at: Optional[str] = None
        self._backend: str = "none"

        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load index from disk if both files exist."""
        if not _INDEX_EMB_PATH.exists() or not _INDEX_META_PATH.exists():
            return
        try:
            emb = np.load(str(_INDEX_EMB_PATH))
            meta = json.loads(_INDEX_META_PATH.read_text())
            with self._lock:
                self._embeddings = emb
                self._meta = meta.get("songs", [])
                self._name_to_idx = {s["name"]: i for i, s in enumerate(self._meta)}
                self._built_at = meta.get("built_at")
                self._backend = meta.get("backend", "clap")
            log.info(
                "[crate_digger] Loaded index: %d songs, backend=%s",
                len(self._meta), self._backend,
            )
        except Exception as exc:
            log.warning("[crate_digger] Failed to load CLAP index: %s", exc)

    def _save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(_INDEX_EMB_PATH), self._embeddings)
        meta = {
            "built_at": self._built_at,
            "backend": self._backend,
            "songs": self._meta,
        }
        _INDEX_META_PATH.write_text(json.dumps(meta, indent=2))

    # ── Index building ────────────────────────────────────────────────────────

    def index_library(
        self,
        library_dir: Optional[Path] = None,
        progress_cb=None,
        force: bool = False,
    ) -> int:
        """
        Build the CLAP embedding index for all songs in the library.

        Reads ``library/<song>/full.wav`` (or any FLAC/MP3 via librosa).
        Songs already indexed (same name) are skipped unless force=True.

        Parameters
        ----------
        library_dir : Path | None
            Override library root; defaults to LIBRARY_DIR from paths.py.
        progress_cb : callable(float) | None
            Progress callback called with [0.0, 1.0] fractions.
        force : bool
            If True, re-embed all songs even if already indexed.

        Returns
        -------
        int
            Number of songs indexed (newly added or updated).
        """
        lib = library_dir or LIBRARY_DIR

        # Try CLAP; fall back to 35-D index if not installed
        try:
            model = _load_clap_model()
            use_clap = True
        except ImportError:
            log.warning(
                "[crate_digger] laion_clap not installed — using music_index.py fallback"
            )
            use_clap = False

        song_dirs = sorted(
            (d for d in lib.iterdir() if d.is_dir()),
            key=lambda d: d.name,
        )
        if not song_dirs:
            return 0

        new_embeddings: List[np.ndarray] = []
        new_meta: List[Dict] = []
        n_indexed = 0

        with self._lock:
            existing = dict(self._name_to_idx)
            existing_emb = self._embeddings
            existing_meta = list(self._meta)

        for i, song_dir in enumerate(song_dirs):
            name = song_dir.name
            if not force and name in existing:
                continue

            # Find audio file
            audio_path = None
            for ext in ("full.wav", "full.flac", "full.mp3"):
                p = song_dir / ext
                if p.exists():
                    audio_path = p
                    break
            if audio_path is None:
                # No full audio — try individual stems
                for stem in ("vocals.flac", "vocals.wav", "other.flac", "other.wav"):
                    p = song_dir / stem
                    if p.exists():
                        audio_path = p
                        break
            if audio_path is None:
                log.debug("[crate_digger] Skipping %s — no audio file found", name)
                continue

            try:
                import librosa
                audio, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=90.0)
            except Exception as exc:
                log.warning("[crate_digger] Cannot load %s: %s", audio_path, exc)
                continue

            try:
                if use_clap:
                    emb = _embed_audio_clap(model, audio, sr)
                    backend = "clap"
                else:
                    emb = self._fallback_embed(name, audio, sr)
                    backend = "music_index"
            except Exception as exc:
                log.warning("[crate_digger] Embedding failed for %s: %s", name, exc)
                continue

            # Read analysis metadata
            meta_entry = {"name": name, "bpm": 0.0, "key": "", "camelot": "", "energy": 0.5}
            analysis_path = song_dir / "analysis.json"
            if analysis_path.exists():
                try:
                    d = json.loads(analysis_path.read_text())
                    meta_entry["bpm"]     = float(d.get("bpm", 0.0))
                    meta_entry["key"]     = str(d.get("key", ""))
                    meta_entry["camelot"] = str(d.get("camelot", ""))
                    meta_entry["energy"]  = float(d.get("energy", 0.5))
                except Exception:
                    pass

            new_embeddings.append(emb)
            new_meta.append(meta_entry)
            n_indexed += 1

            if progress_cb and song_dirs:
                progress_cb(i / len(song_dirs))

        if n_indexed == 0 and not force:
            return 0

        # Merge new into existing (for incremental updates)
        if existing_emb is not None and not force:
            # Keep existing songs not being re-indexed
            keep_idx = [i for i, m in enumerate(existing_meta) if m["name"] not in {m2["name"] for m2 in new_meta}]
            kept_emb  = existing_emb[keep_idx] if keep_idx else np.empty((0, CLAP_DIM), dtype=np.float32)
            kept_meta = [existing_meta[i] for i in keep_idx]
        else:
            kept_emb  = np.empty((0, CLAP_DIM), dtype=np.float32)
            kept_meta = []

        if new_embeddings:
            new_emb_arr = np.stack(new_embeddings, axis=0)
        else:
            new_emb_arr = np.empty((0, CLAP_DIM), dtype=np.float32)

        all_emb = np.concatenate([kept_emb, new_emb_arr], axis=0).astype(np.float32)
        all_meta = kept_meta + new_meta

        with self._lock:
            self._embeddings = all_emb
            self._meta = all_meta
            self._name_to_idx = {m["name"]: i for i, m in enumerate(all_meta)}
            self._built_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._backend = "clap" if use_clap else "music_index"
            self._save()

        log.info("[crate_digger] Index built: %d total songs (%d new)", len(all_meta), n_indexed)
        if progress_cb:
            progress_cb(1.0)
        return n_indexed

    def _fallback_embed(self, name: str, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Fallback 35-D → padded 512-D vector when CLAP is not installed.

        Embeds using the handcrafted feature logic from music_index.py,
        then zero-pads to 512 dims so the index format is consistent.
        """
        from scripts.core.music_index import get_index
        idx = get_index()
        idx.upsert_song(name)
        vec35 = idx._embeddings.get(name)
        if vec35 is None:
            return np.zeros(CLAP_DIM, dtype=np.float32)
        # Pad to CLAP_DIM
        padded = np.zeros(CLAP_DIM, dtype=np.float32)
        padded[: len(vec35)] = vec35
        # Normalise
        norm = np.linalg.norm(padded)
        return padded / norm if norm > 1e-10 else padded

    # ── Search ────────────────────────────────────────────────────────────────

    def find_similar(
        self,
        query_name: Optional[str] = None,
        query_audio: Optional[np.ndarray] = None,
        query_sr: int = 44100,
        query_text: Optional[str] = None,
        k: int = 10,
        camelot_filter: Optional[str] = None,
        bpm_range: Optional[Tuple[float, float]] = None,
    ) -> List[CrateResult]:
        """
        Find the k most similar songs to an audio or text query.

        At least one of (query_name, query_audio, query_text) must be provided.

        Parameters
        ----------
        query_name : str | None
            Name of a song already in the index to use as query.
        query_audio : np.ndarray | None
            Raw audio array to embed and search against.
        query_sr : int
            Sample rate of query_audio.
        query_text : str | None
            Natural-language text query ("dark minimal techno at 130 BPM").
            Ignored when laion_clap is not installed.
        k : int
            Number of results to return.
        camelot_filter : str | None
            If set, only return songs in this Camelot key (e.g. "10A").
        bpm_range : (float, float) | None
            If set, only return songs with BPM in [min, max].

        Returns
        -------
        list[CrateResult]
            Sorted by similarity descending.
        """
        with self._lock:
            if self._embeddings is None or len(self._meta) == 0:
                log.warning("[crate_digger] Index is empty — run index_library() first")
                return []
            emb_matrix = self._embeddings.copy()
            meta = list(self._meta)
            name_to_idx = dict(self._name_to_idx)
            backend = self._backend

        # ── Build query vector ────────────────────────────────────────────────
        query_vec: Optional[np.ndarray] = None

        if query_name and query_name in name_to_idx:
            query_vec = emb_matrix[name_to_idx[query_name]].copy()

        if query_audio is not None:
            if backend == "clap":
                try:
                    model = _load_clap_model()
                    av = _embed_audio_clap(model, query_audio, query_sr)
                    query_vec = av if query_vec is None else (query_vec + av) / 2.0
                except Exception as exc:
                    log.warning("[crate_digger] Audio embedding failed: %s", exc)
            else:
                # Fallback: use music_index search, return early
                return self._fallback_search(
                    query_name=query_name,
                    k=k,
                    camelot_filter=camelot_filter,
                    bpm_range=bpm_range,
                )

        if query_text is not None:
            if backend == "clap":
                try:
                    model = _load_clap_model()
                    tv = _embed_text_clap(model, query_text)
                    query_vec = tv if query_vec is None else (query_vec + tv) / 2.0
                except Exception as exc:
                    log.warning("[crate_digger] Text embedding failed: %s", exc)
            else:
                # Text search on name substring when CLAP unavailable
                if query_vec is None:
                    query_vec = self._text_keyword_fallback(
                        query_text, meta, name_to_idx, emb_matrix
                    )

        if query_vec is None:
            log.warning("[crate_digger] No valid query — provide query_name, query_audio, or query_text")
            return []

        # Normalise query
        qnorm = np.linalg.norm(query_vec)
        if qnorm > 1e-10:
            query_vec = query_vec / qnorm

        # ── Cosine similarity (dot product on unit vectors) ────────────────────
        scores = emb_matrix @ query_vec  # (n,)

        # ── Apply filters ─────────────────────────────────────────────────────
        results: List[CrateResult] = []
        for idx in np.argsort(scores)[::-1]:
            m = meta[idx]
            if m["name"] == query_name:
                continue   # exclude the query song itself
            if camelot_filter and m.get("camelot") != camelot_filter:
                continue
            if bpm_range:
                bpm = float(m.get("bpm", 0.0))
                if bpm > 0 and not (bpm_range[0] <= bpm <= bpm_range[1]):
                    continue
            results.append(CrateResult(
                name=m["name"],
                score=float(np.clip(scores[idx], 0.0, 1.0)),
                bpm=float(m.get("bpm", 0.0)),
                key=str(m.get("key", "")),
                camelot=str(m.get("camelot", "")),
                energy=float(m.get("energy", 0.5)),
                backend=backend,
            ))
            if len(results) >= k:
                break

        return results

    def _fallback_search(
        self,
        query_name: Optional[str],
        k: int,
        camelot_filter: Optional[str],
        bpm_range: Optional[Tuple[float, float]],
    ) -> List[CrateResult]:
        """Use music_index.py for similarity when CLAP unavailable."""
        if query_name is None:
            return []
        try:
            from scripts.core.music_index import get_index
            hits = get_index().search(query_name, k=k * 3)
        except Exception as exc:
            log.warning("[crate_digger] music_index fallback search failed: %s", exc)
            return []

        results = []
        for h in hits:
            if camelot_filter and h.get("camelot") != camelot_filter:
                continue
            bpm = float(h.get("bpm", 0.0))
            if bpm_range and bpm > 0 and not (bpm_range[0] <= bpm <= bpm_range[1]):
                continue
            results.append(CrateResult(
                name=h["name"],
                score=float(h.get("score", 0.0)),
                bpm=bpm,
                key=str(h.get("key", "")),
                camelot=str(h.get("camelot", "")),
                energy=float(h.get("energy", 0.5)),
                backend="music_index",
            ))
            if len(results) >= k:
                break
        return results

    def _text_keyword_fallback(
        self,
        query_text: str,
        meta: List[Dict],
        name_to_idx: Dict[str, int],
        emb_matrix: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        When CLAP is unavailable, find the best-matching song by keyword search
        on the query text vs song names, then return that song's embedding as
        the query vector.
        """
        words = set(query_text.lower().split())
        best_score = -1
        best_idx = None
        for i, m in enumerate(meta):
            name_words = set(m["name"].lower().replace("-", " ").replace("_", " ").split())
            overlap = len(words & name_words)
            if overlap > best_score:
                best_score = overlap
                best_idx = i
        if best_idx is not None and best_score > 0:
            return emb_matrix[best_idx].copy()
        return None

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "n_songs": len(self._meta),
                "backend": self._backend,
                "embedding_dim": int(self._embeddings.shape[1]) if self._embeddings is not None else 0,
                "built_at": self._built_at,
                "index_files": {
                    "embeddings": str(_INDEX_EMB_PATH),
                    "metadata": str(_INDEX_META_PATH),
                },
            }

    def is_empty(self) -> bool:
        with self._lock:
            return self._embeddings is None or len(self._meta) == 0


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_digger: Optional[CrateDigger] = None
_digger_lock = threading.Lock()


def get_digger() -> CrateDigger:
    """Return the module-level CrateDigger singleton (created on first call)."""
    global _digger
    if _digger is None:
        with _digger_lock:
            if _digger is None:
                _digger = CrateDigger()
    return _digger
