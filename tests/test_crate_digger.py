"""
tests/test_crate_digger.py — Unit tests for scripts/core/crate_digger.py.

All tests operate on in-memory state — no disk I/O, no CLAP model, no librosa.
The module's load() is a no-op when index files are absent (test environment).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLAP_DIM = 512   # Must match crate_digger.CLAP_DIM


def _unit(seed: int = 0, dim: int = CLAP_DIM) -> np.ndarray:
    """Return a deterministic unit-norm vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _digger_with_songs(n: int = 5) -> "CrateDigger":
    """
    Return a fresh CrateDigger pre-loaded with n synthetic songs.
    Bypasses disk I/O completely.
    """
    from scripts.core.crate_digger import CrateDigger, CLAP_DIM
    d = CrateDigger.__new__(CrateDigger)
    import threading
    d._lock = threading.RLock()
    d._backend = "clap"
    d._built_at = "2026-01-01T00:00:00Z"

    embs = np.stack([_unit(i) for i in range(n)], axis=0)
    meta = [
        {
            "name": f"Song_{i}",
            "bpm":  120.0 + i * 2,
            "key":  "C major",
            "camelot": f"{i + 1}A" if i < 12 else "1A",
            "energy": 0.4 + i * 0.1,
        }
        for i in range(n)
    ]
    d._embeddings = embs
    d._meta = meta
    d._name_to_idx = {m["name"]: i for i, m in enumerate(meta)}
    return d


# ---------------------------------------------------------------------------
# TestCrateResult
# ---------------------------------------------------------------------------

class TestCrateResult:
    def test_dataclass_fields(self):
        from scripts.core.crate_digger import CrateResult
        r = CrateResult(name="x", score=0.8)
        assert r.name == "x"
        assert r.score == 0.8
        assert r.backend == "clap"

    def test_default_backend_is_clap(self):
        from scripts.core.crate_digger import CrateResult
        r = CrateResult(name="y", score=0.5)
        assert r.backend == "clap"


# ---------------------------------------------------------------------------
# TestFindSimilarByName
# ---------------------------------------------------------------------------

class TestFindSimilarByName:
    def test_returns_list(self):
        d = _digger_with_songs(5)
        results = d.find_similar(query_name="Song_0", k=3)
        assert isinstance(results, list)

    def test_excludes_query_song(self):
        d = _digger_with_songs(5)
        results = d.find_similar(query_name="Song_0", k=10)
        names = [r.name for r in results]
        assert "Song_0" not in names

    def test_returns_at_most_k_results(self):
        d = _digger_with_songs(10)
        results = d.find_similar(query_name="Song_0", k=4)
        assert len(results) <= 4

    def test_scores_in_range(self):
        d = _digger_with_songs(5)
        results = d.find_similar(query_name="Song_0", k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_sorted_descending(self):
        d = _digger_with_songs(8)
        results = d.find_similar(query_name="Song_0", k=8)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_query_name_returns_empty(self):
        d = _digger_with_songs(5)
        results = d.find_similar(query_name="DoesNotExist", k=3)
        assert results == []


# ---------------------------------------------------------------------------
# TestFindSimilarByText — fallback path (CLAP absent)
# ---------------------------------------------------------------------------

class TestFindSimilarByTextFallback:
    """Test the keyword-fallback text search (no CLAP needed)."""

    def test_text_matches_song_name(self):
        """Song_2 should surface when query contains '2'."""
        from scripts.core.crate_digger import CrateDigger, CLAP_DIM
        import threading

        d = CrateDigger.__new__(CrateDigger)
        d._lock = threading.RLock()
        d._backend = "music_index"   # simulate no-CLAP mode
        d._built_at = "2026-01-01T00:00:00Z"

        embs = np.stack([_unit(i) for i in range(4)], axis=0)
        meta = [
            {"name": "Cool Track",  "bpm": 120, "key": "", "camelot": "", "energy": 0.5},
            {"name": "Dark Bass",   "bpm": 128, "key": "", "camelot": "", "energy": 0.7},
            {"name": "Bright Vocal","bpm": 130, "key": "", "camelot": "", "energy": 0.6},
            {"name": "Acid House",  "bpm": 135, "key": "", "camelot": "", "energy": 0.8},
        ]
        d._embeddings = embs
        d._meta = meta
        d._name_to_idx = {m["name"]: i for i, m in enumerate(meta)}

        # "dark" should match "Dark Bass"
        result_vec = d._text_keyword_fallback(
            "dark", meta, d._name_to_idx, embs
        )
        assert result_vec is not None
        # The returned vector should be the embedding of "Dark Bass" (index 1)
        assert np.allclose(result_vec, embs[1])

    def test_no_match_returns_none(self):
        d = _digger_with_songs(3)
        result_vec = d._text_keyword_fallback(
            "xyzzy_no_match", d._meta, d._name_to_idx, d._embeddings
        )
        assert result_vec is None


# ---------------------------------------------------------------------------
# TestFilters
# ---------------------------------------------------------------------------

class TestFilters:
    def test_camelot_filter(self):
        d = _digger_with_songs(10)
        results = d.find_similar(query_name="Song_0", k=10, camelot_filter="2A")
        for r in results:
            assert r.camelot == "2A"

    def test_bpm_range_filter(self):
        d = _digger_with_songs(10)
        results = d.find_similar(query_name="Song_0", k=10, bpm_range=(122, 126))
        for r in results:
            if r.bpm > 0:
                assert 122 <= r.bpm <= 126

    def test_combined_filters_may_return_fewer_than_k(self):
        d = _digger_with_songs(10)
        # Very tight filter — may not match k songs
        results = d.find_similar(
            query_name="Song_0", k=10,
            camelot_filter="9A",
            bpm_range=(140, 145),
        )
        assert len(results) <= 10

    def test_no_filter_returns_up_to_k(self):
        d = _digger_with_songs(10)
        results = d.find_similar(query_name="Song_0", k=5)
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# TestEmptyIndex
# ---------------------------------------------------------------------------

class TestEmptyIndex:
    def test_empty_index_returns_empty_list(self):
        from scripts.core.crate_digger import CrateDigger
        import threading
        d = CrateDigger.__new__(CrateDigger)
        d._lock = threading.RLock()
        d._embeddings = None
        d._meta = []
        d._name_to_idx = {}
        d._built_at = None
        d._backend = "none"
        results = d.find_similar(query_name="anything", k=5)
        assert results == []

    def test_is_empty_when_no_embeddings(self):
        from scripts.core.crate_digger import CrateDigger
        import threading
        d = CrateDigger.__new__(CrateDigger)
        d._lock = threading.RLock()
        d._embeddings = None
        d._meta = []
        d._name_to_idx = {}
        d._built_at = None
        d._backend = "none"
        assert d.is_empty() is True

    def test_is_not_empty_after_indexing(self):
        d = _digger_with_songs(3)
        assert d.is_empty() is False


# ---------------------------------------------------------------------------
# TestGetStats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_stats_fields(self):
        d = _digger_with_songs(5)
        s = d.get_stats()
        assert "n_songs" in s
        assert "backend" in s
        assert "embedding_dim" in s
        assert s["n_songs"] == 5
        assert s["embedding_dim"] == CLAP_DIM

    def test_stats_backend_preserved(self):
        d = _digger_with_songs(3)
        d._backend = "clap"
        s = d.get_stats()
        assert s["backend"] == "clap"


# ---------------------------------------------------------------------------
# TestModuleSingleton
# ---------------------------------------------------------------------------

class TestModuleSingleton:
    def test_get_digger_returns_crate_digger(self):
        from scripts.core.crate_digger import get_digger, CrateDigger
        d = get_digger()
        assert isinstance(d, CrateDigger)

    def test_get_digger_is_idempotent(self):
        from scripts.core.crate_digger import get_digger
        d1 = get_digger()
        d2 = get_digger()
        assert d1 is d2


# ---------------------------------------------------------------------------
# TestIndexLibraryEdgeCases — no real audio needed
# ---------------------------------------------------------------------------

class TestIndexLibraryEdgeCases:
    def test_empty_library_returns_zero(self):
        from scripts.core.crate_digger import CrateDigger
        import threading
        d = CrateDigger.__new__(CrateDigger)
        d._lock = threading.RLock()
        d._embeddings = None
        d._meta = []
        d._name_to_idx = {}
        d._built_at = None
        d._backend = "none"

        with tempfile.TemporaryDirectory() as td:
            n = d.index_library(library_dir=Path(td))
        assert n == 0
