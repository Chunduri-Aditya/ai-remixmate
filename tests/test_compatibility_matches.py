"""
tests/test_compatibility_matches.py — strict ranked compatibility matches.

Pure cache-driven (no librosa): builds a tmp library of meta.json files and
asserts filtering, sorting, and the unanalyzed-skip accounting.
"""

import json
from pathlib import Path

from scripts.core.recommend import top_compatibility_matches


def _song(tmp: Path, name: str, bpm: float, camelot: str, key: str = "G",
          mode: str = "minor", energy: float = 0.6) -> Path:
    d = tmp / name
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({
        "bpm": bpm, "camelot": camelot, "key": key, "mode": mode,
        "energy_mean": energy, "vocal_density": 0.2,
        "spectral_centroid_hz": 2000.0, "genre": "house",
    }))
    return d


class TestTopCompatibilityMatches:
    def test_filters_sorts_and_counts(self, tmp_path):
        src = _song(tmp_path, "source", 126.0, "6A")
        _song(tmp_path, "perfect", 126.0, "6A")            # same key + bpm → top
        _song(tmp_path, "adjacent", 128.0, "7A")           # neighbour key → high
        _song(tmp_path, "clash", 90.0, "12B", key="B", mode="major", energy=0.1)
        (tmp_path / "unanalyzed").mkdir()                  # no meta.json → skipped

        out = top_compatibility_matches(src, tmp_path, min_score=85.0)
        names = [m["name"] for m in out["matches"]]
        scores = [m["score"] for m in out["matches"]]

        assert "clash" not in names, f"low-compat song leaked in: {out['matches']}"
        assert "perfect" in names
        assert scores == sorted(scores, reverse=True), "matches must be sorted best-first"
        assert all(s >= 85.0 for s in scores)
        assert out["scored"] == 3
        assert out["skipped_unanalyzed"] == 1

    def test_min_score_is_respected(self, tmp_path):
        src = _song(tmp_path, "source", 126.0, "6A")
        _song(tmp_path, "perfect", 126.0, "6A")
        strict = top_compatibility_matches(src, tmp_path, min_score=99.9)
        loose = top_compatibility_matches(src, tmp_path, min_score=0.0)
        assert len(strict["matches"]) <= len(loose["matches"])
        assert len(loose["matches"]) == 1

    def test_unanalyzed_source_errors_cleanly(self, tmp_path):
        src = tmp_path / "no-meta"
        src.mkdir()
        out = top_compatibility_matches(src, tmp_path)
        assert out["matches"] == []
        assert "error" in out

    def test_source_excluded_from_its_own_matches(self, tmp_path):
        src = _song(tmp_path, "source", 126.0, "6A")
        out = top_compatibility_matches(src, tmp_path, min_score=0.0)
        assert all(m["name"] != "source" for m in out["matches"])
