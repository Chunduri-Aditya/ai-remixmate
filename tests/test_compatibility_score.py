"""
tests/test_compatibility_score.py — Unit tests for
MetadataClient.compatibility_score().

REMIX_QUALITY_INSIGHTS.md finding #1 regression guard.

Confirmed root cause: compatibility_score() implemented an undocumented
formula (key*0.45 + bpm*0.40 + energy*0.15) instead of the project's own
research-backed SetFlow formula in docs/DJ_THEORY.md section 3:

    compatibility = 0.35*harmonic_match + 0.25*beat_alignment
                  + 0.15*energy_smoothness + 0.15*genre_proximity
                  + 0.10*timbral_similarity - vocal_clash_penalty

genre_proximity, timbral_similarity, and vocal_clash_penalty had zero
implementation anywhere in the codebase. vocal_clash_penalty in particular
is named as catching "the single most audible failure mode in DJ mixing:
two vocals playing over each other" — and there was no detection for it at
all despite Demucs stems (which make vocal-presence detection cheap)
already existing in the pipeline.

These tests lock in the fixed formula and reproduce two specific bugs
verified live during the session that produced REMIX_QUALITY_INSIGHTS.md:
a tritone key clash and a ~2.45x BPM gap each scored "compatible: true"
under the old flat overall>=0.55 threshold.
"""
from __future__ import annotations

import pytest

from scripts.core.track_metadata import MetadataClient, TrackMetadata


@pytest.fixture
def client():
    # Skip MetadataClient.__init__ — it opens a SQLite cache and provider
    # clients we don't need for pure scoring-function tests.
    return MetadataClient.__new__(MetadataClient)


class TestHardDealbreakers:
    """Cases that must never be marked 'compatible', regardless of how
    good the weighted average looks."""

    def test_tritone_key_clash_not_compatible(self, client):
        """Reproduced live: tritone key clash (key_score=0.0) with perfect
        bpm/energy used to average to overall=0.55 -> compatible=True."""
        a = TrackMetadata(bpm=128, camelot="8A", energy=0.6)
        b = TrackMetadata(bpm=128, camelot="2A", energy=0.6)
        result = client.compatibility_score(a, b)
        assert result["key_score"] == 0.0
        assert result["compatible"] is False

    def test_large_bpm_gap_not_compatible(self, client):
        """Reproduced live: a 63.8 vs 156.6 BPM pair (~2.45x, outside even
        the double-time window) used to average to overall=0.60 with
        key/energy perfect -> compatible=True."""
        a = TrackMetadata(bpm=63.8, camelot="8A", energy=0.6)
        b = TrackMetadata(bpm=156.6, camelot="8A", energy=0.6)
        result = client.compatibility_score(a, b)
        assert result["bpm_score"] == 0.0
        assert result["compatible"] is False

    def test_strong_dual_vocal_presence_not_compatible(self, client):
        """New veto: previously there was no vocal_clash_penalty at all, so
        two tracks with strong simultaneous vocals could score
        'compatible' purely on key/bpm/energy."""
        a = TrackMetadata(bpm=128, camelot="8A", energy=0.6, vocal_density=0.8)
        b = TrackMetadata(bpm=128, camelot="8A", energy=0.6, vocal_density=0.8)
        result = client.compatibility_score(a, b)
        assert result["vocal_clash_penalty"] >= 0.25
        assert result["compatible"] is False


class TestNewDimensions:
    def test_genre_proximity_identical_genres(self, client):
        a = TrackMetadata(bpm=128, camelot="8A", genres=["house"])
        b = TrackMetadata(bpm=128, camelot="8A", genres=["house"])
        result = client.compatibility_score(a, b)
        assert result["genre_proximity"] == 1.0

    def test_genre_proximity_disjoint_genres(self, client):
        a = TrackMetadata(bpm=128, camelot="8A", genres=["techno"])
        b = TrackMetadata(bpm=128, camelot="8A", genres=["hip-hop"])
        result = client.compatibility_score(a, b)
        assert result["genre_proximity"] == 0.0

    def test_genre_proximity_neutral_when_missing(self, client):
        """Missing genre data must not penalize the pair — neutral 0.5."""
        a = TrackMetadata(bpm=128, camelot="8A")
        b = TrackMetadata(bpm=128, camelot="8A")
        result = client.compatibility_score(a, b)
        assert result["genre_proximity"] == 0.5

    def test_timbral_similarity_close_centroids(self, client):
        a = TrackMetadata(bpm=128, camelot="8A", spectral_centroid_hz=2000.0)
        b = TrackMetadata(bpm=128, camelot="8A", spectral_centroid_hz=2100.0)
        result = client.compatibility_score(a, b)
        assert result["timbral_similarity"] > 0.9

    def test_timbral_similarity_distant_centroids(self, client):
        a = TrackMetadata(bpm=128, camelot="8A", spectral_centroid_hz=500.0)
        b = TrackMetadata(bpm=128, camelot="8A", spectral_centroid_hz=5000.0)
        result = client.compatibility_score(a, b)
        assert result["timbral_similarity"] < 0.5

    def test_timbral_similarity_neutral_when_missing(self, client):
        a = TrackMetadata(bpm=128, camelot="8A")
        b = TrackMetadata(bpm=128, camelot="8A")
        result = client.compatibility_score(a, b)
        assert result["timbral_similarity"] == 0.5

    def test_vocal_clash_penalty_zero_when_data_missing(self, client):
        """No vocal_density data on either track -> no penalty, not a
        worst-case assumption."""
        a = TrackMetadata(bpm=128, camelot="8A")
        b = TrackMetadata(bpm=128, camelot="8A")
        result = client.compatibility_score(a, b)
        assert result["vocal_clash_penalty"] == 0.0

    def test_vocal_clash_penalty_zero_when_one_track_instrumental(self, client):
        a = TrackMetadata(bpm=128, camelot="8A", vocal_density=0.9)
        b = TrackMetadata(bpm=128, camelot="8A", vocal_density=0.0)
        result = client.compatibility_score(a, b)
        assert result["vocal_clash_penalty"] == 0.0


class TestOverallFormula:
    def test_clean_pair_is_compatible(self, client):
        a = TrackMetadata(bpm=128, camelot="8A", energy=0.55, genres=["house"])
        b = TrackMetadata(bpm=130, camelot="9A", energy=0.50, genres=["house"])
        result = client.compatibility_score(a, b)
        assert result["compatible"] is True
        assert result["overall"] > 0.55

    def test_overall_uses_setflow_weights(self, client):
        """Weighted sum should match the documented SetFlow formula exactly
        (0.35 harmonic + 0.25 beat + 0.15 energy + 0.15 genre + 0.10 timbral
        - vocal_clash), not the old 0.45/0.40/0.15 split."""
        a = TrackMetadata(bpm=128, camelot="8A", energy=0.6, genres=["house"],
                           spectral_centroid_hz=2000.0)
        b = TrackMetadata(bpm=128, camelot="2A", energy=0.4, genres=["techno"],
                           spectral_centroid_hz=2000.0)
        result = client.compatibility_score(a, b)
        expected = (
            result["key_score"] * 0.35
            + result["bpm_score"] * 0.25
            + result["energy_score"] * 0.15
            + result["genre_proximity"] * 0.15
            + result["timbral_similarity"] * 0.10
            - result["vocal_clash_penalty"]
        )
        expected = max(0.0, min(1.0, expected))
        assert abs(result["overall"] - round(expected, 3)) < 0.005

    def test_overall_clamped_to_zero_one(self, client):
        a = TrackMetadata(bpm=63.8, camelot="8A", energy=1.0, vocal_density=1.0)
        b = TrackMetadata(bpm=156.6, camelot="2A", energy=0.0, vocal_density=1.0)
        result = client.compatibility_score(a, b)
        assert 0.0 <= result["overall"] <= 1.0
