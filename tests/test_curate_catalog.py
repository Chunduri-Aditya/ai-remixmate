"""
tests/test_curate_catalog.py — Batch curation driver tests.

Covers the dry-run estimate logic and verifies that --execute mode calls ingest_track
for publishable candidates only.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.core.license import LicenseInfo, LicenseType
from scripts.curate_catalog import curate_catalog, load_manifest, query_key
from scripts.download import CatalogCandidate, IngestResult


@pytest.fixture
def temp_manifest(tmp_path: Path) -> Path:
    """Create a temporary manifest with 2 entries."""
    manifest = [
        {"source": "ccmixter", "query": "test query 1", "pick": 2},
        {"source": "archive", "query": "test query 2", "pick": 3, "notes": "test note"},
    ]
    manifest_path = tmp_path / "test_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return manifest_path


def test_load_manifest(temp_manifest: Path) -> None:
    """Verify manifest loading."""
    manifest = load_manifest(temp_manifest)
    assert len(manifest) == 2
    assert manifest[0]["source"] == "ccmixter"
    assert manifest[0]["query"] == "test query 1"
    assert manifest[0]["pick"] == 2
    assert manifest[1]["source"] == "archive"
    assert manifest[1]["notes"] == "test note"


def test_query_key() -> None:
    """Verify query key generation."""
    entry = {"source": "ccmixter", "query": "glitch hop", "pick": 5}
    assert query_key(entry) == "ccmixter:glitch hop"


@patch("scripts.curate_catalog.search_catalog")
@patch("scripts.curate_catalog.classify_license")
@patch("scripts.curate_catalog.is_publishable")
def test_dry_run_estimates_without_ingest(
    mock_is_publishable: MagicMock,
    mock_classify: MagicMock,
    mock_search: MagicMock,
    temp_manifest: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dry-run mode estimates publishable tracks and storage without calling ingest_track."""
    import logging

    caplog.set_level(logging.INFO)
    # Mock search_catalog to return 2 fake candidates
    cand1 = CatalogCandidate(
        title="Test Track 1",
        artist="Test Artist 1",
        audio_url="http://example.com/track1.mp3",
        source="ccmixter",
    )
    cand2 = CatalogCandidate(
        title="Test Track 2",
        artist="Test Artist 2",
        audio_url="http://example.com/track2.mp3",
        source="ccmixter",
    )
    mock_search.return_value = [cand1, cand2]

    # Mock classify_license to return a CC-BY license
    mock_classify.return_value = LicenseInfo(
        license_type=LicenseType.CC_BY,
        source="ccmixter",
        commercial_ok=True,
        attribution_required=True,
        derivatives_ok=True,
        redistribution_ok=True,
    )

    # Mock is_publishable to return True for first candidate, False for second
    mock_is_publishable.side_effect = [True, False, True, False]

    # Run dry-run
    exit_code = curate_catalog(temp_manifest, dry_run=True, limit_entries=2)

    # Assert exit code 0
    assert exit_code == 0

    # Assert search_catalog was called for both entries
    assert mock_search.call_count == 2

    # Assert classify_license was called 4 times (2 candidates per query)
    assert mock_classify.call_count == 4

    # Assert is_publishable was called 4 times
    assert mock_is_publishable.call_count == 4

    # Assert log output includes dry-run message and estimate
    assert "DRY RUN" in caplog.text
    assert "Publishable tracks: 2" in caplog.text
    assert "Estimated storage" in caplog.text
    assert "0.4 GB" in caplog.text  # 2 tracks × 0.2 GB


@patch("scripts.curate_catalog.ingest_track")
@patch("scripts.curate_catalog.search_catalog")
@patch("scripts.curate_catalog.classify_license")
@patch("scripts.curate_catalog.is_publishable")
@patch("scripts.curate_catalog.save_state")
@patch("scripts.curate_catalog.load_state")
def test_dry_run_does_not_call_ingest_track(
    mock_load_state: MagicMock,
    mock_save_state: MagicMock,
    mock_is_publishable: MagicMock,
    mock_classify: MagicMock,
    mock_search: MagicMock,
    mock_ingest: MagicMock,
    temp_manifest: Path,
) -> None:
    """Verify that dry-run does not call ingest_track."""
    mock_load_state.return_value = {}

    cand = CatalogCandidate(
        title="Test Track",
        artist="Test Artist",
        audio_url="http://example.com/track.mp3",
        source="ccmixter",
    )
    mock_search.return_value = [cand]

    mock_classify.return_value = LicenseInfo(
        license_type=LicenseType.CC_BY,
        source="ccmixter",
        commercial_ok=True,
        derivatives_ok=True,
        redistribution_ok=True,
    )
    mock_is_publishable.return_value = True

    exit_code = curate_catalog(temp_manifest, dry_run=True, limit_entries=1)

    assert exit_code == 0
    mock_ingest.assert_not_called()
    mock_save_state.assert_not_called()


@patch("scripts.curate_catalog.ingest_track")
@patch("scripts.curate_catalog.search_catalog")
@patch("scripts.curate_catalog.classify_license")
@patch("scripts.curate_catalog.is_publishable")
@patch("scripts.curate_catalog.save_state")
@patch("scripts.curate_catalog.load_state")
def test_execute_mode_calls_ingest_for_publishable(
    mock_load_state: MagicMock,
    mock_save_state: MagicMock,
    mock_is_publishable: MagicMock,
    mock_classify: MagicMock,
    mock_search: MagicMock,
    mock_ingest: MagicMock,
    temp_manifest: Path,
) -> None:
    """Verify that --execute mode calls ingest_track for publishable candidates."""
    mock_load_state.return_value = {}

    cand1 = CatalogCandidate(
        title="Publishable Track",
        artist="Test Artist",
        audio_url="http://example.com/publishable.mp3",
        source="ccmixter",
    )
    cand2 = CatalogCandidate(
        title="Non-Publishable Track",
        artist="Test Artist",
        audio_url="http://example.com/nonpublishable.mp3",
        source="ccmixter",
    )
    mock_search.return_value = [cand1, cand2]

    mock_classify.return_value = LicenseInfo(
        license_type=LicenseType.CC_BY,
        source="ccmixter",
    )
    mock_is_publishable.side_effect = [True, False]

    mock_ingest.return_value = IngestResult(
        name="Test Artist - Publishable Track",
        success=True,
    )

    exit_code = curate_catalog(temp_manifest, dry_run=False, limit_entries=1)

    assert exit_code == 0
    # Should call ingest_track once (only for publishable candidate)
    assert mock_ingest.call_count == 1
    mock_ingest.assert_called_once_with(cand1, separate=True, auto_analyze=True)
    # Should save state after processing entry
    assert mock_save_state.call_count >= 1


@patch("scripts.curate_catalog.ingest_track")
@patch("scripts.curate_catalog.search_catalog")
@patch("scripts.curate_catalog.classify_license")
@patch("scripts.curate_catalog.is_publishable")
@patch("scripts.curate_catalog.save_state")
@patch("scripts.curate_catalog.load_state")
def test_execute_mode_exits_1_on_zero_successes(
    mock_load_state: MagicMock,
    mock_save_state: MagicMock,
    mock_is_publishable: MagicMock,
    mock_classify: MagicMock,
    mock_search: MagicMock,
    mock_ingest: MagicMock,
    temp_manifest: Path,
) -> None:
    """Verify that --execute mode exits 1 when no tracks successfully ingest."""
    mock_load_state.return_value = {}

    cand = CatalogCandidate(
        title="Test Track",
        artist="Test Artist",
        audio_url="http://example.com/track.mp3",
        source="ccmixter",
    )
    mock_search.return_value = [cand]

    mock_classify.return_value = LicenseInfo(
        license_type=LicenseType.CC_BY,
        source="ccmixter",
    )
    mock_is_publishable.return_value = True

    # Mock ingest_track to fail
    mock_ingest.return_value = IngestResult(
        name="Test Artist - Test Track",
        success=False,
        error="Mock ingest failure",
    )

    exit_code = curate_catalog(temp_manifest, dry_run=False, limit_entries=1)

    assert exit_code == 1
    assert mock_ingest.call_count == 1
