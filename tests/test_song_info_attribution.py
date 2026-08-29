"""
tests/test_song_info_attribution.py — _song_info surfaces catalogue attribution.

Deployment plan P1: the library API response must carry artist_name and
license_ccurl so the frontend can render a credit line and P8 can build mix
credits. Before this, _song_info read only license_type / source from
license.json.
"""

from __future__ import annotations

import json

from scripts.api.routers._helpers import _song_info
from scripts.core.license import LicenseInfo, LicenseType, save_license


def test_song_info_reads_artist_and_ccurl_from_license_json(tmp_path) -> None:
    song_dir = tmp_path / "Some Artist - Some Track"
    song_dir.mkdir()
    (song_dir / "full.wav").write_bytes(b"\x00")

    save_license(
        song_dir,
        LicenseInfo(
            license_type=LicenseType.CC_BY,
            source="ccmixter",
            license_url="https://creativecommons.org/licenses/by/3.0/",
            artist_name="Some Artist",
            source_url="https://ccmixter.org/files/someartist/12345",
        ),
    )

    info = _song_info(song_dir)
    assert info.license_type == "cc_by"
    assert info.source == "ccmixter"
    assert info.artist_name == "Some Artist"
    assert info.license_ccurl == "https://creativecommons.org/licenses/by/3.0/"


def test_song_info_attribution_is_none_without_license_json(tmp_path) -> None:
    song_dir = tmp_path / "Unlicensed Track"
    song_dir.mkdir()
    (song_dir / "full.wav").write_bytes(b"\x00")

    info = _song_info(song_dir)
    assert info.artist_name is None
    assert info.license_ccurl is None


def test_song_info_survives_a_legacy_license_json_without_new_keys(tmp_path) -> None:
    # license.json written before artist_name/source_url existed.
    song_dir = tmp_path / "Legacy Track"
    song_dir.mkdir()
    (song_dir / "full.wav").write_bytes(b"\x00")
    (song_dir / "license.json").write_text(
        json.dumps({"license_type": "cc_by", "source": "jamendo"}), encoding="utf-8"
    )

    info = _song_info(song_dir)
    assert info.license_type == "cc_by"
    assert info.artist_name is None
    assert info.license_ccurl is None
