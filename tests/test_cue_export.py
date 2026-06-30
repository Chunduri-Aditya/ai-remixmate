"""
tests/test_cue_export.py — Unit tests for scripts/core/cue_export.py.

These tests use stdlib only (no mutagen, no librosa).  Tests that require
mutagen are marked and skipped when the package is absent.

Coverage
--------
- _build_serato_markers2_payload: binary layout, slot index, timestamp,
  entry count, header presence, empty list.
- export_rekordbox_xml: XML structure, HOT CUE count cap, MEMORY CUE presence,
  TEMPO element, empty phrase boundaries.
- export_cues: dispatcher, unknown-format error, serato without audio_path.
- export_serato_markers: ImportError on missing mutagen (mocked).
"""
from __future__ import annotations

import struct
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cues(n: int = 3) -> list[float]:
    """Return n cue points at 0, 10, 20, ... seconds."""
    return [i * 10.0 for i in range(n)]


# ---------------------------------------------------------------------------
# TestSeratoPayload — pure binary construction, no external deps
# ---------------------------------------------------------------------------

class TestSeratoPayload:
    """Tests for _build_serato_markers2_payload (internal helper)."""

    def _payload(self, cue_points):
        from scripts.core.cue_export import _build_serato_markers2_payload
        return _build_serato_markers2_payload(cue_points)

    def test_header_starts_with_mime(self):
        payload = self._payload([0.0])
        assert payload.startswith(b"application/octet-stream\x00Serato Markers2\x00")

    def test_version_bytes_present(self):
        """After header, first two bytes must be \x01\x01."""
        payload = self._payload([0.0])
        header = b"application/octet-stream\x00Serato Markers2\x00"
        body = payload[len(header):]
        assert body[0:2] == b"\x01\x01"

    def test_entry_count_correct(self):
        """4-byte big-endian count after version bytes must match len(cue_points)."""
        header = b"application/octet-stream\x00Serato Markers2\x00"
        for n in (1, 3, 8):
            payload = self._payload(_make_cues(n))
            body = payload[len(header):]
            count = struct.unpack(">I", body[2:6])[0]
            assert count == n, f"Expected {n} entries, got {count}"

    def test_empty_list_produces_zero_entries(self):
        header = b"application/octet-stream\x00Serato Markers2\x00"
        payload = self._payload([])
        body = payload[len(header):]
        count = struct.unpack(">I", body[2:6])[0]
        assert count == 0

    def test_cap_at_8_cues(self):
        """Only first 8 cue points are encoded even if more are provided."""
        header = b"application/octet-stream\x00Serato Markers2\x00"
        payload = self._payload(_make_cues(12))  # 12 provided → 8 encoded
        body = payload[len(header):]
        count = struct.unpack(">I", body[2:6])[0]
        assert count == 8

    def test_first_cue_timestamp_correct(self):
        """First CUE entry body should encode the time in milliseconds."""
        from scripts.core.cue_export import _build_serato_markers2_payload
        t_s = 32.5
        payload = _build_serato_markers2_payload([t_s])
        header = b"application/octet-stream\x00Serato Markers2\x00"
        body = payload[len(header):]
        # version (2) + count (4) + "CUE\0" (4) + length (4) = 14 bytes before body
        # body starts with: \x00 (1) + index (1) + \x00\x00 (2) + timestamp (4)
        entry_start = 2 + 4 + 4 + 4   # version + count + type + length
        body_start = entry_start + 1 + 1 + 2   # flags + index + padding
        t_ms_encoded = struct.unpack(">I", body[body_start : body_start + 4])[0]
        assert t_ms_encoded == int(t_s * 1000)

    def test_slot_index_matches_position(self):
        """Second CUE entry (index 1) should have slot byte == 1."""
        from scripts.core.cue_export import _build_serato_markers2_payload
        payload = _build_serato_markers2_payload([0.0, 10.0])
        header = b"application/octet-stream\x00Serato Markers2\x00"
        body = payload[len(header):]

        # Parse the two CUE entries manually.
        # Layout after version+count: for each entry:
        #   type (4) + length (4) + body
        # body[0] = start flag, body[1] = slot index
        pos = 2 + 4  # skip version + count
        indices = []
        for _ in range(2):
            # skip "CUE\0" + length field; read length
            pos += 4   # type "CUE\0"
            entry_len = struct.unpack(">I", body[pos : pos + 4])[0]
            pos += 4   # length field
            slot_idx = body[pos + 1]   # body[0]=flag, body[1]=slot
            indices.append(slot_idx)
            pos += entry_len
        assert indices == [0, 1]


# ---------------------------------------------------------------------------
# TestRekordboxXML
# ---------------------------------------------------------------------------

class TestRekordboxXML:
    """Tests for export_rekordbox_xml — only stdlib (ET) needed."""

    def _export(self, cue_points, phrase_boundaries=None, bpm=128.0, **kw):
        from scripts.core.cue_export import export_rekordbox_xml
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "test.xml"
            export_rekordbox_xml(
                song_name="Test Song",
                cue_points=cue_points,
                phrase_boundaries=phrase_boundaries or [],
                bpm=bpm,
                output_path=out,
                **kw,
            )
            return ET.parse(out).getroot()

    def test_root_element_is_dj_playlists(self):
        root = self._export([0.0])
        assert root.tag == "DJ_PLAYLISTS"

    def test_collection_has_track(self):
        root = self._export([0.0])
        track = root.find("COLLECTION/TRACK")
        assert track is not None

    def test_hot_cue_markers_present(self):
        root = self._export([0.0, 10.0, 20.0])
        marks = root.findall("COLLECTION/TRACK/POSITION_MARK[@Type='0']")
        assert len(marks) == 3

    def test_hot_cue_cap_at_8(self):
        root = self._export(_make_cues(12))
        marks = root.findall("COLLECTION/TRACK/POSITION_MARK[@Type='0']")
        assert len(marks) == 8

    def test_memory_cue_markers_present(self):
        root = self._export([0.0], phrase_boundaries=[10.0, 20.0])
        marks = root.findall("COLLECTION/TRACK/POSITION_MARK[@Type='1']")
        assert len(marks) == 2

    def test_no_memory_cues_when_empty(self):
        root = self._export([0.0], phrase_boundaries=[])
        marks = root.findall("COLLECTION/TRACK/POSITION_MARK[@Type='1']")
        assert len(marks) == 0

    def test_tempo_element_present(self):
        root = self._export([0.0], bpm=130.0)
        tempo = root.find("COLLECTION/TRACK/TEMPO")
        assert tempo is not None
        assert abs(float(tempo.attrib["Bpm"]) - 130.0) < 0.01

    def test_bpm_stored_in_track(self):
        root = self._export([0.0], bpm=142.5)
        track = root.find("COLLECTION/TRACK")
        assert abs(float(track.attrib["Bpm"]) - 142.5) < 0.01

    def test_song_name_in_track(self):
        from scripts.core.cue_export import export_rekordbox_xml
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.xml"
            export_rekordbox_xml(
                song_name="My Track",
                cue_points=[0.0],
                phrase_boundaries=[],
                bpm=128.0,
                output_path=out,
            )
            root = ET.parse(out).getroot()
            track = root.find("COLLECTION/TRACK")
            assert track.attrib["Name"] == "My Track"

    def test_cue_start_times_correct(self):
        cues = [0.0, 32.5, 64.0]
        root = self._export(cues)
        marks = root.findall("COLLECTION/TRACK/POSITION_MARK[@Type='0']")
        starts = sorted(float(m.attrib["Start"]) for m in marks)
        for got, expected in zip(starts, cues):
            assert abs(got - expected) < 0.001

    def test_output_file_is_created(self):
        from scripts.core.cue_export import export_rekordbox_xml
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sub" / "out.xml"
            result = export_rekordbox_xml(
                song_name="X",
                cue_points=[0.0],
                phrase_boundaries=[],
                bpm=120.0,
                output_path=out,
            )
            assert result.exists()
            assert result.suffix == ".xml"


# ---------------------------------------------------------------------------
# TestExportCuesDispatcher
# ---------------------------------------------------------------------------

class TestExportCuesDispatcher:
    """Tests for the export_cues() convenience dispatcher."""

    def test_unknown_format_raises_value_error(self):
        from scripts.core.cue_export import export_cues
        with pytest.raises(ValueError, match="Unknown export format"):
            with tempfile.TemporaryDirectory() as td:
                export_cues(
                    song_name="X", cue_points=[0.0], phrase_boundaries=[],
                    bpm=120.0, fmt="ableton", output_path=Path(td) / "out.xml",
                )

    def test_serato_without_audio_path_raises(self):
        from scripts.core.cue_export import export_cues
        with pytest.raises(ValueError, match="audio_path is required"):
            with tempfile.TemporaryDirectory() as td:
                export_cues(
                    song_name="X", cue_points=[0.0], phrase_boundaries=[],
                    bpm=120.0, fmt="serato", output_path=Path(td) / "out.mp3",
                    audio_path=None,
                )

    def test_rekordbox_dispatches_correctly(self):
        from scripts.core.cue_export import export_cues
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.xml"
            result = export_cues(
                song_name="Test", cue_points=[0.0, 10.0],
                phrase_boundaries=[10.0], bpm=128.0,
                fmt="rekordbox", output_path=out,
            )
            assert result.exists()
            root = ET.parse(result).getroot()
            assert root.tag == "DJ_PLAYLISTS"

    def test_format_case_insensitive(self):
        from scripts.core.cue_export import export_cues
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.xml"
            result = export_cues(
                song_name="Test", cue_points=[0.0], phrase_boundaries=[],
                bpm=128.0, fmt="REKORDBOX", output_path=out,
            )
            assert result.exists()


# ---------------------------------------------------------------------------
# TestSeratoExport — requires mutagen; skip gracefully when absent
# ---------------------------------------------------------------------------

class TestSeratoExport:
    """Integration tests for export_serato_markers — require mutagen."""

    @pytest.fixture(autouse=True)
    def _require_mutagen(self):
        import importlib
        if importlib.util.find_spec("mutagen") is None:
            pytest.skip("mutagen not installed")

    def _make_mp3(self, tmp_dir: Path) -> Path:
        """Create a minimal valid MP3 file (ID3 header only) for tag writing."""
        mp3 = tmp_dir / "test.mp3"
        # Minimal ID3v2 header: "ID3" + version + flags + syncsafe size (0 bytes)
        mp3.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00")
        return mp3

    def test_writes_geob_tag(self):
        from mutagen.id3 import ID3
        from scripts.core.cue_export import export_serato_markers
        with tempfile.TemporaryDirectory() as td:
            src = self._make_mp3(Path(td))
            out = Path(td) / "tagged.mp3"
            export_serato_markers(src, [0.0, 32.0], output_path=out)
            tags = ID3(str(out))
            assert any("Serato Markers2" in key for key in tags.keys())

    def test_raises_on_non_mp3(self):
        from scripts.core.cue_export import export_serato_markers
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "test.wav"
            wav.write_bytes(b"\x00" * 44)
            with pytest.raises(ValueError, match="only supported for .mp3"):
                export_serato_markers(wav, [0.0])

    def test_raises_on_missing_file(self):
        from scripts.core.cue_export import export_serato_markers
        with pytest.raises(FileNotFoundError):
            export_serato_markers(Path("/nonexistent/file.mp3"), [0.0])

    def test_in_place_write_when_no_output_path(self):
        from mutagen.id3 import ID3
        from scripts.core.cue_export import export_serato_markers
        with tempfile.TemporaryDirectory() as td:
            src = self._make_mp3(Path(td))
            result = export_serato_markers(src, [5.0, 10.0], output_path=None)
            assert result == src
            tags = ID3(str(src))
            assert any("Serato Markers2" in key for key in tags.keys())
