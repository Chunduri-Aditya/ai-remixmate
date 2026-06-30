"""
scripts/core/cue_export.py — Export RemixMate cue points to DJ software formats.

Supported formats
-----------------
rekordbox XML
    Standard rekordbox 6+ collection XML.  HOT CUE markers embedded in a
    TRACK element.  Compatible with rekordbox 6+, DJay Pro (via import),
    and VirtualDJ.

Serato ID3 GEOB
    Serato stores cue points as GEOB (General Encapsulated Object) ID3 tags
    inside .mp3 files.  The binary format is reverse-engineered from the
    Serato community and widely used by third-party tools (e.g., Mixxx).
    Works only with .mp3 files; FLAC/WAV is not supported by Serato's tag
    format.

Usage
-----
    from scripts.core.cue_export import export_rekordbox_xml, export_serato_markers

    # rekordbox
    export_rekordbox_xml(
        song_name="Song Title",
        cue_points=[0.0, 32.5, 64.0],
        phrase_boundaries=[32.5, 64.0],
        bpm=128.0,
        output_path=Path("output/rekordbox_cues.xml"),
    )

    # Serato — writes tags into the mp3 file (or a copy)
    export_serato_markers(
        audio_path=Path("library/Song/full.mp3"),
        cue_points=[0.0, 32.5, 64.0],
        output_path=None,   # None = write in-place
    )
"""

from __future__ import annotations

import logging
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette for cue points
# ---------------------------------------------------------------------------

# rekordbox uses hex RGB colour codes for HOT CUE markers.
# Standard palette: red, orange, yellow, green, cyan, blue, violet, pink
_RB_COLOURS = [
    "#CC0000",   # 0 — red
    "#CC6600",   # 1 — orange
    "#CCCC00",   # 2 — yellow
    "#00CC00",   # 3 — green
    "#00CCCC",   # 4 — cyan
    "#0000CC",   # 5 — blue
    "#6600CC",   # 6 — violet
    "#CC0066",   # 7 — pink
]

# Serato GEOB colours (same logical palette, stored as 3-byte BGR in the tag)
_SERATO_COLOURS: List[bytes] = [
    bytes([0x00, 0x00, 0xCC]),   # red   → BGR
    bytes([0x00, 0x66, 0xCC]),   # orange
    bytes([0x00, 0xCC, 0xCC]),   # yellow
    bytes([0x00, 0xCC, 0x00]),   # green
    bytes([0xCC, 0xCC, 0x00]),   # cyan
    bytes([0xCC, 0x00, 0x00]),   # blue
    bytes([0xCC, 0x00, 0x66]),   # violet
    bytes([0x66, 0x00, 0xCC]),   # pink
]


# ---------------------------------------------------------------------------
# rekordbox XML export
# ---------------------------------------------------------------------------

def export_rekordbox_xml(
    song_name: str,
    cue_points: List[float],
    phrase_boundaries: List[float],
    bpm: float,
    output_path: Path,
    artist: str = "",
    total_time_s: float = 0.0,
    audio_path: Optional[Path] = None,
) -> Path:
    """
    Write a rekordbox-compatible XML collection file with HOT CUE markers.

    Parameters
    ----------
    song_name : str
        Track title shown in rekordbox.
    cue_points : list[float]
        Hot cue times in seconds.  Up to 8 are exported (rekordbox limit).
    phrase_boundaries : list[float]
        RemixMate phrase boundary times (seconds).  Exported as MEMORY CUE
        markers (Type="1") so they appear in the waveform overview without
        occupying hot cue slots.
    bpm : float
        Track BPM.
    output_path : Path
        Destination XML file path.
    artist : str
        Artist name (optional).
    total_time_s : float
        Track duration in seconds (optional; stored in TotalTime attribute).
    audio_path : Path | None
        Absolute path to the audio file (stored in Location attribute).

    Returns
    -------
    Path
        The written output_path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    product = ET.SubElement(root, "PRODUCT",
        Name="RemixMate", Version="1.0", Company="RemixMate")
    collection = ET.SubElement(root, "COLLECTION", Entries=str(1))

    # ── TRACK element ────────────────────────────────────────────────────
    track_attrs: dict[str, str] = {
        "TrackID": "1",
        "Name": song_name,
        "Artist": artist,
        "Tonality": "",
        "Bpm": f"{bpm:.2f}",
        "AverageBpm": f"{bpm:.2f}",
        "Kind": "MP3 File",
        "TotalTime": str(int(total_time_s)),
    }
    if audio_path is not None:
        track_attrs["Location"] = audio_path.as_uri()

    track = ET.SubElement(collection, "TRACK", **track_attrs)

    # ── TEMPO (beat grid) ─────────────────────────────────────────────────
    ET.SubElement(track, "TEMPO",
        Inizio="0.000",
        Bpm=f"{bpm:.2f}",
        Metro="4/4",
        Battito="1",
    )

    # ── HOT CUE markers (up to 8) ─────────────────────────────────────────
    for i, t in enumerate(cue_points[:8]):
        ET.SubElement(track, "POSITION_MARK",
            Name=f"C{i + 1}",
            Type="0",              # HOT CUE
            Start=f"{t:.3f}",
            Num=str(i),
            Red=str(int(_RB_COLOURS[i % len(_RB_COLOURS)][1:3], 16)),
            Green=str(int(_RB_COLOURS[i % len(_RB_COLOURS)][3:5], 16)),
            Blue=str(int(_RB_COLOURS[i % len(_RB_COLOURS)][5:7], 16)),
        )

    # ── MEMORY CUE markers (phrase boundaries) ────────────────────────────
    for j, t in enumerate(phrase_boundaries):
        ET.SubElement(track, "POSITION_MARK",
            Name=f"P{j + 1}",
            Type="1",              # MEMORY CUE (loop/phrase marker)
            Start=f"{t:.3f}",
            Num=str(j + 8),        # offset past hot cue slots
            Red="0", Green="128", Blue="255",
        )

    # ── Write XML ─────────────────────────────────────────────────────────
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)

    log.info(
        "rekordbox XML written: %s (%d hot cues, %d phrase markers)",
        output_path, min(len(cue_points), 8), len(phrase_boundaries),
    )
    return output_path


# ---------------------------------------------------------------------------
# Serato ID3 GEOB tag export
# ---------------------------------------------------------------------------

def _build_serato_markers2_payload(cue_points: List[float]) -> bytes:
    """
    Build the binary payload for Serato's GEOB:Serato Markers2 ID3 tag.

    Format (reverse-engineered, widely documented in Serato community):
      Header: b"application/octet-stream\x00Serato Markers2\x00"
      Payload version: b"\x01\x01"
      Entry count: 4-byte big-endian uint32
      For each CUE entry:
        Type:     b"CUE\x00"  (null-terminated string)
        Length:   4-byte big-endian uint32 (body byte count)
        Body:
          \x00            — start flag
          index (1 byte)  — cue slot 0-7
          \x00\x00        — padding
          position (4-byte big-endian uint32, milliseconds)
          \x00            — end flag
          colour (3 bytes RGB)
          \x00\x00        — padding
          name (null-terminated UTF-8 string, empty = "\x00")

    Reference: https://github.com/digital-dj-tools/serato-data-formats
    """
    entries: list[bytes] = []
    for i, t_s in enumerate(cue_points[:8]):
        t_ms = int(t_s * 1000)
        colour = _SERATO_COLOURS[i % len(_SERATO_COLOURS)]
        name = f"C{i + 1}".encode("utf-8") + b"\x00"

        body = (
            b"\x00"                              # start flag
            + bytes([i])                         # cue slot index (0–7)
            + b"\x00\x00"                        # padding
            + struct.pack(">I", t_ms)            # position in ms (big-endian)
            + b"\x00"                            # end flag
            + colour                             # RGB colour (3 bytes)
            + b"\x00\x00"                        # padding
            + name                               # label (null-terminated)
        )

        entry = (
            b"CUE\x00"
            + struct.pack(">I", len(body))
            + body
        )
        entries.append(entry)

    payload = (
        b"\x01\x01"                              # version
        + struct.pack(">I", len(entries))        # entry count
        + b"".join(entries)
    )

    header = b"application/octet-stream\x00Serato Markers2\x00"
    return header + payload


def export_serato_markers(
    audio_path: Path,
    cue_points: List[float],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Write Serato cue markers as a GEOB ID3 tag into an .mp3 file.

    Requires ``mutagen`` (``pip install mutagen``).

    Parameters
    ----------
    audio_path : Path
        Source .mp3 file.  For in-place write, pass output_path=None.
    cue_points : list[float]
        Cue times in seconds (up to 8).
    output_path : Path | None
        If None, tags are written back to ``audio_path`` in-place.
        If specified, ``audio_path`` is copied to ``output_path`` first.

    Returns
    -------
    Path
        Path to the tagged file.

    Raises
    ------
    ImportError
        If ``mutagen`` is not installed.
    ValueError
        If ``audio_path`` is not an .mp3 file.
    FileNotFoundError
        If ``audio_path`` does not exist.
    """
    try:
        from mutagen.id3 import ID3, GEOB, Encoding
        from mutagen.mp3 import MP3
        import shutil
    except ImportError as e:
        raise ImportError(
            "mutagen is required for Serato export.  Install with: pip install mutagen"
        ) from e

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() != ".mp3":
        raise ValueError(
            f"Serato GEOB tags are only supported for .mp3 files (got {audio_path.suffix!r})"
        )

    # Copy to output_path if requested
    dest = audio_path
    if output_path is not None:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, dest)

    # Load existing ID3 tags (create if missing)
    try:
        tags = ID3(str(dest))
    except Exception:
        tags = ID3()

    payload = _build_serato_markers2_payload(cue_points)
    tags.add(
        GEOB(
            encoding=Encoding.LATIN1,
            mime="application/octet-stream",
            filename="Serato Markers2",
            desc="Serato Markers2",
            data=payload,
        )
    )
    tags.save(str(dest))

    log.info(
        "Serato markers written: %s (%d cues)",
        dest, min(len(cue_points), 8),
    )
    return dest


# ---------------------------------------------------------------------------
# Convenience: detect format from extension or query
# ---------------------------------------------------------------------------

def export_cues(
    song_name: str,
    cue_points: List[float],
    phrase_boundaries: List[float],
    bpm: float,
    fmt: str,
    output_path: Path,
    audio_path: Optional[Path] = None,
    artist: str = "",
    total_time_s: float = 0.0,
) -> Path:
    """
    Dispatch to the correct exporter based on ``fmt``.

    Parameters
    ----------
    fmt : str
        "rekordbox" or "serato".
    """
    fmt = fmt.lower().strip()
    if fmt == "rekordbox":
        return export_rekordbox_xml(
            song_name=song_name,
            cue_points=cue_points,
            phrase_boundaries=phrase_boundaries,
            bpm=bpm,
            output_path=output_path,
            artist=artist,
            total_time_s=total_time_s,
            audio_path=audio_path,
        )
    if fmt == "serato":
        if audio_path is None:
            raise ValueError("audio_path is required for Serato export")
        return export_serato_markers(
            audio_path=audio_path,
            cue_points=cue_points,
            output_path=output_path,
        )
    raise ValueError(f"Unknown export format {fmt!r}. Use 'rekordbox' or 'serato'.")
