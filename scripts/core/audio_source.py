"""
scripts/core/audio_source.py — Source-audio resolution for the render path.

A song's playable source may be stored in any of several forms depending on
where it sits in the download → enhance → stem-separate → prune lifecycle:

    full.wav            original downloaded mix (often pruned away to save space)
    full_enhanced.wav   mastered/enhanced mix kept after pruning
    <stem>.flac / .wav  Demucs stems (vocals/drums/bass/other)

With ``prune_on_download`` enabled, most songs lose ``full.wav`` and keep only
``full_enhanced.wav`` + stems.  The DJ render path historically opened
``full.wav`` directly, so it failed on the bulk of a pruned library.  This
module resolves a song name to usable audio, trying each form in priority
order, so the engine works on the whole library — not just the few songs that
still have ``full.wav``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from scripts.core.paths import song_dir

log = logging.getLogger(__name__)

# Single-file sources, highest fidelity first.
_SOURCE_FILES = ("full.wav", "full_enhanced.wav")
# Demucs stems in canonical order; reconstruct = sum of all four.
_STEMS = ("vocals", "drums", "bass", "other")


def resolve_source_file(song_name: str) -> Optional[Path]:
    """Return an existing single-file source (full.wav → full_enhanced.wav), or None."""
    d = song_dir(song_name)
    for fname in _SOURCE_FILES:
        p = d / fname
        if p.exists():
            return p
    return None


def stem_files(song_name: str) -> List[Path]:
    """Return existing stem files (flac preferred over wav) in canonical order."""
    d = song_dir(song_name)
    out: List[Path] = []
    for s in _STEMS:
        for ext in (".flac", ".wav"):
            p = d / f"{s}{ext}"
            if p.exists():
                out.append(p)
                break
    return out


def has_source_audio(song_name: str) -> bool:
    """True if the song can be loaded — a single-file source or at least one stem."""
    return resolve_source_file(song_name) is not None or len(stem_files(song_name)) > 0


def find_unusable_songs() -> List[str]:
    """
    Return the names of library folders the engine cannot load — no full mix
    (full.wav / full_enhanced.wav) and no stems at all.  These are typically
    download shells left behind when only the metadata (full.info.json /
    license.json) was written but the audio never landed.
    """
    from scripts.core.paths import LIBRARY_DIR

    if not LIBRARY_DIR.exists():
        return []
    return sorted(
        d.name for d in LIBRARY_DIR.iterdir()
        if d.is_dir() and not has_source_audio(d.name)
    )


def purge_unusable_songs(dry_run: bool = True) -> List[str]:
    """
    Delete library folders with no usable source audio (see find_unusable_songs).

    Returns the list of song names identified.  With ``dry_run=True`` (the
    default) nothing is deleted — call with ``dry_run=False`` to remove them.
    Irreversible: the whole song folder (including its metadata JSON) is removed.
    """
    import shutil

    from scripts.core.paths import song_dir

    names = find_unusable_songs()
    if dry_run:
        return names
    for name in names:
        shutil.rmtree(song_dir(name), ignore_errors=True)
        log.info("Purged unusable song %r (no source audio)", name)
    return names


def load_source_audio(
    song_name: str,
    sr: Optional[int] = 44100,
    mono: bool = True,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load a song's source audio as a float32 array.

    Resolution order:
      1. ``full.wav``
      2. ``full_enhanced.wav``
      3. reconstruct by summing the Demucs stems (vocals+drums+bass+other)

    ``sr=None`` keeps the file's native sample rate (stems are summed at their
    shared native rate).  Returns ``(audio, sample_rate)``.

    Raises ``FileNotFoundError`` (with the song name) if no source exists.
    """
    import librosa

    src = resolve_source_file(song_name)
    if src is not None:
        audio, out_sr = librosa.load(str(src), sr=sr, mono=mono, duration=duration)
        return audio.astype(np.float32), int(out_sr)

    # ── Reconstruct from stems ───────────────────────────────────────────────
    stems = stem_files(song_name)
    if not stems:
        raise FileNotFoundError(f"No source audio for {song_name!r}")

    mix: Optional[np.ndarray] = None
    out_sr = sr or 44100
    for p in stems:
        s, file_sr = librosa.load(str(p), sr=sr, mono=mono, duration=duration)
        s = s.astype(np.float32)
        out_sr = int(file_sr)
        if mix is None:
            mix = s
        else:
            n = min(len(mix), len(s))
            mix = mix[:n] + s[:n]

    # Peak-normalise the summed stems so the reconstructed mix can't clip.
    # (The engine peak-normalises its final output anyway, so absolute level
    #  here only matters for headroom during analysis/mixing.)
    peak = float(np.abs(mix).max()) if mix is not None and mix.size else 0.0
    if peak > 1.0:
        mix = mix / peak
    log.info("Reconstructed source for %r from %d stem(s)", song_name, len(stems))
    return mix.astype(np.float32), out_sr
