"""
scripts/core/instrument_lab.py — Instrument Replacement & Combination Engine

Mix-and-match stems from different songs to create hybrid tracks.
Given N songs with Demucs stems (vocals, drums, bass, other), this module:

  1. Enumerates all possible stem combinations
  2. Renders each combo by aligning BPM via time-stretch
  3. Applies per-stem gain normalisation + mastering
  4. Outputs every variant as a separate file

Architecture
────────────
  • Each combo is a dict like {"vocals": "Song A", "drums": "Song B", ...}
  • The "anchor" is the combo's tempo reference (defaults to drums source)
  • All other stems get time-stretched to match the anchor's BPM
  • Optional key-aware pitch shift keeps harmonic stems in tune

Usage
─────
    from scripts.core.instrument_lab import (
        InstrumentCombo, enumerate_combos, render_combo, render_all_combos
    )

    songs = ["Anyma - Abyss", "Dom Dolla - Define"]
    combos = enumerate_combos(songs)
    # → generates all 2^4 = 16 combos (4 stems, 2 songs each)

    result = render_combo(combos[0], sr=44100)
    # → renders a single combo to a numpy array

    results = render_all_combos(songs, output_dir=Path("outputs/lab_xyz"))
    # → renders every combo, saves WAV files, returns metadata
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

STEM_NAMES = ("vocals", "drums", "bass", "other")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InstrumentCombo:
    """A single stem combination — which song each stem comes from."""
    mapping: Dict[str, str]     # stem_name → song_name
    label: str = ""             # human-readable description
    anchor_stem: str = "drums"  # stem whose song sets the BPM

    def __post_init__(self):
        if not self.label:
            parts = [f"{s[0].upper()}:{self.mapping[s][:18]}" for s in STEM_NAMES if s in self.mapping]
            self.label = " | ".join(parts)

    @property
    def anchor_song(self) -> str:
        return self.mapping.get(self.anchor_stem, next(iter(self.mapping.values())))

    @property
    def is_pure(self) -> bool:
        """True if all stems come from the same song (original, not a hybrid)."""
        sources = set(self.mapping.values())
        return len(sources) == 1

    @property
    def unique_sources(self) -> List[str]:
        return sorted(set(self.mapping.values()))

    def short_label(self) -> str:
        """Compact label like 'V:SongA D:SongB B:SongA O:SongB'."""
        songs = list(set(self.mapping.values()))
        song_map = {s: chr(65 + i) for i, s in enumerate(songs)}  # A, B, C...
        return " ".join(f"{stem[0].upper()}:{song_map[song]}" for stem, song in self.mapping.items())


@dataclass
class ComboResult:
    """Result of rendering a single combo."""
    combo: InstrumentCombo
    audio: Optional[np.ndarray] = None
    output_path: Optional[Path] = None
    bpm: float = 0.0
    duration_sec: float = 0.0
    peak_dbfs: float = 0.0
    lufs: float = -70.0
    success: bool = False
    error: Optional[str] = None


@dataclass
class LabSession:
    """Complete results of a render_all_combos run."""
    songs: List[str]
    total_combos: int = 0
    rendered: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[ComboResult] = field(default_factory=list)
    output_dir: Optional[Path] = None


# ---------------------------------------------------------------------------
# Combo enumeration
# ---------------------------------------------------------------------------

def enumerate_combos(
    songs: List[str],
    stems: Tuple[str, ...] = STEM_NAMES,
    include_pure: bool = True,
) -> List[InstrumentCombo]:
    """
    Generate all possible stem combinations across N songs.

    For 2 songs × 4 stems → 2^4 = 16 combos (including the 2 pure originals).
    For 3 songs × 4 stems → 3^4 = 81 combos.

    Parameters
    ----------
    songs : list of song names
    stems : which stems to permute (default: all 4)
    include_pure : whether to include combos where all stems come from one song

    Returns
    -------
    List of InstrumentCombo objects
    """
    if len(songs) < 2:
        raise ValueError("Need at least 2 songs for instrument combinations")

    combos: List[InstrumentCombo] = []

    for combo_tuple in itertools.product(songs, repeat=len(stems)):
        mapping = {stem: song for stem, song in zip(stems, combo_tuple)}
        combo = InstrumentCombo(mapping=mapping)

        if not include_pure and combo.is_pure:
            continue

        combos.append(combo)

    log.info("Enumerated %d combos across %d songs × %d stems",
             len(combos), len(songs), len(stems))
    return combos


def enumerate_targeted_swaps(
    songs: List[str],
    swap_stems: Optional[List[str]] = None,
) -> List[InstrumentCombo]:
    """
    Generate only targeted single-stem swaps (more manageable than full permutation).

    For each pair of songs, swap one stem at a time while keeping the rest from
    the base song. This produces (N*(N-1)) * S combos where S = # stems to swap.

    Example with 2 songs and 4 stems → 8 combos (each song as base, swap 1 stem).
    """
    if swap_stems is None:
        swap_stems = list(STEM_NAMES)

    combos: List[InstrumentCombo] = []

    for base_song in songs:
        for donor_song in songs:
            if donor_song == base_song:
                continue
            for stem in swap_stems:
                mapping = {s: base_song for s in STEM_NAMES}
                mapping[stem] = donor_song
                combos.append(InstrumentCombo(mapping=mapping))

    log.info("Enumerated %d targeted swaps across %d songs", len(combos), len(songs))
    return combos


# ---------------------------------------------------------------------------
# BPM lookup
# ---------------------------------------------------------------------------

def _get_song_bpm(song_name: str) -> float:
    """Look up a song's BPM from its analysis cache or estimate from audio."""
    try:
        from scripts.core.library import get_song_metadata
        meta = get_song_metadata(song_name)
        if meta and meta.get("bpm"):
            return float(meta["bpm"])
    except Exception:
        pass

    # Fallback: try to estimate from the full.wav
    try:
        import librosa
        from scripts.core.paths import song_dir
        wav = song_dir(song_name) / "full.wav"
        if wav.exists():
            y, sr = librosa.load(str(wav), sr=22050, mono=True, duration=30.0)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
    except Exception:
        pass

    return 120.0  # safe default


# ---------------------------------------------------------------------------
# Single combo render
# ---------------------------------------------------------------------------

def render_combo(
    combo: InstrumentCombo,
    sr: int = 44100,
    target_duration: Optional[float] = None,
    normalize: bool = True,
) -> ComboResult:
    """
    Render a single InstrumentCombo by loading, time-stretching, and mixing stems.

    Algorithm
    ---------
    1. Determine anchor BPM from the anchor song
    2. For each stem:
       a. Load the stem audio from the source song
       b. If source BPM ≠ anchor BPM → time-stretch to match
    3. Sum all stems
    4. Normalise to -1 dBFS
    5. Return ComboResult with the mixed audio

    Parameters
    ----------
    combo : InstrumentCombo
    sr : sample rate
    target_duration : if set, trim/pad all stems to this many seconds
    normalize : peak-normalise the output
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError as exc:
        return ComboResult(combo=combo, error=f"Missing dependency: {exc}")

    from scripts.core.dj_engine import _load_stem, _pad_or_trim

    anchor_bpm = _get_song_bpm(combo.anchor_song)
    log.info("Rendering combo: %s  (anchor BPM=%.1f from %s)",
             combo.short_label(), anchor_bpm, combo.anchor_song[:30])

    # ── Load + time-stretch each stem ─────────────────────────────────────
    stem_arrays: Dict[str, np.ndarray] = {}
    min_length = float('inf')

    for stem_name in STEM_NAMES:
        song = combo.mapping.get(stem_name)
        if not song:
            continue

        from scripts.core.paths import song_dir
        sdir = song_dir(song)
        arr = _load_stem(sdir, stem_name, sr)

        if arr is None:
            log.warning("Stem %s not found for %s — using silence", stem_name, song)
            continue

        # Time-stretch if needed (GPU-accelerated when available)
        song_bpm = _get_song_bpm(song)
        if song_bpm > 0 and anchor_bpm > 0 and abs(song_bpm - anchor_bpm) > 1.0:
            ratio = anchor_bpm / song_bpm
            try:
                from scripts.core.gpu import gpu_time_stretch
                arr = gpu_time_stretch(arr, rate=ratio, sr=sr)
                log.info("  %s: stretched %.1f→%.1f BPM (ratio=%.3f) [GPU]",
                         stem_name, song_bpm, anchor_bpm, ratio)
            except ImportError:
                arr = librosa.effects.time_stretch(arr, rate=ratio)
                log.info("  %s: stretched %.1f→%.1f BPM (ratio=%.3f) [CPU]",
                         stem_name, song_bpm, anchor_bpm, ratio)
            except Exception as exc:
                log.warning("  %s: time-stretch failed: %s", stem_name, exc)

        stem_arrays[stem_name] = arr
        min_length = min(min_length, len(arr))

    if not stem_arrays:
        return ComboResult(combo=combo, error="No stems could be loaded")

    # ── Align lengths ─────────────────────────────────────────────────────
    if target_duration:
        target_samples = int(target_duration * sr)
    else:
        target_samples = int(min_length) if min_length < float('inf') else sr * 30

    # ── Mix stems ─────────────────────────────────────────────────────────
    mixed = np.zeros(target_samples, dtype=np.float32)

    for stem_name, arr in stem_arrays.items():
        trimmed = _pad_or_trim(arr, target_samples)
        mixed += trimmed

    # ── Normalise ─────────────────────────────────────────────────────────
    peak = float(np.abs(mixed).max())
    if normalize and peak > 0:
        mixed = mixed / peak * 0.891  # -1 dBFS headroom

    peak_dbfs = float(20.0 * np.log10(float(np.abs(mixed).max()) + 1e-10))

    # ── Compute LUFS if mastering module available ────────────────────────
    lufs = -70.0
    try:
        from scripts.core.mastering import compute_lufs
        lufs = compute_lufs(mixed, sr)
    except Exception:
        pass

    return ComboResult(
        combo=combo,
        audio=mixed,
        bpm=anchor_bpm,
        duration_sec=round(len(mixed) / sr, 1),
        peak_dbfs=round(peak_dbfs, 1),
        lufs=round(lufs, 1),
        success=True,
    )


# ---------------------------------------------------------------------------
# Batch render — all combos
# ---------------------------------------------------------------------------

def render_all_combos(
    songs: List[str],
    output_dir: Path,
    sr: int = 44100,
    mode: str = "all",
    target_duration: Optional[float] = None,
    swap_stems: Optional[List[str]] = None,
    include_pure: bool = False,
    progress_cb=None,
) -> LabSession:
    """
    Enumerate and render all instrument combinations for a set of songs.

    Parameters
    ----------
    songs : list of song names (must have stems in library/)
    output_dir : where to save rendered WAV files
    sr : sample rate
    mode : "all" for full permutation, "targeted" for single-stem swaps only
    target_duration : if set, trim to this many seconds (useful for previews)
    swap_stems : which stems to swap (for targeted mode)
    include_pure : include combos where all stems come from one song
    progress_cb : optional callable(current_idx, total, combo_label, result)

    Returns
    -------
    LabSession with all results
    """
    import soundfile as sf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Enumerate combos ──────────────────────────────────────────────────
    if mode == "targeted":
        combos = enumerate_targeted_swaps(songs, swap_stems=swap_stems)
    else:
        combos = enumerate_combos(songs, include_pure=include_pure)

    session = LabSession(
        songs=songs,
        total_combos=len(combos),
        output_dir=output_dir,
    )

    log.info("Instrument Lab: rendering %d combos for [%s] → %s",
             len(combos), ", ".join(s[:25] for s in songs), output_dir)

    for i, combo in enumerate(combos):
        # ── Progress callback ─────────────────────────────────────────────
        if progress_cb:
            try:
                progress_cb(i, len(combos), combo.short_label(), None)
            except Exception:
                pass

        # ── Skip pure combos if not requested ─────────────────────────────
        if combo.is_pure and not include_pure:
            session.skipped += 1
            continue

        # ── Render ────────────────────────────────────────────────────────
        result = render_combo(combo, sr=sr, target_duration=target_duration)

        if result.success and result.audio is not None:
            # ── Save to WAV ───────────────────────────────────────────────
            filename = _combo_filename(combo, i)
            out_path = output_dir / filename
            sf.write(str(out_path), result.audio, sr, subtype="PCM_16")
            result.output_path = out_path
            result.audio = None  # free memory
            session.rendered += 1
            log.info("  [%d/%d] ✓ %s → %s  (%.1f s, %.1f LUFS)",
                     i + 1, len(combos), combo.short_label(),
                     filename, result.duration_sec, result.lufs)
        else:
            session.failed += 1
            log.warning("  [%d/%d] ✗ %s → %s",
                        i + 1, len(combos), combo.short_label(),
                        result.error or "unknown error")

        session.results.append(result)

        if progress_cb:
            try:
                progress_cb(i, len(combos), combo.short_label(), result)
            except Exception:
                pass

    log.info("Instrument Lab complete: %d rendered, %d failed, %d skipped",
             session.rendered, session.failed, session.skipped)
    return session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _combo_filename(combo: InstrumentCombo, index: int) -> str:
    """Generate a descriptive filename for a combo."""
    songs = list(set(combo.mapping.values()))
    song_map = {s: chr(65 + i) for i, s in enumerate(songs)}

    parts = []
    for stem in STEM_NAMES:
        song = combo.mapping.get(stem, "")
        letter = song_map.get(song, "?")
        parts.append(f"{stem[0]}{letter}")

    tag = "".join(parts)  # e.g. "vAdBbAoB"
    return f"combo_{index:03d}_{tag}.wav"


def get_songs_with_stems(min_stems: int = 2) -> List[str]:
    """Return all library songs that have at least `min_stems` separated stems."""
    from scripts.core.paths import LIBRARY_DIR

    result = []
    if not LIBRARY_DIR.exists():
        return result

    for song_dir in sorted(LIBRARY_DIR.iterdir()):
        if not song_dir.is_dir():
            continue
        stem_count = sum(
            1 for s in STEM_NAMES
            if (song_dir / f"{s}.wav").exists() or (song_dir / f"{s}.flac").exists()
        )
        if stem_count >= min_stems:
            result.append(song_dir.name)

    return result


def preview_combo(
    combo: InstrumentCombo,
    sr: int = 44100,
    duration: float = 15.0,
) -> ComboResult:
    """Render a short preview (default 15s) of a combo — fast for UI."""
    return render_combo(combo, sr=sr, target_duration=duration)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    ap = argparse.ArgumentParser(description="Instrument Lab — stem swap experiments")
    ap.add_argument("songs", nargs="+", help="2+ song names from library/")
    ap.add_argument("--mode", default="targeted", choices=["all", "targeted"],
                    help="Combo generation mode (default: targeted single-stem swaps)")
    ap.add_argument("--duration", type=float, default=None,
                    help="Trim output to N seconds")
    ap.add_argument("--output", default="outputs/instrument_lab",
                    help="Output directory")
    args = ap.parse_args()

    session = render_all_combos(
        songs=args.songs,
        output_dir=Path(args.output),
        mode=args.mode,
        target_duration=args.duration,
        progress_cb=lambda i, t, label, r: print(f"  [{i+1}/{t}] {label}"),
    )

    print(f"\n✅ Instrument Lab: {session.rendered} rendered, "
          f"{session.failed} failed, {session.skipped} skipped")
