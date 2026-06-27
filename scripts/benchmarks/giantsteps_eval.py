"""
scripts/benchmarks/giantsteps_eval.py — GiantSteps key detection benchmark.

Evaluates detect_key() accuracy across four tonal profiles (ks, edma, edmm,
auto) against the GiantSteps dataset of 604 annotated Beatport EDM tracks.

Modes
-----
Real-data mode (requires dataset):
    python scripts/benchmarks/giantsteps_eval.py \\
        --annotations-dir data/giantsteps/annotations \\
        --audio-dir       data/giantsteps/audio

Synthetic mode (CI-friendly, no dataset needed):
    python scripts/benchmarks/giantsteps_eval.py
    # runs when --audio-dir does not exist

Scoring
-------
Uses the MIREX weighted key evaluation scheme:
    Correct (exact match)   : 1.0
    Fifth relation          : 0.5  (±7 or ±5 semitones, same mode)
    Relative major/minor    : 0.3  (same notes, different mode)
    Parallel major/minor    : 0.2  (same tonic, different mode)
    Other                   : 0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROFILES: Tuple[str, ...] = ('ks', 'edma', 'edmm', 'auto')

# Semitone offset from C, including enharmonic equivalents
_NOTE_SEMI: Dict[str, int] = {
    'C': 0,  'C#': 1,  'Db': 1,  'D': 2,  'D#': 3,  'Eb': 3,
    'E': 4,  'F': 5,   'F#': 6,  'Gb': 6, 'G': 7,   'G#': 8,
    'Ab': 8, 'A': 9,   'A#': 10, 'Bb': 10,'B': 11,
}

# 20 synthetic ground-truth keys used in CI mode:
#   7 white-key major + 7 white-key minor + 6 accidentals
_SYNTHETIC_KEYS: List[Tuple[str, str]] = [
    ('C', 'major'), ('D', 'major'), ('E', 'major'), ('F', 'major'),
    ('G', 'major'), ('A', 'major'), ('B', 'major'),
    ('C', 'minor'), ('D', 'minor'), ('E', 'minor'), ('F', 'minor'),
    ('G', 'minor'), ('A', 'minor'), ('B', 'minor'),
    ('C#', 'major'), ('D#', 'major'), ('F#', 'major'),
    ('G#', 'major'), ('A#', 'major'), ('C#', 'minor'),
]


# ---------------------------------------------------------------------------
# MIREX weighted score
# ---------------------------------------------------------------------------

def mirex_score(pred_key: str, pred_mode: str, ref_key: str, ref_mode: str) -> float:
    """
    Return the MIREX-standard weighted key evaluation score.

    Returns one of {1.0, 0.5, 0.3, 0.2, 0.0}.

    Parameters
    ----------
    pred_key, pred_mode : predicted root and mode (e.g. 'G', 'major')
    ref_key, ref_mode   : ground-truth root and mode
    """
    pred_s = _NOTE_SEMI.get(pred_key, -1)
    ref_s  = _NOTE_SEMI.get(ref_key,  -1)
    if pred_s < 0 or ref_s < 0:
        return 0.0

    # Exact match
    if pred_s == ref_s and pred_mode == ref_mode:
        return 1.0

    delta = (pred_s - ref_s) % 12

    # Fifth relation: dominant (+7) or subdominant (+5), same mode
    if pred_mode == ref_mode and delta in {5, 7}:
        return 0.5

    # Relative major/minor (same pitch classes, different tonic and mode)
    if ref_mode == 'major' and pred_mode == 'minor' and delta == 9:
        return 0.3
    if ref_mode == 'minor' and pred_mode == 'major' and delta == 3:
        return 0.3

    # Parallel mode: same root, different mode
    if pred_s == ref_s:
        return 0.2

    return 0.0


def _classify(score: float) -> str:
    """Map a MIREX score to its category name."""
    return {1.0: 'correct', 0.5: 'fifth', 0.3: 'relative', 0.2: 'parallel'}.get(
        score, 'other'
    )


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def _parse_annotation(path: Path) -> Optional[Tuple[str, str]]:
    """
    Parse a GiantSteps .key file → (key_name, mode) or None on failure.

    File format: a single line such as ``A major`` or ``F# minor``.
    """
    try:
        text = path.read_text(encoding='utf-8').strip()
        parts = text.split()
        if len(parts) >= 2:
            # Title-case normalisation: 'c#' → 'C#', 'db' → 'Db'
            raw = parts[0]
            key_norm = raw[0].upper() + raw[1:] if len(raw) > 1 else raw.upper()
            mode = parts[-1].lower()
            if mode in ('major', 'minor') and key_norm in _NOTE_SEMI:
                return key_norm, mode
    except Exception as exc:
        log.warning("Failed to parse annotation %s: %s", path, exc)
    return None


# ---------------------------------------------------------------------------
# Synthetic audio generation
# ---------------------------------------------------------------------------

def _make_key_tone(
    key: str,
    mode: str,
    duration: float = 5.0,
    sr: int = 22050,
) -> np.ndarray:
    """
    Generate a synthetic float32 audio clip whose chroma reflects ``key/mode``.

    Produces the first six partials of a major or minor chord rooted at the
    tonic (root, third, fifth, octave, tenth, twelfth) with decreasing
    amplitude — giving chroma-based detectors a clean, realistic signal
    without requiring a real audio file.
    """
    from scripts.core.key_detection import _note_to_hz

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.zeros(len(t), dtype=np.float64)

    root_hz = _note_to_hz(key)
    # Semitone intervals above root for each chord quality
    intervals = (
        [0, 4, 7, 12, 16, 19]  # major: root, M3, P5, octave, M10, P12
        if mode == 'major' else
        [0, 3, 7, 12, 15, 19]  # minor: root, m3, P5, octave, m10, P12
    )

    for rank, semitone in enumerate(intervals):
        freq = root_hz * (2.0 ** (semitone / 12.0))
        amp  = 1.0 / (rank + 1)
        audio += amp * np.sin(2.0 * np.pi * freq * t)

    peak = float(np.abs(audio).max())
    if peak > 1e-6:
        audio /= peak
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-track evaluation
# ---------------------------------------------------------------------------

def _eval_track(
    audio: np.ndarray,
    sr: int,
    ref_key: str,
    ref_mode: str,
) -> Dict[str, Dict]:
    """
    Run all four profiles on a single audio clip.

    Returns
    -------
    Dict mapping each profile name to a dict with keys
    ``score``, ``category``, ``pred_key``, ``pred_mode``.
    """
    from scripts.core.key_detection import detect_key

    results: Dict[str, Dict] = {}
    for profile in PROFILES:
        try:
            kr    = detect_key(audio, sr, profile=profile)
            score = mirex_score(kr.key_name, kr.mode, ref_key, ref_mode)
            results[profile] = {
                'score':     score,
                'category':  _classify(score),
                'pred_key':  kr.key_name,
                'pred_mode': kr.mode,
            }
        except Exception as exc:
            log.warning("detect_key(profile=%s) failed: %s", profile, exc)
            results[profile] = {
                'score': 0.0, 'category': 'other',
                'pred_key': '?', 'pred_mode': '?',
            }
    return results


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _aggregate(per_track: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    """Compute per-profile aggregate MIREX metrics over all tracks."""
    n = len(per_track)
    totals: Dict[str, Dict[str, float]] = {p: defaultdict(float) for p in PROFILES}

    for track in per_track:
        for profile, res in track.items():
            totals[profile][res['category']] += 1.0
            totals[profile]['weighted_sum']   += res['score']

    summary: Dict[str, Dict] = {}
    for profile in PROFILES:
        t = totals[profile]
        denom = max(n, 1)
        summary[profile] = {
            'correct':  round(t['correct']          / denom, 6),
            'fifth':    round(t['fifth']             / denom, 6),
            'relative': round(t['relative']          / denom, 6),
            'parallel': round(t['parallel']          / denom, 6),
            'other':    round(t['other']             / denom, 6),
            'weighted': round(t['weighted_sum']      / denom, 6),
            'n_tracks': n,
        }
    return summary


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_table(summary: Dict[str, Dict]) -> None:
    header = (
        f"{'Profile':<8} {'Correct':>8} {'Fifth':>8} "
        f"{'Relative':>10} {'Parallel':>9} {'Other':>7} {'Weighted':>9}"
    )
    print()
    print(header)
    print('-' * len(header))
    for profile in PROFILES:
        s = summary[profile]
        print(
            f"{profile:<8}"
            f" {s['correct']:>7.1%}"
            f" {s['fifth']:>8.1%}"
            f" {s['relative']:>10.1%}"
            f" {s['parallel']:>9.1%}"
            f" {s['other']:>7.1%}"
            f" {s['weighted']:>9.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# Synthetic mode (CI-friendly)
# ---------------------------------------------------------------------------

def run_synthetic(sr: int = 22050, verbose: bool = True) -> Dict[str, Dict]:
    """
    Evaluate all four profiles on 20 synthetic chord tones with known ground truth.

    Generates each tone programmatically — no real audio files required.
    Returns per-profile aggregate MIREX metrics dict.

    Parameters
    ----------
    sr : int
        Sample rate for synthesis (default 22050).
    verbose : bool
        Print per-track predictions and summary table if True.

    Returns
    -------
    Dict[str, Dict] — one entry per profile with keys
    ``correct``, ``fifth``, ``relative``, ``parallel``, ``other``,
    ``weighted``, ``n_tracks``.
    """
    if verbose:
        print(
            f"[Synthetic mode] Evaluating {len(_SYNTHETIC_KEYS)} tones "
            f"across {len(PROFILES)} profiles…"
        )

    per_track: List[Dict[str, Dict]] = []
    for key, mode in _SYNTHETIC_KEYS:
        audio   = _make_key_tone(key, mode, sr=sr)
        results = _eval_track(audio, sr, ref_key=key, ref_mode=mode)
        per_track.append(results)

        if verbose:
            preds  = {p: f"{results[p]['pred_key']} {results[p]['pred_mode']}" for p in PROFILES}
            scores = {p: f"{results[p]['score']:.1f}" for p in PROFILES}
            print(f"  GT: {key:3s} {mode:<6}  preds={preds}  scores={scores}")

    summary = _aggregate(per_track)

    if verbose:
        print("\n── Synthetic Benchmark Results ──")
        _print_table(summary)

    return summary


# ---------------------------------------------------------------------------
# Real-data mode
# ---------------------------------------------------------------------------

def run_real(
    annotations_dir: Path,
    audio_dir: Path,
    sr: int = 22050,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate all four profiles against real GiantSteps .key annotations.

    Audio files and annotation files are matched by stem name
    (e.g. ``track_123.mp3`` ↔ ``track_123.key``).

    Parameters
    ----------
    annotations_dir : Path
        Directory containing ``.key`` annotation files.
    audio_dir : Path
        Directory containing audio files (``.mp3``, ``.wav``, ``.flac``).
    sr : int
        Sample rate for audio loading (default 22050).
    verbose : bool
        Print progress and summary table if True.

    Returns
    -------
    Dict[str, Dict] — same structure as run_synthetic(), plus empty dict on
    configuration errors.
    """
    ann_files = list(annotations_dir.glob('*.key'))
    if not ann_files:
        print(f"No .key files found in {annotations_dir}", file=sys.stderr)
        return {}

    ann_map: Dict[str, Tuple[str, str]] = {}
    for af in ann_files:
        parsed = _parse_annotation(af)
        if parsed:
            ann_map[af.stem] = parsed

    if verbose:
        print(
            f"[Real mode] {len(ann_map)} valid annotations in {annotations_dir}"
        )

    audio_files: List[Path] = []
    for ext in ('.mp3', '.wav', '.flac', '.ogg'):
        audio_files.extend(audio_dir.glob(f'*{ext}'))

    matched = [
        (af, ann_map[af.stem])
        for af in audio_files
        if af.stem in ann_map
    ]
    if verbose:
        print(f"[Real mode] {len(matched)} audio↔annotation pairs matched")

    if not matched:
        print("No matched audio+annotation pairs found.", file=sys.stderr)
        return {}

    try:
        import librosa as _librosa
    except ImportError:
        print("librosa is required for real-data mode.", file=sys.stderr)
        return {}

    per_track: List[Dict[str, Dict]] = []
    for i, (audio_path, (ref_key, ref_mode)) in enumerate(matched):
        try:
            audio, _ = _librosa.load(str(audio_path), sr=sr, mono=True, duration=60.0)
            results  = _eval_track(audio, sr, ref_key=ref_key, ref_mode=ref_mode)
            per_track.append(results)
            if verbose and (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(matched)}] processed…")
        except Exception as exc:
            log.warning("Skipping %s: %s", audio_path.name, exc)

    summary = _aggregate(per_track)
    if verbose:
        print(f"\n── GiantSteps Results ({len(per_track)} tracks) ──")
        _print_table(summary)
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Parse CLI arguments and run the appropriate benchmark mode.

    Returns the summary dict for programmatic use.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate detect_key() tonal profiles against GiantSteps key annotations. "
            "Runs in synthetic mode when --audio-dir does not exist."
        )
    )
    parser.add_argument(
        '--annotations-dir',
        type=Path,
        default=Path('data/giantsteps'),
        help='Directory of .key annotation files (default: data/giantsteps)',
    )
    parser.add_argument(
        '--audio-dir',
        type=Path,
        default=Path('data/giantsteps'),
        help='Directory of audio files; if absent, runs in synthetic mode (default: data/giantsteps)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/giantsteps_results.json'),
        help='JSON output path (default: data/giantsteps_results.json)',
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=22050,
        help='Sample rate for audio loading / synthesis (default: 22050)',
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

    if not args.audio_dir.exists():
        print(
            f"Audio directory {args.audio_dir} not found — "
            "running in synthetic mode (no dataset required)."
        )
        summary = run_synthetic(sr=args.sr)
    else:
        summary = run_real(
            annotations_dir=args.annotations_dir,
            audio_dir=args.audio_dir,
            sr=args.sr,
        )

    if summary:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"Results saved → {args.output}")

    return summary


if __name__ == '__main__':
    main()
