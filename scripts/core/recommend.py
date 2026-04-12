"""
scripts/core/recommend.py — Fast BPM-based song recommendation engine.

Strategy
--------
Each song directory gets a lightweight ``meta.json`` cache file holding at
minimum ``{"bpm": <float>}``.  This file is written:

  • By this module on first access (lazy, using a 10-second audio clip).
  • By ``task_analyze`` in tasks.py after a full analysis job (richer data).

On a first call with no warm cache the engine will process up to
``MAX_UNCACHED_PER_CALL`` songs (≈ 0.1–0.2 s each on a modern laptop).
Subsequent calls are instant because every song reads from its cached file.

Usage::

    from scripts.core.recommend import get_recommendations
    from pathlib import Path

    recs = get_recommendations(
        source_dir=Path("library/Anyma - Explore"),
        library_dir=Path("library"),
        limit=5,
    )
    # [{"name": "...", "bpm": 128.0, "bpm_score": 0.97, "overall": 0.97}, ...]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_FILE = "meta.json"

# Max number of *uncached* songs to process per call.
# Keeps worst-case latency under ~20–30 s on first cold call.
MAX_UNCACHED_PER_CALL = 120

# Early-exit: if we already have this many high-quality matches (score ≥ 0.7)
# from the cache, skip uncached songs entirely.
EARLY_EXIT_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _quick_bpm(wav_path: Path, duration: float = 10.0, sr: int = 22050) -> Optional[float]:
    """
    Load a short clip and estimate BPM.  Returns None on any failure.
    Using 10 s is enough for a reliable tempo estimate and loads in < 0.2 s.
    """
    try:
        audio, _ = librosa.load(str(wav_path), sr=sr, mono=True, duration=duration)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        return bpm if bpm > 0 else None
    except Exception as exc:
        log.debug("_quick_bpm failed for %s: %s", wav_path, exc)
        return None


def _load_cache(song_dir: Path) -> Dict[str, Any]:
    """Read meta.json; return empty dict on failure."""
    try:
        cache_path = song_dir / _CACHE_FILE
        if cache_path.exists():
            return json.loads(cache_path.read_text())
    except Exception:
        pass
    return {}


def _save_cache(song_dir: Path, data: Dict[str, Any]) -> None:
    """Write meta.json; silently ignore errors."""
    try:
        (song_dir / _CACHE_FILE).write_text(json.dumps(data))
    except Exception:
        pass


def get_or_cache_bpm(song_dir: Path) -> Optional[float]:
    """
    Return BPM for a song — from cache if available, otherwise compute and
    cache it.  Public so task_analyze can call it after a full analysis.
    """
    data = _load_cache(song_dir)
    if "bpm" in data:
        return float(data["bpm"])

    wav = song_dir / "full.wav"
    if not wav.exists():
        return None

    bpm = _quick_bpm(wav)
    if bpm is not None:
        data["bpm"] = round(bpm, 1)
        _save_cache(song_dir, data)

    return bpm


def write_meta_cache(
    song_dir: Path,
    bpm: float,
    genre: str = "",
    camelot: str = "",
    energy_mean: float = -1.0,
) -> None:
    """
    Called by task_analyze to persist richer metadata after a full analysis.
    Merges with any existing cache so we don't overwrite extra fields.

    Extended fields (from DJ_THEORY.md §3 — Compatibility Scoring Weights):
      camelot     — Camelot Wheel notation (e.g. "8A"), used for harmonic scoring
      energy_mean — Normalised RMS energy 0–1, used for energy smoothness scoring
    """
    data = _load_cache(song_dir)
    data["bpm"] = round(bpm, 1)
    if genre:
        data["genre"] = genre
    if camelot:
        data["camelot"] = camelot
    if energy_mean >= 0.0:
        data["energy_mean"] = round(energy_mean, 4)
    _save_cache(song_dir, data)


# ---------------------------------------------------------------------------
# Compatibility scoring (multi-dimensional)
# ---------------------------------------------------------------------------
# Weights grounded in SetFlow algorithm and Kim et al. (ISMIR 2020) analysis
# of 1,557 real DJ mixes from 1001Tracklists. Full rationale in docs/DJ_THEORY.md §3.
#
#   harmonic_match   0.35  — Camelot Wheel distance (most important DJ constraint)
#   beat_alignment   0.25  — BPM proximity (86.1% of transitions < 5% stretch)
#   energy_smoothness 0.15 — Prevents jarring energy jumps between tracks
#   other/fallback   0.25  — Distributed across missing dimensions
#
# When camelot / energy data are absent (cold cache), the engine gracefully
# falls back to pure BPM scoring so performance is never blocked.

def _bpm_score(bpm_a: float, bpm_b: float) -> float:
    """
    Return a [0, 1] BPM compatibility score.

    Matches at 1×, 0.5×, and 2× speed (half/double-time) are all considered.
    Based on: Kim et al. (ISMIR 2020) — 86.1% of DJ transitions adjust
    tempo < 5%, making BPM proximity the primary mechanical constraint.

    ≥ 0.9  → excellent (will mix without noticeable pitch shifting)
    0.6–0.9 → good (gentle stretch / pitch-lock needed)
    0.3–0.6 → marginal (noticeable adjustment)
    < 0.3  → incompatible
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.0

    best = 0.0
    for ratio in (bpm_a / bpm_b, bpm_a / (bpm_b * 2), (bpm_a * 2) / bpm_b):
        if abs(ratio - 1.0) < 0.01:
            best = max(best, 1.0)
        elif 0.94 <= ratio <= 1.06:
            # Linear drop from 1.0 (0% diff) to 0.7 (6% diff)
            best = max(best, 1.0 - abs(ratio - 1.0) / 0.06 * 0.3)
        elif 0.90 <= ratio <= 1.10:
            # Linear drop from 0.7 (6% diff) to 0.1 (10% diff)
            best = max(best, 0.7 - (abs(ratio - 1.0) - 0.06) / 0.04 * 0.6)

    return round(best, 4)


def _camelot_score(camelot_a: str, camelot_b: str) -> float:
    """
    Return a [0, 1] harmonic compatibility score from Camelot Wheel positions.

    Implements the seven canonical Camelot moves from DJ_THEORY.md §2:

      Same key                  → 1.00  (identical notes)
      ±1 adjacent (same letter) → 0.90  (one note changes, very safe)
      Parallel mode (same num)  → 0.85  (same root, mood shift without clash)
      +7 energy boost           → 0.80  (1-semitone up, euphoric — Armin's move)
      ±2 double step            → 0.65  (slight tension)
      ±3–5 distant              → 0.30–0.50  (significant clash)
      ≥6 clash                  → 0.05  (maximum dissonance — avoid)

    The +7 clockwise move (e.g. 12A → 7A) produces a 1-semitone upward pitch
    shift. It technically clashes during extended overlap but is a deliberate
    DJ technique for euphoric "energy boost" moments. Use with quick transitions.
    """
    if not camelot_a or not camelot_b:
        return 0.5  # unknown → neutral

    def _parse(c: str):
        try:
            return int(c[:-1]), c[-1].upper()
        except (ValueError, IndexError):
            return 8, 'B'

    num_a, let_a = _parse(camelot_a)
    num_b, let_b = _parse(camelot_b)

    dist_cw  = (num_b - num_a) % 12
    dist_ccw = (num_a - num_b) % 12
    dist     = min(dist_cw, dist_ccw)

    # Same key
    if dist == 0 and let_a == let_b:
        return 1.00

    # Parallel mode (same root note, e.g. Am → A major)
    if dist == 0:
        return 0.85

    # +7 clockwise energy boost (1-semitone upward shift — deliberate DJ move)
    if dist_cw == 7 and let_a == let_b:
        return 0.80

    # ±1 adjacent (safest key change — one scale note difference)
    if dist == 1:
        return 0.90

    # ±2 double step (acceptable with short overlap)
    if dist == 2:
        return 0.65

    # ±3–5 distant (risky — use effects-only transitions)
    if dist <= 5:
        return round(0.50 - (dist - 3) * 0.10, 2)

    # ≥6 clash (avoid — maximum dissonance)
    return 0.05


def _energy_score(energy_a: float, energy_b: float) -> float:
    """
    Return a [0, 1] energy smoothness score.

    Penalises large energy gaps between consecutive tracks. Based on:
    DJ_THEORY.md §1 — energy arc management requires smooth steps (±1 level
    is Armin Van Buuren's documented pattern; large jumps disorient the crowd).

    Gap < 0.10 → near-perfect match
    Gap 0.10–0.30 → acceptable
    Gap > 0.50 → harsh jump
    """
    if energy_a < 0 or energy_b < 0:
        return 0.5  # missing data → neutral
    gap = abs(energy_a - energy_b)
    return round(float(np.clip(1.0 - gap * 2.0, 0.0, 1.0)), 4)


def _composite_score(
    bpm_score: float,
    camelot_score: float,
    energy_score: float,
) -> float:
    """
    Weighted composite compatibility score.

    Weights from SetFlow algorithm, validated against 1001Tracklists analysis:
      harmonic  35%  — Camelot (most important DJ constraint)
      BPM       25%  — Tempo proximity
      energy    15%  — Dynamics smoothness
      remainder 25%  — Distributed proportionally when data is present

    When harmonic or energy data is absent (cold cache), BPM gets a higher
    weight so the score remains useful.
    """
    has_harmonic = camelot_score != 0.5  # 0.5 = neutral sentinel
    has_energy   = energy_score  != 0.5

    if has_harmonic and has_energy:
        # Full multi-dimensional score
        return round(
            0.35 * camelot_score
            + 0.25 * bpm_score
            + 0.15 * energy_score
            + 0.25 * (0.5 * camelot_score + 0.5 * bpm_score),  # fill remainder
            4,
        )
    elif has_harmonic:
        return round(0.45 * camelot_score + 0.55 * bpm_score, 4)
    elif has_energy:
        return round(0.65 * bpm_score + 0.35 * energy_score, 4)
    else:
        return bpm_score  # cold cache fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recommendations(
    source_dir: Path,
    library_dir: Path,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return the top ``limit`` songs from ``library_dir`` that are most BPM-
    compatible with the song at ``source_dir``.

    Algorithm
    ---------
    1. Get source BPM (cache-first).
    2. Scan library — separate songs with cached BPM vs uncached.
    3. Score all cached songs instantly (dict lookup + arithmetic).
    4. Early-exit if we already have enough high-quality matches.
    5. Otherwise process up to MAX_UNCACHED_PER_CALL uncached songs (writes
       cache as a side-effect so subsequent calls are faster).
    6. Sort by overall score, return top N.

    Returns a list of dicts::

        [
            {
                "name":      "Anyma - Explore",
                "bpm":       128.0,
                "bpm_score": 0.972,
                "overall":   0.972,
            },
            ...
        ]
    """
    source_bpm   = get_or_cache_bpm(source_dir)
    source_cache = _load_cache(source_dir)   # needed for harmonic + energy dims
    if source_bpm is None:
        log.warning("Could not determine BPM for source song: %s", source_dir.name)
        return []

    # Collect candidate directories (exclude source, require full.wav)
    all_dirs = [
        d for d in library_dir.iterdir()
        if d.is_dir()
        and d.name != source_dir.name
        and (d / "full.wav").exists()
    ]

    # Partition into cached vs uncached
    cached_dirs, uncached_dirs = [], []
    for d in all_dirs:
        cache = _load_cache(d)
        if "bpm" in cache:
            cached_dirs.append((d, float(cache["bpm"])))
        else:
            uncached_dirs.append(d)

    source_camelot     = source_cache.get("camelot", "")
    source_energy_mean = float(source_cache.get("energy_mean", -1.0))

    results: List[Dict[str, Any]] = []

    # ── Phase 1: score cached songs (instant) ──────────────────────────────
    for d, bpm in cached_dirs:
        cache = _load_cache(d)
        bpm_sc  = _bpm_score(source_bpm, bpm)
        cam_sc  = _camelot_score(source_camelot, cache.get("camelot", ""))
        eng_sc  = _energy_score(source_energy_mean, float(cache.get("energy_mean", -1.0)))
        overall = _composite_score(bpm_sc, cam_sc, eng_sc)
        if overall > 0.0:
            results.append({
                "name":           d.name,
                "bpm":            round(bpm, 1),
                "bpm_score":      bpm_sc,
                "harmonic_score": cam_sc if cam_sc != 0.5 else None,
                "energy_score":   eng_sc  if eng_sc  != 0.5 else None,
                "camelot":        cache.get("camelot", ""),
                "overall":        overall,
            })

    # Early exit: if we already have plenty of good matches, skip cold songs
    high_quality = [r for r in results if r["overall"] >= 0.7]
    if len(high_quality) >= EARLY_EXIT_THRESHOLD:
        results.sort(key=lambda x: x["overall"], reverse=True)
        return results[:limit]

    # ── Phase 2: process uncached songs (lazy cache population) ───────────
    for d in uncached_dirs[:MAX_UNCACHED_PER_CALL]:
        bpm = get_or_cache_bpm(d)  # writes meta.json as side-effect
        if bpm is None:
            continue
        cache = _load_cache(d)
        bpm_sc  = _bpm_score(source_bpm, bpm)
        cam_sc  = _camelot_score(source_camelot, cache.get("camelot", ""))
        eng_sc  = _energy_score(source_energy_mean, float(cache.get("energy_mean", -1.0)))
        overall = _composite_score(bpm_sc, cam_sc, eng_sc)
        if overall > 0.0:
            results.append({
                "name":           d.name,
                "bpm":            round(bpm, 1),
                "bpm_score":      bpm_sc,
                "harmonic_score": cam_sc if cam_sc != 0.5 else None,
                "energy_score":   eng_sc  if eng_sc  != 0.5 else None,
                "camelot":        cache.get("camelot", ""),
                "overall":        overall,
            })

    results.sort(key=lambda x: x["overall"], reverse=True)
    return results[:limit]
