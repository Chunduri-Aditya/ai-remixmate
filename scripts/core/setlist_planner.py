"""
scripts/core/setlist_planner.py — Algorithmic Setlist Optimizer

Implements the full pipeline described in DJ theory:
  1. Exportify CSV/Excel import (Spotify playlist export)
  2. Weighted greedy setlist ordering (50% harmonic, 30% BPM, 20% energy)
  3. Energy arc models (Ramp Up, Mountain, Wave, Ramp Down)
  4. Markov chain transition scoring from historical setlists
  5. Spectral flux energy metric (local, no API dependency)

Usage:
    from scripts.core.setlist_planner import SetlistPlanner, EnergyArc, TrackNode
    planner = SetlistPlanner()
    tracks  = planner.load_exportify_csv("my_playlist.csv")
    ordered = planner.optimize(tracks, arc=EnergyArc.MOUNTAIN, start_track="Anyma - Voices In My Head")
    planner.export_csv(ordered, "optimized_set.csv")
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.core.key_detection import camelot_distance, camelot_modulation, CAMELOT

log = logging.getLogger(__name__)

# ── Spotify key integer → Camelot mapping ────────────────────────────────────
# Spotify returns key as 0–11 (0=C, 1=C#, ...) and mode as 0=minor / 1=major
_SPOTIFY_KEY_CAMELOT: Dict[Tuple[int, int], str] = {
    (0, 1): "8B",   # C major
    (1, 1): "3B",   # C# / Db major
    (2, 1): "10B",  # D major
    (3, 1): "5B",   # D# / Eb major
    (4, 1): "12B",  # E major
    (5, 1): "7B",   # F major
    (6, 1): "2B",   # F# / Gb major
    (7, 1): "9B",   # G major
    (8, 1): "4B",   # G# / Ab major
    (9, 1): "11B",  # A major
    (10, 1): "6B",  # A# / Bb major
    (11, 1): "1B",  # B major
    (0, 0): "5A",   # C minor
    (1, 0): "12A",  # C# / Db minor
    (2, 0): "7A",   # D minor
    (3, 0): "2A",   # D# / Eb minor
    (4, 0): "9A",   # E minor
    (5, 0): "4A",   # F minor
    (6, 0): "11A",  # F# / Gb minor
    (7, 0): "6A",   # G minor
    (8, 0): "1A",   # G# / Ab minor
    (9, 0): "8A",   # A minor
    (10, 0): "3A",  # A# / Bb minor
    (11, 0): "10A", # B minor
}

# ── Energy arc definitions ────────────────────────────────────────────────────

class EnergyArc(str, Enum):
    RAMP_UP  = "ramp_up"    # Steady climb warm-up → peak
    MOUNTAIN = "mountain"   # Slow build → single peak → resolve
    WAVE     = "wave"       # Alternating peaks and valleys
    RAMP_DOWN = "ramp_down" # Gradual deceleration (after-hours / sunrise)


def _arc_target_energy(position: float, arc: EnergyArc) -> float:
    """
    Return the ideal energy level [0.0, 1.0] at a fractional set position [0.0, 1.0].

    position: 0.0 = first track, 1.0 = last track.
    """
    t = position
    if arc == EnergyArc.RAMP_UP:
        return 0.3 + 0.7 * t

    elif arc == EnergyArc.MOUNTAIN:
        # Bell-curve peak at 70% of the set
        peak = 0.70
        if t <= peak:
            return 0.2 + 0.8 * (t / peak)
        else:
            return 1.0 - 0.6 * ((t - peak) / (1.0 - peak))

    elif arc == EnergyArc.WAVE:
        # Sinusoidal: 2.5 full cycles across the set
        base = 0.5 + 0.4 * math.sin(2 * math.pi * 2.5 * t)
        # Trend slightly upward overall
        return min(1.0, base + 0.1 * t)

    elif arc == EnergyArc.RAMP_DOWN:
        return max(0.1, 1.0 - 0.8 * t)

    return 0.5


# ── Track data model ──────────────────────────────────────────────────────────

@dataclass
class TrackNode:
    """
    Normalised representation of a track for setlist planning.
    Fields align with Exportify CSV columns.
    """
    name: str
    artist: str = ""
    album: str = ""

    # Audio features (Spotify / Exportify scale)
    bpm: float = 0.0
    energy: float = 0.5       # 0.0–1.0
    danceability: float = 0.5 # 0.0–1.0
    valence: float = 0.5      # 0.0–1.0 (emotional positivity)
    loudness_db: float = -10.0
    speechiness: float = 0.0
    instrumentalness: float = 0.5
    acousticness: float = 0.0
    liveness: float = 0.1

    # Harmonic
    camelot: str = ""
    spotify_key: int = -1     # 0–11
    spotify_mode: int = -1    # 0=minor, 1=major

    # Meta
    duration_ms: int = 0
    popularity: int = 0
    genres: List[str] = field(default_factory=list)
    spotify_id: str = ""

    @property
    def display(self) -> str:
        return f"{self.artist} — {self.name}" if self.artist else self.name

    @property
    def energy_level(self) -> int:
        """Integer energy level 1–10 (for display and arc planning)."""
        return max(1, min(10, round(self.energy * 10)))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "artist": self.artist,
            "album": self.album,
            "bpm": round(self.bpm, 1),
            "energy": round(self.energy, 3),
            "energy_level": self.energy_level,
            "danceability": round(self.danceability, 3),
            "valence": round(self.valence, 3),
            "camelot": self.camelot,
            "loudness_db": round(self.loudness_db, 1),
            "duration_ms": self.duration_ms,
            "popularity": self.popularity,
            "spotify_id": self.spotify_id,
        }


# ── Transition cost calculator ────────────────────────────────────────────────

@dataclass
class TransitionScore:
    """Breakdown of how well two tracks transition into each other."""
    harmonic_cost: float   # 0.0 = perfect, 1.0 = worst
    bpm_cost: float
    energy_cost: float
    total_cost: float
    modulation: dict       # camelot_modulation() result
    bpm_delta: float
    energy_delta: float


def transition_cost(
    a: TrackNode,
    b: TrackNode,
    arc_position: float = 0.5,
    arc: EnergyArc = EnergyArc.MOUNTAIN,
    w_harmonic: float = 0.50,
    w_bpm: float = 0.30,
    w_energy: float = 0.20,
) -> TransitionScore:
    """
    Compute the weighted transition cost from track A to track B.

    Weights follow the academic consensus from the DJ theory literature:
      50% harmonic compatibility
      30% BPM similarity
      20% energy flow

    Lower cost = better transition.
    """
    # ── Harmonic cost (Camelot distance) ──────────────────────────────────────
    if a.camelot and b.camelot:
        mod = camelot_modulation(a.camelot, b.camelot)
        harmonic_cost = mod["cost"]
    else:
        mod = {"type": "unknown", "cost": 0.5}
        harmonic_cost = 0.5

    # ── BPM cost ──────────────────────────────────────────────────────────────
    bpm_delta = abs(a.bpm - b.bpm) if a.bpm > 0 and b.bpm > 0 else 15.0
    # Penalty: 0 at same BPM, 1.0 at ±30 BPM deviation (beyond that, hard cut territory)
    bpm_cost = min(1.0, bpm_delta / 30.0)

    # ── Energy flow cost ──────────────────────────────────────────────────────
    ideal_energy = _arc_target_energy(arc_position, arc)
    energy_delta = abs(b.energy - ideal_energy)
    # Also penalise skipping more than 2 energy levels from A to B
    direct_jump = abs(b.energy - a.energy)
    energy_cost = min(1.0, energy_delta * 0.6 + direct_jump * 0.4)

    total = (
        w_harmonic * harmonic_cost
        + w_bpm * bpm_cost
        + w_energy * energy_cost
    )

    return TransitionScore(
        harmonic_cost=round(harmonic_cost, 3),
        bpm_cost=round(bpm_cost, 3),
        energy_cost=round(energy_cost, 3),
        total_cost=round(total, 3),
        modulation=mod,
        bpm_delta=round(bpm_delta, 1),
        energy_delta=round(energy_delta, 3),
    )


# ── Markov transition model ───────────────────────────────────────────────────

class MarkovTransitionModel:
    """
    Learns transition probability distributions from historical setlists.

    State = (camelot_code, energy_level_bucket).
    Trained on track sequences — predicts the best next state based on
    observed human DJ behaviour rather than pure cost minimisation.

    The Markov model acts as a tiebreaker and cultural-plausibility filter
    on top of the greedy cost function.
    """

    def __init__(self) -> None:
        # counts[state_a][state_b] = number of observed transitions a→b
        self._counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._total: Dict[str, int] = defaultdict(int)

    def _state(self, track: TrackNode) -> str:
        """Encode a track as a Markov state string."""
        bucket = max(1, min(5, round(track.energy * 5)))  # 1–5
        return f"{track.camelot or 'XX'}:E{bucket}"

    def train(self, setlist: List[TrackNode]) -> None:
        """Record a sequence of transitions from a historical setlist."""
        for i in range(len(setlist) - 1):
            sa = self._state(setlist[i])
            sb = self._state(setlist[i + 1])
            self._counts[sa][sb] += 1
            self._total[sa] += 1

    def score(self, a: TrackNode, b: TrackNode) -> float:
        """
        Return a transition plausibility score [0.0, 1.0].
        1.0 = frequently observed transition; 0.0 = never seen / unknown.
        """
        sa = self._state(a)
        sb = self._state(b)
        total = self._total.get(sa, 0)
        if total == 0:
            return 0.5  # uninformed prior
        count = self._counts[sa].get(sb, 0)
        return count / total

    def save(self, path: str | Path) -> None:
        data = {
            "counts": {k: dict(v) for k, v in self._counts.items()},
            "total": dict(self._total),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> "MarkovTransitionModel":
        data = json.loads(Path(path).read_text())
        for k, v in data["counts"].items():
            self._counts[k] = defaultdict(int, v)
        for k, v in data["total"].items():
            self._total[k] = v
        return self


# ── Exportify CSV parser ──────────────────────────────────────────────────────

def parse_exportify_csv(content: str | bytes) -> List[TrackNode]:
    """
    Parse an Exportify CSV export into TrackNode objects.

    Exportify (https://exportify.net) exports Spotify playlists with the
    following standard columns (subset we use):
      Spotify ID, Track Name, Artist Name(s), Album Name, Duration (ms),
      Popularity, Danceability, Energy, Key, Loudness, Mode, Speechiness,
      Acousticness, Instrumentalness, Liveness, Valence, Tempo, Genres

    Args:
        content: CSV text or bytes (UTF-8)

    Returns:
        List of TrackNode objects with Camelot keys resolved.
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")  # strip BOM if present

    tracks: List[TrackNode] = []
    reader = csv.DictReader(io.StringIO(content))

    for i, row in enumerate(reader):
        try:
            # Flexible column name matching (Exportify has changed headers over time)
            def _get(*keys: str, default: str = "") -> str:
                for k in keys:
                    for col, val in row.items():
                        if col.strip().lower() == k.lower():
                            return str(val).strip()
                return default

            def _float(*keys: str, default: float = 0.0) -> float:
                val = _get(*keys)
                try:
                    return float(val) if val else default
                except ValueError:
                    return default

            def _int(*keys: str, default: int = 0) -> int:
                val = _get(*keys)
                try:
                    return int(float(val)) if val else default
                except ValueError:
                    return default

            name   = _get("Track Name", "name", "title")
            artist = _get("Artist Name(s)", "artist", "artists")
            album  = _get("Album Name", "album")

            if not name:
                continue  # skip header rows or empty

            bpm   = _float("Tempo", "bpm")
            energy = _float("Energy", "energy")
            dance  = _float("Danceability", "danceability")
            valence = _float("Valence", "valence")
            loud  = _float("Loudness", "loudness", default=-10.0)
            speech = _float("Speechiness", "speechiness")
            instr  = _float("Instrumentalness", "instrumentalness")
            acoust = _float("Acousticness", "acousticness")
            live   = _float("Liveness", "liveness")

            s_key  = _int("Key", "spotify_key", default=-1)
            s_mode = _int("Mode", "spotify_mode", default=-1)
            dur    = _int("Duration (ms)", "duration_ms")
            pop    = _int("Popularity", "popularity")
            sp_id  = _get("Spotify ID", "spotify_id", "id")

            genres_raw = _get("Genres", "genre")
            genres = [g.strip() for g in genres_raw.split(",") if g.strip()]

            # Resolve Camelot from Spotify key/mode
            camelot = ""
            if s_key >= 0 and s_mode >= 0:
                camelot = _SPOTIFY_KEY_CAMELOT.get((s_key, s_mode), "")

            tracks.append(TrackNode(
                name=name,
                artist=artist,
                album=album,
                bpm=bpm,
                energy=energy,
                danceability=dance,
                valence=valence,
                loudness_db=loud,
                speechiness=speech,
                instrumentalness=instr,
                acousticness=acoust,
                liveness=live,
                camelot=camelot,
                spotify_key=s_key,
                spotify_mode=s_mode,
                duration_ms=dur,
                popularity=pop,
                genres=genres,
                spotify_id=sp_id,
            ))

        except Exception as e:
            log.warning("Exportify parser: skipping row %d: %s", i, e)

    log.info("Exportify: parsed %d tracks", len(tracks))
    return tracks


# ── Setlist optimizer ─────────────────────────────────────────────────────────

class SetlistPlanner:
    """
    Full setlist optimizer: greedy + Markov + energy arc.

    The core algorithm:
      1. Start from an anchor track (first or user-specified)
      2. At each step, evaluate all remaining tracks and score them with
         transition_cost() (weighted: 50% harmonic, 30% BPM, 20% energy)
      3. Optionally blend in Markov plausibility score to add human-feel
      4. Pick the lowest-cost next track

    The greedy approach runs in O(n²) — fine for playlist sizes up to ~1000.
    For larger libraries, a beam search fallback is available.
    """

    def __init__(self, markov_model: Optional[MarkovTransitionModel] = None) -> None:
        self.markov = markov_model

    def optimize(
        self,
        tracks: List[TrackNode],
        arc: EnergyArc = EnergyArc.MOUNTAIN,
        start_track: Optional[str] = None,
        w_harmonic: float = 0.50,
        w_bpm: float = 0.30,
        w_energy: float = 0.20,
        markov_weight: float = 0.15,
    ) -> List[dict]:
        """
        Return an optimally ordered setlist.

        Args:
            tracks:        List of TrackNode objects to order.
            arc:           Energy arc model (RAMP_UP, MOUNTAIN, WAVE, RAMP_DOWN).
            start_track:   Track name or artist–name string to anchor the set.
                           If None, picks the lowest-energy track (warm-up logic).
            w_harmonic:    Weight for harmonic compatibility (default 0.50).
            w_bpm:         Weight for BPM continuity (default 0.30).
            w_energy:      Weight for energy arc adherence (default 0.20).
            markov_weight: How much to blend in Markov plausibility. 0 = pure greedy.

        Returns:
            List of dicts, each containing the track data plus:
              position (int), transition_to_next (dict), cumulative_cost (float)
        """
        if not tracks:
            return []

        remaining = list(tracks)
        ordered: List[TrackNode] = []
        n = len(tracks)

        # ── Pick start track ──────────────────────────────────────────────────
        if start_track:
            start_track_lower = start_track.lower()
            for t in remaining:
                if (start_track_lower in t.name.lower()
                        or start_track_lower in t.display.lower()):
                    ordered.append(t)
                    remaining.remove(t)
                    break

        if not ordered:
            # Pick lowest-energy track as default warm-up anchor
            if arc in (EnergyArc.RAMP_UP, EnergyArc.MOUNTAIN):
                anchor = min(remaining, key=lambda t: t.energy)
            elif arc == EnergyArc.RAMP_DOWN:
                anchor = max(remaining, key=lambda t: t.energy)
            else:
                anchor = remaining[0]
            ordered.append(anchor)
            remaining.remove(anchor)

        cumulative_cost = 0.0
        results: List[dict] = []

        # ── Greedy loop ───────────────────────────────────────────────────────
        while remaining:
            current = ordered[-1]
            position = len(ordered) / n  # fractional position in set [0, 1]

            best_track: Optional[TrackNode] = None
            best_score: float = float("inf")
            best_ts: Optional[TransitionScore] = None

            for candidate in remaining:
                ts = transition_cost(
                    current, candidate,
                    arc_position=position,
                    arc=arc,
                    w_harmonic=w_harmonic,
                    w_bpm=w_bpm,
                    w_energy=w_energy,
                )
                score = ts.total_cost

                # Blend Markov plausibility if model is loaded
                if self.markov and markov_weight > 0:
                    markov_plausibility = self.markov.score(current, candidate)
                    # Markov score: higher = better, so flip to cost
                    markov_cost = 1.0 - markov_plausibility
                    score = (1 - markov_weight) * score + markov_weight * markov_cost

                if score < best_score:
                    best_score = score
                    best_track = candidate
                    best_ts = ts

            ordered.append(best_track)
            remaining.remove(best_track)
            cumulative_cost += best_score

            # Record this step's transition
            results.append({
                **current.to_dict(),
                "position": len(ordered) - 1,
                "arc_target_energy": round(_arc_target_energy(position, arc), 3),
                "transition_to_next": {
                    "next_track": best_track.display,
                    "next_camelot": best_track.camelot,
                    "modulation_type": best_ts.modulation.get("type", ""),
                    "modulation_impact": best_ts.modulation.get("impact", ""),
                    "recommendation": best_ts.modulation.get("recommendation", ""),
                    "safe_to_blend": best_ts.modulation.get("safe_to_blend", False),
                    "bpm_delta": best_ts.bpm_delta,
                    "harmonic_cost": best_ts.harmonic_cost,
                    "bpm_cost": best_ts.bpm_cost,
                    "energy_cost": best_ts.energy_cost,
                    "total_cost": best_ts.total_cost,
                },
                "cumulative_cost": round(cumulative_cost, 3),
            })

        # Append the last track with no transition info
        results.append({
            **ordered[-1].to_dict(),
            "position": n - 1,
            "arc_target_energy": round(_arc_target_energy(1.0, arc), 3),
            "transition_to_next": None,
            "cumulative_cost": round(cumulative_cost, 3),
        })

        log.info(
            "Setlist optimised: %d tracks, arc=%s, total_cost=%.3f",
            n, arc.value, cumulative_cost
        )
        return results

    def export_csv(self, results: List[dict], path: str | Path) -> None:
        """Write an optimized setlist to CSV (human-readable)."""
        if not results:
            return

        fieldnames = [
            "position", "artist", "name", "camelot", "bpm", "energy_level",
            "energy", "arc_target_energy", "modulation_type", "safe_to_blend",
            "bpm_delta", "total_cost", "recommendation",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                t = r.get("transition_to_next") or {}
                writer.writerow({
                    "position": r["position"] + 1,
                    "artist": r.get("artist", ""),
                    "name": r.get("name", ""),
                    "camelot": r.get("camelot", ""),
                    "bpm": r.get("bpm", ""),
                    "energy_level": r.get("energy_level", ""),
                    "energy": r.get("energy", ""),
                    "arc_target_energy": r.get("arc_target_energy", ""),
                    "modulation_type": t.get("modulation_type", ""),
                    "safe_to_blend": t.get("safe_to_blend", ""),
                    "bpm_delta": t.get("bpm_delta", ""),
                    "total_cost": t.get("total_cost", ""),
                    "recommendation": t.get("recommendation", ""),
                })

        log.info("Setlist exported to %s", path)

    def summary(self, results: List[dict]) -> dict:
        """Return a high-level summary of the optimised setlist."""
        if not results:
            return {}

        mods = [
            r["transition_to_next"]["modulation_type"]
            for r in results if r.get("transition_to_next")
        ]
        mod_counts: Dict[str, int] = defaultdict(int)
        for m in mods:
            mod_counts[m] += 1

        bpm_values = [r["bpm"] for r in results if r.get("bpm")]
        energy_values = [r["energy"] for r in results if r.get("energy")]

        return {
            "total_tracks": len(results),
            "total_cost": results[-1].get("cumulative_cost", 0.0),
            "bpm_range": [round(min(bpm_values), 1), round(max(bpm_values), 1)] if bpm_values else [],
            "energy_range": [round(min(energy_values), 3), round(max(energy_values), 3)] if energy_values else [],
            "modulation_breakdown": dict(mod_counts),
            "safe_blend_count": sum(
                1 for r in results
                if r.get("transition_to_next") and r["transition_to_next"].get("safe_to_blend")
            ),
            "hard_cut_count": sum(
                1 for r in results
                if r.get("transition_to_next") and not r["transition_to_next"].get("safe_to_blend")
            ),
        }


# ── Spectral flux energy metric ───────────────────────────────────────────────

def compute_spectral_flux_energy(audio: np.ndarray, sr: int) -> float:
    """
    Compute a Spectral Flux-based energy metric [0.0, 1.0].

    Spectral Flux measures the rate of change in the power spectrum over time.
    It captures the "punchiness" and danceability of a track better than raw
    RMS amplitude — critical for dense electronic music where multiple drum
    transients per bar spike the flux score.

    Formula:
      flux(t) = sum_over_bins( max(|X(t)| - |X(t-1)|, 0)² )
    Normalised to [0, 1] via max-norm across the signal.

    Args:
        audio: Mono audio array (float32)
        sr:    Sample rate

    Returns:
        Normalised spectral flux energy score [0.0, 1.0]
    """
    try:
        import librosa

        # STFT for the spectrogram
        hop_length = 512
        S = np.abs(librosa.stft(audio, hop_length=hop_length))  # shape: (bins, frames)

        # Positive flux only (onset detection focus)
        diff = np.diff(S, axis=1)
        flux = np.sum(np.maximum(diff, 0) ** 2, axis=0)  # shape: (frames,)

        if flux.max() > 0:
            flux_norm = flux / flux.max()
        else:
            return 0.0

        # Mean flux as the track's "punchiness" score
        score = float(np.mean(flux_norm))
        return min(1.0, score * 3.0)   # scale up — mean is typically 0.15–0.4

    except Exception as e:
        log.warning("spectral_flux_energy: %s", e)
        return 0.5


def compute_rms_energy(audio: np.ndarray, sr: int) -> float:
    """
    Compute RMS-based energy from the spectrogram (more accurate than time-domain RMS).
    Consistent with how Spotify approximated energy before the API deprecation.

    Returns value in [0.0, 1.0].
    """
    try:
        import librosa
        S = np.abs(librosa.stft(audio))
        rms = librosa.feature.rms(S=S)[0]
        # Normalise: typical RMS in dance music peaks around 0.25–0.45
        score = float(np.mean(rms)) / 0.45
        return min(1.0, max(0.0, score))
    except Exception as e:
        log.warning("rms_energy: %s", e)
        return 0.5


def compute_energy_features(audio: np.ndarray, sr: int) -> dict:
    """
    Compute both RMS and Spectral Flux energy and return a blended score.

    The blended score mirrors Spotify's Energy metric (0.0–1.0) using
    local DSP, replacing the deprecated /audio-features endpoint.

    Returns:
        {
          "energy": float [0,1],    # blended score (70% flux, 30% RMS)
          "spectral_flux": float,
          "rms": float,
        }
    """
    rms = compute_rms_energy(audio, sr)
    flux = compute_spectral_flux_energy(audio, sr)
    blended = 0.70 * flux + 0.30 * rms
    return {
        "energy": round(min(1.0, blended), 4),
        "spectral_flux": round(flux, 4),
        "rms": round(rms, 4),
    }
