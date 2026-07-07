"""
scripts/core/spectral_matching.py — learned spectrogram matching for transitions.

Picks Song B's entry point by comparing mel-band spectral profiles: A's outro
window (the transition_seconds before its exit cue) is profiled once, then every
phrase-boundary candidate window in B is profiled and scored against it. The
score is a weighted per-band similarity, and the band weights LEARN from user
feedback (👍/👎 on finished mixes via POST /dj-remix/feedback):

  liked   → bands that matched well in that mix gain weight
  disliked → bands that matched well but produced a bad mix lose weight

All updates are deterministic numpy math; state persists as JSON in data/:
  spectral_match_model.json    — per-band weights + update count
  spectral_match_history.json  — per-session band similarities awaiting feedback

This operates on the TransitionPlan produced by either mixing engine (classic
or current) — it mutates plan.entry_bar_b / plan.entry_time_b before rendering
and never touches the engines themselves (the classic package stays frozen).
Gated by config `dj.spectral_matching`; failures degrade to the planner's
original entry point, never break a render.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

N_BANDS = 24                 # mel bands in a transition profile
_PHRASE_BARS = 8             # candidate entry points sit on 8-bar phrase boundaries
_MAX_CANDIDATES = 8          # cap on B-entry candidates per render
_HISTORY_CAP = 200           # renders kept awaiting feedback
_WEIGHT_MIN, _WEIGHT_MAX = 0.1, 3.0

_lock = threading.Lock()


def _data_dir() -> Path:
    from scripts.core.paths import DATA_DIR
    return Path(DATA_DIR)


def model_path() -> Path:
    return _data_dir() / "spectral_match_model.json"


def history_path() -> Path:
    return _data_dir() / "spectral_match_history.json"


# ---------------------------------------------------------------------------
# Profiles and similarity
# ---------------------------------------------------------------------------

def band_profile(audio: np.ndarray, sr: int, n_bands: int = N_BANDS) -> np.ndarray:
    """Log-mel band energy distribution of an audio window.

    Returns a length-n_bands vector normalized to sum 1 (a distribution over
    bands), so profiles of windows with different lengths/levels compare fairly.
    """
    import librosa

    if audio.size == 0:
        return np.full(n_bands, 1.0 / n_bands, dtype=np.float64)
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=sr, n_mels=n_bands, fmax=sr / 2
    )
    band = np.log1p(mel).mean(axis=1).astype(np.float64)
    total = float(band.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(n_bands, 1.0 / n_bands, dtype=np.float64)
    return band / total


def band_similarity(profile_a: np.ndarray, profile_b: np.ndarray) -> np.ndarray:
    """Per-band similarity in [0, 1] (1 = identical share of energy).

    Canberra-style per band: 1 − |a−b| / (a+b). Bounded regardless of scale.
    """
    a = np.asarray(profile_a, dtype=np.float64)
    b = np.asarray(profile_b, dtype=np.float64)
    denom = a + b
    sims = np.ones_like(a)
    nz = denom > 1e-12
    sims[nz] = 1.0 - np.abs(a[nz] - b[nz]) / denom[nz]
    return np.clip(sims, 0.0, 1.0)


_HOP = 512  # trajectory frame hop — one profile every ~12 ms at 44.1 kHz


def mel_trajectory(audio: np.ndarray, sr: int, n_bands: int = N_BANDS) -> np.ndarray:
    """Time-varying band profiles of a whole signal: (n_bands × frames).

    Each frame column is normalized to a distribution over bands, so the
    trajectory captures how the spectral *shape* evolves through the song,
    independent of level. Computed once per song and sliced per window.
    """
    import librosa

    if audio.size == 0:
        return np.full((n_bands, 1), 1.0 / n_bands, dtype=np.float64)
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=sr, n_mels=n_bands, fmax=sr / 2, hop_length=_HOP
    ).astype(np.float64)
    total = mel.sum(axis=0, keepdims=True)
    total[total <= 1e-12] = 1e-12
    return mel / total


def trajectory_similarity(traj_a: np.ndarray, traj_b: np.ndarray) -> np.ndarray:
    """Per-band similarity of two aligned trajectories → vector of N_BANDS.

    Frames are compared pairwise (both windows span the same musical time —
    the engines tempo-match them), truncated to the shorter trajectory, and
    the per-frame Canberra similarities are averaged per band. The result
    feeds the same learned weights / feedback loop as static profiles.
    """
    frames = min(traj_a.shape[1], traj_b.shape[1])
    if frames == 0:
        return np.ones(traj_a.shape[0])
    a = traj_a[:, :frames]
    b = traj_b[:, :frames]
    denom = a + b
    sims = np.ones_like(a)
    nz = denom > 1e-12
    sims[nz] = 1.0 - np.abs(a[nz] - b[nz]) / denom[nz]
    return np.clip(sims.mean(axis=1), 0.0, 1.0)


def match_score(sims: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean of per-band similarities → scalar in [0, 1]."""
    w = np.asarray(weights, dtype=np.float64)
    s = np.asarray(sims, dtype=np.float64)
    total = float(w.sum())
    if total <= 0.0:
        return float(s.mean())
    return float(np.dot(w, s) / total)


# ---------------------------------------------------------------------------
# Learned model
# ---------------------------------------------------------------------------

@dataclass
class SpectralMatchModel:
    """Per-band weights, learned online from mix feedback."""
    weights: np.ndarray = field(default_factory=lambda: np.ones(N_BANDS, dtype=np.float64))
    n_updates: int = 0

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SpectralMatchModel":
        p = path or model_path()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            w = np.asarray(raw["weights"], dtype=np.float64)
            if w.shape != (N_BANDS,) or not np.all(np.isfinite(w)):
                raise ValueError("bad weights shape/values")
            return cls(weights=w, n_updates=int(raw.get("n_updates", 0)))
        except FileNotFoundError:
            return cls()
        except Exception as exc:  # corrupted file → fresh model, don't break renders
            log.warning(f"spectral match model unreadable ({exc}) — starting fresh")
            return cls()

    def save(self, path: Optional[Path] = None) -> None:
        p = path or model_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "version": 1,
            "n_bands": N_BANDS,
            "weights": [round(float(w), 6) for w in self.weights],
            "n_updates": self.n_updates,
        }, indent=2), encoding="utf-8")

    def update(self, sims: np.ndarray, liked: bool, lr: float = 0.2) -> None:
        """One online step from a single mix's band similarities.

        Bands whose similarity exceeded this mix's mean move with the verdict:
        up when liked, down when disliked. Weights are clipped and mean-
        renormalized to 1 so the score scale stays stable over time.
        """
        s = np.asarray(sims, dtype=np.float64)
        if s.shape != self.weights.shape or not np.all(np.isfinite(s)):
            return
        delta = float(lr) * (s - s.mean())
        self.weights = self.weights + (delta if liked else -delta)
        self.weights = np.clip(self.weights, _WEIGHT_MIN, _WEIGHT_MAX)
        mean = float(self.weights.mean())
        if mean > 0:
            self.weights = self.weights / mean
        self.n_updates += 1


# ---------------------------------------------------------------------------
# Candidate search
# ---------------------------------------------------------------------------

def _drop_alignment_factor(
    entry_time: float,
    drop_time: Optional[float],
    transition_seconds: float,
    bar_seconds: float,
) -> Tuple[float, Optional[float]]:
    """Score multiplier for how B's drop lands relative to the handover.

    DJ practice: the incoming track's drop should hit shortly AFTER the
    crossfade completes — the payoff moment belongs to B alone. A drop
    inside the overlap fights the outgoing track, so it's penalized.

      drop within 16 bars after handover → up to +15% (closer = bigger)
      drop inside the overlap            → −10%
      no drop / far away                 → neutral

    Returns (factor, delta_seconds) where delta = drop − handover.
    """
    if drop_time is None or bar_seconds <= 0:
        return 1.0, None
    handover = entry_time + transition_seconds
    delta = float(drop_time) - handover
    if -transition_seconds < delta < 0.0:
        return 0.9, round(delta, 2)
    window = 16.0 * bar_seconds
    if 0.0 <= delta <= window:
        return 1.0 + 0.15 * (1.0 - delta / window), round(delta, 2)
    return 1.0, round(delta, 2)


def choose_entry_bar(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sr: int,
    exit_time_a: float,
    transition_seconds: float,
    bars_b: List[Tuple[float, float]],
    weights: Optional[np.ndarray] = None,
    drop_time_b: Optional[float] = None,
    bpm_b: float = 0.0,
) -> Dict:
    """Score B's phrase-boundary entry candidates against A's outro spectrum.

    Dynamic per-song matching: A's outro and every candidate window in B are
    compared as mel-band *trajectories* (frame-by-frame spectral shape over
    the whole window), so a candidate must track how A's outro evolves — a
    static timbre average can't fake it. B's trajectory is computed once for
    the full song and sliced per candidate.

    Returns {"entry_bar", "entry_time", "score", "sims", "candidates"} for the
    best candidate. Falls back to bar 0 when B is too short for alternatives.
    """
    if weights is None:
        weights = SpectralMatchModel.load().weights

    win = max(1.0, float(transition_seconds))
    a_start = max(0.0, float(exit_time_a) - win)
    outro = audio_a[int(a_start * sr): int(float(exit_time_a) * sr)]
    traj_a = mel_trajectory(outro, sr)

    dur_b = len(audio_b) / float(sr)
    candidate_bars: List[int] = []
    for bar in range(0, len(bars_b), _PHRASE_BARS):
        start_t = bars_b[bar][0]
        if start_t + win > dur_b:
            break
        candidate_bars.append(bar)
        if len(candidate_bars) >= _MAX_CANDIDATES:
            break
    if not candidate_bars:
        candidate_bars = [0]

    traj_b_full = mel_trajectory(audio_b, sr)   # once per song, sliced below
    win_frames = max(1, int(win * sr / _HOP))
    bar_seconds = (4.0 * 60.0 / bpm_b) if bpm_b > 0 else 0.0

    best: Optional[Dict] = None
    scored: List[Dict] = []
    for bar in candidate_bars:
        t = float(bars_b[bar][0]) if bar < len(bars_b) else 0.0
        f0 = int(t * sr / _HOP)
        window_traj = traj_b_full[:, f0: f0 + win_frames]
        sims = trajectory_similarity(traj_a, window_traj)
        spectral = match_score(sims, weights)
        drop_factor, drop_delta = _drop_alignment_factor(t, drop_time_b, win, bar_seconds)
        score = spectral * drop_factor
        entry = {
            "entry_bar": bar,
            "entry_time": t,
            "score": round(score, 4),
            "spectral": round(spectral, 4),
            "drop_delta_s": drop_delta,
        }
        scored.append(entry)
        if best is None or score > best["score"]:
            best = {**entry, "sims": sims}

    assert best is not None  # candidate_bars is never empty
    best["candidates"] = scored
    return best


def apply_spectral_matching(plan, audio_a: np.ndarray, audio_b: np.ndarray,
                            struct_b, sr: int) -> Optional[Dict]:
    """Mutate plan's B-entry to the best spectral match; return match info.

    Engine-agnostic: only touches plan.entry_bar_b / plan.entry_time_b, which
    both renderers consume. Returns None (plan untouched) when disabled,
    when B has no bar grid, or on any internal failure.
    """
    try:
        from scripts.core.config import cfg
        if not bool(getattr(getattr(cfg, "dj", None), "spectral_matching", True)):
            return None
    except Exception:
        pass

    try:
        bars_b = list(getattr(struct_b, "bars", []) or [])
        if len(bars_b) < _PHRASE_BARS:
            return None
        best = choose_entry_bar(
            audio_a, audio_b, sr,
            exit_time_a=float(plan.exit_time_a),
            transition_seconds=float(plan.transition_seconds),
            bars_b=bars_b,
            drop_time_b=getattr(struct_b, "drop_position", None),
            bpm_b=float(getattr(struct_b, "bpm", 0.0) or 0.0),
        )
        original_bar = int(plan.entry_bar_b)
        plan.entry_bar_b = int(best["entry_bar"])
        plan.entry_time_b = float(best["entry_time"])
        info = {
            "score": best["score"],
            "spectral": best.get("spectral"),
            "drop_delta_s": best.get("drop_delta_s"),
            "entry_bar": int(best["entry_bar"]),
            "original_entry_bar": original_bar,
            "candidates": best["candidates"],
            "sims": [round(float(x), 4) for x in best["sims"]],
        }
        log.info(
            "Spectral match: B enters at bar %d (score %.3f, drop Δ %s s, "
            "%d candidates; planner said bar %d)",
            info["entry_bar"], info["score"], info["drop_delta_s"],
            len(info["candidates"]), original_bar,
        )
        return info
    except Exception as exc:  # never break a render over cue optimization
        log.warning(f"spectral matching skipped: {exc}")
        return None


# ---------------------------------------------------------------------------
# Feedback loop
# ---------------------------------------------------------------------------

def record_render(session_id: str, info: Dict) -> None:
    """Stash a render's band similarities so later feedback can train on them."""
    if not info or "sims" not in info:
        return
    with _lock:
        p = history_path()
        try:
            history = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            history = {}
        history[session_id] = {
            "sims": info["sims"],
            "entry_bar": info.get("entry_bar"),
            "score": info.get("score"),
            "ts": time.time(),
        }
        if len(history) > _HISTORY_CAP:  # drop oldest beyond cap
            for key, _ in sorted(history.items(), key=lambda kv: kv[1].get("ts", 0))[
                : len(history) - _HISTORY_CAP
            ]:
                history.pop(key, None)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(history, indent=2), encoding="utf-8")


def apply_feedback(session_id: str, liked: bool, lr: Optional[float] = None) -> Dict:
    """Train the band weights from one mix verdict. Returns a summary dict."""
    if lr is None:
        try:
            from scripts.core.config import cfg
            lr = float(getattr(getattr(cfg, "dj", None), "spectral_matching_lr", 0.2))
        except Exception:
            lr = 0.2

    with _lock:
        try:
            history = json.loads(history_path().read_text(encoding="utf-8"))
        except Exception:
            history = {}
        entry = history.get(session_id)
        if entry is None:
            return {"updated": False, "reason": f"no spectral record for session {session_id}"}

        model = SpectralMatchModel.load()
        model.update(np.asarray(entry["sims"], dtype=np.float64), liked=liked, lr=lr)
        model.save()
        # One verdict per render: consume the record
        history.pop(session_id, None)
        history_path().write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "updated": True,
        "liked": liked,
        "n_updates": model.n_updates,
        "weights_min": round(float(model.weights.min()), 4),
        "weights_max": round(float(model.weights.max()), 4),
    }
