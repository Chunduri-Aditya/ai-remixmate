#!/usr/bin/env python3
"""
analyze_similarity.py

Hybrid similarity ranking for remix candidates:
1) Tempo shortlist (fast prefilter)
2) Audio similarity (MFCC + Chroma, cosine)
3) Lyrics semantic similarity (SBERT) — optional, falls back gracefully

Weights renormalize automatically when any signal (tempo/audio/lyrics) is missing.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Project core utilities (from the refactor)
from scripts.core.paths import SEPARATED, other_path, lyrics_path
from scripts.core.features import (
    load_audio, tempo_feature, chroma_feature, mfcc_feature, cosine_similarity
)

# --- Config ---
MAX_CANDIDATES = 10         # shortlist size after tempo filtering
TOP_K_DEFAULT = 5           # final top-K to return
N_MFCC = 20                 # MFCC count for audio vector
WEIGHTS = dict(tempo=0.2, audio=0.5, lyrics=0.3)

# Lazy import for sentence-transformers (lyrics embedding is optional)
_SBERT_MODEL = None
def _get_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SBERT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _SBERT_MODEL = None
    return _SBERT_MODEL


def _read_text(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    try:
        return p.read_text(errors="ignore").strip()
    except Exception:
        return None


def _lyrics_embedding(song: str) -> Optional[np.ndarray]:
    """Return SBERT embedding for lyrics.txt if available and model present."""
    model = _get_sbert()
    if model is None:
        return None
    txt = _read_text(lyrics_path(song))
    if not txt:
        return None
    vec = model.encode([txt])[0]
    return np.asarray(vec, dtype=np.float32)


def _audio_vector(path_other: Path, n_mfcc: int = N_MFCC) -> Optional[np.ndarray]:
    """Concatenate mean Chroma (12) + MFCC (n_mfcc) into a single vector."""
    if not path_other.exists():
        return None
    try:
        y, sr = load_audio(path_other)  # mono, TARGET_SR
        chroma = chroma_feature(y, sr)          # (12,)
        mfcc = mfcc_feature(y, sr, n_mfcc)      # (n_mfcc,)
        v = np.concatenate([chroma, mfcc]).astype(np.float32)
        # simple normalization to unit norm for robust cosine
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    except Exception:
        return None


def _tempo_score(base_tempo: Optional[float], cand_tempo: Optional[float], max_delta: float = 40.0) -> Optional[float]:
    """Return [0,1] score; None if any tempo missing."""
    if base_tempo is None or cand_tempo is None:
        return None
    delta = abs(base_tempo - cand_tempo)
    raw = 1.0 - (delta / max_delta)
    return float(np.clip(raw, 0.0, 1.0))


def _audio_score(base_vec: Optional[np.ndarray], cand_vec: Optional[np.ndarray]) -> Optional[float]:
    """Cosine similarity in [-1,1] mapped to [0,1]; None if missing."""
    if base_vec is None or cand_vec is None:
        return None
    sim = cosine_similarity(base_vec, cand_vec)   # [-1, 1]
    return float((sim + 1.0) / 2.0)              # map to [0,1]


def _lyrics_score(base_emb: Optional[np.ndarray], cand_emb: Optional[np.ndarray]) -> Optional[float]:
    """Cosine similarity in [-1,1] mapped to [0,1]; None if missing or model not available."""
    if base_emb is None or cand_emb is None:
        return None
    # manual cosine to avoid importing torch/util; consistent with audio
    denom = (np.linalg.norm(base_emb) * np.linalg.norm(cand_emb))
    if denom == 0:
        return None
    sim = float(np.dot(base_emb, cand_emb) / denom)  # [-1,1]
    return (sim + 1.0) / 2.0                         # [0,1]


def _list_songs() -> List[str]:
    """All stemmed song folder names under separated/htdemucs."""
    if not SEPARATED.exists():
        return []
    return sorted([d.name for d in SEPARATED.iterdir() if d.is_dir()])


def _tempo_shortlist(base: str, base_tempo: Optional[float], max_candidates: int = MAX_CANDIDATES) -> List[str]:
    """Rank candidates by absolute tempo difference; return top-N names."""
    names = [s for s in _list_songs() if s != base]
    scored: List[Tuple[str, float]] = []
    for s in names:
        co = other_path(s)
        if not co.exists():
            continue
        try:
            yc, src = load_audio(co)
            t = tempo_feature(yc, src)
            diff = abs(t - base_tempo) if base_tempo is not None else 9999.0
            scored.append((s, diff))
        except Exception:
            continue
    scored.sort(key=lambda x: x[1])
    return [s for s, _ in scored[:max_candidates]]


def rank_matches(base: str, top_k: int = TOP_K_DEFAULT) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    Return: list of (song_name, final_score, {'tempo':..,'audio':..,'lyrics':..}) sorted desc by score.
    Scores are in [0,1]. If *all* signals missing for a candidate, it receives -1 and is filtered out.
    """
    # Prepare base signals
    base_other = other_path(base)
    base_vec = _audio_vector(base_other)
    base_tempo = None
    if base_other.exists():
        try:
            yb, srb = load_audio(base_other)
            base_tempo = tempo_feature(yb, srb)
        except Exception:
            base_tempo = None
    base_emb = _lyrics_embedding(base)

    # Candidate shortlist (tempo-first)
    candidates = _tempo_shortlist(base, base_tempo, MAX_CANDIDATES)
    results: List[Tuple[str, float, Dict[str, float]]] = []

    for cand in candidates:
        co = other_path(cand)
        cand_vec = _audio_vector(co)
        cand_tempo = None
        if co.exists():
            try:
                yc, src = load_audio(co)
                cand_tempo = tempo_feature(yc, src)
            except Exception:
                cand_tempo = None
        cand_emb = _lyrics_embedding(cand)

        # individual scores (each in [0,1] or None)
        t_s = _tempo_score(base_tempo, cand_tempo)
        a_s = _audio_score(base_vec, cand_vec)
        l_s = _lyrics_score(base_emb, cand_emb)

        # weight renormalization for available signals
        parts = {
            "tempo": 0.0 if t_s is None else float(t_s),
            "audio": 0.0 if a_s is None else float(a_s),
            "lyrics": 0.0 if l_s is None else float(l_s),
        }
        masks = np.array([t_s is not None, a_s is not None, l_s is not None], dtype=bool)
        w = np.array([WEIGHTS["tempo"], WEIGHTS["audio"], WEIGHTS["lyrics"]], dtype=np.float32)
        if masks.sum() == 0:
            final = -1.0  # no usable signal — drop
        else:
            w = w * masks
            w = w / w.sum()
            final = float(w[0]*parts["tempo"] + w[1]*parts["audio"] + w[2]*parts["lyrics"])

        if final >= 0:
            results.append((cand, final, parts))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base song folder name under separated/htdemucs/")
    ap.add_argument("--topk", type=int, default=TOP_K_DEFAULT, help="How many matches to return")
    ap.add_argument("--show-tempo", action="store_true", help="Print shortlist with tempo diffs")
    args = ap.parse_args()

    # Preview tempo shortlist (optional)
    base_other = other_path(args.base)
    base_tempo = None
    if base_other.exists():
        try:
            yb, srb = load_audio(base_other)
            base_tempo = tempo_feature(yb, srb)
        except Exception:
            pass

    shortlist = _tempo_shortlist(args.base, base_tempo, MAX_CANDIDATES)
    if args.show_tempo:
        print(f"\n🔎 Top {MAX_CANDIDATES} tempo-close songs to {args.base}:\n")
        for s in shortlist:
            try:
                yc, src = load_audio(other_path(s))
                t = tempo_feature(yc, src)
                d = abs(t - base_tempo) if base_tempo is not None else float('nan')
                print(f" - {s:<35} | tempo diff: {d:.2f}")
            except Exception:
                print(f" - {s:<35} | tempo diff: n/a")

    ranked = rank_matches(args.base, top_k=args.topk)
    print(f"\n🎯 Top matches for: {args.base}\n")
    for i, (name, score, parts) in enumerate(ranked, 1):
        print(f"{i:>2}. {name:<30} | score={score:.3f}  (tempo={parts['tempo']:.3f}, audio={parts['audio']:.3f}, lyrics={parts['lyrics']:.3f})")


if __name__ == "__main__":
    main()