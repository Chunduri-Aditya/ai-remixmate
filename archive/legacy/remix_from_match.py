#!/usr/bin/env python3
# remix_from_match.py — Robust, tempo-aware remix generator

from __future__ import annotations
import argparse
from difflib import get_close_matches
from pathlib import Path
from typing import List, Optional

from scripts.core.paths import SEPARATED, other_path
from scripts.core.features import load_audio, tempo_feature
from scripts.core.remix import remix_songs
from scripts.analyze_similarity import rank_matches


# --- Helpers ---------------------------------------------------------------

def list_available_songs() -> List[str]:
    if not SEPARATED.exists():
        return []
    return sorted([d.name for d in SEPARATED.iterdir() if d.is_dir()])


def resolve_song_name(input_name: str) -> Optional[str]:
    """Resolve a possibly fuzzy song name to an existing separated folder."""
    available = list_available_songs()
    if input_name in available:
        return input_name

    matches = get_close_matches(input_name, available, n=5, cutoff=0.4)
    if not matches:
        print("❌ No similar song names found.")
        return None

    print("\n🧠 Did you mean one of these songs?")
    for idx, name in enumerate(matches, 1):
        print(f"{idx}. {name}")
    try:
        choice = int(input("👉 Enter the number of the correct song (or 0 to cancel): ").strip())
        if choice == 0:
            return None
        return matches[choice - 1]
    except Exception:
        print("❌ Invalid choice.")
        return None


def estimate_tempo_from_other(song_name: str) -> Optional[float]:
    """Estimate BPM from the 'other.wav' stem if present."""
    p = other_path(song_name)
    if not p.exists():
        return None
    try:
        y, sr = load_audio(p)
        return float(tempo_feature(y, sr))
    except Exception:
        return None


# --- Main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Remix base vocals with another song's instrumental (tempo-aware).")
    ap.add_argument("--base", help="Base song folder name under separated/htdemucs (will use its VOCALS)")
    ap.add_argument("--topk", type=int, default=5, help="How many match candidates to show")
    ap.add_argument("--fade", type=float, default=0.5, help="Fade in/out seconds for the final mix")
    ap.add_argument("--auto", action="store_true", help="Auto-pick the top match without prompting")
    args = ap.parse_args()

    # Resolve base song
    base_input = args.base or input("🎧 Enter base song name (folder under separated/htdemucs): ").strip()
    base_song = resolve_song_name(base_input)
    if not base_song:
        return

    # Rank candidates
    print(f"🔍 Finding best matches for: {base_song}")
    candidates = rank_matches(base_song, top_k=args.topk)
    if not candidates:
        print("❌ No matches found (ensure stems and features are available).")
        return

    print("\n🎯 Top Matches:")
    for i, (name, score, parts) in enumerate(candidates, 1):
        print(f"{i}. {name:<30} | score={score:.4f}  (tempo={parts['tempo']:.3f}, audio={parts['audio']:.3f}, lyrics={parts['lyrics']:.3f})")

    # Choose match
    if args.auto:
        match_name = candidates[0][0]
        print(f"\n⚙️ Auto-selected top match: {match_name}")
    else:
        try:
            choice = int(input(f"\n👉 Enter the number of your match choice (1–{len(candidates)}): ").strip())
            if not 1 <= choice <= len(candidates):
                raise ValueError
            match_name = candidates[choice - 1][0]
        except Exception:
            print("❌ Invalid choice.")
            return

    # Compute tempos for basic alignment
    base_tempo = estimate_tempo_from_other(base_song)
    match_tempo = estimate_tempo_from_other(match_name)

    # Remix (vocals from base + instrumental from match)
    print(f"\n🎶 Remixing vocals from '{base_song}' with instruments from '{match_name}' …")
    try:
        out_path: Path = remix_songs(
            base_song=base_song,
            match_song=match_name,
            base_tempo=base_tempo,
            match_tempo=match_tempo,
            fade_sec=args.fade,
        )
        print(f"✅ Remix saved to: {out_path}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Remix failed: {e}")


if __name__ == "__main__":
    main()