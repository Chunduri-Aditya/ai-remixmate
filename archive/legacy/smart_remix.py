#!/usr/bin/env python3
# scripts/smart_remix.py
"""
Smart Remix:
- Finds top matches for a base song using rank_matches (tempo+audio+lyrics).
- Lets you auto-pick or interactively choose a candidate.
- Remixes base VOCALS with match INSTRUMENTAL (other.wav), tempo-aware.

Usage:
  python scripts/smart_remix.py --base "YourBaseSong"
  python scripts/smart_remix.py --base "YourBaseSong" --auto --fade 0.5
"""

from __future__ import annotations
import argparse
from difflib import get_close_matches
from typing import List, Optional

from scripts.core.paths import SEPARATED, other_path
from scripts.core.features import load_audio, tempo_feature
from scripts.core.remix import remix_songs
from scripts.analyze_similarity import rank_matches


def list_available_songs() -> List[str]:
    if not SEPARATED.exists():
        return []
    return sorted([d.name for d in SEPARATED.iterdir() if d.is_dir()])


def resolve_song_name(input_name: str) -> Optional[str]:
    available = list_available_songs()
    if input_name in available:
        return input_name
    matches = get_close_matches(input_name, available, n=5, cutoff=0.4)
    if not matches:
        print("❌ No similar song names found.")
        return None
    print("\n🧠 Did you mean one of these?")
    for i, m in enumerate(matches, 1):
        print(f"{i}. {m}")
    try:
        pick = int(input("👉 Enter number (or 0 to cancel): ").strip())
        if pick == 0:
            return None
        return matches[pick - 1]
    except Exception:
        print("❌ Invalid choice.")
        return None


def estimate_tempo(song_name: str) -> Optional[float]:
    p = other_path(song_name)
    if not p.exists():
        return None
    try:
        y, sr = load_audio(p)
        return float(tempo_feature(y, sr))
    except Exception:
        return None


def ask_user_to_choose(candidates):
    print("\n🎯 Top Matches:")
    for i, (name, score, parts) in enumerate(candidates, 1):
        print(f"{i}. {name:<30} | score={score:.3f}  (tempo={parts['tempo']:.3f}, audio={parts['audio']:.3f}, lyrics={parts['lyrics']:.3f})")
    choice = input(f"\nPick a match [1-{len(candidates)}] or press Enter for top match: ").strip()
    if not choice:
        return candidates[0][0]
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx][0]
    except Exception:
        pass
    print("⚠️ Invalid input, using top match.")
    return candidates[0][0]


def main():
    ap = argparse.ArgumentParser(description="🎛 Smart Remix Generator")
    ap.add_argument("--base", required=True, help="Base song (folder under separated/htdemucs)")
    ap.add_argument("--topk", type=int, default=5, help="How many candidates to consider")
    ap.add_argument("--fade", type=float, default=0.5, help="Fade in/out seconds for the final mix")
    ap.add_argument("--auto", action="store_true", help="Auto-pick the top match without prompting")
    args = ap.parse_args()

    # Resolve base song (supports fuzzy input)
    base_song = resolve_song_name(args.base)
    if not base_song:
        return

    print(f"\n🎵 Starting smart remix for: {base_song}")
    candidates = rank_matches(base_song, top_k=args.topk)
    if not candidates:
        print("❌ No similar songs found (check stems/lyrics/features).")
        return

    match_song = candidates[0][0] if args.auto else ask_user_to_choose(candidates)

    # Estimate tempos for gentle alignment
    base_tempo = estimate_tempo(base_song)
    match_tempo = estimate_tempo(match_song)

    print(f"\n🔀 Remixing {base_song} (VOCALS) with {match_song} (INSTRUMENTAL)…")
    try:
        out = remix_songs(
            base_song=base_song,
            match_song=match_song,
            base_tempo=base_tempo,
            match_tempo=match_tempo,
            fade_sec=args.fade,
        )
        print(f"✅ Remix saved to: {out}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Remix failed: {e}")


if __name__ == "__main__":
    main()