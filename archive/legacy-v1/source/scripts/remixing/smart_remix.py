import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.analyze_similarity import find_best_match
from scripts.remixing.remix_from_match import generate_smooth_remix, resolve_song_name


def ask_user_to_choose(matches):
    print("\n🎯 Top Matches:")
    for i, (song, score) in enumerate(matches):
        print(f"{i+1}. {song:<40} | Similarity: {score:.4f}")
    choice = input("\nPick a match [1-5] or press Enter to use top match: ")
    try:
        idx = int(choice.strip()) - 1
        if 0 <= idx < len(matches):
            return matches[idx][0]
    except Exception:
        pass
    return matches[0][0]


def main(base_song):
    base_song = resolve_song_name(base_song)
    if not base_song:
        return 1

    print(f"\n🎵 Starting smart remix for: {base_song}")
    matches = find_best_match(base_song, top_k=5)

    if not matches:
        print("❌ No similar songs found.")
        return 1

    match_song = resolve_song_name(ask_user_to_choose(matches))
    if not match_song:
        return 1

    print(f"\n🔀 Remixing {base_song} with {match_song}...\n")
    return 0 if generate_smooth_remix(base_song, match_song) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart remix generator")
    parser.add_argument("--base", required=True, help="Base song name (folder name under separated/htdemucs)")
    args = parser.parse_args()
    sys.exit(main(args.base))
