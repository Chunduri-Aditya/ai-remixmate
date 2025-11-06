# scripts/smart_remix.py
import argparse
from scripts.analyze_similarity import find_best_match
from remix import remix_songs  # assumes remix.py is in the same folder

def ask_user_to_choose(matches):
    print("\n🎯 Top Matches:")
    for i, (song, score) in enumerate(matches):
        print(f"{i+1}. {song:<40} | Similarity: {score:.4f}")
    choice = input("\nPick a match [1-5] or press Enter to use top match: ")
    try:
        idx = int(choice.strip()) - 1
        if 0 <= idx < len(matches):
            return matches[idx][0]
    except:
        pass
    return matches[0][0]  # default to top match

def main(base_song):
    print(f"\n🎵 Starting smart remix for: {base_song}")
    matches = find_best_match(base_song, top_k=5)

    if not matches:
        print("❌ No similar songs found.")
        return

    match_song = ask_user_to_choose(matches)
    print(f"\n🔀 Remixing {base_song} with {match_song}...\n")
    remix_songs(base_song, match_song, auto_mode=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🎛 Smart Remix Generator")
    parser.add_argument("--base", required=True, help="Base song name (folder name)")
    args = parser.parse_args()
    main(args.base)