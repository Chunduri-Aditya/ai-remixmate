# scripts/download_from_csv.py

import csv
import os
import subprocess

AUDIO_FOLDER = "../audio_input"

def download_song(url_or_search, title=None):
    safe_title = title.replace("/", "_").replace("\\", "_").strip()
    output_path = os.path.join(AUDIO_FOLDER, f"{safe_title}.wav")

    if os.path.exists(output_path):
        print(f"⏩ Skipping (already downloaded): {safe_title}")
        return

    print(f"🎧 Downloading: {safe_title}")
    command = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--output", f"{AUDIO_FOLDER}/%(title)s.%(ext)s",
        url_or_search
    ]
    subprocess.run(command)

def process_csv(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header

        for row in reader:
            if not row or len(row) < 1 or not row[0].strip():
                continue

            title = row[0].strip()
            url = None

            if len(row) > 1 and "http" in row[1]:
                url = row[1].strip()

            if url:
                download_song(url, title)
            else:
                search_query = f"ytsearch:{title}"
                download_song(search_query, title)

if __name__ == "__main__":
    csv_path = input("📄 Enter path to playlist CSV: ").strip()
    if not os.path.exists(AUDIO_FOLDER):
        os.makedirs(AUDIO_FOLDER)

    if os.path.exists(csv_path):
        process_csv(csv_path)
    else:
        print("❌ CSV file not found. Please check the path.")

        