import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_FOLDER = PROJECT_ROOT / "audio_input"


def download_song(url_or_search, title=None, audio_folder=AUDIO_FOLDER):
    audio_folder = Path(audio_folder)
    audio_folder.mkdir(parents=True, exist_ok=True)

    safe_title = (title or url_or_search).replace("/", "_").replace("\\", "_").strip()
    output_path = audio_folder / f"{safe_title}.wav"

    if output_path.exists():
        print(f"⏩ Skipping (already downloaded): {safe_title}")
        return

    print(f"🎧 Downloading: {safe_title}")
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-x",
        "--audio-format",
        "wav",
        "--output",
        str(audio_folder / "%(title)s.%(ext)s"),
        url_or_search,
    ]
    subprocess.run(command, check=False)


def process_csv(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)

        for row in reader:
            if not row or len(row) < 1 or not row[0].strip():
                continue

            title = row[0].strip()
            url = row[1].strip() if len(row) > 1 and "http" in row[1] else None

            if url:
                download_song(url, title)
            else:
                download_song(f"ytsearch:{title}", title)


if __name__ == "__main__":
    csv_path = Path(input("📄 Enter path to playlist CSV: ").strip()).expanduser()

    if csv_path.exists():
        process_csv(csv_path)
    else:
        print("❌ CSV file not found. Please check the path.")
