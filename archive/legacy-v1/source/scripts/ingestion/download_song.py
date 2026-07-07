import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_INPUT_DIR = PROJECT_ROOT / "audio_input"


def _get_yt_dlp():
    try:
        import yt_dlp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing optional ingestion dependency. Install it with: "
            "python -m pip install -e '.[ingestion]'"
        ) from exc

    return yt_dlp


def download_song(url, output_folder=AUDIO_INPUT_DIR):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(str(output_folder), "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": False,
    }

    yt_dlp = _get_yt_dlp()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"🎵 Downloading: {url}")
        ydl.download([url])


if __name__ == "__main__":
    url = input("Paste the YouTube Music or YouTube link here: ")
    download_song(url)
