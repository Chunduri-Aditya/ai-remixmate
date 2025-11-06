# scripts/download_song.py
# To install the required package, run: pip install yt_dlp

import os
import yt_dlp

def download_song(url, output_folder):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"🎵 Downloading: {url}")
        ydl.download([url])

if __name__ == "__main__":
    url = input("Paste the YouTube Music or YouTube link here: ")
    download_song(url, "../audio_input")