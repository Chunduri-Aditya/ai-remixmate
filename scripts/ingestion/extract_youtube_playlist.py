"""
YouTube Playlist Extractor with skip logic for already downloaded songs.
"""
import os
import sys
import json
import yt_dlp
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

AUDIO_INPUT_DIR = os.path.join(os.path.dirname(__file__), "../../audio_input")
DB_PATH = os.path.join(os.path.dirname(__file__), "../../models/song_embeddings.json")
PLAYLISTS_META_PATH = os.path.join(os.path.dirname(__file__), "../../models/playlists.json")

os.makedirs(AUDIO_INPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def check_song_exists(song_title, output_folder, formats=['.wav', '.mp3', '.m4a']):
    """Check if song file already exists (case-insensitive)."""
    song_title_lower = song_title.lower()
    
    for root, dirs, files in os.walk(output_folder):
        for file in files:
            file_lower = file.lower()
            # Check if filename matches (case-insensitive)
            if any(file_lower.endswith(ext) and song_title_lower in file_lower for ext in formats):
                return True
            # Also check without extension
            file_no_ext = os.path.splitext(file_lower)[0]
            if song_title_lower in file_no_ext or file_no_ext in song_title_lower:
                return True
    
    return False

def check_song_in_database(song_title):
    """Check if song is already in database."""
    if not os.path.exists(DB_PATH):
        return False
    
    try:
        with open(DB_PATH, 'r') as f:
            db = json.load(f)
        
        # Case-insensitive check
        song_title_lower = song_title.lower()
        for db_title in db.keys():
            if db_title.lower() == song_title_lower:
                return True
        
        return False
    except Exception:
        return False

def extract_playlist_info(playlist_url):
    """Extract playlist information without downloading."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            
            playlist_info = {
                'title': info.get('title', 'Unknown Playlist'),
                'description': info.get('description', ''),
                'uploader': info.get('uploader', 'Unknown'),
                'songs': []
            }
            
            for entry in info.get('entries', []):
                if entry:
                    playlist_info['songs'].append({
                        'title': entry.get('title', 'Unknown'),
                        'url': entry.get('url', ''),
                        'duration': entry.get('duration', 0)
                    })
            
            return playlist_info
    except Exception as e:
        print(f"❌ Error extracting playlist info: {e}")
        return None

def download_playlist_songs(playlist_url, output_folder=AUDIO_INPUT_DIR,
                           playlist_name=None, add_to_database=False,
                           download_format='wav'):
    """Download all songs from playlist, skipping already downloaded ones."""
    
    # Extract playlist info
    print("📋 Extracting playlist information...")
    playlist_info = extract_playlist_info(playlist_url)
    
    if not playlist_info:
        print("❌ Failed to extract playlist information")
        return None
    
    print(f"\n✅ Found playlist: {playlist_info['title']}")
    print(f"📊 Total songs: {len(playlist_info['songs'])}")
    print(f"👤 Uploader: {playlist_info['uploader']}")
    
    # Set playlist folder
    if playlist_name:
        playlist_folder = os.path.join(output_folder, playlist_name)
    else:
        playlist_folder = os.path.join(output_folder, playlist_info['title'])
    
    os.makedirs(playlist_folder, exist_ok=True)
    
    # Download options
    if download_format == 'wav':
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(playlist_folder, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': False,
        }
    else:  # mp3
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(playlist_folder, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': False,
        }
    
    # Download each song
    successful = 0
    skipped = 0
    failed = 0
    skipped_songs = []
    failed_songs = []
    
    print("\n" + "=" * 60)
    
    for i, song in enumerate(playlist_info['songs'], 1):
        song_title = song['title']
        song_url = song['url']
        
        print(f"\n[{i}/{len(playlist_info['songs'])}] 🎵 Processing: {song_title}")
        
        # Check if already downloaded
        if check_song_exists(song_title, playlist_folder):
            print(f"   ⏭️  Skipped (already downloaded)")
            skipped += 1
            skipped_songs.append(song_title)
            
            # Still add to database if needed
            if add_to_database and not check_song_in_database(song_title):
                try:
                    from database.song_database import add_song_to_db
                    song_path = os.path.join(playlist_folder, f"{song_title}.{download_format}")
                    if os.path.exists(song_path):
                        add_song_to_db(song_path, song_title)
                        print(f"   📝 Added to database")
                except Exception as e:
                    print(f"   ⚠️  Database add failed: {e}")
            continue
        
        # Check if in database
        if check_song_in_database(song_title):
            print(f"   ⏭️  Skipped (already in database)")
            skipped += 1
            skipped_songs.append(song_title)
            continue
        
        # Download
        try:
            ydl_opts['outtmpl'] = os.path.join(playlist_folder, f'{song_title}.%(ext)s')
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([song_url])
            
            print(f"   ✅ Download complete")
            
            # Add to database if requested
            if add_to_database:
                try:
                    from database.song_database import add_song_to_db
                    song_path = os.path.join(playlist_folder, f"{song_title}.{download_format}")
                    if os.path.exists(song_path):
                        add_song_to_db(song_path, song_title)
                        print(f"   📝 Added to database")
                except Exception as e:
                    print(f"   ⚠️  Database add failed: {e}")
            
            successful += 1
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed += 1
            failed_songs.append((song_title, str(e)))
    
    # Save playlist metadata
    try:
        if os.path.exists(PLAYLISTS_META_PATH):
            with open(PLAYLISTS_META_PATH, 'r') as f:
                playlists_meta = json.load(f)
        else:
            playlists_meta = {}
        
        playlists_meta[playlist_info['title']] = {
            'title': playlist_info['title'],
            'description': playlist_info['description'],
            'uploader': playlist_info['uploader'],
            'total_songs': len(playlist_info['songs']),
            'downloaded': successful,
            'skipped': skipped,
            'failed': failed
        }
        
        with open(PLAYLISTS_META_PATH, 'w') as f:
            json.dump(playlists_meta, f, indent=2)
    except Exception as e:
        print(f"⚠️  Failed to save playlist metadata: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Playlist: {playlist_info['title']}")
    print(f"Total songs: {len(playlist_info['songs'])}")
    print(f"✅ Downloaded: {successful}")
    print(f"⏭️  Skipped (already exists): {skipped}")
    print(f"❌ Failed: {failed}")
    
    if add_to_database:
        print(f"\n📝 Added to database: {successful} songs")
        print(f"ℹ️  Already in database: {skipped} songs")
    
    if skipped_songs:
        print(f"\n⏭️  Skipped songs (already downloaded):")
        for song in skipped_songs[:10]:  # Show first 10
            print(f"   - {song}")
        if len(skipped_songs) > 10:
            print(f"   ... and {len(skipped_songs) - 10} more")
    
    if failed_songs:
        print(f"\n⚠️ Failed downloads:")
        for song, error in failed_songs:
            print(f"   - {song}: {error}")
    
    print(f"\n💾 Playlist metadata saved to: {PLAYLISTS_META_PATH}")
    print(f"✅ Done! Songs saved to: {playlist_folder}")
    
    return {
        'successful': successful,
        'skipped': skipped,
        'failed': failed,
        'playlist_folder': playlist_folder
    }

if __name__ == "__main__":
    print("🎵 YouTube Playlist Extractor for AI RemixMate")
    print("=" * 60)
    print("\nOptions:")
    print("1. Extract and download playlist")
    print("2. List extracted playlists")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        playlist_url = input("📎 Paste YouTube playlist URL: ").strip()
        playlist_name = input("📝 Playlist name (leave empty for auto): ").strip() or None
        
        print("\n🎵 Audio format:")
        print("1. WAV (recommended for remixing)")
        print("2. MP3 (smaller file size)")
        format_choice = input("Select format (1-2, default=1): ").strip() or "1"
        download_format = 'wav' if format_choice == "1" else 'mp3'
        
        add_to_db = input("📝 Add songs to database for recommendations? (y/n, default=n): ").strip().lower() == 'y'
        
        download_playlist_songs(
            playlist_url,
            playlist_name=playlist_name,
            add_to_database=add_to_db,
            download_format=download_format
        )
    elif choice == "2":
        if os.path.exists(PLAYLISTS_META_PATH):
            with open(PLAYLISTS_META_PATH, 'r') as f:
                playlists = json.load(f)
            print("\n📋 Extracted Playlists:")
            for name, info in playlists.items():
                print(f"\n- {name}")
                print(f"  Uploader: {info.get('uploader', 'Unknown')}")
                print(f"  Songs: {info.get('downloaded', 0)}/{info.get('total_songs', 0)}")
        else:
            print("No playlists extracted yet.")
    else:
        print("Goodbye!")

