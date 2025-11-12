"""
Playlist management: create, add, remove, view playlists.
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PLAYLISTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models", "user_playlists.json"
)

# Ensure models directory exists
os.makedirs(os.path.dirname(PLAYLISTS_FILE), exist_ok=True)

def _load_playlists():
    """Load playlists from JSON file."""
    if os.path.exists(PLAYLISTS_FILE):
        try:
            with open(PLAYLISTS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_playlists(playlists):
    """Save playlists to JSON file."""
    with open(PLAYLISTS_FILE, 'w') as f:
        json.dump(playlists, f, indent=2)

def create_playlist(name, description="", user_id="default"):
    """Create a new playlist."""
    playlists = _load_playlists()
    
    # Generate ID
    playlist_id = f"{user_id}_{name.lower().replace(' ', '_')}"
    
    # Check if exists
    if playlist_id in playlists:
        logger.warning(f"Playlist '{name}' already exists")
        return playlists[playlist_id]
    
    playlist = {
        "id": playlist_id,
        "name": name,
        "description": description,
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "songs": [],
        "song_count": 0
    }
    
    playlists[playlist_id] = playlist
    _save_playlists(playlists)
    
    logger.info(f"Created playlist: {name}")
    return playlist

def add_song_to_playlist(playlist_id, song_name, song_path):
    """Add a song to a playlist."""
    playlists = _load_playlists()
    
    if playlist_id not in playlists:
        raise ValueError(f"Playlist '{playlist_id}' not found")
    
    playlist = playlists[playlist_id]
    
    # Check if already exists
    for song in playlist["songs"]:
        if song["name"] == song_name and song["path"] == song_path:
            logger.warning(f"Song '{song_name}' already in playlist")
            return playlist
    
    # Add song
    song_entry = {
        "name": song_name,
        "path": song_path,
        "added_at": datetime.now().isoformat()
    }
    
    playlist["songs"].append(song_entry)
    playlist["song_count"] = len(playlist["songs"])
    playlist["updated_at"] = datetime.now().isoformat()
    
    _save_playlists(playlists)
    logger.info(f"Added '{song_name}' to playlist '{playlist['name']}'")
    
    return playlist

def remove_song_from_playlist(playlist_id, song_name, song_path):
    """Remove a song from a playlist."""
    playlists = _load_playlists()
    
    if playlist_id not in playlists:
        raise ValueError(f"Playlist '{playlist_id}' not found")
    
    playlist = playlists[playlist_id]
    
    # Remove song
    playlist["songs"] = [
        s for s in playlist["songs"]
        if not (s["name"] == song_name and s["path"] == song_path)
    ]
    
    playlist["song_count"] = len(playlist["songs"])
    playlist["updated_at"] = datetime.now().isoformat()
    
    _save_playlists(playlists)
    logger.info(f"Removed '{song_name}' from playlist '{playlist['name']}'")
    
    return playlist

def get_playlists(user_id="default"):
    """Get all playlists for a user."""
    playlists = _load_playlists()
    return [p for p in playlists.values() if p.get("user_id") == user_id]

def get_playlist(playlist_id):
    """Get a specific playlist."""
    playlists = _load_playlists()
    return playlists.get(playlist_id)

def get_playlist_songs(playlist_id):
    """Get all songs in a playlist."""
    playlist = get_playlist(playlist_id)
    if playlist:
        return playlist.get("songs", [])
    return []

def delete_playlist(playlist_id):
    """Delete a playlist."""
    playlists = _load_playlists()
    
    if playlist_id not in playlists:
        raise ValueError(f"Playlist '{playlist_id}' not found")
    
    playlist_name = playlists[playlist_id]["name"]
    del playlists[playlist_id]
    
    _save_playlists(playlists)
    logger.info(f"Deleted playlist: {playlist_name}")
    
    return True

def get_available_songs():
    """Get list of available songs from audio_input and database."""
    songs = []
    
    # From audio_input directory
    audio_input_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "audio_input"
    )
    
    if os.path.exists(audio_input_dir):
        for root, dirs, files in os.walk(audio_input_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.aiff')):
                    full_path = os.path.join(root, file)
                    songs.append({
                        "name": os.path.splitext(file)[0],
                        "path": full_path
                    })
    
    # From database
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "models", "song_embeddings.json"
    )
    
    if os.path.exists(db_path):
        try:
            with open(db_path, 'r') as f:
                db = json.load(f)
            for song_name in db.keys():
                # Try to find file path
                # This is simplified - in production, store paths in DB
                songs.append({
                    "name": song_name,
                    "path": f"db:{song_name}"  # Placeholder
                })
        except Exception:
            pass
    
    return songs

