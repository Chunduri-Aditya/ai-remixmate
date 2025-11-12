"""
Configuration settings for AI RemixMate
"""
import os

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIO_INPUT_DIR = os.path.join(BASE_DIR, "audio_input")
AUDIO_OUTPUT_DIR = os.path.join(BASE_DIR, "audio_output")
SEPARATED_DIR = os.path.join(BASE_DIR, "separated")
STEMS_DIR = os.path.join(BASE_DIR, "stems")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LYRICS_DIR = os.path.join(BASE_DIR, "lyrics")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLAYLISTS_DIR = os.path.join(BASE_DIR, "playlists")
CONVERTED_DIR = os.path.join(BASE_DIR, "temp", "converted")

# Audio settings
SAMPLE_RATE = 44100
MAX_DURATION_SEC = None  # No limit - support any length files

# Create directories if they don't exist
for dir_path in [AUDIO_INPUT_DIR, AUDIO_OUTPUT_DIR, SEPARATED_DIR, STEMS_DIR, 
                 MODELS_DIR, LYRICS_DIR, OUTPUT_DIR, PLAYLISTS_DIR, CONVERTED_DIR]:
    os.makedirs(dir_path, exist_ok=True)

