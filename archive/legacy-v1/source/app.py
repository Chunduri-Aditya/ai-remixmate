#!/usr/bin/env python3
"""
AI RemixMate - Web Application
Checks dependencies and launches the Gradio web interface.
"""
import sys
import os
import importlib.util
from importlib import metadata

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

MIN_PYTHON = (3, 10)
MAX_PYTHON_EXCLUSIVE = (3, 13)

APP_REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "pydub": "pydub",
    "scikit-learn": "sklearn",
    "gradio": "gradio",
}

OPTIONAL_IMPORT_GROUPS = {
    "separation": {
        "demucs": "demucs",
        "torch": "torch",
    },
    "lyrics": {
        "openai-whisper": "whisper",
        "SpeechRecognition": "speech_recognition",
        "pocketsphinx": "pocketsphinx",
    },
    "ingestion": {
        "yt-dlp": "yt_dlp",
    },
}


def _is_python_supported():
    """Return True when the running Python version is supported."""
    return MIN_PYTHON <= sys.version_info[:2] < MAX_PYTHON_EXCLUSIVE


def _find_missing(import_map):
    """Return package names whose import modules are unavailable."""
    missing = []
    for package_name, module_name in import_map.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def _pydantic_version_issue():
    """Return a version issue string if installed pydantic is outside the supported range."""
    try:
        version = metadata.version("pydantic")
    except metadata.PackageNotFoundError:
        return None

    try:
        major, minor, *_ = version.split(".")
        if int(major) > 2 or (int(major) == 2 and int(minor) >= 12):
            return f"pydantic {version} installed; expected >=2.0.0,<2.12.0"
    except ValueError:
        return f"pydantic {version} installed; expected >=2.0.0,<2.12.0"

    return None


def _install_command(extras):
    """Build the recommended editable install command for the given extras."""
    extras_spec = ",".join(extras)
    return f'{sys.executable} -m pip install -e ".[{extras_spec}]"'


def check_runtime_environment():
    """Check dependency availability without installing packages."""
    print("🔍 Checking dependencies...")

    if not _is_python_supported():
        current = ".".join(str(part) for part in sys.version_info[:3])
        print(f"   ❌ Python {current} is unsupported.")
        print("   Use Python >=3.10,<3.13.")
        return False

    missing_required = _find_missing(APP_REQUIRED_IMPORTS)
    pydantic_issue = _pydantic_version_issue()

    if missing_required or pydantic_issue:
        if missing_required:
            print("   ❌ Missing required packages:")
            for package_name in missing_required:
                print(f"      - {package_name}")

        if pydantic_issue:
            print(f"   ❌ {pydantic_issue}")

        print("\nInstall the web app dependencies with:")
        print(f"   {_install_command(['web'])}")
        print("\nFor the full local feature set, use:")
        print(f"   {_install_command(['web', 'separation', 'lyrics', 'ingestion', 'data'])}")
        return False

    print("✅ Required web app dependencies are installed.")

    missing_optional = {
        group_name: _find_missing(import_map)
        for group_name, import_map in OPTIONAL_IMPORT_GROUPS.items()
    }
    missing_optional = {
        group_name: packages
        for group_name, packages in missing_optional.items()
        if packages
    }

    if missing_optional:
        print("\nOptional feature packages are missing:")
        for group_name, packages in missing_optional.items():
            print(f"   - {group_name}: {', '.join(packages)}")
        print("\nInstall optional extras as needed, for example:")
        print(f"   {_install_command(['separation'])}")
        print(f"   {_install_command(['lyrics'])}")

    return True

def main():
    """Main application entry point."""
    print("=" * 60)
    print("🎵 AI RemixMate - Web Application")
    print("=" * 60)
    print()
    
    # Check dependencies without mutating the environment.
    if not check_runtime_environment():
        print("\n❌ Dependency check failed. Install the packages above and run again.")
        sys.exit(1)
    
    print("\n🚀 Starting web application...")
    print()
    
    # Import after dependencies are checked
    try:
        import gradio as gr
        import logging
        import numpy as np
        import soundfile as sf
        from remixmate import (
            config, remix_core, dj_mixing, recommendations,
            lyrics_extraction, ml_audio_features, auto_mode_selector,
            playlist_manager, structure_detection
        )
        from remixmate.recommendations import analyze_track_for_display, format_recommendations, find_compatible_songs
        from remixmate.auto_mode_selector import select_auto_mode
        from remixmate.ml_audio_features import get_strategy_engine
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed.")
        sys.exit(1)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize strategy engine
    strategy_engine = get_strategy_engine()
    
    # ==================== UI Functions ====================
    
    def analyze_track1(file):
        """Analyze Track 1 and show recommendations."""
        if file is None:
            return "", ""
        
        try:
            file_path = file.name if hasattr(file, 'name') else str(file)
            analysis = analyze_track_for_display(file_path)
            recommendations_text = ""
            
            # Get recommendations
            recs = find_compatible_songs(file_path, top_k=5)
            if recs:
                recommendations_text = format_recommendations(recs)
            else:
                recommendations_text = "No recommendations available. Add songs to the database first."
            
            return analysis, recommendations_text
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return f"Error analyzing track: {e}", ""
    
    def analyze_track2(file):
        """Analyze Track 2."""
        if file is None:
            return ""
        
        try:
            file_path = file.name if hasattr(file, 'name') else str(file)
            return analyze_track_for_display(file_path)
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return f"Error analyzing track: {e}"
    
    def remix_files(file1, file2, remix_mode, use_intelligent, use_arrangement,
                   aggressiveness, energy_shape):
        """Main remix function."""
        if file1 is None or file2 is None:
            return None, "Please upload both tracks"
        
        try:
            # Get file paths (Gradio File object)
            file1_path = file1.name if hasattr(file1, 'name') else str(file1)
            file2_path = file2.name if hasattr(file2, 'name') else str(file2)
            
            # Validate files exist
            if not os.path.exists(file1_path):
                return None, f"Error: File 1 not found: {file1_path}"
            if not os.path.exists(file2_path):
                return None, f"Error: File 2 not found: {file2_path}"
            
            # Check file sizes (prevent processing empty/corrupted files)
            if os.path.getsize(file1_path) < 1000:  # Less than 1KB
                return None, "Error: File 1 appears to be empty or corrupted"
            if os.path.getsize(file2_path) < 1000:
                return None, "Error: File 2 appears to be empty or corrupted"
            
            # Auto mode selection
            if remix_mode == "🎯 Auto (Recommended)":
                try:
                    remix_mode = select_auto_mode(file1_path, file2_path)
                    logger.info(f"Auto-selected mode: {remix_mode}")
                except Exception as e:
                    logger.warning(f"Auto mode selection failed: {e}, using mashup")
                    remix_mode = "mashup"
            
            # Map UI mode to function mode
            mode_map = {
                "Mashup (Mix Both Tracks)": "mashup",
                "Track 1 Vocals + Track 2 Instruments": "base_vocals_match_instr",
                "Track 1 Instruments + Track 2 Vocals": "base_instr_match_vocals"
            }
            actual_mode = mode_map.get(remix_mode, remix_mode)
            
            # Intelligent mixing
            if use_intelligent:
                try:
                    track1_features = recommendations.analyze_track_characteristics(file1_path)
                    track2_features = recommendations.analyze_track_characteristics(file2_path)
                    
                    strategy = strategy_engine.predict_best_mixing_strategy(
                        track1_features, track2_features,
                        track1_features.get("key"), track2_features.get("key")
                    )
                    
                    mixing_technique = strategy["technique"]
                    crossfade_length = strategy["crossfade_length"]
                    apply_beatmatching = strategy["apply_beatmatching"]
                    apply_harmonic_mixing = strategy["apply_harmonic_mixing"]
                    bass_swap = (mixing_technique == "bass_swap")
                except Exception as e:
                    logger.warning(f"Intelligent mixing failed: {e}, using defaults")
                    mixing_technique = "crossfade"
                    crossfade_length = 8.0
                    apply_beatmatching = True
                    apply_harmonic_mixing = False
                    bass_swap = False
            else:
                mixing_technique = "crossfade"
                crossfade_length = 8.0
                apply_beatmatching = True
                apply_harmonic_mixing = False
                bass_swap = False
            
            # Remix
            output_path = remix_core.remix_two_files(
                file1_path, file2_path,
                mode=actual_mode,
                use_arrangement_mixing=use_arrangement,
                use_intelligent_mixing=use_intelligent,
                crossfade_length=crossfade_length,
                mixing_technique=mixing_technique,
                apply_beatmatching=apply_beatmatching,
                apply_harmonic_mixing=apply_harmonic_mixing,
                bass_swap=bass_swap,
                aggressiveness=aggressiveness,
                energy_shape=energy_shape
            )
            
            if output_path and os.path.exists(output_path):
                try:
                    remix_metadata = remix_core.load_remix_metadata(output_path)
                    stem_status = remix_core.format_stem_status(remix_metadata)
                except Exception as e:
                    logger.warning(f"Could not load remix metadata: {e}")
                    stem_status = "Stem status is unavailable."

                return output_path, f"✅ Remix created successfully!\n{stem_status}"
            else:
                return None, "Error: Remix file was not created"
        except Exception as e:
            logger.error(f"Remix error: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error creating remix: {str(e)[:200]}"
    
    def toggle_arrangement_controls(use_arrangement):
        """Show/hide arrangement controls."""
        return gr.update(visible=use_arrangement)
    
    # ==================== UI Layout ====================
    
    theme = gr.themes.Soft()

    with gr.Blocks(title="AI RemixMate") as app:
        gr.Markdown("""
        # 🎵 AI RemixMate - Arrangement-Level AI DJ Mixer
        
        Create professional remixes with AI-powered stem separation, intelligent mixing, and arrangement-level DJ techniques.
        """)
        
        with gr.Tabs():
            # ========== Remixing Tab ==========
            with gr.Tab("🎵 Remix Creator"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Track 1")
                        track1 = gr.File(label="Upload Track 1", file_types=["audio"])
                        track1_analysis = gr.Markdown("**Track Analysis:** Upload a file to see analysis")
                        recommendations_display = gr.Markdown("")
                    
                    with gr.Column():
                        gr.Markdown("### Track 2")
                        track2 = gr.File(label="Upload Track 2", file_types=["audio"])
                        track2_analysis = gr.Markdown("**Track Analysis:** Upload a file to see analysis")
                
                with gr.Accordion("🎛️ Remix Settings", open=True):
                    remix_mode = gr.Dropdown(
                        choices=[
                            "🎯 Auto (Recommended)",
                            "Mashup (Mix Both Tracks)",
                            "Track 1 Vocals + Track 2 Instruments",
                            "Track 1 Instruments + Track 2 Vocals"
                        ],
                        value="🎯 Auto (Recommended)",
                        label="Remix Mode"
                    )
                    
                    use_intelligent = gr.Checkbox(
                        label="🧠 Intelligent Mix (Recommended)",
                        value=True,
                        info="Automatically optimizes mixing settings"
                    )
                    
                    use_arrangement = gr.Checkbox(
                        label="🎬 Arrangement-Level Mixing",
                        value=False,
                        info="DJ-style arrangement with sections and timeline"
                    )
                    
                    with gr.Row(visible=False) as arrangement_controls:
                        aggressiveness = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="🎚️ Remix Aggressiveness",
                            info="0.0 = Conservative, 1.0 = Wild"
                        )
                        
                        energy_shape = gr.Dropdown(
                            choices=["chill_to_peak", "rollercoaster", "slow_build", "double_peak"],
                            value="chill_to_peak",
                            label="📈 Energy Shape"
                        )
                    
                submit_btn = gr.Button("🎵 Create Remix", variant="primary", size="lg")
                
                with gr.Row():
                    output_audio = gr.Audio(label="Remixed Output", type="filepath")
                    status = gr.Textbox(label="Status", interactive=False)
                
                # Event handlers
                track1.change(analyze_track1, inputs=[track1], outputs=[track1_analysis, recommendations_display])
                track2.change(analyze_track2, inputs=[track2], outputs=[track2_analysis])
                use_arrangement.change(
                    toggle_arrangement_controls,
                    inputs=[use_arrangement],
                    outputs=[arrangement_controls]
                )
                submit_btn.click(
                    remix_files,
                    inputs=[track1, track2, remix_mode, use_intelligent, use_arrangement,
                           aggressiveness, energy_shape],
                    outputs=[output_audio, status]
                )
            
            # ========== Playlist Manager Tab ==========
            with gr.Tab("📋 Playlist Manager"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Create Playlist")
                        playlist_name = gr.Textbox(label="Playlist Name", placeholder="My Favorite Mixes")
                        playlist_desc = gr.Textbox(label="Description (Optional)", placeholder="Best songs for remixing")
                        create_btn = gr.Button("➕ Create Playlist", variant="primary")
                        create_status = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("### Manage Playlists")
                        playlist_dropdown = gr.Dropdown(
                            choices=["(No playlists yet)"],
                            value=None,
                            label="Select Playlist",
                            interactive=True,
                            allow_custom_value=False
                        )
                        playlist_info = gr.Markdown("Select a playlist to view details")
                        
                        refresh_btn = gr.Button("🔄 Refresh Playlists")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Add Song to Playlist")
                        available_songs = gr.Dropdown(
                            choices=["(No songs available)"],
                            value=None,
                            label="Available Songs",
                            interactive=True,
                            allow_custom_value=False
                        )
                        add_btn = gr.Button("➕ Add Song to Playlist")
                    
                    with gr.Column():
                        gr.Markdown("### Remove Song from Playlist")
                        playlist_songs = gr.Dropdown(
                            choices=["(No songs in playlist)"],
                            value=None,
                            label="Songs in Playlist",
                            interactive=True,
                            allow_custom_value=False
                        )
                        remove_btn = gr.Button("➖ Remove Song")
                
                delete_btn = gr.Button("🗑️ Delete Playlist", variant="stop")
                
                def refresh_playlists():
                    """Refresh playlist list."""
                    playlists = playlist_manager.get_playlists()
                    choices = [p["name"] for p in playlists] if playlists else ["(No playlists yet)"]
                    songs = playlist_manager.get_available_songs()
                    song_choices = [f"{s['name']}" for s in songs] if songs else ["(No songs available)"]
                    return gr.update(choices=choices, value=None), gr.update(choices=song_choices, value=None)
                
                def create_playlist(name, desc):
                    """Create a new playlist."""
                    if not name:
                        return "Please enter a playlist name", refresh_playlists()[0], refresh_playlists()[1]
                    try:
                        playlist_manager.create_playlist(name, desc)
                        return f"✅ Created playlist: {name}", *refresh_playlists()
                    except Exception as e:
                        return f"❌ Error: {e}", refresh_playlists()[0], refresh_playlists()[1]
                
                def update_playlist_info(playlist_name):
                    """Update playlist info display."""
                    if not playlist_name or playlist_name.startswith("(No "):
                        return "Select a playlist", gr.update(choices=["(No songs in playlist)"], value=None)
                    try:
                        playlists = playlist_manager.get_playlists()
                        playlist = next((p for p in playlists if p["name"] == playlist_name), None)
                        if playlist:
                            songs = playlist_manager.get_playlist_songs(playlist["id"])
                            song_choices = [f"{s['name']}" for s in songs] if songs else ["(No songs in playlist)"]
                            
                            info = f"""
**Playlist:** {playlist['name']}
**Description:** {playlist.get('description', 'None')}
**Songs:** {playlist['song_count']}
**Created:** {playlist.get('created_at', 'Unknown')}
"""
                            return info, gr.update(choices=song_choices, value=None)
                        return "Playlist not found", gr.update(choices=["(No songs in playlist)"], value=None)
                    except Exception as e:
                        return f"Error: {e}", gr.update(choices=["(No songs in playlist)"], value=None)
                
                def add_song(playlist_name, song_name):
                    """Add song to playlist."""
                    if not playlist_name or playlist_name.startswith("(No ") or not song_name or song_name.startswith("(No "):
                        return "Please select playlist and song", *refresh_playlists()
                    try:
                        playlists = playlist_manager.get_playlists()
                        playlist = next((p for p in playlists if p["name"] == playlist_name), None)
                        if playlist:
                            songs = playlist_manager.get_available_songs()
                            song = next((s for s in songs if s["name"] == song_name), None)
                            if song:
                                playlist_manager.add_song_to_playlist(playlist["id"], song["name"], song["path"])
                                return f"✅ Added {song_name} to {playlist_name}", *refresh_playlists()
                        return "Error adding song", *refresh_playlists()
                    except Exception as e:
                        return f"Error: {e}", *refresh_playlists()
                
                def remove_song(playlist_name, song_name):
                    """Remove song from playlist."""
                    if not playlist_name or playlist_name.startswith("(No ") or not song_name or song_name.startswith("(No "):
                        return "Please select playlist and song", *refresh_playlists()
                    try:
                        playlists = playlist_manager.get_playlists()
                        playlist = next((p for p in playlists if p["name"] == playlist_name), None)
                        if playlist:
                            songs = playlist_manager.get_playlist_songs(playlist["id"])
                            song = next((s for s in songs if s["name"] == song_name), None)
                            if song:
                                playlist_manager.remove_song_from_playlist(playlist["id"], song["name"], song["path"])
                                return f"✅ Removed {song_name} from {playlist_name}", *refresh_playlists()
                        return "Error removing song", *refresh_playlists()
                    except Exception as e:
                        return f"Error: {e}", *refresh_playlists()
                
                def delete_playlist(playlist_name):
                    """Delete a playlist."""
                    if not playlist_name or playlist_name.startswith("(No "):
                        return "Please select a playlist", *refresh_playlists()
                    try:
                        playlists = playlist_manager.get_playlists()
                        playlist = next((p for p in playlists if p["name"] == playlist_name), None)
                        if playlist:
                            playlist_manager.delete_playlist(playlist["id"])
                            return f"✅ Deleted playlist: {playlist_name}", *refresh_playlists()
                        return "Playlist not found", *refresh_playlists()
                    except Exception as e:
                        return f"Error: {e}", *refresh_playlists()
                
                # Event handlers
                refresh_btn.click(refresh_playlists, outputs=[playlist_dropdown, available_songs])
                create_btn.click(create_playlist, inputs=[playlist_name, playlist_desc],
                               outputs=[create_status, playlist_dropdown, available_songs])
                playlist_dropdown.change(update_playlist_info, inputs=[playlist_dropdown],
                                       outputs=[playlist_info, playlist_songs])
                add_btn.click(add_song, inputs=[playlist_dropdown, available_songs],
                            outputs=[create_status, playlist_dropdown, available_songs])
                remove_btn.click(remove_song, inputs=[playlist_dropdown, playlist_songs],
                              outputs=[create_status, playlist_dropdown, available_songs])
                delete_btn.click(delete_playlist, inputs=[playlist_dropdown],
                               outputs=[create_status, playlist_dropdown, available_songs])
                
                # Initial refresh
                app.load(refresh_playlists, outputs=[playlist_dropdown, available_songs])
        
        gr.Markdown("""
        ---
        ### 📖 Features
        - ✅ Multi-format audio support (MP3, WAV, M4A, FLAC, etc.)
        - ✅ Intelligent volume balancing with vocal enhancement
        - ✅ Dynamic crossfade curves
        - ✅ Beatmatching and conservative harmonic key-shifting
        - ✅ ML-based genre classification
        - ✅ Arrangement-level DJ mixing with energy curve planning
        - ✅ Playlist management
        - ✅ Song recommendations
        """)
    
    # Launch app
    print("\n" + "=" * 60)
    print("🌐 Web application starting...")
    print("=" * 60)
    print("\n📱 Open your browser to the URL shown below")
    print("   (Usually: http://127.0.0.1:7860)")
    print("\n💡 Press Ctrl+C to stop the server")
    print()
    
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=theme)

if __name__ == "__main__":
    main()
